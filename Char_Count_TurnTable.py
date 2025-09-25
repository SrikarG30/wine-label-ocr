import cv2
import time
import re
import numpy as np
import pytesseract
from collections import deque

import os

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _yolo_import_error = e

# ---- Camera / OCR cadence ----
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
OCR_INTERVAL = 0.25       # seconds between OCR calls

# ---- Smoothing ----
ALPHA = 0.25              # EMA smoothing factor (higher = more responsive)
WINDOW_SEC = 5.0          # keep last 5s for basic stats 

# ---- Bottle detection ----
BOTTLE_CLASS_ID = 39
YOLO_CONF = 0.4

# ---- ROI (green rectangle inside bottle bbox) ----
ROI_WIDTH_FRAC = 1.0      # bbox width
ROI_HEIGHT_FRAC = 0.9     # bbox height
MIN_ROI_PIXELS = 120*120

# ---- Simplified peak/band guidance ----
DROP_TO_START = 5.0       # how far below peak to decide we've passed it (begin reverse)
MATCH_BAND    = 0       # accept counts in [peak - MATCH_BAND, peak] as centered

# ---- Peak update/unlock ------
PEAK_BUMP_MIN = 1.0       # only raise peak if new value exceeds by this many chars
UNLOCK_MARGIN = 3.0       # drop lock if we fall below peak
UNLOCK_DWELL  = 0.6       # seconds below threshold to actually unlock

#--- Multi Bottle -----
PRESENT_THRES = 5
ABSENT_THRESH = 8



CAPTURE_DIR = "captures"

FONT = cv2.FONT_HERSHEY_SIMPLEX

CAPTURE_PATH = "captures/capture.png"


def count_alnum(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]", text))


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 15)
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th


def ocr_char_count(img) -> int:
    cfg = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    text = pytesseract.image_to_string(img, config=cfg)
    return count_alnum(text)


def ema_update(prev, new):
    return ALPHA * new + (1 - ALPHA) * prev if prev is not None else float(new)


class Ring:
    def __init__(self):
        self.ts = deque()
        self.val = deque()
    def append(self, t, v):
        self.ts.append(t); self.val.append(v)
    def trim(self, now, horizon):
        while self.ts and (now - self.ts[0] > horizon):
            self.ts.popleft(); self.val.popleft()
    def range(self):
        if not self.val:
            return 0.0
        return float(max(self.val) - min(self.val))


def choose_bottle_box(results):
    best = None
    best_area = 0
    for r in results:
        if r.boxes is None: continue
        for b in r.boxes:
            cls = int(b.cls.item()) if hasattr(b, 'cls') else -1
            if cls != BOTTLE_CLASS_ID: continue
            conf = float(b.conf.item()) if hasattr(b, 'conf') else 0.0
            if conf < YOLO_CONF: continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            area = max(0, x2-x1) * max(0, y2-y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2, conf)
    return best


def roi_from_bbox(frame, box):
    x1, y1, x2, y2, _ = box
    w = x2 - x1; h = y2 - y1
    roi_w = int(max(10, ROI_WIDTH_FRAC * w))
    roi_h = int(max(10, ROI_HEIGHT_FRAC * h))
    cx = x1 + w // 2; cy = y1 + h // 2
    rx1 = max(x1, cx - roi_w // 2)
    rx2 = min(x2, cx + roi_w // 2)
    ry1 = max(y1 + (h - roi_h)//2, y1)
    ry2 = min(ry1 + roi_h, y2)
    roi = frame[ry1:ry2, rx1:rx2]
    return roi, (rx1, ry1, rx2, ry2)


def main():
    if YOLO is None:
        raise RuntimeError(f"Failed to import ultralytics.YOLO: {_yolo_import_error}\nInstall with 'pip install ultralytics'.")

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Change CAM_INDEX.")

    # --- Session state (multi-bottle) ---
    armed = False          # SPACE toggles this
    in_session = False     # true while tracking the current bottle
    present_run = 0        # consecutive frames with a bottle
    absent_run  = 0        # consecutive frames without a bottle
    session_id  = 1

    # --- Guidance state ---
    last_ocr = 0.0
    smoothed = None
    window = Ring()

    phase = 'SCAN_CLOCKWISE'   # 'SCAN_CLOCKWISE' -> 'POST_PEAK_SEEK' -> lock
    peak_count = None
    center_lock = False
    unlock_timer = None
    paused = False

    captured_center = False        # ensure we save only once per lock
    saved_banner_until = 0.0       # HUD banner timer

    # Small helper: reset per-bottle guidance
    def start_session(start_reason=""):
        nonlocal in_session, phase, peak_count, center_lock, unlock_timer
        nonlocal smoothed, window, captured_center, present_run, absent_run
        in_session = True
        phase = 'SCAN_CLOCKWISE'
        peak_count = None
        center_lock = False
        unlock_timer = None
        smoothed = None
        window = Ring()
        captured_center = False
        present_run = 0
        absent_run = 0
        print(f"[session {session_id}] started ({start_reason})")

    def end_session():
        nonlocal in_session, session_id, armed, present_run, absent_run
        in_session = False
        armed = False  # require SPACE again for the next bottle
        session_id += 1
        present_run = 0
        absent_run = 0
        print(f"[session {session_id-1}] ended (bottle removed) — press SPACE to arm next bottle")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        display = frame.copy()

        # --- YOLO detect bottle ---
        yolo_out = model.predict(source=frame, classes=[BOTTLE_CLASS_ID], conf=YOLO_CONF, verbose=False)
        box = choose_bottle_box(yolo_out)   # (x1,y1,x2,y2,conf) or None

        roi_rect = None
        if box is not None:
            (x1, y1, x2, y2, conf) = box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(display, f"bottle {conf:.2f}", (x1, max(0, y1-8)), FONT, 0.5, (0, 180, 255), 1, cv2.LINE_AA)

            roi, (rx1, ry1, rx2, ry2) = roi_from_bbox(frame, box)
            roi_rect = (rx1, ry1, rx2, ry2)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

        # --- Presence/absence counters ---
        if box is not None:
            present_run += 1
            absent_run = 0
        else:
            present_run = 0
            absent_run += 1

        # --- Immediate start logic ---
        # If armed and not in-session:
        #   - If a bottle is visible *now*, start immediately (no debounce wait).
        #   - Otherwise keep waiting; when a bottle appears, the check above will start it.
        if armed and (not in_session) and (box is not None):
            start_session("immediate")

        # --- Auto-end if bottle is removed for a while ---
        if in_session and (absent_run >= ABSENT_THRESH):
            end_session()

        # --- OCR / Center finding runs ONLY when armed & in-session & not paused ---
        if (box is not None) and roi_rect and (now - last_ocr) >= OCR_INTERVAL and not paused and armed and in_session:
            last_ocr = now

            # OCR preprocess + count
            rx1, ry1, rx2, ry2 = roi_rect
            roi_for_ocr = frame[ry1:ry2, rx1:rx2]
            proc = preprocess(roi_for_ocr)
            hR, wR = proc.shape[:2]
            if min(hR, wR) < 120:
                scale = 120.0 / float(min(hR, wR))
                proc = cv2.resize(proc, (int(wR*scale), int(hR*scale)), interpolation=cv2.INTER_CUBIC)

            cnt = ocr_char_count(proc)
            smoothed = ema_update(smoothed, cnt)
            window.append(now, smoothed)
            window.trim(now, WINDOW_SEC)

            # Peak tracking
            if (peak_count is None) or (smoothed >= peak_count + PEAK_BUMP_MIN):
                peak_count = smoothed

            # Phase logic -> lock
            if not center_lock:
                if phase == 'SCAN_CLOCKWISE':
                    if (peak_count is not None) and (smoothed <= peak_count - DROP_TO_START):
                        phase = 'POST_PEAK_SEEK'
                elif phase == 'POST_PEAK_SEEK':
                    if (peak_count is not None) and (smoothed >= max(0.0, peak_count - MATCH_BAND)):
                        center_lock = True
                        unlock_timer = None

                        # === SAVE: only once per lock, bbox only, overwrite image.jpg ===
                        if not captured_center and box is not None:
                            H, W = frame.shape[:2]
                            x1, y1, x2, y2, _ = box
                            x1 = max(0, min(int(x1), W - 1))
                            x2 = max(0, min(int(x2), W))
                            y1 = max(0, min(int(y1), H - 1))
                            y2 = max(0, min(int(y2), H))
                            if x2 > x1 and y2 > y1:
                                bbox_crop = frame[y1:y2, x1:x2]
                                if bbox_crop.size > 0:
                                    ok = cv2.imwrite(CAPTURE_PATH, bbox_crop)
                                    print(f"[session {session_id}] saved center bbox -> {CAPTURE_PATH}, ok={ok}")
                                    if ok:
                                        captured_center = True
                                        saved_banner_until = now + 1.5
            else:
                # Unlock if we drift away from the match band for a bit
                if (peak_count is not None) and (smoothed < (peak_count - (MATCH_BAND + UNLOCK_MARGIN))):
                    unlock_timer = unlock_timer or now
                    if (now - unlock_timer) >= UNLOCK_DWELL:
                        center_lock = False
                        phase = 'POST_PEAK_SEEK'
                        unlock_timer = None
                        captured_center = False
                else:
                    unlock_timer = None

        # --- HUD ---
        h, w = display.shape[:2]
        cur_range = window.range()
        if not armed:
            status = "Press SPACE to ARM"
            color = (200, 200, 200)
        else:
            if not in_session:
                status = "Armed — show a bottle to start"
                color = (0, 200, 255)
            else:
                if center_lock:
                    status = "CENTER FOUND — HOLD"
                    color = (0, 255, 0)
                else:
                    status = "Scanning…" if phase == 'SCAN_CLOCKWISE' else "← Spin LEFT (reverse to center)"
                    color = (0, 200, 255)

        cv2.rectangle(display, (10, 10), (w - 10, 160), (0, 0, 0), -1)
        cc_text = "Char count: …" if smoothed is None else f"Char count (EMA): {smoothed:.1f}"
        pk_text = "Peak: …" if peak_count is None else f"Peak: {peak_count:.1f}"
        cv2.putText(display, cc_text, (20, 50), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"Win range: {cur_range:.1f}  |  {pk_text}", (20, 80), FONT, 0.6, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(display, status, (20, 115), FONT, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(display, f"DROP_TO_START={DROP_TO_START:.1f}  MATCH_BAND={MATCH_BAND:.1f}", (20, 145), FONT, 0.5, (170, 220, 170), 1, cv2.LINE_AA)

        if now < saved_banner_until:
            cv2.putText(display, "Saved center bbox", (20, 185), FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # OCR preview thumbnail
        if roi_rect is not None:
            rx1, ry1, rx2, ry2 = roi_rect
            roi_vis = frame[ry1:ry2, rx1:rx2]
            thumb = preprocess(roi_vis)
            thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            th_h = 140
            th_w = int(thumb.shape[1] * th_h / thumb.shape[0])
            thumb = cv2.resize(thumb, (th_w, th_h))
            display[10:10+th_h, w - 10 - th_w:w - 10] = thumb
            cv2.rectangle(display, (w - 10 - th_w, 10), (w - 10, 10 + th_h), (255, 255, 255), 1)
            cv2.putText(display, "OCR view", (w - 10 - th_w, 10 + th_h + 20), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Bottle OCR Guidance — Multi-bottle (immediate start)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # SPACE toggles arming; session starts *immediately* on next frame if a bottle is visible
            armed = not armed
            if not armed and in_session:
                # manual disarm also ends the session
                in_session = False
                session_id += 1
                print(f"[session {session_id-1}] ended (manual)")
            print("[armed]" if armed else "[disarmed]")
        elif key == ord('p'):
            paused = not paused
            print("[paused]" if paused else "[resumed]")
        elif key == ord('r'):
            # Hard reset everything; requires SPACE again
            armed = False
            in_session = False
            present_run = 0
            absent_run = 0
            phase = 'SCAN_CLOCKWISE'
            peak_count = None
            center_lock = False
            unlock_timer = None
            smoothed = None
            window = Ring()
            captured_center = False
            saved_banner_until = 0.0
            print("[reset] press SPACE to arm")

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
