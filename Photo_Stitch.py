import cv2
import numpy as np
import time
import subprocess
import sys
import tempfile
import uuid
import json
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit(
        f"Install ultralytics first: pip install ultralytics\n\n{e}")

# ---------- paths for auto-run of your scanner ----------
ROOT = Path(__file__).resolve().parent
SCAN_SCRIPT = ROOT / "ocr_scripts" / "scan_and_store.py"
WEIGHTS = ROOT / "weights.pt"
DB_PATH = ROOT / "storing_images" / "cellar.json"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists

CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BOTTLE_CLASS_ID = 39          # coco "bottle"
YOLO_CONF = 0.40
GUIDE_BOX_FRAC = (0.40, 0.80)  # (width_frac, height_frac)


def draw_guide_box(img, frac=(0.4, 0.8)):
    # --- Giant Box -----
    h, w = img.shape[:2]
    gw = int(w * frac[0])
    gh = int(h * frac[1])
    x1 = (w - gw) // 2
    y1 = (h - gh) // 2
    x2 = x1 + gw
    y2 = y1 + gh
    # ---- box -----
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 220, 60), 2)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
    # ----- crosshair lines -----
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.line(img, (cx, y1), (cx, y2), (60, 220, 60), 1)
    cv2.line(img, (x1, cy), (x2, cy), (60, 220, 60), 1)
    return (x1, y1, x2, y2)


def choose_bottle_box(results):
    best = None
    best_area = 0
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            try:
                cls = int(b.cls.item())
                if cls != BOTTLE_CLASS_ID:
                    continue
                conf = float(b.conf.item())
                if conf < YOLO_CONF:
                    continue
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2, conf)
            except Exception:
                continue
    return best


def safe_crop(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(int(x1), W - 1))
    x2 = max(0, min(int(x2), W))
    y1 = max(0, min(int(y1), H - 1))
    y2 = max(0, min(int(y2), H))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def stitch_horizontal(img1, img2):
    # ----- Return side-by-side stitch with matched heights (no saving).----
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2:
        scale = h1 / float(h2)
        img2 = cv2.resize(img2, (int(w2 * scale), h1),
                          interpolation=cv2.INTER_CUBIC)
    return np.hstack((img1, img2))

# ---------- helper: run your existing CLI on a stitched numpy image ----------


def run_scan_on_np(img_bgr):
    """
    Saves stitched BGR image to a temp file, runs your CLI scanner,
    returns parsed JSON dict. Temp file is removed automatically.
    """
    tmp = Path(tempfile.gettempdir()) / f"stitch_{uuid.uuid4().hex}.jpg"
    try:
        ok = cv2.imwrite(str(tmp), img_bgr)
        if not ok:
            print("Failed to write temp stitched image.")
            return {}
        # Use same interpreter (venv-safe) and absolute paths
        cmd = [sys.executable, str(SCAN_SCRIPT), str(
            tmp), str(WEIGHTS), str(DB_PATH)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(res.stdout)
    except subprocess.CalledProcessError as e:
        print("scan_and_store failed:", e.stderr or e.stdout)
        return {}
    except Exception as e:
        print("scan_and_store error:", e)
        return {}
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise SystemExit("Camera not available. Change CAM_INDEX.")

    front_img = None
    back_img = None
    info = "Press SPACE to capture FRONT label"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()
        guide = draw_guide_box(display, GUIDE_BOX_FRAC)

        # ----- YOLO detect bottle ------
        yolo_out = model.predict(source=frame, classes=[
                                 BOTTLE_CLASS_ID], conf=YOLO_CONF, verbose=False)
        box = choose_bottle_box(yolo_out)

        # ------ Draw detection box ------
        if box is not None:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(display, f"bottle {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)

        cv2.putText(display, info, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "SPACE: capture | R: reset | Q: quit", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow("Bottle Capture (Front + Back, then Stitch)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            front_img = None
            back_img = None
            info = "Reset. Press SPACE to capture FRONT label."

        elif key == ord(' '):
            # ------ only capture if bottle is present -----
            if box is None:
                info = "No bottle detected. Center it inside the green box."
                continue

            x1, y1, x2, y2, _ = box
            crop = safe_crop(frame, x1, y1, x2, y2)
            if crop is None:
                info = "Bad crop. Adjust bottle and try again."
                continue

            if front_img is None:
                front_img = crop
                info = "Front captured. Now rotate to BACK and press SPACE."
            elif back_img is None:
                back_img = crop
                # ----- Stitch in memory & show ------
                stitched = stitch_horizontal(
                    front_img, back_img)  # <------ PHOTO

                # >>> Auto-run your existing OCR+store CLI <<<
                result = run_scan_on_np(stitched)
                # Optional overlay
                maker = (result.get("maker_name") or "?")
                year = (result.get("vintage") or "?")
                seen = result.get("seen")
                tag = "SEEN" if seen else ("NEW" if seen is not None else "?")
                color = (0, 255, 0) if seen else (0, 200, 255)
                cv2.putText(stitched, f"{maker}  {year}  {tag}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                print(json.dumps(result or {}, indent=2))

                cv2.imshow("Stitched (Front | Back)", stitched)
                info = "Stitched shown. Press R to redo or Q to quit."
            else:
                info = "Already stitched. Press R to restart or Q to quit."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
