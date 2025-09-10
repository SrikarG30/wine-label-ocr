"""
bottle_singlepass_stop_on_seen_preview.py
-----------------------------------------
Single-pass pipeline that:
- Enforces MAX 1 bottle per frame (yolom11.pt).
- Segments labels (label-segmentation.pt) and overlays filled masks.
- Runs classifier.pt on Top 30% / Bottom 70% of bottle ROI + each label crop (masked only; no rotation).
- Clips detections to bottle ROI and color-codes boxes per class.
- Shows a static right legend (~800 px wide) with spaced class counters.
- Builds a fingerprint (classifier signature + HSV hist inside best label mask + simple geometry),
  compares against a persistent bottle_library.json, and
  **stops early** (saving partial annotated video to ./output/) if bottle is recognized as already seen.
- Live preview window (press 'q' to stop).

Deps:
    pip install ultralytics opencv-python numpy
Files in the same folder:
    wine.avi, yolom11.pt, label-segmentation.pt, (optional) classifier.pt
"""

import os, json, tempfile, time
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ============================== Config ==================================
VIDEO_PATH        = "wine-4.avi"
DET_MODEL_PATH    = "yolo11m.pt"             # bottle detector
SEG_MODEL_PATH    = "label-segmentation.pt"  # label instance segmentation
CLS_MODEL_PATH    = "classifier.pt"          # optional parts detector
LIB_PATH          = "bottle_library.json"    # persistent library (JSON)

# Detector / segmenter thresholds
DET_CONF          = 0.30
SEG_CONF          = 0.25
SEG_IOU           = 0.60

# ROI expansion (analysis only)
BBOX_MARGIN       = 0.08

# Process cadence
PROCESS_EVERY_N   = 1

# Output
OUT_DIR           = "output"
OUT_FPS_FALLBACK  = 30.0

# Live preview
SHOW_PREVIEW      = True   # set False to disable preview window

# Simple tracker params
TRACK_IOU_THR     = 0.30
TRACK_MAX_AGE     = 45
TRACK_MIN_HITS    = 2

# Re-ID threshold & weights
REID_MATCH_THR    = 0.80
W_CLS             = 0.55   # classifier signature cosine (dict-based)
W_COLOR           = 0.35   # 1/(1+chi2) on HSV hist inside best label mask
W_GEOM            = 0.10   # similarity of [num_labels, max_area%]

# UI
FONT              = cv2.FONT_HERSHEY_SIMPLEX
FS_OVERLAY        = 1.1
THICK             = 2
ALPHA_MASK        = 0.45
COLOR_MASK_FILL   = (80,160,255)

# Legend panel
PANEL_W           = 800
PANEL_BG          = (20,20,20)
PANEL_TXT         = (255,255,255)
PANEL_PAD_OUTER   = 16
PANEL_PADX        = 22
PANEL_PADY        = 22
LINE_DY_HEADER    = 48
LINE_DY_SUBHEAD   = 38
LINE_DY_CLASS     = 42
SWATCH_W, SWATCH_H = 22, 22

# 'distinctlogo' gates
LOGO_MIN_REL_AREA = 0.01
LOGO_MAX_REL_AREA = 0.15
LOGO_BAND_TOP     = 0.10
LOGO_BAND_BOTTOM  = 0.90
TEXTINESS_VETO_T  = 0.52

# When to decide "already seen?"
EARLY_PRINT_AFTER_FRAMES = 18

# ========================================================================

def clamp(v, lo, hi): return max(lo, min(hi, v))

def expand_bbox(xyxy, W, H, m):
    x1,y1,x2,y2 = map(int, xyxy)
    bw, bh = x2-x1, y2-y1
    dx, dy = int(bw*m), int(bh*m)
    return clamp(x1-dx,0,W-1), clamp(y1-dy,0,H-1), clamp(x2+dx,0,W-1), clamp(y2+dy,0,H-1)

def iou_box(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = max(0,ax2-ax1)*max(0,ay2-ay1) + max(0,bx2-bx1)*max(0,by2-by1) - inter
    return inter/ua if ua>0 else 0.0

def cosine_sim_from_dicts(a: Dict[str,float], b: Dict[str,float]) -> float:
    if not a or not b: return 0.0
    keys = set(a.keys()) | set(b.keys())
    v1 = np.array([a.get(k,0.0) for k in keys], dtype=np.float32)
    v2 = np.array([b.get(k,0.0) for k in keys], dtype=np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 0.0
    return float(np.dot(v1,v2) / (n1*n2))

def chi2_dist(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    num = (p - q) ** 2
    den = p + q + eps
    return float(0.5 * np.sum(num / den))

def class_color(name: str) -> Tuple[int,int,int]:
    import hashlib
    if not name: name = "unknown"
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    hue = (h % 180); sat = 200 + (h % 55); val = 180 + ((h//2) % 75)
    color_bgr = tuple(int(c) for c in cv2.cvtColor(
        np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR
    )[0,0])
    return color_bgr

def draw_filled_box(img, x1,y1,x2,y2, label_text, color_bgr, alpha=0.20):
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color_bgr, -1)
    img[:] = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
    cv2.rectangle(img, (x1,y1), (x2,y2), color_bgr, 2)
    (tw, th), _ = cv2.getTextSize(label_text, FONT, FS_OVERLAY, THICK)
    chip_w, chip_h = tw + 10, th + 10
    cx1, cy1 = x1, max(0, y1 - chip_h - 3)
    cx2, cy2 = cx1 + chip_w, cy1 + chip_h
    cv2.rectangle(img, (cx1,cy1), (cx2,cy2), color_bgr, -1)
    b,g,r = color_bgr
    luminance = 0.299*r + 0.587*g + 0.114*b
    tcol = (0,0,0) if luminance > 160 else (255,255,255)
    cv2.putText(img, label_text, (cx1+5, cy2-5), FONT, FS_OVERLAY, tcol, THICK, cv2.LINE_AA)

def clip_box_to_roi(box_xyxy, roi_rect):
    x1,y1,x2,y2 = map(int, box_xyxy)
    rx1,ry1,rx2,ry2 = roi_rect
    X1 = clamp(x1, rx1, rx2-1)
    Y1 = clamp(y1, ry1, ry2-1)
    X2 = clamp(x2, rx1, rx2-1)
    Y2 = clamp(y2, ry1, ry2-1)
    if X2 <= X1 or Y2 <= Y1: return (0,0,0,0), False
    return (X1,Y1,X2,Y2), True

def rasterize_polygon(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    m = np.zeros((h,w), np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(m, [poly.astype(np.int32).reshape(-1,1,2)], 1)
    return m.astype(bool)

def textiness_native(gray_img: np.ndarray, mask_bool: np.ndarray) -> float:
    if gray_img is None or mask_bool is None or not mask_bool.any():
        return 0.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray_img)
    vals = g[mask_bool]
    if vals.size < 50: return 0.0
    thr,_ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bi = ((g >= thr).astype(np.uint8) & mask_bool.astype(np.uint8)) * 255
    if bi[mask_bool].mean() < 127:
        bi[mask_bool] = 255 - bi[mask_bool]
    ft = (bi>0).astype(np.uint8)
    num,_,stats,_ = cv2.connectedComponentsWithStats(ft, 8)
    cnt=0
    for i in range(1,num):
        x,y,w,h,a = stats[i]
        if 20<=a<=7000 and 0.2<=w/max(h,1)<=6.0: cnt+=1
    cc = cnt / max(mask_bool.sum()/1000.0, 1.0)
    rows = (ft>0).sum(axis=1).astype(np.float32)
    m = rows.mean(); s = rows.std()
    peak = 0.0 if m<1e-6 else float(np.tanh((s/m)/2.0))
    dist = cv2.distanceTransform(ft, cv2.DIST_L2, 3)
    dv = dist[ft>0]
    swc = 0.0 if dv.size==0 else float(1.0 / (1.0 + dv.std()/(dv.mean()+1e-9)))
    return float(max(0.0, min(1.0, 0.45*cc + 0.30*peak + 0.25*swc)))

# ---------------------- Simple IOU Tracker ----------------------
class SimpleTrack:
    def __init__(self, box, tid):
        self.box = np.array(box, dtype=float)
        self.id = tid
        self.hits = 1
        self.age = 0
        self.confirmed = False
        # aggregates
        self.cls_sig: Dict[str,float] = {}
        self.num_labels_seen = 0
        self.max_label_area_pct = 0.0
        self.frames_accumulated = 0
        self.last_roi_frame: Optional[np.ndarray] = None

        # Paired best label snapshot (prevents mask/image size mismatch)
        self.best_label_mask = None        # ROI-local bool/uint8 mask (HxW of the saved ROI)
        self.best_label_roi  = None        # BGR ROI image captured with the mask
        self.best_label_rect = None        # (rx1, ry1, rx2, ry2)
        self.best_label_area_pct = 0.0

class SimpleTracker:
    def __init__(self, iou_thr=0.3, max_age=45, min_hits=2):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: Dict[int, SimpleTrack] = {}
        self.next_id = 1
    def update(self, dets_xyxy_conf: np.ndarray):
        for t in self.tracks.values(): t.age += 1
        for det in dets_xyxy_conf:
            box = det[:4]
            best_iou, best_id = 0.0, None
            for tid, tr in self.tracks.items():
                iou = iou_box(box, tr.box)
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_iou >= self.iou_thr:
                tr = self.tracks[best_id]
                tr.box = box; tr.hits += 1; tr.age = 0
                tr.confirmed = tr.hits >= self.min_hits
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = SimpleTrack(box, tid)
        dead = [tid for tid,t in self.tracks.items() if t.age > self.max_age]
        for tid in dead: del self.tracks[tid]
        out=[]
        for tid, tr in self.tracks.items():
            x1,y1,x2,y2 = tr.box.astype(int).tolist()
            out.append((x1,y1,x2,y2, tid, tr.confirmed))
        return out

# ----------------------- Models & helpers ----------------------
def get_bottle_id(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for k,v in items:
        if "bottle" in str(v).lower():
            return int(k)
    return None

def run_classifier(model: Optional[YOLO], img_bgr: np.ndarray):
    if model is None: return []
    r = model.predict(source=img_bgr, conf=0.25, iou=0.50, imgsz=960, verbose=False, stream=False)[0]
    out=[]
    if r.boxes is None or len(r.boxes)==0: return out
    xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    if getattr(r.boxes, "cls", None) is not None:
        cls = r.boxes.cls.cpu().numpy().astype(int)
    else:
        cls = np.zeros_like(conf, dtype=int)
    names = r.names if hasattr(r, "names") else {}
    for i in range(len(conf)):
        nm = str(names.get(int(cls[i]), f"class{int(cls[i])}")).lower().replace(" ","")
        out.append((xyxy[i], float(conf[i]), nm))
    return out

def get_classifier_classes(model: Optional[YOLO]) -> List[str]:
    if model is None: return []
    try:
        names = model.model.names if hasattr(model, "model") else model.names
    except Exception:
        names = None
    if not names: return []
    items = names.items() if isinstance(names, dict) else enumerate(names)
    cls = [str(v).lower().replace(" ","") for _,v in items]
    return sorted(list(set(cls)))

def calc_hsv_hist(img_bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2], mask, [8,8,8], [0,180, 0,256, 0,256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    return hist.astype(np.float32)

def geom_vec(num_labels_seen: int, max_label_area_pct: float) -> np.ndarray:
    n = min(num_labels_seen, 4) / 4.0
    a = max(0.0, min(1.0, max_label_area_pct/100.0))
    return np.array([n, a], dtype=np.float32)

def fingerprint_similarity(cls_dict_A: Dict[str,float],
                           hsv_hist_A: np.ndarray,
                           geom_A: np.ndarray,
                           cls_dict_B: Dict[str,float],
                           hsv_hist_B: np.ndarray,
                           geom_B: np.ndarray) -> float:
    s_cls = cosine_sim_from_dicts(cls_dict_A, cls_dict_B)
    d = chi2_dist(hsv_hist_A, hsv_hist_B) if hsv_hist_A.size and hsv_hist_B.size else 1.0
    s_col = 1.0 / (1.0 + d)
    s_geom = float(1.0 - np.mean(np.minimum(1.0, np.abs(geom_A - geom_B)))) if geom_A.size and geom_B.size else 0.0
    score = W_CLS*s_cls + W_COLOR*s_col + W_GEOM*s_geom
    return float(max(0.0, min(1.0, score)))

# ----------------------------- Legend panel ------------------------------
def class_color_chip(name: str) -> Tuple[int,int,int]:
    return class_color(name)

def draw_legend_panel(canvas: np.ndarray, classes: List[str], cum_counts: Dict[str,int], frame_idx: int):
    H, W = canvas.shape[:2]
    x0 = W - PANEL_W - PANEL_PAD_OUTER
    x1 = W - PANEL_PAD_OUTER
    y0, y1 = PANEL_PAD_OUTER, H - PANEL_PAD_OUTER
    cv2.rectangle(canvas, (x0, y0), (x1, y1), PANEL_BG, -1)
    x = x0 + PANEL_PADX
    y = y0 + PANEL_PADY
    cv2.putText(canvas, f"Frame {frame_idx}", (x, y), FONT, 1.8, PANEL_TXT, 3, cv2.LINE_AA); y += LINE_DY_HEADER
    if classes:
        cv2.putText(canvas, "Classifier counts (cumulative):", (x, y), FONT, 1.4, PANEL_TXT, 3, cv2.LINE_AA); y += LINE_DY_SUBHEAD
        for name in classes:
            col = class_color_chip(name)
            cv2.rectangle(canvas, (x, y-SWATCH_H+10), (x+SWATCH_W, y+10), col, -1)
            cnt = int(cum_counts.get(name, 0))
            cv2.putText(canvas, f"{name:<20} n={cnt}", (x+SWATCH_W+14, y),
                        FONT, 1.3, PANEL_TXT, 3, cv2.LINE_AA)
            y += LINE_DY_CLASS
    else:
        cv2.putText(canvas, "classifier.pt not found", (x, y), FONT, 1.4, (200,200,200), 3, cv2.LINE_AA)

# -------------------------- Persistence helpers --------------------------
def load_library(path: str) -> List[dict]:
    if not os.path.isfile(path): return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def save_library(path: str, entries: List[dict]) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="libtmp_", suffix=".json")
    os.close(tmp_fd)
    with open(tmp_path, "w") as f:
        json.dump(entries, f)
    os.replace(tmp_path, path)

# ================================ Main ====================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"result_{timestamp}.mp4")

    # Load models
    det = YOLO(DET_MODEL_PATH)
    seg = YOLO(SEG_MODEL_PATH)
    cls = YOLO(CLS_MODEL_PATH) if os.path.isfile(CLS_MODEL_PATH) else None
    cls_names = get_classifier_classes(cls)
    cum_counts = {n: 0 for n in cls_names}

    # bottle class id (optional)
    try:
        BOTTLE_ID = get_bottle_id(det.model.names)
    except Exception:
        BOTTLE_ID = None

    # persistent library
    library = load_library(LIB_PATH)  # list of {"cls":{...}, "hsv":[...], "geom":[...], "seen_count":int}

    # Video IO
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {VIDEO_PATH}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or OUT_FPS_FALLBACK
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = OUT_FPS_FALLBACK

    out_size = (W + PANEL_W + PANEL_PAD_OUTER*2, H)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)

    tracker = SimpleTracker(iou_thr=TRACK_IOU_THR, max_age=TRACK_MAX_AGE, min_hits=TRACK_MIN_HITS)

    frame_idx = -1
    stop_now = False

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        vis = frame.copy()
        cv2.putText(vis, f"Frame {frame_idx}", (12, 30), FONT, FS_OVERLAY, (255,255,0), THICK, cv2.LINE_AA)

        # -------- Detect bottle (MAX 1) --------
        dets = []
        r = det.predict(source=frame, conf=DET_CONF, verbose=False, stream=False, max_det=1)[0]
        if r.boxes is not None and len(r.boxes)>0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            if getattr(r.boxes,"cls",None) is not None:
                cls_det = r.boxes.cls.cpu().numpy().astype(int)
            else:
                cls_det = np.zeros_like(conf, dtype=int)
            for b, c, k in zip(xyxy, conf, cls_det):
                if (BOTTLE_ID is None) or (k==BOTTLE_ID):
                    dets.append([int(b[0]),int(b[1]),int(b[2]),int(b[3]), float(c)])
        dets = np.array(dets, dtype=float) if dets else np.zeros((0,5), dtype=float)
        if dets.shape[0] > 1:
            order = np.argsort(dets[:,4])[::-1]
            dets = dets[order[:1]]

        # -------- Update tracker --------
        tracks = tracker.update(dets)

        # -------- Analyze confirmed track --------
        for (bx1,by1,bx2,by2, tid, confirmed) in tracks:
            col = (0,255,0) if confirmed else (128,128,128)
            cv2.rectangle(vis, (bx1,by1), (bx2,by2), col, 2)
            cv2.putText(vis, f"Track {tid} {'CONF' if confirmed else '...'}", (bx1, max(0,by1-8)),
                        FONT, FS_OVERLAY, col, THICK, cv2.LINE_AA)

            rx1,ry1,rx2,ry2 = expand_bbox((bx1,by1,bx2,by2), W,H, BBOX_MARGIN)
            roi = frame[ry1:ry2, rx1:rx2]
            rh,rw = roi.shape[:2]
            if rh<8 or rw<8: continue
            cv2.rectangle(vis, (rx1,ry1), (rx2,ry2), (0,180,255), 1)

            tr = tracker.tracks[tid]
            tr.last_roi_frame = roi.copy()

            if frame_idx % PROCESS_EVERY_N != 0:
                # compose canvas with legend
                canvas = np.zeros((H, W + PANEL_W + PANEL_PAD_OUTER*2, 3), np.uint8)
                canvas[:, :W] = vis
                draw_legend_panel(canvas, cls_names, cum_counts, frame_idx)
                writer.write(canvas)
                if SHOW_PREVIEW:
                    cv2.imshow("Preview", canvas)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_now = True
                        break
                if stop_now: break
                continue

            roi_area = float(max(1, rw*rh))

            # Split: top 30% / bottom 70%
            h_top = int(round(0.30*rh))
            y_top0, y_bot0 = 0, h_top
            cv2.line(vis, (rx1, ry1+y_bot0), (rx2, ry1+y_bot0), (0,255,255), 2)
            cv2.putText(vis, "TOP 30%", (rx2+4, ry1+min(22, max(12,h_top-6))), FONT, 0.7, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(vis, "BOT 70%", (rx2+4, ry1+y_bot0+min(22, max(12,(rh-h_top)-6))), FONT, 0.7, (0,255,255), 2, cv2.LINE_AA)

            # Segment labels within ROI
            masks=[]; polys_full=[]
            s = seg.predict(source=roi, conf=SEG_CONF, iou=SEG_IOU, verbose=False, stream=False)[0]
            if getattr(s,"masks",None) is not None and s.masks is not None and s.masks.data is not None:
                mstack = s.masks.data.cpu().numpy()
                for m in mstack:
                    rr = cv2.resize(m,(rw,rh),interpolation=cv2.INTER_NEAREST)
                    cnts,_=cv2.findContours((rr>0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for c in cnts:
                        pr = c.reshape(-1,2).astype(np.float32)
                        if len(pr)>=3:
                            mask = np.zeros((rh,rw), np.uint8)
                            cv2.fillPoly(mask, [pr.astype(np.int32).reshape(-1,1,2)], 1)
                            masks.append(mask.astype(bool))
                            pf = pr.copy(); pf[:,0]+=rx1; pf[:,1]+=ry1
                            polys_full.append(pf)

            # Overlay label masks (filled)
            for pf in polys_full:
                ov = vis.copy()
                cv2.fillPoly(ov, [pf.astype(np.int32).reshape(-1,1,2)], COLOR_MASK_FILL)
                vis = cv2.addWeighted(ov, ALPHA_MASK, vis, 1-ALPHA_MASK, 0)

            # Classifier passes
            boxes_top, boxes_bot = [], []
            if cls is not None:
                top_img = roi[y_top0:y_top0+h_top, :]
                bot_img = roi[y_bot0:rh, :]
                boxes_top = run_classifier(cls, top_img)
                boxes_bot = run_classifier(cls, bot_img)

            # Iterate labels
            for mask in masks:
                area = float(mask.sum())
                if area <= 10: continue

                ys, xs = np.where(mask)
                lbx, lby = int(xs.min()), int(ys.min())
                lbw, lbh = int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
                label_bbox_area = float(max(1, lbw*lbh))
                area_pct = 100.0 * area / roi_area

                tr.max_label_area_pct = max(tr.max_label_area_pct, area_pct)
                # Save paired best label snapshot
                if tr.best_label_mask is None or area_pct > tr.best_label_area_pct:
                    tr.best_label_mask = mask.copy()
                    tr.best_label_roi  = roi.copy()
                    tr.best_label_rect = (rx1, ry1, rx2, ry2)
                    tr.best_label_area_pct = area_pct

                # Masked label crop (no rotation)
                crop = roi[lby:lby+lbh, lbx:lbx+lbw].copy()
                crop_mask = mask[lby:lby+lbh, lbx:lbx+lbw]
                crop[~crop_mask] = 127
                boxes_crop = run_classifier(cls, crop) if cls is not None else []

                for (cx1,cy1,cx2,cy2), cconf, cname in boxes_crop:
                    ccx = clamp(int(0.5*(cx1+cx2)), 0, crop_mask.shape[1]-1)
                    ccy = clamp(int(0.5*(cy1+cy2)), 0, crop_mask.shape[0]-1)
                    if not crop_mask[ccy, ccx]: continue

                    RX1, RY1 = cx1 + lbx + rx1, cy1 + lby + ry1
                    RX2, RY2 = cx2 + lbx + rx1, cy2 + lby + ry1

                    accept = True
                    if cname == "distinctlogo":
                        rel_y = (ccy / max(lbh,1))
                        patch_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)[lby: lby+lbh, lbx: lbx+lbw]
                        texty = textiness_native(patch_gray, crop_mask)
                        box_area = float(max(0, RX2-RX1) * max(0, RY2-RY1))
                        rel_area = box_area / max(label_bbox_area,1.0)
                        if not (LOGO_BAND_TOP <= rel_y <= LOGO_BAND_BOTTOM): accept = False
                        if rel_area < LOGO_MIN_REL_AREA or rel_area > LOGO_MAX_REL_AREA: accept = False
                        if texty > TEXTINESS_VETO_T: accept = False

                    if accept:
                        clipped, valid = clip_box_to_roi((RX1,RY1,RX2,RY2), (rx1,ry1,rx2,ry2))
                        if not valid: continue
                        color = class_color(cname)
                        draw_filled_box(vis, *clipped, f"{cname}:{cconf:.2f}", color)

                        # update cumulative legend counts and signature
                        if cname in cum_counts:
                            cum_counts[cname] += 1
                        tr.cls_sig[cname] = tr.cls_sig.get(cname, 0.0) + float(cconf)

            # Gate slice boxes by union mask (simple)
            if cls is not None and masks:
                rh, rw = roi.shape[:2]
                mask_union = np.zeros((rh, rw), np.uint8)
                for m in masks: mask_union |= m.astype(np.uint8)

                # TOP slice
                for (tx1,ty1,tx2,ty2), tconf, tname in boxes_top:
                    tcx = clamp(int(0.5*(tx1+tx2)), 0, rw-1)
                    tcy = clamp(int(0.5*(ty1+ty2)), 0, int(round(0.30*rh))-1)
                    if not mask_union[tcy, tcx]: continue
                    RX1, RY1 = tx1 + rx1, ty1 + ry1
                    RX2, RY2 = tx2 + rx1, ty2 + ry1
                    clipped, valid = clip_box_to_roi((RX1,RY1,RX2,RY2), (rx1,ry1,rx2,ry2))
                    if not valid: continue
                    color = class_color(tname)
                    draw_filled_box(vis, *clipped, f"{tname}:{tconf:.2f}", color)
                    if tname in cum_counts: cum_counts[tname] += 1
                    tr.cls_sig[tname] = tr.cls_sig.get(tname, 0.0) + float(tconf)

                # BOTTOM slice
                for (bx1_,by1_,bx2_,by2_), bconf, bname in boxes_bot:
                    bcx = clamp(int(0.5*(bx1_+bx2_)), 0, rw-1)
                    bcy = clamp(int(0.5*(by1_+by2_)), 0, rh - int(round(0.30*rh)) - 1)
                    if not mask_union[bcy + int(round(0.30*rh)), bcx]: continue
                    RX1, RY1 = bx1_ + rx1, by1_ + int(round(0.30*rh)) + ry1
                    RX2, RY2 = bx2_ + rx1, by2_ + int(round(0.30*rh)) + ry1
                    clipped, valid = clip_box_to_roi((RX1,RY1,RX2,RY2), (rx1,ry1,rx2,ry2))
                    if not valid: continue
                    color = class_color(bname)
                    draw_filled_box(vis, *clipped, f"{bname}:{bconf:.2f}", color)
                    if bname in cum_counts: cum_counts[bname] += 1
                    tr.cls_sig[bname] = tr.cls_sig.get(bname, 0.0) + float(bconf)

            # Track state updates
            tr.frames_accumulated += 1
            tr.num_labels_seen = max(tr.num_labels_seen, len(masks))

            # Decide "already seen?" and STOP if matched
            if confirmed and tr.frames_accumulated >= EARLY_PRINT_AFTER_FRAMES and not stop_now:
                # Build fingerprint snapshot
                if tr.best_label_mask is not None and tr.best_label_roi is not None:
                    mask_u8 = (tr.best_label_mask.astype(np.uint8)) * 255
                    hsv_hist = calc_hsv_hist(tr.best_label_roi, mask_u8)
                elif tr.last_roi_frame is not None:
                    hsv_hist = calc_hsv_hist(tr.last_roi_frame, None)
                else:
                    hsv_hist = np.zeros((8*8*8,), np.float32)

                geom = geom_vec(tr.num_labels_seen, tr.max_label_area_pct)

                best_score, best_idx = 0.0, -1
                for i, fp in enumerate(library):
                    cls_old = fp.get("cls", {})
                    hsv_old = np.array(fp.get("hsv", []), dtype=np.float32)
                    geom_old = np.array(fp.get("geom", []), dtype=np.float32)
                    s = fingerprint_similarity(tr.cls_sig, hsv_hist, geom, cls_old, hsv_old, geom_old)
                    if s > best_score:
                        best_score, best_idx = s, i

                if best_idx >= 0 and best_score >= REID_MATCH_THR:
                    library[best_idx]["seen_count"] = int(library[best_idx].get("seen_count", 1)) + 1
                    print(f">>> Already seen bottle? YES (confidence={best_score:.3f}) | seen_count={library[best_idx]['seen_count']}")
                    stop_now = True   # early stop on recognized bottle

        # Compose canvas with legend & write
        canvas = np.zeros((H, W + PANEL_W + PANEL_PAD_OUTER*2, 3), np.uint8)
        canvas[:, :W] = vis
        draw_legend_panel(canvas, cls_names, cum_counts, frame_idx)
        writer.write(canvas)

        if SHOW_PREVIEW:
            cv2.imshow("Preview", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_now = True

        if stop_now:
            break

    # --------------- finalize & persist library ----------------
    # If we stopped early because bottle was already seen, do NOT add a new entry.
    # If we ran to completion without a match, add or merge the entry.
    if not stop_now and tracker.tracks:
        for tid, tr in list(tracker.tracks.items()):
            if tr.frames_accumulated >= max(10, EARLY_PRINT_AFTER_FRAMES//2):
                if tr.best_label_mask is not None and tr.best_label_roi is not None:
                    mask_u8 = (tr.best_label_mask.astype(np.uint8)) * 255
                    hsv_hist = calc_hsv_hist(tr.best_label_roi, mask_u8)
                elif tr.last_roi_frame is not None:
                    hsv_hist = calc_hsv_hist(tr.last_roi_frame, None)
                else:
                    hsv_hist = np.zeros((8*8*8,), np.float32)
                geom = geom_vec(tr.num_labels_seen, tr.max_label_area_pct)

                best_score, best_idx = 0.0, -1
                for i, fp in enumerate(library):
                    cls_old = fp.get("cls", {})
                    hsv_old = np.array(fp.get("hsv", []), dtype=np.float32)
                    geom_old = np.array(fp.get("geom", []), dtype=np.float32)
                    s = fingerprint_similarity(tr.cls_sig, hsv_hist, geom, cls_old, hsv_old, geom_old)
                    if s > best_score:
                        best_score, best_idx = s, i

                if best_idx >= 0 and best_score >= REID_MATCH_THR:
                    entry = library[best_idx]
                    entry_cls = entry.get("cls", {})
                    for k,v in tr.cls_sig.items():
                        entry_cls[k] = float(entry_cls.get(k,0.0) + v)
                    entry["cls"] = entry_cls
                    old_hsv = np.array(entry.get("hsv", []), dtype=np.float32)
                    old_geom = np.array(entry.get("geom", []), dtype=np.float32)
                    entry["hsv"]  = ((old_hsv + hsv_hist)/2.0).tolist() if old_hsv.size==hsv_hist.size and old_hsv.size>0 else hsv_hist.tolist()
                    entry["geom"] = ((old_geom + geom)/2.0).tolist()     if old_geom.size==geom.size and old_geom.size>0 else geom.tolist()
                    entry["seen_count"] = int(entry.get("seen_count", 1)) + 1
                else:
                    library.append({
                        "cls": tr.cls_sig,
                        "hsv": hsv_hist.tolist(),
                        "geom": geom.tolist(),
                        "seen_count": 1
                    })

    save_library(LIB_PATH, library)

    # Cleanup
    cap.release()
    writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    print(f"Saved annotated video: {out_path}")
    print(f"Library saved: {LIB_PATH} | entries={len(library)}")

if __name__ == "__main__":
    main()
