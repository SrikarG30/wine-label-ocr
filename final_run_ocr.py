# final_run_ocr.py
# One-call OCR: (image) -> (CustomID, MakerName, Vintage)
# - Caches YOLO and PaddleOCR once to avoid reloading GBs of weights
# - Accepts an image path or a cv2/numpy image
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import re
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- Lazy singletons to prevent memory spikes ----------
_YOLO = None
_YOLO_PATH = None

def _get_yolo(weights_path: Union[str, Path]) -> YOLO:
    """Load YOLO once and reuse."""
    global _YOLO, _YOLO_PATH
    wp = str(weights_path)
    if _YOLO is None or _YOLO_PATH != wp:
        _YOLO = YOLO(wp)
        _YOLO_PATH = wp
    return _YOLO

# PaddleOCR is optional at import; error only when we actually call OCR
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None  # type: ignore

_OCR = None
def _get_ocr():
    """Create PaddleOCR once and reuse. Compatible with builds that don't support show_log."""
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR
        import inspect
        kwargs = dict(lang='en', use_angle_cls=True)
        try:
            # pass show_log=False only if supported by this PaddleOCR
            if 'show_log' in inspect.signature(PaddleOCR.__init__).parameters:
                kwargs['show_log'] = False
        except Exception:
            pass
        try:
            _OCR = PaddleOCR(**kwargs)
        except TypeError:
            # very old builds: fall back to minimal ctor
            _OCR = PaddleOCR()
    return _OCR


# ---------- OCR helpers (adapted from your ocr_wine.py) ----------
def _pad_box(xyxy, img_w, img_h, pad=0.08):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * w)); y1 = max(0, int(y1 - pad * h))
    x2 = min(img_w, int(x2 + pad * w)); y2 = min(img_h, int(y2 + pad * h))
    return [x1, y1, x2, y2]

def _enhance_for_ocr(crop_bgr):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    g = cv2.bilateralFilter(g, d=7, sigmaColor=50, sigmaSpace=50)
    h, w = g.shape
    if max(h, w) < 250:
        g = cv2.resize(g, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    sharp = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
    binar = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    return sharp, binar

def _run_paddle_ocr(img_bgr) -> Tuple[List[str], List[float]]:
    if img_bgr is None or img_bgr.size == 0:
        return [], []
    if len(img_bgr.shape) == 2:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    result = _get_ocr().ocr(rgb)
    if not result:
        return [], []

    lines = result[0] if (isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple, dict))) else result
    texts, confs = [], []
    for line in lines or []:
        txt, score = None, None
        if isinstance(line, dict):
            txt = line.get('rec_text') or line.get('label') or line.get('text')
            score = line.get('rec_score') or line.get('score')
        elif isinstance(line, (list, tuple)):
            if len(line) >= 2 and isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                txt, score = line[1][0], line[1][1]
            elif len(line) >= 3 and isinstance(line[1], str):
                txt, score = line[1], line[2]
            elif len(line) >= 2 and isinstance(line[0], str):
                txt, score = line[0], line[1]
        if txt:
            try:
                conf = float(score) if score is not None else 0.0
            except Exception:
                conf = 0.0
            texts.append(txt.strip()); confs.append(conf)
    return texts, confs

def _best_ocr_text(crop_bgr):
    sharp, binar = _enhance_for_ocr(crop_bgr)
    t1, c1 = _run_paddle_ocr(sharp)
    t2, c2 = _run_paddle_ocr(binar)
    m1 = np.mean(c1) if c1 else 0.0
    m2 = np.mean(c2) if c2 else 0.0
    texts, confs = (t1, c1) if m1 >= m2 else (t2, c2)
    return " ".join(texts).strip(), (float(np.mean(confs)) if confs else 0.0)

def _extract_fields(image_bgr, detections):
    """Return {'maker_name': str|None, 'vintage': str|None, 'raw': {...}}"""
    H, W = image_bgr.shape[:2]
    out: Dict[str, Any] = {"maker_name": None, "vintage": None, "raw": {}}
    for det in detections:
        cls = det["class"]
        x1, y1, x2, y2 = _pad_box(det["box"], W, H, pad=0.08)
        crop = image_bgr[y1:y2, x1:x2]
        txt, conf = _best_ocr_text(crop)
        cls_lower = cls.replace("-", "_").lower()
        if cls_lower in ["maker_name", "producer", "winery"]:
            cleaned = "".join(ch for ch in txt if ch.isalnum() or ch in " &'-").upper()
            cleaned = " ".join(cleaned.split())
            if not out["maker_name"] or (cleaned and len(cleaned) > len(out["maker_name"])):
                out["maker_name"] = cleaned if cleaned else None
            out["raw"].setdefault("maker_name_candidates", []).append((cleaned, conf))
        elif cls_lower in ["vintage", "year"]:
            m = re.search(r"\b(19|20)\d{2}\b", txt)
            if m:
                out["vintage"] = m.group(0)
            else:
                digits = "".join(ch for ch in txt if ch.isdigit())
                if len(digits) >= 4:
                    for i in range(0, len(digits) - 3):
                        cand = digits[i:i+4]
                        if cand.startswith(("19", "20")):
                            out["vintage"] = cand
                            break
            out["raw"].setdefault("vintage_candidates", []).append((txt, conf))
        else:
            out["raw"].setdefault(cls_lower, []).append((txt, conf))
    return out

# ---------- Post-processing helpers ----------
ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 &'-")

def _normalize_maker(s: Optional[str]) -> str:
    if not s: return ""
    s = s.upper()
    s = "".join(ch for ch in s if ch in ALLOWED)
    return " ".join(s.split())

def _extract_best_maker_from_raw(raw: Dict[str, Any]) -> str:
    cands = [txt for (txt, _sc) in raw.get("maker_name_candidates", []) if txt]
    for k, arr in raw.items():
        if k == "maker_name_candidates": continue
        for txt, _sc in arr:
            if isinstance(txt, str) and len(txt.strip()) >= 3 and txt.strip() != ".":
                cands.append(txt)
    cands = [_normalize_maker(t) for t in cands if t]
    cands = [t for t in cands if t]
    return max(cands, key=len) if cands else ""

def _to_int_year(v: Optional[Union[str, int]]) -> Optional[int]:
    if v is None: return None
    s = str(v).strip()
    return int(s) if (len(s) == 4 and s.isdigit()) else None

# ---------- Public API ----------
def final_run_ocr(
    image: Union[str, np.ndarray],
    weights_path: Union[str, Path],
    id_to_name: Dict[int, str] = None
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Args:
        image: path to image or BGR numpy array
        weights_path: path to your YOLO/Roboflow weights (e.g., 'weights.pt')
        id_to_name: optional class map; default matches your training
                    {0:'Distinct Logo', 1:'Maker-Name', 2:'Vintage'}

    Returns:
        (CustomID, MakerName, Vintage)
        - CustomID = "MakerName|Vintage" when both exist else None
        - MakerName = normalized uppercase or None
        - Vintage = 4-digit int or None
    """
    # Load image
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        img = image
        if img is None or not hasattr(img, "shape"):
            raise ValueError("image must be a path or a cv2/numpy image")

    # YOLO inference (cached model)
    model = _get_yolo(weights_path)
    pred = model(img, verbose=False)[0]

    # Class mapping
    if id_to_name is None:
        id_to_name = {0: "Distinct Logo", 1: "Maker-Name", 2: "Vintage"}

    # Build detections list for OCR
    detections = []
    if pred.boxes is not None:
        for b in pred.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            cls_id = int(b.cls[0])
            detections.append({
                "class": id_to_name.get(cls_id, str(cls_id)),
                "box": [x1, y1, x2, y2]
            })

    # OCR on detected regions
    fields = _extract_fields(img, detections)
    maker_raw = fields.get("maker_name")
    raw = fields.get("raw") or {}

    maker_norm = _normalize_maker(maker_raw) if maker_raw else ""
    if not maker_norm:
        maker_norm = _extract_best_maker_from_raw(raw)
    maker_out = maker_norm or None

    vintage_int = _to_int_year(fields.get("vintage"))
    custom_id = f"{maker_out}|{vintage_int}" if (maker_out and vintage_int) else None
    return custom_id, maker_out, vintage_int

# ---------- CLI ----------
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) != 3:
        print("Usage: python final_run_ocr.py <image_path> <weights_path>")
        sys.exit(1)
    c, m, v = final_run_ocr(sys.argv[1], sys.argv[2])
    print(json.dumps({"CustomID": c, "MakerName": m, "Vintage": v}, indent=2))
