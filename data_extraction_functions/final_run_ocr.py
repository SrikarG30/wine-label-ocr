# Enhanced OCR with fallback vintage detection when YOLO fails
# Includes both targeted detection and full-image OCR fallback

from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import re
import cv2
import numpy as np
from ultralytics import YOLO

# --- logging silence helpers for PaddleOCR ---
import logging, contextlib, sys, io, inspect

@contextlib.contextmanager
def _silence_ppocr_and_stdout(level=logging.ERROR):
    """Mute PaddleOCR logs and any print()s it does inside the block."""
    targets = [
        "ppocr", "ppocr.utils", "ppocr.data",
        "ppocr.postprocess", "ppocr.infer"
    ]
    saved = []
    for name in targets:
        lg = logging.getLogger(name)
        saved.append((lg, lg.level, lg.propagate, list(lg.handlers)))
        lg.setLevel(level)
        lg.propagate = False
        lg.handlers = [logging.NullHandler()]
    old_out, old_err = sys.stdout, sys.stderr
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        sys.stdout, sys.stderr = buf_out, buf_err
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        for lg, lvl, prop, handlers in saved:
            lg.setLevel(lvl)
            lg.propagate = prop
            lg.handlers = handlers

# ---------- Lazy singletons ----------
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

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

_OCR = None
def _get_ocr():
    """Create PaddleOCR once and reuse."""
    global _OCR
    if _OCR is None:
        if PaddleOCR is None:
            from paddleocr import PaddleOCR as _POCR
        else:
            _POCR = PaddleOCR

        kwargs = dict(lang='en', use_angle_cls=True)
        try:
            if 'show_log' in inspect.signature(_POCR.__init__).parameters:
                kwargs['show_log'] = False
        except Exception:
            pass

        with _silence_ppocr_and_stdout():
            try:
                _OCR = _POCR(**kwargs)
            except TypeError:
                _OCR = _POCR()
    return _OCR

# ---------- OCR helpers ----------
def _pad_box(xyxy, img_w, img_h, pad=0.08):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * w))
    y1 = max(0, int(y1 - pad * h))
    x2 = min(img_w, int(x2 + pad * w))
    y2 = min(img_h, int(y2 + pad * h))
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

    with _silence_ppocr_and_stdout():
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
            texts.append(txt.strip())
            confs.append(conf)
    return texts, confs

def _best_ocr_text(crop_bgr):
    sharp, binar = _enhance_for_ocr(crop_bgr)
    t1, c1 = _run_paddle_ocr(sharp)
    t2, c2 = _run_paddle_ocr(binar)
    m1 = float(np.mean(c1)) if c1 else 0.0
    m2 = float(np.mean(c2)) if c2 else 0.0
    texts, confs = (t1, c1) if m1 >= m2 else (t2, c2)
    return " ".join(texts).strip(), (float(np.mean(confs)) if confs else 0.0)

def _extract_year_from_text(txt: str) -> Optional[str]:
    """Enhanced year extraction with multiple patterns."""
    if not txt:
        return None
    
    # Pattern 1: Standard 4-digit year
    patterns = [
        r'\b(19[0-9]{2}|20[0-9]{2})\b',  # 1900-2099
        r'(?:VIN|VINTAGE|VNT|V\.)\s*(\d{4})',
        r'(\d{4})(?:\s*(?:VINTAGE|VIN))',  
        r'[^\d](\d{4})[^\d]',  # Isolated 4 digits
    ]
    
    for pattern in patterns:
        m = re.search(pattern, txt, re.IGNORECASE)
        if m:
            year_str = m.group(1) if '(' in pattern else m.group(0)
            year_str = re.sub(r'\D', '', year_str)  # Keep only digits
            if len(year_str) == 4 and year_str.startswith(('19', '20')):
                year_int = int(year_str)
                # Validate reasonable wine vintage range
                if 1900 <= year_int <= 2030:
                    return year_str
    
    # Fallback: Look for any 4 consecutive digits starting with 19 or 20
    digits = re.findall(r'\d{4}', txt)
    for d in digits:
        if d.startswith(('19', '20')):
            year_int = int(d)
            if 1900 <= year_int <= 2030:
                return d
    
    return None

def _full_image_vintage_search(image_bgr) -> Optional[str]:
    """
    Enhanced fallback: Run OCR on multiple preprocessed versions to find vintage.
    """
    H, W = image_bgr.shape[:2]
    
    # Try multiple preprocessing approaches
    versions_to_try = []
    
    # 1. Original image
    versions_to_try.append(("original", image_bgr))
    
    # 2. Enhance red channel (for red wine labels)
    b, g, r = cv2.split(image_bgr)
    red_enhanced = cv2.merge([b*0.5, g*0.5, r*1.5])
    versions_to_try.append(("red_enhanced", red_enhanced.astype(np.uint8)))
    
    # 3. Convert to HSV and extract red/burgundy colors
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Red color range (includes burgundy)
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Dilate to connect components
    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)
    
    # Apply mask to get only red text
    red_only = cv2.bitwise_and(image_bgr, image_bgr, mask=red_mask)
    versions_to_try.append(("red_mask", red_only))
    
    # 4. High contrast version
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    versions_to_try.append(("high_contrast", enhanced_bgr))
    
    # 5. Edge-aware sharpening (helps with curved text)
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), 1.0)
    sharpened = cv2.addWeighted(image_bgr, 2.0, blurred, -1.0, 0)
    versions_to_try.append(("sharpened", sharpened))
    
    # Try each version
    for version_name, processed in versions_to_try:
        # Define search regions focusing on where vintage typically appears
        search_regions = [
            # Full image
            (0, 0, W, H, 1.0),
            
            (0, 0, W, H//2, 1.0),
            # Middle band
            (0, H//3, W, 2*H//3, 1.0),
            # Left and right edges (for curved text)
            (0, 0, W//3, H, 1.0),
            (2*W//3, 0, W, H, 1.0),
        ]
        
        all_texts = []
        for x1, y1, x2, y2, scale in search_regions:
            crop = processed[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            if scale < 1.0:
                new_w = int(crop.shape[1] * scale)
                new_h = int(crop.shape[0] * scale)
                crop = cv2.resize(crop, (new_w, new_h))
            
            # Try OCR on this region
            texts, _ = _run_paddle_ocr(crop)
            all_texts.extend(texts)
            
            # Also try with rotation correction for curved text
            for angle in [-5, 0, 5]:  # Try slight rotations
                if angle != 0:
                    center = (crop.shape[1]//2, crop.shape[0]//2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]))
                    texts, _ = _run_paddle_ocr(rotated)
                    all_texts.extend(texts)
        
        # Search for vintage in all collected text
        full_text = " ".join(all_texts)
        
        # More aggressive year extraction
        # Remove spaces that might be between digits
        text_no_spaces = re.sub(r'(\d)\s+(\d)', r'\1\2', full_text)
        
        # Try multiple patterns
        patterns = [
            r'(202[0-9])',  # Any 2020s year
            r'(20[0-9]{2})',  # Any 2000s year
            r'(\d{4})',  # Any 4 digits
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_no_spaces)
            for match in matches:
                if match.startswith(('19', '20')):
                    year_int = int(match)
                    if 1900 <= year_int <= 2030:
                        print(f"Found vintage {match} using {version_name} preprocessing")
                        return match
    
    return None

def _extract_fields(image_bgr, detections):
    """Return {'maker_name': str|None, 'vintage': str|None, 'raw': {...}}"""
    H, W = image_bgr.shape[:2]
    out: Dict[str, Any] = {"maker_name": None, "vintage": None, "raw": {}}
    vintage_detected_by_yolo = False
    
    for det in detections:
        cls = det["class"]
        x1, y1, x2, y2 = _pad_box(det["box"], W, H, pad=0.08)
        crop = image_bgr[y1:y2, x1:x2]
        txt, conf = _best_ocr_text(crop)
        cls_lower = cls.replace("-", "_").lower()
        
        if cls_lower in ["maker_name", "producer", "winery"]:
            # Keep the OCR text more raw for maker name since YOLO is accurate
            # Just basic cleaning without aggressive filtering
            cleaned = txt.strip()
            # Remove only truly problematic characters, keep most text
            cleaned = re.sub(r'[^\w\s&\'-]', ' ', cleaned)
            cleaned = " ".join(cleaned.split())
            cleaned = cleaned.upper()
            
            # Only update if we got meaningful text
            if cleaned and len(cleaned) >= 2:
                if not out["maker_name"] or len(cleaned) > len(out["maker_name"]):
                    out["maker_name"] = cleaned
            out["raw"].setdefault("maker_name_candidates", []).append((cleaned, conf))
            
            # Still check for vintage in maker region as backup
            year_in_maker = _extract_year_from_text(txt)
            if year_in_maker and not out["vintage"]:
                out["vintage"] = year_in_maker
                out["raw"].setdefault("vintage_from_maker_region", []).append((year_in_maker, conf))
                
        elif cls_lower in ["vintage", "year"]:
            vintage_detected_by_yolo = True
            year = _extract_year_from_text(txt)
            if year:
                out["vintage"] = year
            out["raw"].setdefault("vintage_candidates", []).append((txt, conf))
        else:
            out["raw"].setdefault(cls_lower, []).append((txt, conf))
    
    # FALLBACK: If YOLO didn't detect vintage region, search entire image
    if not vintage_detected_by_yolo and not out["vintage"]:
        print("Note: YOLO did not detect vintage region. Performing full image search...")
        fallback_vintage = _full_image_vintage_search(image_bgr)
        if fallback_vintage:
            out["vintage"] = fallback_vintage
            out["raw"]["vintage_from_fallback"] = fallback_vintage
            print(f"Found vintage via fallback: {fallback_vintage}")
    
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
        if k in ["maker_name_candidates", "vintage_candidates", "vintage_from_maker_region", "vintage_from_fallback"]:
            continue
        for item in arr:
            if isinstance(item, tuple) and len(item) >= 1:
                txt = item[0]
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
    id_to_name: Dict[int, str] = None,
    confidence_threshold: float = 0.01,  # Lower threshold to catch more detections
    debug: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    # Load image
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        img = image
        if img is None or not hasattr(img, "shape"):
            raise ValueError("image must be a path or a cv2/numpy image")

    # YOLO inference with lower confidence threshold
    model = _get_yolo(weights_path)
    pred = model(img, conf=confidence_threshold, verbose=False)[0]

    # Class mapping
    if id_to_name is None:
        id_to_name = {0: "Distinct Logo", 1: "Maker-Name", 2: "Vintage"}

    # Build detections list
    detections = []
    if pred.boxes is not None:
        for b in pred.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            det = {
                "class": id_to_name.get(cls_id, str(cls_id)),
                "box": [x1, y1, x2, y2],
                "confidence": conf
            }
            detections.append(det)
            if debug:
                print(f"Detection: {det['class']} (conf: {conf:.3f})")

    # OCR on detected regions + fallback
    fields = _extract_fields(img, detections)
    maker_raw = fields.get("maker_name")
    raw = fields.get("raw") or {}

    if debug and raw:
        print(f"Raw OCR results: {raw}")

    maker_norm = _normalize_maker(maker_raw) if maker_raw else ""
    if not maker_norm:
        maker_norm = _extract_best_maker_from_raw(raw)
    maker_out = maker_norm or None

    vintage_int = _to_int_year(fields.get("vintage"))
    custom_id = f"{maker_out}|{vintage_int}" if (maker_out and vintage_int) else None
    
    if debug:
        print(f"Final results: CustomID={custom_id}, Maker={maker_out}, Vintage={vintage_int}")
    
    return custom_id, maker_out, vintage_int