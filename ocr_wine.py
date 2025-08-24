import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize a global OCR instance
ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)

def _pad_box(xyxy, img_w, img_h, pad=0.05):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    x1 = max(0, int(x1 - pad * w))
    y1 = max(0, int(y1 - pad * h))
    x2 = min(img_w, int(x2 + pad * w))
    y2 = min(img_h, int(y2 + pad * h))
    return [x1, y1, x2, y2]

def _enhance_for_ocr(crop_bgr):
    # Convert to grayscale and apply CLAHE
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    # Bilateral filter to reduce noise while keeping edges
    g = cv2.bilateralFilter(g, d=7, sigmaColor=50, sigmaSpace=50)
    h, w = g.shape
    # Upscale small crops
    scale = 2 if max(h, w) < 250 else 1
    if scale > 1:
        g = cv2.resize(g, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    # Unsharp mask to sharpen
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    sharp = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
    # Adaptive threshold
    binar = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 7)
    return sharp, binar

def _run_paddle_ocr(img):
    # If grayscale, convert to RGB
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = ocr.ocr(rgb, cls=True)
    texts = []
    confs = []
    if result:
        for line in result[0]:
            txt = line[1][0].strip()
            conf = float(line[1][1])
            if txt:
                texts.append(txt)
                confs.append(conf)
    return texts, confs

def best_ocr_text(crop_bgr):
    sharp, binar = _enhance_for_ocr(crop_bgr)
    t1, c1 = _run_paddle_ocr(sharp)
    t2, c2 = _run_paddle_ocr(binar)
    m1 = np.mean(c1) if c1 else 0.0
    m2 = np.mean(c2) if c2 else 0.0
    if m1 >= m2:
        texts, confs = t1, c1
    else:
        texts, confs = t2, c2
    return " ".join(texts).strip(), (np.mean(confs) if confs else 0.0)

def extract_fields(image_bgr, detections):
    """
    Extracts maker name and vintage from detections.
    detections: list of dicts with {"class": str, "box": [x1,y1,x2,y2]}
    """
    H, W = image_bgr.shape[:2]
    out = {"maker_name": None, "vintage": None, "raw": {}}
    for det in detections:
        cls = det["class"]
        x1, y1, x2, y2 = _pad_box(det["box"], W, H, pad=0.08)
        crop = image_bgr[y1:y2, x1:x2]
        txt, conf = best_ocr_text(crop)
        # Process by class
        cls_lower = cls.replace("-", "_").lower()
        if cls_lower in ["maker_name", "maker_name", "producer", "winery"]:
            cleaned = "".join(ch for ch in txt if ch.isalnum() or ch in " &'-").upper()
            cleaned = " ".join(cleaned.split())
            if not out["maker_name"] or (cleaned and len(cleaned) > len(out["maker_name"])):
                out["maker_name"] = cleaned if cleaned else None
            out["raw"].setdefault("maker_name_candidates", []).append((cleaned, conf))
        elif cls_lower in ["vintage", "year"]:
            import re
            m = re.search(r"\b(19|20)\d{2}\b", txt)
            if m:
                out["vintage"] = m.group(0)
            else:
                digits = "".join(ch for ch in txt if ch.isdigit())
                if len(digits) >= 4:
                    for i in range(0, len(digits)-3):
                        cand = digits[i:i+4]
                        if cand.startswith(("19", "20")):
                            out["vintage"] = cand
                            break
            out["raw"].setdefault("vintage_candidates", []).append((txt, conf))
        else:
            out["raw"].setdefault(cls_lower, []).append((txt, conf))
    return out
