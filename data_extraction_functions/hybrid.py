#!/usr/bin/env python3
"""
Hybrid Wine Label Analysis System  (masking DISABLED)

Usage:
    python hybrid.py /path/to/image.jpg  --use-image-as-mask --skip-alignment --debug-out debug
    python3 hybrid.py /path/to/image.jpg --use-image-as-mask --skip-alignment --debug-out debug
"""

from __future__ import annotations

import os
import json
import cv2
import numpy as np
import hashlib
import imutils
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict

# YOLO is optional; only used if you ask to crop
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    YOLO = None
    HAS_YOLO = False


# ----------------------------
# Core analyzer (same logic you liked, just tidy)
# ----------------------------
class WineLabelAnalyzer:
    def __init__(self):
        self.database_path: Optional[str] = None
        self.known_wines: Dict[str, Dict] = {}
        self.label_detector = None  # YOLO model, optional

    def load_label_detector(self, weights_path: str = "crop_weights.pt"):
        """Load YOLO for cropping if available."""
        if not weights_path:
            print("(Cropping) No weights path provided; YOLO disabled.")
            return
        if not HAS_YOLO:
            print("(Cropping) Ultralytics not installed; YOLO disabled.")
            return
        if os.path.exists(weights_path):
            self.label_detector = YOLO(weights_path)
            print(f"(Cropping) Loaded YOLO weights: {weights_path}")
        else:
            print(
                f"(Cropping) Weights not found at {weights_path}; YOLO disabled.")

    def load_database(self, db_path: str):
        self.database_path = db_path
        self.known_wines = {}
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        wid = rec.get("wine_id")
                        if wid:
                            self.known_wines[wid] = rec

    def find_label_roi_with_yolo(
        self,
        img_bgr: np.ndarray,
        confidence_threshold: float = 0.8,
        enable: bool = False
    ) -> Tuple[int, int, int, int]:
        """If enable=False → full image; else try YOLO then heuristic."""
        H, W = img_bgr.shape[:2]
        if not enable:
            return 0, 0, W, H

        # YOLO first if available
        if self.label_detector is not None:
            results = self.label_detector(
                img_bgr, conf=confidence_threshold, verbose=False)
            best_box, best_conf = None, 0.0
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        if conf > best_conf:
                            best_conf, best_box = conf, (x1, y1, x2, y2)
            if best_box:
                x1, y1, x2, y2 = best_box
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(W, x2 + pad)
                y2 = min(H, y2 + pad)
                return x1, y1, x2, y2

        # Heuristic fallback
        return self.find_label_roi(img_bgr, force_full=False)

    # Original heuristic (kept intact)
    def find_label_roi(self, img_bgr: np.ndarray, force_full: bool = False) -> Tuple[int, int, int, int]:
        H, W = img_bgr.shape[:2]
        if force_full:
            return 0, 0, W, H

        scale = min(1.0, 800.0 / max(H, W))
        small = cv2.resize(img_bgr, (int(W * scale), int(H * scale))
                           ) if scale < 1.0 else img_bgr.copy()
        h, w = small.shape[:2]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box, best_score = None, -1
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            if area < 0.1 * w * h or area > 0.9 * w * h:
                continue
            ar = cw / ch
            if ar < 0.2 or ar > 5.0:
                continue
            cx, cy = (x + cw/2) / w, (y + ch/2) / h
            center_score = 1.0 - (abs(cx - 0.5) + abs(cy - 0.5))
            size_score = area / (w * h)
            roi = gray[y:y+ch, x:x+cw]
            content_variance = np.var(roi)
            content_score = min(1.0, content_variance / 1000.0)
            score = center_score * 0.4 + size_score * 0.3 + content_score * 0.3
            if score > best_score:
                best_score = score
                best_box = (x, y, x + cw, y + ch)

        if best_box is None:
            margin_w, margin_h = int(w * 0.1), int(h * 0.1)
            best_box = (margin_w, margin_h, w - margin_w, h - margin_h)

        x1, y1, x2, y2 = best_box
        inv = 1.0 / scale
        return max(0, int(x1 * inv)), max(0, int(y1 * inv)), min(W, int(x2 * inv)), min(H, int(y2 * inv))

    def align_vertically(self, src: np.ndarray) -> Tuple[np.ndarray, float]:
        """Simple alignment (you can skip it from the public API)."""
        gray = cv2.cvtColor(
            src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src.copy()
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        degree, scores = [], []
        step = 1
        initial = gray.shape[1] - np.count_nonzero(binary.sum(axis=0))
        test_rot = imutils.rotate(binary, 1)
        pos_score = test_rot.shape[1] - np.count_nonzero(test_rot.sum(axis=0))
        rng = range(0, 45, step) if pos_score > initial else range(
            0, -45, -step)

        for deg in rng:
            rot = imutils.rotate(binary, deg)
            score = rot.shape[1] - np.count_nonzero(rot.sum(axis=0))
            degree.append(deg)
            scores.append(score)

        best_angle = degree[int(np.argmax(scores))]
        return imutils.rotate(src, best_angle), float(best_angle)

    def create_text_mask(self, roi_bgr: np.ndarray, use_image_as_mask: bool = False) -> Tuple[np.ndarray, Dict]:
        H, W = roi_bgr.shape[:2]
        if not use_image_as_mask:
            mask = np.zeros((H, W), dtype=np.uint8)
            return mask, {"method": "disabled", "foreground_ratio": 0.0}

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if (
            roi_bgr.ndim == 3 and roi_bgr.shape[2] == 3) else roi_bgr.copy()
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), 1)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
        fg_ratio = float(np.count_nonzero(mask)) / float(H * W if H * W else 1)
        return mask, {"method": "image_as_mask", "foreground_ratio": fg_ratio, "note": "Binary from prefiltered image"}

    def extract_smart_blobs(self, mask: np.ndarray, min_area: float = 30.0) -> List[Dict]:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        blobs: List[Dict] = []
        H, W = mask.shape[:2]
        roi_area = float(H * W) if H and W else 1.0

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < min_area or w < 3 or h < 3:
                continue
            if area > 0.1 * roi_area:
                continue
            cx, cy = centroids[i]
            ar = w / h if h > 0 else 0.0

            # local solidity/extent
            blob_mask = (labels == i).astype(np.uint8) * 255
            blob_region = blob_mask[y:y+h, x:x+w]
            contours, _ = cv2.findContours(
                blob_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            solidity = 0.5
            if contours:
                hull = cv2.convexHull(contours[0])
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / float(hull_area)
            extent = float(area) / float(w * h) if (w * h) > 0 else 0.0

            if (0.02 <= ar <= 50.0) and (solidity >= 0.05) and (extent >= 0.05):
                blobs.append({
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "cx": float(cx), "cy": float(cy),
                    "area": float(area),
                    "aspect_ratio": float(ar),
                    "solidity": float(solidity),
                    "extent": float(extent)
                })
        return blobs

    def generate_blob_fingerprint(self, blobs: List[Dict], roi_shape: Tuple[int, int]) -> str:
        if not blobs:
            return "empty"
        H, W = roi_shape
        feats = []
        for b in sorted(blobs, key=lambda x: (x["cy"], x["cx"])):
            feats.append((
                round(b["cx"] / W, 3),
                round(b["cy"] / H, 3),
                round(b["w"] / W, 3),
                round(b["h"] / H, 3),
                round(b["area"] / (W * H), 4),
            ))
        return hashlib.md5(str(feats).encode()).hexdigest()[:16]

    def save_wine_record(
        self,
        image_path: str,
        roi_bbox: Tuple[int, int, int, int],
        mask_info: Dict,
        blobs: List[Dict],
        fingerprint: str,
        alignment_angle: float,
    ):
        if not self.database_path:
            return
        x1, y1, x2, y2 = roi_bbox
        roi_shape = [y2 - y1, x2 - x1]
        rec = {
            "image": os.path.basename(image_path),
            "timestamp": os.path.getmtime(image_path) if os.path.exists(image_path) else None,
            "roi_bbox": list(roi_bbox),
            "roi_shape": roi_shape,
            "alignment_angle": float(alignment_angle),
            "mask_info": mask_info,
            "blob_count": len(blobs),
            "blob_fingerprint": fingerprint,
            "blobs": blobs,
            "wine_id": fingerprint,
        }
        self.known_wines[fingerprint] = rec
        with open(self.database_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------
# Public API (what Wine_Tracker.py will call)
# ----------------------------
def final_run_blobs(
    image_path: Union[str, os.PathLike],
    *,
    use_image_as_mask: bool = False,
    crop_label: bool = False,
    skip_alignment: bool = True,
    database: str = "wine_database.jsonl",
    crop_weights: Optional[str] = "crop_weights.pt",
    min_blob_area: float = 30.0,
    similarity_threshold: float = 0.9,
    label_confidence: float = 0.8,
    debug_out: Optional[str] = None
) -> Dict:
    """
    Run blob pipeline and return a JSON-serializable dict.

    Returns:
      {
        'image': str,
        'roi_bbox': [x1,y1,x2,y2],
        'roi_shape': [H,W],
        'alignment_angle': float,
        'mask_info': {...},
        'blob_count': int,
        'blobs': [ {x,y,w,h,cx,cy,area,aspect_ratio,solidity,extent}, ... ],
        'blob_fingerprint': str,
        'match': {'wine_id':..., 'similarity_score':..., 'match_type':...} | None,
        'debug_paths': {'composite': path_or_None, 'blobs_only': path_or_None}
      }
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    analyzer = WineLabelAnalyzer()
    analyzer.load_database(database)
    analyzer.load_label_detector(crop_weights or "")

    # ROI (full by default; crop if requested)
    x1, y1, x2, y2 = analyzer.find_label_roi_with_yolo(
        img, confidence_threshold=label_confidence, enable=crop_label
    )
    roi = img[y1:y2, x1:x2].copy()
    roi_shape = roi.shape[:2]

    # Optional alignment
    alignment_angle = 0.0
    if not skip_alignment:
        roi, alignment_angle = analyzer.align_vertically(roi)

    # Mask + blobs
    mask, mask_info = analyzer.create_text_mask(
        roi, use_image_as_mask=use_image_as_mask)
    blobs = analyzer.extract_smart_blobs(mask, min_blob_area)
    fingerprint = analyzer.generate_blob_fingerprint(blobs, roi.shape[:2])

    # Save DB
    analyzer.save_wine_record(
        str(image_path),
        (x1, y1, x2, y2),
        mask_info,
        blobs,
        fingerprint,
        alignment_angle
    )

    # Debug composites (optional)
    comp_path = blobs_only_path = None
    if debug_out:
        os.makedirs(debug_out, exist_ok=True)
        base = os.path.splitext(os.path.basename(str(image_path)))[0]
        comp_path = os.path.join(debug_out, f"{base}_debug.jpg")
        # Reuse the overlay creator
        blobs_only_path = _create_debug_visualization(
            roi, mask, blobs, comp_path)

    return {
        "image": os.path.basename(str(image_path)),
        "roi_bbox": [x1, y1, x2, y2],
        "roi_shape": list(roi_shape),
        "alignment_angle": alignment_angle,
        "mask_info": mask_info,
        "blob_count": len(blobs),
        "blobs": blobs,
        "blob_fingerprint": fingerprint,
        "debug_paths": {"composite": comp_path, "blobs_only": blobs_only_path}
    }


# Small helper using the same visualization routine you had
def _create_debug_visualization(roi_bgr: np.ndarray, mask: np.ndarray, blobs: List[Dict], output_path: str) -> Optional[str]:
    H, W = roi_bgr.shape[:2]
    debug_img = np.zeros((H * 2 + 40, W * 2 + 40, 3), dtype=np.uint8)
    pad = 20
    debug_img[pad:H+pad, pad:W+pad] = roi_bgr
    cv2.putText(debug_img, 'Original ROI', (pad + 5, pad + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    debug_img[pad:H+pad, W+pad*2:W*2+pad*2] = mask_colored
    cv2.putText(debug_img, f'Text Mask ({np.sum(mask > 0)} pixels)', (
        W + pad*2 + 5, pad + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    overlay = roi_bgr.copy()
    mask_overlay = np.zeros_like(roi_bgr)
    mask_overlay[:, :, 2] = mask
    overlay = cv2.addWeighted(overlay, 0.8, mask_overlay, 0.2, 0)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 255, 128), (255, 128, 128)]
    for i, b in enumerate(blobs):
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        color = colors[i % len(colors)]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        text = str(i)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x, max(0, y - th - 5)),
                      (x + tw + 5, y), color, -1)
        cv2.putText(overlay, text, (x + 2, max(th, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cx, cy = int(b['cx']), int(b['cy'])
        cv2.circle(overlay, (cx, cy), 3, color, -1)

    debug_img[H+pad*2:H*2+pad*2, pad:W+pad] = overlay
    cv2.putText(debug_img, f'Blob Detection - {len(blobs)} blobs', (pad + 5,
                H + pad*2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    stats_img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.putText(stats_img, f'BLOB STATS ({len(blobs)} total)',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    debug_img[H+pad*2:H*2+pad*2, W+pad*2:W*2+pad*2] = stats_img

    try:
        cv2.imwrite(output_path, debug_img)
        simple_overlay_path = output_path.replace(
            '_debug.jpg', '_blobs_only.jpg')
        cv2.imwrite(simple_overlay_path, overlay)
        return simple_overlay_path
    except Exception:
        return None


# ----------------------------
# Optional CLI for ad-hoc testing
# ----------------------------
if __name__ == "__main__":
    import argparse
    import sys
    p = argparse.ArgumentParser(description="Blob fingerprint (module & CLI)")
    p.add_argument("image", help="Path to wine image")
    p.add_argument("--database", default="wine_database.jsonl")
    p.add_argument("--crop-weights", default="crop_weights.pt")
    p.add_argument("--min-blob-area", type=float, default=30.0)
    p.add_argument("--similarity-threshold", type=float, default=0.9)
    p.add_argument("--label-confidence", type=float, default=0.8)
    p.add_argument("--use-image-as-mask", action="store_true")
    p.add_argument("--crop-label", action="store_true")
    p.add_argument("--skip-alignment", action="store_true")
    p.add_argument("--debug-out", default=None)
    args = p.parse_args()

    out = final_run_blobs(
        args.image,
        use_image_as_mask=args.use_image_as_mask,
        crop_label=args.crop_label,
        skip_alignment=args.skip_alignment,
        database=args.database,
        crop_weights=args.crop_weights,
        min_blob_area=args.min_blob_area,
        similarity_threshold=args.similarity_threshold,
        label_confidence=args.label_confidence,
        debug_out=args.debug_out
    )
    print(json.dumps(out, indent=2))
    sys.exit(0)
