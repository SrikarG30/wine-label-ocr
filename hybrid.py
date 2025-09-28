#!/usr/bin/env python3
"""
Hybrid Wine Label Analysis System  (cropping + masking DISABLED)

Changes in this version:
- Cropping disabled: ROI is always the full image (ignores YOLO / heuristics).
- Masking disabled: create_text_mask() returns an empty mask and notes 'disabled'.

Usage:
  python hybrid_wine_system.py /path/to/image.jpg --database wine_db.jsonl --debug-out debug
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import hashlib
import imutils
from ultralytics import YOLO


class WineLabelAnalyzer:
    def __init__(self):
        self.database_path = None
        self.known_wines = {}
        self.label_detector = None

    def load_label_detector(self, weights_path: str = "crop_weights.pt"):
        """NO-OP now that cropping is disabled (kept for compatibility/logging)."""
        if os.path.exists(weights_path):
            # Still instantiate so callers don't crash, but we won't use it.
            self.label_detector = YOLO(weights_path)
            print(
                f"(Cropping disabled) Model {weights_path} loaded but will be ignored.")
        else:
            print(
                f"(Cropping disabled) {weights_path} not found (and not needed).")

    def find_label_roi_with_yolo(self, img_bgr: np.ndarray, confidence_threshold: float = 0.8) -> Tuple[int, int, int, int]:
        """Cropping disabled: always return full image ROI."""
        H, W = img_bgr.shape[:2]
        print("Cropping is DISABLED: using full image as ROI.")
        return 0, 0, W, H

    def load_database(self, db_path: str):
        """Load existing wine database"""
        self.database_path = db_path
        self.known_wines = {}

        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        wine_id = record.get('wine_id')
                        if wine_id:
                            self.known_wines[wine_id] = record

    # keep the original heuristic method unmodified (unused now)
    def find_label_roi(self, img_bgr: np.ndarray, force_full: bool = False) -> Tuple[int, int, int, int]:
        """Enhanced ROI detection using wine-specific heuristics (UNUSED when cropping disabled)."""
        H, W = img_bgr.shape[:2]
        if force_full:
            return 0, 0, W, H

        scale = min(1.0, 800.0 / max(H, W))
        if scale < 1.0:
            small = cv2.resize(img_bgr, (int(W * scale), int(H * scale)))
        else:
            small = img_bgr.copy()

        h, w = small.shape[:2]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        best_score = -1
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            if area < 0.1 * w * h or area > 0.9 * w * h:
                continue
            aspect_ratio = cw / ch
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            cx, cy = (x + cw/2) / w, (y + ch/2) / h
            center_score = 1.0 - (abs(cx - 0.5) + abs(cy - 0.5))
            size_score = area / (w * h)
            roi = gray[y:y+ch, x:x+cw]
            content_variance = np.var(roi)
            content_score = min(1.0, content_variance / 1000.0)
            combined_score = center_score * 0.4 + size_score * 0.3 + content_score * 0.3
            if combined_score > best_score:
                best_score = combined_score
                best_box = (x, y, x + cw, y + ch)

        if best_box is None:
            margin_w, margin_h = int(w * 0.1), int(h * 0.1)
            best_box = (margin_w, margin_h, w - margin_w, h - margin_h)

        x1, y1, x2, y2 = best_box
        inv_scale = 1.0 / scale
        X1 = max(0, int(x1 * inv_scale))
        Y1 = max(0, int(y1 * inv_scale))
        X2 = min(W, int(x2 * inv_scale))
        Y2 = min(H, int(y2 * inv_scale))
        return X1, Y1, X2, Y2

    def align_vertically(self, src: np.ndarray) -> Tuple[np.ndarray, float]:
        """Align image vertically (unchanged)."""
        if len(src.shape) == 3:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            gray = src.copy()
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        degree = []
        vertical_scores = []
        step = 1
        initial_score = gray.shape[1] - np.count_nonzero(binary.sum(axis=0))
        test_rot = imutils.rotate(binary, 1)
        pos_score = test_rot.shape[1] - np.count_nonzero(test_rot.sum(axis=0))
        angle_range = range(
            0, 45, step) if pos_score > initial_score else range(0, -45, -step)

        for deg in angle_range:
            rotated = imutils.rotate(binary, deg)
            vertical_score = rotated.shape[1] - \
                np.count_nonzero(rotated.sum(axis=0))
            degree.append(deg)
            vertical_scores.append(vertical_score)

        max_score = max(vertical_scores)
        best_angle = degree[vertical_scores.index(max_score)]
        aligned_src = imutils.rotate(src, best_angle)
        return aligned_src, best_angle

    def create_text_mask(self, roi_bgr: np.ndarray, use_image_as_mask: bool = False) -> Tuple[np.ndarray, Dict]:
        H, W = roi_bgr.shape[:2]
        if not use_image_as_mask:
            # masking disabled path (previous behavior)
            mask = np.zeros((H, W), dtype=np.uint8)
            return mask, {"method": "disabled", "foreground_ratio": 0.0}

        # --- Use the prefiltered ROI as the mask ---
        # If it’s RGB, convert to gray
        if roi_bgr.ndim == 3 and roi_bgr.shape[2] == 3:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_bgr.copy()

        # Your prefiltered image has bright edges on dark background.
        # Keep bright strokes as foreground -> THRESH_BINARY (NOT _INV).
        # Otsu picks a threshold automatically.
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Light cleanup to remove salt noise and connect thin strokes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)

        fg_ratio = float(np.count_nonzero(mask)) / (H * W)
        return mask, {"method": "image_as_mask", "foreground_ratio": fg_ratio, "note": "Binary from prefiltered image"}

    def extract_smart_blobs(self, mask: np.ndarray, min_area: float = 30.0) -> List[Dict]:
        """Unchanged (will naturally return 0 blobs with an empty mask)."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        blobs = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            if w < 3 or h < 3:
                continue
            roi_area = mask.shape[0] * mask.shape[1]
            if area > 0.1 * roi_area:
                continue
            cx, cy = centroids[i]
            aspect_ratio = w / h if h > 0 else 0
            blob_mask = (labels == i).astype(np.uint8) * 255
            blob_region = blob_mask[y:y+h, x:x+w]
            contours, _ = cv2.findContours(
                blob_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            solidity = 0.5
            if contours:
                hull = cv2.convexHull(contours[0])
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
            extent = area / (w * h) if (w * h) > 0 else 0
            is_text_like = (0.02 <= aspect_ratio <= 50.0 and solidity >=
                            0.05 and extent >= 0.05 and area >= min_area)
            if is_text_like:
                blobs.append({
                    'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                    'cx': float(cx), 'cy': float(cy),
                    'area': float(area),
                    'aspect_ratio': float(aspect_ratio),
                    'solidity': float(solidity),
                    'extent': float(extent)
                })
        return blobs

    def generate_blob_fingerprint(self, blobs: List[Dict], roi_shape: Tuple[int, int]) -> str:
        if not blobs:
            return "empty"
        H, W = roi_shape
        sorted_blobs = sorted(blobs, key=lambda x: (x['cy'], x['cx']))
        fingerprint_features = []
        for blob in sorted_blobs:
            normalized = {
                'cx': round(blob['cx'] / W, 3),
                'cy': round(blob['cy'] / H, 3),
                'w': round(blob['w'] / W, 3),
                'h': round(blob['h'] / H, 3),
                'area': round(blob['area'] / (W * H), 4)
            }
            fingerprint_features.append(tuple(normalized.values()))
        fingerprint_str = str(fingerprint_features)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:16]

    def calculate_position_similarity(self, pos1, pos2) -> float:
        if not pos1 or not pos2:
            return 0.0
        len_diff = abs(len(pos1) - len(pos2))
        max_len = max(len(pos1), len(pos2))
        len_penalty = len_diff / max_len * 0.6
        distances = np.zeros((len(pos1), len(pos2)))
        for i, p1 in enumerate(pos1):
            for j, p2 in enumerate(pos2):
                distances[i, j] = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        matched = []
        dist_copy = distances.copy()
        for _ in range(min(len(pos1), len(pos2))):
            idx = np.unravel_index(np.argmin(dist_copy), dist_copy.shape)
            matched.append(dist_copy[idx])
            dist_copy[idx[0], :] = np.inf
            dist_copy[:, idx[1]] = np.inf
        if not matched:
            return 0.0
        avg = float(np.mean(matched))
        distance_similarity = max(0, 1.0 - avg / 0.05)
        return max(0.0, min(1.0, distance_similarity * (1.0 - len_penalty)))

    def find_similar_wine(self, current_fingerprint: str, current_blobs: List[Dict],
                          roi_shape: Tuple[int, int], threshold: float = 0.9) -> Optional[Dict]:
        best_match = None
        best_similarity = 0
        H, W = roi_shape
        current_positions = [(b['cx']/W, b['cy']/H) for b in current_blobs]
        for wine_id, wine_data in self.known_wines.items():
            stored_fingerprint = wine_data.get('blob_fingerprint', '')
            if stored_fingerprint == current_fingerprint:
                exact_match = wine_data.copy()
                exact_match['similarity_score'] = 1.0
                exact_match['match_type'] = 'exact'
                return exact_match
            stored_blobs = wine_data.get('blobs', [])
            if not stored_blobs:
                continue
            sH, sW = wine_data.get('roi_shape', [H, W])
            stored_positions = [(b['cx']/sW, b['cy']/sH) for b in stored_blobs]
            similarity = self.calculate_position_similarity(
                current_positions, stored_positions)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = wine_data.copy()
                best_match['similarity_score'] = similarity
                best_match['match_type'] = 'similar'
        return best_match

    def save_wine_record(self, image_path: str, roi_bbox: Tuple[int, int, int, int],
                         mask_info: Dict, blobs: List[Dict], fingerprint: str,
                         alignment_angle: float, match_info: Optional[Dict] = None):
        if not self.database_path:
            return
        roi_shape = [roi_bbox[3] - roi_bbox[1], roi_bbox[2] - roi_bbox[0]]
        record = {
            'image': os.path.basename(image_path),
            'timestamp': os.path.getmtime(image_path),
            'roi_bbox': list(roi_bbox),
            'roi_shape': roi_shape,
            'alignment_angle': float(alignment_angle),
            'mask_info': mask_info,
            'blob_count': len(blobs),
            'blob_fingerprint': fingerprint,
            'blobs': blobs,
            'wine_id': fingerprint,
            'is_new_wine': match_info is None,
            'similar_wine': match_info
        }
        self.known_wines[fingerprint] = record
        with open(self.database_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def create_debug_visualization(self, roi_bgr: np.ndarray, mask: np.ndarray,
                                   blobs: List[Dict], output_path: str):
        H, W = roi_bgr.shape[:2]
        debug_img = np.zeros((H * 2 + 40, W * 2 + 40, 3), dtype=np.uint8)
        pad = 20
        debug_img[pad:H+pad, pad:W+pad] = roi_bgr
        cv2.putText(debug_img, 'Original ROI', (pad + 5, pad + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        debug_img[pad:H+pad, W+pad*2:W*2+pad*2] = mask_colored
        cv2.putText(debug_img, f'Text Mask ({np.sum(mask > 0)} pixels)',
                    (W + pad*2 + 5, pad + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        overlay = roi_bgr.copy()
        mask_overlay = np.zeros_like(roi_bgr)
        mask_overlay[:, :, 2] = mask
        overlay = cv2.addWeighted(overlay, 0.8, mask_overlay, 0.2, 0)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 255, 128), (255, 128, 128)]
        for i, blob in enumerate(blobs):
            x, y, w, h = blob['x'], blob['y'], blob['w'], blob['h']
            color = colors[i % len(colors)]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            text = str(i)
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(overlay, (x, max(0, y - text_h - 5)),
                          (x + text_w + 5, y), color, -1)
            cv2.putText(overlay, text, (x + 2, max(text_h, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            center_x, center_y = int(blob['cx']), int(blob['cy'])
            cv2.circle(overlay, (center_x, center_y), 3, color, -1)
        debug_img[H+pad*2:H*2+pad*2, pad:W+pad] = overlay
        cv2.putText(debug_img, f'Blob Detection - {len(blobs)} blobs',
                    (pad + 5, H + pad*2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        stats_img = np.zeros((H, W, 3), dtype=np.uint8)
        y_offset = 25
        line_height = 18
        font_scale = 0.4
        cv2.putText(stats_img, f'BLOB STATISTICS ({len(blobs)} total)',
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        y_offset += 30
        for i, blob in enumerate(blobs):
            if y_offset > H - 25:
                break
            color = colors[i % len(colors)]
            text1 = f"#{i}: Area={blob['area']:.0f}, AR={blob['aspect_ratio']:.2f}"
            text2 = f"  Pos:({blob['cx']:.0f},{blob['cy']:.0f}), Sol={blob['solidity']:.2f}"
            cv2.putText(stats_img, text1, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            y_offset += line_height
            cv2.putText(stats_img, text2, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 180, 180), 1)
            y_offset += line_height + 3
        debug_img[H+pad*2:H*2+pad*2, W+pad*2:W*2+pad*2] = stats_img
        cv2.putText(debug_img, 'Statistics', (W + pad*2 + 5, H + pad*2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(output_path, debug_img)
        simple_overlay_path = output_path.replace(
            '_debug.jpg', '_blobs_only.jpg')
        cv2.imwrite(simple_overlay_path, overlay)
        return simple_overlay_path


def main():
    parser = argparse.ArgumentParser(
        description='Wine Label Analysis with YOLO Detection (cropping/masking disabled)')
    parser.add_argument('image', help='Path to wine label image')
    parser.add_argument(
        '--database', default='wine_database.jsonl', help='Path to wine database file')
    parser.add_argument('--debug-out', help='Directory for debug images')
    parser.add_argument('--min-blob-area', type=float,
                        default=30.0, help='Minimum blob area in pixels')
    parser.add_argument('--similarity-threshold', type=float,
                        default=0.9, help='Similarity threshold for wine matching')
    parser.add_argument('--label-confidence', type=float, default=0.8,
                        help='(Ignored) Confidence threshold for YOLO label detection')
    parser.add_argument('--crop-weights', default='crop_weights.pt',
                        help='(Optional) YOLO weights (ignored when cropping disabled)')
    parser.add_argument('--skip-alignment', action='store_true',
                        help='Skip vertical alignment step')

    parser.add_argument('--use-image-as-mask', action='store_true',
                        help='Use the input/ROI image (thresholded) directly as the mask')

    args = parser.parse_args()

    analyzer = WineLabelAnalyzer()
    analyzer.load_database(args.database)
    analyzer.load_label_detector(args.crop_weights)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Cannot read image {args.image}")
        return 1

    print(f"Processing image: {args.image}")

    # ROI: full image
    x1, y1, x2, y2 = analyzer.find_label_roi_with_yolo(
        img, args.label_confidence)
    roi = img[y1:y2, x1:x2].copy()
    roi_shape = roi.shape[:2]
    print(
        f"Label ROI (full image): ({x1}, {y1}) to ({x2}, {y2}), size: {roi_shape}")

    # Optional alignment
    alignment_angle = 0.0
    if not args.skip_alignment:
        roi, alignment_angle = analyzer.align_vertically(roi)
        print(f"Vertical alignment: rotated by {alignment_angle:.1f} degrees")

    # Masking disabled
    mask, mask_info = analyzer.create_text_mask(
        roi, use_image_as_mask=args.use_image_as_mask
    )
    print(f"Text mask status: {mask_info['method']}")

    # Blobs (will be zero with empty mask)
    blobs = analyzer.extract_smart_blobs(mask, args.min_blob_area)
    print(f"Found {len(blobs)} text blobs")

    fingerprint = analyzer.generate_blob_fingerprint(blobs, roi.shape[:2])
    print(f"Wine fingerprint: {fingerprint}")

    similar_wine = analyzer.find_similar_wine(
        fingerprint, blobs, roi.shape[:2], args.similarity_threshold)
    if similar_wine:
        match_type = similar_wine.get('match_type', 'unknown')
        similarity_score = similar_wine.get('similarity_score', 0)
        print(f"MATCH FOUND! Similar to wine: {similar_wine.get('wine_id')}")
        if match_type == 'exact':
            print(f"Match type: EXACT fingerprint match")
        else:
            print(f"Match type: Similar (score: {similarity_score:.3f})")
        print(f"Original image: {similar_wine.get('image')}")
        is_new = False
    else:
        print("NEW WINE - No similar wines found in database")
        is_new = True

    analyzer.save_wine_record(args.image, (x1, y1, x2, y2), mask_info,
                              blobs, fingerprint, alignment_angle, similar_wine)

    if args.debug_out:
        os.makedirs(args.debug_out, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        debug_path = os.path.join(args.debug_out, f"{base_name}_debug.jpg")
        simple_overlay_path = analyzer.create_debug_visualization(
            roi, mask, blobs, debug_path)
        print(f"Debug visualization saved: {debug_path}")
        print(f"Simple blob overlay saved: {simple_overlay_path}")

    result = {
        'wine_id': fingerprint,
        'is_new_wine': is_new,
        'blob_count': len(blobs),
        'roi_size': roi_shape,
        'alignment_angle': alignment_angle,
        'foreground_ratio': mask_info['foreground_ratio'],
        'similar_wine_id': similar_wine.get('wine_id') if similar_wine else None,
        'similarity_score': similar_wine.get('similarity_score') if similar_wine else None,
        'match_type': similar_wine.get('match_type') if similar_wine else None
    }

    print("\n" + "="*50)
    print("ANALYSIS SUMMARY:")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
