# Photo_Stitch.py
# Capture FRONT + BACK using OAK (color stream),
# compute a label box on the COLOR frame, but DO NOT crop color.
# Apply that box to the EDGE frame (Canny) and crop there.
# Save two stitched JPGs:
#   1) normal (color, full frames)  -> normalFramePath
#   2) edge   (edge, cropped by box)-> edgeFramePath
#
# Usage:
#   from Photo_Stitch import stitchedImagePath
#   normalPath, edgePath = stitchedImagePath()
#   if normalPath and edgePath:
#       # normalPath -> OCR/Barcode/cropping coords (if needed)
#       # edgePath   -> blob detection
#
# Keys: SPACE (capture), R (reset), Q (quit)

from pathlib import Path
from typing import Union, Optional, Tuple
import uuid
import tempfile
import cv2
import numpy as np
import depthai as dai

# --- DepthAI (OAK) support ---
try:
    import depthai as dai
    HAVE_DAI = True
except Exception:
    HAVE_DAI = False

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit(
        f"Install ultralytics first: pip install ultralytics\n\n{e}"
    )

# ---------- Config ----------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BOTTLE_CLASS_ID = 39            # COCO "bottle"
YOLO_CONF = 0.40
GUIDE_BOX_FRAC = (0.40, 0.80)   # (width_frac, height_frac)
YOLO_WEIGHTS_FOR_DETECTION = "yolov8n.pt"  # change if you want

# DepthAI toggles
USE_DEPTHAI = True                 # must stay True, no cv2 fallback
DAI_DEVICE_MXID: Optional[str] = "19443010E1EFF24800"


def draw_guide_box(img, frac=(0.4, 0.8)) -> Tuple[int, int, int, int]:
    """Draws a centered translucent guide box + crosshairs. Returns (x1,y1,x2,y2)."""
    h, w = img.shape[:2]
    gw = int(w * frac[0])
    gh = int(h * frac[1])
    x1 = (w - gw) // 2
    y1 = (h - gh) // 2
    x2 = x1 + gw
    y2 = y1 + gh

    # box
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 220, 60), 2)

    # dim surroundings
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)

    # crosshair
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.line(img, (cx, y1), (cx, y2), (60, 220, 60), 1)
    cv2.line(img, (x1, cy), (x2, cy), (60, 220, 60), 1)
    return (x1, y1, x2, y2)


def choose_best_box(results, conf_thresh=0.40):
    """Pick highest-confidence box from any class."""
    best = None
    best_conf = -1.0
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            try:
                conf = float(b.conf.item())
                if conf < conf_thresh:
                    continue
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                if conf > best_conf:
                    best_conf = conf
                    best = (x1, y1, x2, y2, conf)
            except Exception:
                pass
    return best


def safe_crop(img, x1, y1, x2, y2):
    """Safely crop; return None on invalid region."""
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
    """Return side-by-side stitch with matched heights."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2:
        scale = h1 / float(h2 if h2 else 1)
        img2 = cv2.resize(img2, (max(1, int(w2 * scale)), h1),
                          interpolation=cv2.INTER_CUBIC)
    return np.hstack((img1, img2))


def _make_temp_jpg_path(prefix="stitch_", suffix=".jpg") -> Path:
    tmp_dir = Path(tempfile.gettempdir())
    return tmp_dir / f"{prefix}{uuid.uuid4().hex}{suffix}"


# -------- DepthAI helpers --------
def _open_depthai(frame_width: int, frame_height: int, device_mxid: Optional[str] = None):
    if not HAVE_DAI:
        raise SystemExit("DepthAI not installed. Run: pip install depthai")

    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)
    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080
    cam.setVideoSize(FRAME_WIDTH, FRAME_HEIGHT)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)

    if device_mxid is None:
        device = dai.Device(pipeline)
    else:
        info = dai.DeviceInfo(device_mxid)
        device = dai.Device(pipeline, info)

    q_rgb = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    return device, q_rgb


def _edge_from_color_bgr(bgr: np.ndarray) -> np.ndarray:
    """Host-side edge map (Canny + light morphology)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    # optional: thicken slightly for connectivity
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
    return edges


def stitchedImagePath(
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
    guide_box_frac=GUIDE_BOX_FRAC,
    yolo_weights: Union[str, Path] = YOLO_WEIGHTS_FOR_DETECTION,
    outfile_normal: Optional[Union[str, Path]] = None,
    outfile_edge: Optional[Union[str, Path]] = None,
    mirror_horizontally: bool = False,
    use_depthai: bool = USE_DEPTHAI,
    dai_device_mxid: Optional[str] = DAI_DEVICE_MXID,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Opens OAK camera UI, capture FRONT then BACK (press SPACE twice).

    - On each SPACE:
        * Run YOLO on COLOR frame to get crop coords (fallback = guide box).
        * DO NOT crop the color frame (store full color frame).
        * Crop the EDGE frame with those coords (store cropped edge).

    - After 2 captures:
        * Save STITCHED COLOR (full frames)      -> normalFramePath
        * Save STITCHED EDGE  (cropped by coords)-> edgeFramePath

    Returns: (normalFramePath, edgeFramePath)
             (None, None) if aborted.
    """
    if not HAVE_DAI:
        raise SystemExit("DepthAI not installed. Run: pip install depthai")
    if not use_depthai:
        raise SystemExit("This build is DepthAI-only. Set use_depthai=True.")

    model = YOLO(str(yolo_weights))

    devs = dai.Device.getAllAvailableDevices()
    if not devs:
        raise SystemExit(
            "No DepthAI device found. Plug in your OAK and try again.")
    chosen_mxid = dai_device_mxid or devs[0].getMxId()
    device, q_rgb = _open_depthai(frame_width, frame_height, chosen_mxid)
    try:
        print(f"[DepthAI] Using MXID={chosen_mxid}")
    except Exception:
        pass

    def _read_frame():
        msg = q_rgb.get()
        return True, msg.getCvFrame()

    # storage for the two shots
    color_front_full = None
    color_back_full = None
    edge_front_cropped = None
    edge_back_cropped = None

    info = "Press SPACE to capture FRONT label"
    normal_save = Path(outfile_normal) if outfile_normal else _make_temp_jpg_path(
        prefix="normal_")
    edge_save = Path(outfile_edge) if outfile_edge else _make_temp_jpg_path(
        prefix="edge_")

    try:
        while True:
            ok, frame_bgr = _read_frame()
            if not ok or frame_bgr is None:
                break

            if mirror_horizontally:
                frame_bgr = cv2.flip(frame_bgr, 1)

            # compute edge view (host-side)
            edge = _edge_from_color_bgr(frame_bgr)

            # For display HUD, draw guide on a copy of color
            display = frame_bgr.copy()
            guide = draw_guide_box(display, guide_box_frac)

            # YOLO detect bottle ON COLOR
            yolo_out = model.predict(
                source=frame_bgr, conf=YOLO_CONF, verbose=False)
            box = choose_best_box(yolo_out, YOLO_CONF)

            # visualize color det box
            if box is not None:
                x1, y1, x2, y2, conf = box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(display, f"bottle {conf:.2f}",
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)

            # HUD
            cv2.putText(display, info, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "SPACE: capture | R: reset | Q: quit",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 220, 255), 1, cv2.LINE_AA)

            # Show side-by-side preview (color | edge)
            edge_vis = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
            preview = stitch_horizontal(display, edge_vis)
            cv2.imshow(
                "Bottle Capture [Color | Edge] (Front + Back, then Stitch)", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                return None, None

            elif key == ord('r'):
                color_front_full = None
                color_back_full = None
                edge_front_cropped = None
                edge_back_cropped = None
                info = "Reset. Press SPACE to capture FRONT label."

            elif key == ord(' '):
                # choose coords from COLOR (yolo or guide)
                if box is None:
                    x1, y1, x2, y2 = guide
                else:
                    x1, y1, x2, y2, _ = box

                # crop the EDGE frame using coords from COLOR
                edge_crop = safe_crop(edge, x1, y1, x2, y2)
                if edge_crop is None:
                    info = "Bad crop. Adjust bottle and try again."
                    continue

                # store full COLOR + cropped EDGE per shot
                if color_front_full is None:
                    color_front_full = frame_bgr.copy()          # FULL color
                    edge_front_cropped = edge_crop.copy()        # CROPPED edge
                    info = "Front captured. Now rotate to BACK and press SPACE."
                elif color_back_full is None:
                    color_back_full = frame_bgr.copy()           # FULL color
                    edge_back_cropped = edge_crop.copy()         # CROPPED edge

                    # stitch both products
                    stitched_color = stitch_horizontal(
                        color_front_full, color_back_full)
                    stitched_edge = stitch_horizontal(
                        edge_front_cropped, edge_back_cropped)

                    ok1 = cv2.imwrite(str(normal_save), stitched_color)
                    ok2 = cv2.imwrite(str(edge_save),   stitched_edge)
                    if not ok1 or not ok2:
                        print("Failed to write stitched outputs.")
                        return None, None

                    # quick preview windows
                    cv2.imshow("Stitched COLOR (Front | Back)", stitched_color)
                    cv2.imshow(
                        "Stitched EDGE  (Front | Back, Cropped)", stitched_edge)
                    cv2.waitKey(300)
                    try:
                        cv2.destroyWindow("Stitched COLOR (Front | Back)")
                        cv2.destroyWindow(
                            "Stitched EDGE  (Front | Back, Cropped)")
                    except Exception:
                        pass
                    for _ in range(3):
                        cv2.waitKey(1)

                    # return both paths
                    return str(normal_save), str(edge_save)

                else:
                    info = "Already stitched. Press R to restart or Q to quit."

    finally:
        try:
            device.close()
        finally:
            cv2.destroyAllWindows()

    return None, None


# Simple CLI test
if __name__ == "__main__":
    normalPath, edgePath = stitchedImagePath()
    if normalPath and edgePath:
        print("Color (full)  stitched image:", normalPath)
        print("Edge  (cropped) stitched image:", edgePath)
    else:
        print("No stitched images were produced.")
