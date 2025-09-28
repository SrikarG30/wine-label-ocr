# Photo_Stitch.py
# Capture FRONT + BACK label crops using YOLO bottle detection,
# stitch them side-by-side, save to a temp JPG, and return the file path.
#
# Usage from another module:
#   from Photo_Stitch import stitchedImagePath
#   path = stitchedImagePath()
#   if path:
#       final_run_ocr(path, "weights.pt")

from pathlib import Path
from typing import Union, Optional, Tuple
import uuid
import tempfile
import cv2
import numpy as np

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
BOTTLE_CLASS_ID = 39           # COCO "bottle"
YOLO_CONF = 0.40
GUIDE_BOX_FRAC = (0.40, 0.80)  # (width_frac, height_frac)
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


def choose_bottle_box(results):
    """Pick the largest confident bottle bbox from YOLO results."""
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
        scale = h1 / float(h2)
        img2 = cv2.resize(img2, (int(w2 * scale), h1),
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
    # Instead of preview (low res), configure full sensor output
    cam.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_1080_P)  # or THE_4_K
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    # Use video output instead of preview
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


def stitchedImagePath(
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
    guide_box_frac=GUIDE_BOX_FRAC,
    yolo_weights: Union[str, Path] = YOLO_WEIGHTS_FOR_DETECTION,
    outfile: Optional[Union[str, Path]] = None,
    mirror_horizontally: bool = False,
    use_depthai: bool = USE_DEPTHAI,
    dai_device_mxid: Optional[str] = DAI_DEVICE_MXID,
) -> Optional[str]:
    """
    Opens OAK camera UI, capture FRONT then BACK (press SPACE twice),
    stitches them, saves to a JPG, and RETURNS the file path. Auto-exits.
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
        print(
            f"[DepthAI] Using {device.getDeviceName()} | MXID={chosen_mxid} | USB={device.getUsbSpeed()}")
    except Exception:
        print(f"[DepthAI] Using MXID={chosen_mxid}")

    def _read_frame():
        msg = q_rgb.get()
        return True, msg.getCvFrame()

    front_img = None
    back_img = None
    info = "Press SPACE to capture FRONT label"

    save_path = Path(outfile) if outfile else _make_temp_jpg_path()

    try:
        while True:
            ok, frame = _read_frame()
            if not ok or frame is None:
                break

            if mirror_horizontally:
                frame = cv2.flip(frame, 1)

            display = frame.copy()
            guide = draw_guide_box(display, guide_box_frac)

            # YOLO detect bottle
            yolo_out = model.predict(
                source=frame, classes=[
                    BOTTLE_CLASS_ID], conf=YOLO_CONF, verbose=False
            )
            box = choose_bottle_box(yolo_out)

            # Draw detection box
            if box is not None:
                x1, y1, x2, y2, conf = box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(
                    display, f"bottle {conf:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,
                                                    180, 255), 1, cv2.LINE_AA
                )

            # HUD
            cv2.putText(display, info, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "SPACE: capture | R: reset | Q: quit",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 220, 255), 1, cv2.LINE_AA)

            cv2.imshow(
                "Bottle Capture [OAK DepthAI] (Front + Back, then Stitch)", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                return None

            elif key == ord('r'):
                front_img = None
                back_img = None
                info = "Reset. Press SPACE to capture FRONT label."

            elif key == ord(' '):
                if box is None:
                    x1, y1, x2, y2 = guide
                else:
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

                    stitched = stitch_horizontal(front_img, back_img)
                    ok = cv2.imwrite(str(save_path), stitched)
                    if not ok:
                        print("Failed to write stitched image to:", save_path)
                        return None

                    cv2.imshow("Stitched (Front | Back)", stitched)
                    cv2.waitKey(250)
                    cv2.destroyWindow("Stitched (Front | Back)")
                    for _ in range(3):
                        cv2.waitKey(1)

                    return str(save_path)

                else:
                    info = "Already stitched. Press R to restart or Q to quit."

    finally:
        try:
            device.close()
        finally:
            cv2.destroyAllWindows()

    return None


# Simple CLI test
if __name__ == "__main__":
    path = stitchedImagePath()
    if path:
        print("Stitched image saved at:", path)
    else:
        print("No stitched image was produced.")
