# QRCodeScanner.py
# Capture a frame from OAK (DepthAI) and scan for barcodes using pyzbar.

import cv2
from pyzbar import pyzbar
import depthai as dai

# ---------- Config ----------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DAI_DEVICE_MXID = None  # put your MXID here if you want to lock to a specific device


def ask_yes_no(prompt: str) -> bool:
    """
    Repeatedly prompt the user until they enter yes/y or no/n.
    Returns True for yes, False for no.
    """
    valid_yes = {"y", "yes"}
    valid_no = {"n", "no"}
    while True:
        try:
            ans = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCanceled.")
            return False
        if ans in valid_yes:
            return True
        if ans in valid_no:
            return False
        print("Please enter yes/y or no/n.")


def _open_depthai(frame_width: int, frame_height: int, device_mxid: str = None):
    """
    Open DepthAI pipeline for color camera frames.
    Returns (device, q_rgb).
    """
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)

    if device_mxid is None:
        device = dai.Device(pipeline)
    else:
        device = dai.Device(pipeline, dai.DeviceInfo(device_mxid))

    q_rgb = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    return device, q_rgb


def scanBarcode(timeout: int = 0) -> str:
    has_barcode = ask_yes_no("Does the wine bottle have a barcode? (y/n): ")
    if not has_barcode:
        return None
    """
    Open OAK camera stream, look for a barcode, return the first decoded string.
    Press Q to quit if needed.
    timeout: milliseconds to wait (0 = wait forever).
    """
    device, q_rgb = _open_depthai(FRAME_WIDTH, FRAME_HEIGHT, DAI_DEVICE_MXID)

    print("[DepthAI] Starting QR/barcode scanner...")
    barcode_data = None

    try:
        while True:
            frame = q_rgb.get().getCvFrame()
            if frame is None:
                continue

            # Optional: mirror to behave like webcam selfie
            frame = cv2.flip(frame, 1)

            # detect barcodes
            barcodes = pyzbar.decode(frame)
            for bc in barcodes:
                x, y, w, h = bc.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = bc.data.decode("utf-8")
                barcode_data = text
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"[DepthAI] Detected barcode: {text}")
                # return immediately after first detection
                cv2.imshow("Barcode Scanner [OAK DepthAI]", frame)
                cv2.waitKey(500)  # small pause so user sees it
                return barcode_data

            # show live stream
            cv2.imshow("Barcode Scanner [OAK DepthAI]", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        device.close()
        cv2.destroyAllWindows()

    return barcode_data
