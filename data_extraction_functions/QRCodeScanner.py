import cv2
import numpy as np
from pyzbar.pyzbar import decode

def ask_yes_no(prompt: str) -> bool:
    """
    Repeatedly prompt the user until they enter yes/y or no/n.
    Returns True for yes, False for no.
    """
    valid_yes = {"y", "yes"}
    valid_no  = {"n", "no"}
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

def scanBarcode(cam_index=0):
    has_barcode = ask_yes_no("Does the wine bottle have a barcode? (y/n): ")
    if not has_barcode:
        return None

    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera (index {cam_index}).")

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)

    try:
        # Warm up a few frames
        for _ in range(5):
            cap.read()

        while True:
            success, img = cap.read()
            if not success or img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray)

            codes = decode(gray)
            if codes:
                print("Scanned")
                myData = codes[0].data.decode('utf-8', errors='replace')
                return myData

            cv2.imshow('Result', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None
    finally:
        cap.release()