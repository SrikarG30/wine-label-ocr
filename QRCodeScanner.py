import cv2
import numpy as np
from pyzbar.pyzbar import decode

# 0 for default webcam
def scanBarcode(cameraIndex):
    cap = cv2.VideoCapture(cameraIndex)
    cap.set(3,640)
    cap.set(4,480)

    while True:
        success, img = cap.read()
        for barcode in decode(img):
            print("Scanned")
            myData = barcode.data.decode('utf-8')
            cap.release()
            cv2.destroyAllWindows()
            return(myData)
        cv2.imshow('Result', img)
        cv2.waitKey(1)

print(scanBarcode(0))
