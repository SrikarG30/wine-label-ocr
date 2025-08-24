from ultralytics import YOLO
import cv2
import json
import sys
from ocr_wine import extract_fields

def main(image_path, weights_path):
    model = YOLO(weights_path)
    img = cv2.imread(image_path)
    # Run inference
    pred = model(img, verbose=False)[0]
    # Map class IDs to names; adjust according to your model
    id_to_name = {0: "Distinct Logo", 1: "Maker-Name", 2: "Vintage"}
    detections = []
    for b in pred.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        cls_id = int(b.cls[0])
        detections.append({
            "class": id_to_name.get(cls_id, str(cls_id)),
            "box": [x1, y1, x2, y2]
        })
    result = extract_fields(img, detections)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_ocr.py <image_path> <weights_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
