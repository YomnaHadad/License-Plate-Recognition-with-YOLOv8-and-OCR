from ultralytics import YOLO
from config import MODEL_PATH

model = YOLO(MODEL_PATH)

def detect_plates(image, conf):
    results = model.predict(image, conf=conf, verbose=False)
    return results