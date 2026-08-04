import cv2
from ocr import recognize_plate
from yolo_detector import detect_plates
from config import CONF_THRESHOLD, ALLOWLIST


def process_image_for_plates_streamlit(img):
    '''Image Processing Function'''
    annotated_image = img.copy()
    detected_plates = []

    results = detect_plates(img, conf=CONF_THRESHOLD)

    for result in results:
        for *xyxy, conf, cls in result.boxes.data:
            x1, y1, x2, y2 = map(int, xyxy)

            plate_crop = annotated_image[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]

            if plate_crop.shape[0] > 0 and plate_crop.shape[1] > 0:
                recognized_text = recognize_plate(plate_crop)

                if recognized_text:
                    detected_plates.append(recognized_text)

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_image, recognized_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return annotated_image, ", ".join(detected_plates) if detected_plates else "No plates detected."
