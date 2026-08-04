import cv2
import gradio as gr
from config import CONF_THRESHOLD
from yolo_detector import detect_plates
from ocr import recognize_plate

def process_image_for_plates(img):
    annotated_image = img.copy()
    detected_plates = []

    results = detect_plates(img, conf=CONF_THRESHOLD)

    for result in results:
        for *xyxy, conf, cls in result.boxes.data:
            x1, y1, x2, y2 = map(int, xyxy)

            # Crop license plate region, ensuring coordinates are within bounds
            # And convert to numpy array if it's a torch tensor or similar
            plate_crop = annotated_image[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]

            if plate_crop.shape[0] > 0 and plate_crop.shape[1] > 0:
                # recognize plate using OCR
                recognized_text = recognize_plate(plate_crop)

                if recognized_text:
                    detected_plates.append(recognized_text)

                    # Draw bounding box and text on the annotated image
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                    cv2.putText(annotated_image, recognized_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return annotated_image, ", ".join(detected_plates) if detected_plates else "No plates detected."

iface = gr.Interface(
    fn=process_image_for_plates,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(type="numpy", label="Annotated Image"),
        gr.Textbox(label="Detected Plate Numbers")
    ],
    title="License Plate Recognition with YOLOv8 and OCR",
    description="Upload an image to detect and recognize license plates."
)

iface.launch(debug=True,share=True)