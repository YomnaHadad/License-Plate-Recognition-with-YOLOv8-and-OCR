# YOLO+OCR for License Plate Recognition

## Project Overview
This project develops an automated system for detecting and recognizing license plates in both images and video streams. It integrates YOLO (You Only Look Once) for efficient object detection of license plates with EasyOCR for optical character recognition. The system is designed to provide stable recognition results, especially in video, by using a voting mechanism. It offers two user interfaces: a Gradio application for quick image inference and a more robust Streamlit application capable of processing both images and videos.

* **Interactive Web Interfaces**: Provides [Streamlit](https://license-plate-recognition-with-yolov8-and-ocr.streamlit.app/) application for Advanced interface supporting both image and video uploads, offering a cleaner UI and download options for processed videos.

## Features
-   **License Plate Detection**: Employs a fine-tuned YOLOv8 model to accurately locate license plates within frames.
-   **Optical Character Recognition (OCR)**: Uses EasyOCR to extract alphanumeric characters from the detected license plate regions.
-   **Plate Text Stabilization**: For video processing, a history-based voting mechanism is implemented to reduce flickering and improve the stability of recognized plate numbers over time.
-   **Flexible Plate Pattern Matching**: Utilizes a regular expression to validate recognized plate numbers, allowing for diverse global license plate formats.
-   **Video Processing**: Processes video files frame by frame, annotates them with bounding boxes and recognized plate text, and saves the output as a new video file.

## Technologies Used
-   **Ultralytics YOLOv8**
-   **EasyOCR**
-   **OpenCV**
-   **Roboflow**
-   **Gradio**
-   **Streamlit**
-   **Python**
-   **GPU Acceleration**

## Project Workflow
The general processing pipeline for detecting and recognizing license plates is as follows:

```
Frame/Image Input
   │
   ▼
 YOLO 
   │
   ▼
Crop Detected Plate Region
   │
   ▼
Image Preprocessing for OCR
   │
   ▼
EasyOCR 
   │
   ▼
Recognized Text
   │
   ▼
(For video) Plate Stabilization (Majority Vote)
   │
   ▼
Annotated Frame/Image (Bounding Box + Text)
```

## Setup and Installation

To run this project, you'll need to install the required Python libraries and set up the YOLO model.

### 1. Install Dependencies
```bash
!pip install -q roboflow ultralytics easyocr opencv-python-headless streamlit
```

### 2. Roboflow Dataset Download and YOLO Training
This project uses a custom YOLOv8 model trained on a license plate dataset from Roboflow.

First, set your Roboflow API key (e.g., in Google Colab secrets or environment variables).

```python
from roboflow import Roboflow
import os
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY") 
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(13)
dataset = version.download("yolov8")
```

Next, fine-tune a YOLOv8n model on the downloaded dataset. This step requires a GPU for efficient training.

```python
from ultralytics import YOLO
import shutil, os

model = YOLO("yolov8n.pt") 

results = model.train(
    data = dataset.location + "/data.yaml",
    epochs = 5,      
    imgsz = 640,     
    batch = 16,      
    patience = 3,    
    workers = 4,     
    device = 0       
)
os.makedirs("save_models", exist_ok=True)
shutil.copy("runs/detect/train/weights/best.pt",
            "save_models/best_license_plate_recognition.pt")
print("Weights saved in saved_models/")
```

## Usage

### 1. Gradio Web Interface

```python
import gradio as gr
# ... (Gradio setup code)

iface.launch(debug=True, share=True)
```
This will launch a local server and provide a public URL (if run in Colab) where you can upload images to see the annotated results.

### 2. Streamlit Web Application

The newly generated Streamlit code provides a dynamic interface for both image and video uploads. To run the Streamlit app:

Run the application from your terminal:
    ```bash
    streamlit run app.py
    ```

**Streamlit App Features:**
-   **File Uploader**: Automatically distinguishes between image and video uploads.
-   **Image Processing**: Upload an image, and it displays the original and annotated image with recognized plate numbers.
-   **Video Processing**: Upload a video, and it processes it frame by frame, displays a progress bar, and then plays the annotated video within the app. It also provides a download button for the processed video.
-   **Clean UI**: Uses Streamlit's native components for a modern and interactive user experience.
