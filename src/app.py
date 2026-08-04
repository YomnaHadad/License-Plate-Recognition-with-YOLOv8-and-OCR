# Streamlit Application for License Plate Recognition
import cv2, os
import tempfile
import numpy as np
import streamlit as st
from ocr import recognize_plate
from yolo_detector import detect_plates
from config import PHOTO_TYPES, CONF_THRESHOLD
from preprocessor import process_image_for_plates_streamlit
from stabilizer import get_stable_plate, get_box_id, plate_final, plate_history

st.set_page_config(page_title="License Plate Recognition", layout="centered")
st.title("🚗 License Plate Recognition App")
st.markdown("Upload an image or a video to detect and recognize license plates.")

uploaded_file = st.file_uploader("Choose a file...", type=PHOTO_TYPES) 

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        st.header("Image Processing")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img, caption="Original Image", use_column_width=True)

        st.subheader("Processing...")
        annotated_image, plate_numbers = process_image_for_plates_streamlit(img)

        st.image(annotated_image, caption="Annotated Image", use_column_width=True)
        st.info(f"**Detected Plate Numbers:** {plate_numbers}")

    elif "video" in file_type:
        st.header("Video Processing")
        st.warning("Video processing can take some time depending on its length. Please be patient.")

        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type.split('/')[-1]}") as temp_input_file:
            temp_input_file.write(uploaded_file.read())
            temp_input_path = temp_input_file.name

        # Prepare output video path
        output_video_path = temp_input_path.replace(f".{file_type.split('/')[-1]}", "_processed.mp4")

        cap = cv2.VideoCapture(temp_input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        progress_bar = st.progress(0)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Clear stabilization history for a new video
        plate_history.clear()
        plate_final.clear()

        processing_status = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Update progress bar
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            processing_status.text(f"Processing frame {frame_count}/{total_frames}...")

            # Perform YOLO inference
            results = detect_plates(frame, conf=CONF_THRESHOLD)

            for result in results:
                for *xyxy, conf, cls in result.boxes.data:
                    x1, y1, x2, y2 = map(int, xyxy)

                    plate_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

                    if plate_crop.shape[0] > 0 and plate_crop.shape[1] > 0:
                        recognized_text = recognize_plate(plate_crop)

                        box_id = get_box_id(x1, y1, x2, y2)
                        stable_plate_text = get_stable_plate(box_id, recognized_text)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if stable_plate_text:
                            cv2.putText(frame, stable_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        # Clean up temporary input file
        os.remove(temp_input_path)

        processing_status.success(f"Processed {frame_count} frames. Video saved to output.")

        st.subheader("Processed Video")
        st.video(output_video_path)

        with open(output_video_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_license_plates.mp4",
                mime="video/mp4"
            )
        # Clean up temporary output file after display/download
        os.remove(output_video_path)

    else:
        st.error("Unsupported file type. Please upload an image (jpg, png) or a video (mp4, mov, avi).")
