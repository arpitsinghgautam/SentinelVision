import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLOv10
import supervision as sv
import os

# Load YOLOv10 model
model = YOLOv10('8_best.pt')

# Function to apply gamma correction
def apply_gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function to apply CLAHE
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Function to process video and annotate
def process_video(input_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        st.error("Error: Unable to open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_output_path = 'temp_annotated_video.avi'
    output_video = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    # Create the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gun = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply CLAHE
        enhanced_frame = apply_clahe(frame)

        # Apply gamma correction
        enhanced_frame = apply_gamma_correction(enhanced_frame, gamma=1.5)

        # Reduce noise using Gaussian blur
        denoised = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)

        # Sharpen the image
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Perform detection
        results = model(sharpened)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter detections by confidence
        high_conf_indices = detections.confidence > 0.4
        high_conf_detections = detections[high_conf_indices]

        # Annotate the frame if detections exist
        if high_conf_detections.xyxy.size > 0:
            annotated_image = sv.BoundingBoxAnnotator().annotate(scene=sharpened, detections=high_conf_detections)
            annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=high_conf_detections)
            gun = 1
        else:
            annotated_image = sharpened

        # Write annotated frame to output video
        output_video.write(annotated_image)
    if gun ==1:
        st.error("Gun detected! Calling police...")
    # Release the video capture and writer objects
    cap.release()
    output_video.release()

    return temp_output_path

# Streamlit UI
st.title("SentinelVision Gun Detection")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.write(uploaded_file)
    # Save uploaded video to a temporary location
    temp_video_path = "temp_uploaded_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.write("Uploaded video:")
    st.video(uploaded_file)

    # Process the video and get annotated video path
    annotated_video_path = process_video(temp_video_path)

    # Display annotated video
    if annotated_video_path:
        st.video(annotated_video_path)

    # # Clean up temporary files
    # if os.path.exists(temp_video_path):
    #     os.remove(temp_video_path)
    # if os.path.exists(annotated_video_path):
    #     os.remove(annotated_video_path)
