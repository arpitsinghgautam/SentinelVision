import cv2
import supervision as sv
from ultralytics import YOLOv10
import numpy as np

model = YOLOv10('train9_best.pt')

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture("test_videos/shop_armed_robbery.mp4")

# Create the sharpening kernel
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

def apply_gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

if not cap.isOpened():
    print("Unable to read video feed")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Apply CLAHE
    enhanced_frame = apply_clahe(frame)
    
    # Apply gamma correction
    enhanced_frame = apply_gamma_correction(enhanced_frame, gamma=1.5)
    
    # Reduce noise using Gaussian blur
    denoised = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)
    
    # Sharpen the image
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    results = model(sharpened)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Filter detections by confidence
    high_conf_indices = detections.confidence > 0.4
    high_conf_detections = detections[high_conf_indices]

    # Annotate the filtered detections
    if high_conf_detections.xyxy.size > 0:
        annotated_image = bounding_box_annotator.annotate(
            scene=sharpened, detections=high_conf_detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=high_conf_detections)
        
        # Save frames with detections
        cv2.imwrite(f"detected_frames/frame_{frame_count}.jpg", annotated_image)
    else:
        annotated_image = sharpened

    # Display the frame
    cv2.imshow('YOLO Detection', annotated_image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
