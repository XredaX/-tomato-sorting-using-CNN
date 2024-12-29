from ultralytics import YOLO
import cv2
import requests
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# Step 3: Deploy Model for Real-Time Sorting
# Load the best model from training
model = YOLO("model.pt")

# Define color labels
color_labels = {
    0: "Unripe (Green)",
    1: "Semi-ripe (Orange)",
    2: "Ripe (Bright Red)"
}

# Define bounding box colors for each label
bbox_colors = {
    0: (0, 255, 0),    # Green
    1: (0, 165, 255),  # Orange
    2: (0, 0, 255)     # Red
}

# Function to determine size category based on diameter
def determine_size(diameter_mm):
    if diameter_mm < 20:
        return "Small"
    elif 20 <= diameter_mm <= 25:
        return "Medium"
    else:
        return "Large"

# Function to process real-time video stream
def process_frame(frame, model):
    results = model.predict(source=frame, imgsz=640, conf=0.5)  # Adjust confidence threshold as needed
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            color_label = color_labels[cls]
            bbox_color = bbox_colors[cls]
            
            # Calculate diameter (assuming circular tomatoes)
            diameter_px = max(abs(x2 - x1), abs(y2 - y1))
            diameter_mm = diameter_px / frame.shape[1] * 100  # Approximation based on field of view
            size_label = determine_size(diameter_mm)
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            label = f"{color_label}, {size_label} ({diameter_mm:.1f} mm)"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

    return frame

# Real-time video capture using IP Webcam
ip_camera_url = "http://192.168.0.101:8080/video"
cap = cv2.VideoCapture(ip_camera_url)

while True:
    ret, frame = cap.read()
    if not ret:
        # Attempt to fetch the frame from the IP webcam
        response = requests.get(ip_camera_url, stream=True)
        if response.status_code == 200:
            bytes_data = bytes()
            for chunk in response.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b + 2]
                    bytes_data = bytes_data[b + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    break
        else:
            print("Failed to fetch the frame from IP webcam.")
            break

    processed_frame = process_frame(frame, model)

    # Resize window to fit within screen boundaries
    resized_frame = cv2.resize(processed_frame, (960, 540))  # Adjust dimensions as needed
    cv2.imshow("Cherry Tomato Sorting", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Step 4: Dashboard Integration (Mock Implementation)
# This would typically involve a separate web app using Flask/Django/Streamlit
# Here we'll display a simple real-time statistic
sorted_count = {"Ripe": 0, "Semi-ripe": 0, "Unripe": 0}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, imgsz=640, conf=0.5)  # Predict multiple tomatoes in real-time

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())
            label = color_labels[cls]
            sorted_count[label] += 1

print("Sorting Statistics:", sorted_count)