from ultralytics import YOLO
import cv2
# Step 3: Deploy Model for Real-Time Sorting
# Load the best model from training
model = YOLO("model.pt")

# Define color labels
color_labels = {
    0: "Unripe (Green)",
    1: "Semi-ripe (Orange)",
    2: "Ripe (Bright Red)"
}
results = []
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
            
            # Calculate diameter (assuming circular tomatoes)
            diameter_px = max(abs(x2 - x1), abs(y2 - y1))
            diameter_mm = diameter_px / frame.shape[1] * 100  # Approximation based on field of view
            size_label = determine_size(diameter_mm)
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{color_label}, {size_label}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Real-time video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera; replace with video file path if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame, model)
    
    cv2.imshow("Cherry Tomato Sorting", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Step 4: Dashboard Integration (Mock Implementation)
# This would typically involve a separate web app using Flask/Django/Streamlit
# Here we'll display a simple real-time statistic
sorted_count = {"Ripe": 0, "Semi-ripe": 0, "Unripe": 0}

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0].item())
        label = color_labels[cls]
        sorted_count[label] += 1

print("Sorting Statistics:", sorted_count)