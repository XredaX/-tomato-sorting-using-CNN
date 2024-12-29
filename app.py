import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import plotly.express as px

# Load the YOLO model
model = YOLO("models/model.pt")

# Define color labels
color_labels = {
    0: "Unripe (Green)",
    1: "Semi-ripe (Orange)",
    2: "Ripe (Bright Red)"
}

# Expected size range for cherry tomatoes (in mm)
MIN_DIAMETER_MM = 15
MAX_DIAMETER_MM = 30

# Function to determine size category based on diameter
def determine_size(diameter_mm):
    if diameter_mm < 20:
        return "Small"
    elif 20 <= diameter_mm <= 25:
        return "Medium"
    else:
        return "Large"

# Initialize Streamlit app
st.title("Cherry Tomato Sorting Dashboard")

# Sidebar for real-time video feed
st.sidebar.header("Camera Feed Settings")
ip_camera_url = st.sidebar.text_input("IP Camera URL", "http://192.168.0.101:8080/video")

# Placeholder for the video feed
video_placeholder = st.empty()

# Metrics placeholders
col1, col2, col3 = st.columns(3)
total_tomatoes_placeholder = col1.metric("Total Tomatoes", "0")
color_distribution_placeholder = col2.metric("Color Distribution", "N/A")
size_distribution_placeholder = col3.metric("Size Distribution", "N/A")

# Data for plots
color_counts = {"Unripe (Green)": 0, "Semi-ripe (Orange)": 0, "Ripe (Bright Red)": 0}
size_counts = {"Small": 0, "Medium": 0, "Large": 0}

# Start processing video feed
if st.sidebar.button("Start Processing"):
    cap = cv2.VideoCapture(ip_camera_url)
    total_tomatoes = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to fetch frame from camera.")
            break

        results = model.predict(source=frame, imgsz=640, conf=0.6)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0].item())

                # Skip objects that are not cherry tomatoes
                if cls not in color_labels:
                    continue

                # Calculate diameter (assuming circular tomatoes)
                diameter_px = max(abs(x2 - x1), abs(y2 - y1))
                diameter_mm = diameter_px / frame.shape[1] * 100

                # Filter by size
                if diameter_mm < MIN_DIAMETER_MM or diameter_mm > MAX_DIAMETER_MM:
                    continue

                color_label = color_labels[cls]
                size_label = determine_size(diameter_mm)

                # Update counts
                total_tomatoes += 1
                color_counts[color_label] += 1
                size_counts[size_label] += 1

        # Update metrics
        total_tomatoes_placeholder.metric("Total Tomatoes", total_tomatoes)
        color_distribution_placeholder.metric("Color Distribution", str(color_counts))
        size_distribution_placeholder.metric("Size Distribution", str(size_counts))

        # Display video feed
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Generate visualizations
if st.sidebar.button("Generate Visualizations"):
    # Pie chart for color distribution
    fig_color = px.pie(
        names=list(color_counts.keys()),
        values=list(color_counts.values()),
        title="Color Distribution"
    )
    st.plotly_chart(fig_color)

    # Pie chart for size distribution
    fig_size = px.pie(
        names=list(size_counts.keys()),
        values=list(size_counts.values()),
        title="Size Distribution"
    )
    st.plotly_chart(fig_size)