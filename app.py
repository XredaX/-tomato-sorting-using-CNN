import streamlit as st
import cv2
from threading import Thread
from queue import Queue
import numpy as np
from ultralytics import YOLO
import plotly.express as px
from collections import deque
import time

# Load the YOLO model
model = YOLO("models/model.pt")

# Define color labels and their BGR colors for visualization
color_labels = {
    0: ("Unripe (Green)", (0, 255, 0)),
    1: ("Semi-ripe (Orange)", (0, 165, 255)),
    2: ("Ripe (Bright Red)", (0, 0, 255))
}

# Size parameters
MIN_DIAMETER_MM = 15
MAX_DIAMETER_MM = 30

def determine_size(diameter_mm):
    if diameter_mm < 20:
        return "Small"
    elif 20 <= diameter_mm <= 25:
        return "Medium"
    else:
        return "Large"

# App Configuration
st.set_page_config(page_title="Cherry Tomatoes Sorting Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize Streamlit app
st.title("🍅 Cherry Tomato Sorting Dashboard")
st.markdown("""Easily classify tomatoes by ripeness and size..""")

# Sidebar settings
st.sidebar.header("Camera Feed Settings")
ip_camera_url = st.sidebar.text_input("IP Camera URL", "http://192.168.0.101:8080/video")

# Create placeholder for video feed
video_placeholder = st.empty()

# Create columns for metrics
col1, col2, col3 = st.columns(3)
with col1:
    total_counter = st.empty()
with col2:
    color_metrics = {label: st.empty() for label, _ in color_labels.values()}
with col3:
    size_metrics = {size: st.empty() for size in ["Small", "Medium", "Large"]}

# Create placeholders for real-time charts
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    color_chart_placeholder = st.empty()
with chart_col2:
    size_chart_placeholder = st.empty()

# Tracking variables
tracked_tomatoes = {}
last_seen = {}
TRACKING_THRESHOLD = 30  # pixels
DISAPPEAR_THRESHOLD = 2.0  # seconds
UPDATE_INTERVAL = 0.5  # seconds
last_chart_update = 0
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame
FRAME_WIDTH = 640  # Reduced frame width
UPDATE_INTERVAL = 1.0  # Increase chart update interval to 1 second

def update_charts(color_counts, size_counts, current_time):
    global last_chart_update
    
    # Only update charts every UPDATE_INTERVAL seconds
    if current_time - last_chart_update >= UPDATE_INTERVAL:
        # Update color distribution chart
        fig_color = px.pie(
            names=list(color_counts.keys()),
            values=list(color_counts.values()),
            title="Color Distribution"
        )
        color_chart_placeholder.plotly_chart(fig_color, use_container_width=True, key=f"color_chart_{current_time}")

        # Update size distribution chart
        fig_size = px.pie(
            names=list(size_counts.keys()),
            values=list(size_counts.values()),
            title="Size Distribution"
        )
        size_chart_placeholder.plotly_chart(fig_size, use_container_width=True, key=f"size_chart_{current_time}")
        
        last_chart_update = current_time

def process_frame(frame, current_time):
    color_counts = {label: 0 for label, _ in color_labels.values()}
    size_counts = {"Small": 0, "Medium": 0, "Large": 0}
    
    results = model.predict(source=frame, imgsz=640, conf=0.6)
    
    # Current detected centroids
    current_centroids = []
    
    for result in results[0]:
        x1, y1, x2, y2 = map(int, result.boxes.xyxy[0].tolist())
        cls = int(result.boxes.cls[0].item())
        conf = float(result.boxes.conf[0].item())
        
        if cls not in color_labels:
            continue
            
        # Calculate centroid
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        current_centroids.append((centroid, cls, (x1, y1, x2, y2)))
        
        # Calculate diameter and size
        diameter_px = max(abs(x2 - x1), abs(y2 - y1))
        diameter_mm = diameter_px / frame.shape[1] * 100
        size_category = determine_size(diameter_mm)
        
        # Draw bounding box and enhanced label
        color_name, color_bgr = color_labels[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Create multi-line label
        label_lines = [
            f"{color_name}",
            f"{diameter_mm:.1f}mm ({size_category})",
            f"Conf: {conf:.2f}"
        ]
        
        # Draw background rectangle for text
        text_size = cv2.getTextSize(max(label_lines, key=len), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, 
                     (x1, y1 - 10 - (len(label_lines) * 15)), 
                     (x1 + text_size[0], y1),
                     color_bgr, 
                     -1)  # Filled rectangle
        
        # Draw each line of text
        for i, line in enumerate(label_lines):
            y_pos = y1 - 5 - ((len(label_lines) - 1 - i) * 15)
            cv2.putText(frame, 
                       line,
                       (x1, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 255, 255),  # White text
                       2)
        
        # Update tracking
        tracked = False
        for tracked_id, tracked_info in tracked_tomatoes.items():
            tracked_centroid = tracked_info['centroid']
            distance = np.sqrt((centroid[0] - tracked_centroid[0])**2 + 
                             (centroid[1] - tracked_centroid[1])**2)
            
            if distance < TRACKING_THRESHOLD:
                tracked = True
                last_seen[tracked_id] = current_time
                tracked_tomatoes[tracked_id].update({
                    'centroid': centroid,
                    'class': cls,
                    'size': size_category
                })
                break
                
        if not tracked:
            new_id = max(tracked_tomatoes.keys(), default=-1) + 1
            tracked_tomatoes[new_id] = {
                'centroid': centroid,
                'class': cls,
                'size': size_category
            }
            last_seen[new_id] = current_time
    
    # Remove disappeared tomatoes
    disappeared = []
    for tracked_id, last_time in last_seen.items():
        if current_time - last_time > DISAPPEAR_THRESHOLD:
            disappeared.append(tracked_id)
    
    for tracked_id in disappeared:
        del tracked_tomatoes[tracked_id]
        del last_seen[tracked_id]
    
    # Count current tomatoes
    for tracked_info in tracked_tomatoes.values():
        color_name = color_labels[tracked_info['class']][0]
        size_category = tracked_info['size']
        color_counts[color_name] += 1
        size_counts[size_category] += 1
    
    return frame, color_counts, size_counts

if st.sidebar.button("Start Processing"):
    try:
        cap = cv2.VideoCapture(ip_camera_url)
        frame_count = 0
        
        # Create queues for async processing
        frame_queue = Queue(maxsize=1)
        result_queue = Queue(maxsize=1)
        
        def process_frames():
            while cap.isOpened():
                if not frame_queue.empty():
                    frame = frame_queue.get()
                    current_time = time.time()
                    processed_frame, color_counts, size_counts = process_frame(frame, current_time)
                    
                    # Put results in queue, overwriting old results
                    if result_queue.full():
                        result_queue.get()
                    result_queue.put((processed_frame, color_counts, size_counts, current_time))
        
        # Start processing thread
        process_thread = Thread(target=process_frames, daemon=True)
        process_thread.start()
        
        last_processed_frame = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to fetch frame from camera.")
                break
            
            frame_count += 1
            
            # Resize frame for better performance
            frame = cv2.resize(frame, (FRAME_WIDTH, int(FRAME_WIDTH * frame.shape[0] / frame.shape[1])))
            
            # Process every Nth frame
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                # Update frame queue, dropping old frame if necessary
                if frame_queue.full():
                    frame_queue.get()
                frame_queue.put(frame.copy())
            
            # Update metrics and get processed frame if available
            if not result_queue.empty():
                last_processed_frame, color_counts, size_counts, current_time = result_queue.get()
                
                # Update metrics
                total_counter.metric("Total Tomatoes", sum(color_counts.values()))
                
                # Update color metrics
                for color, count in color_counts.items():
                    color_metrics[color].metric(color, count)
                
                # Update size metrics
                for size, count in size_counts.items():
                    size_metrics[size].metric(size, count)
                
                # Update charts with throttling
                update_charts(color_counts, size_counts, current_time)
            
            # Display the processed frame if available, otherwise display raw frame
            display_frame = last_processed_frame if last_processed_frame is not None else frame
            video_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), 
                                 channels="RGB",
                                 use_container_width=True)
            
            # Small delay to prevent CPU overload
            time.sleep(0.001)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()