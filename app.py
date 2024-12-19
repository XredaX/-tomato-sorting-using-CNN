import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
model = load_model('tomato_sorting_cnn.h5')  # Update with your model path
class_labels = {0: 'Red', 1: 'Green', 2: 'Orange'}

# Helper function for classification
def classify_tomato(image):
    image = image.resize((150, 150))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    return class_labels[np.argmax(prediction)], np.max(prediction)

# Real-time webcam function
def webcam_stream():
    st.title("Tomato Sorting - Webcam Stream")
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        # Preprocess and classify frame (optional)
    camera.release()

# App Configuration
st.set_page_config(page_title="Tomato Sorting Dashboard", layout="wide", initial_sidebar_state="expanded")

# Sidebar Configuration
st.sidebar.title("Tomato Sorting Parameters")
st.sidebar.markdown("### Upload or stream tomatoes for sorting")
mode = st.sidebar.radio("Select Mode", ("Upload Images", "Real-time Webcam"))

# Main UI
st.title("🍅 Tomato Sorting Dashboard")
st.markdown("""
### Welcome to the Tomato Sorting Dashboard
Easily classify tomatoes by ripeness and size. Use the options below to upload images or start a real-time webcam stream.
""")

if mode == "Upload Images":
    st.markdown("### Upload Multiple Tomato Images")
    uploaded_files = st.file_uploader("Upload tomato images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.markdown("### Classification Results")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

            # Classify the tomato
            label, confidence = classify_tomato(image)
            st.write(f"**Classification**: {label} (Confidence: {confidence:.2f})")

elif mode == "Real-time Webcam":
    webcam_stream()

# Footer
st.markdown("---")
st.markdown("""
**Tomato Sorting Dashboard**  
Modern and scalable tomato sorting for industrial needs.  
Built with ❤️ using Streamlit.
""")
