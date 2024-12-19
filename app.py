import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('tomato_sorting_cnn.h5')  # Update with your model path
class_labels = {0: 'Green', 1: 'Orange', 2: 'Red'}


# Helper function for classification
def classify_tomato(image):
    image = image.resize((150, 150))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    return class_labels[np.argmax(prediction)], np.max(prediction)


# Helper function for diameter estimation
def estimate_diameter(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Apply GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assume the largest contour corresponds to the tomato
        largest_contour = max(contours, key=cv2.contourArea)
        # Fit a bounding circle to the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        diameter = radius * 2
        return diameter  # Return diameter in pixels
    return None


# Map diameter to size category
def classify_size(diameter):
    if diameter is None:
        return "Unknown"
    elif diameter < 20:
        return "Small"
    elif 20 <= diameter <= 25:
        return "Medium"
    else:
        return "Large"


# App Configuration
st.set_page_config(page_title="Tomato Sorting Dashboard", layout="wide", initial_sidebar_state="expanded")

# Sidebar Configuration
st.sidebar.title("Tomato Sorting Parameters")
st.sidebar.markdown("### Sorting Thresholds")
threshold_small_medium = st.sidebar.slider("Diameter Threshold (Small - Medium)", 15, 25, 20, step=1)
threshold_medium_large = st.sidebar.slider("Diameter Threshold (Medium - Large)", 25, 35, 25, step=1)

# Main UI
st.title("🍅 Tomato Sorting Dashboard")
st.markdown("""
### Welcome to the Tomato Sorting Dashboard
Easily classify tomatoes by ripeness and size. Use the options below to upload images for sorting.
""")

st.markdown("### Upload Multiple Tomato Images")
uploaded_files = st.file_uploader("Upload tomato images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.markdown("### Classification Results")
    results = {"Red": 0, "Green": 0, "Orange": 0, "Small": 0, "Medium": 0, "Large": 0}

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}")

        # Classify the tomato
        label, confidence = classify_tomato(image)
        diameter = estimate_diameter(image)
        size_label = classify_size(diameter)

        st.markdown(f"**Color Classification**: `{label}` (Confidence: `{confidence:.2f}`)")
        st.markdown(f"**Estimated Diameter**: `{diameter:.2f}px` ({size_label})")

        # Update results summary
        results[label] += 1
        if size_label in results:
            results[size_label] += 1

    # Display summary
    st.markdown("### Sorting Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Color Classification")
        st.metric("Red", results['Red'], delta=None)
        st.metric("Green", results['Green'], delta=None)
        st.metric("Orange", results['Orange'], delta=None)

    with col2:
        st.subheader("Size Classification")
        st.metric("Small", results['Small'], delta=None)
        st.metric("Medium", results['Medium'], delta=None)
        st.metric("Large", results['Large'], delta=None)

    # Add a pie chart for visualization
    st.markdown("---")
    st.markdown("### Visualization")
    color_data = [results['Red'], results['Green'], results['Orange']]
    size_data = [results['Small'], results['Medium'], results['Large']]

    color_chart = st.columns(2)
    with color_chart[0]:
        st.subheader("Color Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie(color_data, labels=['Red', 'Green', 'Orange'], autopct='%1.1f%%',
                colors=['#FF6347', '#90EE90', '#FFA500'])
        st.pyplot(fig1)

    with color_chart[1]:
        st.subheader("Size Distribution")
        fig2, ax2 = plt.subplots()
        ax2.pie(size_data, labels=['Small', 'Medium', 'Large'], autopct='%1.1f%%',
                colors=['#4682B4', '#32CD32', '#FFD700'])
        st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("""
**Tomato Sorting Dashboard**  
Modern and scalable tomato sorting for industrial needs.  
Built with ❤️ using Streamlit.
""")
