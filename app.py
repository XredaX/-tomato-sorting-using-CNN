import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

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
st.sidebar.slider("Diameter Threshold (Small - Medium)", 15, 25, 20, step=1)
st.sidebar.slider("Diameter Threshold (Medium - Large)", 25, 35, 25, step=1)

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

        st.write(f"**Color Classification**: {label} (Confidence: {confidence:.2f})")
        st.write(f"**Estimated Diameter**: {diameter:.2f}px ({size_label})")

        # Update results summary
        results[label] += 1
        if size_label in results:
            results[size_label] += 1

    # Display summary
    st.markdown("### Sorting Summary")
    st.write("**Color Classification:**")
    st.write(f"- Red: {results['Red']}")
    st.write(f"- Green: {results['Green']}")
    st.write(f"- Orange: {results['Orange']}")
    st.write("**Size Classification:**")
    st.write(f"- Small: {results['Small']}")
    st.write(f"- Medium: {results['Medium']}")
    st.write(f"- Large: {results['Large']}")

# Footer
st.markdown("---")
st.markdown("""
**Tomato Sorting Dashboard**  
Modern and scalable tomato sorting for industrial needs.  
Built with ❤️ using Streamlit.
""")
