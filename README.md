# 🍅 Cherry Tomato Sorting System

An intelligent computer vision system for automated cherry tomato sorting based on ripeness and size using YOLOv11 and Streamlit.

## 📝 Overview

This project implements a real-time cherry tomato sorting system that:

- Detects and classifies cherry tomatoes by ripeness level (Unripe, Semi-ripe, Ripe)
- Measures tomato sizes and categorizes them (Small, Medium, Large)
- Provides real-time analytics and visualization
- Supports IP camera integration for continuous monitoring

## 🔧 Features

- **Real-time Detection**: Process live video feed with bounding boxes and labels
- **Multi-class Classification**:
  - Unripe (Green)
  - Semi-ripe (Orange)
  - Ripe (Bright Red)
- **Size Measurement**:
  - Small: < 20mm
  - Medium: 20-25mm
  - Large: > 25mm
- **Live Analytics**:
  - Total tomato count
  - Color distribution
  - Size distribution
- **Interactive Dashboard**:
  - Real-time video feed
  - Dynamic charts
  - Live metrics

## 🛠️ Technologies Used

- Python 3.8+
- YOLOv8 for object detection
- Streamlit for web interface
- OpenCV for image processing
- Plotly for data visualization
- NumPy for numerical operations

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/abdellatif-laghjaj/tomato-sorting-using-CNN
cd tomato-sorting-using-CNN
```
