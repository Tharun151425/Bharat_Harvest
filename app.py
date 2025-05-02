import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import torch
from ultralytics import YOLO
import io
import base64
from datetime import datetime
import time

# Set page config
st.set_page_config(
    page_title="Vegetable Detection App",
    page_icon="🥕",
    layout="wide",
)

# Custom CSS for styling
def local_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #4CAF50; /* Green */
        --text-color: #333333;    /* Dark text */
        --bg-color: #FFFFFF;      /* White background */
        --accent-color: #2E7D32;  /* Dark green */
        --secondary-bg: #F5F5F5;  /* Light gray */
    }
    
    /* Dark mode colors */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #4CAF50;  /* Green */
            --text-color: #E0E0E0;     /* Light text */
            --bg-color: #121212;       /* Dark background */
            --accent-color: #81C784;   /* Light green */
            --secondary-bg: #1E1E1E;   /* Darker gray */
        }
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    h1, h2, h3 {
        color: var(--primary-color) !important;
        font-weight: 600 !important;
    }
    
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .upload-container {
        border: 2px dashed var(--primary-color);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: var(--accent-color);
        background-color: var(--secondary-bg);
    }
    
    .result-container {
        padding: 15px;
        border-radius: 10px;
        background-color: var(--secondary-bg);
        margin-top: 20px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--secondary-bg);
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: var(--text-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    .about-section {
        background-color: var(--secondary-bg);
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
    }
    
    footer {
        margin-top: 30px;
        text-align: center;
        color: var(--text-color);
        opacity: 0.8;
        font-size: 0.8rem;
    }
    
    .badge {
        background-color: var(--primary-color);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    
    .stImage {
        border-radius: 10px;
        transition: transform 0.3s ease;
    }
    
    .stImage:hover {
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# Call the CSS function
local_css()

# Load the model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Load your model file (make sure it's in the same directory)
    return model

# Function to make predictions
def predict(image, model):
    results = model(image)
    return results

# Function to plot results
def plot_results(results, img):
    res_plotted = results[0].plot()
    return Image.fromarray(res_plotted)

# Function to get detection details
def get_detection_details(results):
    if not results:
        return []
    
    boxes = results[0].boxes
    
    if not boxes:
        return []
    
    class_names = results[0].names
    
    detection_details = []
    for box in boxes:
        class_id = int(box.cls[0].item())
        class_name = class_names[class_id]
        confidence = box.conf[0].item()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        detection_details.append({
            "class_name": class_name,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        })
    
    return detection_details

# Function to download image with annotations
def get_image_download_link(img, filename="detected_image.jpg", text="Download Annotated Image"):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}" class="download-btn">{text}</a>'
    return href

# Function to run webcam
def run_webcam(model):
    st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
    webcam_placeholder = st.empty()
    stop_button_container = st.container()
    
    # Create a counter for the processed frames
    frame_counter = st.empty()
    counter = 0
    fps_counter = st.empty()
    
    stop_button_pressed = stop_button_container.button("Stop Webcam")
    
    camera = cv2.VideoCapture(0)
    
    start_time = time.time()
    frame_count = 0
    
    while not stop_button_pressed:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access webcam")
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run prediction
        results = predict(frame_rgb, model)
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        webcam_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
        
        # Update counters
        counter += 1
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= 1.0:  # Update FPS every second
            fps = frame_count / elapsed_time
            fps_counter.markdown(f"<div style='text-align: center;'><strong>FPS:</strong> {fps:.2f}</div>", unsafe_allow_html=True)
            start_time = current_time
            frame_count = 0
            
        frame_counter.markdown(f"<div style='text-align: center;'><strong>Frames Processed:</strong> {counter}</div>", unsafe_allow_html=True)
        
        # Check again if stop button was pressed
        if stop_button_container.button("Stop Webcam", key=f"stop_button_{counter}"):
            break
            
    # Release the camera
    camera.release()
    st.markdown('</div>', unsafe_allow_html=True)
    st.success("Webcam stopped")
    
# Main app header
st.markdown("<h1 style='text-align: center;'>🥕 Vegetable Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image or use your webcam to detect vegetables using YOLOv8</p>", unsafe_allow_html=True)

# Create tabs for different input methods
tabs = st.tabs(["📷 Upload Image", "🎥 Use Webcam"])

# Load the model
with st.spinner("Loading YOLOv8 model..."):
    model = load_model()
    st.success("Model loaded successfully!")

with tabs[0]:  # Upload Image Tab
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    st.markdown("### 📤 Drag and Drop Image")
    st.markdown("Drop your image here or click to browse")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file)
            results = predict(image, model)
            result_image = plot_results(results, image)
            detection_details = get_detection_details(results)
            
            # Side-by-side layout for images
            st.markdown("<div class='result-container' style='box-shadow:0 2px 16px rgba(44,62,80,0.08);'>", unsafe_allow_html=True)
            img_cols = st.columns(2, gap="large")
            with img_cols[0]:
                st.markdown("<div style='text-align:center;'><strong>Original Image</strong></div>", unsafe_allow_html=True)
                st.image(image, use_column_width=True)
            with img_cols[1]:
                st.markdown("<div style='text-align:center;'><strong>Detection Result</strong></div>", unsafe_allow_html=True)
                st.image(result_image, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Detection details and download
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            if detection_details:
                st.markdown("### Detected Vegetables")
                for i, det in enumerate(detection_details):
                    cols = st.columns([1, 2])
                    with cols[0]:
                        st.markdown(f"**{i+1}. {det['class_name'].capitalize()}**")
                    with cols[1]:
                        st.markdown(f"Confidence: **{det['confidence']*100:.2f}%**")
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"vegetable_detection_{now}.jpg"
                st.markdown(get_image_download_link(result_image, filename=filename, text="Download Annotated Image"), unsafe_allow_html=True)
            else:
                st.info("No vegetables detected in the image.")
            st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:  # Webcam Tab
    st.markdown("### 🎥 Live Webcam Detection")
    st.markdown("Use your webcam to detect vegetables in real-time")
    
    if st.button("Start Webcam"):
        run_webcam(model)

# About section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='about-section'>", unsafe_allow_html=True)
st.markdown("## About YOLOv8 Vegetable Detection")

# Create two columns for the About section
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### How YOLOv8 Works
    
    **YOLO (You Only Look Once)** is a state-of-the-art object detection algorithm that processes images in a single pass through a neural network. YOLOv8 is the latest version with improved accuracy and speed.
    
    **Key features of YOLOv8:**
    - Real-time object detection
    - High accuracy and precision
    - Single-stage detection process
    - Anchor-free detection system
    - Improved small object detection
    
    This application uses a YOLOv8 model trained on a custom dataset of vegetables, capable of detecting 10 different types: beans, brinjal, cabbage, capsicum, carrot, cauliflower, chilli, onion, potato, and tomato.
    """)
    
with col2:
    st.markdown("""
    ### Training Dataset
    
    The model was trained on a custom dataset from Roboflow:
    - **Dataset**: Mini Project Vegetable Detection
    - **Classes**: 10 vegetable types
    - **Version**: 4
    - **License**: CC BY 4.0
    
    The training process involved:
    1. Data collection and annotation
    2. Data augmentation (rotation, scaling, etc.)
    3. Training the YOLOv8 architecture
    4. Validation and fine-tuning
    5. Testing on real-world images
    
    The model achieves high accuracy in identifying vegetables in various lighting conditions and backgrounds.
    """)

st.markdown("""
### Detection Process
1. **Input**: The system takes an image/video frame as input
2. **Feature Extraction**: The neural network extracts features from the image
3. **Grid Division**: The image is divided into a grid 
4. **Bounding Box Prediction**: For each grid cell, the model predicts:
   - Bounding box coordinates
   - Confidence scores
   - Class probabilities
5. **Non-Maximum Suppression**: Overlapping detections are removed
6. **Output**: Final detections with class labels and confidence scores
""")

st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
st.markdown("### Vegetable Classes")
class_names = ['Beans', 'Brinjal', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Chilli', 'Onion', 'Potato', 'Tomato']
cols = st.columns(5)
for i, class_name in enumerate(class_names):
    with cols[i % 5]:
        st.markdown(f"<span class='badge'>{class_name}</span>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<footer>", unsafe_allow_html=True)
st.markdown("© 2025 Vegetable Detection App | YOLOv8 Powered", unsafe_allow_html=True)
st.markdown("</footer>", unsafe_allow_html=True)