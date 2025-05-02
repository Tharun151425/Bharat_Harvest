# 🥕 Vegetable Detection App

A streamlined Streamlit web application for detecting vegetables using YOLOv8 with both image upload and webcam capabilities.

## Features

- **Drag and Drop Image Upload**: Easily upload images for vegetable detection
- **Webcam Integration**: Use your camera for real-time vegetable detection
- **Elegant UI**: Clean interface with green, white, and black theme
- **Responsive Design**: Works on desktop and mobile devices
- **Detailed Results**: View detection confidence scores and bounding boxes
- **Download Results**: Save annotated images with detections
- **Dark/Light Mode Support**: Automatic theme switching based on system preferences

## Vegetables Detected

The model can detect 10 different types of vegetables:
- Beans
- Brinjal (Eggplant)
- Cabbage
- Capsicum (Bell Pepper)
- Carrot
- Cauliflower
- Chilli
- Onion
- Potato
- Tomato

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/vegetable-detection-app.git
cd vegetable-detection-app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLOv8 model file (best.pt) and place it in the project root directory.

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Use the app by either:
   - Uploading an image through the drag and drop interface
   - Using your webcam for real-time detection

## Model Information

This application uses a YOLOv8 model trained on a custom dataset from Roboflow:
- **Dataset**: Mini Project Vegetable Detection
- **Version**: 4
- **License**: CC BY 4.0
- **URL**: https://universe.roboflow.com/srihitha-4xrye/mini-project-qd9pd/dataset/4

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- PIL

## License

This project is licensed under the MIT License.