# 🧠 Real-Time Object Detection System using OpenCV, YOLOv4 & MobileNet-SSD

A complete computer vision project that performs **real-time object detection** in **images, videos, and webcam streams**.  
Built using **OpenCV’s DNN module**, **YOLOv4**, and **MobileNet-SSD**, with a clean and easy-to-use **Tkinter GUI** for model selection and file browsing.

---

## 🚀 Features

- 🖼 **Image detection**
- 🎞 **Video file detection**
- 🎥 **Real-time webcam detection**
- 🔁 **Model switching:** YOLOv4 ↔ MobileNet-SSD  
- 📦 **Tkinter GUI** (no command-line required)
- 🟩 Non-Max Suppression (NMS)
- 🎯 Multi-class object detection
- ⚡ Optimized resizing for proper visualization
- 🔧 Threaded detection — GUI never freezes

---

## 📁 Project Structure

ObjectDetectionProject/
│
├── yolo/
│ ├── yolov4.weights
│ ├── yolov4.cfg
│ └── coco.names
│
├── mobilenet/
│ ├── MobileNetSSD_deploy.prototxt
│ └── MobileNetSSD_deploy.caffemodel
│
├── object_detection_gui.py # Tkinter GUI App
└── object_detection_combined.py


---

## 🔧 Installation

### 1️⃣ Install Dependencies

pip install opencv-python numpy


📥 Download Required Model Files
YOLOv4 (place inside yolo/)

yolov4.cfg

yolov4.weights

coco.names

MobileNet-SSD (place inside mobilenet/)

MobileNetSSD_deploy.prototxt

MobileNetSSD_deploy.caffemodel


▶️ Run the Application
Start GUI
python object_detection_gui.py


GUI Options:

Detect Image

Detect Video

Open Webcam

Model Selection: YOLOv4 or MobileNet-SSD

🧪 Supported Models
YOLOv4

High detection accuracy

Supports 80 COCO classes

Good for real-time webcam detection

MobileNet-SSD

Lightweight

Fast on CPU

Supports 20 classes

📊 Technical Highlights

Preprocessing using cv2.dnn.blobFromImage

YOLOv4 + SSD integration in same GUI

Multi-threaded detection loop

Non-Max Suppression (NMS)

Optimized resizing (1280×720)

Handles Windows path issues

Clean, readable Python code structure

🧩 Skills Demonstrated

Python

OpenCV (Deep Neural Network module)

YOLOv4 & MobileNet-SSD

Tkinter GUI development

Multithreading

Computer Vision

Real-time processing

📄 License

This project is open-source. Feel free to modify and extend it.
