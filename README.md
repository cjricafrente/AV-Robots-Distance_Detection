# AV-Robots Distance Detection

A lightweight **monocular computer vision pipeline** for estimating object distance and supporting **stop/go and overtaking decisions** for indoor autonomous robots.

This project implements a real-time perception system using **YOLOv8n object detection**, **Ground-Plane Metricization (GPM)**, and **temporal filtering** to estimate the metric distance of obstacles from a **single camera**. The system is designed to run on **embedded hardware such as the Raspberry Pi 5** and integrate with a navigation stack using **ROS2 and ORB-SLAM3**.

---

# Project Overview

Modern robot navigation systems often rely on expensive sensors such as LiDAR or stereo cameras. This project demonstrates that **reliable short-range obstacle awareness can be achieved using only a monocular camera and lightweight algorithms**.

The perception pipeline detects objects, estimates their distance relative to the ground plane, and evaluates collision risk to produce interpretable navigation cues:

* **GO**
* **SLOW**
* **STOP**
* **OVERTAKE**

These outputs are published to a **navigation system (handled by a partner project)** which performs localization and path planning.

---

# System Architecture

The perception pipeline follows the workflow below:

```
Camera Input
      │
      ▼
YOLOv8n Object Detection
      │
      ▼
Footpoint Extraction
      │
      ▼
Ground-Plane Metricization (GPM)
      │
      ▼
Temporal Filtering
(Median + Exponential + Kalman)
      │
      ▼
Static / Dynamic Classification
      │
      ▼
Obstacle State & Risk Evaluator
      │
      ▼
Navigation Commands
(GO / SLOW / STOP / OVERTAKE)
```

---

# Key Features

### Monocular Distance Estimation

Computes real-world object distance using **a single camera** and ground-plane geometry.

### Lightweight Detection

Uses **YOLOv8n**, a compact real-time detector optimized for embedded systems.

### Ground-Plane Metricization (GPM)

Converts 2D object detections into **metric distances (meters)** using camera calibration and planar geometry.

### Temporal Filtering

Reduces noise and detection jitter using:

* Median filtering
* Exponential smoothing
* Kalman filtering

### Static / Dynamic Classification

Determines whether obstacles are:

* **STATIC** (walls, boxes, furniture)
* **DYNAMIC** (moving people or objects)

### Risk Evaluation

Computes **relative speed** and **Time-to-Collision (TTC)** to generate navigation signals.

---

# Hardware Setup

The prototype platform consists of a small robotic vehicle equipped with a monocular camera.

| Component       | Specification  |
| --------------- | -------------- |
| Platform        | 1:12 RC Car    |
| Camera          | USB Webcam     |
| Resolution      | 1280 × 720     |
| Frame Rate      | 30 FPS         |
| Camera Height   | ~23 cm         |
| Camera Tilt     | ~12° downward  |
| Processing Unit | Raspberry Pi 5 |

The system was tested in an **indoor hallway-style diorama environment** with static and moving obstacles.

---

# Software Stack

* **Python**
* **YOLOv8 (Ultralytics)**
* **OpenCV**
* **NumPy**
* **ROS2 (for communication with navigation stack)**

Optional components used during development:

* Label Studio (annotation)
* PyTorch
* Matplotlib

---

# Installation

### 1. Clone the Repository

```bash
git clone https://github.com/cjricafrente/AV-Robots-Distance_Detection.git
cd AV-Robots-Distance_Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If YOLOv8 is not included in requirements:

```bash
pip install ultralytics
```

---

# Running the System

Run the main detection pipeline:

```bash
python main.py
```

The system will:

1. Capture frames from the camera
2. Detect objects using YOLOv8n
3. Estimate object distances
4. Apply temporal filtering
5. Classify obstacle state
6. Output navigation recommendations

Example output:

```
Obstacle: Box
Distance: 0.72 m
Relative Speed: -0.14 m/s
TTC: 5.1 s
Decision: SLOW
```

---

# Calibration

Camera calibration is performed using a **ChArUco board**.

Calibration estimates:

* Camera intrinsic matrix
* Distortion coefficients
* Camera pose relative to the ground plane

This information is used by the **Ground-Plane Metricization module** to convert pixel coordinates into metric distances.

---

# Dataset Preparation

Training data is generated from recorded runs inside the diorama environment.

Steps:

1. Record videos using the RC platform
2. Extract frames
3. Annotate obstacles using Label Studio
4. Train YOLOv8n on the custom dataset

Example training command:

```bash
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```

---

# Example Output

The system overlays detection and distance estimation directly on the camera feed:

```
[Person] 0.83m
State: Dynamic
Risk: SLOW
```

Additional information displayed:

* Bounding boxes
* Estimated distance
* Relative velocity
* Navigation decision

---

# Performance Goals

Target performance on Raspberry Pi 5:

| Metric              | Target   |
| ------------------- | -------- |
| Frame Rate          | ~10 FPS  |
| Mean Distance Error | ≈ 0.1 m  |
| Input Resolution    | 720p     |
| Processing          | CPU-only |

---

# Research Context

This repository is part of the thesis:

**"Robot Navigation with ORB-SLAM3 and Monocular Object Distance Estimation for Stop/Go and Overtaking Decisions"** 

The perception module developed here is designed to operate collaboratively with a **separate localization and navigation system** that uses:

* ORB-SLAM3 for localization
* A* path planning
* ROS2 communication

---

# Future Improvements

Planned enhancements include:

* Real-time **top-down obstacle map**
* **Multi-object tracking improvements**
* Integration with **hardware motor control**
* Optimization using **TensorRT or OpenVINO**
* Extended testing in **larger indoor environments**

---

# Contributors

* John Paul B. Baroña
* Andrei Miguel A. David
* Paul Christian M. Mandap
* Carl Joshua D. Ricafrente

Technological Institute of the Philippines – Manila

---

# License

This project is intended for **academic and research purposes**.

---

# Acknowledgements

Special thanks to the advisers and panelists who supported the development of this thesis project and the research on lightweight monocular perception systems.
