---
layout: page
title: Real-time Hand Gesture Recognition for AR Interaction
description: A sophisticated system combining computer vision and deep learning for hand tracking and gesture recognition in augmented reality
img: assets/img/project-3/hero_image.png
importance: 4
category: work
related_publications: false
---

### 1. Overview

A sophisticated real-time hand gesture recognition system developed during my internship at Hanon Systems, implementing a hybrid architecture that combines classical computer vision with deep learning approaches. The system leverages depth sensing camera's capabilities enhanced by Extended Kalman filtering for precise 3D tracking, while incorporating both MediaPipe-based gesture recognition and optimized ONNX neural network implementations for robust hand detection and pose estimation.

<div style="text-align: center;">
    <img src="/assets/img/project-3/virtuhand_sys.png" alt="System Architecture" style="width: 3000%; max-width: 800px;">
    <p><em> System pipeline showing data flow from camera through Python backend to Unity frontend</em></p>
</div>

<div style="text-align: center;">
    <a href="https://youtu.be/eRFWZjJbcgI" target="_blank">
        <img src="/assets/img/project-3/full_demo_gesture_fast.gif" alt="Full Demo" style="width: 100%; max-width: 700px;">
        <p><em>📌 For full quality, watch the video on <a href="https://youtu.be/eRFWZjJbcgI" target="_blank">YouTube</a></em></p>
    </a>
</div>

---

### 2. Hand Detection and Tracking 

#### 2.1 MediaPipe Landmark Detection

The system uses Google's MediaPipe Hands library as the foundation for initial hand detection and landmark extraction:

- Extracts 21 keypoints from each hand in real-time
- Provides a skeletal representation of hand pose
- Computationally efficient with high accuracy
- Enables detection of both left and right hands independently

<div style="text-align: center;">
    <img src="/assets/img/project-3/mediapipe_landmarks.png" alt="MediaPipe Landmarks" style="width: 100%; max-width: 800px;">
    <p><em>Hand detection with MediaPipe showing 21 landmark points and their connections</em></p>
</div>

#### 2.2 3D Tracking with Extended Kalman Filter

To achieve precise and stable 3D tracking, the system combines MediaPipe landmarks with depth data and applies Extended Kalman Filtering:

- **Multi-stage Depth Filtering Pipeline:** 
  - Spatial filtering to reduce noise
  - Temporal filtering for consistency
  - Kalman filtering for smooth tracking
  
- **Extended Kalman Filter Implementation:**
  - 2D state vector (position, velocity) estimation
  - Optimized noise matrices for hand motion
  - 30Hz update rate with dynamic time-step handling
  - Advanced outlier rejection for robust tracking

<div style="text-align: center;">
    <img src="/assets/img/project-3/depth_filter.png" alt="Depth Filtering" style="width: 30%; max-width: 800px;">
    <p><em>Visualization of the multi-stage depth filtering process showing raw depth data being transformed into smooth 3D positions</em></p>
</div>

<div style="text-align: center;">
    <img src="/assets/img/project-3/EFK_depth.drawio.png" alt="EKF Flowchart" style="width: 100%; max-width: 800px;">
    <p><em>Extended Kalman Filter implementation flowchart showing the prediction-correction cycle</em></p>
</div>

#### 2.3 ONNX Neural Network Integration

The system incorporates an optimized ONNX-based pipeline to improve performance:

- **Two-stage Detection System:**
  - Palm Detection (192x192 input)
  - Hand Landmark Detection (224x224 input)

- **Performance Optimizations:**
  - FP16 quantization reducing model size by 50%
  - Unity Barracuda engine for GPU acceleration
  - Custom tensor preprocessing pipeline
  - Overall inference time <33ms (30+ FPS)

<div style="text-align: center;">
    <img src="/assets/img/project-3/ONNX.png" alt="ONNX Pipeline" style="width: 30%; max-width: 350px;">
    <p><em>ONNX neural network pipeline with parallel palm detection and hand landmark models</em></p>
</div>

---

### 3. Gesture Recognition System

#### 3.1 Static Gesture Recognition

The static gesture recognition system employs geometric analysis of hand landmarks:

- **Joint Angle Calculation:** Analysis of angles between finger joints
- **Finger State Detection:** Adaptive thresholds for open/closed/bent states
- **Palm Orientation Analysis:** Using normal vectors to determine hand orientation
- **Real-time Confidence Scoring:** Certainty evaluation for each classification

<div style="text-align: center;">
    <img src="/assets/img/project-3/static_gestures.gif" alt="Static Gestures" style="width: 100%; max-width: 700px;">
    <p><em>Demonstration of supported static gestures: GRAB, OPEN_PALM, PINCH, and POINT</em></p>
</div>

#### 3.2 Dynamic Gesture Recognition

For dynamic gestures, the system uses a Gated Recurrent Unit (GRU) neural network combined with custom motion pattern analysis:

- **GRU Neural Network:**
  - Input: Sequence of 30 frames (63 features per frame)
  - Architecture: 2 hidden layers with 32 units each
  - Output: Classification with confidence scores for each dynamic gesture

- **Motion Pattern Analysis:**
  - Velocity component extraction (dx, dy)
  - Horizontal/vertical motion ratio analysis
  - Specialized pattern detectors for SWIPE, CIRCLE, and WAVE gestures

<div style="text-align: center;">
    <img src="/assets/img/project-3/GRU.png" alt="GRU Architecture" style="width: 30%; max-width: 300px;">
    <p><em>GRU-based dynamic gesture recognition pipeline with sequence preprocessing and temporal smoothing</em></p>
</div>

<div style="text-align: center;">
    <img src="/assets/img/project-3/dynamic_gestures.gif" alt="Dynamic Gestures" style="width: 100%; max-width: 700px;">
    <p><em>Demonstration of supported dynamic gestures: SWIPE_LEFT, SWIPE_RIGHT, and CIRCLE</em></p>
</div>

---

### 4. Unity Integration and AR Interaction

#### 4.1 Real-time Hand Rigging

The Unity frontend provides visualization and interaction capabilities:

- **Hand Model:** Fully articulated 3D hand with 21 joints
- **Inverse Kinematics:** Realistic hand movement based on tracking data
- **Real-time Physics:** Dynamic object interaction with collision detection
- **WebSocket Communication:** Low-latency data streaming between Python backend and Unity frontend

<div style="text-align: center;">
    <img src="/assets/img/project-3/hand_rig_2x.gif" alt="Hand Rigging" style="width: 100%; max-width: 700px;">
    <p><em>Real-time hand rigging in Unity. The 3D hand model accurately mirrors the user's hand movements and gestures</em></p>
</div>

#### 4.2 AR Interaction Demo

The system was integrated into an AR environment for demonstration purposes:

- **Intuitive Interactions:** Users can grab, move, and place virtual objects
- **Gesture-Triggered Events:** Different gestures trigger different interactions
- **Real-time Response:** The system maintains 30+ FPS for seamless user experience

<p style="font-style: italic; font-weight: underline;">
   <span style="font-weight: bold; font-style: italic;">Note:</span> The virtual flower arrangement scene shown in the demo was created solely for demonstration purposes. During my internship at Hanon Systems, the actual implementation was focused on enabling automotive technicians to practice precise component placement and assembly procedures for virtual HVAC systems in an automotive manufacturing context.
</p>



---

### 5. Performance Metrics

<div class="table-responsive">
  <table class="table">
    <thead>
      <tr>
        <th>Component</th>
        <th>Metric</th>
        <th>Value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Hand Tracking</td>
        <td>Tracking Precision</td>
        <td>&lt;7.5mm error</td>
      </tr>
      <tr>
        <td>Static Gestures</td>
        <td>Recognition Accuracy</td>
        <td>97%</td>
      </tr>
      <tr>
        <td>Dynamic Gestures</td>
        <td>Recognition Latency</td>
        <td>&lt;33ms</td>
      </tr>
      <tr>
        <td>ONNX Models</td>
        <td>Palm Detection Inference</td>
        <td>8-10ms</td>
      </tr>
      <tr>
        <td>ONNX Models</td>
        <td>Landmark Detection Inference</td>
        <td>12-15ms</td>
      </tr>
      <tr>
        <td>Overall System</td>
        <td>Frame Rate</td>
        <td>30+ FPS</td>
      </tr>
    </tbody>
  </table>
</div>

---

### 6. Contribution

During my internship at Hanon Systems, I contributed to the HVAC Systems Simulation Team as a Machine Learning Engineer Intern by architecting and implementing a real-time 3D hand tracking and gesture recognition system for augmented reality (AR) applications within Unity. This system enabled over 50 automotive technicians to practice component placement in virtual HVAC systems.

My key contributions included:

- Integrating a depth-sensing camera for capturing 3D data and utilizing MediaPipe and an Extended Kalman Filter for robust 3D hand tracking, achieving less than 7.5mm ground truth tracking accuracy
- Designing and implementing both static gesture recognition (using geometric analysis) and dynamic gesture recognition (using a custom-trained GRU network and motion analysis), achieving 97% accuracy for static gestures and under 30ms latency for dynamic gestures
- Optimizing the pipeline with ONNX, achieving 33% faster inference with 50% smaller model size than MediaPipe
- Establishing real-time communication between a Python backend and the Unity AR frontend using WebSockets, enabling a 30Hz data streaming rate with packets under 1KB
- Implementing the 3D hand model rigging and inverse kinematics for realistic hand movement in the Unity environment

The demonstration scene shown in this portfolio was created after the internship to showcase the capabilities of the system, while the actual implementation at Hanon Systems was focused on HVAC component visualization and interaction.

---

### 7. Skills & Technologies Used

- **Languages & Frameworks**: Python, C++, PyTorch, ONNX, MediaPipe, WebSockets, OpenCV, NumPy
- **Machine Learning**: 3D Tracking, Landmark Detection, Depth Sensing, Geometric Analysis, Kalman Filtering
- **Deep Learning**: Recurrent Neural Networks (GRU), Model Optimization (ONNX), GPU Inference (Unity Barracuda)
- **Computer Vision**: Landmark Detection, Depth Filtering, Motion Analysis, Real-time Processing
- **Augmented Reality**: Unity Development, 3D Interaction Design, Virtual Object Manipulation

---

### 8. Project Repository

- [VirtuHand](https://github.com/Srecharan/VirtuHand.git): Main project repository