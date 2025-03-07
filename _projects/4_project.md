---
layout: page
title: Multi-Camera Vision System for Automated Material Detection and Sorting
description: A real-time computer vision system for material recovery and worker safety monitoring on industrial conveyor belts
img: assets/img/project-4/heroo.png
importance: 4
category: work
related_publications: false
---

### 1. Overview

A comprehensive real-time computer vision system developed during my engagement at VEE ESS Engineering for enhancing high-value material recovery and worker safety monitoring on industrial conveyor belts. The system combines YOLOv5 for precise material detection and instance segmentation with intelligent background subtraction for motion analysis. The architecture features camera-specific region-of-interest (ROI) processing, worker-interaction filtering to minimize false positives, and a robust counting mechanism. The semi-automation of data collection using Mask R-CNN significantly reduced manual annotation efforts while maintaining high-quality dataset generation.

<div style="text-align: center;">
    <img src="/assets/img/project-4/sys_pipe.png" alt="System Pipeline" style="width: 100%; max-width: 1000px;">
    <p><em>End-to-end system architecture showing the complete pipeline from data collection to deployment</em></p>
</div>

---

### 2. System Architecture

#### 2.1 Data Acquisition and Processing Pipeline

The system architecture consists of several interconnected components that work together to create a robust, real-time material detection and sorting system:

- **Multi-Camera Input**: Multiple cameras monitor different sections of the conveyor belt, providing comprehensive coverage
- **Parallel Processing Pipelines**: Separate pipelines for material detection and worker safety monitoring
- **ROI-Based Processing**: Camera-specific regions of interest to focus computational resources on relevant areas
- **Real-Time Detection System**: YOLOv5-based detection models for materials and workers
- **False Positive Filtering**: Worker interaction detection to prevent miscounting

#### 2.2 Semi-Automated Data Annotation

Creating a robust dataset was one of the key challenges. A multi-stage approach was implemented:

- **Initial Dataset Creation**: A small initial dataset of approximately 800 images (200 per material class) was created using:
  - Manual annotation with LabelMe
  - Traditional computer vision techniques (Canny edge detection, adaptive thresholding)
  - Semi-automated annotation using bounding boxes from a pre-trained object detector

- **Mask R-CNN Fine-Tuning**: The initial dataset was used to fine-tune a pre-trained Mask R-CNN model, specifically adapting it to detect and segment the target materials

<div style="text-align: center;">
    <img src="/assets/img/project-4/labelme.png" alt="Manual Annotation" style="width: 60%; max-width: 700px;">
    <p><em>Initial dataset creation using manual annotation</em></p>
</div>
<div style="text-align: center;">
    <img src="/assets/img/project-4/maskrcnn.drawio.png" alt="Mask R-CNN Architecture" style="width: 100%; max-width: 700px;">
    <p><em>Simple Mask R-CNN Architecture Diagram</em></p>
</div>

#### 2.3 Automated Material Segmentation

Using the fine-tuned Mask R-CNN model, the system could automatically generate segmentation masks for a much larger dataset:

- **Automated Mask Generation**: The fine-tuned model processed video feeds through predefined ROIs
- **Segmented Instance Extraction**: Over 43,000 segmented material instances were automatically extracted
- **Dataset Expansion**: This approach dramatically improved data collection efficiency

<div style="text-align: center;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-4/trash1.png" alt="Segmented Material 1" style="width: 22%; max-width: 250px;">
        <img src="/assets/img/project-4/trash2.png" alt="Segmented Material 2" style="width: 22%; max-width: 250px;">
        <img src="/assets/img/project-4/trash3.png" alt="Segmented Material 3" style="width: 22%; max-width: 250px;">
        <img src="/assets/img/project-4/trash4.png" alt="Segmented Material 4" style="width: 22%; max-width: 250px;">
    </div>
    <p><em>Individual material instances segmented and extracted from the conveyor belt stream</em></p>
</div>

#### 2.4 Data Augmentation Strategy

To create a dataset that generalizes well to real-world conditions, a sophisticated data augmentation strategy was implemented:

- **Object-Background Compositing**: Segmented material images were overlaid onto images of empty conveyor belt bins
- **Multiple Transformations**: Each composite image underwent various transformations:
  - Rotations to simulate different orientations
  - Scaling to account for size variations
  - Brightness and contrast adjustments to handle lighting changes
- **Dual Dataset Creation**: Two distinct datasets were generated:
  - Detection dataset for YOLOv5 training
  - Segmentation dataset for instance segmentation

<div style="text-align: center;">
    <img src="/assets/img/project-4/get_coord_op.png" alt="ROI Definition" style="width: 60%; max-width: 550px;">
    <p><em>Defining the ROI for environmental (bin) extraction for data augmentationt</em></p>
</div>

<div style="text-align: center;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-4/aug1.jpg" alt="Augmented Data 1" style="width: 23%; max-width: 275px;">
        <img src="/assets/img/project-4/aug2.jpg" alt="Augmented Data 2" style="width: 23%; max-width: 275px;">
        <img src="/assets/img/project-4/aug3.jpg" alt="Augmented Data 3" style="width: 23%; max-width: 275px;">
        <img src="/assets/img/project-4/aug4.jpg" alt="Augmented Data 4" style="width: 23%; max-width: 275px;">
    </div>
    <p><em>Data augmentation showing segmented materials overlaid on bin backgrounds with various transformations</em></p>
</div>

---

### 3. Worker Safety Monitoring

#### 3.1 Worker Detection System

A separate but integrated worker detection system was implemented to ensure worker safety and prevent false positive material counts:

- **Color-Based Initial Detection**: A fast color-based detection method identified high-visibility safety vests
- **YOLOv5 Worker Detection**: A specialized YOLOv5 model was trained for robust worker detection
- **Bounding Box Generation**: Precise bounding boxes around workers enabled interaction detection

<div style="text-align: center;">
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-4/people_det.jpg" alt="Worker Detection 1" style="width: 45%; max-width: 400px;">
        <img src="/assets/img/project-4/people_det2.jpg" alt="Worker Detection 2" style="width: 45%; max-width: 400px;">
        <img src="/assets/img/project-4/people_det3.jpg" alt="Worker Detection 3" style="width: 45%; max-width: 400px;">
        <img src="/assets/img/project-4/person_det4.jpg" alt="Worker Detection 4" style="width: 45%; max-width: 400px;">
    </div>
    <p><em>Worker detection system identifying safety vest-wearing personnel with precise bounding boxes</em></p>
</div>

#### 3.2 False Positive Filtering

A critical innovation in this system was the ability to prevent false positive material counts when workers interact with the conveyor belt:

- **Worker Overlap Detection**: Materials detected within worker bounding boxes are not counted
- **Temporal Cooldown**: After counting an object, a cooldown period prevents immediate recounting
- **Background Subtraction**: MOG2 background subtractor further refines motion detection within ROIs

<div style="text-align: center;">
    <img src="/assets/img/project-4/false_positves.png" alt="False Positive Filtering" style="width: 100%; max-width: 700px;">
    <p><em>Examples of false positives detected and filtered by the system</em></p>
</div>

---

### 4. Real-Time Processing

The real-time processing system integrates all components to deliver accurate material detection and counting while ensuring worker safety:

- **Frame Acquisition**: Continuous frame capture from multiple camera feeds
- **ROI Processing**: Camera-specific regions are processed separately
- **Background Subtraction**: Intelligent background subtraction identifies moving objects
- **False Positive Filtering**: Worker overlap detection prevents miscounting
- **Material Classification**: Detected materials are classified and counted

<div style="text-align: center;">
    <img src="/assets/img/project-4/trash_mask.gif" alt="Real-time Material Detection" style="width: 60%; max-width: 700px;">
    <p><em>Real-time segmentation of materials on the conveyor belt using ROI-based detection</em></p>
</div>

<div style="text-align: center;">
    <img src="/assets/img/project-4/data_aug_op.jpg" alt="Real-time Detection Results" style="width: 60%; max-width: 700px;">
    <p><em>Real-time detection results showing system performance with detected materials in the production environment</em></p>
</div>

---

### 5. Performance Metrics

The system achieved exceptional performance across various metrics:

<div style="text-align: center;">
  <table class="table" style="width: 90%; margin: 0 auto; border-collapse: collapse; border: 1px solid #ddd;">
    <thead>
      <tr style="background-color: #f2f2f2;">
        <th style="padding: 12px; border: 1px solid #ddd;">Model</th>
        <th style="padding: 12px; border: 1px solid #ddd;">mAP50</th>
        <th style="padding: 12px; border: 1px solid #ddd;">mAP50-95</th>
        <th style="padding: 12px; border: 1px solid #ddd;">Precision</th>
        <th style="padding: 12px; border: 1px solid #ddd;">Recall</th>
        <th style="padding: 12px; border: 1px solid #ddd;">F1 Score</th>
        <th style="padding: 12px; border: 1px solid #ddd;">IOU</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 12px; border: 1px solid #ddd;"><strong>Material Detection</strong></td>
        <td style="padding: 12px; border: 1px solid #ddd;">0.995</td>
        <td style="padding: 12px; border: 1px solid #ddd;">0.968</td>
        <td style="padding: 12px; border: 1px solid #ddd;">0.999</td>
        <td style="padding: 12px; border: 1px solid #ddd;">0.999</td>
        <td style="padding: 12px; border: 1px solid #ddd;">0.999</td>
        <td style="padding: 12px; border: 1px solid #ddd;">~0.92</td>
      </tr>
      <tr style="background-color: #f9f9f9;">
        <td style="padding: 12px; border: 1px solid #ddd;"><strong>Worker Detection</strong></td>
        <td style="padding: 12px; border: 1px solid #ddd;">0.977</td>
        <td style="padding: 12px; border: 1px solid #ddd;">0.745</td>
        <td style="padding: 12px; border: 1px solid #ddd;">0.938</td>
        <td style="padding: 12px; border: 1px solid #ddd;">0.960</td>
        <td style="padding: 12px; border: 1px solid #ddd;">0.950</td>
        <td style="padding: 12px; border: 1px solid #ddd;">~0.86</td>
      </tr>
    </tbody>
  </table>
</div>

<p style="margin-top: 20px;">
<strong>System Performance</strong><br>
• Real-time processing: 30+ FPS<br>
• Inference latency: <15ms<br>
• False positive rate: <0.5%<br>
• Robust to lighting variations and occlusions
</p>

---

### 6. Key Contributions

My key contributions to this project at VEE ESS Engineering included:

- Pipeline architecture designed and implemented the end-to-end computer vision pipeline for automated material detection and sorting
- Semi-automated data collection created a semi-automated data collection and annotation system that reduced manual labeling effort by 90%
- Worker safety system developed the worker detection and false positive filtering system that increased counting accuracy by 35%
- ROI management designed the interactive ROI definition system that enabled flexible deployment across different conveyor configurations
- Model training trained and optimized the YOLOv5 models for both material detection and worker safety monitoring

---

### 7. Technologies & Skills Used

- **Languages**: Python, C++
- **Frameworks**: PyTorch, OpenCV, ROS (Noetic), NumPy, Pandas
- **Machine Learning**: YOLOv5, Mask R-CNN, Transfer Learning, Data Augmentation
- **Computer Vision**: Object Detection, Instance Segmentation, Background Subtraction, ROI Processing

---

### 8. Project Repository

[CVAnnotate](https://github.com/Srecharan/CVAnnotate.git)