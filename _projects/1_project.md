---
layout: page
title: Hybrid CV-ML Approach for Autonomous Leaf Grasping
description: A novel vision system combining geometric computer vision with deep learning for leaf detection and grasp point optimization
img: assets/img/project-1/hybrid_op.png
importance: 1
category: work
related_publications: false
---

### Overview

A real-time vision system for leaf manipulation combining geometric computer vision techniques with deep learning. This hybrid system integrates YOLOv8 for leaf segmentation, RAFT-Stereo for depth estimation, and a custom CNN (GraspPointCNN) for grasp point optimization. The architecture features self-supervised learning that eliminates manual annotation, and a confidence-weighted decision framework that dynamically balances traditional CV algorithms with CNN predictions to achieve superior grasping performance.

<div style="text-align: center;">
    <img src="/assets/img/project-1/REX.drawio_f.png" alt="System Architecture" style="width: 100%; max-width: 3000px;">
    <p><em>Multi-stage perception pipeline integrating traditional computer vision with deep learning</em></p>
</div>

### Multi-Stage Perception Pipeline

The system employs a three-stage perception pipeline:

1. **Instance Segmentation (YOLOv8)**: Fine-tuned on a custom dataset of ~900 images, achieving 68% mAP@[0.5:0.95] for leaf mask generation

<div style="text-align: center;">
    <img src="/assets/img/project-1/yolo_output.png" alt="YOLOv8 Segmentation Output" style="width: 100%; max-width: 800px;">
    <p><em>YOLOv8 instance segmentation results showing precise leaf mask generation with high confidence scores</em></p>
</div>

2. **Depth Estimation (RAFT-Stereo)**: High-precision depth maps with sub-pixel accuracy (<0.5px) from stereo pairs, enabling detailed 3D reconstruction

<div style="text-align: center; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
    <img src="/assets/img/project-1/rgb_input.png" alt="RGB Input" style="width: 30%; max-width: 300px;">
    <img src="/assets/img/project-1/depth0.png" alt="Depth Map" style="width: 30%; max-width: 300px;">
    <img src="/assets/img/project-1/plant_pcd-2x.gif" alt="Point Cloud" style="width: 30%; max-width: 300px;">
</div>
<div style="text-align: center;">
    <p><em>Stereo vision pipeline: RGB input (left), depth map visualization (center), and 3D point cloud reconstruction (right)</em></p>
</div>

3. **Hybrid Grasp Point Selection**: Combines geometric CV with machine learning refinement, which includes the traditional CV pipeline and ML enhancement described below.

#### Traditional Computer Vision Pipeline

The geometric CV component uses Pareto optimization for leaf selection based on:

- **Clutter Score (35%)**: Isolation from other leaves using Signed Distance Fields
- **Distance Score (35%)**: Proximity to camera with exponential falloff
- **Visibility Score (30%)**: Completeness of view and position in frame

Grasp point selection employs weighted scoring criteria:

- **Flatness Analysis (25%)**: Surface smoothness using depth gradients
- **Approach Vector Quality (40%)**: Optimal approach direction
- **Accessibility (15%)**: Position relative to camera
- **Edge Awareness (20%)**: Distance from leaf boundaries

<div style="text-align: center; display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="/assets/img/project-1/cv_op1.png" alt="CV Output 1" style="max-width: 400px;">
    <img src="/assets/img/project-1/cv_op2.png" alt="CV Output 2" style="max-width: 400px;">
</div>
<div style="text-align: center;">
    <p><em>Traditional CV pipeline output: Segmented leaf visualization with grasp point selection (left), and raw stereo camera image with detected leaf midrib (right)</em></p>
</div>

#### ML-Enhanced Decision Making

The machine learning component features a custom CNN architecture (GraspPointCNN) with:

- **Self-Supervised Learning**: CV pipeline acts as an expert teacher
- **Data Collection**: Automated generation of positive/negative samples
- **9-Channel Input Features**: Depth map, binary mask, and 7 score maps
- **Attention Mechanism**: Enables focus on most relevant patch regions

<div style="text-align: center;">
    <img src="/assets/img/project-1/CNN_grasp.drawio.png" alt="CNN Architecture" style="width: 50%; max-width: 400px;">
    <p><em>GraspPointCNN architecture: A 9-channel input feature map processed through three encoder blocks with an attention mechanism, followed by dense layers and global average pooling</em></p>
</div>

### Hybrid Decision Integration

The system implements a dynamic integration strategy:

- Traditional CV generates candidate grasp points
- ML model evaluates candidates with confidence scores
- Weighted average combines both approaches:
  ```python
  ml_conf = 1.0 - abs(ml_score - 0.5) * 2
  ml_weight = min(0.3, ml_conf * 0.6)
  final_score = (1 - ml_weight) * trad_score + ml_weight * ml_score
  ```
- ML influence varies (10-30%) based on prediction confidence
- Falls back to traditional CV (70-90%) for low-confidence predictions

<div style="text-align: center;">
    <img src="/assets/img/project-1/hybrid_op.png" alt="Hybrid Output" style="width: 100%; max-width: 800px;">
    <p><em>Hybrid CV-ML grasp point selection optimized for reliable leaf manipulation</em></p>
</div>

<div style="text-align: center;">
    <img src="/assets/img/project-1/rex_grasp_4x.gif" alt="System Operation" style="width: 100%; max-width: 800px;">
    <p><em>Complete pipeline in action: Once 3D coordinates are determined, the system executes precise leaf grasping</em></p>
</div>

### Results & Performance

#### Model Metrics

| Metric                | Value  | Description |
|----------------------|--------|-------------|
| Validation Accuracy  | 93.14% | Overall model accuracy |
| Positive Accuracy    | 97.09% | Accuracy for successful grasp points |
| Precision           | 92.59% | True positives / predicted positives |
| Recall              | 97.09% | True positives / actual positives |
| F1 Score            | 94.79% | Balanced measure of precision and recall |

#### System Performance (150 test cases)

| Metric                     | Classical CV | Hybrid (CV+ML) | Improvement |
|---------------------------|--------------|----------------|-------------|
| Accuracy (px)             | 25.3         | 27.1          | +1.8        |
| Feature Alignment (%)     | 80.67        | 83.33         | +2.66       |
| Edge Case Handling (%)    | 75.33        | 77.33         | +2.00       |
| Overall Success Rate (%)  | 78.00        | 82.66         | +4.66       |

### Contribution

I was responsible for the complete development of this computer vision system, including:

- Preparing and annotating the custom dataset for YOLOv8 instance segmentation
- Fine-tuning the RAFT-Stereo model for high-precision depth estimation
- Engineering the traditional computer vision pipeline with multiple scoring mechanisms 
- Designing and implementing the custom CNN architecture (GraspPointCNN)
- Creating the hybrid decision integration framework that balances traditional CV with ML refinement
- Testing and validating system performance through multiple experimental trials

This work was completed under the guidance of Prof. Abhishek Silval and Prof. George Cantor.

### Technologies Used

- **Languages**: Python, C++
- **Frameworks**: PyTorch, CUDA, OpenCV, Scikit-learn, Numpy, Pandas, Matplotlib
- **Computer Vision**: Instance Segmentation, Depth Estimation, Point Cloud Processing, SDF, 3D Perception
- **Deep Learning**: CNN Architecture Design, Self-Supervised Learning, Model Training & Optimization, Attention Mechanisms
- **Cloud Computing**: AWS EC2

### Resources

- [GitHub Repository](https://github.com/Srecharan/LeafGrasp-Vision-ML)
- [YOLOv8 Segmentation Node](https://github.com/Srecharan/YoloV8Seg-REX.git)
- [RAFT-Stereo Node](https://github.com/Srecharan/RAFTStereo-REX.git)