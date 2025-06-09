---
layout: page
title: Hybrid CV-ML Approach for Autonomous Leaf Grasping
description:  A novel vision system combining geometric computer vision with deep learning for leaf detection and grasp point optimization
img: assets/img/project-1/hero.png
importance: 1
category: work
related_publications: false
---

### 1. Overview

A real-time vision system for leaf manipulation combining geometric computer vision techniques with deep learning. This hybrid system integrates YOLOv8 for leaf segmentation, RAFT-Stereo for depth estimation, and a custom CNN (GraspPointCNN) for grasp point optimization. The architecture features **self-supervised learning that eliminates 100% manual annotation requirements**, and a confidence-weighted decision framework that dynamically balances traditional CV algorithms with CNN predictions to achieve superior grasping performance. The system includes production-ready optimizations through **custom CUDA kernels**, **TensorRT acceleration**, and **Docker containerization** for real-world deployment.

<div style="text-align: center;">
    <img src="/assets/img/project-1/REX.drawio_f.png" alt="System Architecture" style="width: 100%; max-width: 3000px;">
    <p><em>Multi-stage perception pipeline for leaf manipulation with production optimizations</em></p>
</div>

---

### 2. Multi-Stage Perception Pipeline

The system employs a three-stage perception pipeline with advanced optimization techniques:

#### 2.1 Instance Segmentation (YOLOv8)
Fine-tuned on a custom dataset of ~900 images, achieving 68% mAP@[0.5:0.95] for leaf mask generation with **TensorRT optimization** for real-time performance

<div style="text-align: center;">
    <img src="/assets/img/project-1/yolo_output.png" alt="YOLOv8 Segmentation Output" style="width: 100%; max-width: 800px;">
    <p><em>YOLOv8 instance segmentation results showing precise leaf mask generation with high confidence scores</em></p>
</div>

#### 2.2 Depth Estimation (RAFT-Stereo)
High-precision depth maps with sub-pixel accuracy (<0.5px) from stereo pairs, enhanced with **custom CUDA kernels** for accelerated point cloud generation

<div style="text-align: center; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
    <img src="/assets/img/project-1/rgb_input.png" alt="RGB Input" style="width: 30%; max-width: 300px;">
    <img src="/assets/img/project-1/depth0.png" alt="Depth Map" style="width: 30%; max-width: 300px;">
    <img src="/assets/img/project-1/plant_pcd-2x.gif" alt="Point Cloud" style="width: 30%; max-width: 300px;">
</div>
<div style="text-align: center;">
    <p><em>Stereo vision pipeline: RGB input (left), depth map visualization (center), and 3D point cloud reconstruction (right)</em></p>
</div>

#### 2.3 Hybrid Grasp Point Selection
Combines geometric CV with machine learning refinement, which includes the traditional CV pipeline and ML enhancement described below.

##### 2.3.1 Traditional Computer Vision Pipeline

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
    <p><em>Traditional CV pipeline output: Raw stereo camera image with detected leaf midrib (left); Segmented leaf visualization with grasp point selection (right)</em></p>
</div>

##### 2.3.2 ML-Enhanced Decision Making with MLflow Tracking

The machine learning component features a custom CNN architecture (GraspPointCNN) with **comprehensive experiment tracking**:

- **Self-Supervised Learning**: CV pipeline acts as an expert teacher, eliminating 100% manual annotation requirements
- **MLflow Experiment Management**: Systematic tracking of 60+ model experiments across different attention mechanisms and architectural variants
- **Data Collection**: Automated generation of positive/negative samples with 10x acceleration over manual methods
- **9-Channel Input Features**: Depth map, binary mask, and 7 score maps
- **Attention Mechanism**: Enables focus on most relevant patch regions

**MLflow Integration for Systematic Optimization:**
```python
# Experiment tracking across model configurations
mlflow.log_params({
    "attention_type": "spatial",
    "architecture": "standard", 
    "learning_rate": 0.0005,
    "batch_size": 16
})

mlflow.log_metrics({
    "train_loss": loss.item(),
    "val_accuracy": accuracy,
    "recall": 97.09,
    "precision": 92.59
})
```

<div style="text-align: center;">
    <img src="/assets/img/project-1/CNN_grasp.drawio.png" alt="CNN Architecture" style="width: 50%; max-width: 400px;">
    <p><em>GraspPointCNN architecture with attention mechanism</em></p>
</div>

##### 2.3.3 Hybrid Decision Integration

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
    <p><em>Hybrid CV-ML grasp point selection: Left - Original camera view with leaf midrib ; Right - Segmented leaves with grasp point visualization
</em></p>
</div>
<p style="font-style: italic; font-weight: underline;">
   <span style="font-weight: bold; font-style: italic;">Note:</span> Grasping at the leaf tip often fails as the REX robot struggles to secure it, leading to missed grasps or leaf displacement. The hybrid grasp point selection method outperforms traditional CV, achieving a 4.66% improvement over 150 test cases.
</p>

---

### 3. Production Optimization and Deployment

#### 3.1 Custom CUDA Kernel Development
**High-Performance Point Cloud Generation**

Developed custom CUDA kernels to address CPU bottlenecks in RAFT-Stereo post-processing:

- **Problem**: Sequential processing of 1.5M pixels for 3D coordinate transformation
- **Solution**: Parallelized GPU computation with one thread per pixel
- **Performance**: 5x speedup (150ms → 30ms) for real-time operation
- **Implementation**: Memory-coalesced access patterns and optimized thread organization

#### 3.2 TensorRT Inference Optimization
**System-Wide Model Acceleration**

Compiled all vision models into TensorRT engines for production performance:

- **Models Optimized**: YOLOv8, RAFT-Stereo, and GraspPointCNN
- **Techniques**: FP16 precision, operator fusion, graph optimization
- **Performance**: 35% throughput improvement (20 → 27 FPS)
- **Benefits**: Maintained accuracy while achieving real-time processing

#### 3.3 Docker Containerization and Deployment
**Production-Ready System Deployment**

Containerized the complete vision pipeline for robust deployment:

- **Dependency Management**: PyTorch, CUDA, ROS, OpenCV, and custom libraries
- **Container Features**: Multi-stage builds, GPU acceleration, volume mounting
- **Deployment Results**: 82% grasp success rate in field trials across 150+ test cases
- **Production Benefits**: Environment reproducibility, easy deployment, version control

<div style="text-align: center;">
    <img src="/assets/img/project-1/rex_grasp_4x.gif" alt="System Operation" style="width: 30%; max-width: 240px;">
    <p><em>Complete optimized pipeline in action: Real-time leaf grasping with production-ready performance</em></p>
</div>

---

### 4. Results & Performance

#### Model Metrics

<div class="table-responsive">
  <table class="table">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Value</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Validation Accuracy</td>
        <td>93.14%</td>
        <td>Overall model accuracy</td>
      </tr>
      <tr>
        <td>Positive Accuracy</td>
        <td>97.09%</td>
        <td>Accuracy for successful grasp points</td>
      </tr>
      <tr>
        <td>Precision</td>
        <td>92.59%</td>
        <td>True positives / predicted positives</td>
      </tr>
      <tr>
        <td>Recall</td>
        <td>97.09%</td>
        <td>True positives / actual positives</td>
      </tr>
      <tr>
        <td>F1 Score</td>
        <td>94.79%</td>
        <td>Balanced measure of precision and recall</td>
      </tr>
    </tbody>
  </table>
</div>

#### System Performance (150 test cases)

<div class="table-responsive">
  <table class="table">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Classical CV</th>
        <th>Hybrid (CV+ML)</th>
        <th>Improvement</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Accuracy (px)</td>
        <td>25.3</td>
        <td>27.1</td>
        <td>+1.8</td>
      </tr>
      <tr>
        <td>Feature Alignment (%)</td>
        <td>80.67</td>
        <td>83.33</td>
        <td>+2.66</td>
      </tr>
      <tr>
        <td>Edge Case Handling (%)</td>
        <td>75.33</td>
        <td>77.33</td>
        <td>+2.00</td>
      </tr>
      <tr>
        <td>Overall Success Rate (%)</td>
        <td>78.00</td>
        <td>82.66</td>
        <td>+4.66</td>
      </tr>
    </tbody>
  </table>
</div>

#### Production Performance Metrics

<div class="table-responsive">
  <table class="table">
    <thead>
      <tr>
        <th>Component</th>
        <th>Baseline</th>
        <th>Optimized</th>
        <th>Improvement</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Point Cloud Generation</td>
        <td>150ms</td>
        <td>30ms</td>
        <td>5x speedup</td>
      </tr>
      <tr>
        <td>Inference Throughput</td>
        <td>20 FPS</td>
        <td>27 FPS</td>
        <td>35% improvement</td>
      </tr>
      <tr>
        <td>Dataset Creation</td>
        <td>Manual annotation</td>
        <td>Self-supervised</td>
        <td>100% elimination</td>
      </tr>
      <tr>
        <td>Field Trial Success</td>
        <td>-</td>
        <td>82%</td>
        <td>Production deployment</td>
      </tr>
    </tbody>
  </table>
</div>

---

### 5. Contribution

I was responsible for the complete system development including:

- Preparation and annotation of custom dataset for YOLOv8 instance segmentation, along with training and validating the model
- Fine-tuning the RAFT-Stereo model for high-precision depth estimation and 3D reconstruction
- Engineered the traditional computer vision pipeline with multiple scoring mechanisms, optimal leaf selection, and grasping point detection
- Designed and implemented the custom CNN architecture (GraspPointCNN)
- Created the hybrid decision integration framework that balances traditional CV with ML refinement
- Testing and validating system performance through multiple experimental trials

This research is carried out under the guidance of Prof. Abhisesh Silwal and Prof. George A. Kantor.


---

### 6. Skills and Technologies Used

- **Languages**: Python, C++, CUDA
- **Frameworks**: PyTorch, CUDA, OpenCV, Scikit-learn, Numpy, Pandas, Matplotlib, ROS2
- **Computer Vision**: Instance Segmentation, Depth Estimation, Point Cloud Processing, SDF, 3D Perception
- **Deep Learning**: CNN Architecture Design, Self-Supervised Learning, Model Training & Optimization, Attention Mechanisms
- **MLOps**: MLflow (Experiment Tracking), Model Versioning, Hyperparameter Optimization
- **Performance Optimization**: Custom CUDA Kernels, TensorRT (FP16 Precision, Operator Fusion), GPU Acceleration
- **Cloud Computing**: AWS EC2, Distributed Training
- **DevOps**: Docker (Containerization), Production Deployment, Environment Management

---

### 7. Project Repositories

- [LeafGrasp-Vision-ML](https://github.com/Srecharan/Leaf-Grasping-Vision-ML.git): Main Project Repository with Hybrid CV-ML System
- [YOLOv8 Segmentation Node](https://github.com/Srecharan/YoloV8Seg-REX.git): Real-time Leaf Instance Segmentation
- [RAFT-Stereo Node](https://github.com/Srecharan/RAFTStereo-REX.git): High-Precision Depth Estimation with CUDA Optimization
- [REX-Robot](https://github.com/Srecharan/REX-Robot.git): 6-DOF Gantry Robot Control System

---

### 8. References
[1] Srecharan Selvam, Abhisesh Silwal, George Kantor ”Self-Supervised Learning for Robotic Leaf Manipulation: A Hybrid Geometric-Neural Approach”, https://arxiv.org/pdf/2505.0370, Under review at ICCV 2025.

[2] Silwal, A., Zhang, X. M., Hadlock, T., Neice, J., Haque, S., Kaundanya, A., Lu, C., Vinatzer, B. A., Kantor, G., & Li, S. (2024). Towards an AI-Driven Cyber-Physical System for Closed-Loop Control of Plant Diseases. *Proceedings of the AAAI Symposium Series*, *4*(1), 432-435. https://doi.org/10.1609/aaaiss.v4i1.31828