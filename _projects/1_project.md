---
layout: page
title: Vision-Language-Action Enhanced Robotic Leaf Grasping: A Hybrid Foundation Model Approach
description: A novel vision system combining geometric computer vision, deep learning, and Vision-Language-Action models for intelligent leaf manipulation
img: assets/img/project-1/hero.png
importance: 1
category: work
related_publications: false
---

### 1. Overview

A real-time vision system for leaf manipulation that combines geometric computer vision with deep learning and **Vision-Language-Action (VLA) models**. This hybrid system integrates YOLOv8 for segmentation, RAFT-Stereo for depth estimation, and a custom CNN enhanced with **LLaVA-1.6-Mistral-7B for intelligent grasp reasoning**. The architecture features **self-supervised learning eliminating 100% manual annotation**, **LoRA fine-tuning achieving 88% validation accuracy**, and a confidence-weighted framework dynamically balancing traditional CV, ML, and VLA predictions. Production optimizations include **custom CUDA kernels**, **TensorRT acceleration**, and **Docker containerization** with **AWS GPU training infrastructure**.

<div style="text-align: center;">
    <img src="/assets/img/project-1/REX.drawiof.png" alt="System Architecture" style="width: 100%; max-width: 3000px;">
    <p><em>Multi-stage perception pipeline enhanced with Vision-Language-Action integration</em></p>
</div>

---

### 2. Vision-Language-Action (VLA) System

#### 2.1 LLaVA Integration and Fine-tuning
**Foundation Model Enhancement for Grasp Reasoning**

Integrated LLaVA-1.6-Mistral-7B foundation model with parameter-efficient LoRA fine-tuning:

- **Base Model**: LLaVA-1.6-Mistral-7B (CLIP + Vicuna) for vision-language understanding
- **Fine-tuning**: LoRA adaptation (rank=8, alpha=32) for leaf grasping tasks
- **Training Infrastructure**: AWS GPU acceleration with MLflow experiment tracking
- **Performance**: 88.0% validation accuracy through systematic hyperparameter optimization
- **Experiments**: 4 systematic configurations (baseline_5e5, higher_lr_1e4, larger_rank_16, optimized_config)

#### 2.2 Hybrid CV-VLA Decision Framework
**Dynamic Confidence-Based Integration**

The system implements intelligent fusion between traditional CV, ML, and VLA predictions:

```python
# Confidence-based weighting strategy
if vla_confidence > 0.8:
    weights = (0.4, 0.3, 0.3)  # CV, ML, VLA
elif vla_confidence > 0.5:
    weights = (0.7, 0.2, 0.1)  # Conservative VLA influence
else:
    weights = (0.9, 0.1, 0.0)  # Pure CV fallback
```

- **High VLA Confidence**: Balanced three-way integration
- **Medium Confidence**: CV-dominant with VLA assistance  
- **Low Confidence**: Traditional CV fallback for reliability
- **Adaptive Learning**: VLA influence grows with operational experience

---

### 3. Multi-Stage Perception Pipeline

#### 3.1 Instance Segmentation (YOLOv8)
Fine-tuned on ~900 images achieving 68% mAP@[0.5:0.95] with TensorRT optimization for real-time performance.

#### 3.2 Depth Estimation (RAFT-Stereo)
High-precision depth maps with sub-pixel accuracy (<0.5px) enhanced with custom CUDA kernels for accelerated point cloud generation.

<div style="text-align: center; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
    <img src="/assets/img/project-1/rgb_input.png" alt="RGB Input" style="width: 30%; max-width: 300px;">
    <img src="/assets/img/project-1/depth0.png" alt="Depth Map" style="width: 30%; max-width: 300px;">
    <img src="/assets/img/project-1/plant_pcd-2x.gif" alt="Point Cloud" style="width: 30%; max-width: 300px;">
</div>
<div style="text-align: center;">
    <p><em>Stereo vision pipeline: RGB input → depth estimation → 3D reconstruction</em></p>
</div>

#### 3.3 Hybrid Grasp Point Selection

##### Traditional CV Pipeline
Pareto optimization for leaf selection with geometric scoring:
- **Clutter Score (35%)**: Isolation using Signed Distance Fields
- **Distance Score (35%)**: Camera proximity with exponential falloff  
- **Visibility Score (30%)**: Frame position and completeness

Grasp point selection criteria:
- **Flatness Analysis (25%)**: Surface smoothness via depth gradients
- **Approach Vector Quality (40%)**: Optimal robot orientation
- **Accessibility (15%)**: Camera-relative positioning
- **Edge Awareness (20%)**: Boundary distance analysis

##### ML Enhancement with MLflow Tracking
Custom GraspPointCNN with comprehensive experiment management:
- **Self-Supervised Learning**: CV-generated training data (100% annotation-free)
- **MLflow Integration**: 60+ tracked experiments across attention mechanisms
- **Architecture**: 9-channel input with spatial/channel attention
- **Performance**: 93.14% validation accuracy, 94.79% F1 score

<div style="text-align: center;">
    <img src="/assets/img/project-1/CNN_grasp.drawio.png" alt="CNN Architecture" style="width: 50%; max-width: 400px;">
    <p><em>GraspPointCNN with attention mechanism for grasp quality prediction</em></p>
</div>

##### VLA-Enhanced Decision Making
Language-guided grasp reasoning with confidence weighting:
- **Prompt Engineering**: Structured queries for grasp point evaluation
- **Confidence Scoring**: Dynamic assessment of VLA prediction quality
- **Fallback Strategy**: Robust degradation to proven CV algorithms
- **Continuous Learning**: Adaptation through operational feedback

<div style="text-align: center; display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="/assets/img/project-1/cv_op1.png" alt="CV Output 1" style="max-width: 400px;">
    <img src="/assets/img/project-1/cv_op2.png" alt="CV Output 2" style="max-width: 400px;">
</div>
<div style="text-align: center;">
    <p><em>Hybrid CV-ML-VLA pipeline: Traditional geometric analysis (left) enhanced with foundation model reasoning (right)</em></p>
</div>

---

### 4. Production Optimization

#### 4.1 Custom CUDA Kernel Development
Developed GPU kernels addressing CPU bottlenecks in point cloud generation:
- **Performance**: 5x speedup (150ms → 30ms) for real-time operation
- **Implementation**: Parallelized 1.5M pixel processing with memory optimization

#### 4.2 TensorRT & Model Optimization  
System-wide acceleration with model compilation:
- **Models**: YOLOv8, RAFT-Stereo, GraspPointCNN, and LLaVA components
- **Techniques**: FP16 precision, operator fusion, graph optimization
- **Results**: 35% throughput improvement (20 → 27 FPS)

#### 4.3 AWS Training Infrastructure
Cloud-based training pipeline for VLA fine-tuning:
- **Infrastructure**: g4dn.xlarge instances with Tesla T4 GPUs
- **Cost Efficiency**: LoRA fine-tuning reduces computational requirements
- **Scalability**: MLflow tracking across distributed experiments
- **Deployment**: Docker containerization for environment consistency

<div style="text-align: center;">
    <img src="/assets/img/project-1/rex_grasp_4x.gif" alt="System Operation" style="width: 30%; max-width: 240px;">
    <p><em>Production-optimized VLA-enhanced grasping system in operation</em></p>
</div>

---

### 5. Results & Performance

#### VLA System Performance

<div class="table-responsive">
  <table class="table">
    <thead>
      <tr>
        <th>Model</th>
        <th>Configuration</th>
        <th>Validation Accuracy</th>
        <th>Training Infrastructure</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>LLaVA-1.6-Mistral-7B</td>
        <td>baseline_5e5</td>
        <td><strong>88.0%</strong></td>
        <td>AWS GPU + MLflow</td>
      </tr>
      <tr>
        <td>GraspPointCNN</td>
        <td>Spatial Attention</td>
        <td>93.14%</td>
        <td>Self-supervised</td>
      </tr>
      <tr>
        <td>Hybrid Integration</td>
        <td>Confidence-weighted</td>
        <td>82.66% field success</td>
        <td>Production deployment</td>
      </tr>
    </tbody>
  </table>
</div>

#### System Performance Comparison (150 test cases)

<div class="table-responsive">
  <table class="table">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Classical CV</th>
        <th>Hybrid (CV+ML+VLA)</th>
        <th>Improvement</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Overall Success Rate (%)</td>
        <td>78.00</td>
        <td><strong>82.66</strong></td>
        <td>+4.66</td>
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
        <td>Accuracy (px)</td>
        <td>25.3</td>
        <td>27.1</td>
        <td>+1.8</td>
      </tr>
    </tbody>
  </table>
</div>

#### Production Optimization Results

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
        <td>VLA Training (AWS)</td>
        <td>CPU-only</td>
        <td>GPU acceleration</td>
        <td>3x speedup</td>
      </tr>
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
    </tbody>
  </table>
</div>

---

### 6. Key Contributions

**System Development:**
- Complete VLA integration pipeline with LLaVA-1.6-Mistral-7B foundation model
- LoRA fine-tuning achieving 88% validation accuracy through systematic optimization
- Hybrid decision framework balancing CV, ML, and VLA predictions with confidence weighting
- Self-supervised learning eliminating manual annotation requirements
- AWS GPU training infrastructure with MLflow experiment tracking

**Performance Optimization:**
- Custom CUDA kernels for 5x point cloud generation speedup
- TensorRT model compilation for 35% inference improvement
- Production Docker deployment achieving 82.66% field success rate

This research is conducted under Prof. Abhisesh Silwal and Prof. George A. Kantor.

---

### 7. Skills and Technologies

- **Foundation Models**: LLaVA-1.6-Mistral-7B, LoRA Fine-tuning, Vision-Language Integration
- **Languages**: Python, C++, CUDA
- **Deep Learning**: PyTorch, CNN Architecture, Self-Supervised Learning, Attention Mechanisms
- **Computer Vision**: Instance Segmentation, Depth Estimation, Point Cloud Processing, 3D Perception
- **MLOps**: MLflow Experiment Tracking, Model Versioning, Hyperparameter Optimization
- **Cloud & Performance**: AWS EC2/GPU, Custom CUDA Kernels, TensorRT Optimization
- **Production**: Docker Containerization, ROS2 Integration, Real-time Systems

---

### 8. Project Repositories

- [LeafGrasp-Vision-ML](https://github.com/Srecharan/Leaf-Grasping-Vision-ML.git): **Main Repository with VLA System Integration**
- [YOLOv8 Segmentation](https://github.com/Srecharan/YoloV8Seg-REX.git): Real-time Leaf Instance Segmentation  
- [RAFT-Stereo](https://github.com/Srecharan/RAFTStereo-REX.git): High-Precision Depth Estimation with CUDA
- [REX-Robot](https://github.com/Srecharan/REX-Robot.git): 6-DOF Gantry Robot Control System

---

### 9. References
[1] Srecharan Selvam, Abhisesh Silwal, George Kantor "Self-Supervised Learning for Robotic Leaf Manipulation: A Hybrid Geometric-Neural Approach", https://arxiv.org/pdf/2505.0370, Under review at ICCV 2025.

[2] Silwal, A., Zhang, X. M., Hadlock, T., Neice, J., Haque, S., Kaundanya, A., Lu, C., Vinatzer, B. A., Kantor, G., & Li, S. (2024). Towards an AI-Driven Cyber-Physical System for Closed-Loop Control of Plant Diseases. *Proceedings of the AAAI Symposium Series*, *4*(1), 432-435.