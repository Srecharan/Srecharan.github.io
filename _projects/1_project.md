---
layout: page
title: Vision-Language-Action Enhanced Robotic Leaf Manipulation
description: A novel vision system combining geometric computer vision, deep learning, and Vision-Language-Action models for intelligent leaf manipulation
img: assets/img/project-1/hero.png
importance: 1
category: work
related_publications: false
---

## Overview

A real-time vision system for leaf manipulation that combines geometric computer vision with deep learning and **Vision-Language-Action (VLA) models**. This hybrid system integrates YOLOv8 for segmentation, RAFT-Stereo for depth estimation, and a custom CNN enhanced with **LLaVA-1.6-Mistral-7B for intelligent grasp reasoning**.

Key achievements:
- **Self-supervised learning eliminating 100% manual annotation**
- **LoRA fine-tuning achieving 88% validation accuracy**
- **Confidence-weighted framework** dynamically balancing traditional CV, ML, and VLA predictions
- **Custom CUDA kernels** and **TensorRT acceleration**
- **AWS GPU training infrastructure** with **Docker containerization**

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project-1/REX.drawiof.png" title="System Architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Multi-stage perception pipeline enhanced with Vision-Language-Action integration
</div>

---

## Vision-Language-Action (VLA) System

### LLaVA Integration and Fine-tuning
**Foundation Model Enhancement for Grasp Reasoning**

Integrated LLaVA-1.6-Mistral-7B foundation model with parameter-efficient LoRA fine-tuning:

- **Base Model**: LLaVA-1.6-Mistral-7B (CLIP + Vicuna) for vision-language understanding
- **Fine-tuning**: LoRA adaptation (rank=8, alpha=32) for leaf grasping tasks
- **Training Infrastructure**: AWS GPU acceleration with MLflow experiment tracking
- **Performance**: 88.0% validation accuracy through systematic hyperparameter optimization
- **Experiments**: 4 systematic configurations with comprehensive evaluation

### Hybrid CV-VLA Decision Framework
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

## Multi-Stage Perception Pipeline

### Instance Segmentation (YOLOv8)
Fine-tuned on approximately 900 images achieving 68% mAP@[0.5:0.95] with TensorRT optimization for real-time performance.

### Depth Estimation (RAFT-Stereo)
High-precision depth maps with sub-pixel accuracy (<0.5px) enhanced with custom CUDA kernels for accelerated point cloud generation.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project-1/rgb_input.png" title="RGB Input" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project-1/depth0.png" title="Depth Map" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project-1/plant_pcd-2x.gif" title="Point Cloud" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Stereo vision pipeline: RGB input → depth estimation → 3D reconstruction
</div>

### Hybrid Grasp Point Selection

#### Traditional CV Pipeline
Pareto optimization for leaf selection with geometric scoring:
- **Clutter Score (35%)**: Isolation using Signed Distance Fields
- **Distance Score (35%)**: Camera proximity with exponential falloff  
- **Visibility Score (30%)**: Frame position and completeness

Grasp point selection criteria:
- **Flatness Analysis (25%)**: Surface smoothness via depth gradients
- **Approach Vector Quality (40%)**: Optimal robot orientation
- **Accessibility (15%)**: Camera-relative positioning
- **Edge Awareness (20%)**: Boundary distance analysis

#### ML Enhancement with MLflow Tracking
Custom GraspPointCNN with comprehensive experiment management:
- **Self-Supervised Learning**: CV-generated training data (100% annotation-free)
- **MLflow Integration**: 60+ tracked experiments across attention mechanisms
- **Architecture**: 9-channel input with spatial/channel attention
- **Performance**: 93.14% validation accuracy, 94.79% F1 score

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project-1/CNN_grasp.drawio.png" title="CNN Architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    GraspPointCNN with attention mechanism for grasp quality prediction
</div>

#### VLA-Enhanced Decision Making
Language-guided grasp reasoning with confidence weighting:
- **Prompt Engineering**: Structured queries for grasp point evaluation
- **Confidence Scoring**: Dynamic assessment of VLA prediction quality
- **Fallback Strategy**: Robust degradation to proven CV algorithms
- **Continuous Learning**: Adaptation through operational feedback

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project-1/cv_op1.png" title="CV Output 1" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project-1/cv_op2.png" title="CV Output 2" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Hybrid CV-ML-VLA pipeline: Traditional geometric analysis (left) enhanced with foundation model reasoning (right)
</div>

---

## Production Optimization

### Custom CUDA Kernel Development
Developed GPU kernels addressing CPU bottlenecks in point cloud generation:
- **Performance**: 5x speedup (150ms → 30ms) for real-time operation
- **Implementation**: Parallelized 1.5M pixel processing with memory optimization

### TensorRT & Model Optimization  
System-wide acceleration with model compilation:
- **Models**: YOLOv8, RAFT-Stereo, GraspPointCNN, and LLaVA components
- **Techniques**: FP16 precision, operator fusion, graph optimization
- **Results**: 35% throughput improvement (20 → 27 FPS)

### AWS Training Infrastructure
Cloud-based training pipeline for VLA fine-tuning:
- **Infrastructure**: g4dn.xlarge instances with Tesla T4 GPUs
- **Cost Efficiency**: LoRA fine-tuning reduces computational requirements
- **Scalability**: MLflow tracking across distributed experiments
- **Deployment**: Docker containerization for environment consistency

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project-1/rex_grasp_4x.gif" title="System Operation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Production-optimized VLA-enhanced grasping system in operation
</div>

---

## Results & Performance

### Performance Results

#### VLA System Performance

**Model Performance Metrics:**
- **LLaVA-1.6-Mistral-7B**: 88.0% validation accuracy with LoRA fine-tuning (AWS GPU + MLflow)
- **GraspPointCNN**: 93.14% validation accuracy with spatial attention (self-supervised)
- **Hybrid Integration**: 82.66% field success rate with confidence weighting (production deployment)

#### System Performance Comparison (150 test cases)

**Performance Improvements:**
- **Overall Success Rate**: 78.00% → **82.66%** (+4.66% improvement)
- **Feature Alignment**: 80.67% → 83.33% (+2.66% improvement)  
- **Edge Case Handling**: 75.33% → 77.33% (+2.00% improvement)
- **Accuracy**: 25.3px → 27.1px (+1.8px improvement)

#### Production Optimization Results

**System Optimizations:**
- **VLA Training (AWS)**: CPU-only → GPU acceleration (3x speedup)
- **Point Cloud Generation**: 150ms → 30ms (5x speedup)
- **Inference Throughput**: 20 FPS → 27 FPS (35% improvement)
- **Dataset Creation**: Manual annotation → Self-supervised (100% elimination)

---

## Key Contributions

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

## Skills and Technologies

- **Foundation Models**: LLaVA-1.6-Mistral-7B, LoRA Fine-tuning, Vision-Language Integration
- **Languages**: Python, C++, CUDA
- **Deep Learning**: PyTorch, CNN Architecture, Self-Supervised Learning, Attention Mechanisms
- **Computer Vision**: Instance Segmentation, Depth Estimation, Point Cloud Processing, 3D Perception
- **MLOps**: MLflow Experiment Tracking, Model Versioning, Hyperparameter Optimization
- **Cloud & Performance**: AWS EC2/GPU, Custom CUDA Kernels, TensorRT Optimization
- **Production**: Docker Containerization, ROS2 Integration, Real-time Systems

---

## Project Repositories

- [LeafGrasp-Vision-ML](https://github.com/Srecharan/Leaf-Grasping-Vision-ML.git): **Main Repository with VLA System Integration**
- [YOLOv8 Segmentation](https://github.com/Srecharan/YoloV8Seg-REX.git): Real-time Leaf Instance Segmentation  
- [RAFT-Stereo](https://github.com/Srecharan/RAFTStereo-REX.git): High-Precision Depth Estimation with CUDA
- [REX-Robot](https://github.com/Srecharan/REX-Robot.git): 6-DOF Gantry Robot Control System

---

## References
[1] Srecharan Selvam, Abhisesh Silwal, George Kantor "Self-Supervised Learning for Robotic Leaf Manipulation: A Hybrid Geometric-Neural Approach", https://arxiv.org/pdf/2505.0370, Under review at ICCV 2025.

[2] Silwal, A., Zhang, X. M., Hadlock, T., Neice, J., Haque, S., Kaundanya, A., Lu, C., Vinatzer, B. A., Kantor, G., & Li, S. (2024). Towards an AI-Driven Cyber-Physical System for Closed-Loop Control of Plant Diseases. *Proceedings of the AAAI Symposium Series*, *4*(1), 432-435.