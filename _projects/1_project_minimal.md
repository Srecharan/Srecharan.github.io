---
layout: page
title: Vision-Language-Action Enhanced Robotic Leaf Grasping
description: A novel vision system combining computer vision, deep learning, and Vision-Language-Action models
img: assets/img/project-1/hero.png
importance: 8
category: work
related_publications: false
---

### Overview

A real-time vision system for leaf manipulation that combines geometric computer vision with deep learning and Vision-Language-Action (VLA) models. This hybrid system integrates YOLOv8 for segmentation, RAFT-Stereo for depth estimation, and a custom CNN enhanced with LLaVA-1.6-Mistral-7B for intelligent grasp reasoning.

### Key Features

- **VLA System**: LoRA fine-tuning of LLaVA-1.6-Mistral-7B foundation model
- **Self-supervised Learning**: Eliminates 100% manual annotation requirements  
- **Production Optimization**: Custom CUDA kernels and TensorRT acceleration
- **AWS Infrastructure**: GPU training with MLflow experiment tracking

### Results

The system achieves 82% leaf grasp success rate in field tests with 88% validation accuracy through systematic hyperparameter optimization.

### Technologies Used

- Python, C++, CUDA
- PyTorch, CNN Architecture, Self-Supervised Learning
- AWS EC2/GPU, Custom CUDA Kernels, TensorRT Optimization
- Docker Containerization, ROS2 Integration

### Project Repository

[LeafGrasp-Vision-ML](https://github.com/Srecharan/Leaf-Grasping-Vision-ML.git) 