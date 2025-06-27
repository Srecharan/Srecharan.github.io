---
layout: page
title: GenAI for Synthetic Data Augmentation
description: Validating synthetic data augmentation for computer vision with measurable accuracy improvements
img: assets/img/project-5/hero.png
importance: 9
category: work
related_publications: false
---

### Overview

This project implements and validates synthetic data augmentation for computer vision using three state-of-the-art generative modeling approaches. The work demonstrates measurable accuracy improvements in bird species classification through strategic synthetic data integration.

### Key Contributions

- 4.1% accuracy gains with diffusion-based synthetic data augmentation
- 12.9% improvement in low-data scenarios (10% training data)
- 18K synthetic images generated across 200 bird species
- Comprehensive evaluation using ResNet-50 on CUB-200-2011 dataset

### Technical Implementation

The implementation explores three fundamentally different approaches to deep generative modeling:

**Generative Adversarial Networks (GANs)**
- Vanilla GAN, LSGAN, WGAN-GP variants
- Best performance: WGAN-GP with 33.07 FID score

**Variational Autoencoders (VAEs)**  
- Beta-VAE with annealing schedule
- Optimal beta parameter of 0.8 for classification utility

**Diffusion Models**
- DDPM and DDIM sampling strategies
- Best classification improvements: +5.1% with DDPM

### Results

All three approaches show synthetic data improves real classification performance, with diffusion models achieving the highest gains. The benefit is amplified in low-data scenarios, making this particularly valuable for domains with limited labeled data.

### Technologies Used

- Python, PyTorch, NumPy, scikit-learn
- GANs, VAEs, Diffusion Models (DDPM/DDIM)
- Computer Vision, Image Classification, Data Augmentation

### Project Repository

[GenVision: Synthetic Data Augmentation](https://github.com/Srecharan/GenVision.git) 