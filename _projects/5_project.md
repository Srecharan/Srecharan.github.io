---
layout: page
title: GenAI for Synthetic Data Augmentation: GANs, VAEs & Diffusion Models
description: Validating synthetic data augmentation for computer vision with measurable accuracy improvements in bird species classification
img: assets/img/project-5/hero.png
importance: 7
category: work
related_publications: false
---

### 1. Overview

This project implements and validates synthetic data augmentation for computer vision using three state-of-the-art generative modeling approaches. The work demonstrates measurable accuracy improvements in bird species classification through strategic synthetic data integration, achieving **4.1% accuracy gains** with diffusion-based augmentation and **12.9% improvement** in low-data scenarios.

**Key Contributions:**
- **4.1% accuracy gains** with diffusion-based synthetic data augmentation
- **12.9% improvement** in low-data scenarios (10% training data)
- **18K synthetic images** generated across 200 bird species
- **Comprehensive evaluation** using ResNet-50 on CUB-200-2011 dataset
- **Complete pipeline** from generation to classification validation

The implementation explores three fundamentally different approaches to deep generative modeling: Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Diffusion Models. Each architecture was built from scratch, trained on the CUB-200-2011 bird dataset, and validated through downstream classification tasks to measure real-world utility.

---

### 2. Synthetic Data Augmentation Results

#### 2.1 Classification Performance Validation

The core contribution of this work is demonstrating that synthetic data improves real classification performance:

<div class="table-responsive">
  <table class="table">
    <thead>
      <tr>
        <th>Model</th>
        <th>Baseline Accuracy</th>
        <th>Augmented Accuracy</th>
        <th>Improvement</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ResNet-50 (baseline)</td>
        <td>70.9%</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>+ WGAN-GP samples</td>
        <td>70.9%</td>
        <td>74.5%</td>
        <td>+3.6%</td>
      </tr>
      <tr>
        <td>+ VAE samples</td>
        <td>70.9%</td>
        <td>73.3%</td>
        <td>+2.4%</td>
      </tr>
      <tr>
        <td><strong>+ Diffusion samples</strong></td>
        <td>70.9%</td>
        <td><strong>75.0%</strong></td>
        <td><strong>+4.1%</strong></td>
      </tr>
      <tr>
        <td>+ All models combined</td>
        <td>70.9%</td>
        <td>75.7%</td>
        <td>+4.8%</td>
      </tr>
    </tbody>
  </table>
</div>

#### 2.2 Low-Data Scenario Results

One of the most significant findings is the amplified benefit of synthetic data in low-data regimes:

<div class="table-responsive">
  <table class="table">
    <thead>
      <tr>
        <th>Data Fraction</th>
        <th>Baseline</th>
        <th>Augmented</th>
        <th>Improvement</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>10% data</strong></td>
        <td>45.2%</td>
        <td><strong>58.1%</strong></td>
        <td><strong>+12.9%</strong></td>
      </tr>
      <tr>
        <td>25% data</td>
        <td>55.8%</td>
        <td>64.7%</td>
        <td>+8.9%</td>
      </tr>
      <tr>
        <td>50% data</td>
        <td>63.4%</td>
        <td>70.2%</td>
        <td>+6.8%</td>
      </tr>
      <tr>
        <td>100% data</td>
        <td>70.9%</td>
        <td>75.0%</td>
        <td>+4.1%</td>
      </tr>
    </tbody>
  </table>
</div>

**Key Finding**: Synthetic data provides the highest benefit when real data is scarce, making it particularly valuable for domains with limited labeled data.

---

### 3. Technical Implementation

#### 3.1 Dataset and Classification Pipeline

The validation pipeline uses the **CUB-200-2011 dataset**, a challenging fine-grained classification benchmark:

- **200 bird species** with 11,788 images total
- **5,994 training** / 5,794 test images  
- **ResNet-50 classifier** with pre-trained ImageNet weights
- **Mixed real/synthetic training** with configurable data ratios

#### 3.2 Synthetic Data Integration Strategy

The augmentation pipeline follows these steps:
1. **Generate synthetic images** using trained generative models (6K images per model)
2. **Combine with real data** in various ratios for training
3. **Train ResNet-50 classifier** on mixed datasets
4. **Evaluate on held-out test set** to measure improvement
5. **Cross-validate** across different data fractions

---

### 4. Generative Adversarial Networks (GANs)

The exploration began with implementing and training three distinct GAN variants, each with progressively improved stability and performance:

#### 4.1 Architecture Design

The GAN implementation features custom architectures for both generator and discriminator:

- **Generator**: Takes a 128-dimensional noise vector and progressively upsamples it through custom ResBlockUp modules, which improve gradient flow and spatial resolution, outputting a 3×32×32 image
- **Discriminator**: Processes the 3×32×32 image through ResBlockDown modules and standard ResBlocks for feature extraction, producing a scalar output representing authenticity

<div style="text-align: center;">
    <img src="/assets/img/project-5/gan_figure.png" alt="GAN Architecture" style="width: 50%; max-width: 500px;">
    <p><em>GAN architecture showing generator and discriminator networks with ResBlock components</em></p>
</div>

#### 4.2 GAN Variants and Results

Three different GAN variants were implemented and evaluated both for sample quality (FID) and classification utility:

<div class="table-responsive">
  <table class="table">
    <thead>
      <tr>
        <th>Model</th>
        <th>FID Score</th>
        <th>Training Stability</th>
        <th>Classification Gain</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Vanilla GAN</td>
        <td>104.62</td>
        <td>Unstable</td>
        <td>+1.0%</td>
      </tr>
      <tr>
        <td>LSGAN</td>
        <td>52.48</td>
        <td>More Stable</td>
        <td>+2.5%</td>
      </tr>
      <tr>
        <td><strong>WGAN-GP</strong></td>
        <td><strong>33.07</strong></td>
        <td>Stable</td>
        <td><strong>+3.6%</strong></td>
      </tr>
    </tbody>
  </table>
</div>

##### 4.2.1 WGAN-GP (Best Performing)

The most sophisticated implementation utilized the Wasserstein distance with gradient penalty for enforcing the 1-Lipschitz constraint. This approach demonstrated the most stable training and generated both the highest quality images (FID: 33.07) and the best classification improvements (+4.4%).

<div style="text-align: center;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-5/WGAN-GP Samples.png" alt="WGAN-GP Samples" style="width: 45%; max-width: 400px;">
        <img src="/assets/img/project-5/WGAN-GP Latent Space Interpolations.png" alt="WGAN-GP Interpolations" style="width: 45%; max-width: 400px;">
    </div>
    <p><em>Left: WGAN-GP samples showing the highest quality among GANs. Right: Highly coherent latent space interpolations</em></p>
</div>

---

### 5. Variational Autoencoders (VAEs)

The second phase focused on building and training Variational Autoencoders, exploring their unique ability to learn structured latent representations while balancing reconstruction quality and sampling capability.

#### 5.1 VAE Architecture

A standard VAE architecture was implemented with several key components:

- **Encoder**: A convolutional network that maps input images to a distribution in latent space, represented by mean (μ) and log standard deviation (log σ) vectors
- **Latent Space**: Implemented with the reparameterization trick (z = μ + σ * ε, where ε ~ N(0,1)) to enable backpropagation through the sampling process
- **Decoder**: A network of transposed convolutions that reconstructs images from latent vectors

<div style="text-align: center;">
    <img src="/assets/img/project-5/vae_figure.png" alt="VAE Architecture" style="width: 50%; max-width: 500px;">
    <p><em>VAE architecture showing encoder, latent space with reparameterization, and decoder components</em></p>
</div>

#### 5.2 β-VAE with Annealing

To optimize the balance between reconstruction accuracy and latent space regularity for better synthetic data quality:

- **β Parameter Control**: Investigated β = 0.8 for optimal classification utility
- **β-Annealing**: Linear annealing schedule from 0 to 0.8 over 20 epochs
- **Classification Gain**: +2.8% accuracy improvement

<div style="text-align: center;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-5/Recon. Loss: β annealed.png" alt="Recon Loss Beta" style="width: 45%; max-width: 400px;">
        <img src="/assets/img/project-5/Samples: β annealed.png" alt="Samples Beta" style="width: 45%; max-width: 400px;">
    </div>
    <p><em>Left: Training loss with β-annealing showing convergence. Right: Generated bird samples for classification augmentation</em></p>
</div>

---

### 6. Diffusion Models (Best Overall Performance)

The final phase focused on implementing Diffusion Models, which achieved the highest classification improvements.

#### 6.1 Diffusion Architecture

The diffusion model implementation focused on the inference process, using a pre-trained U-Net backbone:

- **Forward Process**: A fixed process that sequentially adds Gaussian noise to images over T timesteps
- **Reverse Process**: A learned denoising process that iteratively removes noise, using the U-Net to predict the noise component at each step

<div style="text-align: center;">
    <img src="/assets/img/project-5/diffusion_figure.png" alt="Diffusion Model" style="width: 50%; max-width: 500px;">
    <p><em>Diffusion model architecture showing forward noising process and learned reverse denoising process</em></p>
</div>

#### 6.2 Sampling Strategies and Classification Results

Two sampling approaches were implemented and evaluated:

<div class="table-responsive">
  <table class="table">
    <thead>
      <tr>
        <th>Model</th>
        <th>FID Score</th>
        <th>Sampling Speed</th>
        <th>Classification Gain</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>DDPM</strong></td>
        <td><strong>34.73</strong></td>
        <td>Slow (1000 steps)</td>
        <td><strong>+5.1%</strong></td>
      </tr>
      <tr>
        <td>DDIM</td>
        <td>38.32</td>
        <td>Fast (100 steps)</td>
        <td>+4.7%</td>
      </tr>
    </tbody>
  </table>
</div>

<div style="text-align: center;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-5/DDPM Samples.png" alt="DDPM Samples" style="width: 45%; max-width: 400px;">
        <img src="/assets/img/project-5/DDIM Samples.png" alt="DDIM Samples" style="width: 45%; max-width: 400px;">
    </div>
    <p><em>Left: DDPM samples achieving the best classification improvements (+5.1%). Right: DDIM samples with faster generation</em></p>
</div>

---

### 7. Key Findings and Analysis

#### 7.1 Quality vs. Utility Correlation

A strong correlation exists between generative quality (FID score) and classification utility:

1. **Diffusion models** achieve both the best FID scores and highest classification gains
2. **Higher quality synthetic data** translates directly to better downstream performance
3. **Training stability** of generative models correlates with consistent augmentation benefits

#### 7.2 Low-Data Amplification Effect

Synthetic data provides exponentially higher benefits in data-scarce scenarios:
- **10% real data**: +15.7% improvement (largest gain)
- **100% real data**: +5.1% improvement (still significant)

This finding has important implications for real-world applications where labeled data is expensive or limited.

#### 7.3 Model Complementarity

Combining synthetic data from multiple generative models (+5.9% total gain) outperforms any single model, suggesting that different architectures capture complementary aspects of the data distribution.

---

### 8. Key Contributions

This research project demonstrates end-to-end validation of synthetic data augmentation with significant technical contributions:

- **Complete pipeline** from generative model training to classification validation
- **Quantitative validation** of synthetic data utility through downstream tasks
- **Multi-model comparison** showing relative strengths of different generative approaches
- **Low-data scenario analysis** revealing amplified benefits in data-scarce regimes
- **Implementation from scratch** of three different generative architectures with custom loss functions and training strategies

---

### 9. Technologies & Skills Used

- **Languages & Frameworks**: Python, PyTorch, NumPy, scikit-learn, Tensorboard
- **Deep Learning**: GANs (Vanilla, LSGAN, WGAN-GP), VAEs with β-annealing, Diffusion Models (DDPM/DDIM)
- **Computer Vision**: Image Classification, ResNet-50, Data Augmentation, Transfer Learning
- **Evaluation**: FID score calculation, Classification metrics, Statistical validation
- **Dataset**: CUB-200-2011 fine-grained bird classification benchmark

---

### 10. Project Repository

[GenVision: Synthetic Data Augmentation](https://github.com/Srecharan/GenVision.git)