---
layout: page
title: Deep Image Synthesis with GANs, VAEs, and Diffusion Models
description: A comprehensive implementation of three leading generative model architectures for image synthesis
img: assets/img/project-5/hero.png
importance: 5
category: work
related_publications: false
---

### 1. Overview

This project explores three fundamentally different approaches to deep generative modeling for image synthesis, implementing Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Diffusion Models from scratch. Each architecture was built, trained, and evaluated on the challenging CUB-200-2011 bird dataset, which requires models to capture subtle visual features and details. The implementation includes custom architectures, loss functions, and training strategies, with rigorous evaluation using the Fréchet Inception Distance (FID) metric to quantitatively assess the quality of generated images. The project demonstrates the effectiveness of different generative approaches, with WGAN-GP emerging as the strongest performer, achieving an FID score of 33.07.

---

### 2. Generative Adversarial Networks (GANs)

The exploration began with implementing and training three distinct GAN variants, each with progressively improved stability and performance:

#### 2.1 Architecture Design

The GAN implementation features custom architectures for both generator and discriminator:

- **Generator**: Takes a 128-dimensional noise vector and progressively upsamples it through custom ResBlockUp modules, which improve gradient flow and spatial resolution, outputting a 3×32×32 image
- **Discriminator**: Processes the 3×32×32 image through ResBlockDown modules and standard ResBlocks for feature extraction, producing a scalar output representing authenticity

<div style="text-align: center;">
    <img src="/assets/img/project-5/gan_figure.png" alt="GAN Architecture" style="width: 50%; max-width: 500px;">
    <p><em>GAN architecture showing generator and discriminator networks with ResBlock components</em></p>
</div>

#### 2.2 GAN Variants and Loss Functions

Three different GAN variants were implemented and trained from scratch, each with its unique loss function and training dynamics:

##### 2.2.1 Vanilla GAN

The original GAN formulation was implemented using Binary Cross-Entropy loss. The generator was trained to minimize the log-probability of the discriminator correctly identifying fake images, while the discriminator was trained to maximize the log-probability of correct classification.

Training this model revealed the challenges of the original GAN approach, including significant training instability and mode collapse. Despite these issues, the model managed to learn basic bird shapes and features, achieving an FID score of 104.62.

<div style="text-align: center;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-5/Vanilla_GAN_Samples.png" alt="Vanilla GAN Samples" style="width: 45%; max-width: 400px;">
        <img src="/assets/img/project-5/Vanilla GAN Latent Space Interpolations.png" alt="Vanilla GAN Interpolations" style="width: 45%; max-width: 400px;">
    </div>
    <p><em>Left: Vanilla GAN samples. Right: Latent space interpolations showing transitions between generated images</em></p>
</div>

##### 2.2.2 Least Squares GAN (LSGAN)

Building upon the Vanilla GAN, the LSGAN replaced the sigmoid cross-entropy with a least-squares loss, minimizing the squared difference between the discriminator's output and target labels. This modification produced more stable training dynamics and improved sample diversity.

The LSGAN implementation successfully reduced training instability and generated more diverse and realistic bird images, with an FID score of 52.48, a significant improvement over the Vanilla GAN.

<div style="text-align: center;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-5/LS-GAN Samples.png" alt="LSGAN Samples" style="width: 45%; max-width: 400px;">
        <img src="/assets/img/project-5/LS-GAN Latent Space Interpolations.png" alt="LSGAN Interpolations" style="width: 45%; max-width: 400px;">
    </div>
    <p><em>Left: LSGAN samples showing improved diversity. Right: Smooth latent space interpolations</em></p>
</div>

##### 2.2.3 Wasserstein GAN with Gradient Penalty (WGAN-GP)

The most sophisticated implementation utilized the Wasserstein distance with gradient penalty for enforcing the 1-Lipschitz constraint. The discriminator (now a "critic") was trained to estimate the Wasserstein distance between real and generated distributions.

A key innovation in this implementation was the gradient penalty term, calculated by sampling interpolated points between real and fake images and penalizing deviations of the gradient norm from 1. This approach demonstrated the most stable training and generated the highest quality images, with an FID score of 33.07.

<div style="text-align: center;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-5/WGAN-GP Samples.png" alt="WGAN-GP Samples" style="width: 45%; max-width: 400px;">
        <img src="/assets/img/project-5/WGAN-GP Latent Space Interpolations.png" alt="WGAN-GP Interpolations" style="width: 45%; max-width: 400px;">
    </div>
    <p><em>Left: WGAN-GP samples showing the highest quality. Right: Highly coherent latent space interpolations</em></p>
</div>

---

### 3. Variational Autoencoders (VAEs)

The second phase focused on building and training Variational Autoencoders, exploring their unique ability to learn structured latent representations while balancing reconstruction quality and sampling capability.

#### 3.1 VAE Architecture

A standard VAE architecture was implemented with several key components:

- **Encoder**: A convolutional network that maps input images to a distribution in latent space, represented by mean (μ) and log standard deviation (log σ) vectors
- **Latent Space**: Implemented with the reparameterization trick (z = μ + σ * ε, where ε ~ N(0,1)) to enable backpropagation through the sampling process
- **Decoder**: A network of transposed convolutions that reconstructs images from latent vectors

<div style="text-align: center;">
    <img src="/assets/img/project-5/vae_figure.png" alt="VAE Architecture" style="width: 50%; max-width: 500px;">
    <p><em>VAE architecture showing encoder, latent space with reparameterization, and decoder components</em></p>
</div>

#### 3.2 Latent Space Experiments

Experiments were conducted with different latent space dimensions to explore the trade-offs:

- **Latent Size 16**: Limited capacity resulted in blurry reconstructions with high reconstruction loss (~200)
- **Latent Size 128**: Provided a balance between reconstruction quality and sample diversity with moderate loss (~75)
- **Latent Size 1024**: Achieved the sharpest reconstructions with lowest loss (~40) but potentially less structured latent space

<div style="text-align: center;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-5/Reconstructions: (size 16).png" alt="Reconstructions Size 16" style="width: 30%; max-width: 275px;">
        <img src="/assets/img/project-5/Reconstructions: (size 128).png" alt="Reconstructions Size 128" style="width: 30%; max-width: 275px;">
        <img src="/assets/img/project-5/Reconstructions: (size 1024).png" alt="Reconstructions Size 1024" style="width: 30%; max-width: 275px;">
    </div>
    <p><em>Reconstructions with increasing latent dimensions (16, 128, 1024), showing improved quality with larger latent spaces</em></p>
</div>

#### 3.3 β-VAE and Annealing

To optimize the balance between reconstruction accuracy and latent space regularity, the following techniques were implemented and trained:

- **β Parameter Control**: Different values of β (0.8, 1.0, 1.2) were investigated, which controls the weight of the KL divergence term in the loss function
- **β-Annealing**: A linear annealing schedule was implemented that gradually increased β from 0 to the target value (0.8) over 20 epochs

<div style="text-align: center;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-5/constant_beta_samples.png" alt="Constant Beta Samples" style="width: 45%; max-width: 400px;">
        <img src="/assets/img/project-5/annelaed_beta.png" alt="Annealed Beta Samples" style="width: 45%; max-width: 400px;">
    </div>
    <p><em>Left: Samples with constant β=0.8. Right: Samples after β-annealing, showing improved quality and diversity</em></p>
</div>

---

### 4. Diffusion Models

The final phase focused on implementing Diffusion Models, specifically investigating sampling strategies for a pre-trained model.

#### 4.1 Diffusion Architecture

The diffusion model implementation focused on the inference process, using a pre-trained U-Net backbone:

- **Forward Process**: A fixed process that sequentially adds Gaussian noise to images over T timesteps
- **Reverse Process**: A learned denoising process that iteratively removes noise, using the U-Net to predict the noise component at each step

<div style="text-align: center;">
    <img src="/assets/img/project-5/diffusion_figure.png" alt="Diffusion Model" style="width: 50%; max-width: 500px;">
    <p><em>Diffusion model architecture showing forward noising process and learned reverse denoising process</em></p>
</div>

#### 4.2 Sampling Strategies

Two sampling approaches were implemented and thoroughly tested:

- **DDPM (Denoising Diffusion Probabilistic Models)**: The original sampling approach requiring approximately 1000 sequential denoising steps. This produced high-quality samples with an FID score of 34.73.

- **DDIM (Denoising Diffusion Implicit Models)**: An accelerated sampling approach using a non-Markovian diffusion process, requiring only 100 steps. This maintained comparable quality with an FID score of 38.32, demonstrating a significant efficiency improvement.

<div style="text-align: center;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <img src="/assets/img/project-5/DDPM Samples.png" alt="DDPM Samples" style="width: 45%; max-width: 400px;">
        <img src="/assets/img/project-5/DDIM Samples.png" alt="DDIM Samples" style="width: 45%; max-width: 400px;">
    </div>
    <p><em>Left: DDPM samples (1000 steps). Right: DDIM samples (100 steps), showing comparable quality with 10× faster sampling</em></p>
</div>

---

### 5. Performance Evaluation

A comprehensive quantitative evaluation was conducted using the Fréchet Inception Distance (FID) to measure the quality and diversity of generated images:

<div style="text-align: center;">
  <table class="table" style="width: 80%; margin: 0 auto; border-collapse: collapse; border: 1px solid #ddd;">
    <thead>
      <tr style="background-color: #f2f2f2;">
        <th style="padding: 12px; border: 1px solid #ddd;">Model</th>
        <th style="padding: 12px; border: 1px solid #ddd;">FID Score</th>
        <th style="padding: 12px; border: 1px solid #ddd;">Training Stability</th>
        <th style="padding: 12px; border: 1px solid #ddd;">Sampling Speed</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 12px; border: 1px solid #ddd;">Vanilla GAN</td>
        <td style="padding: 12px; border: 1px solid #ddd;">104.62</td>
        <td style="padding: 12px; border: 1px solid #ddd;">Unstable</td>
        <td style="padding: 12px; border: 1px solid #ddd;">Fast</td>
      </tr>
      <tr style="background-color: #f9f9f9;">
        <td style="padding: 12px; border: 1px solid #ddd;">LSGAN</td>
        <td style="padding: 12px; border: 1px solid #ddd;">52.48</td>
        <td style="padding: 12px; border: 1px solid #ddd;">More Stable</td>
        <td style="padding: 12px; border: 1px solid #ddd;">Fast</td>
      </tr>
      <tr>
        <td style="padding: 12px; border: 1px solid #ddd;">WGAN-GP</td>
        <td style="padding: 12px; border: 1px solid #ddd;">33.07</td>
        <td style="padding: 12px; border: 1px solid #ddd;">Stable</td>
        <td style="padding: 12px; border: 1px solid #ddd;">Fast</td>
      </tr>
      <tr style="background-color: #f9f9f9;">
        <td style="padding: 12px; border: 1px solid #ddd;">DDPM</td>
        <td style="padding: 12px; border: 1px solid #ddd;">34.73</td>
        <td style="padding: 12px; border: 1px solid #ddd;">Stable</td>
        <td style="padding: 12px; border: 1px solid #ddd;">Slow (1000 steps)</td>
      </tr>
      <tr>
        <td style="padding: 12px; border: 1px solid #ddd;">DDIM</td>
        <td style="padding: 12px; border: 1px solid #ddd;">38.32</td>
        <td style="padding: 12px; border: 1px solid #ddd;">Stable</td>
        <td style="padding: 12px; border: 1px solid #ddd;">Medium (100 steps)</td>
      </tr>
    </tbody>
  </table>
</div>

**Key Findings:**
- WGAN-GP emerged as the best-performing model with an FID score of 33.07
- Diffusion models (DDPM) produced comparable quality to WGAN-GP but required significantly more sampling steps
- VAEs with β-annealing showed improved sample quality, though quantitative comparison was not performed
- Training stability improved considerably from Vanilla GAN to LSGAN to WGAN-GP

---

### 6. Key Contributions

This independent research project was completed as an individual endeavor, with significant technical contributions including:

- Implementation and training of three different GAN architectures from scratch with custom ResBlock components and loss functions
- Development and training of the gradient penalty mechanism for WGAN-GP to enforce the 1-Lipschitz constraint
- Building and training VAEs with exploration of latent space dimensions and implementation of β-annealing for improved sample quality
- Implementation of both DDPM and DDIM sampling strategies for diffusion models
- Evaluation and analysis of different generative architectures on the same dataset

---

### 7. Technologies & Skills Used

- **Languages & Frameworks**: Python, PyTorch, TensorFlow, NumPy, OpenCV, scikit-learn
- **Deep Learning**: Generative Adversarial Networks, Variational Autoencoders, Diffusion Models, Model Training, Hyperparameter Tuning
- **Loss Functions**: Adversarial Loss, Reconstruction Loss, KL Divergence, Gradient Penalty
- **Computer Vision**: Image Synthesis, Image Generation, Feature Visualization

---

### 8. Project Repository

[GenVision](https://github.com/Srecharan/GenVision.git)