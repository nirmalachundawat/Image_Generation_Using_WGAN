# Image Generation using DCGAN & WGAN-GP

## Overview
This project implements **image generation using Generative Adversarial Networks (GANs), specifically:
* **DCGAN (Deep Convolutional GAN)**
* **WGAN-GP (Wasserstein GAN with Gradient Penalty)**
The model is trained on the **CelebA dataset** to generate realistic human face images.
---
## Key Concepts
### 1. Generative Adversarial Network (GAN)
A GAN consists of two neural networks:
* **Generator (G):** Generates fake images from random noise
* **Discriminator (D):** Distinguishes real images from fake ones
They are trained in a **minimax game**:
* Generator tries to fool the discriminator
* Discriminator tries to correctly classify real vs fake
---
## Architecture
### Generator (DCGAN-based)

* Input: Random noise vector (size = 100)
* Fully connected layer → reshape to 4×4×512
* Series of **Conv2DTranspose (deconvolution)** layers
* Batch Normalization + LeakyReLU
* Output: 64×64×3 image (tanh activation)

### 🔹 Discriminator (DCGAN-based)

* Input: 64×64×3 image
* Series of **Conv2D layers**
* LeakyReLU activations
* Downsampling via strides
* Output: Single scalar (real/fake score)
---

## Data Pipeline
* Dataset: CelebA images
* Preprocessing:
  * Resize to **64×64**
  * Normalize pixel values to **[-1, 1]**
* TensorFlow `tf.data` pipeline:
  * Shuffle → Batch → Prefetch
---
## Training Strategy
### 1. DCGAN Loss (Binary Cross Entropy)

#### Discriminator (Critic) Loss:
L_D = -E[log(D(x))] - E[log(1 - D(G(z)))]

#### Generator Loss:
L_G = -E[log(D(G(z)))]

### 2. WGAN Loss (Wasserstein Distance)

Instead of probabilities, discriminator outputs **real-valued scores**.

#### Discriminator (Critic) Loss:
L_D = E[D(fake)] - E[D(real)]

#### Generator Loss:
L_G = -E[D(fake)]

### 3. Gradient Penalty (WGAN-GP)

To enforce Lipschitz constraint:

L_GP = λ * E[(||∇x̂ D(x̂)||₂ - 1)²]

x̂ = εx + (1 - ε)G(z)
ε ~ Uniform(0,1)

### Final Critic Loss (WGAN-GP)

L_D_total = E[D(fake)] - E[D(real)] + λ * E[(||∇x̂ D(x̂)||₂ - 1)²]

---

## Training Workflow

For each epoch:
1. Train **Discriminator (Critic)** multiple times (`N_CRITIC`)
2. Train **Generator** once
3. Apply **learning rate decay**
4. Save checkpoints periodically
5. Generate sample images
---

## Checkpointing

* Uses TensorFlow checkpoint system
* Saves:
  * Generator weights
  * Discriminator weights
  * Optimizer states
* Automatically restores the latest checkpoint if available
---

##  Output
* Generated images saved during training
* Visualization using Matplotlib
* Fixed noise vector used for consistent progress tracking
---

### Frechet Inception Distance (FID)
FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2 * sqrt(Σ_real * Σ_fake))
where: 
μ_real  = mean of real image features  
μ_fake  = mean of generated image features  

Σ_real  = covariance of real features  
Σ_fake  = covariance of generated features  

Tr(.)   = trace of matrix  
sqrt(.) = matrix square root  

## Observations
* **DCGAN** may suffer from instability and mode collapse
* **WGAN-GP** provides:

  * More stable training
  * Better convergence
  * Higher quality images
---

##  Conclusion
This project demonstrates a complete pipeline for **image generation using GANs**, progressing from standard DCGAN to a more stable **WGAN-GP implementation**, highlighting practical differences in training stability and output quality.
---
