# Satellite-Super-Resolution-4-Enhancement-via-SRGAN


A deep learning project that enhances low-resolution satellite images (24×24) into high-resolution images (96×96) using a Generative Adversarial Network (GAN) architecture.
The model performs 4× super-resolution and improves visual quality and spatial detail compared to traditional interpolation methods.

📌 Project Overview

Satellite images often suffer from limited resolution due to sensor limitations and transmission constraints.
This project builds a deep learning pipeline to reconstruct high-resolution images from low-resolution inputs using an ESRGAN-inspired architecture.

The system uses:

RRDB Generator to reconstruct detailed images

PatchGAN Discriminator to enforce realistic textures

Perceptual + Adversarial + L1 Loss for high quality outputs

The model is trained on the UC Merced Land Use Dataset, which contains aerial images across 21 terrain classes.

🚀 Features

4× Satellite Image Super-Resolution

GAN-based architecture

ESRGAN-inspired RRDB Generator

PatchGAN Discriminator

VGG19 Perceptual Loss

Automatic GPU / CPU detection

PSNR and SSIM evaluation metrics

Training visualization and result saving

Single image inference support

Dataset streaming from HuggingFace (no manual download)

🧠 Model Architecture
Generator

The generator uses Residual-in-Residual Dense Blocks (RRDB).

Key characteristics:

16 RRDB blocks

Dense connections

Residual scaling

PixelShuffle upsampling

~16.7M parameters

This architecture allows the model to reconstruct fine spatial details and textures.

Discriminator

A PatchGAN discriminator evaluates small patches of the image instead of the entire image.

Benefits:

Improves local texture realism

Stabilizes GAN training

Encourages sharper outputs

Total parameters: ~5.2M

📊 Loss Function

Training combines three loss functions:

Perceptual Loss (VGG19) – compares deep image features
Adversarial Loss – encourages realistic outputs
L1 Pixel Loss – preserves ground truth similarity

Loss weighting:

Perceptual Loss → 1.0
Adversarial Loss → 1e-3
Pixel Loss → 1e-2

📂 Dataset

UC Merced Land Use Dataset

2,100 aerial images

21 land categories

256×256 resolution

0.3 m spatial resolution

Examples of categories:

Forest

Airport

Freeway

Agricultural

Beach

Harbor

Residential

Buildings

Dataset is streamed directly from HuggingFace.
