# MonetAI Architecture Documentation

This document provides detailed information about the MonetAI project architecture, implementation details, and design decisions.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Data Pipeline](#data-pipeline)
- [Training Process](#training-process)
- [Implementation Details](#implementation-details)

## 🎯 Overview

MonetAI implements CycleGAN (Cycle-Consistent Adversarial Networks) for unpaired image-to-image translation, specifically transforming photographs into Monet-style paintings. The project emphasizes clean code architecture, modularity, and educational value.

### Key Components

1. **CycleGAN Model**: Core GAN architecture with cycle consistency
2. **Data Pipeline**: TFRecord-based data loading with augmentation
3. **Training Framework**: Modular training loop with checkpointing
4. **Evaluation Suite**: Quality metrics and FID score calculation
5. **Utility Functions**: Visualization, logging, and helper functions

## 📁 Project Structure

```
MonetAI/
├── src/monetai/              # Main package
│   ├── __init__.py          # Package initialization
│   ├── models.py            # CycleGAN implementation
│   ├── data.py              # Data loading and preprocessing
│   └── utils.py             # Utility functions
├── scripts/                  # Executable scripts
│   ├── train.py             # Training script
│   ├── generate.py          # Image generation script
│   └── evaluate.py          # Model evaluation script
├── notebooks/                # Jupyter notebooks
│   └── cyclegan_demo.ipynb  # Interactive demo
├── config/                   # Configuration files
│   └── config.yaml          # Training and model configuration
├── docs/                     # Documentation
│   └── ARCHITECTURE.md      # This file
├── tests/                    # Unit tests (to be implemented)
├── checkpoints/              # Model checkpoints (gitignored)
├── logs/                     # Training logs (gitignored)
├── outputs/                  # Generated outputs (gitignored)
└── datasets/                 # Dataset files (gitignored)
```

## 🏗 Model Architecture

### CycleGAN Components

The CycleGAN consists of four main components:

#### Generators (G and F)
- **Generator G**: Photo → Monet (X → Y)
- **Generator F**: Monet → Photo (Y → X)
- **Architecture**: U-Net with instance normalization
- **Features**: Skip connections, residual blocks

```python
# U-Net Generator Architecture
Input (256x256x3)
    ↓ Encoder (Downsampling)
    Conv2D + InstanceNorm + LeakyReLU
    ↓ (128x128x64)
    Conv2D + InstanceNorm + LeakyReLU  
    ↓ (64x64x128)
    Conv2D + InstanceNorm + LeakyReLU
    ↓ (32x32x256)
    # ... (bottleneck)
    ↑ Decoder (Upsampling with Skip Connections)
    Conv2DTranspose + InstanceNorm + ReLU + Concatenate
    ↑ (64x64x128)
    Conv2DTranspose + InstanceNorm + ReLU + Concatenate
    ↑ (128x128x64)
    Conv2DTranspose + Tanh
Output (256x256x3)
```

#### Discriminators (D_X and D_Y)
- **Discriminator D_X**: Distinguishes real photos from fake photos
- **Discriminator D_Y**: Distinguishes real Monet from fake Monet
- **Architecture**: PatchGAN discriminator
- **Features**: Instance normalization, no target conditioning

```python
# PatchGAN Discriminator Architecture
Input (256x256x3)
    ↓
    Conv2D + LeakyReLU (no normalization for first layer)
    ↓ (128x128x64)
    Conv2D + InstanceNorm + LeakyReLU
    ↓ (64x64x128)
    Conv2D + InstanceNorm + LeakyReLU
    ↓ (32x32x256)
    Conv2D + InstanceNorm + LeakyReLU
    ↓ (16x16x512)
    Conv2D (final classification)
Output (16x16x1)
```

### Loss Functions

#### 1. Adversarial Loss
```python
L_GAN(G, D_Y, X, Y) = E_y[log D_Y(y)] + E_x[log(1 - D_Y(G(x)))]
```

#### 2. Cycle Consistency Loss
```python
L_cyc(G, F) = E_x[||F(G(x)) - x||_1] + E_y[||G(F(y)) - y||_1]
```

#### 3. Identity Loss
```python
L_identity(G, F) = E_y[||G(y) - y||_1] + E_x[||F(x) - x||_1]
```

#### 4. Total Loss
```python
L(G, F, D_X, D_Y) = L_GAN(G, D_Y) + L_GAN(F, D_X) 
                   + λ_cyc * L_cyc(G, F) + λ_id * L_identity(G, F)
```

Where:
- `λ_cyc = 10.0` (cycle consistency weight)
- `λ_id = 5.0` (identity loss weight)

## 📊 Data Pipeline

### TFRecord Format
The project uses TensorFlow's TFRecord format for efficient data loading:

```python
# TFRecord structure
{
    'image': tf.io.FixedLenFeature([], tf.string),      # JPEG encoded image
    'target': tf.io.FixedLenFeature([], tf.string),     # Label/category
    'image_name': tf.io.FixedLenFeature([], tf.string), # Original filename
}
```

### Data Preprocessing Pipeline

1. **Loading**: Parse TFRecord files
2. **Decoding**: JPEG → RGB tensor
3. **Resizing**: Scale to 256×256
4. **Normalization**: [0,1] → [-1,1]
5. **Augmentation**: Random jitter and flip (training only)

```python
# Training augmentation pipeline
def random_jitter(image):
    # Resize to 286x286
    image = tf.image.resize(image, [286, 286])
    # Random crop to 256x256
    image = tf.image.random_crop(image, [256, 256, 3])
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    return image
```

### Dataset Organization
```
datasets/
├── photo_tfrec/
│   ├── trainA/          # Training photos
│   └── testA/           # Test photos  
└── monet_tfrec/
    ├── trainB/          # Training Monet paintings
    └── testB/           # Test Monet paintings
```

## 🔄 Training Process

### Training Loop Architecture

```python
@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        fake_y = generator_g(real_x, training=True)
        fake_x = generator_f(real_y, training=True)
        
        # Cycle reconstruction
        cycled_x = generator_f(fake_y, training=True)
        cycled_y = generator_g(fake_x, training=True)
        
        # Identity mapping
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)
        
        # Calculate losses
        # ... (loss calculations)
    
    # Calculate and apply gradients
    # ... (gradient updates)
```

### Optimization Strategy

- **Optimizer**: Adam with β₁=0.5, β₂=0.999
- **Learning Rate**: 2×10⁻⁴ with linear decay
- **Batch Size**: 1 (following original paper)
- **Training Schedule**: 50-200 epochs

### Checkpoint Management

```python
# Checkpoint structure
checkpoint = tf.train.Checkpoint(
    generator_g=generator_g,
    generator_f=generator_f,
    discriminator_x=discriminator_x,
    discriminator_y=discriminator_y,
    gen_g_optimizer=gen_g_optimizer,
    gen_f_optimizer=gen_f_optimizer,
    disc_x_optimizer=disc_x_optimizer,
    disc_y_optimizer=disc_y_optimizer
)
```

## 🔧 Implementation Details

### Instance Normalization
Unlike batch normalization, instance normalization normalizes across spatial dimensions for each sample independently:

```python
# Instance normalization for style transfer
mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
var = tf.reduce_mean(tf.square(x - mean), axis=[1, 2], keepdims=True)
x = (x - mean) / tf.sqrt(var + epsilon)
```

### Memory Optimization
- **Gradient Tapes**: Use `persistent=True` for multiple gradient calculations
- **Data Pipeline**: Prefetching and parallel processing
- **Model Compilation**: `@tf.function` for graph optimization

### Numerical Stability
- **Label Smoothing**: Soft labels for better training stability  
- **Gradient Clipping**: Prevent exploding gradients
- **Learning Rate Scheduling**: Linear decay after warmup

### Configuration Management
The project uses YAML configuration files for easy parameter tuning:

```yaml
model:
  image_size: 256
  lambda_cycle: 10.0
  lambda_identity: 5.0
  learning_rate: 2e-4

training:
  epochs: 50
  batch_size: 1
  checkpoint_freq: 10
```

## 📈 Evaluation Metrics

### Implemented Metrics
1. **Generator/Discriminator Losses**: Training stability indicators
2. **Image Quality Metrics**: Brightness, contrast, color diversity
3. **Visual Inspection**: Generated sample comparisons

### Future Metrics (Extensible)
1. **FID Score**: Fréchet Inception Distance
2. **LPIPS**: Learned Perceptual Image Patch Similarity
3. **SSIM**: Structural Similarity Index

## 🚀 Design Principles

### Modularity
- **Separation of Concerns**: Data, model, training, evaluation
- **Plugin Architecture**: Easy to swap components
- **Configuration-Driven**: Minimal code changes for experiments

### Extensibility
- **Base Classes**: Easy to extend for new model variants
- **Abstract Interfaces**: Support for different data formats
- **Callback System**: Custom training hooks

### Maintainability
- **Type Hints**: Better code documentation and IDE support
- **Comprehensive Logging**: Detailed progress tracking
- **Unit Tests**: Ensure code reliability (to be implemented)

### Performance
- **TensorFlow Optimizations**: Graph compilation, mixed precision
- **Efficient Data Loading**: TFRecord format, parallel processing
- **Memory Management**: Gradient checkpointing, model pruning

## 🎓 Educational Value

This implementation prioritizes:
- **Clear Code Structure**: Easy to understand and modify
- **Comprehensive Comments**: Detailed explanations
- **Progressive Complexity**: Build understanding step by step
- **Best Practices**: Industry-standard patterns and techniques

The architecture serves as both a functional implementation and a learning resource for understanding CycleGAN and modern deep learning project organization.
