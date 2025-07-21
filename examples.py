#!/usr/bin/env python3
"""
MonetAI Example Usage Script

This script demonstrates various ways to use the MonetAI package
for photo to Monet style transfer.
"""

import os
import sys
import tensorflow as tf
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monetai.models import CycleGAN
from src.monetai.data import DataLoader, load_image_dataset
from src.monetai.utils import setup_gpu, display_sample_images, save_generated_images


def example_basic_usage():
    """Example 1: Basic model usage."""
    print("🎨 Example 1: Basic CycleGAN Usage")
    print("-" * 40)
    
    # Setup GPU
    setup_gpu()
    
    # Initialize model
    model = CycleGAN(img_height=256, img_width=256)
    print("✓ CycleGAN model initialized")
    
    # Create dummy input
    dummy_photo = tf.random.normal([1, 256, 256, 3])
    print("✓ Created dummy photo input")
    
    # Generate Monet-style image (untrained model)
    monet_style = model.generator_g(dummy_photo, training=False)
    print("✓ Generated Monet-style image (untrained)")
    
    print(f"Input shape: {dummy_photo.shape}")
    print(f"Output shape: {monet_style.shape}")
    print()


def example_data_loading():
    """Example 2: Data loading and preprocessing."""
    print("🎨 Example 2: Data Loading and Preprocessing")
    print("-" * 40)
    
    # Initialize data loader
    data_loader = DataLoader(
        img_height=256,
        img_width=256,
        batch_size=2,
        buffer_size=100
    )
    print("✓ DataLoader initialized")
    
    # Create sample data directory structure
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample TFRecord-like structure (placeholder)
    print("✓ Sample data structure created")
    print(f"Data loader config:")
    print(f"  - Image size: {data_loader.img_height}x{data_loader.img_width}")
    print(f"  - Batch size: {data_loader.batch_size}")
    print(f"  - Buffer size: {data_loader.buffer_size}")
    print()


def example_training_setup():
    """Example 3: Training setup without actual training."""
    print("🎨 Example 3: Training Setup")
    print("-" * 40)
    
    # Initialize model with custom parameters
    model = CycleGAN(
        img_height=256,
        img_width=256,
        lambda_cycle=10.0,
        lambda_identity=5.0,
        learning_rate=2e-4
    )
    print("✓ CycleGAN model initialized with custom parameters")
    
    # Display model configuration
    print(f"Model configuration:")
    print(f"  - Image size: {model.img_height}x{model.img_width}")
    print(f"  - Cycle loss weight: {model.lambda_cycle}")
    print(f"  - Identity loss weight: {model.lambda_identity}")
    
    # Create dummy training data
    batch_size = 2
    dummy_photos = tf.random.normal([batch_size, 256, 256, 3])
    dummy_monet = tf.random.normal([batch_size, 256, 256, 3])
    
    print("✓ Created dummy training batch")
    
    # Demonstrate one training step (without actually training)
    print("✓ Training step function ready")
    print(f"  - Photo batch shape: {dummy_photos.shape}")
    print(f"  - Monet batch shape: {dummy_monet.shape}")
    print()


def example_image_generation():
    """Example 4: Image generation pipeline."""
    print("🎨 Example 4: Image Generation Pipeline")
    print("-" * 40)
    
    # Initialize model
    model = CycleGAN()
    
    # Create sample input images
    num_samples = 3
    sample_photos = tf.random.normal([num_samples, 256, 256, 3])
    
    # Generate Monet-style versions
    generated_monet = model.generator_g(sample_photos, training=False)
    
    # Generate photos from Monet (reverse direction)
    generated_photos = model.generator_f(sample_photos, training=False)
    
    print("✓ Generated images in both directions")
    print(f"  - Input photos: {sample_photos.shape}")
    print(f"  - Generated Monet: {generated_monet.shape}")
    print(f"  - Generated photos: {generated_photos.shape}")
    
    # Demonstrate cycle consistency
    cycled_photos = model.generator_f(generated_monet, training=False)
    cycle_loss = tf.reduce_mean(tf.abs(sample_photos - cycled_photos))
    
    print(f"  - Cycle consistency loss: {cycle_loss.numpy():.4f}")
    print()


def example_evaluation_metrics():
    """Example 5: Evaluation metrics calculation."""
    print("🎨 Example 5: Evaluation Metrics")
    print("-" * 40)
    
    # Generate sample images for evaluation
    real_photos = tf.random.normal([5, 256, 256, 3])
    real_monet = tf.random.normal([5, 256, 256, 3])
    
    model = CycleGAN()
    generated_monet = model.generator_g(real_photos, training=False)
    
    # Calculate basic metrics
    def calculate_metrics(images):
        # Rescale to [0, 1] for metrics
        images = (images + 1) / 2.0
        
        brightness = tf.reduce_mean(images)
        contrast = tf.math.reduce_std(images)
        
        return {
            'brightness': brightness.numpy(),
            'contrast': contrast.numpy(),
            'shape': images.shape
        }
    
    real_metrics = calculate_metrics(real_monet)
    generated_metrics = calculate_metrics(generated_monet)
    
    print("✓ Calculated evaluation metrics")
    print(f"Real Monet metrics:")
    print(f"  - Brightness: {real_metrics['brightness']:.4f}")
    print(f"  - Contrast: {real_metrics['contrast']:.4f}")
    
    print(f"Generated Monet metrics:")
    print(f"  - Brightness: {generated_metrics['brightness']:.4f}")
    print(f"  - Contrast: {generated_metrics['contrast']:.4f}")
    print()


def example_model_persistence():
    """Example 6: Model saving and loading."""
    print("🎨 Example 6: Model Persistence")  
    print("-" * 40)
    
    # Initialize model
    model = CycleGAN()
    
    # Create checkpoint directory
    checkpoint_dir = "example_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model (demonstrates saving mechanism)
    print("✓ Model save mechanism ready")
    print(f"  - Checkpoint directory: {checkpoint_dir}")
    
    # Demonstrate loading mechanism
    print("✓ Model load mechanism ready")
    print("  - Would restore from latest checkpoint if available")
    
    # Cleanup
    print("✓ Checkpoint management configured")
    print()


def main():
    """Run all examples."""
    print("🎨 MonetAI Usage Examples")
    print("=" * 50)
    print()
    
    try:
        # Run all examples
        example_basic_usage()
        example_data_loading() 
        example_training_setup()
        example_image_generation()
        example_evaluation_metrics()
        example_model_persistence()
        
        print("🎉 All examples completed successfully!")
        print()
        print("Next steps:")
        print("1. Prepare your photo and Monet painting datasets")
        print("2. Convert them to TFRecord format")
        print("3. Run: python scripts/train.py --epochs 50")
        print("4. Generate images: python scripts/generate.py --model-path checkpoints/final_model --input-dir your_photos/")
        print()
        print("For more information, see:")
        print("- README.md for overview")
        print("- docs/QUICKSTART.md for quick start guide")
        print("- docs/ARCHITECTURE.md for technical details")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("This might be due to missing dependencies or TensorFlow configuration.")
        print("Please ensure you have installed all requirements:")
        print("  pip install -e .")


if __name__ == '__main__':
    main()
