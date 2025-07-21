"""
Training script for CycleGAN model.

This script handles the complete training pipeline including data loading,
model initialization, training loop, and checkpoint management.
"""

import argparse
import os
import time
import yaml
from pathlib import Path
import tensorflow as tf
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monetai.models import CycleGAN
from src.monetai.data import DataLoader
from src.monetai.utils import setup_gpu, log_training_progress, TrainingLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CycleGAN for Photo to Monet style transfer')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory to save training logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Data paths
    parser.add_argument('--photo-train-dir', type=str, 
                       default='datasets/photo_jpg',
                       help='Directory containing training photos')
    parser.add_argument('--monet-train-dir', type=str,
                       default='datasets/monet_jpg', 
                       help='Directory containing training Monet paintings')
    parser.add_argument('--photo-test-dir', type=str,
                       default='datasets/photo_jpg',
                       help='Directory containing test photos')
    parser.add_argument('--monet-test-dir', type=str,
                       default='datasets/monet_jpg',
                       help='Directory containing test Monet paintings')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        print("Using default configuration")
        return {}


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    # Override config with command line arguments
    epochs = args.epochs or training_config.get('epochs', 50)
    batch_size = args.batch_size or model_config.get('batch_size', 1)
    learning_rate = args.learning_rate or model_config.get('learning_rate', 2e-4)
    
    # Setup GPU
    setup_gpu()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader(
        img_height=model_config.get('image_size', 256),
        img_width=model_config.get('image_size', 256),
        batch_size=batch_size,
        buffer_size=training_config.get('buffer_size', 1000)
    )
    
    print("Loading datasets...")
    try:
        train_photos, train_monet, test_photos, test_monet = data_loader.create_datasets(
            photo_train_dir=args.photo_train_dir,
            monet_train_dir=args.monet_train_dir,
            photo_test_dir=args.photo_test_dir if os.path.exists(args.photo_test_dir) else None,
            monet_test_dir=args.monet_test_dir if os.path.exists(args.monet_test_dir) else None
        )
        print("✓ Datasets loaded successfully")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure dataset paths are correct")
        return
    
    # Initialize model
    model = CycleGAN(
        img_height=model_config.get('image_size', 256),
        img_width=model_config.get('image_size', 256),
        lambda_cycle=model_config.get('lambda_cycle', 10.0),
        lambda_identity=model_config.get('lambda_identity', 5.0),
        learning_rate=learning_rate
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if model.load_model(args.resume):
            # Extract epoch from checkpoint path if possible
            try:
                start_epoch = int(args.resume.split('-')[-1])
                print(f"Resuming from epoch {start_epoch}")
            except:
                print("Could not determine start epoch from checkpoint name")
    
    # Initialize logger
    log_file = os.path.join(args.log_dir, f'training_{int(time.time())}.csv')
    logger = TrainingLogger(log_file)
    
    # Get sample images for visualization
    sample_photo = next(iter(train_photos))
    sample_monet = next(iter(train_monet))
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Logs: {log_file}")
    print("=" * 60)
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        
        # Reset metrics
        model.gen_g_loss_metric.reset_state()
        model.gen_f_loss_metric.reset_state()
        model.disc_x_loss_metric.reset_state()
        model.disc_y_loss_metric.reset_state()
        
        # Training step
        step_count = 0
        for image_x, image_y in tf.data.Dataset.zip((train_photos, train_monet)):
            model.train_step(image_x, image_y)
            
            if step_count % training_config.get('log_freq', 100) == 0:
                print('.', end='', flush=True)
            step_count += 1
        
        epoch_time = time.time() - start_time
        
        # Collect losses
        losses = {
            'gen_g_loss': model.gen_g_loss_metric.result().numpy(),
            'gen_f_loss': model.gen_f_loss_metric.result().numpy(),
            'disc_x_loss': model.disc_x_loss_metric.result().numpy(),
            'disc_y_loss': model.disc_y_loss_metric.result().numpy()
        }
        
        # Log progress
        log_training_progress(epoch, epochs, start_time, losses)
        logger.log(epoch, losses, epoch_time)
        
        # Generate sample images
        if epoch % training_config.get('sample_freq', 5) == 0:
            print("Generating sample images...")
            sample_output_path = os.path.join(args.log_dir, f'sample_epoch_{epoch:03d}.png')
            model.generate_images(sample_photo, save_path=sample_output_path)
        
        # Save checkpoint
        if (epoch + 1) % training_config.get('checkpoint_freq', 10) == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint-{epoch:03d}')
            model.save_model(checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Final checkpoint
    final_checkpoint = os.path.join(args.checkpoint_dir, 'final_model')
    model.save_model(final_checkpoint)
    print(f"✓ Final model saved: {final_checkpoint}")
    
    # Plot training curves
    print("Generating training plots...")
    plot_path = os.path.join(args.log_dir, 'training_curves.png')
    logger.plot_losses(save_path=plot_path)
    
    # Generate final test images if test data available
    if test_photos is not None:
        print("Generating test results...")
        test_output_dir = os.path.join(args.log_dir, 'test_results')
        os.makedirs(test_output_dir, exist_ok=True)
        
        for i, test_batch in enumerate(test_photos.take(5)):
            output_path = os.path.join(test_output_dir, f'test_result_{i:03d}.png')
            model.generate_images(test_batch, save_path=output_path)
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Logs saved to: {args.log_dir}")
    print(f"💾 Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
