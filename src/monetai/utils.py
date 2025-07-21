"""
Utility functions for MonetAI project.

This module contains helper functions for visualization, evaluation,
and general utility operations.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from typing import Optional, List, Tuple
import time
from PIL import Image


def save_generated_images(
    dataset: tf.data.Dataset,
    generator: tf.keras.Model,
    output_dir: str,
    prefix: str = "generated"
) -> int:
    """
    Generate and save images using the trained generator.
    
    Args:
        dataset: Input dataset
        generator: Trained generator model
        output_dir: Directory to save generated images
        prefix: Prefix for generated image names
        
    Returns:
        Number of images generated
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_count = 0
    for batch in dataset:
        generated_images = generator(batch, training=False)
        
        for img in generated_images:
            # Rescale to [0, 1] for saving
            img = (img + 1) / 2.0
            img = tf.clip_by_value(img, 0, 1)
            
            output_path = os.path.join(output_dir, f"{prefix}_{image_count:04d}.jpg")
            tf.keras.preprocessing.image.save_img(output_path, img.numpy())
            image_count += 1
    
    return image_count


def clear_directory(directory: str) -> None:
    """
    Clear all files in a directory.
    
    Args:
        directory: Directory path to clear
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)


def create_image_zip(source_dir: str, zip_filename: str) -> None:
    """
    Create a zip file containing all images from source directory.
    
    Args:
        source_dir: Directory containing images to zip
        zip_filename: Output zip filename
    """
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=file)
    
    print(f"Created zip file: {zip_filename}")


def display_sample_images(
    dataset: tf.data.Dataset,
    num_samples: int = 4,
    title: str = "Sample Images"
) -> None:
    """
    Display sample images from dataset.
    
    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples to display
        title: Plot title
    """
    samples = list(dataset.take(num_samples))
    
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 4))
    if len(samples) == 1:
        axes = [axes]
    
    for i, sample in enumerate(samples):
        image = sample[0] if isinstance(sample, tuple) else sample
        axes[i].imshow(image * 0.5 + 0.5)  # Rescale to [0,1] for display
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def compare_images(
    original: tf.Tensor,
    generated: tf.Tensor,
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Display original and generated images side by side.
    
    Args:
        original: Original image tensor
        generated: Generated image tensor
        titles: Custom titles for images
        save_path: Path to save comparison image
    """
    if titles is None:
        titles = ['Original', 'Generated']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original
    axes[0].imshow(original * 0.5 + 0.5)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    # Display generated
    axes[1].imshow(generated * 0.5 + 0.5)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def setup_gpu():
    """
    Configure GPU settings for optimal performance.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU")


def log_training_progress(
    epoch: int,
    total_epochs: int,
    start_time: float,
    losses: dict
) -> None:
    """
    Log training progress with losses and timing.
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        start_time: Start time of current epoch
        losses: Dictionary of loss values
    """
    elapsed_time = time.time() - start_time
    
    print(f"\nEpoch {epoch+1}/{total_epochs}")
    print(f"Time: {elapsed_time:.2f}s")
    
    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value:.4f}")
    
    print("-" * 50)


def create_model_summary(model: tf.keras.Model, save_path: Optional[str] = None) -> str:
    """
    Create and optionally save model architecture summary.
    
    Args:
        model: Keras model
        save_path: Path to save summary text file
        
    Returns:
        Model summary as string
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    summary_text = '\n'.join(summary_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary_text)
        print(f"Model summary saved to {save_path}")
    
    return summary_text


def validate_dataset_paths(*paths: str) -> bool:
    """
    Validate that all dataset paths exist.
    
    Args:
        paths: Dataset directory paths
        
    Returns:
        True if all paths exist
    """
    for path in paths:
        if not os.path.exists(path):
            print(f"Error: Path does not exist: {path}")
            return False
    return True


def calculate_dataset_stats(dataset: tf.data.Dataset) -> dict:
    """
    Calculate basic statistics for a dataset.
    
    Args:
        dataset: TensorFlow dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    total_samples = 0
    pixel_sum = 0
    pixel_count = 0
    
    for batch in dataset:
        if isinstance(batch, tuple):
            batch = batch[0]  # Extract images if tuple
        
        batch_size = tf.shape(batch)[0]
        total_samples += batch_size.numpy()
        
        pixel_sum += tf.reduce_sum(batch).numpy()
        pixel_count += tf.size(batch).numpy()
    
    mean_pixel_value = pixel_sum / pixel_count if pixel_count > 0 else 0
    
    return {
        'total_samples': total_samples,
        'mean_pixel_value': mean_pixel_value,
        'pixel_range': '[-1, 1]' if mean_pixel_value < 0 else '[0, 1]'
    }


class TrainingLogger:
    """Simple training logger for tracking metrics."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics = []
        
        # Create log file with headers
        with open(log_file, 'w') as f:
            f.write("epoch,gen_g_loss,gen_f_loss,disc_x_loss,disc_y_loss,time\n")
    
    def log(self, epoch: int, losses: dict, epoch_time: float):
        """Log metrics for one epoch."""
        entry = {
            'epoch': epoch,
            'time': epoch_time,
            **losses
        }
        self.metrics.append(entry)
        
        # Append to CSV file
        with open(self.log_file, 'a') as f:
            gen_g = losses.get('gen_g_loss', 0)
            gen_f = losses.get('gen_f_loss', 0)
            disc_x = losses.get('disc_x_loss', 0)
            disc_y = losses.get('disc_y_loss', 0)
            f.write(f"{epoch},{gen_g:.6f},{gen_f:.6f},{disc_x:.6f},{disc_y:.6f},{epoch_time:.2f}\n")
    
    def plot_losses(self, save_path: Optional[str] = None):
        """Plot training losses."""
        if not self.metrics:
            print("No metrics to plot")
            return
        
        epochs = [m['epoch'] for m in self.metrics]
        gen_losses = [m.get('gen_g_loss', 0) + m.get('gen_f_loss', 0) for m in self.metrics]
        disc_losses = [m.get('disc_x_loss', 0) + m.get('disc_y_loss', 0) for m in self.metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, gen_losses, label='Generator Loss')
        plt.plot(epochs, disc_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
