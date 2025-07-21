"""
Data loading and preprocessing utilities for CycleGAN training.

This module handles loading image datasets, applying data augmentation,
and creating training/testing data pipelines.
"""

import tensorflow as tf
import os
from typing import Tuple, Optional, List


class DataLoader:
    """
    Data loader for CycleGAN training with image file support.
    
    Handles loading and preprocessing of photo and Monet painting datasets
    with appropriate data augmentation for training.
    """
    
    def __init__(
        self, 
        img_height: int = 256,
        img_width: int = 256,
        batch_size: int = 1,
        buffer_size: int = 1000
    ):
        """
        Initialize data loader.
        
        Args:
            img_height: Target image height
            img_width: Target image width
            batch_size: Training batch size
            buffer_size: Buffer size for dataset shuffling
        """
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
    def normalize(self, image: tf.Tensor) -> tf.Tensor:
        """Normalize image to [-1, 1] range."""
        return (image * 2.0) - 1.0
    
    def random_crop(self, image: tf.Tensor) -> tf.Tensor:
        """Randomly crop image to target size."""
        return tf.image.random_crop(image, size=[self.img_height, self.img_width, 3])
    
    def random_jitter(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply random jittering data augmentation.
        
        Args:
            image: Input image tensor
            
        Returns:
            Augmented image tensor
        """
        # Resize to 286x286
        image = tf.image.resize(
            image, [286, 286], 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        
        # Random crop to target size
        image = self.random_crop(image)
        
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        return image
    
    def preprocess_train(self, image: tf.Tensor) -> tf.Tensor:
        """
        Preprocess image for training.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        image = self.random_jitter(image)
        image = self.normalize(image)
        return image
    
    def preprocess_test(self, image: tf.Tensor) -> tf.Tensor:
        """
        Preprocess image for testing.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        image = self.normalize(image)
        return image
    
    def load_image_dataset(self, image_dir: str) -> tf.data.Dataset:
        """
        Load dataset from JPG/PNG image files in directory.
        
        Args:
            image_dir: Directory containing image files
            
        Returns:
            Dataset of preprocessed images
        """
        # Get list of image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(tf.io.gfile.glob(os.path.join(image_dir, ext)))
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in {image_dir}")
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        def load_and_preprocess_image(image_path):
            """Load and preprocess a single image."""
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image.set_shape([None, None, 3])  # Set shape to enable resize
            image = tf.image.resize(image, [self.img_height, self.img_width])
            image = tf.cast(image, tf.float32) / 255.0
            return image
        
        dataset = tf.data.Dataset.from_tensor_slices(image_files)
        dataset = dataset.map(
            load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset
    
    def prepare_dataset(
        self, 
        dataset: tf.data.Dataset, 
        training: bool = True
    ) -> tf.data.Dataset:
        """
        Prepare dataset for training or testing.
        
        Args:
            dataset: Input dataset
            training: Whether to apply training preprocessing
            
        Returns:
            Prepared dataset
        """
        if training:
            dataset = dataset.cache().map(
                self.preprocess_train,
                num_parallel_calls=tf.data.AUTOTUNE
            ).shuffle(self.buffer_size).batch(self.batch_size)
        else:
            dataset = dataset.map(
                self.preprocess_test,
                num_parallel_calls=tf.data.AUTOTUNE
            ).cache().batch(self.batch_size)
        
        return dataset.prefetch(tf.data.AUTOTUNE)
    
    def create_datasets(
        self,
        photo_train_dir: str,
        monet_train_dir: str,
        photo_test_dir: Optional[str] = None,
        monet_test_dir: Optional[str] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset], Optional[tf.data.Dataset]]:
        """
        Create training and testing datasets from image directories.
        
        Args:
            photo_train_dir: Directory with training photos
            monet_train_dir: Directory with training Monet paintings
            photo_test_dir: Directory with test photos (optional)
            monet_test_dir: Directory with test Monet paintings (optional)
            
        Returns:
            Tuple of (train_photos, train_monet, test_photos, test_monet)
        """
        # Load training datasets
        train_photos_raw = self.load_image_dataset(photo_train_dir)
        train_monet_raw = self.load_image_dataset(monet_train_dir)
        
        train_photos = self.prepare_dataset(train_photos_raw, training=True)
        train_monet = self.prepare_dataset(train_monet_raw, training=True)
        
        # Load test datasets if provided
        test_photos = None
        test_monet = None
        
        if photo_test_dir and os.path.exists(photo_test_dir):
            test_photos_raw = self.load_image_dataset(photo_test_dir)
            test_photos = self.prepare_dataset(test_photos_raw, training=False)
            
        if monet_test_dir and os.path.exists(monet_test_dir):
            test_monet_raw = self.load_image_dataset(monet_test_dir)
            test_monet = self.prepare_dataset(test_monet_raw, training=False)
        
        return train_photos, train_monet, test_photos, test_monet


def load_image_dataset(
    image_dir: str,
    img_height: int = 256,
    img_width: int = 256,
    batch_size: int = 1
) -> tf.data.Dataset:
    """
    Load images from directory for inference.
    
    Args:
        image_dir: Directory containing images
        img_height: Target image height
        img_width: Target image width
        batch_size: Batch size
        
    Returns:
        Dataset of preprocessed images
    """
    def preprocess_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [img_height, img_width])
        image = tf.cast(image, tf.float32) / 255.0
        image = (image * 2.0) - 1.0  # Normalize to [-1, 1]
        return image
    
    image_paths = tf.data.Dataset.list_files(os.path.join(image_dir, "*"))
    dataset = image_paths.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset
