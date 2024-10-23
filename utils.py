import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

def load_data(batch_size=1):
    """
    Load and preprocess the Monet and photo datasets.
    Returns: TensorFlow dataset
    """
    # TODO: Implement data loading
    # 1. Load images from monet_jpg and photo_jpg directories
    # 2. Preprocess images (resize, normalize, etc.)
    # 3. Create tf.data.Dataset
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [256, 256])
        img = tf.cast(img, tf.float32)
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        return img

    # Load Monet images
    monet_files = glob.glob("datasets/monet_jpg/*.jpg")
    monet_ds = tf.data.Dataset.from_tensor_slices(monet_files)
    monet_ds = monet_ds.map(load_image, num_parallel_calls=AUTOTUNE)

    # Create batched dataset
    monet_ds = monet_ds.batch(batch_size)
    
    return monet_ds

def save_images(images, epoch, output_dir="outputs"):
    """
    Save generated images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for idx, image in enumerate(images):
        # Convert from [-1, 1] to [0, 255]
        image = ((image + 1) * 127.5).numpy().astype(np.uint8)
        im = Image.fromarray(image)
        im.save(os.path.join(output_dir, f"image_epoch_{epoch}_{idx}.jpg"))

def calculate_fid(real_images, generated_images):
    """
    Calculate the Fréchet Inception Distance between real and generated images.
    """
    # TODO: Implement FID calculation
    # This will be used for evaluation
    pass