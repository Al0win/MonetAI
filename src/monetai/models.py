"""
CycleGAN Model Implementation

This module contains the CycleGAN model architecture including generators,
discriminators, and the main training loop.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow_examples.models.pix2pix import pix2pix
import os
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class CycleGAN:
    """
    CycleGAN implementation for unpaired image-to-image translation.
    
    This class implements the complete CycleGAN architecture with two generators
    (G: X->Y, F: Y->X) and two discriminators (D_X, D_Y) for bidirectional
    image translation without paired training data.
    """
    
    def __init__(
        self,
        img_height: int = 256,
        img_width: int = 256,
        channels: int = 3,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        learning_rate: float = 2e-4
    ):
        """
        Initialize CycleGAN model.
        
        Args:
            img_height: Height of input images
            img_width: Width of input images  
            channels: Number of image channels (3 for RGB)
            lambda_cycle: Weight for cycle consistency loss
            lambda_identity: Weight for identity loss
            learning_rate: Learning rate for optimizers
        """
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # Initialize models
        self.generator_g = pix2pix.unet_generator(
            channels, norm_type='instancenorm'
        )  # Photo -> Monet
        self.generator_f = pix2pix.unet_generator(
            channels, norm_type='instancenorm'
        )  # Monet -> Photo
        
        self.discriminator_x = pix2pix.discriminator(
            norm_type='instancenorm', target=False
        )  # Discriminates photos
        self.discriminator_y = pix2pix.discriminator(
            norm_type='instancenorm', target=False
        )  # Discriminates Monet paintings
        
        # Initialize optimizers
        self.gen_g_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.gen_f_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.disc_x_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.disc_y_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        
        # Initialize loss function
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        # Metrics
        self.gen_g_loss_metric = tf.keras.metrics.Mean()
        self.gen_f_loss_metric = tf.keras.metrics.Mean()
        self.disc_x_loss_metric = tf.keras.metrics.Mean()
        self.disc_y_loss_metric = tf.keras.metrics.Mean()
    
    def discriminator_loss(self, real: tf.Tensor, generated: tf.Tensor) -> tf.Tensor:
        """Calculate discriminator loss."""
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        return (real_loss + generated_loss) * 0.5
    
    def generator_loss(self, generated: tf.Tensor) -> tf.Tensor:
        """Calculate generator loss."""
        return self.loss_obj(tf.ones_like(generated), generated)
    
    def cycle_loss(self, real_image: tf.Tensor, cycled_image: tf.Tensor) -> tf.Tensor:
        """Calculate cycle consistency loss."""
        return self.lambda_cycle * tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    def identity_loss(self, real_image: tf.Tensor, same_image: tf.Tensor) -> tf.Tensor:
        """Calculate identity loss."""
        return self.lambda_identity * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))
    
    @tf.function
    def train_step(self, real_x: tf.Tensor, real_y: tf.Tensor):
        """
        Perform one training step.
        
        Args:
            real_x: Batch of real photos
            real_y: Batch of real Monet paintings
        """
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)
            
            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)
            
            # Identity mapping
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)
            
            # Discriminator outputs
            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)
            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)
            
            # Calculate losses
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)
            
            total_cycle_loss = (
                self.cycle_loss(real_x, cycled_x) + 
                self.cycle_loss(real_y, cycled_y)
            )
            
            # Total generator losses
            total_gen_g_loss = (
                gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            )
            total_gen_f_loss = (
                gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)
            )
            
            # Discriminator losses
            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
        
        # Calculate gradients
        gen_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        gen_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)
        disc_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        disc_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)
        
        # Apply gradients
        self.gen_g_optimizer.apply_gradients(zip(gen_g_gradients, self.generator_g.trainable_variables))
        self.gen_f_optimizer.apply_gradients(zip(gen_f_gradients, self.generator_f.trainable_variables))
        self.disc_x_optimizer.apply_gradients(zip(disc_x_gradients, self.discriminator_x.trainable_variables))
        self.disc_y_optimizer.apply_gradients(zip(disc_y_gradients, self.discriminator_y.trainable_variables))
        
        # Update metrics
        self.gen_g_loss_metric.update_state(total_gen_g_loss)
        self.gen_f_loss_metric.update_state(total_gen_f_loss)
        self.disc_x_loss_metric.update_state(disc_x_loss)
        self.disc_y_loss_metric.update_state(disc_y_loss)
    
    def generate_images(self, test_input: tf.Tensor, save_path: Optional[str] = None):
        """
        Generate and display transformed images.
        
        Args:
            test_input: Input images to transform
            save_path: Optional path to save the generated images
        """
        prediction = self.generator_g(test_input, training=False)
        
        plt.figure(figsize=(12, 6))
        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Generated Image']
        
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def save_model(self, checkpoint_path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = tf.train.Checkpoint(
            generator_g=self.generator_g,
            generator_f=self.generator_f,
            discriminator_x=self.discriminator_x,
            discriminator_y=self.discriminator_y,
            gen_g_optimizer=self.gen_g_optimizer,
            gen_f_optimizer=self.gen_f_optimizer,
            disc_x_optimizer=self.disc_x_optimizer,
            disc_y_optimizer=self.disc_y_optimizer
        )
        
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_path, max_to_keep=5
        )
        checkpoint_manager.save()
        
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = tf.train.Checkpoint(
            generator_g=self.generator_g,
            generator_f=self.generator_f,
            discriminator_x=self.discriminator_x,
            discriminator_y=self.discriminator_y,
            gen_g_optimizer=self.gen_g_optimizer,
            gen_f_optimizer=self.gen_f_optimizer,
            disc_x_optimizer=self.disc_x_optimizer,
            disc_y_optimizer=self.disc_y_optimizer
        )
        
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_path, max_to_keep=5
        )
        
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f'Restored from {checkpoint_manager.latest_checkpoint}')
            return True
        else:
            print('No checkpoint found')
            return False
    
    def transform(self, image: tf.Tensor) -> tf.Tensor:
        """
        Transform a single image from photo to Monet style.
        
        Args:
            image: Input image tensor
            
        Returns:
            Transformed image tensor
        """
        return self.generator_g(image, training=False)
