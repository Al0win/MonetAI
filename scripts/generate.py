"""
Generate images using trained CycleGAN model.

This script loads a trained model and generates Monet-style images
from input photographs.
"""

import argparse
import os
import tensorflow as tf
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monetai.models import CycleGAN
from src.monetai.data import load_image_dataset
from src.monetai.utils import setup_gpu, save_generated_images, create_image_zip


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate Monet-style images using trained CycleGAN')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing input photos')
    parser.add_argument('--output-dir', type=str, default='generated_images',
                       help='Directory to save generated images')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for generation')
    parser.add_argument('--image-size', type=int, default=256,
                       help='Image size for processing')
    parser.add_argument('--create-zip', action='store_true',
                       help='Create zip file of generated images')
    parser.add_argument('--zip-name', type=str, default='monet_generated.zip',
                       help='Name for zip file')
    
    return parser.parse_args()


def main():
    """Main generation function."""
    args = parse_args()
    
    # Setup GPU
    setup_gpu()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print("Initializing CycleGAN model...")
    model = CycleGAN(
        img_height=args.image_size,
        img_width=args.image_size
    )
    
    # Load model checkpoint
    print(f"Loading model from: {args.model_path}")
    if not model.load_model(args.model_path):
        print("Error: Could not load model checkpoint")
        return
    
    print("✓ Model loaded successfully")
    
    # Load input dataset
    print(f"Loading images from: {args.input_dir}")
    try:
        dataset = load_image_dataset(
            args.input_dir,
            img_height=args.image_size,
            img_width=args.image_size,
            batch_size=args.batch_size
        )
        print("✓ Input images loaded")
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # Generate images
    print(f"Generating Monet-style images...")
    print(f"Output directory: {args.output_dir}")
    
    try:
        num_generated = save_generated_images(
            dataset=dataset,
            generator=model.generator_g,
            output_dir=args.output_dir,
            prefix="monet"
        )
        print(f"✓ Generated {num_generated} images successfully")
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    
    # Create zip file if requested
    if args.create_zip:
        print(f"Creating zip file: {args.zip_name}")
        try:
            create_image_zip(args.output_dir, args.zip_name)
            print("✓ Zip file created successfully")
        except Exception as e:
            print(f"Error creating zip file: {e}")
    
    print("\n🎨 Image generation completed!")
    print(f"📁 Generated images saved to: {args.output_dir}")
    if args.create_zip:
        print(f"📦 Zip file: {args.zip_name}")


if __name__ == '__main__':
    main()
