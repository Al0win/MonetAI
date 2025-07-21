"""
Command line interface for MonetAI package.

This module provides CLI access to training, generation, and evaluation scripts.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='MonetAI: CycleGAN for Photo to Monet Style Transfer',
        prog='monetai'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train CycleGAN model')
    train_parser.add_argument('--config', type=str, default='config/config.yaml')
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch-size', type=int, default=1)
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate Monet-style images')
    gen_parser.add_argument('--model-path', type=str, required=True)
    gen_parser.add_argument('--input-dir', type=str, required=True)
    gen_parser.add_argument('--output-dir', type=str, default='generated_images')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--model-path', type=str, required=True)
    eval_parser.add_argument('--test-photos', type=str, required=True)
    eval_parser.add_argument('--real-monet', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from scripts.train import main as train_main
        # Override sys.argv for the train script
        sys.argv = ['train.py'] + [f'--{k.replace("_", "-")}={v}' for k, v in vars(args).items() if k != 'command' and v is not None]
        train_main()
    elif args.command == 'generate':
        from scripts.generate import main as generate_main
        sys.argv = ['generate.py'] + [f'--{k.replace("_", "-")}={v}' for k, v in vars(args).items() if k != 'command' and v is not None]
        generate_main()
    elif args.command == 'evaluate':
        from scripts.evaluate import main as evaluate_main
        sys.argv = ['evaluate.py'] + [f'--{k.replace("_", "-")}={v}' for k, v in vars(args).items() if k != 'command' and v is not None]
        evaluate_main()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
