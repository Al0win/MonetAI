# MonetAI: Style Transfer with CycleGAN

<div align="center">

![MonetAI Banner](https://img.shields.io/badge/MonetAI-Style%20Transfer-blue?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange?logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Transform ordinary photographs into Monet-style paintings using CycleGAN*

</div>

## 🎨 Overview

MonetAI is an implementation of CycleGAN (Cycle-Consistent Adversarial Networks) for unpaired image-to-image translation. This project focuses on transforming regular photographs into paintings that mimic Claude Monet's distinctive artistic style. The model learns to capture the essence of Monet's impressionistic techniques without requiring paired training data.

### Key Highlights

- **Unpaired Learning**: No need for matching photo-painting pairs
- **Cycle Consistency**: Ensures meaningful transformations through cycle loss
- **Instance Normalization**: Better style transfer compared to batch normalization
- **Identity Loss**: Preserves content when input already matches target domain

## 🚀 Features

- ✨ **Photo-to-Monet Translation**: Convert photographs to Monet-style paintings
- 🔄 **Bidirectional Translation**: Support for both photo→painting and painting→photo
- 📊 **Comprehensive Evaluation**: FID score calculation for quality assessment
- 🎯 **Modular Design**: Clean, extensible codebase following best practices
- 📱 **Easy to Use**: Simple CLI interface and notebook examples
- 🔧 **Configurable**: Adjustable hyperparameters and training settings

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 2GB+ storage for datasets

## 🛠 Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Al0win/MonetAI.git
cd MonetAI

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Development Setup

```bash
# Create virtual environment
python -m venv monetai_env
source monetai_env/bin/activate  # On Windows: monetai_env\Scripts\activate

# Install in development mode with extra dependencies
pip install -e .[dev]
```

## 📖 Usage

### Command Line Interface

```bash
# Train the model
python scripts/train.py --epochs 50 --batch-size 1

# Generate images
python scripts/generate.py --model-path checkpoints/final_model --input-dir photos/ --output-dir monet_style/

# Evaluate model
python scripts/evaluate.py --model-path checkpoints/final_model --test-photos photos/ --real-monet monet_paintings/
```

### Jupyter Notebook

Explore the complete workflow in the interactive notebook:

```bash
jupyter notebook notebooks/cyclegan_demo.ipynb
```

### Python API

```python
from monetai import MonetGAN

# Initialize model
model = MonetGAN()

# Load pre-trained weights
model.load_weights('checkpoints/monet_cyclegan.h5')

# Transform image
monet_style_image = model.transform(your_photo)
```

## 🏗 Architecture

MonetAI implements the CycleGAN architecture with the following components:

### Generators
- **U-Net Architecture**: Encoder-decoder with skip connections
- **Instance Normalization**: Better for style transfer tasks
- **Residual Blocks**: Preserve fine details during transformation

### Discriminators
- **PatchGAN**: Discriminates on patch level for better texture details
- **Instance Normalization**: Consistent with generator normalization

### Loss Functions
- **Adversarial Loss**: Makes generated images indistinguishable from real ones
- **Cycle Consistency Loss**: Ensures F(G(x)) ≈ x and G(F(y)) ≈ y
- **Identity Loss**: Preserves images already in target domain


## 🎯 Dataset

The model is trained on:
- **Monet Paintings**: ~300 high-quality Monet artwork images
- **Landscape Photos**: ~7,000 natural landscape photographs
- **Image Size**: 256×256 pixels with data augmentation

Data augmentation includes:
- Random horizontal flipping
- Random cropping from 286×286 to 256×256
- Normalization to [-1, 1] range

## 🔧 Configuration

Key hyperparameters can be adjusted in `config/config.yaml`:

```yaml
model:
  image_size: 256
  batch_size: 1
  epochs: 50
  learning_rate: 0.0002
  lambda_cycle: 10.0
  lambda_identity: 5.0

training:
  checkpoint_freq: 5
  log_freq: 100
  sample_freq: 1000
```

## 📊 Results

While focusing on the learning process and architectural implementation, the model demonstrates:

- **Style Transfer Capability**: Successfully captures Monet's color palette and brushstroke patterns
- **Content Preservation**: Maintains structural elements of input photographs
- **Artistic Interpretation**: Creates visually appealing artistic renditions

## 📚 References

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Alankrit Singh**
- Email: f20221186@goa.bits-pilani.ac.in
- GitHub: [@Al0win](https://github.com/Al0win)

## 🙏 Acknowledgments

- BITS Pilani Goa Campus - Generative AI Course
- TensorFlow team for the excellent deep learning framework
- Original CycleGAN authors for the groundbreaking research

---




