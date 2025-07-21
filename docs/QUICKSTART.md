# MonetAI Quick Start Guide

Get up and running with MonetAI in just a few steps!

## 🚀 Quick Installation

### Prerequisites
- Python 3.8+ 
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 2GB+ free storage

### 1. Clone and Setup
```bash
git clone https://github.com/Al0win/MonetAI.git
cd MonetAI
pip install -e .
```

### 2. Download Sample Data
```bash
# Create sample directories
mkdir -p datasets/sample_photos
mkdir -p datasets/sample_monet

# Add your own images or download sample datasets
# Photos should go in datasets/sample_photos/
# Monet paintings should go in datasets/sample_monet/
```

## 🎨 Using MonetAI

### Method 1: Python API (Recommended)
```python
from src.monetai.models import CycleGAN
from src.monetai.data import load_image_dataset

# Initialize model
model = CycleGAN()

# Load pre-trained weights (if available)
model.load_model('checkpoints/monet_cyclegan')

# Transform images
import tensorflow as tf

# Load and transform a single image
image_path = "path/to/your/photo.jpg"
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3)
image = tf.image.resize(image, [256, 256])
image = tf.cast(image, tf.float32) / 255.0
image = (image * 2.0) - 1.0  # Normalize to [-1, 1]
image = tf.expand_dims(image, 0)  # Add batch dimension

# Generate Monet-style version
monet_style = model.transform(image)

# Save result
output_image = (monet_style[0] + 1) / 2.0  # Rescale to [0, 1]
tf.keras.preprocessing.image.save_img('output.jpg', output_image.numpy())
```

### Method 2: Command Line Interface
```bash
# Train a new model
python scripts/train.py --epochs 10 --batch-size 1

# Generate images from trained model  
python scripts/generate.py \
    --model-path checkpoints/final_model \
    --input-dir datasets/sample_photos \
    --output-dir generated_monet

# Evaluate model performance
python scripts/evaluate.py \
    --model-path checkpoints/final_model \
    --test-photos datasets/sample_photos \
    --real-monet datasets/sample_monet
```

### Method 3: Jupyter Notebook
```bash
jupyter notebook notebooks/cyclegan_demo.ipynb
```

## 📁 Project Structure After Setup

```
MonetAI/
├── src/monetai/           # Core package
├── scripts/               # Training & evaluation scripts  
├── notebooks/             # Interactive demos
├── config/                # Configuration files
├── checkpoints/           # Model saves (created after training)
├── logs/                  # Training logs (created during training)
├── outputs/               # Generated images (created during inference)
└── datasets/              # Your image datasets
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  image_size: 256          # Image resolution
  batch_size: 1            # Batch size (keep at 1 for best results)
  learning_rate: 2e-4      # Learning rate
  lambda_cycle: 10.0       # Cycle consistency weight
  lambda_identity: 5.0     # Identity loss weight

training:
  epochs: 50               # Training epochs
  checkpoint_freq: 10      # Save checkpoint every N epochs
  sample_freq: 5           # Generate samples every N epochs
```

## 🎯 Common Use Cases

### 1. Transform Personal Photos
```python
# Transform your vacation photos to Monet style
from src.monetai.models import CycleGAN

model = CycleGAN()
model.load_model('checkpoints/trained_model')

# Process directory of photos
import os
for photo_file in os.listdir('my_photos/'):
    if photo_file.endswith('.jpg'):
        # Transform photo (code from Method 1 above)
        pass
```

### 2. Batch Processing
```bash
# Process entire directory
python scripts/generate.py \
    --input-dir my_vacation_photos/ \
    --output-dir monet_vacation/ \
    --create-zip \
    --zip-name vacation_monet.zip
```

### 3. Training from Scratch
```bash
# With your own dataset
python scripts/train.py \
    --photo-train-dir datasets/my_photos/ \
    --monet-train-dir datasets/monet_paintings/ \
    --epochs 100 \
    --checkpoint-dir my_checkpoints/
```

## 🔧 Troubleshooting

### Common Issues

**GPU not detected:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Out of memory:**
- Reduce batch size to 1
- Use smaller images (reduce image_size in config)
- Close other GPU applications

**Poor results:**
- Train for more epochs (100-200 recommended)
- Ensure good quality, diverse training data
- Adjust loss function weights in config

**Import errors:**
```bash
# Reinstall in development mode
pip uninstall monetai
pip install -e .
```

## 📊 Expected Results

With proper training (50+ epochs), you should see:
- Photos transformed to have impressionistic style
- Maintained scene structure and content
- Monet-like color palettes and brush stroke effects
- Artistic interpretation while preserving recognizable elements

## 🎓 Learning Path

1. **Start**: Run the Jupyter notebook demo
2. **Experiment**: Try different photos and settings
3. **Train**: Use your own dataset for custom styles
4. **Extend**: Modify loss functions or architecture
5. **Evaluate**: Implement additional quality metrics

## 📞 Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Documentation**: Check `docs/ARCHITECTURE.md` for details
- **Contributing**: See `CONTRIBUTING.md` to contribute
- **Contact**: f20221186@goa.bits-pilani.ac.in

Happy creating! 🎨✨
