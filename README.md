Hey!

This is the project I created for my Generative AI Course.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/MonetAI.git
cd MonetAI
```

2. Create and activate conda environment:
```bash
conda create -n monet python=3.8
conda activate monet
```

3. Install the package:
```bash
# For basic installation
pip install -e .

# For development installation (includes jupyter notebooks)
pip install -e ".[dev]"
```

## Development Workflow

1. Before starting new work:
```bash
git pull origin main  # Get latest changes
```

2. After making changes:
```bash
git add .
git commit -m "Description of changes"
git push origin main  # Or your feature branch
```

## Usage

```python
from monetai.models.gan import Generator, Discriminator
from monetai import utils

# Your code here
```