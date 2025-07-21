# MonetAI Makefile
# Common operations for development and usage

.PHONY: install install-dev clean test lint format train generate evaluate demo help

# Variables
PYTHON = python3
PIP = pip3
PROJECT_NAME = monetai
SRC_DIR = src/$(PROJECT_NAME)
TEST_DIR = tests
CHECKPOINT_DIR = checkpoints
OUTPUT_DIR = outputs
LOG_DIR = logs

# Default target
help:
	@echo "MonetAI - CycleGAN for Photo to Monet Style Transfer"
	@echo "======================================================"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install package and dependencies"
	@echo "  install-dev  - Install in development mode with dev dependencies"
	@echo "  clean        - Clean up generated files and directories"
	@echo "  test         - Run unit tests"
	@echo "  lint         - Run code linting with flake8"
	@echo "  format       - Format code with black"
	@echo "  train        - Train CycleGAN model with default settings"
	@echo "  generate     - Generate sample images (requires trained model)"
	@echo "  evaluate     - Evaluate model performance"
	@echo "  demo         - Run demonstration examples"
	@echo "  setup        - Initial setup for new users"
	@echo "  help         - Show this help message"

# Installation targets
install:
	@echo "📦 Installing MonetAI..."
	$(PIP) install -e .
	@echo "✅ Installation complete!"

install-dev:
	@echo "📦 Installing MonetAI in development mode..."
	$(PIP) install -e .[dev]
	@echo "✅ Development installation complete!"

# Setup for new users
setup: install
	@echo "🚀 Setting up MonetAI for first use..."
	mkdir -p $(CHECKPOINT_DIR)
	mkdir -p $(OUTPUT_DIR)
	mkdir -p $(LOG_DIR)
	mkdir -p datasets/sample_photos
	mkdir -p datasets/sample_monet
	@echo "✅ Directory structure created!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Add your photo datasets to datasets/sample_photos/"
	@echo "2. Add Monet paintings to datasets/sample_monet/"
	@echo "3. Run 'make train' to start training"

# Development targets
test:
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest $(TEST_DIR) -v
	@echo "✅ Tests complete!"

test-basic:
	@echo "🧪 Running basic tests..."
	$(PYTHON) tests/test_monetai.py
	@echo "✅ Basic tests complete!"

lint:
	@echo "🔍 Running linting..."
	$(PYTHON) -m flake8 $(SRC_DIR) scripts/ examples.py --max-line-length=100 --ignore=E203,W503
	@echo "✅ Linting complete!"

format:
	@echo "🎨 Formatting code..."
	$(PYTHON) -m black $(SRC_DIR) scripts/ examples.py tests/ --line-length=100
	@echo "✅ Code formatting complete!"

# Cleanup targets
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage .pytest_cache/
	@echo "✅ Cleanup complete!"

clean-all: clean
	@echo "🧹 Deep cleaning..."
	rm -rf $(CHECKPOINT_DIR)/*
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(LOG_DIR)/*
	rm -rf .venv/ venv/ ENV/
	@echo "✅ Deep cleanup complete!"

# Training and inference targets
train:
	@echo "🎯 Starting CycleGAN training..."
	@if [ ! -d "datasets" ]; then \
		echo "❌ Error: datasets/ directory not found!"; \
		echo "Please add your datasets first. See README.md for details."; \
		exit 1; \
	fi
	$(PYTHON) scripts/train.py --epochs 50 --batch-size 1
	@echo "✅ Training complete!"

train-quick:
	@echo "🎯 Quick training (5 epochs for testing)..."
	$(PYTHON) scripts/train.py --epochs 5 --batch-size 1
	@echo "✅ Quick training complete!"

generate:
	@echo "🎨 Generating Monet-style images..."
	@if [ ! -d "$(CHECKPOINT_DIR)" ] || [ -z "$$(ls -A $(CHECKPOINT_DIR) 2>/dev/null)" ]; then \
		echo "❌ Error: No trained model found in $(CHECKPOINT_DIR)/"; \
		echo "Please train a model first with 'make train'"; \
		exit 1; \
	fi
	$(PYTHON) scripts/generate.py \
		--model-path $(CHECKPOINT_DIR)/final_model \
		--input-dir datasets/sample_photos \
		--output-dir $(OUTPUT_DIR)/generated_monet \
		--create-zip
	@echo "✅ Image generation complete!"

evaluate:
	@echo "📊 Evaluating model performance..."
	@if [ ! -d "$(CHECKPOINT_DIR)" ] || [ -z "$$(ls -A $(CHECKPOINT_DIR) 2>/dev/null)" ]; then \
		echo "❌ Error: No trained model found in $(CHECKPOINT_DIR)/"; \
		echo "Please train a model first with 'make train'"; \
		exit 1; \
	fi
	$(PYTHON) scripts/evaluate.py \
		--model-path $(CHECKPOINT_DIR)/final_model \
		--test-photos datasets/sample_photos \
		--real-monet datasets/sample_monet \
		--output-dir $(OUTPUT_DIR)/evaluation
	@echo "✅ Evaluation complete!"

# Demo and examples
demo:
	@echo "🎭 Running MonetAI demonstration..."
	$(PYTHON) examples.py
	@echo "✅ Demo complete!"

notebook:
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook notebooks/cyclegan_demo.ipynb

# Docker targets (future enhancement)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t $(PROJECT_NAME):latest .
	@echo "✅ Docker image built!"

docker-run:
	@echo "🐳 Running Docker container..."
	docker run --gpus all -it -v $(PWD):/workspace $(PROJECT_NAME):latest
	@echo "✅ Docker container started!"

# Documentation targets
docs:
	@echo "📚 Opening documentation..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open README.md; \
	elif command -v open > /dev/null; then \
		open README.md; \
	else \
		echo "Please open README.md manually"; \
	fi

# Quick status check
status:
	@echo "📊 MonetAI Project Status"
	@echo "========================="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "TensorFlow version: $$($(PYTHON) -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "GPU available: $$($(PYTHON) -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices(\"GPU\")) > 0)' 2>/dev/null || echo 'Unknown')"
	@echo ""
	@echo "Directory status:"
	@echo "  Checkpoints: $$(ls -la $(CHECKPOINT_DIR) 2>/dev/null | wc -l || echo '0') files"
	@echo "  Outputs: $$(ls -la $(OUTPUT_DIR) 2>/dev/null | wc -l || echo '0') files"
	@echo "  Logs: $$(ls -la $(LOG_DIR) 2>/dev/null | wc -l || echo '0') files"
	@echo "  Datasets: $$(ls -la datasets 2>/dev/null | wc -l || echo '0') directories"

# Development workflow
dev-setup: install-dev setup
	@echo "🚀 Development environment ready!"
	@echo "Run 'make test' to verify everything works"

# Complete workflow for new users
quickstart: setup train-quick generate
	@echo "🎉 Quickstart complete!"
	@echo "Check $(OUTPUT_DIR)/ for generated images"
