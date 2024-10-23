from setuptools import setup, find_packages

setup(
    name="monetai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8.0",
        "numpy>=1.19.2",
        "pandas>=1.1.3",
        "matplotlib>=3.3.2",
        "pillow>=8.0.1",
        "opencv-python>=4.4.0",
        "tqdm>=4.50.2",
        "scikit-learn>=0.23.2",
    ],
    extras_require={
        "dev": [
            "jupyter",
            "notebook",
        ],
    },
    python_requires=">=3.8,<3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="GAN-based Monet style transfer project",
    keywords="gan, deep-learning, art, style-transfer",
)