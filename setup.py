from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="monetai",
    version="0.1.0",
    author="Alankrit Singh",
    author_email="f20221186@goa.bits-pilani.ac.in",
    description="CycleGAN implementation for Photo to Monet style transfer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Al0win/MonetAI",
    project_urls={
        "Bug Tracker": "https://github.com/Al0win/MonetAI/issues",
        "Documentation": "https://github.com/Al0win/MonetAI/blob/main/README.md",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "evaluation": [
            "pytorch-fid>=0.2.0",
            "lpips>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "monetai-train=scripts.train:main",
            "monetai-generate=scripts.generate:main", 
            "monetai-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "cyclegan", 
        "gan", 
        "deep-learning", 
        "style-transfer", 
        "monet", 
        "art", 
        "computer-vision",
        "tensorflow",
        "image-processing"
    ],
)