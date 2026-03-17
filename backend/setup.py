"""
Retina-GPT: Foundation Model for Retinal Image Analysis
Install with: pip install -e .
"""

from setuptools import find_packages, setup

setup(
    name="retina_gpt",
    version="1.0.0",
    description="Production-grade AI pipeline for retinal fundus image analysis and clinical report generation",
    author="Retina-GPT Engineering Team",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.38.0",
        "monai>=1.3.0",
        "opencv-python>=4.9.0.80",
        "Pillow>=10.2.0",
        "numpy>=1.26.0",
        "einops>=0.7.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.6.0",
        "albumentations>=1.3.1",
        "timm>=0.9.12",
        "scikit-learn>=1.4.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0.1",
        "pandas>=2.2.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "black>=24.1.0", "isort>=5.13.0", "flake8>=7.0.0"],
        "logging": ["wandb>=0.16.0", "tensorboard>=2.16.0"],
    },
    entry_points={
        "console_scripts": [
            "retina-train=scripts.train:main",
            "retina-api=api.main:app",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
