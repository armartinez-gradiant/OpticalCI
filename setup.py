#!/usr/bin/env python3
"""
Setup script for PtONN-TESTS - A modern, updated version of pytorch-onn
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Leer README
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "PtONN-TESTS - A modern PyTorch Library for Photonic Integrated Circuit Simulation"

# Leer requirements de manera más robusta
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    if not requirements_path.exists():
        # Si no existe requirements.txt, usar dependencias básicas
        return [
            "torch>=2.0.0,<2.8.0",
            "torchvision>=0.15.0,<0.20.0",
            "torchaudio>=2.0.0,<2.8.0",
            "numpy>=1.19.0,<2.0.0",
            "scipy>=1.7.0,<1.13.0",
            "matplotlib>=3.3.0,<4.0.0",
            "pyyaml>=5.4.0,<7.0.0",
            "tqdm>=4.60.0,<5.0.0",
        ]
    
    requirements = []
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Saltar líneas vacías y comentarios
            if line and not line.startswith("#"):
                requirements.append(line)
    
    return requirements

# Obtener versión del paquete
def get_version():
    version_path = Path(__file__).parent / "torchonn" / "__init__.py"
    if version_path.exists():
        with open(version_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="ptonn-tests",
    version=get_version(),
    description="A modern, updated PyTorch Library for Photonic Integrated Circuit Simulation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Gradiant Technology Center",
    author_email="info@gradiant.org",
    url="https://github.com/armartinez-gradiant/PtONN-TESTS",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.8,<3.13",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="pytorch photonics optical neural networks ONN simulation",
    project_urls={
        "Bug Reports": "https://github.com/armartinez-gradiant/PtONN-TESTS/issues",
        "Source": "https://github.com/armartinez-gradiant/PtONN-TESTS",
    },
    # Datos adicionales del paquete
    package_data={
        "torchonn": ["py.typed"],
    },
    # Configuración adicional
    zip_safe=False,
    # Dependencias opcionales
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.900",
        ],
        "all": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.900",
            "psutil>=5.8.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
)
