from setuptools import setup, find_packages

setup(
    name="ptonn-tests",
    version="1.0.0",
    description="A modern PyTorch Library for Photonic Computing",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.19.0,<2.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.8",
) 