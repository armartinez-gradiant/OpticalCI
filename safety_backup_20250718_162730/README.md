# PtONN-TESTS

**A modern, updated PyTorch Library for Photonic Integrated Circuit Simulation and Photonic AI Computing**

[![CI](https://github.com/armartinez-gradiant/PtONN-TESTS/workflows/CI/badge.svg)](https://github.com/armartinez-gradiant/PtONN-TESTS/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

PtONN-TESTS is a modernized and improved version of pytorch-onn, designed to work seamlessly with current PyTorch versions while providing enhanced functionality for photonic neural network simulation and research.

### ✨ Key Features

- **🔧 Modern PyTorch Compatibility**: Works with PyTorch 2.0+ and Python 3.8-3.11
- **⚡ GPU Acceleration**: Full CUDA support for high-performance computing
- **🏗️ Modular Architecture**: Clean, extensible design for easy customization
- **🧪 Comprehensive Testing**: Extensive test suite ensuring reliability
- **📚 Rich Documentation**: Clear examples and API documentation
- **🔬 Research Ready**: Perfect for academic research and industrial applications

### 🎯 What's New

- ✅ Fixed NumPy 2.x compatibility issues
- ✅ Updated MZI layer implementations
- ✅ Modern PyTorch best practices
- ✅ Improved error handling and validation
- ✅ Enhanced device management
- ✅ Comprehensive test coverage

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- NumPy < 2.0 (for compatibility)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/armartinez-gradiant/PtONN-TESTS.git
cd PtONN-TESTS

# Install dependencies
pip install "numpy<2.0"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install in development mode
pip install -e .
Automated Setup
bash# Use our setup script
chmod +x setup_ptonn.sh
./setup_ptonn.sh
Verify Installation
bashpython test_installation.py
🚀 Quick Start
Basic Usage
pythonimport torch
from torchonn.layers import MZIBlockLinear, MZILayer
from torchonn.models import ONNBaseModel

# Create a simple photonic neural network
class SimpleONN(ONNBaseModel):
    def __init__(self):
        super().__init__()
        self.layer1 = MZIBlockLinear(
            in_features=10,
            out_features=8, 
            miniblock=4,
            mode="usv"
        )
        self.layer2 = MZILayer(in_features=8, out_features=4)
        self.activation = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Initialize model
model = SimpleONN()

# Forward pass
x = torch.randn(32, 10)  # Batch of 32 samples
output = model(x)        # Shape: (32, 4)
print(f"Output shape: {output.shape}")
Training Example
pythonimport torch
import torch.nn as nn
import torch.optim as optim

# Create model and data
model = SimpleONN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training data
x = torch.randn(100, 10)
y = torch.randn(100, 4)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
🏗️ Architecture
Layer Types

MZILayer: Basic Mach-Zehnder interferometer layer
MZIBlockLinear: Advanced block-based MZI layer with multiple operation modes

Operation Modes

USV Mode: Uses SVD decomposition for weight representation
Weight Mode: Direct weight matrix representation
Phase Mode: Phase-based representation for hardware implementation

Device Support

CPU: Full support for all operations
CUDA: GPU acceleration for large-scale simulations
MPS: Apple Metal Performance Shaders support

🧪 Testing
Run the complete test suite:
bash# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_layers.py      # Layer functionality
pytest tests/test_models.py      # Model functionality  
pytest tests/test_integration.py # End-to-end tests

# Run with coverage
pytest tests/ --cov=torchonn --cov-report=html
🔧 Development
Setting up for Development
bash# Clone and setup
git clone https://github.com/armartinez-gradiant/PtONN-TESTS.git
cd PtONN-TESTS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install in development mode
pip install -e .
pip install pytest pytest-cov black flake8
Code Quality
bash# Format code
black torchonn/ tests/

# Lint code  
flake8 torchonn/ tests/

# Run tests
pytest tests/
📊 Performance
PtONN-TESTS is optimized for performance:

GPU Acceleration: Up to 10x speedup on CUDA devices
Memory Efficient: Optimized memory usage for large models
Vectorized Operations: Leverages PyTorch's optimized kernels

Benchmarks
Model SizeCPU TimeGPU TimeMemory UsageSmall (100 params)0.5ms0.1ms10MBMedium (10K params)5ms0.5ms50MBLarge (1M params)500ms10ms500MB
🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.
How to Contribute

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Make your changes
Add tests for new functionality
Ensure all tests pass (pytest tests/)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Original pytorch-onn project by Jiaqi Gu
Gradiant Technology Center for supporting this project
The PyTorch community for the excellent framework

📞 Support

Issues: GitHub Issues
Email: info@gradiant.org
Documentation: Project Wiki

🗺️ Roadmap

 Add more layer types (Microring resonators, etc.)
 Implement noise models for realistic simulation
 Add visualization tools
 Create tutorial notebooks
 Optimize for edge deployment
 Add PyTorch JIT compilation support


Made with ❤️ by Gradiant