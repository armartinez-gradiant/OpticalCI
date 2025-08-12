# OpticalCI

A PyTorch library for photonic neural network simulation and optical computing research.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](#testing)

## Overview

OpticalCI provides PyTorch-compatible components for simulating photonic integrated circuits and optical neural networks. It includes physically-grounded models of Mach-Zehnder interferometers, microring resonators, and other photonic devices commonly used in optical computing research.

## Features

- **MZI Layers**: Mach-Zehnder interferometer implementations with Reck scheme decomposition
- **Microring Components**: Resonator models with realistic Q-factors and coupling parameters
- **Physical Accuracy**: Models include fabrication tolerances and material properties
- **PyTorch Integration**: Full compatibility with PyTorch 2.0+ and standard training workflows
- **GPU Support**: CUDA acceleration for large-scale simulations

## Installation

```bash
git clone https://github.com/armartinez-gradiant/OpticalCI.git
cd OpticalCI
pip install -e .
```

## Quick Start

```python
import torch
from torchonn.layers import MZIBlockLinear
from torchonn.models import ONNBaseModel

class SimpleONN(ONNBaseModel):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = MZIBlockLinear(input_size, hidden_size, mode="usv")
        self.layer2 = MZIBlockLinear(hidden_size, output_size, mode="phase")
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)

# Create and use the model
model = SimpleONN(8, 6, 4)
x = torch.randn(32, 8)
output = model(x)
```

## Components

### Core Layers

- `MZILayer`: Basic Mach-Zehnder interferometer layer
- `MZIBlockLinear`: Advanced MZI blocks with multiple operation modes
- `MicroringResonator`: Single microring with configurable parameters
- `DirectionalCoupler`: Beam splitting and combining

### Advanced Systems

- `WDMMultiplexer`: Wavelength division multiplexing
- `PhotoDetector`: Optical-to-electrical conversion
- `PhaseChangeCell`: Non-volatile memory elements

## Examples

Check the `examples/` directory for complete usage examples, including:

- Basic photonic layer usage
- Training optical neural networks
- Component characterization and validation

## Testing

Run the test suite to verify installation:

```bash
pytest tests/ -v
```

The test suite includes physics validation to ensure components behave according to optical principles.

## Requirements

- Python 3.8-3.12
- PyTorch 2.0+
- NumPy < 2.0
- SciPy
- Matplotlib

## Documentation

- **API Reference**: Check docstrings in source code
- **Examples**: See `examples/` directory
- **Issues**: [GitHub Issues](https://github.com/armartinez-gradiant/OpticalCI/issues)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Author

**Anxo Rodríguez Martínez**  
Email: armartinez@gradiant.org  
Organization: Gradiant

## License

All rights reserved. For licensing inquiries, contact armartinez@gradiant.org

## Citation

If you use OpticalCI in your research:

```bibtex
@software{rodriguez2025opticalci,
  title={OpticalCI: A PyTorch Library for Photonic Neural Networks},
  author={Anxo Rodríguez Martínez},
  year={2025},
  url={https://github.com/armartinez-gradiant/OpticalCI}
}
```