# PtONN-TESTS

**A modern PyTorch library for photonic neural network simulation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Proprietary-yellow.svg)](#license)

PtONN-TESTS is a modernized PyTorch library for simulating photonic integrated circuits and optical neural networks. It provides physically-accurate models of photonic components like Mach-Zehnder interferometers, microring resonators, and wavelength division multiplexing systems.

## Features

- **Physical accuracy**: Real physics-based models of photonic components
- **Modern PyTorch**: Compatible with PyTorch 2.0+ and Python 3.8-3.11
- **GPU acceleration**: Full CUDA support for high-performance computing
- **Modular design**: Clean, extensible architecture for research and development
- **Comprehensive testing**: Extensive test suite ensuring reliability

## Quick Start

### Installation

```bash
git clone https://github.com/anxo-rodriguez/PtONN-TESTS.git
cd PtONN-TESTS
pip install -e .
```

### Basic Usage

```python
import torch
from torchonn.layers import MZIBlockLinear, MicroringResonator
from torchonn.models import ONNBaseModel

# Create a simple photonic neural network
class PhotonicNN(ONNBaseModel):
    def __init__(self):
        super().__init__()
        self.layer1 = MZIBlockLinear(10, 8, mode="usv")
        self.layer2 = MZIBlockLinear(8, 4, mode="phase")
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

# Initialize and use
model = PhotonicNN()
x = torch.randn(32, 10)
output = model(x)
print(f"Output shape: {output.shape}")  # torch.Size([32, 4])
```

## Components

### Basic Layers
- **MZILayer**: Mach-Zehnder interferometer with unitary matrix decomposition
- **MZIBlockLinear**: Advanced MZI layer with multiple operation modes (USV, weight, phase)

### Photonic Components
- **MicroringResonator**: Wavelength-selective filtering with thermal effects
- **AddDropMRR**: Four-port add-drop microring with realistic physics
- **DirectionalCoupler**: Beam splitting and combining
- **Photodetector**: Optical-to-electrical conversion

### Advanced Components
- **PhaseChangeCell**: Non-volatile memory using phase change materials
- **WDMMultiplexer**: Wavelength division multiplexing systems
- **MRRWeightBank**: Microring-based weight matrices for optical computing

## Examples

### Microring Resonator

```python
from torchonn.layers import MicroringResonator

# Create a microring with specific parameters
mrr = MicroringResonator(
    radius=10e-6,           # 10 μm radius
    coupling_strength=0.3,  # 30% coupling
    q_factor=15000,         # Quality factor
    center_wavelength=1550e-9  # Telecom wavelength
)

# Simulate wavelength sweep
wavelengths = torch.linspace(1530e-9, 1570e-9, 100)
input_signal = torch.randn(1, 100)

output = mrr(input_signal, wavelengths)
through_port = output['through']
drop_port = output['drop']
```

### Wavelength Division Multiplexing

```python
from torchonn.components import WDMMultiplexer

# Create WDM system
wdm = WDMMultiplexer(wavelengths=[
    1530e-9, 1540e-9, 1550e-9, 1560e-9
])

# Multiplex channels
channels = [torch.randn(32) for _ in range(4)]
multiplexed = wdm.multiplex(channels)

# Demultiplex
demuxed = wdm.demultiplex(multiplexed)
```

## Architecture

```
torchonn/
├── layers/              # Basic photonic layers
│   ├── mzi_layer.py        # Mach-Zehnder interferometers
│   ├── mzi_block_linear.py # Advanced MZI layers
│   ├── microring.py        # Microring resonators
│   ├── couplers.py         # Directional couplers
│   └── detectors.py        # Photodetectors
├── components/          # Specialized components
│   ├── memory.py           # Phase change materials
│   └── wdm.py             # WDM systems
├── models/              # Base model classes
├── ops/                 # Core operations
└── utils/               # Helper functions
```

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher  
- NumPy < 2.0 (for compatibility)
- Optional: CUDA for GPU acceleration

## Testing

Run the test suite to verify your installation:

```bash
# Run all tests
pytest tests/

# Run specific component tests
pytest tests/test_layers.py -v
pytest tests/test_components.py -v

# Run with coverage
pytest tests/ --cov=torchonn
```

## Performance

The library is optimized for performance with:

- **Vectorized operations** using PyTorch's optimized kernels
- **GPU acceleration** for large-scale simulations
- **Memory-efficient** implementations
- **Physical validation** to ensure energy conservation

Typical performance on modern hardware:

| Model Size | CPU Time | GPU Time | Memory |
|-----------|----------|----------|---------|
| Small (1K params) | 0.5ms | 0.1ms | 10MB |
| Medium (100K params) | 5ms | 0.5ms | 50MB |
| Large (10M params) | 500ms | 10ms | 500MB |

## Physical Validation

All components are validated against known physical principles:

- Energy conservation for unitary operations
- Proper resonance behavior in ring resonators
- Realistic coupling in directional couplers
- Causality and stability in time-domain simulations

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-component`)
3. Make your changes and add tests
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/anxo-rodriguez/PtONN-TESTS.git
cd PtONN-TESTS
pip install -e .
pip install -r requirements-dev.txt
```

## Roadmap

**Version 1.0.0**
- Microring arrays and meshes
- Advanced thermal modeling
- Visualization tools
- Tutorial notebooks

**Version 1.1.0**
- Complete nonlinear optics suite
- Hardware integration
- Performance optimizations

## License

This software is proprietary and confidential. All rights reserved.

For licensing inquiries, contact: armartinez@gradiant.org

## Author

**Anxo Rodríguez Martínez**


- Email: armartinez@gradiant.org
- GitHub: [@armartinez-gradiant]

## Citation

If you use PtONN-TESTS in your research, please cite:

```bibtex
@software{rodriguez2025ptonn,
  title={PtONN-TESTS: A Modern PyTorch Library for Photonic Neural Networks},
  author={Anxo Rodríguez Martínez},
  year={2025},
  url={https://github.com/anxo-rodriguez/PtONN-TESTS}
}
```

## Acknowledgments

This project builds upon research in photonic computing and neural networks. Special thanks to the PyTorch team for providing the foundational framework. 