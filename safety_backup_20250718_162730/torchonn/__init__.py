"""
TorchONN - Framework para Redes Neuronales Ópticas
===================================================

Framework modular y profesional para el diseño, simulación y entrenamiento
de redes neuronales ópticas (ONNs) basado en PyTorch.
"""

__version__ = "2.0.0"
__author__ = "PtONN-TESTS Team"

# Core imports
from . import layers
from . import models
from . import devices
from . import ops
from . import utils

# Try to import components
try:
    from . import components
except ImportError:
    pass

# Key classes
from .layers import MZILayer, MZIBlockLinear
from .models import ONNBaseModel

__all__ = [
    'layers', 'models', 'devices', 'ops', 'utils',
    'MZILayer', 'MZIBlockLinear', 'ONNBaseModel'
]
