"""
Capas Fotónicas - TorchONN
=========================

Módulo de capas neuronales fotónicas.
"""

from .mzi_layer import MZILayer
from .mzi_block_linear import MZIBlockLinear

__all__ = ['MZILayer', 'MZIBlockLinear']

# Import optional layers
try:
    from .mrr_weight_bank import MRRWeightBank
    __all__.append('MRRWeightBank')
except ImportError:
    pass

try:
    from .photonic_linear import PhotonicLinear
    __all__.append('PhotonicLinear')
except ImportError:
    pass

try:
    from .photonic_conv2d import PhotonicConv2D
    __all__.append('PhotonicConv2D')
except ImportError:
    pass
