"""
Capas Fotónicas - TorchONN
=========================

Módulo de capas neuronales fotónicas.
"""

# Imports seguros
__all__ = []

try:
    from .mzi_layer import MZILayer
    __all__.append('MZILayer')
except ImportError as e:
    print(f"Warning: No se pudo importar MZILayer: {e}")

try:
    from .mzi_block_linear import MZIBlockLinear
    __all__.append('MZIBlockLinear')
except ImportError as e:
    print(f"Warning: No se pudo importar MZIBlockLinear: {e}")

# Capas nuevas (si existen)
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
