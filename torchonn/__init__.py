"""
PtONN-TESTS - A modern, updated PyTorch Library for Photonic Integrated Circuit Simulation

This is an updated and improved version of pytorch-onn that works with modern PyTorch versions.
"""

__version__ = "1.0.0"
__author__ = "Anxo Rodríguez Martínez"
__email__ = "anxo.martinez@email.com"

# Importar módulos principales
try:
    from . import layers
    from . import models
    from . import devices
    from . import ops
    from . import components
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import some modules: {e}")

# Configuración por defecto
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = "float32"

def get_version():
    """Obtener versión del paquete."""
    return __version__

def get_device():
    """Obtener dispositivo por defecto."""
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return DEFAULT_DEVICE

# Información del paquete
__all__ = [
    "layers",
    "models", 
    "devices",
    "ops",
    "components",
    "get_version",
    "get_device",
    "__version__",
]