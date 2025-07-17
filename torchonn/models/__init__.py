"""
Modelos ONN - TorchONN
=====================

Módulo de modelos y arquitecturas para redes neuronales ópticas.
"""

# Import del modelo base si existe
try:
    from .base_model import ONNBaseModel
    __all__ = ['ONNBaseModel']
except ImportError:
    # Crear modelo base si no existe
    import torch.nn as nn
    
    class ONNBaseModel(nn.Module):
        """Modelo base para ONNs - implementación temporal"""
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            return x
    
    __all__ = ['ONNBaseModel']

# Para compatibilidad hacia atrás
BaseONNModel = ONNBaseModel
__all__.append('BaseONNModel')
