"""
ONN Base Model - Modelo base para redes neuronales ópticas
=========================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Union

class ONNBaseModel(nn.Module):
    """
    Clase base para modelos de redes neuronales ópticas.
    """
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.to(self.device)
    
    def reset_parameters(self):
        """Reinicializar todos los parámetros del modelo"""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - debe ser implementado por subclases"""
        raise NotImplementedError("Subclasses must implement forward method")
