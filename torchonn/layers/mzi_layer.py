"""
MZI Layer - Basic implementation for PtONN-TESTS
"""

import torch
import torch.nn as nn
from typing import Optional, Union

class MZILayer(nn.Module):
    """
    Basic MZI Layer implementation
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MZILayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Configurar device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        
        # Parámetros simples
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinicializar parámetros con mayor aleatoriedad."""
        with torch.no_grad():
            # ✅ FIX: Usar inicialización más agresiva para garantizar cambios
            nn.init.xavier_uniform_(self.weight, gain=1.0)
            # ✅ FIX: Añadir ruido extra para garantizar que los valores cambien
            self.weight.add_(torch.randn_like(self.weight) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.mm(x, self.weight.t())
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"