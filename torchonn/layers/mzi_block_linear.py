"""
MZI Block Linear Layer - Modern implementation for PtONN-TESTS
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

class MZIBlockLinear(nn.Module):
    """
    MZI Block Linear Layer - Compatible with modern PyTorch versions
    
    A modern implementation of the MZI Block Linear layer that works with
    current PyTorch versions and handles compatibility issues.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        miniblock: int = 4,
        mode: str = "usv",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize MZI Block Linear Layer
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            miniblock: Size of miniblock (default: 4)
            mode: Operation mode ("usv", "weight", "phase")
            device: Device to place the layer on
            dtype: Data type for the layer
        """
        super(MZIBlockLinear, self).__init__()
        
        # Configuración básica
        self.in_features = in_features
        self.out_features = out_features
        self.miniblock = miniblock
        self.mode = mode
        
        # Configurar device y dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        
        # Validar parámetros
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if miniblock <= 0:
            raise ValueError("miniblock must be positive")
        if mode not in ["usv", "weight", "phase"]:
            raise ValueError("mode must be 'usv', 'weight', or 'phase'")
            
        # Inicializar parámetros
        self._init_parameters()
        
    def _init_parameters(self):
        """Inicializar parámetros de la capa."""
        # Calcular dimensiones
        self.weight_shape = (self.out_features, self.in_features)
        
        if self.mode == "usv":
            # Modo USV: usar descomposición SVD
            self.u_matrix = nn.Parameter(torch.randn(self.out_features, self.out_features, device=self.device, dtype=self.dtype))
            self.s_matrix = nn.Parameter(torch.randn(min(self.out_features, self.in_features), device=self.device, dtype=self.dtype))
            self.v_matrix = nn.Parameter(torch.randn(self.in_features, self.in_features, device=self.device, dtype=self.dtype))
        elif self.mode == "weight":
            # Modo weight: usar matriz de pesos directa
            self.weight = nn.Parameter(torch.randn(self.weight_shape, device=self.device, dtype=self.dtype))
        elif self.mode == "phase":
            # Modo phase: usar fases
            self.phases = nn.Parameter(torch.randn(self.in_features + self.out_features, device=self.device, dtype=self.dtype))
            
        # Inicializar parámetros con valores razonables
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reinicializar parámetros con mayor aleatoriedad."""
        with torch.no_grad():
            if self.mode == "usv":
                # ✅ FIX: Usar inicializaciones más aleatorias
                nn.init.orthogonal_(self.u_matrix)
                # ✅ FIX: Valores singulares aleatorios en lugar de siempre 1s
                nn.init.uniform_(self.s_matrix, 0.1, 2.0)  # Valores positivos aleatorios
                nn.init.orthogonal_(self.v_matrix)
                
                # ✅ FIX: Añadir ruido extra para garantizar cambios
                self.u_matrix.add_(torch.randn_like(self.u_matrix) * 0.01)
                self.v_matrix.add_(torch.randn_like(self.v_matrix) * 0.01)
                
            elif self.mode == "weight":
                # ✅ FIX: Usar inicialización más agresiva
                nn.init.xavier_uniform_(self.weight, gain=1.0)
                # Añadir ruido extra
                self.weight.add_(torch.randn_like(self.weight) * 0.01)
                
            elif self.mode == "phase":
                # ✅ FIX: Fases completamente aleatorias
                nn.init.uniform_(self.phases, -np.pi, np.pi)
                
    def _get_weight_matrix(self) -> torch.Tensor:
        """Obtener la matriz de pesos según el modo."""
        if self.mode == "usv":
            # Reconstruir desde USV
            s_diag = torch.diag(self.s_matrix)
            if self.out_features > self.in_features:
                s_diag = torch.cat([s_diag, torch.zeros(self.out_features - self.in_features, self.in_features, device=self.device, dtype=self.dtype)], dim=0)
            elif self.out_features < self.in_features:
                s_diag = torch.cat([s_diag, torch.zeros(self.out_features, self.in_features - self.out_features, device=self.device, dtype=self.dtype)], dim=1)
            
            weight = torch.mm(torch.mm(self.u_matrix, s_diag), self.v_matrix.t())
            return weight
        elif self.mode == "weight":
            return self.weight
        elif self.mode == "phase":
            # Construir matriz desde fases (simplificado)
            weight = torch.zeros(self.weight_shape, device=self.device, dtype=self.dtype)
            for i in range(min(self.out_features, self.in_features)):
                weight[i, i] = torch.cos(self.phases[i])
            return weight
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Validar entrada
        if x.size(-1) != self.in_features:
            raise ValueError(f"Input size mismatch: expected {self.in_features}, got {x.size(-1)}")
            
        # Obtener matriz de pesos
        weight = self._get_weight_matrix()
        
        # Aplicar transformación lineal
        output = torch.mm(x, weight.t())
        
        return output
        
    def extra_repr(self) -> str: 
        """Representación extra para debugging."""
        return f"in_features={self.in_features}, out_features={self.out_features}, miniblock={self.miniblock}, mode={self.mode}"