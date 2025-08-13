"""
Generadores de Datos - OpticalCI Utils
=====================================

Utilidades para generar diferentes tipos de datos sintéticos.
"""

import torch
import numpy as np
from typing import Tuple, Optional

def generate_linear_transform_data(
    n_samples: int = 1000,
    input_size: int = 4,
    output_size: int = 4,
    noise_level: float = 0.1,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generar datos de transformación lineal."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Matriz de transformación aleatoria
    W = torch.randn(output_size, input_size, device=device) * 0.5
    
    # Datos de entrada
    X = torch.randn(n_samples, input_size, device=device) * 2.0
    
    # Aplicar transformación
    Y = torch.matmul(X, W.t())
    
    # Añadir ruido
    if noise_level > 0:
        Y += torch.randn_like(Y) * noise_level
    
    return X, Y

def generate_rotation_data(
    n_samples: int = 1000,
    size: int = 3,
    angle_degrees: float = 45.0,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generar datos de rotación (2D/3D)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    angle = np.radians(angle_degrees)
    
    # Datos de entrada
    X = torch.randn(n_samples, size, device=device)
    
    if size == 2:
        # Rotación 2D
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)
        Y = torch.matmul(X, R.t())
    elif size == 3:
        # Rotación 3D alrededor del eje Z
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], device=device)
        Y = torch.matmul(X, R.t())
    else:
        raise ValueError("Rotación solo soportada para 2D y 3D")
    
    return X, Y