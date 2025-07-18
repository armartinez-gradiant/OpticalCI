"""
Photodetector - Componente Fotónico
=====================================

Componente extraído y refactorizado del sistema PtONN-TESTS.
Parte del framework TorchONN para redes neuronales ópticas.

Autor: PtONN-TESTS Team
Fecha: 2025-07-17
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from .base_component import BasePhotonicComponent

class Photodetector(nn.Module):
    """
    Photodetector - Conversión óptico-eléctrica.
    
    Convierte potencia óptica a corriente eléctrica.
    """
    
    def __init__(
        self,
        responsivity: float = 1.0,  # A/W
        dark_current: float = 1e-9,  # A
        thermal_noise: float = 1e-12,  # A²/Hz
        bandwidth: float = 10e9,  # Hz
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.responsivity = responsivity
        self.dark_current = dark_current
        self.thermal_noise = thermal_noise
        self.bandwidth = bandwidth
    
    def forward(self, optical_signal: torch.Tensor) -> torch.Tensor:
        """
        Convertir señal óptica a eléctrica.
        
        Args:
            optical_signal: [batch_size, n_wavelengths] - Campo óptico
            
        Returns:
            electrical_current: [batch_size, n_wavelengths] - Corriente
        """
        # Optical power = |E|²
        optical_power = torch.abs(optical_signal)**2
        
        # Photocurrent = Responsivity × Power
        photocurrent = self.responsivity * optical_power
        
        # Add dark current
        photocurrent += self.dark_current
        
        # Add thermal noise (si está en entrenamiento)
        if self.training:
            noise_std = torch.sqrt(torch.tensor(self.thermal_noise * self.bandwidth, device=self.device))
            thermal_noise = torch.randn_like(photocurrent) * noise_std
            photocurrent += thermal_noise
        
        return photocurrent

# Exports del módulo
__all__ = ['Photodetector']
