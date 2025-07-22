"""
Detector Components for PtONN-TESTS  

Implementation of photodetectors and optical-to-electrical
conversion components for photonic neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import math
import warnings

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

def test_basic_components():
    """Test básico de componentes fotónicos."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🧪 Testing basic photonic components...")
    
    # Test MicroringResonator if available
    if "MicroringResonator" in globals():
        mrr = MicroringResonator(device=device)
        wavelengths = torch.linspace(1530e-9, 1570e-9, 8, device=device)
        input_signal = torch.randn(2, 8, device=device)
        output = mrr(input_signal, wavelengths)
        print("  ✅ MicroringResonator working")
    
    print("✅ Basic components test completed")
    return True
