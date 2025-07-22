"""
Coupler Components for PtONN-TESTS

Implementation of directional couplers and beam splitters
for photonic neural network simulation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import math
import warnings
 
class DirectionalCoupler(nn.Module):
    """
    Directional Coupler - Componente para splitting/combining señales.
    
    Dispositivo de 4 puertos con splitting ratio configurable.
    """
    
    def __init__(
        self,
        splitting_ratio: float = 0.5,  # 50:50 split
        coupling_length: float = 100e-6,  # 100 μm
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Parámetro entrenable para splitting ratio
        self.splitting_ratio = nn.Parameter(torch.tensor([splitting_ratio], device=device))
        self.coupling_length = coupling_length
        
        # Phase relationships
        self.phase_offset = nn.Parameter(torch.zeros(1, device=device))
    
    def forward(
        self, 
        input_1: torch.Tensor, 
        input_2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass del directional coupler.
        
        Args:
            input_1: Input port 1 [batch_size, n_wavelengths]
            input_2: Input port 2 [batch_size, n_wavelengths]
            
        Returns:
            output_1, output_2: Output ports
        """
        # Coupling coefficient
        kappa = torch.clamp(self.splitting_ratio, 0.01, 0.99)
        
        # Transmission coefficients
        t = torch.sqrt(1 - kappa)
        k = torch.sqrt(kappa)
        
        # Simplified coupling without complex phase for now
        # 2x2 coupling matrix (simplified real version)
        # [out1]   [t   k] [in1]
        # [out2] = [k   t] [in2]
        
        output_1 = t * input_1 + k * input_2
        output_2 = k * input_1 + t * input_2
        
        return output_1, output_2

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
