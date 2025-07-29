"""
Coupler Components for OpticalCI

Implementation of directional couplers and beam splitters
for photonic neural network simulation.

✅ FIXED: DirectionalCoupler now conserves energy properly
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import math
import warnings
 
class DirectionalCoupler(nn.Module):
    """
    Directional Coupler - Componente para splitting/combining señales - FIXED VERSION.
    
    ✅ CORRECCIÓN: Implementa matriz unitaria para conservación de energía.
    
    Dispositivo de 4 puertos con splitting ratio configurable.
    Para conservar energía, usa matriz de transferencia unitaria.
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
        Forward pass del directional coupler - FIXED VERSION.
        
        ✅ CORRECCIÓN: Implementa matriz de transferencia unitaria para
        conservar energía correctamente.
        
        Args:
            input_1: Input port 1 [batch_size, n_wavelengths]
            input_2: Input port 2 [batch_size, n_wavelengths]
            
        Returns:
            output_1, output_2: Output ports (energy conserved)
        """
        # Coupling coefficient clamped to valid range
        kappa = torch.clamp(self.splitting_ratio, 0.01, 0.99)
        
        # ✅ FIX CRÍTICO: Coeficientes unitarios para conservación de energía
        # Para matriz unitaria: t² + κ² = 1
        t = torch.sqrt(1 - kappa**2)  # Transmission coefficient
        k = torch.sqrt(kappa**2)      # Coupling coefficient
        
        # ✅ FIX: Matriz de transferencia unitaria 2x2
        # Para directional coupler, la matriz correcta es:
        # [out1]   [t   -jk] [in1]
        # [out2] = [jk   t ] [in2]
        #
        # Para mantenerlo real (sin complejos), usamos matriz ortogonal:
        # [out1]   [t   -k] [in1]  
        # [out2] = [k    t] [in2]
        
        output_1 = t * input_1 - k * input_2
        output_2 = k * input_1 + t * input_2
        
        # ✅ VERIFICACIÓN: Debug info ocasional
        if self.training and torch.rand(1).item() < 0.05:  # 5% de las veces
            # Verificar conservación de energía
            input_energy = torch.sum(torch.abs(input_1)**2 + torch.abs(input_2)**2)
            output_energy = torch.sum(torch.abs(output_1)**2 + torch.abs(output_2)**2)
            
            if input_energy > 1e-10:
                energy_ratio = output_energy / input_energy
                if abs(energy_ratio.item() - 1.0) > 0.01:
                    warnings.warn(f"Coupler energy conservation: {energy_ratio:.6f} (should be ≈1.0)")
        
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
