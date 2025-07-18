"""
AddDropMRR - Componente Fotónico
==================================

Componente extraído y refactorizado del sistema PtONN-TESTS.
Parte del framework TorchONN para redes neuronales ópticas.

Autor: PtONN-TESTS Team
Fecha: 2025-07-17
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_component import BasePhotonicComponent
from .microring_resonator import MicroringResonator

class AddDropMRR(nn.Module):
    """
    Add-Drop Microring Resonator - Configuración de 4 puertos.
    
    Puertos: Input, Through, Add, Drop
    Usado para WDM add/drop multiplexing.
    """
    
    def __init__(
        self,
        radius: float = 10e-6,
        coupling_strength_1: float = 0.3,  # Input-ring coupling
        coupling_strength_2: float = 0.3,  # Ring-drop coupling
        **kwargs
    ):
        super().__init__()
        
        self.coupling_1 = coupling_strength_1
        self.coupling_2 = coupling_strength_2
        
        # Microring central
        self.ring = MicroringResonator(
            radius=radius,
            coupling_strength=coupling_strength_1,
            **kwargs
        )
        
        # Segundo acoplador para drop port
        self.coupling_2_param = nn.Parameter(torch.tensor([coupling_strength_2], device=self.ring.device))
    
    def forward(
        self, 
        input_signal: torch.Tensor,
        add_signal: torch.Tensor,
        wavelengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass del Add-Drop MRR.
        
        Args:
            input_signal: Input port [batch_size, n_wavelengths]
            add_signal: Add port [batch_size, n_wavelengths]  
            wavelengths: [n_wavelengths]
            
        Returns:
            Dict con 'through' y 'drop' outputs
        """
        # Ring response para input
        ring_response = self.ring(input_signal, wavelengths)
        
        # Add signal se combina en el ring
        kappa_2 = torch.clamp(self.coupling_2_param, 0.1, 0.9)
        
        # Combinar add signal con ring
        combined_in_ring = ring_response['drop'] + add_signal * torch.sqrt(kappa_2)
        
        # Output ports
        through_output = ring_response['through']
        drop_output = combined_in_ring * torch.sqrt(1 - kappa_2**2)
        
        return {
            'through': through_output,
            'drop': drop_output
        }

# Exports del módulo
__all__ = ['AddDropMRR']
