"""
PhaseChangeCell - Componente Fotónico
=======================================

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

class PhaseChangeCell(nn.Module):
    """
    Phase Change Material Cell - Para pesos reconfigurables no-volátiles.
    
    Simula GST (Ge₂Sb₂Te₅) o otros PCMs para switching óptico.
    """
    
    def __init__(
        self,
        initial_state: float = 0.0,  # 0=amorphous, 1=crystalline
        switching_energy: float = 1e-12,  # J
        retention_time: float = 10.0,  # years
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Estado del PCM (entrenable)
        self.pcm_state = nn.Parameter(torch.tensor([initial_state], device=device))
        
        self.switching_energy = switching_energy
        self.retention_time = retention_time
        
        # ✅ FIX: Convert complex numbers to tensors
        # Índices refractivos para diferentes estados
        self.n_amorphous_real = torch.tensor(5.5, device=device, dtype=torch.float32)
        self.n_amorphous_imag = torch.tensor(0.3, device=device, dtype=torch.float32)
        self.n_crystalline_real = torch.tensor(6.9, device=device, dtype=torch.float32)
        self.n_crystalline_imag = torch.tensor(0.9, device=device, dtype=torch.float32)
    
    def get_optical_properties(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtener propiedades ópticas según estado PCM."""
        # Estado normalizado [0, 1]
        state = torch.clamp(self.pcm_state, 0, 1)
        
        # Interpolación lineal entre estados
        n_real = (1 - state) * self.n_amorphous_real + state * self.n_crystalline_real
        n_imag = (1 - state) * self.n_amorphous_imag + state * self.n_crystalline_imag
        
        return n_real, n_imag
    
    def switch_state(self, energy_pulse: torch.Tensor):
        """Cambiar estado PCM con pulso de energía."""
        # ✅ FIX: Convert energy_pulse to float for comparison
        energy_val = energy_pulse.item() if torch.is_tensor(energy_pulse) else energy_pulse
        
        if energy_val > self.switching_energy:
            # Switch towards crystalline
            self.pcm_state.data = torch.clamp(self.pcm_state.data + 0.1, 0, 1)
        elif energy_val < -self.switching_energy:
            # Switch towards amorphous  
            self.pcm_state.data = torch.clamp(self.pcm_state.data - 0.1, 0, 1)
    
    def forward(self, optical_signal: torch.Tensor) -> torch.Tensor:
        """
        Aplicar modulación PCM a señal óptica.
        
        Args:
            optical_signal: [batch_size, n_wavelengths]
            
        Returns:
            modulated_signal: [batch_size, n_wavelengths]
        """
        # Obtener propiedades ópticas actuales
        n_real, n_imag = self.get_optical_properties()
        
        # ✅ FIX: Use tensor operations for complex calculations
        # Transmisión dependiente del estado PCM
        wavelength = torch.tensor(1550e-9, device=self.device, dtype=torch.float32)
        thickness = torch.tensor(100e-9, device=self.device, dtype=torch.float32)
        
        transmission = torch.exp(-4 * np.pi * n_imag / wavelength * thickness)
        phase_shift = 2 * np.pi * (n_real - 1) / wavelength * thickness
        
        # Aplicar a señal óptica (simplified - only magnitude modulation)
        modulated_signal = optical_signal * transmission
        
        return modulated_signal

# Exports del módulo
__all__ = ['PhaseChangeCell']
