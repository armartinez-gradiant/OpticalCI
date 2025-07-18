"""
MicroringResonator - Componente Fotónico
==========================================

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

class MicroringResonator(nn.Module):
    """
    Microring Resonator - Componente fundamental para filtrado y switching.
    
    Implementa la física completa de un microring con:
    - Resonancia wavelength-selective
    - Efectos no-lineales
    - Thermal tuning
    - Free carrier effects
    """
    
    def __init__(
        self,
        radius: float = 10e-6,  # 10 μm radio
        coupling_strength: float = 0.3,  # Acoplamiento
        q_factor: float = 10000,  # Factor Q
        center_wavelength: float = 1550e-9,  # Wavelength central
        fsr: float = None,  # Free Spectral Range
        thermal_coefficient: float = 8.6e-5,  # Coef. termo-óptico /K
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.radius = radius
        self.coupling_strength = coupling_strength
        self.q_factor = q_factor
        self.center_wavelength = center_wavelength
        self.thermal_coefficient = thermal_coefficient
        
        # FSR calculation: FSR = λ²/(n_g * L)
        if fsr is None:
            n_group = 4.2  # Group index for silicon
            circumference = 2 * np.pi * radius
            self.fsr = center_wavelength**2 / (n_group * circumference)
        else:
            self.fsr = fsr
        
        # Parámetros entrenables
        self.phase_shift = nn.Parameter(torch.zeros(1, device=device))
        self.coupling_tuning = nn.Parameter(torch.tensor([coupling_strength], device=device))
        
        # Estado interno (para efectos no-lineales)
        self.register_buffer('photon_energy', torch.zeros(1, device=device))
        self.register_buffer('temperature_shift', torch.zeros(1, device=device))
    
    def get_transmission(self, wavelengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcular transmisión en through y drop ports.
        
        Args:
            wavelengths: Tensor de wavelengths [n_wavelengths]
            
        Returns:
            through_transmission, drop_transmission
        """
        # Detuning from resonance
        delta_lambda = wavelengths - self.center_wavelength
        
        # Resonance condition with thermal shift
        thermal_shift = self.temperature_shift * self.thermal_coefficient
        effective_wavelength = self.center_wavelength + thermal_shift
        
        # Phase per round trip
        detuning = 2 * np.pi * delta_lambda / self.fsr
        total_phase = detuning + self.phase_shift
        
        # Coupling coefficient (adjustable)
        kappa = torch.clamp(self.coupling_tuning, 0.1, 0.9)
        
        # Transmission coefficient
        t = torch.sqrt(1 - kappa**2)
        
        # ✅ FIX: Convert alpha to tensor
        alpha_val = 1 - (np.pi / self.q_factor)
        alpha = torch.tensor(alpha_val, device=self.device, dtype=torch.float32)
        
        # Transfer function (simplified Lorentzian)
        denominator = 1 - alpha * t * torch.exp(1j * total_phase)
        
        # Through port (transmitted)
        through_transmission = torch.abs((t - alpha * torch.exp(1j * total_phase)) / denominator)**2
        
        # Drop port (coupled)
        drop_transmission = torch.abs(kappa * torch.sqrt(alpha) / denominator)**2
        
        return through_transmission, drop_transmission
    
    def apply_nonlinear_effects(self, input_power: torch.Tensor):
        """Aplicar efectos no-lineales (Kerr, TPA)."""
        # ✅ FIX: Convert constants to tensors
        tpa_coefficient = torch.tensor(0.8e-11, device=self.device, dtype=torch.float32)  # m/W for silicon
        kerr_coefficient = torch.tensor(2.7e-18, device=self.device, dtype=torch.float32)  # m²/W for silicon
        
        # Update internal state
        self.photon_energy += input_power * 0.1  # Simplified accumulation
        
        # Thermal heating from TPA
        thermal_power = tpa_coefficient * input_power**2
        self.temperature_shift += thermal_power * 0.01  # Simplified thermal model
        
        # Phase shift from Kerr effect
        kerr_phase = kerr_coefficient * input_power
        
        return kerr_phase
    
    def forward(self, input_signal: torch.Tensor, wavelengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass del microring.
        
        Args:
            input_signal: [batch_size, n_wavelengths]
            wavelengths: [n_wavelengths]
            
        Returns:
            Dict con 'through' y 'drop' outputs
        """
        batch_size = input_signal.size(0)
        n_wavelengths = wavelengths.size(0)
        
        # Aplicar efectos no-lineales
        input_power = torch.abs(input_signal)**2
        kerr_phase = self.apply_nonlinear_effects(input_power.mean())
        
        # Ajustar fase por Kerr effect
        self.phase_shift.data += kerr_phase * 0.1
        
        # Calcular transmisión para cada wavelength
        through_trans, drop_trans = self.get_transmission(wavelengths)
        
        # Aplicar a la señal de entrada
        through_output = input_signal * through_trans.unsqueeze(0)
        drop_output = input_signal * drop_trans.unsqueeze(0)
        
        return {
            'through': through_output,
            'drop': drop_output,
            'transmission_through': through_trans,
            'transmission_drop': drop_trans
        }

# Exports del módulo
__all__ = ['MicroringResonator']
