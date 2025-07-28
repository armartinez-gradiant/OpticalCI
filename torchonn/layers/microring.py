"""
Microring Photonic Components for PtONN-TESTS

Implementation of microring resonators and related components
for photonic neural network simulation.
""" 

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import math
import warnings

class MicroringResonator(nn.Module):
    """
    Microring Resonator - PARÁMETROS REALISTAS v4.0
    
    ✅ CAMBIOS FUNDAMENTALES v4.0:
    - Q=5,000 (no 20,000) → resonancia visible
    - κ=0.1 (no 0.017) → ecuaciones estables  
    - Rango optimizado para UNA resonancia
    - Extinction ratio 15-20 dB realista
    """
    
    def __init__(
        self,
        radius: float = 10e-6,  # 10 μm radio
        coupling_strength: float = 0.1,  # ✅ REALISTA y estable
        q_factor: float = 5000,  # ✅ REALISTA para demos
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
        self.q_factor = q_factor
        self.center_wavelength = center_wavelength
        self.thermal_coefficient = thermal_coefficient
        self.coupling_strength = coupling_strength
        
        # Física correcta pero parámetros prácticos
        self.n_eff = 2.4   # Effective index
        self.n_g = 4.2     # Group index
        self.circumference = 2 * np.pi * radius
        
        # FSR correcto
        if fsr is None:
            self.fsr = (center_wavelength**2) / (self.n_g * self.circumference)
            print(f"🔧 FSR: {self.fsr*1e12:.1f} pm (R={radius*1e6:.1f}μm)")
        else:
            self.fsr = fsr
        
        # ✅ PARÁMETROS REALISTAS
        alpha = np.exp(-np.pi / q_factor)
        kappa_critical = np.sqrt(1 - alpha**2)
        
        print(f"   📊 Q={q_factor} (realista para demos)")
        print(f"   📊 α={alpha:.6f}")
        print(f"   📊 κ_critical={kappa_critical:.4f}, κ={coupling_strength:.4f}")
        print(f"   📊 Extinction ratio esperado: {10*np.log10(4*q_factor*(coupling_strength**2)):.1f} dB")
        
        # Parámetros entrenables
        self.phase_shift = nn.Parameter(torch.zeros(1, device=device))
        self.coupling_tuning = nn.Parameter(torch.tensor([coupling_strength], device=device))
        
        # Estado interno
        self.register_buffer('photon_energy', torch.zeros(1, device=device))
        self.register_buffer('temperature_shift', torch.zeros(1, device=device))
        
        # ✅ GARANTIZAR resonancia visible en 1550nm exacto
        self._set_exact_resonance()
    
    def _set_exact_resonance(self):
        """Configurar resonancia exacta en center_wavelength."""
        # Calcular phase para resonancia perfecta
        target_phase = 2 * np.pi * self.n_eff * self.circumference / self.center_wavelength
        
        # Ajustar para resonancia exacta (φ = 2πm)
        resonance_order = round(target_phase / (2 * np.pi))
        exact_phase = 2 * np.pi * resonance_order
        
        # Phase shift para centrar resonancia
        self.phase_shift.data.fill_(exact_phase - target_phase)
        
        print(f"   🎯 Resonancia centrada: m={resonance_order}, φ_shift={self.phase_shift.item():.4f}")
    
    def get_transmission(self, wavelengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ TRANSMISIÓN con PARÁMETROS REALISTAS v4.0
        
        Q=5000 y κ=0.1 → ecuaciones estables, resonancia visible
        """
        # Thermal shift mínimo
        thermal_shift = self.temperature_shift * self.thermal_coefficient * 0.01
        
        # ✅ FASE round-trip
        phase_round_trip = (2 * np.pi * self.n_eff * self.circumference / wavelengths) + self.phase_shift
        
        # ✅ COUPLING realista y estable
        kappa = torch.clamp(self.coupling_tuning, 0.01, 0.99)
        t = torch.sqrt(1 - kappa**2)
        
        # ✅ LOSS con Q realista
        alpha = torch.exp(torch.tensor(-np.pi / self.q_factor, device=self.device, dtype=torch.float32))
        
        # ✅ ECUACIONES ESTABLES
        cos_phi = torch.cos(phase_round_trip)
        sin_phi = torch.sin(phase_round_trip)
        
        # Denominador estable (no cerca de cero)
        denom_real = 1 - alpha * t * cos_phi
        denom_imag = alpha * t * sin_phi
        denom_magnitude_sq = denom_real**2 + denom_imag**2
        
        # THROUGH PORT 
        through_real = t - alpha * cos_phi
        through_imag = alpha * sin_phi
        through_magnitude_sq = through_real**2 + through_imag**2
        through_transmission = through_magnitude_sq / denom_magnitude_sq
        
        # DROP PORT
        drop_transmission = (kappa**2 * alpha) / denom_magnitude_sq
        
        # ✅ NO necesita renormalización con parámetros realistas
        return through_transmission, drop_transmission
    
    def apply_nonlinear_effects(self, input_power: torch.Tensor):
        """Efectos no-lineales mínimos."""
        # Efectos muy pequeños para no perturbar
        tpa_coefficient = torch.tensor(0.8e-11, device=self.device, dtype=torch.float32)
        kerr_coefficient = torch.tensor(2.7e-18, device=self.device, dtype=torch.float32)
        
        # Updates muy pequeños
        self.photon_energy += input_power * 0.0001
        
        # Thermal muy pequeño
        thermal_power = tpa_coefficient * input_power**2
        self.temperature_shift += thermal_power * 1e-6
        
        # Kerr muy pequeño
        kerr_phase = kerr_coefficient * input_power
        
        return kerr_phase
    
    def forward(self, input_signal: torch.Tensor, wavelengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass con parámetros realistas."""
        batch_size = input_signal.size(0)
        n_wavelengths = wavelengths.size(0)
        
        # Efectos no-lineales muy pequeños
        input_power = torch.abs(input_signal)**2
        kerr_phase = self.apply_nonlinear_effects(input_power.mean())
        
        # Phase shift casi nulo
        self.phase_shift.data += kerr_phase * 1e-6
        
        # Calcular transmisión
        through_trans, drop_trans = self.get_transmission(wavelengths)
        
        # Aplicar
        through_output = input_signal * through_trans.unsqueeze(0)
        drop_output = input_signal * drop_trans.unsqueeze(0)
        
        return {
            'through': through_output,
            'drop': drop_output,
            'transmission_through': through_trans,
            'transmission_drop': drop_trans
        }

class AddDropMRR(nn.Module):
    """Add-Drop Microring Resonator - PARÁMETROS REALISTAS v4.0"""

    def __init__(
        self,
        radius: float = 10e-6,
        coupling_strength_1: float = 0.1,  # ✅ REALISTA
        coupling_strength_2: float = 0.1,  # ✅ REALISTA  
        q_factor: float = 5000,  # ✅ REALISTA
        center_wavelength: float = 1550e-9,
        n_eff: float = 2.4,
        n_g: float = 4.2,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.radius = radius
        self.center_wavelength = center_wavelength
        self.n_eff = n_eff
        self.q_factor = q_factor

        print(f"🔧 Add-Drop MRR v4: R={radius*1e6:.1f}μm, Q={q_factor}")

        self.coupling_1 = nn.Parameter(torch.tensor([coupling_strength_1], device=device))
        self.coupling_2 = nn.Parameter(torch.tensor([coupling_strength_2], device=device))

        # Phase shifts
        self.phi_1 = nn.Parameter(torch.zeros(1, device=device))
        self.phi_2 = nn.Parameter(torch.zeros(1, device=device))
        self.phi_ring = nn.Parameter(torch.zeros(1, device=device))

        # Ring parameters
        self.circumference = 2 * np.pi * radius
        self.fsr = center_wavelength**2 / (n_g * self.circumference)

        # Round-trip loss
        loss_per_round_trip = 2 * np.pi / q_factor
        self.register_buffer("alpha", torch.tensor(np.exp(-loss_per_round_trip/2), device=device))

        print(f"   FSR: {self.fsr*1e12:.1f} pm")

    def get_ring_round_trip_phase(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """Calcular fase de round-trip."""
        beta = 2 * np.pi * self.n_eff / wavelengths
        phi_propagation = beta * self.circumference
        phi_total = phi_propagation + self.phi_ring
        return phi_total

    def forward(
        self, 
        input_signal: torch.Tensor,
        add_signal: torch.Tensor, 
        wavelengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass con parámetros realistas."""
        
        # Ecuaciones simplificadas y estables
        phase_rt = self.get_ring_round_trip_phase(wavelengths)
        
        kappa_1 = torch.clamp(self.coupling_1, 0.01, 0.99)
        kappa_2 = torch.clamp(self.coupling_2, 0.01, 0.99)
        
        t1 = torch.sqrt(1 - kappa_1**2)
        t2 = torch.sqrt(1 - kappa_2**2)
        
        # Transfer functions
        cos_phi = torch.cos(phase_rt)
        sin_phi = torch.sin(phase_rt)
        
        denom_real = 1 - self.alpha * t1 * t2 * cos_phi
        denom_imag = self.alpha * t1 * t2 * sin_phi
        denom_sq = denom_real**2 + denom_imag**2
        
        # Through response
        through_num_real = t1 * t2 - self.alpha * cos_phi
        through_num_imag = self.alpha * sin_phi
        through_response = (through_num_real**2 + through_num_imag**2) / denom_sq
        
        # Drop response
        drop_response = (kappa_1 * kappa_2 * self.alpha) / denom_sq
        
        # Apply to signals
        through_output = input_signal * through_response.unsqueeze(0)
        drop_output = input_signal * drop_response.unsqueeze(0) + add_signal * 0.01

        return {
            'through': through_output,
            'drop': drop_output,
            'coupling_1': self.coupling_1.detach(),
            'coupling_2': self.coupling_2.detach(), 
            'round_trip_loss': (1 - self.alpha**2).detach(),
            'fsr': self.fsr
        }

def test_basic_components():
    """Test con parámetros realistas."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🧪 Testing REALISTIC microring parameters...")
    
    # Test con Q=5000, κ=0.1
    mrr = MicroringResonator(device=device)
    
    # ✅ RANGO REALISTA: ±50pm para ver UNA resonancia clara
    wavelengths = torch.linspace(1549.95e-9, 1550.05e-9, 1000, device=device)  # ±50pm
    input_signal = torch.ones(1, 1000, device=device)
    output = mrr(input_signal, wavelengths)
    
    # Verificar extinction ratio
    through_response = output['transmission_through']
    min_val = torch.min(through_response).item()
    max_val = torch.max(through_response).item()
    
    print(f"  📊 Through: min={min_val:.4f}, max={max_val:.4f}")
    
    if min_val > 0 and max_val > min_val:
        extinction_ratio_db = 10 * np.log10(max_val / min_val)
        print(f"  📊 Extinction ratio: {extinction_ratio_db:.1f} dB")
        
        if extinction_ratio_db > 10:
            print("  🎉 ¡RESONANCIA VISIBLE!")
            return True
        else:
            print(f"  ⚠️ Extinción baja: {extinction_ratio_db:.1f} dB")
            return False
    
    return False
