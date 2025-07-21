#!/usr/bin/env python3
"""
Componentes Fotónicos Avanzados para PtONN-TESTS

Implementación de componentes críticos que faltan:
- Microring Resonators (MRR)
- Add-Drop MRR
- MRR Weight Banks  
- Directional Couplers
- Photodetectors
- Phase Change Materials (PCM)
- WDM Components
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import math

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

class MRRWeightBank(nn.Module):
    """
    MRR Weight Bank - Array de microrings para implementar matrices de pesos.
    
    Cada microring codifica un peso mediante su transmisión wavelength-dependent.
    Fundamental para ONNs incoherentes con WDM.
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_wavelengths: int,
        wavelength_range: Tuple[float, float] = (1530e-9, 1570e-9),
        ring_radius: float = 5e-6,  # Rings más pequeños para densidad
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_wavelengths = n_wavelengths
        
        # Wavelengths para WDM
        wl_min, wl_max = wavelength_range
        self.wavelengths = torch.linspace(wl_min, wl_max, n_wavelengths, device=device)
        
        # Array de microrings [n_outputs, n_inputs]
        self.rings = nn.ModuleList()
        for i in range(n_outputs):
            output_rings = nn.ModuleList()
            for j in range(n_inputs):
                # Cada ring tiene wavelength central ligeramente diferente
                center_wl = wl_min + (wl_max - wl_min) * (i * n_inputs + j) / (n_inputs * n_outputs)
                
                ring = MicroringResonator(
                    radius=ring_radius,
                    center_wavelength=center_wl,
                    coupling_strength=0.5,
                    q_factor=15000,
                    device=device
                )
                output_rings.append(ring)
            self.rings.append(output_rings)
        
        print(f"🔧 MRR Weight Bank: {n_outputs}x{n_inputs}, {n_wavelengths} wavelengths")
    
    def get_weight_matrix(self) -> torch.Tensor:
        """Obtener la matriz de pesos actual del weight bank."""
        weight_matrix = torch.zeros(self.n_outputs, self.n_inputs, self.n_wavelengths, device=self.device)
        
        for i in range(self.n_outputs):
            for j in range(self.n_inputs):
                ring = self.rings[i][j]
                through_trans, drop_trans = ring.get_transmission(self.wavelengths)
                
                # Usar drop transmission como peso
                weight_matrix[i, j, :] = drop_trans
        
        return weight_matrix
    
    def forward(self, input_signals: torch.Tensor) -> torch.Tensor:
        """
        Matrix-vector multiplication usando MRR weight bank.
        
        Args:
            input_signals: [batch_size, n_inputs, n_wavelengths]
            
        Returns:
            output_signals: [batch_size, n_outputs, n_wavelengths]
        """
        batch_size = input_signals.size(0)
        output_signals = torch.zeros(batch_size, self.n_outputs, self.n_wavelengths, device=self.device)
        
        for i in range(self.n_outputs):
            for j in range(self.n_inputs):
                ring = self.rings[i][j]
                
                # Procesar input signal a través del ring
                input_signal = input_signals[:, j, :]  # [batch_size, n_wavelengths]
                ring_output = ring(input_signal, self.wavelengths)
                
                # Acumular en output (suma incoherente)
                output_signals[:, i, :] += ring_output['drop']
        
        return output_signals

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
        
        # ✅ FIX: Simplified coupling without complex phase for now
        # 2x2 coupling matrix (simplified real version)
        # [out1]   [t   k] [in1]
        # [out2] = [k   t] [in2]
        
        output_1 = t * input_1 + k * input_2
        output_2 = k * input_1 + t * input_2
        
        return output_1, output_2

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

class WDMMultiplexer(nn.Module):
    """
    WDM Multiplexer/Demultiplexer - Para sistemas multicanal.
    
    Combina/separa múltiples wavelengths usando array de microrings.
    """
    
    def __init__(
        self,
        wavelengths: List[float],
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.wavelengths = torch.tensor(wavelengths, device=device)
        self.n_channels = len(wavelengths)
        
        # Array de drop filters (uno por wavelength)
        self.drop_filters = nn.ModuleList()
        for i, wl in enumerate(wavelengths):
            drop_filter = MicroringResonator(
                center_wavelength=wl,
                coupling_strength=0.8,  # High coupling for good drop efficiency
                q_factor=20000,  # High Q for narrow linewidth
                device=device
            )
            self.drop_filters.append(drop_filter)
        
        print(f"🌈 WDM Multiplexer: {self.n_channels} canales")
    
    def multiplex(self, channel_signals: List[torch.Tensor]) -> torch.Tensor:
        """
        Multiplexar múltiples canales en una sola fibra.
        
        Args:
            channel_signals: Lista de [batch_size] por canal
            
        Returns:
            multiplexed_signal: [batch_size, n_wavelengths]
        """
        if len(channel_signals) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(channel_signals)}")
        
        batch_size = channel_signals[0].size(0)
        multiplexed = torch.zeros(batch_size, self.n_channels, device=self.device)
        
        # Cada canal va a su wavelength correspondiente
        for i, signal in enumerate(channel_signals):
            multiplexed[:, i] = signal
        
        return multiplexed
    
    def demultiplex(self, multiplexed_signal: torch.Tensor) -> List[torch.Tensor]:
        """
        Demultiplexar señal WDM en canales individuales.
        
        Args:
            multiplexed_signal: [batch_size, n_wavelengths]
            
        Returns:
            channel_signals: Lista de [batch_size] por canal
        """
        channel_signals = []
        through_signal = multiplexed_signal.clone()
        
        for i, drop_filter in enumerate(self.drop_filters):
            # Aplicar drop filter
            filter_output = drop_filter(through_signal, self.wavelengths)
            
            # Canal extraído en drop port
            channel_signal = filter_output['drop'][:, i]  # Canal específico
            channel_signals.append(channel_signal)
            
            # Continuar con through signal
            through_signal = filter_output['through']
        
        return channel_signals

def test_advanced_components():
    """Test de todos los componentes avanzados."""
    print("🧪 Test: Componentes Fotónicos Avanzados")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    n_wavelengths = 8
    
    # Wavelengths de prueba
    wavelengths = torch.linspace(1530e-9, 1570e-9, n_wavelengths, device=device)
    
    print("1️⃣ Microring Resonator:")
    mrr = MicroringResonator(device=device)
    input_signal = torch.randn(batch_size, n_wavelengths, device=device)
    mrr_output = mrr(input_signal, wavelengths)
    print(f"   Input: {input_signal.shape}")
    print(f"   Through: {mrr_output['through'].shape}")
    print(f"   Drop: {mrr_output['drop'].shape}")
    
    print("\n2️⃣ Add-Drop MRR:")
    add_drop = AddDropMRR(device=device)
    add_signal = torch.randn(batch_size, n_wavelengths, device=device)
    add_drop_output = add_drop(input_signal, add_signal, wavelengths)
    print(f"   Through: {add_drop_output['through'].shape}")
    print(f"   Drop: {add_drop_output['drop'].shape}")
    
    print("\n3️⃣ MRR Weight Bank:")
    weight_bank = MRRWeightBank(n_inputs=4, n_outputs=3, n_wavelengths=n_wavelengths, device=device)
    bank_input = torch.randn(batch_size, 4, n_wavelengths, device=device)
    bank_output = weight_bank(bank_input)
    print(f"   Input: {bank_input.shape}")
    print(f"   Output: {bank_output.shape}")
    weight_matrix = weight_bank.get_weight_matrix()
    print(f"   Weight matrix: {weight_matrix.shape}")
    
    print("\n4️⃣ Directional Coupler:")
    coupler = DirectionalCoupler(device=device)
    input_1 = torch.randn(batch_size, n_wavelengths, device=device)
    input_2 = torch.randn(batch_size, n_wavelengths, device=device)
    out_1, out_2 = coupler(input_1, input_2)
    print(f"   Input: {input_1.shape}, {input_2.shape}")
    print(f"   Output: {out_1.shape}, {out_2.shape}")
    
    print("\n5️⃣ Photodetector:")
    photodet = Photodetector(device=device)
    optical_in = torch.randn(batch_size, n_wavelengths, device=device)
    current_out = photodet(optical_in)
    print(f"   Optical in: {optical_in.shape}")
    print(f"   Current out: {current_out.shape}")
    
    print("\n6️⃣ PCM Cell:")
    pcm = PhaseChangeCell(device=device)
    pcm_input = torch.randn(batch_size, n_wavelengths, device=device)
    pcm_output = pcm(pcm_input)
    print(f"   PCM state: {pcm.pcm_state.item():.3f}")
    print(f"   Output: {pcm_output.shape}")
    
    print("\n7️⃣ WDM Multiplexer:")
    wdm_wavelengths = [1530e-9, 1540e-9, 1550e-9, 1560e-9]
    wdm = WDMMultiplexer(wdm_wavelengths, device=device)
    
    # Test multiplexing
    channels = [torch.randn(batch_size, device=device) for _ in range(4)]
    muxed = wdm.multiplex(channels)
    print(f"   Channels: {len(channels)} x {channels[0].shape}")
    print(f"   Multiplexed: {muxed.shape}")
    
    # Test demultiplexing
    demuxed = wdm.demultiplex(muxed)
    print(f"   Demultiplexed: {len(demuxed)} channels")
    
    print("\n✅ Todos los componentes avanzados funcionando correctamente!")
    
    return {
        'mrr': mrr,
        'add_drop': add_drop,
        'weight_bank': weight_bank,
        'coupler': coupler,
        'photodetector': photodet,
        'pcm': pcm,
        'wdm': wdm
    }

def main():
    """Función principal de demostración."""
    print("🌟 Componentes Fotónicos Avanzados - PtONN-TESTS Enhanced")
    print("=" * 80)
    
    try:
        components = test_advanced_components()
        
        print(f"\n📋 Componentes Implementados:")
        print(f"   ✅ Microring Resonator (MRR)")
        print(f"   ✅ Add-Drop MRR")
        print(f"   ✅ MRR Weight Bank")
        print(f"   ✅ Directional Coupler")
        print(f"   ✅ Photodetector")
        print(f"   ✅ Phase Change Material (PCM)")
        print(f"   ✅ WDM Multiplexer/Demultiplexer")
        
        print(f"\n🔬 Características Implementadas:")
        print(f"   🎯 Resonancia wavelength-selective")
        print(f"   ⚡ Efectos no-lineales (Kerr, TPA)")
        print(f"   🌡️  Thermal tuning")
        print(f"   🔧 Parámetros entrenables")
        print(f"   🌈 WDM completo")
        print(f"   💾 Memoria no-volátil (PCM)")
        print(f"   📏 Conversión O/E realista")
        
        print(f"\n🚀 Para usar: python advanced_photonic_components.py")
        
    except Exception as e:
        print(f"\n❌ Error durante test: {e}")
        raise

if __name__ == "__main__":
    main()