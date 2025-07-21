#!/usr/bin/env python3
"""
Sistema Fotónico Completo - Demostración de Todos los Componentes

Integra todos los componentes avanzados en un sistema completo:
- Red neuronal óptica con MRR weight banks
- Sistema WDM multicanal
- Reconfigurabilidad con PCM
- Interfaz O/E con photodetectors
- Comparación con redes tradicionales
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import time
import math

# Importar nuestros componentes base
from torchonn.layers import MZIBlockLinear, MZILayer
from torchonn.models import ONNBaseModel

# ==================================================================================
# COMPONENTES FOTÓNICOS AVANZADOS - Incluidos para standalone execution
# ==================================================================================

class MicroringResonator(nn.Module):
    """Microring Resonator - Componente fundamental para filtrado y switching."""
    
    def __init__(
        self,
        radius: float = 10e-6,
        coupling_strength: float = 0.3,
        q_factor: float = 10000,
        center_wavelength: float = 1550e-9,
        fsr: float = None,
        thermal_coefficient: float = 8.6e-5,
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
        
        if fsr is None:
            n_group = 4.2
            circumference = 2 * np.pi * radius
            self.fsr = center_wavelength**2 / (n_group * circumference)
        else:
            self.fsr = fsr
        
        self.phase_shift = nn.Parameter(torch.zeros(1, device=device))
        self.coupling_tuning = nn.Parameter(torch.tensor([coupling_strength], device=device))
        
        self.register_buffer('photon_energy', torch.zeros(1, device=device))
        self.register_buffer('temperature_shift', torch.zeros(1, device=device))
    
    def get_transmission(self, wavelengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcular transmisión en through y drop ports."""
        delta_lambda = wavelengths - self.center_wavelength
        thermal_shift = self.temperature_shift * self.thermal_coefficient
        effective_wavelength = self.center_wavelength + thermal_shift
        
        detuning = 2 * np.pi * delta_lambda / self.fsr
        total_phase = detuning + self.phase_shift
        
        kappa = torch.clamp(self.coupling_tuning, 0.1, 0.9)
        t = torch.sqrt(1 - kappa**2)
        
        # ✅ FIX: Convert alpha to tensor
        alpha_val = 1 - (np.pi / self.q_factor)
        alpha = torch.tensor(alpha_val, device=self.device, dtype=torch.float32)
        
        denominator = 1 - alpha * t * torch.exp(1j * total_phase)
        through_transmission = torch.abs((t - alpha * torch.exp(1j * total_phase)) / denominator)**2
        drop_transmission = torch.abs(kappa * torch.sqrt(alpha) / denominator)**2
        
        return through_transmission, drop_transmission
    
    def apply_nonlinear_effects(self, input_power: torch.Tensor):
        """Aplicar efectos no-lineales."""
        # ✅ FIX: Convert constants to tensors
        tpa_coefficient = torch.tensor(0.8e-11, device=self.device, dtype=torch.float32)
        kerr_coefficient = torch.tensor(2.7e-18, device=self.device, dtype=torch.float32)
        
        self.photon_energy += input_power * 0.1
        thermal_power = tpa_coefficient * input_power**2
        self.temperature_shift += thermal_power * 0.01
        kerr_phase = kerr_coefficient * input_power
        
        return kerr_phase
    
    def forward(self, input_signal: torch.Tensor, wavelengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass del microring."""
        batch_size = input_signal.size(0)
        input_power = torch.abs(input_signal)**2
        kerr_phase = self.apply_nonlinear_effects(input_power.mean())
        
        self.phase_shift.data += kerr_phase * 0.1
        through_trans, drop_trans = self.get_transmission(wavelengths)
        
        through_output = input_signal * through_trans.unsqueeze(0)
        drop_output = input_signal * drop_trans.unsqueeze(0)
        
        return {
            'through': through_output,
            'drop': drop_output,
            'transmission_through': through_trans,
            'transmission_drop': drop_trans
        }

class MRRWeightBank(nn.Module):
    """MRR Weight Bank - Array de microrings para implementar matrices de pesos."""
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_wavelengths: int,
        wavelength_range: Tuple[float, float] = (1530e-9, 1570e-9),
        ring_radius: float = 5e-6,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_wavelengths = n_wavelengths
        
        wl_min, wl_max = wavelength_range
        self.wavelengths = torch.linspace(wl_min, wl_max, n_wavelengths, device=device)
        
        self.rings = nn.ModuleList()
        for i in range(n_outputs):
            output_rings = nn.ModuleList()
            for j in range(n_inputs):
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
    
    def forward(self, input_signals: torch.Tensor) -> torch.Tensor:
        """Matrix-vector multiplication usando MRR weight bank."""
        batch_size = input_signals.size(0)
        output_signals = torch.zeros(batch_size, self.n_outputs, self.n_wavelengths, device=self.device)
        
        for i in range(self.n_outputs):
            for j in range(self.n_inputs):
                ring = self.rings[i][j]
                input_signal = input_signals[:, j, :]
                ring_output = ring(input_signal, self.wavelengths)
                output_signals[:, i, :] += ring_output['drop']
        
        return output_signals

class PhaseChangeCell(nn.Module):
    """Phase Change Material Cell - Para pesos reconfigurables no-volátiles."""
    
    def __init__(
        self,
        initial_state: float = 0.0,
        switching_energy: float = 1e-12,
        retention_time: float = 10.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.pcm_state = nn.Parameter(torch.tensor([initial_state], device=device))
        self.switching_energy = switching_energy
        self.retention_time = retention_time
        
        # ✅ FIX: Convert complex numbers to tensors
        self.n_amorphous_real = torch.tensor(5.5, device=device, dtype=torch.float32)
        self.n_amorphous_imag = torch.tensor(0.3, device=device, dtype=torch.float32)
        self.n_crystalline_real = torch.tensor(6.9, device=device, dtype=torch.float32)
        self.n_crystalline_imag = torch.tensor(0.9, device=device, dtype=torch.float32)
    
    def get_optical_properties(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtener propiedades ópticas según estado PCM."""
        state = torch.clamp(self.pcm_state, 0, 1)
        n_real = (1 - state) * self.n_amorphous_real + state * self.n_crystalline_real
        n_imag = (1 - state) * self.n_amorphous_imag + state * self.n_crystalline_imag
        return n_real, n_imag
    
    def switch_state(self, energy_pulse: torch.Tensor):
        """Cambiar estado PCM con pulso de energía."""
        energy_val = energy_pulse.item() if torch.is_tensor(energy_pulse) else energy_pulse
        
        if energy_val > self.switching_energy:
            self.pcm_state.data = torch.clamp(self.pcm_state.data + 0.1, 0, 1)
        elif energy_val < -self.switching_energy:
            self.pcm_state.data = torch.clamp(self.pcm_state.data - 0.1, 0, 1)
    
    def forward(self, optical_signal: torch.Tensor) -> torch.Tensor:
        """Aplicar modulación PCM a señal óptica."""
        n_real, n_imag = self.get_optical_properties()
        
        # ✅ FIX: Use tensor operations
        wavelength = torch.tensor(1550e-9, device=self.device, dtype=torch.float32)
        thickness = torch.tensor(100e-9, device=self.device, dtype=torch.float32)
        
        transmission = torch.exp(-4 * np.pi * n_imag / wavelength * thickness)
        modulated_signal = optical_signal * transmission
        
        return modulated_signal

class Photodetector(nn.Module):
    """Photodetector - Conversión óptico-eléctrica."""
    
    def __init__(
        self,
        responsivity: float = 1.0,
        dark_current: float = 1e-9,
        thermal_noise: float = 1e-12,
        bandwidth: float = 10e9,
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
        """Convertir señal óptica a eléctrica."""
        optical_power = torch.abs(optical_signal)**2
        photocurrent = self.responsivity * optical_power
        photocurrent += self.dark_current
        
        if self.training:
            noise_std = torch.sqrt(torch.tensor(self.thermal_noise * self.bandwidth, device=self.device))
            thermal_noise = torch.randn_like(photocurrent) * noise_std
            photocurrent += thermal_noise
        
        return photocurrent

class WDMMultiplexer(nn.Module):
    """WDM Multiplexer/Demultiplexer - Para sistemas multicanal."""
    
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
        
        self.drop_filters = nn.ModuleList()
        for i, wl in enumerate(wavelengths):
            drop_filter = MicroringResonator(
                center_wavelength=wl,
                coupling_strength=0.8,
                q_factor=20000,
                device=device
            )
            self.drop_filters.append(drop_filter)
        
        print(f"🌈 WDM Multiplexer: {self.n_channels} canales")
    
    def multiplex(self, channel_signals: List[torch.Tensor]) -> torch.Tensor:
        """Multiplexar múltiples canales en una sola fibra."""
        if len(channel_signals) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(channel_signals)}")
        
        batch_size = channel_signals[0].size(0)
        multiplexed = torch.zeros(batch_size, self.n_channels, device=self.device)
        
        for i, signal in enumerate(channel_signals):
            multiplexed[:, i] = signal
        
        return multiplexed
    
    def demultiplex(self, multiplexed_signal: torch.Tensor) -> List[torch.Tensor]:
        """Demultiplexar señal WDM en canales individuales."""
        channel_signals = []
        through_signal = multiplexed_signal.clone()
        
        for i, drop_filter in enumerate(self.drop_filters):
            filter_output = drop_filter(through_signal, self.wavelengths)
            channel_signal = filter_output['drop'][:, i]
            channel_signals.append(channel_signal)
            through_signal = filter_output['through']
        
        return channel_signals

# ==================================================================================
# SISTEMAS COMPLETOS
# ==================================================================================

class CompletePhotonicNeuralNetwork(ONNBaseModel):
    """
    Red Neuronal Óptica Completa usando todos los componentes avanzados.
    
    Arquitectura híbrida que combina:
    - MRR weight banks para capas densas
    - WDM para paralelismo
    - PCM para reconfigurabilidad
    - Photodetectors para output
    """
    
    def __init__(
        self,
        input_size: int = 16,
        hidden_sizes: List[int] = [12, 8],
        output_size: int = 4,
        n_wavelengths: int = 8,
        use_pcm_weights: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__(device=device)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.n_wavelengths = n_wavelengths
        self.use_pcm_weights = use_pcm_weights
        
        # Wavelengths para WDM
        self.wavelengths = torch.linspace(1530e-9, 1570e-9, n_wavelengths, device=self.device)
        
        # Construir la red fotónica
        self._build_photonic_network()
        
        print(f"🌟 Red Neuronal Óptica Completa:")
        print(f"   📏 Arquitectura: {input_size} → {hidden_sizes} → {output_size}")
        print(f"   🌈 Wavelengths: {n_wavelengths}")
        print(f"   💾 PCM weights: {'✅' if use_pcm_weights else '❌'}")
    
    def _build_photonic_network(self):
        """Construir la red fotónica completa."""
        
        # 1. Input preprocessing con WDM
        self.input_wdm = WDMMultiplexer(
            wavelengths=self.wavelengths.tolist(),
            device=self.device
        )
        
        # 2. Capas de MRR weight banks
        self.mrr_layers = nn.ModuleList()
        prev_size = self.input_size
        
        for hidden_size in self.hidden_sizes:
            weight_bank = MRRWeightBank(
                n_inputs=prev_size,
                n_outputs=hidden_size,
                n_wavelengths=self.n_wavelengths,
                device=self.device
            )
            self.mrr_layers.append(weight_bank)
            prev_size = hidden_size
        
        # Output layer
        output_bank = MRRWeightBank(
            n_inputs=prev_size,
            n_outputs=self.output_size,
            n_wavelengths=self.n_wavelengths,
            device=self.device
        )
        self.mrr_layers.append(output_bank)
        
        # 3. PCM cells para reconfigurabilidad (si habilitado)
        if self.use_pcm_weights:
            self.pcm_cells = nn.ModuleList()
            for layer in self.mrr_layers:
                layer_pcm = nn.ModuleList()
                for i in range(layer.n_outputs):
                    for j in range(layer.n_inputs):
                        pcm_cell = PhaseChangeCell(device=self.device)
                        layer_pcm.append(pcm_cell)
                self.pcm_cells.append(layer_pcm)
        
        # 4. Nonlinear activation usando MRR nonlinearity
        self.activation_rings = nn.ModuleList()
        for _ in range(len(self.mrr_layers) - 1):  # No activation en output layer
            activation_ring = MicroringResonator(
                q_factor=5000,  # Lower Q for broader nonlinearity
                coupling_strength=0.7,
                device=self.device
            )
            self.activation_rings.append(activation_ring)
        
        # 5. Output photodetectors
        self.photodetectors = nn.ModuleList()
        for _ in range(self.output_size):
            photodet = Photodetector(
                responsivity=0.8,
                device=self.device
            )
            self.photodetectors.append(photodet)
        
        # 6. Output WDM demultiplexer
        self.output_wdm = WDMMultiplexer(
            wavelengths=self.wavelengths.tolist(),
            device=self.device
        )
    
    def apply_pcm_modulation(self, layer_idx: int, layer_input: torch.Tensor) -> torch.Tensor:
        """Aplicar modulación PCM a una capa."""
        if not self.use_pcm_weights or layer_idx >= len(self.pcm_cells):
            return layer_input
        
        modulated_input = layer_input.clone()
        layer_pcm = self.pcm_cells[layer_idx]
        
        # Aplicar PCM modulation (simplificado)
        for pcm_idx, pcm_cell in enumerate(layer_pcm):
            if pcm_idx < modulated_input.size(1) * modulated_input.size(2):
                i = pcm_idx // modulated_input.size(2)
                j = pcm_idx % modulated_input.size(2)
                
                if i < modulated_input.size(1) and j < modulated_input.size(2):
                    modulated_input[:, i, j] = pcm_cell(modulated_input[:, i, j:j+1]).squeeze(-1)
        
        return modulated_input
    
    def optical_activation(self, x: torch.Tensor, ring_idx: int) -> torch.Tensor:
        """Aplicar activación no-lineal usando MRR."""
        if ring_idx >= len(self.activation_rings):
            return x
        
        activation_ring = self.activation_rings[ring_idx]
        batch_size, n_features, n_wl = x.shape
        
        activated = torch.zeros_like(x)
        
        for i in range(n_features):
            feature_signal = x[:, i, :]  # [batch_size, n_wavelengths]
            ring_output = activation_ring(feature_signal, self.wavelengths)
            
            # Usar respuesta no-lineal del ring como activación
            activated[:, i, :] = ring_output['through']
        
        return activated
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass completo a través de la red fotónica.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            output: [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # 1. Expandir input a múltiples wavelengths
        # Cada feature se mapea a diferentes wavelengths
        current = torch.zeros(batch_size, self.input_size, self.n_wavelengths, device=self.device)
        
        for i in range(self.input_size):
            # Distribuir cada input feature across wavelengths
            for j in range(self.n_wavelengths):
                current[:, i, j] = x[:, i] * (0.5 + 0.5 * np.sin(2 * np.pi * j / self.n_wavelengths))
        
        # 2. Procesar a través de capas MRR
        for layer_idx, mrr_layer in enumerate(self.mrr_layers):
            
            # Aplicar modulación PCM
            current = self.apply_pcm_modulation(layer_idx, current)
            
            # Procesar a través de MRR weight bank
            current = mrr_layer(current)
            
            # Aplicar activación óptica (excepto en output layer)
            if layer_idx < len(self.activation_rings):
                current = self.optical_activation(current, layer_idx)
        
        # 3. Photodetection para convertir a señales eléctricas
        electrical_outputs = torch.zeros(batch_size, self.output_size, device=self.device)
        
        for i in range(self.output_size):
            photodet = self.photodetectors[i]
            optical_signal = current[:, i, :]  # [batch_size, n_wavelengths]
            
            # Photodetection
            photocurrent = photodet(optical_signal)
            
            # Integrar sobre wavelengths
            electrical_outputs[:, i] = photocurrent.sum(dim=1)
        
        return electrical_outputs
    
    def get_network_analysis(self) -> Dict:
        """Análisis completo de la red."""
        analysis = {
            'architecture': 'Complete Photonic Neural Network',
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'mrr_layers': len(self.mrr_layers),
            'total_microrings': sum(layer.n_inputs * layer.n_outputs for layer in self.mrr_layers),
            'wavelengths': self.n_wavelengths,
            'pcm_cells': len([p for pcm_layer in self.pcm_cells for p in pcm_layer]) if self.use_pcm_weights else 0,
            'photodetectors': len(self.photodetectors),
        }
        
        return analysis

class HybridPhotonicSystem(nn.Module):
    """
    Sistema Fotónico Híbrido que combina componentes ópticos y electrónicos.
    
    Demuestra co-procesamiento óptico-electrónico optimizado.
    """
    
    def __init__(
        self,
        input_size: int = 32,
        optical_hidden_size: int = 16,
        electronic_hidden_size: int = 8,
        output_size: int = 4,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Parte óptica - procesamiento inicial rápido
        self.optical_processor = CompletePhotonicNeuralNetwork(
            input_size=input_size,
            hidden_sizes=[optical_hidden_size],
            output_size=optical_hidden_size,
            n_wavelengths=4,
            use_pcm_weights=False,
            device=device
        )
        
        # Parte electrónica - procesamiento de precisión
        self.electronic_processor = nn.Sequential(
            nn.Linear(optical_hidden_size, electronic_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(electronic_hidden_size, output_size)
        ).to(device)
        
        print(f"🔀 Sistema Híbrido Óptico-Electrónico:")
        print(f"   ⚡ Óptico: {input_size} → {optical_hidden_size}")
        print(f"   🔧 Electrónico: {optical_hidden_size} → {output_size}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass híbrido.
        
        Returns:
            optical_output, final_output
        """
        # Procesamiento óptico (rápido, paralelo)
        optical_output = self.optical_processor(x)
        
        # Procesamiento electrónico (preciso, flexible)
        final_output = self.electronic_processor(optical_output)
        
        return optical_output, final_output

def test_complete_photonic_system():
    """Test del sistema fotónico completo."""
    print("🌟 Test: Sistema Fotónico Neural Completo")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear red fotónica completa
    photonic_net = CompletePhotonicNeuralNetwork(
        input_size=8,
        hidden_sizes=[6, 4],
        output_size=3,
        n_wavelengths=4,
        use_pcm_weights=True,
        device=device
    )
    
    # Análisis de la red
    analysis = photonic_net.get_network_analysis()
    print(f"\n📊 Análisis de la Red:")
    for key, value in analysis.items():
        print(f"   {key}: {value}")
    
    # Test de forward pass
    batch_size = 10
    test_input = torch.randn(batch_size, 8, device=device)
    
    print(f"\n🔄 Test de Forward Pass:")
    print(f"   Input: {test_input.shape}")
    
    start_time = time.time()
    with torch.no_grad():
        output = photonic_net(test_input)
    forward_time = time.time() - start_time
    
    print(f"   Output: {output.shape}")
    print(f"   Forward time: {forward_time*1000:.2f}ms")
    print(f"   Throughput: {batch_size/forward_time:.1f} samples/s")
    
    return photonic_net

def test_hybrid_system():
    """Test del sistema híbrido óptico-electrónico."""
    print("\n🔀 Test: Sistema Híbrido Óptico-Electrónico")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear sistema híbrido
    hybrid_system = HybridPhotonicSystem(
        input_size=16,
        optical_hidden_size=8,
        electronic_hidden_size=6,
        output_size=4,
        device=device
    )
    
    # Test de rendimiento
    batch_size = 32
    test_input = torch.randn(batch_size, 16, device=device)
    
    print(f"\n⚡ Test de Rendimiento:")
    print(f"   Input: {test_input.shape}")
    
    start_time = time.time()
    with torch.no_grad():
        optical_out, final_out = hybrid_system(test_input)
    processing_time = time.time() - start_time
    
    print(f"   Optical output: {optical_out.shape}")
    print(f"   Final output: {final_out.shape}")
    print(f"   Processing time: {processing_time*1000:.2f}ms")
    print(f"   Throughput: {batch_size/processing_time:.1f} samples/s")
    
    return hybrid_system

def compare_with_traditional():
    """Comparar con redes neuronales tradicionales."""
    print("\n📊 Comparación: Fotónico vs Tradicional")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_size, hidden_size, output_size = 16, 8, 4
    batch_size = 100
    
    # Red fotónica
    photonic_net = CompletePhotonicNeuralNetwork(
        input_size=input_size,
        hidden_sizes=[hidden_size],
        output_size=output_size,
        n_wavelengths=4,
        use_pcm_weights=False,
        device=device
    )
    
    # Red tradicional
    traditional_net = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    ).to(device)
    
    # Datos de prueba
    test_data = torch.randn(batch_size, input_size, device=device)
    
    # Benchmark fotónico
    photonic_net.eval()
    start_time = time.time()
    with torch.no_grad():
        photonic_output = photonic_net(test_data)
    photonic_time = time.time() - start_time
    
    # Benchmark tradicional
    traditional_net.eval()
    start_time = time.time()
    with torch.no_grad():
        traditional_output = traditional_net(test_data)
    traditional_time = time.time() - start_time
    
    # Estadísticas
    photonic_params = sum(p.numel() for p in photonic_net.parameters())
    traditional_params = sum(p.numel() for p in traditional_net.parameters())
    
    photonic_throughput = batch_size / photonic_time
    traditional_throughput = batch_size / traditional_time
    
    print(f"📈 Resultados:")
    print(f"   Fotónico:")
    print(f"     Tiempo: {photonic_time*1000:.2f}ms")
    print(f"     Throughput: {photonic_throughput:.1f} samples/s")
    print(f"     Parámetros: {photonic_params:,}")
    print(f"     Output shape: {photonic_output.shape}")
    
    print(f"   Tradicional:")
    print(f"     Tiempo: {traditional_time*1000:.2f}ms")
    print(f"     Throughput: {traditional_throughput:.1f} samples/s")  
    print(f"     Parámetros: {traditional_params:,}")
    print(f"     Output shape: {traditional_output.shape}")
    
    print(f"   Comparación:")
    print(f"     Speedup: {traditional_time/photonic_time:.2f}x")
    print(f"     Param ratio: {photonic_params/traditional_params:.2f}x")
    
    return {
        'photonic_time': photonic_time,
        'traditional_time': traditional_time,
        'photonic_throughput': photonic_throughput,
        'traditional_throughput': traditional_throughput
    }

def demonstrate_advanced_features():
    """Demostrar características avanzadas únicas de fotónica."""
    print("\n🔬 Demostración: Características Avanzadas")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("1️⃣ Paralelismo WDM:")
    # Procesar múltiples inputs simultáneamente en diferentes wavelengths
    wdm_system = WDMMultiplexer([1530e-9, 1540e-9, 1550e-9, 1560e-9], device=device)
    
    parallel_inputs = [torch.randn(5, device=device) for _ in range(4)]
    muxed = wdm_system.multiplex(parallel_inputs)
    demuxed = wdm_system.demultiplex(muxed)
    
    print(f"   Inputs paralelos: {len(parallel_inputs)} canales")
    print(f"   Multiplexed: {muxed.shape}")
    print(f"   Demultiplexed: {len(demuxed)} canales")
    
    print("\n2️⃣ Reconfigurabilidad PCM:")
    pcm_cell = PhaseChangeCell(device=device)
    
    print(f"   Estado inicial: {pcm_cell.pcm_state.item():.3f}")
    
    # Simular switching
    energy_pulse = torch.tensor(2e-12, device=device)  # ✅ FIX: Create tensor
    pcm_cell.switch_state(energy_pulse)  # Above threshold
    print(f"   Después de switch: {pcm_cell.pcm_state.item():.3f}")
    
    print("\n3️⃣ Efectos No-lineales:")
    nonlinear_ring = MicroringResonator(q_factor=8000, device=device)
    
    # Test con diferentes potencias
    wavelengths = torch.tensor([1550e-9], device=device)
    
    for power_level in [0.1, 1.0, 10.0]:
        input_signal = torch.ones(1, 1, device=device) * power_level
        output = nonlinear_ring(input_signal, wavelengths)
        
        transmission = output['transmission_through'].item()
        print(f"   Potencia {power_level:4.1f}: Transmisión = {transmission:.3f}")
    
    print("\n4️⃣ Conversión O/E:")
    photodet = Photodetector(responsivity=0.8, device=device)
    
    optical_powers = torch.tensor([0.1, 1.0, 10.0], device=device).unsqueeze(0)
    photocurrents = photodet(optical_powers)
    
    print(f"   Potencias ópticas: {optical_powers.numpy()}")
    print(f"   Fotocorrientes: {photocurrents.detach().numpy()}")

def visualize_system_performance():
    """Visualizar rendimiento del sistema."""
    print("\n📊 Generando Visualización del Sistema...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Ejecutar comparaciones
        comparison_results = compare_with_traditional()
        
        # Crear visualización
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Comparación de throughput
        systems = ['Fotónico', 'Tradicional']
        throughputs = [comparison_results['photonic_throughput'], 
                      comparison_results['traditional_throughput']]
        
        ax1.bar(systems, throughputs, color=['lightblue', 'lightcoral'])
        ax1.set_title('Throughput Comparison')
        ax1.set_ylabel('Samples/s')
        
        # 2. Comparación de tiempo
        times = [comparison_results['photonic_time']*1000, 
                comparison_results['traditional_time']*1000]
        
        ax2.bar(systems, times, color=['lightgreen', 'lightyellow'])
        ax2.set_title('Processing Time')
        ax2.set_ylabel('Time (ms)')
        
        # 3. Respuesta espectral simulada
        wavelengths_nm = np.linspace(1530, 1570, 100)
        
        # Simular respuesta MRR
        center_wl = 1550
        q_factor = 10000
        fsr = 10  # nm
        
        detuning = 2 * np.pi * (wavelengths_nm - center_wl) / fsr
        transmission = 1 / (1 + (2 * q_factor * np.sin(detuning/2))**2)
        
        ax3.plot(wavelengths_nm, transmission, 'b-', label='Through port')
        ax3.plot(wavelengths_nm, 1-transmission, 'r-', label='Drop port')
        ax3.set_title('MRR Spectral Response')
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Transmission')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Información del sistema
        ax4.axis('off')
        ax4.text(0.1, 0.9, 'Sistema Fotónico Completo', fontsize=14, fontweight='bold')
        
        # ✅ FIX: Remove emojis for matplotlib compatibility
        info_text = [
            '[OK] Microring Resonators',
            '[OK] Add-Drop MRR', 
            '[OK] MRR Weight Banks',
            '[OK] WDM Multiplexing',
            '[OK] PCM Reconfigurability',
            '[OK] Photodetection',
            '[OK] Nonlinear Effects',
            '[OK] Hybrid O/E Processing'
        ]
        
        for i, text in enumerate(info_text):
            ax4.text(0.1, 0.8 - i*0.08, text, fontsize=10)
        
        plt.tight_layout()
        
        # Guardar
        import os
        output_dir = "examples"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, "complete_photonic_system_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualización guardada en: {output_path}")
        
    except ImportError:
        print("⚠️  Matplotlib no disponible - saltando visualización")
        print("💡 Para instalar: pip install matplotlib")
    except Exception as e:
        print(f"⚠️  Error en visualización: {e}")
        print("📊 Continuando sin visualización...")

def main():
    """Función principal de demostración."""
    print("🌟 Sistema Fotónico Neural Completo - PtONN-TESTS Ultimate")
    print("=" * 80)
    
    try:
        # 1. Test sistema fotónico completo
        photonic_net = test_complete_photonic_system()
        
        # 2. Test sistema híbrido
        hybrid_system = test_hybrid_system()
        
        # 3. Comparación con tradicional
        comparison = compare_with_traditional()
        
        # 4. Características avanzadas
        demonstrate_advanced_features()
        
        # 5. Visualización
        visualize_system_performance()
        
        print(f"\n🎉 Sistema Fotónico Completo Implementado!")
        print(f"\n📋 Componentes Integrados:")
        print(f"   ✅ Redes Neuronales Ópticas completas")
        print(f"   ✅ Sistemas Híbridos O/E")
        print(f"   ✅ Paralelismo WDM")
        print(f"   ✅ Reconfigurabilidad PCM")
        print(f"   ✅ Efectos no-lineales")
        print(f"   ✅ Photodetection realista")
        print(f"   ✅ Comparación con métodos tradicionales")
        
        print(f"\n🚀 Para ejecutar: python examples/complete_photonic_system.py")
        
    except Exception as e:
        print(f"\n❌ Error durante demostración: {e}")
        raise

if __name__ == "__main__":
    main()