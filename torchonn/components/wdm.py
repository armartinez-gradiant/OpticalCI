"""
WDM Components for PtONN-TESTS

Implementation of wavelength division multiplexing
and related systems for photonic neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import math
import warnings

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
