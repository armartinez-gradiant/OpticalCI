#!/usr/bin/env python3
"""
🌟 Ejemplo Actualizado: Componentes Fotónicos - PtONN-TESTS

Demostración de uso de los componentes fotónicos migrados a la estructura principal.
"""

import torch
from torchonn.layers import (
    MZILayer, MZIBlockLinear,
    MicroringResonator, AddDropMRR, 
    DirectionalCoupler, Photodetector
)
from torchonn.components import (
    PhaseChangeCell, WDMMultiplexer, MRRWeightBank
)

def demo_basic_components():
    """Demostración de componentes básicos."""
    print("🔧 Demo: Componentes Básicos")
    print("-" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    n_wavelengths = 8
    
    # Wavelengths de prueba
    wavelengths = torch.linspace(1530e-9, 1570e-9, n_wavelengths, device=device)
    
    print("1️⃣ MicroringResonator:")
    mrr = MicroringResonator(device=device)
    input_signal = torch.randn(batch_size, n_wavelengths, device=device)
    mrr_output = mrr(input_signal, wavelengths)
    print(f"   Through: {mrr_output['through'].shape}")
    print(f"   Drop: {mrr_output['drop'].shape}")
    
    print("\n2️⃣ Add-Drop MRR:")
    add_drop = AddDropMRR(device=device)
    add_signal = torch.randn(batch_size, n_wavelengths, device=device)
    add_drop_output = add_drop(input_signal, add_signal, wavelengths)
    print(f"   Through: {add_drop_output['through'].shape}")
    print(f"   Drop: {add_drop_output['drop'].shape}")
    
    print("\n3️⃣ DirectionalCoupler:")
    coupler = DirectionalCoupler(device=device)
    input_1 = torch.randn(batch_size, n_wavelengths, device=device)
    input_2 = torch.randn(batch_size, n_wavelengths, device=device)
    out_1, out_2 = coupler(input_1, input_2)
    print(f"   Output 1: {out_1.shape}")
    print(f"   Output 2: {out_2.shape}")

def demo_specialized_components():
    """Demostración de componentes especializados."""
    print("\n🔬 Demo: Componentes Especializados") 
    print("-" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    
    print("1️⃣ PhaseChangeCell:")
    pcm = PhaseChangeCell(device=device)
    pcm_input = torch.randn(batch_size, 8, device=device)
    pcm_output = pcm(pcm_input)
    print(f"   PCM state: {pcm.pcm_state.item():.3f}")
    print(f"   Output: {pcm_output.shape}")
    
    print("\n2️⃣ WDMMultiplexer:")
    wdm_wavelengths = [1530e-9, 1540e-9, 1550e-9, 1560e-9]
    wdm = WDMMultiplexer(wdm_wavelengths, device=device)
    channels = [torch.randn(batch_size, device=device) for _ in range(4)]
    muxed = wdm.multiplex(channels)
    print(f"   Multiplexed: {muxed.shape}")

def main():
    """Función principal."""
    print("🌟 Componentes Fotónicos - PtONN-TESTS")
    print("=" * 60)
    print("✅ Usando nueva estructura de importaciones!")
    print()
    
    try:
        demo_basic_components()
        demo_specialized_components()
        
        print("\n🎉 ¡Todos los componentes funcionando correctamente!")
        print("\n📚 Para más información:")
        print("   - Documentación: torchonn.layers y torchonn.components")
        print("   - Tests: python test_migration.py")
        
    except Exception as e:
        print(f"\n❌ Error durante demo: {e}")
        raise

if __name__ == "__main__":
    main()
