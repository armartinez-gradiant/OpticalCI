#!/usr/bin/env python3
"""
🧪 Test Rápido - Verificar AddDropMRR Fixes
"""
import torch
import sys
from pathlib import Path

def quick_test():
    print("🧪 QUICK TEST - AddDropMRR Fixes")
    print("=" * 40)
    
    try:
        # Import from the fixed file
        # sys.path.append("examples")  # No longer needed
        # Updated imports:
# from torchonn.layers import AddDropMRR  # or from torchonn.components import AddDropMRR
        
        # Create with fixed parameters
        add_drop = AddDropMRR(
            radius=10e-6,
            coupling_strength_1=0.05,  # Lower for better extinction
            coupling_strength_2=0.05,  # Lower for better extinction
            q_factor=15000
        )
        
        # Test wavelengths around resonance
        wavelengths = torch.linspace(1549e-9, 1551e-9, 101)  # Higher resolution
        
        # Get transfer function
        through_tf, drop_tf = add_drop.get_transfer_function(wavelengths)
        
        # Calculate metrics
        extinction_ratio = add_drop.get_extinction_ratio(wavelengths)
        q_measured = add_drop.get_q_factor_measured(wavelengths)
        
        print(f"✅ FSR: {add_drop.fsr*1e12:.1f} pm")
        print(f"✅ Extinction ratio: {extinction_ratio:.1f} dB")
        print(f"✅ Q measured: {q_measured:.0f}")
        print(f"✅ Q design: {add_drop.q_factor}")
        
        print(f"✅ Through range: {through_tf.min():.3f} - {through_tf.max():.3f}")
        print(f"✅ Drop range: {drop_tf.min():.3f} - {drop_tf.max():.3f}")
        
        if extinction_ratio > 5:
            print("🎉 Extinction ratio IMPROVED!")
        if 5000 < q_measured < 50000:
            print("🎉 Q factor measurement IMPROVED!")
        
        # Test forward pass
        print("\n🔧 Testing forward pass...")
        batch_size = 2
        n_wavelengths = len(wavelengths)
        
        input_signal = torch.randn(batch_size, n_wavelengths) * 0.1
        add_signal = torch.randn(batch_size, n_wavelengths) * 0.01
        
        output = add_drop(input_signal, add_signal, wavelengths)
        
        print(f"✅ Input signal shape: {input_signal.shape}")
        print(f"✅ Add signal shape: {add_signal.shape}")
        print(f"✅ Through output shape: {output['through'].shape}")
        print(f"✅ Drop output shape: {output['drop'].shape}")
        
        print("\n🎉 ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ Test completado exitosamente")
    else:
        print("\n❌ Test falló")
        sys.exit(1)