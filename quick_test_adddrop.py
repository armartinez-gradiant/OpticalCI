#!/usr/bin/env python3
"""
🧪 Test Rápido - Verificar AddDropMRR Fixes
"""
import torch
import sys
from pathlib import Path
sys.path.append('examples')

def quick_test():
    print("🧪 QUICK TEST - AddDropMRR Fixes")
    print("=" * 40)
    
    try:
        from advanced_photonic_components import AddDropMRR
        
        # Create with fixed parameters
        add_drop = AddDropMRR(
            radius=10e-6,
            coupling_strength_1=0.05,  # Even lower for better extinction
            coupling_strength_2=0.05,
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
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    quick_test()
