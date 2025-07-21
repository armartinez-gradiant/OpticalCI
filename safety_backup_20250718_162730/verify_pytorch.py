#!/usr/bin/env python3
'''
Verification script for PyTorch installation
'''

import sys
import torch
import traceback

def main():
    print("🧪 Verificación PyTorch Post-Instalación")
    print("=" * 50)
    
    try:
        # Test 1: Basic import
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Test 2: Tensor operations
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        z = torch.mm(x, y)
        print(f"✅ Matrix multiplication: {x.shape} @ {y.shape} -> {z.shape}")
        
        # Test 3: Autograd
        x = torch.randn(2, 3, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        print(f"✅ Autograd: gradients computed")
        
        # Test 4: Device info
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA devices: {torch.cuda.device_count()}")
        
        # Test 5: Common operations
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        c = torch.dot(a, b)
        print(f"✅ Dot product: {c.item()}")
        
        print("\n🎉 ¡Todas las verificaciones pasaron!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en verificación: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
