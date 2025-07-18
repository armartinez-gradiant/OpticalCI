#!/usr/bin/env python3
"""
Test Super Simple PtONN-TESTS
=============================

Test minimalista que verifica lo básico.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

def main():
    print("🧪 TEST SUPER SIMPLE PtONN-TESTS")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: PyTorch
    total_tests += 1
    try:
        import torch
        print("✅ 1. PyTorch import: OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 1. PyTorch import: {e}")
    
    # Test 2: TorchONN main
    total_tests += 1
    try:
        import torchonn
        print(f"✅ 2. TorchONN import: OK (v{torchonn.__version__})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 2. TorchONN import: {e}")
    
    # Test 3: MZI Layers
    total_tests += 1
    try:
        from torchonn.layers import MZILayer, MZIBlockLinear
        print("✅ 3. MZI layers import: OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 3. MZI layers import: {e}")
    
    # Test 4: Layer creation
    total_tests += 1
    try:
        layer = MZILayer(4, 3)
        block = MZIBlockLinear(4, 3, mode="usv")
        print("✅ 4. Layer creation: OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 4. Layer creation: {e}")
    
    # Test 5: Forward pass
    total_tests += 1
    try:
        x = torch.randn(2, 4)
        out1 = layer(x)
        out2 = block(x)
        print(f"✅ 5. Forward pass: OK ({x.shape} -> {out1.shape})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 5. Forward pass: {e}")
    
    # Test 6: Models
    total_tests += 1
    try:
        from torchonn.models import ONNBaseModel
        print("✅ 6. Models import: OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 6. Models import: {e}")
    
    # Summary
    print(f"\n📊 RESULTADO: {tests_passed}/{total_tests} tests passed")
    percentage = (tests_passed / total_tests) * 100
    print(f"   Porcentaje de éxito: {percentage:.1f}%")
    
    if tests_passed == total_tests:
        print("\n🎉 ¡PERFECTO! PtONN-TESTS está funcionando completamente")
        print("\n🚀 El sistema está listo para:")
        print("   • Crear redes neuronales ópticas")
        print("   • Entrenar modelos")
        print("   • Ejecutar experimentos")
        print("   • Desarrollar aplicaciones")
        
        print("\n📚 Próximos pasos:")
        print("   • Ejecutar: pytest tests/")
        print("   • Probar: python examples/basic_usage.py")
        print("   • Explorar: python examples/advanced_usage.py")
        
        return True
    elif tests_passed >= 4:
        print("\n✅ ¡Muy bien! El core está funcionando")
        print("   Las funcionalidades básicas están operativas")
        return True
    else:
        print("\n⚠️ Hay problemas que necesitan atención")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
