#!/usr/bin/env python3
"""Test básico de instalación para PtONN-TESTS."""

import sys

def test_imports():
    """Test de importaciones básicas."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import torchonn
        print(f"✅ TorchONN importado")
        
        from torchonn.layers import MZILayer
        print("✅ MZILayer importado")
        
        return True
    except Exception as e:
        print(f"❌ Error en imports: {e}")
        return False

def test_basic_functionality():
    """Test de funcionalidad básica."""
    try:
        import torch
        from torchonn.layers import MZILayer
        
        # Test básico
        layer = MZILayer(in_features=4, out_features=3)
        x = torch.randn(2, 4)
        output = layer(x)
        
        assert output.shape == (2, 3), f"Shape incorrecto: {output.shape}"
        print("✅ Test básico de funcionalidad")
        return True
    except Exception as e:
        print(f"❌ Error en funcionalidad: {e}")
        return False

def main():
    """Función principal."""
    print("🧪 Test de Instalación PtONN-TESTS")
    print("-" * 40)
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    if success:
        print("\n🎉 ¡Instalación verificada exitosamente!")
        return 0
    else:
        print("\n❌ Problemas detectados en la instalación")
        return 1

if __name__ == "__main__":
    sys.exit(main())
