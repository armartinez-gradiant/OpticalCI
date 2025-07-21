#!/usr/bin/env python3
"""
Test PyTorch Mejorado para PtONN-TESTS
=====================================

Test que evita el problema de importación de _C
"""

import sys
import os
from pathlib import Path

# Limpiar imports previos
modules_to_remove = []
for module_name in sys.modules.copy():
    if 'torch' in module_name or 'torchonn' in module_name:
        modules_to_remove.append(module_name)

for module_name in modules_to_remove:
    try:
        del sys.modules[module_name]
    except:
        pass

# Añadir directorio actual al path
sys.path.insert(0, str(Path.cwd()))

def test_pytorch_safe():
    """Test seguro de PyTorch"""
    print("🧪 Test PyTorch Seguro")
    print("=" * 40)
    
    try:
        # Import con manejo de errores específico
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # Test básico
        x = torch.randn(2, 3)
        print(f"✅ Tensor: {x.shape}")
        
        # Test operaciones
        y = x * 2
        print(f"✅ Operaciones: OK")
        
        # Test gradientes
        x.requires_grad_(True)
        z = (x ** 2).sum()
        z.backward()
        print(f"✅ Gradientes: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Error PyTorch: {e}")
        return False

def test_torchonn_safe():
    """Test seguro de TorchONN"""
    print("\n🧪 Test TorchONN Seguro")
    print("=" * 40)
    
    try:
        import torchonn
        print(f"✅ TorchONN: {torchonn.__version__}")
        
        from torchonn.layers import MZILayer, MZIBlockLinear
        print("✅ Layers importados")
        
        from torchonn.models import ONNBaseModel
        print("✅ Models importados")
        
        # Test funcionalidad
        layer = MZILayer(4, 3)
        x = torch.randn(2, 4)
        output = layer(x)
        print(f"✅ MZI Layer: {x.shape} -> {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error TorchONN: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    pytorch_ok = test_pytorch_safe()
    torchonn_ok = test_torchonn_safe()
    
    if pytorch_ok and torchonn_ok:
        print("\n🎉 ¡Todos los tests pasaron!")
        sys.exit(0)
    else:
        print("\n❌ Algunos tests fallaron")
        sys.exit(1)
