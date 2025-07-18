#!/usr/bin/env python3
"""
Corrección Directa de Sintaxis PtONN-TESTS
==========================================

Corrección super directa de los errores persistentes de **variable** → __variable__
que siguen apareciendo en el diagnóstico.

Uso: python direct_syntax_fix.py
"""

from pathlib import Path

def fix_file_directly(file_path: Path, expected_bad_content: str, correct_content: str):
    """Corregir archivo de forma directa"""
    if not file_path.exists():
        print(f"❌ Archivo no encontrado: {file_path}")
        return False
    
    try:
        # Leer contenido actual
        with open(file_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        
        # Verificar si tiene el contenido problemático
        if expected_bad_content in current_content:
            print(f"🔧 Corrigiendo: {file_path}")
            
            # Escribir contenido correcto
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(correct_content)
            
            print(f"✅ Corregido: {file_path}")
            return True
        else:
            print(f"ℹ️ Ya correcto: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error en {file_path}: {e}")
        return False

def main():
    """Función principal"""
    print("🔧 Corrección Directa de Sintaxis - PtONN-TESTS")
    print("=" * 55)
    
    fixes_applied = 0
    
    # Corrección 1: torchonn/__init__.py
    print("\n1️⃣ Corrigiendo torchonn/__init__.py")
    
    init_correct = '''"""
TorchONN - Framework para Redes Neuronales Ópticas
===================================================

Framework modular y profesional para el diseño, simulación y entrenamiento
de redes neuronales ópticas (ONNs) basado en PyTorch.
"""

__version__ = "2.0.0"
__author__ = "PtONN-TESTS Team"

# Core imports
from . import layers
from . import models
from . import devices
from . import ops
from . import utils

# Try to import components
try:
    from . import components
except ImportError:
    pass

# Key classes
from .layers import MZILayer, MZIBlockLinear
from .models import ONNBaseModel

__all__ = [
    'layers', 'models', 'devices', 'ops', 'utils',
    'MZILayer', 'MZIBlockLinear', 'ONNBaseModel'
]
'''
    
    if fix_file_directly(Path("torchonn/__init__.py"), "**version**", init_correct):
        fixes_applied += 1
    
    # Corrección 2: torchonn/layers/__init__.py  
    print("\n2️⃣ Corrigiendo torchonn/layers/__init__.py")
    
    layers_correct = '''"""
Capas Fotónicas - TorchONN
=========================

Módulo de capas neuronales fotónicas.
"""

from .mzi_layer import MZILayer
from .mzi_block_linear import MZIBlockLinear

__all__ = ['MZILayer', 'MZIBlockLinear']

# Import optional layers
try:
    from .mrr_weight_bank import MRRWeightBank
    __all__.append('MRRWeightBank')
except ImportError:
    pass

try:
    from .photonic_linear import PhotonicLinear
    __all__.append('PhotonicLinear')
except ImportError:
    pass

try:
    from .photonic_conv2d import PhotonicConv2D
    __all__.append('PhotonicConv2D')
except ImportError:
    pass
'''
    
    if fix_file_directly(Path("torchonn/layers/__init__.py"), "**all**", layers_correct):
        fixes_applied += 1
    
    # Verificación rápida
    print(f"\n🎯 Resultado: {fixes_applied} archivos corregidos")
    
    if fixes_applied > 0:
        print("\n✅ Correcciones aplicadas")
        print("📋 Ejecuta ahora: python quick_test.py")
        
        # Test rápido de import
        print("\n🧪 Test rápido de import:")
        try:
            import sys
            sys.path.insert(0, str(Path.cwd()))
            
            import torchonn
            version = getattr(torchonn, '__version__', 'unknown')
            print(f"✅ TorchONN v{version} importado correctamente")
            
            from torchonn.layers import MZILayer
            print("✅ MZILayer importado correctamente")
            
        except Exception as e:
            print(f"❌ Error en test: {e}")
    else:
        print("\nℹ️ No se necesitaron correcciones")
    
    return fixes_applied > 0

if __name__ == "__main__":
    main()