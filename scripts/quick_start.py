#!/usr/bin/env python3
"""
Quick Start - OpticalCI
=======================

Script de inicio rápido para probar OpticalCI de inmediato.
"""

import sys
from pathlib import Path

# Añadir path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def quick_demo():
    """Demo rápido de 30 segundos."""
    print("🌟 OpticalCI - Quick Start Demo")
    print("=" * 40)
    
    try:
        from torchonn.layers import MZILayer
        from torchonn.utils.phase_shifter_extractor import quick_extract
        import torch
        
        # Crear red simple
        model = MZILayer(3, 3)
        print(f"✅ Red 3x3 creada con {model.get_phase_shifter_count()} phase shifters")
        
        # Test forward pass
        x = torch.randn(1, 3)
        y = model(x)
        print(f"✅ Forward pass: {x.squeeze().numpy()} -> {y.squeeze().detach().numpy()}")
        
        # Extraer valores
        values = quick_extract(model)
        print(f"✅ Valores extraídos: {values['summary']['total_phase_shifters']} phase shifters")
        
        print("\n🎉 ¡OpticalCI funcionando correctamente!")
        print("🚀 Próximos pasos:")
        print("   - Ejecuta: python scripts/interactive/manual_3x3.py")
        print("   - O usa: python scripts/experiments/network_designer.py --interactive")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Instala OpticalCI: pip install -e .")
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    quick_demo()