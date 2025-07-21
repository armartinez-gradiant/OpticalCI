#!/usr/bin/env python3
"""
Ejemplo básico de uso de PtONN-TESTS
"""

import torch
import numpy as np
from torchonn.layers import MZILayer, MZIBlockLinear
from torchonn.models import ONNBaseModel
from torchonn.devices import get_default_device

def test_basic_functionality():
    """Test básico de funcionalidad."""
    print("🚀 Probando PtONN-TESTS...")
    
    # 1. Configuración del dispositivo
    device_config = get_default_device()
    print(f"📱 Dispositivo: {device_config.device}")
    
    # 2. Crear capas individuales
    print("\n🔧 Creando capas MZI...")
    
    # Capa MZI básica
    mzi_layer = MZILayer(in_features=8, out_features=4)
    print(f"   MZI Layer: {mzi_layer.in_features} → {mzi_layer.out_features}")
    
    # Capa MZI Block en diferentes modos
    modes = ["usv", "weight", "phase"]
    for mode in modes:
        mzi_block = MZIBlockLinear(in_features=8, out_features=4, mode=mode)
        print(f"   MZI Block ({mode}): {mzi_block.in_features} → {mzi_block.out_features}")
    
    # 3. Test de forward pass
    print("\n⚡ Probando forward pass...")
    x = torch.randn(5, 8)  # Batch de 5 muestras, 8 características
    print(f"   Input shape: {x.shape}")
    
    # Test cada capa
    output_basic = mzi_layer(x)
    print(f"   MZI Layer output: {output_basic.shape}")
    
    for mode in modes:
        mzi_block = MZIBlockLinear(in_features=8, out_features=4, mode=mode)
        output_block = mzi_block(x)
        print(f"   MZI Block ({mode}) output: {output_block.shape}")
    
    # 4. Test de gradientes
    print("\n🎯 Probando gradientes...")
    x_grad = torch.randn(3, 8, requires_grad=True)
    output = mzi_layer(x_grad)
    loss = output.sum()
    loss.backward()
    
    print(f"   Gradiente del input: {x_grad.grad is not None}")
    print(f"   Gradiente de los parámetros: {mzi_layer.weight.grad is not None}")
    
    print("\n✅ ¡Test básico completado exitosamente!")

if __name__ == "__main__":
    test_basic_functionality()