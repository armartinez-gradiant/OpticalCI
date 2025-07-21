#!/usr/bin/env python3
"""Test simple para verificar que todo funciona"""
import torch
import torchonn
from torchonn.layers import MZILayer

print("🧪 Test Simple PtONN-TESTS")
print("=" * 40)

# Test PyTorch
x = torch.randn(2, 4)
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ Tensor: {x.shape}")

# Test TorchONN
layer = MZILayer(4, 3)
output = layer(x)
print(f"✅ TorchONN: {torchonn.__version__}")
print(f"✅ MZI Layer: {x.shape} -> {output.shape}")

# Test gradientes
x.requires_grad_(True)
output = layer(x)
loss = output.sum()
loss.backward()
print(f"✅ Gradientes: OK")

print("\n🎉 ¡Todo funciona correctamente!")
