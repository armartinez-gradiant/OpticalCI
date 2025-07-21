#!/usr/bin/env python3
"""
Script de exploración interactiva para PtONN-TESTS
"""

import torch
import numpy as np
from torchonn.layers import MZILayer, MZIBlockLinear
from torchonn.models import ONNBaseModel
from torchonn.devices import get_default_device
from torchonn.ops import apply_noise
from torchonn.utils import check_torch_version

def show_header():
    """Mostrar cabecera del programa."""
    print("🔬 PtONN-TESTS - Explorador Interactivo")
    print("=" * 50)
    
    # Info del sistema
    version_info = check_torch_version()
    device_config = get_default_device()
    
    print(f"📊 PyTorch: {version_info['torch_version']}")
    print(f"🖥️  Dispositivo: {device_config.device}")
    print(f"🐍 Python compatible: {version_info['version_compatible']}")
    print()

def demo_layers():
    """Demostrar uso de capas."""
    print("🧪 Demo: Capas MZI")
    print("-" * 30)
    
    # Crear datos de ejemplo
    batch_size, input_size = 5, 8
    x = torch.randn(batch_size, input_size)
    print(f"📥 Input: {x.shape}")
    
    # 1. MZI Layer básica
    print("\n1️⃣ MZI Layer:")
    mzi_layer = MZILayer(in_features=8, out_features=4)
    output_basic = mzi_layer(x)
    print(f"   Output: {output_basic.shape}")
    print(f"   Parámetros: {sum(p.numel() for p in mzi_layer.parameters())}")
    
    # 2. MZI Block en diferentes modos
    print("\n2️⃣ MZI Block Linear:")
    modes = ["usv", "weight", "phase"]
    
    for mode in modes:
        block = MZIBlockLinear(in_features=8, out_features=4, mode=mode)
        output = block(x)
        params = sum(p.numel() for p in block.parameters())
        print(f"   Modo {mode:6s}: {output.shape}, {params} parámetros")
    
    return x

def demo_model_creation():
    """Demostrar creación de modelos."""
    print("\n🏗️  Demo: Creación de Modelos")
    print("-" * 30)
    
    class SimpleONN(ONNBaseModel):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.layer1 = MZIBlockLinear(input_size, hidden_size, mode="usv", device=self.device)
            self.layer2 = MZILayer(hidden_size, output_size, device=self.device)
            self.activation = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.layer1(x)
            x = self.activation(x)
            x = self.layer2(x)
            return x
    
    # Crear modelo
    model = SimpleONN(input_size=10, hidden_size=8, output_size=3)
    
    print(f"📦 Modelo creado:")
    print(f"   Capas: {len(list(model.children()))}")
    print(f"   Parámetros totales: {sum(p.numel() for p in model.parameters())}")
    print(f"   Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Test forward
    x = torch.randn(4, 10)
    output = model(x)
    print(f"   Test forward: {x.shape} → {output.shape}")
    
    return model

def demo_training_loop():
    """Demostrar un loop de entrenamiento básico."""
    print("\n🎯 Demo: Loop de Entrenamiento")
    print("-" * 30)
    
    # Modelo simple
    model = MZIBlockLinear(in_features=4, out_features=2, mode="weight")
    
    # Datos sintéticos
    X = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    
    # Configuración
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("🚀 Entrenando por 10 épocas...")
    
    for epoch in range(10):
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 3 == 0:
            print(f"   Época {epoch}: Loss = {loss.item():.4f}")
    
    print("✅ Entrenamiento completado")

def demo_reset_parameters():
    """Demostrar reset de parámetros."""
    print("\n🔄 Demo: Reset de Parámetros")
    print("-" * 30)
    
    layer = MZILayer(in_features=6, out_features=3)
    
    # Parámetros originales
    original_weight = layer.weight.clone()
    print(f"📊 Peso original (muestra): {original_weight[0, :3].detach().numpy()}")
    
    # Reset
    layer.reset_parameters()
    new_weight = layer.weight
    print(f"📊 Peso después reset:     {new_weight[0, :3].detach().numpy()}")
    
    # Verificar que cambió
    changed = not torch.equal(original_weight, new_weight)
    print(f"✅ Parámetros cambiaron: {changed}")

def demo_noise_robustness():
    """Demostrar robustez al ruido."""
    print("\n🌪️  Demo: Robustez al Ruido")
    print("-" * 30)
    
    model = MZIBlockLinear(in_features=8, out_features=4, mode="usv")
    model.eval()
    
    # Datos limpios
    x_clean = torch.randn(3, 8)
    output_clean = model(x_clean)
    
    # Datos con ruido
    noise_levels = [0.1, 0.3, 0.5]
    
    print("🔍 Comparando salidas:")
    print(f"   Sin ruido: {output_clean[0, :2].detach().numpy()}")
    
    for noise_level in noise_levels:
        x_noisy = apply_noise(x_clean, noise_level=noise_level)
        output_noisy = model(x_noisy)
        
        # Calcular correlación
        correlation = torch.corrcoef(torch.stack([
            output_clean.flatten(),
            output_noisy.flatten()
        ]))[0, 1].item()
        
        print(f"   Ruido {noise_level:.1f}: {output_noisy[0, :2].detach().numpy()} (corr: {correlation:.3f})")

def demo_modes_comparison():
    """Comparar diferentes modos de MZI Block."""
    print("\n⚖️  Demo: Comparación de Modos")
    print("-" * 30)
    
    x = torch.randn(5, 6)
    modes = ["usv", "weight", "phase"]
    
    print(f"📥 Input común: {x.shape}")
    
    for mode in modes:
        layer = MZIBlockLinear(in_features=6, out_features=3, mode=mode)
        
        # Forward pass
        output = layer(x)
        
        # Estadísticas
        mean_output = output.mean().item()
        std_output = output.std().item()
        params = sum(p.numel() for p in layer.parameters())
        
        print(f"   {mode:6s}: μ={mean_output:6.3f}, σ={std_output:6.3f}, params={params}")

def interactive_menu():
    """Menú interactivo."""
    demos = {
        "1": ("Capas MZI", demo_layers),
        "2": ("Creación de Modelos", demo_model_creation),
        "3": ("Loop de Entrenamiento", demo_training_loop),
        "4": ("Reset de Parámetros", demo_reset_parameters),
        "5": ("Robustez al Ruido", demo_noise_robustness),
        "6": ("Comparación de Modos", demo_modes_comparison),
        "0": ("Ejecutar todas las demos", None),
    }
    
    print("\n🎮 Menú Interactivo:")
    print("-" * 20)
    for key, (name, _) in demos.items():
        print(f"   {key}. {name}")
    
    choice = input("\n👉 Elige una opción (Enter para todas): ").strip()
    
    if choice == "" or choice == "0":
        # Ejecutar todas
        for key, (name, func) in demos.items():
            if func is not None:
                func()
    elif choice in demos and demos[choice][1] is not None:
        demos[choice][1]()
    else:
        print("❌ Opción no válida")

def main():
    """Función principal."""
    show_header()
    
    try:
        interactive_menu()
        print("\n🎉 ¡Exploración completada!")
        print("\nPara ejecutar: python explore.py")
        
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()