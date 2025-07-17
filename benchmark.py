#!/usr/bin/env python3
"""
Benchmark de rendimiento para PtONN-TESTS
"""

import torch
import torch.nn as nn
import time
import numpy as np
from torchonn.layers import MZILayer, MZIBlockLinear
from torchonn.models import ONNBaseModel

class StandardNN(nn.Module):
    """Red neuronal estándar para comparación."""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ONNMZI(ONNBaseModel):
    """Red ONN con capas MZI."""
    
    def __init__(self, input_size, hidden_sizes, output_size, mode="usv"):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layer = MZIBlockLinear(
                in_features=prev_size,
                out_features=hidden_size,
                mode=mode,
                device=self.device
            )
            self.layers.append(layer)
            prev_size = hidden_size
        
        self.output_layer = MZILayer(
            in_features=prev_size,
            out_features=output_size,
            device=self.device
        )
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        
        x = self.output_layer(x)
        return x

def benchmark_forward_pass():
    """Benchmark de forward pass."""
    print("⚡ Benchmark: Forward Pass")
    print("-" * 40)
    
    # Configuraciones de prueba
    configs = [
        {"input": 32, "hidden": [16, 8], "output": 4, "batch": 64},
        {"input": 64, "hidden": [32, 16], "output": 8, "batch": 128},
        {"input": 128, "hidden": [64, 32, 16], "output": 10, "batch": 256},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n🧪 Configuración {i+1}: {config['input']}→{config['hidden']}→{config['output']}")
        
        # Crear modelos
        standard_model = StandardNN(config['input'], config['hidden'], config['output'])
        onn_model = ONNMZI(config['input'], config['hidden'], config['output'])
        
        # Datos de prueba
        x = torch.randn(config['batch'], config['input'])
        
        # Benchmark standard
        standard_model.eval()
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = standard_model(x)
        standard_time = time.time() - start_time
        
        # Benchmark ONN
        onn_model.eval()
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = onn_model(x)
        onn_time = time.time() - start_time
        
        # Resultados
        print(f"   Standard NN: {standard_time:.4f}s (100 forward passes)")
        print(f"   ONN MZI:     {onn_time:.4f}s (100 forward passes)")
        print(f"   Ratio:       {onn_time/standard_time:.2f}x")

def benchmark_memory_usage():
    """Benchmark de uso de memoria."""
    print("\n💾 Benchmark: Uso de Memoria")
    print("-" * 40)
    
    def get_model_size(model):
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024**2  # MB
    
    # Configuración de prueba
    input_size, hidden_sizes, output_size = 64, [32, 16], 8
    
    # Modelos
    standard_model = StandardNN(input_size, hidden_sizes, output_size)
    onn_usv = ONNMZI(input_size, hidden_sizes, output_size, mode="usv")
    onn_weight = ONNMZI(input_size, hidden_sizes, output_size, mode="weight")
    onn_phase = ONNMZI(input_size, hidden_sizes, output_size, mode="phase")
    
    # Calcular tamaños
    print(f"   Standard NN:      {get_model_size(standard_model):.2f} MB")
    print(f"   ONN MZI (USV):    {get_model_size(onn_usv):.2f} MB")
    print(f"   ONN MZI (Weight): {get_model_size(onn_weight):.2f} MB")
    print(f"   ONN MZI (Phase):  {get_model_size(onn_phase):.2f} MB")
    
    # Contar parámetros
    print(f"\n🔢 Número de Parámetros:")
    print(f"   Standard NN:      {sum(p.numel() for p in standard_model.parameters()):,}")
    print(f"   ONN MZI (USV):    {sum(p.numel() for p in onn_usv.parameters()):,}")
    print(f"   ONN MZI (Weight): {sum(p.numel() for p in onn_weight.parameters()):,}")
    print(f"   ONN MZI (Phase):  {sum(p.numel() for p in onn_phase.parameters()):,}")

def benchmark_training_speed():
    """Benchmark de velocidad de entrenamiento."""
    print("\n🏃 Benchmark: Velocidad de Entrenamiento")
    print("-" * 40)
    
    # Configuración
    input_size, hidden_sizes, output_size = 32, [16, 8], 4
    batch_size = 64
    
    # Datos sintéticos
    X = torch.randn(batch_size, input_size)
    y = torch.randint(0, output_size, (batch_size,))
    
    # Modelos
    models = {
        "Standard NN": StandardNN(input_size, hidden_sizes, output_size),
        "ONN MZI (USV)": ONNMZI(input_size, hidden_sizes, output_size, mode="usv"),
        "ONN MZI (Weight)": ONNMZI(input_size, hidden_sizes, output_size, mode="weight"),
    }
    
    criterion = nn.CrossEntropyLoss()
    
    for name, model in models.items():
        print(f"\n🔧 {name}:")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Benchmark training
        start_time = time.time()
        
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        training_time = time.time() - start_time
        print(f"   50 épocas: {training_time:.4f}s")
        print(f"   Por época: {training_time/50:.4f}s")

def benchmark_numerical_stability():
    """Benchmark de estabilidad numérica."""
    print("\n🎯 Benchmark: Estabilidad Numérica")
    print("-" * 40)
    
    # Casos extremos
    test_cases = [
        ("Valores pequeños", torch.randn(10, 20) * 0.001),
        ("Valores grandes", torch.randn(10, 20) * 100),
        ("Valores cero", torch.zeros(10, 20)),
        ("Valores negativos", -torch.abs(torch.randn(10, 20))),
    ]
    
    model = ONNMZI(20, [16, 8], 4)
    model.eval()
    
    for case_name, x in test_cases:
        try:
            with torch.no_grad():
                output = model(x)
                
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                max_val = output.abs().max().item()
                
                status = "✅" if not (has_nan or has_inf) and max_val < 1000 else "❌"
                print(f"   {case_name:15s}: {status} (max: {max_val:.2e})")
                
        except Exception as e:
            print(f"   {case_name:15s}: ❌ Error: {str(e)}")

def main():
    """Función principal de benchmark."""
    print("🏁 Benchmark PtONN-TESTS")
    print("=" * 60)
    
    # Configurar semilla
    torch.manual_seed(42)
    
    try:
        benchmark_forward_pass()
        benchmark_memory_usage()
        benchmark_training_speed()
        benchmark_numerical_stability()
        
        print("\n🎉 ¡Benchmark completado!")
        print("\nPara ejecutar: python benchmark.py")
        
    except Exception as e:
        print(f"\n❌ Error durante benchmark: {e}")
        raise

if __name__ == "__main__":
    main()