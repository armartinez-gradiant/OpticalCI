"""
Advanced usage examples for PtONN-TESTS
"""

import torch
import torch.nn as nn
import numpy as np
from torchonn.layers import MZILayer, MZIBlockLinear
from torchonn.models import ONNBaseModel
from torchonn.devices import DeviceConfig, get_default_device
from torchonn.ops import apply_noise, matrix_decomposition
from torchonn.utils import print_model_summary, benchmark_function

class AdvancedONN(ONNBaseModel):
    """Advanced ONN with multiple features."""
    
    def __init__(self, input_size=32, hidden_sizes=[24, 16, 12], output_size=10):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layer = MZIBlockLinear(
                in_features=prev_size,
                out_features=hidden_size,
                miniblock=4,
                mode="usv"
            )
            self.layers.append(layer)
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = MZILayer(prev_size, output_size)
        
        # Activations and regularization
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.ModuleList([
            nn.BatchNorm1d(size) for size in hidden_sizes
        ])
    
    def forward(self, x):
        # Hidden layers with batch norm and dropout
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.batch_norm[i](x)
            x = self.activation(x)
            if self.training:
                x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        return x

def device_management_example():
    """Example of device management."""
    print("🔧 Device Management Example")
    
    # Get default device config
    device_config = get_default_device()
    print(f"Default device: {device_config.device}")
    print(f"Precision: {device_config.precision}")
    
    # Create model on specific device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedONN().to(device)
    
    print(f"Model device: {next(model.parameters()).device}")
    
    return model

def noise_robustness_example():
    """Example of testing noise robustness."""
    print("\n🔬 Noise Robustness Example")
    
    model = AdvancedONN()
    model.eval()
    
    # Clean input
    x_clean = torch.randn(16, 32)
    output_clean = model(x_clean)
    
    # Add noise to input
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    
    print("Noise Level | Output Correlation")
    print("-" * 35)
    
    for noise_level in noise_levels:
        x_noisy = apply_noise(x_clean, noise_level=noise_level)
        output_noisy = model(x_noisy)
        
        # Calculate correlation
        correlation = torch.corrcoef(torch.stack([
            output_clean.flatten(),
            output_noisy.flatten()
        ]))[0, 1].item()
        
        print(f"{noise_level:10.2f} | {correlation:15.3f}")

def matrix_decomposition_example():
    """Example of matrix decomposition operations."""
    print("\n🔧 Matrix Decomposition Example")
    
    # Create a random matrix
    matrix = torch.randn(8, 8)
    print(f"Original matrix shape: {matrix.shape}")
    
    # SVD decomposition
    U, S, V = matrix_decomposition(matrix, method="svd")
    print(f"SVD shapes: U{U.shape}, S{S.shape}, V{V.shape}")
    
    # Reconstruct matrix
    reconstructed = torch.mm(torch.mm(U, torch.diag(S)), V.t())
    reconstruction_error = torch.norm(matrix - reconstructed)
    print(f"Reconstruction error: {reconstruction_error.item():.6f}")

def performance_benchmark_example():
    """Example of performance benchmarking."""
    print("\n⚡ Performance Benchmark Example")
    
    model = AdvancedONN()
    x = torch.randn(32, 32)
    
    def forward_pass():
        with torch.no_grad():
            return model(x)
    
    # Benchmark forward pass
    results = benchmark_function(forward_pass, num_runs=100)
    
    print(f"Forward pass timing (100 runs):")
    print(f"  Mean: {results['mean']*1000:.2f} ms")
    print(f"  Std:  {results['std']*1000:.2f} ms")
    print(f"  Min:  {results['min']*1000:.2f} ms")
    print(f"  Max:  {results['max']*1000:.2f} ms")

def gradient_analysis_example():
    """Example of gradient analysis."""
    print("\n📊 Gradient Analysis Example")
    
    model = AdvancedONN()
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass
    x = torch.randn(16, 32, requires_grad=True)
    y = torch.randint(0, 10, (16,))
    
    output = model(x)
    loss = criterion(output, y)
    
    # Backward pass
    loss.backward()
    
    # Analyze gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            print(f"{name:30} | Grad norm: {grad_norm:.6f}")
    
    print(f"\nAverage gradient norm: {np.mean(grad_norms):.6f}")
    print(f"Max gradient norm: {np.max(grad_norms):.6f}")

def model_summary_example():
    """Example of model summary."""
    print("\n📋 Model Summary Example")
    
    model = AdvancedONN(input_size=32, hidden_sizes=[24, 16, 12], output_size=10)
    print_model_summary(model, input_size=(32,))

if __name__ == "__main__":
    print("🚀 PtONN-TESTS Advanced Examples")
    print("=" * 50)
    
    # Run advanced examples
    model = device_management_example()
    noise_robustness_example()
    matrix_decomposition_example()
    performance_benchmark_example()
    gradient_analysis_example()
    model_summary_example()
    
    print("\n✅ All advanced examples completed successfully!")