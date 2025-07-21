"""
Helper utilities for TorchONN
============================

Utility functions for the photonic neural network framework.
"""

import torch
import time
import platform
from typing import Dict, Any, Tuple, Optional, Callable
from pathlib import Path


def get_package_info() -> Dict[str, Any]:
    """Get package information."""
    try:
        # Try to get version from __init__.py
        init_file = Path(__file__).parent.parent / "__init__.py"
        version = "unknown"
        author = "unknown"
        
        if init_file.exists():
            with open(init_file, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.strip().startswith('__version__'):
                        version = line.split('=')[1].strip().strip('"').strip("'")
                    elif line.strip().startswith('__author__'):
                        author = line.split('=')[1].strip().strip('"').strip("'")
        
        return {
            "name": "torchonn",
            "version": version,
            "author": author,
            "description": "Framework for Photonic Neural Networks"
        }
    except Exception:
        return {"name": "torchonn", "version": "unknown", "author": "unknown"}


def check_torch_version() -> Dict[str, Any]:
    """Check PyTorch version and compatibility."""
    try:
        import torch
        version = torch.__version__
        
        # Check if version is compatible
        major, minor = version.split('.')[:2]
        major, minor = int(major), int(minor)
        
        # Compatible with PyTorch 1.12+ and 2.x
        compatible = (major == 1 and minor >= 12) or (major >= 2)
        
        return {
            "torch_version": version,
            "version_compatible": compatible,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except Exception as e:
        return {
            "torch_version": "unknown",
            "version_compatible": False,
            "error": str(e)
        }


def validate_tensor_shape(
    tensor: torch.Tensor, 
    expected_shape: Tuple[int, ...],
    allow_batch: bool = True
) -> bool:
    """
    Validate tensor shape.
    
    Args:
        tensor: Input tensor
        expected_shape: Expected shape
        allow_batch: Whether to allow batch dimension
    
    Returns:
        True if shape is valid
    """
    actual_shape = tensor.shape
    
    if allow_batch and len(actual_shape) == len(expected_shape) + 1:
        # Check shape ignoring batch dimension
        return actual_shape[1:] == expected_shape
    else:
        return actual_shape == expected_shape


def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]):
    """Print model summary."""
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    try:
        dummy_input = torch.randn(1, *input_size)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output size: {output.shape}")
    except Exception as e:
        print(f"Could not compute output size: {e}")


def benchmark_function(
    func: Callable, 
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark a function.
    
    Args:
        func: Function to benchmark
        num_runs: Number of runs for timing
        warmup_runs: Number of warmup runs
    
    Returns:
        Timing statistics
    """
    # Warmup
    for _ in range(warmup_runs):
        func()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        func()
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = torch.tensor(times)
    
    return {
        "mean": times.mean().item(),
        "std": times.std().item(),
        "min": times.min().item(),
        "max": times.max().item(),
        "median": times.median().item()
    }


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "machine": platform.machine()
    }


__all__ = [
    'get_package_info',
    'check_torch_version',
    'validate_tensor_shape',
    'print_model_summary', 
    'benchmark_function',
    'get_system_info'
]
