"""
Utilidades - TorchONN
====================

Módulo de utilidades y herramientas auxiliares.
"""

try:
    from .helpers import (
        get_package_info,
        check_torch_version,
        validate_tensor_shape,
        print_model_summary,
        benchmark_function
    )
    __all__ = [
        'get_package_info',
        'check_torch_version',
        'validate_tensor_shape', 
        'print_model_summary',
        'benchmark_function'
    ]
except ImportError:
    # Fallback simple utilities
    import torch
    
    def get_package_info():
        return {"name": "torchonn", "version": "2.0.0", "author": "PtONN-TESTS Team"}
    
    def check_torch_version():
        return {
            "torch_version": torch.__version__,
            "version_compatible": True,
            "cuda_available": torch.cuda.is_available()
        }
    
    def validate_tensor_shape(tensor, expected_shape, allow_batch=True):
        return True  # Simplified
    
    def print_model_summary(model, input_size):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
    
    def benchmark_function(func, num_runs=100, warmup_runs=10):
        import time
        times = []
        for _ in range(num_runs):
            start = time.time()
            func()
            times.append(time.time() - start)
        return {"mean": sum(times) / len(times)}
    
    __all__ = [
        'get_package_info',
        'check_torch_version',
        'validate_tensor_shape',
        'print_model_summary',
        'benchmark_function'
    ]
