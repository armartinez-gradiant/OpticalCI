"""
Benchmarks module for ONNs

Contiene benchmarks y demos para Optical Neural Networks.
"""

try:
    from .mnist_optical import (
        run_quick_demo, 
        OpticalMNIST, 
        OpticalMNISTBenchmark,  # Alias
        create_optical_mnist_benchmark
    )
    
    __all__ = [
        "run_quick_demo",
        "OpticalMNIST",
        "OpticalMNISTBenchmark",  # Para compatibilidad con tests
        "create_optical_mnist_benchmark",
    ]
    
except ImportError as e:
    # Si falta algún archivo, al menos el __init__.py existe
    import warnings
    warnings.warn(f"Could not import all benchmark modules: {e}")
    __all__ = []

def get_available_benchmarks():
    """Obtener lista de benchmarks disponibles."""
    return [name for name in __all__ if "Benchmark" in name or "MNIST" in name]

def run_benchmark(name: str, **kwargs):
    """Ejecutar benchmark por nombre."""
    if name in ["OpticalMNIST", "OpticalMNISTBenchmark"]:
        return OpticalMNIST(**kwargs).run_comparison_benchmark()
    else:
        raise ValueError(f"Unknown benchmark: {name}")

# Para compatibilidad con testonn.py
def OpticalMNIST_factory(**kwargs):
    """Factory function para crear OpticalMNIST."""
    return OpticalMNIST(**kwargs)
