"""
ONN Architectures Module

Implementaciones de diferentes arquitecturas de redes neuronales ópticas
basadas en literatura científica y usando componentes OpticalCI.

Arquitecturas Disponibles:
- BaseONN: Clase base para todas las ONNs
- CoherentONN: Red coherente usando mesh de MZIs (Shen et al. 2017)
- IncoherentONN: Red incoherente usando microring arrays (Tait et al. 2017)
- HybridONN: Red híbrida combinando coherent + incoherent (OpticalCI Original)

Todas las arquitecturas:
✅ Conservan propiedades físicas (energía, unitaridad)
✅ Usan componentes OpticalCI existentes
✅ Incluyen validación automática
✅ Compatibles con PyTorch training
"""

# Import de la clase base
from .base_onn import BaseONN, validate_onn_implementation

# Import de arquitecturas específicas (cuando estén implementadas)
try:
    from .coherent_onn import CoherentONN
    COHERENT_AVAILABLE = True
except ImportError:
    COHERENT_AVAILABLE = False

try:
    from .incoherent_onn import IncoherentONN
    INCOHERENT_AVAILABLE = True
except ImportError:
    INCOHERENT_AVAILABLE = False

# NUEVO: Import HybridONN
try:
    from .hybrid_onn import HybridONN, HybridMode
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

# Lista de arquitecturas disponibles
__all__ = ["BaseONN", "validate_onn_implementation"]

if COHERENT_AVAILABLE:
    __all__.append("CoherentONN")

if INCOHERENT_AVAILABLE:
    __all__.append("IncoherentONN")

# NUEVO: Agregar HybridONN
if HYBRID_AVAILABLE:
    __all__.extend(["HybridONN", "HybridMode"])

# Función helper para listar arquitecturas (ACTUALIZADA)
def list_available_architectures():
    """Lista todas las arquitecturas ONN disponibles."""
    architectures = {
        "BaseONN": {
            "available": True,
            "description": "Clase base para todas las ONNs",
            "type": "base"
        }
    }
    
    if COHERENT_AVAILABLE:
        architectures["CoherentONN"] = {
            "available": True,
            "description": "Red coherente usando mesh de MZIs",
            "type": "coherent",
            "reference": "Shen et al. (2017)"
        }
    else:
        architectures["CoherentONN"] = {
            "available": False,
            "description": "Red coherente usando mesh de MZIs (en desarrollo)",
            "type": "coherent"
        }
    
    if INCOHERENT_AVAILABLE:
        architectures["IncoherentONN"] = {
            "available": True,
            "description": "Red incoherente usando microring arrays + WDM",
            "type": "incoherent",
            "reference": "Tait et al. (2017)"
        }
    else:
        architectures["IncoherentONN"] = {
            "available": False,
            "description": "Red incoherente usando microring arrays (en desarrollo)",
            "type": "incoherent"
        }
    
    # NUEVO: Agregar HybridONN
    if HYBRID_AVAILABLE:
        architectures["HybridONN"] = {
            "available": True,
            "description": "Red híbrida coherent + incoherent con transiciones físicas",
            "type": "hybrid",
            "reference": "OpticalCI Original (2025)",
            "modes": ["alternating", "front_coherent", "adaptive", "custom"],
            "key_features": [
                "Combina precisión coherente + escalabilidad incoherent",
                "Transiciones C↔I físicamente realistas",
                "Optimización automática de arquitectura",
                "5+ modos de configuración flexible"
            ]
        }
    else:
        architectures["HybridONN"] = {
            "available": False,
            "description": "Red híbrida coherent + incoherent (en desarrollo)",
            "type": "hybrid"
        }
    
    return architectures

# Función helper para crear arquitecturas (ACTUALIZADA)
def create_onn(architecture: str, **kwargs):
    """
    Factory function para crear arquitecturas ONN.
    
    Args:
        architecture: Nombre de la arquitectura ("CoherentONN", "IncoherentONN", "HybridONN")
        **kwargs: Argumentos para el constructor
        
    Returns:
        Instancia de la arquitectura solicitada
    """
    if architecture == "CoherentONN" and COHERENT_AVAILABLE:
        return CoherentONN(**kwargs)
    elif architecture == "IncoherentONN" and INCOHERENT_AVAILABLE:
        return IncoherentONN(**kwargs)
    elif architecture == "HybridONN" and HYBRID_AVAILABLE:  # NUEVO
        return HybridONN(**kwargs)
    elif architecture == "BaseONN":
        return BaseONN(**kwargs)
    else:
        available = list_available_architectures()
        available_names = [name for name, info in available.items() if info["available"]]
        raise ValueError(
            f"Architecture '{architecture}' not available. "
            f"Available: {available_names}"
        )

# NUEVO: Función helper específica para HybridONN
def create_hybrid_onn_for_task(task: str, **kwargs):
    """
    Factory function para crear HybridONN optimizado para tareas específicas.
    
    Args:
        task: "image_processing", "signal_processing", "large_scale", "balanced"
        **kwargs: Argumentos adicionales
    
    Returns:
        HybridONN configurado para la tarea específica
    """
    if not HYBRID_AVAILABLE:
        raise ImportError("HybridONN not available")
    
    from .hybrid_onn import (
        create_image_processing_hybrid,
        create_signal_processing_hybrid, 
        create_large_scale_hybrid
    )
    
    if task == "image_processing":
        input_size = kwargs.pop('input_size', 3072)  # CIFAR-10 default
        n_classes = kwargs.pop('n_classes', 10)
        return create_image_processing_hybrid(input_size, n_classes, **kwargs)
        
    elif task == "signal_processing":
        input_size = kwargs.pop('input_size', 128)
        output_size = kwargs.pop('output_size', 16)
        return create_signal_processing_hybrid(input_size, output_size, **kwargs)
        
    elif task == "large_scale":
        layer_sizes = kwargs.pop('layer_sizes', [1024, 512, 256, 64])
        return create_large_scale_hybrid(layer_sizes, **kwargs)
        
    elif task == "balanced":
        layer_sizes = kwargs.get('layer_sizes', [8, 12, 8, 4])
        kwargs['hybrid_mode'] = HybridMode.ADAPTIVE
        return HybridONN(**kwargs)
        
    else:
        available_tasks = ["image_processing", "signal_processing", "large_scale", "balanced"]
        raise ValueError(f"Task '{task}' not recognized. Available: {available_tasks}")