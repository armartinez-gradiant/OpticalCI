"""
ONN Architectures Module

Implementaciones de diferentes arquitecturas de redes neuronales ópticas
basadas en literatura científica y usando componentes OpticalCI.

Arquitecturas Disponibles:
- BaseONN: Clase base para todas las ONNs
- CoherentONN: Red coherente usando mesh de MZIs (Shen et al. 2017)
- IncoherentONN: Red incoherente usando microring arrays (Tait et al. 2017)

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

# Lista de arquitecturas disponibles
__all__ = ["BaseONN", "validate_onn_implementation"]

if COHERENT_AVAILABLE:
    __all__.append("CoherentONN")

# Función helper para listar arquitecturas
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
    
    return architectures

# Función helper para crear arquitecturas
def create_onn(architecture: str, **kwargs):
    """
    Factory function para crear arquitecturas ONN.
    
    Args:
        architecture: Nombre de la arquitectura ("CoherentONN", etc.)
        **kwargs: Argumentos para el constructor
        
    Returns:
        Instancia de la arquitectura solicitada
    """
    if architecture == "CoherentONN" and COHERENT_AVAILABLE:
        return CoherentONN(**kwargs)
    elif architecture == "BaseONN":
        return BaseONN(**kwargs)
    else:
        available = list_available_architectures()
        available_names = [name for name, info in available.items() if info["available"]]
        raise ValueError(
            f"Architecture '{architecture}' not available. "
            f"Available: {available_names}"
        )