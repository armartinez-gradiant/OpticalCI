"""
Optical Neural Networks (ONNs) Module for OpticalCI

Este módulo implementa redes neuronales ópticas usando los componentes
fotónicos fundamentales de OpticalCI como building blocks.

🎯 Objetivo: Simulaciones rigurosas de ONNs basadas en literatura científica
📚 Referencias: Shen et al. (2017), Tait et al. (2017), Hughes et al. (2018)

Arquitecturas Implementadas:
- CoherentONN: Redes coherentes usando mesh de MZIs (interferométricas)
- IncoherentONN: Redes incoherentes usando arrays de microrings (intensity-based)
- HybridONN: Redes híbridas combinando ambos enfoques (OpticalCI Original)

Características:
✅ Física realista usando componentes OpticalCI existentes
✅ Conservación de energía y propiedades unitarias
✅ Entrenamiento adaptado para limitaciones ópticas
✅ Benchmarks estándar (MNIST, clasificación)
✅ Optimización automática de arquitecturas híbridas
"""

__version__ = "1.1.0"
__author__ = "OpticalCI ONNs Module"

# Imports principales (solo cuando estén implementados)
try:
    from .architectures import CoherentONN, BaseONN
    from .benchmarks import OpticalMNIST
    __all__ = [
        "CoherentONN",
        "BaseONN", 
        "OpticalMNIST",
    ]
except ImportError:
    # Durante desarrollo incremental, algunos módulos pueden no existir aún
    __all__ = []

# NUEVO: Import IncoherentONN
try:
    from .architectures import IncoherentONN
    __all__.append("IncoherentONN")
except ImportError:
    pass

# NUEVO: Import HybridONN
try:
    from .architectures import HybridONN, HybridMode
    __all__.extend(["HybridONN", "HybridMode"])
except ImportError:
    pass

# NUEVO: Import utility functions
try:
    from .architectures import create_hybrid_onn_for_task
    __all__.append("create_hybrid_onn_for_task")
except ImportError:
    pass

# Configuración por defecto para ONNs (ACTUALIZADA)
DEFAULT_COHERENT_PRECISION = "float32"  # Para matrices unitarias
DEFAULT_WAVELENGTH_CHANNELS = 8         # Para WDM en ONNs
DEFAULT_OPTICAL_POWER = 1.0            # Potencia óptica normalizada
DEFAULT_TRANSITION_LOSS = 0.15         # NUEVO: 15% loss realista para HybridONN

def get_onn_info():
    """Información del módulo ONNs (ACTUALIZADA)."""
    return {
        "version": __version__,
        "architectures_available": __all__,
        "base_components": [
            "MZILayer", "MZIBlockLinear", "MicroringResonator", 
            "WDMMultiplexer", "Photodetector"
        ],
        "hybrid_features": [  # NUEVO
            "Coherent + Incoherent integration",
            "Physical transition modeling",
            "Automatic architecture optimization",
            "Multiple hybrid modes",
            "Task-specific configurations"
        ],
        "status": "Production Ready - Phase 2"
    }