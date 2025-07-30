"""
Optical Neural Networks (ONNs) Module for OpticalCI

Este módulo implementa redes neuronales ópticas usando los componentes
fotónicos fundamentales de OpticalCI como building blocks.

🎯 Objetivo: Simulaciones rigurosas de ONNs basadas en literatura científica
📚 Referencias: Shen et al. (2017), Tait et al. (2017), Hughes et al. (2018)

Arquitecturas Implementadas:
- CoherentONN: Redes coherentes usando mesh de MZIs (interferométricas)
- IncoherentONN: Redes incoherentes usando arrays de microrings (intensity-based)

Características:
✅ Física realista usando componentes OpticalCI existentes
✅ Conservación de energía y propiedades unitarias
✅ Entrenamiento adaptado para limitaciones ópticas
✅ Benchmarks estándar (MNIST, clasificación)
"""

__version__ = "1.0.0"
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

# Configuración por defecto para ONNs
DEFAULT_COHERENT_PRECISION = "float32"  # Para matrices unitarias
DEFAULT_WAVELENGTH_CHANNELS = 8         # Para WDM en ONNs
DEFAULT_OPTICAL_POWER = 1.0            # Potencia óptica normalizada

def get_onn_info():
    """Información del módulo ONNs."""
    return {
        "version": __version__,
        "architectures_available": __all__,
        "base_components": [
            "MZILayer", "MZIBlockLinear", "MicroringResonator", 
            "WDMMultiplexer", "Photodetector"
        ],
        "status": "Development - Fase 1"
    }