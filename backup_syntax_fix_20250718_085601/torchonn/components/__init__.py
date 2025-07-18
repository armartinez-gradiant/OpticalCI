"""
Componentes Fotónicos - TorchONN
===============================
"""

from .base_component import BasePhotonicComponent, WaveguideComponent, ResonatorComponent

# Imports seguros (solo si el archivo existe y es válido)
__all__ = ['BasePhotonicComponent', 'WaveguideComponent', 'ResonatorComponent']

def _safe_import(module_name, class_name):
    """Import seguro que no falla si hay errores"""
    try:
        module = __import__(f'torchonn.components.{module_name}', fromlist=[class_name])
        return getattr(module, class_name)
    except Exception:
        return None

# Intentar imports seguros
MicroringResonator = _safe_import('microring_resonator', 'MicroringResonator')
if MicroringResonator:
    __all__.append('MicroringResonator')

AddDropMRR = _safe_import('add_drop_mrr', 'AddDropMRR')  
if AddDropMRR:
    __all__.append('AddDropMRR')

DirectionalCoupler = _safe_import('directional_coupler', 'DirectionalCoupler')
if DirectionalCoupler:
    __all__.append('DirectionalCoupler')

Photodetector = _safe_import('photodetector', 'Photodetector')
if Photodetector:
    __all__.append('Photodetector')

PhaseChangeCell = _safe_import('phase_change_cell', 'PhaseChangeCell')
if PhaseChangeCell:
    __all__.append('PhaseChangeCell')

Waveguide = _safe_import('waveguide', 'Waveguide')
if Waveguide:
    __all__.append('Waveguide')

LaserSource = _safe_import('laser_source', 'LaserSource')
if LaserSource:
    __all__.append('LaserSource')
