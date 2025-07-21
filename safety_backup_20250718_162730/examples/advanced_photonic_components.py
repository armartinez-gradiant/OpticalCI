"""
Advanced Photonic Components - Imports Modulares
================================================

Este archivo ahora importa los componentes desde sus módulos modulares
en lugar de definirlos directamente, manteniendo compatibilidad hacia atrás.

Los componentes han sido movidos a:
- torchonn.components.*: Componentes básicos
- torchonn.systems.*: Sistemas completos

Autor: PtONN-TESTS Team
Fecha: 2025-07-17
"""

# Imports modulares de los componentes refactorizados
from torchonn.components.microring_resonator import MicroringResonator
from torchonn.components.add_drop_mrr import AddDropMRR
from torchonn.components.directional_coupler import DirectionalCoupler
from torchonn.components.photodetector import Photodetector
from torchonn.components.phase_change_cell import PhaseChangeCell

# Re-exports para compatibilidad hacia atrás
__all__ = [
    'MicroringResonator', 'AddDropMRR', 'MRRWeightBank', 'DirectionalCoupler', 'Photodetector', 'PhaseChangeCell', 'WDMMultiplexer'
]

# Información de migración
_MIGRATION_INFO = {
    'migrated_classes': ['MicroringResonator', 'AddDropMRR', 'MRRWeightBank', 'DirectionalCoupler', 'Photodetector', 'PhaseChangeCell', 'WDMMultiplexer'],
    'migration_date': '2025-07-17T14:43:29.182116',
    'new_structure': 'torchonn.components.*'
}

def get_migration_info():
    """Obtener información sobre la migración de componentes"""
    return _MIGRATION_INFO
