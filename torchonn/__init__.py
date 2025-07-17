"""
TorchONN - Framework para Redes Neuronales Ópticas
===================================================

Framework modular y profesional para el diseño, simulación y entrenamiento
de redes neuronales ópticas (ONNs) basado en PyTorch.
"""

__version__ = "2.0.0"
__author__ = "PtONN-TESTS Team"

# Imports seguros de módulos
def _safe_import_module(module_name):
    """Import seguro de módulos"""
    try:
        return __import__(f'torchonn.{module_name}', fromlist=[''])
    except Exception as e:
        print(f"Warning: No se pudo importar {module_name}: {e}")
        return None

# Imports de módulos principales
components = _safe_import_module('components')
layers = _safe_import_module('layers')
models = _safe_import_module('models')
devices = _safe_import_module('devices')
ops = _safe_import_module('ops')
utils = _safe_import_module('utils')

# Imports seguros de clases específicas
def _safe_import_class(module_path, class_name):
    """Import seguro de clases"""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception:
        return None

# Componentes principales
BasePhotonicComponent = _safe_import_class('torchonn.components', 'BasePhotonicComponent')
MicroringResonator = _safe_import_class('torchonn.components', 'MicroringResonator')
AddDropMRR = _safe_import_class('torchonn.components', 'AddDropMRR')

# Capas principales
MZILayer = _safe_import_class('torchonn.layers', 'MZILayer')
MZIBlockLinear = _safe_import_class('torchonn.layers', 'MZIBlockLinear')

# Modelos principales
BaseONNModel = _safe_import_class('torchonn.models', 'BaseONNModel')
ONNBaseModel = _safe_import_class('torchonn.models', 'ONNBaseModel')

# Lista de exports (solo los que se importaron exitosamente)
__all__ = ['components', 'layers', 'models', 'devices', 'ops', 'utils']

if BasePhotonicComponent:
    __all__.append('BasePhotonicComponent')
if MicroringResonator:
    __all__.append('MicroringResonator')
if AddDropMRR:
    __all__.append('AddDropMRR')
if MZILayer:
    __all__.append('MZILayer')
if MZIBlockLinear:
    __all__.append('MZIBlockLinear')
if BaseONNModel:
    __all__.append('BaseONNModel')
if ONNBaseModel:
    __all__.append('ONNBaseModel')
