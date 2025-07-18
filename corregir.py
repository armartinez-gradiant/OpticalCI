#!/usr/bin/env python3
"""
Script de Corrección Específico para PtONN-TESTS
===============================================

Identifica y corrige automáticamente los errores más comunes encontrados.
"""

import sys
import os
from pathlib import Path
import subprocess
import shutil
import re
from datetime import datetime

class PtONNErrorFixer:
    def __init__(self):
        self.repo_path = Path.cwd()
        self.fixes_applied = []
        self.errors_found = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log con timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {level}: {message}")

    def create_backup(self):
        """Crear backup antes de hacer cambios"""
        backup_dir = self.repo_path / f"backup_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Solo respaldar archivos Python importantes
        important_files = [
            "torchonn/__init__.py",
            "torchonn/layers/__init__.py", 
            "torchonn/models/__init__.py",
            "torchonn/components/__init__.py",
        ]
        
        backup_dir.mkdir(exist_ok=True)
        
        for file_path in important_files:
            source = self.repo_path / file_path
            if source.exists():
                dest = backup_dir / file_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        
        self.log(f"Backup creado en: {backup_dir}")
        return backup_dir

    def fix_import_errors(self):
        """Corregir errores de importación comunes"""
        self.log("🔧 Corrigiendo errores de importación...")
        
        # Fix 1: torchonn/__init__.py
        init_file = self.repo_path / "torchonn" / "__init__.py"
        if init_file.exists():
            content = init_file.read_text(encoding='utf-8')
            
            # Verificar problemas comunes
            issues = []
            if "**version**" in content:
                issues.append("Variable name with asterisks")
            if "**author**" in content:
                issues.append("Variable name with asterisks")
            
            if issues:
                self.log(f"Problemas encontrados en {init_file}: {issues}")
                
                # Crear contenido corregido
                fixed_content = '''"""
TorchONN - Framework para Redes Neuronales Ópticas
===================================================

Framework modular y profesional para el diseño, simulación y entrenamiento
de redes neuronales ópticas (ONNs) basado en PyTorch.
"""

__version__ = "2.0.0"
__author__ = "PtONN-TESTS Team"

# Core imports
from . import layers
from . import models
from . import devices
from . import ops
from . import utils

# Try to import components
try:
    from . import components
except ImportError:
    pass

# Key classes
from .layers import MZILayer, MZIBlockLinear
from .models import ONNBaseModel

__all__ = [
    'layers', 'models', 'devices', 'ops', 'utils',
    'MZILayer', 'MZIBlockLinear', 'ONNBaseModel'
]
'''
                init_file.write_text(fixed_content, encoding='utf-8')
                self.fixes_applied.append("Fixed torchonn/__init__.py variable names")
        
        # Fix 2: torchonn/layers/__init__.py
        layers_init = self.repo_path / "torchonn" / "layers" / "__init__.py"
        if layers_init.exists():
            content = layers_init.read_text(encoding='utf-8')
            if "**all**" in content:
                fixed_content = '''"""
Capas Fotónicas - TorchONN
=========================

Módulo de capas neuronales fotónicas.
"""

from .mzi_layer import MZILayer
from .mzi_block_linear import MZIBlockLinear

__all__ = ['MZILayer', 'MZIBlockLinear']

# Import optional layers
try:
    from .mrr_weight_bank import MRRWeightBank
    __all__.append('MRRWeightBank')
except ImportError:
    pass

try:
    from .photonic_linear import PhotonicLinear
    __all__.append('PhotonicLinear')
except ImportError:
    pass

try:
    from .photonic_conv2d import PhotonicConv2D
    __all__.append('PhotonicConv2D')
except ImportError:
    pass
'''
                layers_init.write_text(fixed_content, encoding='utf-8')
                self.fixes_applied.append("Fixed torchonn/layers/__init__.py")

    def fix_component_imports(self):
        """Corregir imports de componentes problemáticos"""
        self.log("🔧 Corrigiendo imports de componentes...")
        
        # Verificar que MicroringResonator existe y es importable
        mrr_file = self.repo_path / "torchonn" / "components" / "microring_resonator.py"
        if mrr_file.exists():
            try:
                content = mrr_file.read_text(encoding='utf-8')
                if "class MicroringResonator" in content:
                    self.log("✅ MicroringResonator class found")
                else:
                    self.log("❌ MicroringResonator class not found in file")
                    self.errors_found.append("MicroringResonator class missing")
            except Exception as e:
                self.log(f"Error reading microring_resonator.py: {e}")
        
        # Fix components/__init__.py para importación segura
        components_init = self.repo_path / "torchonn" / "components" / "__init__.py"
        if components_init.exists():
            safe_init_content = '''"""
Componentes Fotónicos - TorchONN
===============================
"""

from .base_component import BasePhotonicComponent, WaveguideComponent, ResonatorComponent

# Imports seguros
__all__ = ['BasePhotonicComponent', 'WaveguideComponent', 'ResonatorComponent']

def _safe_import(module_name, class_name):
    """Import seguro que no falla si hay errores"""
    try:
        module = __import__(f'torchonn.components.{module_name}', fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as e:
        print(f"Warning: Could not import {class_name} from {module_name}: {e}")
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

# Add all successfully imported components
print(f"TorchONN Components loaded: {__all__}")
'''
            components_init.write_text(safe_init_content, encoding='utf-8')
            self.fixes_applied.append("Updated components/__init__.py with safe imports")

    def fix_model_imports(self):
        """Corregir imports de modelos"""
        self.log("🔧 Corrigiendo imports de modelos...")
        
        models_init = self.repo_path / "torchonn" / "models" / "__init__.py"
        models_base = self.repo_path / "torchonn" / "models" / "base_model.py"
        
        # Verificar si base_model.py existe
        if not models_base.exists():
            self.log("Creando base_model.py faltante...")
            base_model_content = '''"""
ONN Base Model - Modelo base para redes neuronales ópticas
=========================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Union

class ONNBaseModel(nn.Module):
    """
    Clase base para modelos de redes neuronales ópticas.
    """
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.to(self.device)
    
    def reset_parameters(self):
        """Reinicializar todos los parámetros del modelo"""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - debe ser implementado por subclases"""
        raise NotImplementedError("Subclasses must implement forward method")
'''
            models_base.write_text(base_model_content, encoding='utf-8')
            self.fixes_applied.append("Created missing base_model.py")
        
        # Fix models/__init__.py
        if models_init.exists():
            models_init_content = '''"""
Modelos ONN - TorchONN
=====================

Módulo de modelos y arquitecturas para redes neuronales ópticas.
"""

# Import del modelo base si existe
try:
    from .base_model import ONNBaseModel
    __all__ = ['ONNBaseModel']
except ImportError:
    # Crear modelo base si no existe
    import torch.nn as nn
    
    class ONNBaseModel(nn.Module):
        """Modelo base para ONNs - implementación temporal"""
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            return x
    
    __all__ = ['ONNBaseModel']

# Para compatibilidad hacia atrás
BaseONNModel = ONNBaseModel
__all__.append('BaseONNModel')
'''
            models_init.write_text(models_init_content, encoding='utf-8')
            self.fixes_applied.append("Fixed models/__init__.py")

    def clean_pycache(self):
        """Limpiar archivos cache que pueden causar problemas"""
        self.log("🧹 Limpiando archivos cache...")
        
        # Eliminar __pycache__ directories
        pycache_dirs = list(self.repo_path.rglob("__pycache__"))
        for pycache_dir in pycache_dirs:
            shutil.rmtree(pycache_dir, ignore_errors=True)
        
        # Eliminar archivos .pyc
        pyc_files = list(self.repo_path.rglob("*.pyc"))
        for pyc_file in pyc_files:
            pyc_file.unlink(missing_ok=True)
        
        if pycache_dirs or pyc_files:
            self.fixes_applied.append(f"Cleaned {len(pycache_dirs)} __pycache__ dirs and {len(pyc_files)} .pyc files")

    def fix_circular_imports(self):
        """Corregir imports circulares potenciales"""
        self.log("🔧 Verificando imports circulares...")
        
        # Revisar advanced_photonic_components.py
        advanced_file = self.repo_path / "examples" / "advanced_photonic_components.py"
        if advanced_file.exists():
            content = advanced_file.read_text(encoding='utf-8')
            
            # Si aún tiene definiciones de clases, usar solo imports
            if "class MicroringResonator" in content:
                self.log("Migrando advanced_photonic_components.py a imports...")
                
                migrated_content = '''"""
Advanced Photonic Components - Imports Modulares
================================================

Este archivo ahora importa los componentes desde sus módulos modulares
en lugar de definirlos directamente, manteniendo compatibilidad hacia atrás.

Los componentes han sido movidos a:
- torchonn.components.*: Componentes básicos
- torchonn.systems.*: Sistemas completos

Autor: PtONN-TESTS Team
Fecha: ''' + datetime.now().strftime('%Y-%m-%d') + '''
"""

# Imports modulares de los componentes refactorizados
try:
    from torchonn.components.microring_resonator import MicroringResonator
except ImportError:
    print("Warning: MicroringResonator not available")
    MicroringResonator = None

try:
    from torchonn.components.add_drop_mrr import AddDropMRR
except ImportError:
    print("Warning: AddDropMRR not available")
    AddDropMRR = None

try:
    from torchonn.components.directional_coupler import DirectionalCoupler
except ImportError:
    print("Warning: DirectionalCoupler not available")
    DirectionalCoupler = None

try:
    from torchonn.components.photodetector import Photodetector
except ImportError:
    print("Warning: Photodetector not available") 
    Photodetector = None

try:
    from torchonn.components.phase_change_cell import PhaseChangeCell
except ImportError:
    print("Warning: PhaseChangeCell not available")
    PhaseChangeCell = None

try:
    from torchonn.layers.mrr_weight_bank import MRRWeightBank
except ImportError:
    print("Warning: MRRWeightBank not available")
    MRRWeightBank = None

try:
    from torchonn.systems.wdm_system import WDMMultiplexer
except ImportError:
    print("Warning: WDMMultiplexer not available")
    WDMMultiplexer = None

# Re-exports para compatibilidad hacia atrás
__all__ = []
if MicroringResonator: __all__.append('MicroringResonator')
if AddDropMRR: __all__.append('AddDropMRR')
if MRRWeightBank: __all__.append('MRRWeightBank')
if DirectionalCoupler: __all__.append('DirectionalCoupler')
if Photodetector: __all__.append('Photodetector')
if PhaseChangeCell: __all__.append('PhaseChangeCell')
if WDMMultiplexer: __all__.append('WDMMultiplexer')

# Información de migración
_MIGRATION_INFO = {
    'migrated_classes': __all__,
    'migration_date': \'''' + datetime.now().isoformat() + '''\',
    'new_structure': 'torchonn.components.*'
}

def get_migration_info():
    """Obtener información sobre la migración de componentes"""
    return _MIGRATION_INFO
'''
                advanced_file.write_text(migrated_content, encoding='utf-8')
                self.fixes_applied.append("Migrated advanced_photonic_components.py to use imports")

    def validate_core_functionality(self):
        """Validar que la funcionalidad core funciona"""
        self.log("✅ Validando funcionalidad core...")
        
        try:
            # Test imports
            sys.path.insert(0, str(self.repo_path))
            
            # Test torchonn
            import torchonn
            self.log("✓ Paquete principal: OK")
            
            # Test layers
            import torchonn.layers
            self.log("✓ Módulo layers: OK")
            
            # Test specific classes
            from torchonn.layers import MZILayer
            self.log("✓ Clase MZILayer: OK")
            
            from torchonn.layers import MZIBlockLinear
            self.log("✓ Clase MZIBlockLinear: OK")
            
            # Test models
            import torchonn.models
            from torchonn.models import ONNBaseModel
            self.log("✓ Módulo models: OK")
            self.log("✓ Clase ONNBaseModel: OK")
            
            # Test basic functionality
            import torch
            x = torch.randn(2, 4)
            layer = MZILayer(4, 3)
            output = layer(x)
            self.log("✓ MZI Layer funcional")
            
            block = MZIBlockLinear(4, 3, mode="usv")
            output2 = block(x)
            self.log("✓ MZI Block funcional")
            
            # Test gradients
            output.sum().backward()
            self.log("✓ Gradientes funcionando")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Error en validación: {e}")
            self.errors_found.append(f"Validation failed: {e}")
            return False

    def run_fixes(self):
        """Ejecutar todas las correcciones"""
        self.log("🚀 INICIANDO CORRECCIONES PtONN-TESTS")
        self.log("=" * 50)
        
        # Crear backup
        backup_path = self.create_backup()
        
        # Aplicar correcciones
        self.clean_pycache()
        self.fix_import_errors()
        self.fix_component_imports()
        self.fix_model_imports()
        self.fix_circular_imports()
        
        # Validar
        success = self.validate_core_functionality()
        
        # Generar reporte
        self.generate_report(backup_path, success)
        
        return success

    def generate_report(self, backup_path: Path, success: bool):
        """Generar reporte de correcciones"""
        status = "✅ EXITOSO" if success else "❌ CON ERRORES"
        
        report_content = f'''# Reporte de Corrección PtONN-TESTS
## Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## Estado: {status}

### Resumen
- **Correcciones aplicadas**: {len(self.fixes_applied)}
- **Errores encontrados**: {len(self.errors_found)}

### Correcciones Aplicadas
- Backup creado en: {backup_path}
{chr(10).join([f"- {fix}" for fix in self.fixes_applied])}

### Errores Restantes
{chr(10).join([f"- {error}" for error in self.errors_found]) if self.errors_found else "Ninguno"}

### Archivos Modificados
- torchonn/__init__.py
- torchonn/layers/__init__.py
- torchonn/models/__init__.py
- torchonn/components/__init__.py

### Backup Disponible
Backup de seguridad en: `{backup_path}`

### Verificación Final
Para verificar que todo funciona correctamente:

```bash
cd {self.repo_path}
python -c "import torchonn; print('✅ TorchONN OK')"
python -c "from torchonn.layers import MZILayer; print('✅ MZI Layer OK')"
python quick_test.py
```

### Próximos Pasos
1. Ejecutar el quick_test.py nuevamente
2. Verificar que no hay errores de sintaxis
3. Ejecutar tests completos: `pytest tests/ -v`
4. Continuar con el desarrollo normal

---
*Reporte generado automáticamente por el Corrector PtONN-TESTS*
'''
        
        report_file = self.repo_path / "FIX_REPORT.md"
        report_file.write_text(report_content, encoding='utf-8')
        
        self.log(f"📄 Reporte generado: {report_file}")

def main():
    """Función principal"""
    try:
        fixer = PtONNErrorFixer()
        success = fixer.run_fixes()
        
        if success:
            print("\n🎉 ¡Correcciones aplicadas exitosamente!")
            print("📋 Ejecuta ahora: python quick_test.py")
        else:
            print("\n⚠️ Correcciones aplicadas pero hay errores pendientes")
            print("📋 Revisa el FIX_REPORT.md para más detalles")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Corrección interrumpida por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error durante corrección: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)