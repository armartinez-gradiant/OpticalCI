
# ========================================
# MIGRACIÓN REQUERIDA:
# Los componentes fotónicos han sido movidos:
# 
# ANTES:
# from advanced_photonic_components import MicroringResonator, AddDropMRR
# 
# DESPUÉS:
# from torchonn.layers import MicroringResonator, AddDropMRR, DirectionalCoupler, Photodetector
# from torchonn.components import PhaseChangeCell, WDMMultiplexer, MRRWeightBank
# ========================================
#!/usr/bin/env python3
"""
🔧 PtONN-TESTS: Migración de Componentes Fotónicos Avanzados

Este script migra cuidadosamente los componentes fotónicos desde examples/ 
hacia la estructura principal de torchonn/, organizándolos en módulos apropiados.

Autor: Migración automatizada para PtONN-TESTS
Fecha: 2025-07-22
"""

import os
import shutil
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
from datetime import datetime

class PhotonicComponentsMigrator:
    """Migrador seguro de componentes fotónicos."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.backup_dir = self.repo_root / f"migration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configuración de la migración
        self.source_file = self.repo_root / "examples" / "advanced_photonic_components.py"
        self.components_config = self._setup_components_config()
        
        print(f"🎯 Migrador inicializado en: {self.repo_root}")
        print(f"📁 Backup se guardará en: {self.backup_dir}")
    
    def _setup_components_config(self) -> Dict:
        """Configurar qué componentes van a dónde."""
        return {
            # Tier 1: Componentes básicos → torchonn.layers
            "layers": {
                "microring.py": [
                    "MicroringResonator",
                    "AddDropMRR"
                ],
                "couplers.py": [
                    "DirectionalCoupler"
                ],
                "detectors.py": [
                    "Photodetector"
                ]
            },
            # Tier 2: Componentes especializados → torchonn.components (nuevo)
            "components": {
                "memory.py": [
                    "PhaseChangeCell"
                ],
                "wdm.py": [
                    "WDMMultiplexer",
                    "MRRWeightBank"
                ]
            }
        }
    
    def create_backup(self):
        """Crear backup completo antes de cualquier cambio."""
        print("📋 Creando backup de seguridad...")
        
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup de archivos específicos que cambiaremos
        backup_files = [
            "examples/advanced_photonic_components.py",
            "examples/advanced_photonic_components_fixed.py", 
            "examples/advanced_photonic_components_backup.py",
            "quick_test_adddrop.py",
            "quick_test_adddrop_fixed.py",
            "torchonn/layers/__init__.py",
            "torchonn/__init__.py"
        ]
        
        for file_path in backup_files:
            src = self.repo_root / file_path
            if src.exists():
                dst = self.backup_dir / file_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  ✅ Backup: {file_path}")
        
        # Backup de todo el directorio torchonn por seguridad
        torchonn_backup = self.backup_dir / "torchonn_full"
        if (self.repo_root / "torchonn").exists():
            shutil.copytree(self.repo_root / "torchonn", torchonn_backup)
            print(f"  ✅ Backup completo de torchonn/")
        
        print(f"✅ Backup completado en: {self.backup_dir}")
    
    def verify_source_files(self) -> bool:
        """Verificar que los archivos fuente existen."""
        print("🔍 Verificando archivos fuente...")
        
        # Buscar el mejor archivo fuente (preferir _fixed.py si existe)
        candidates = [
            self.repo_root / "examples" / "advanced_photonic_components_fixed.py",
            self.repo_root / "examples" / "advanced_photonic_components.py"
        ]
        
        for candidate in candidates:
            if candidate.exists():
                self.source_file = candidate
                print(f"  ✅ Usando archivo fuente: {candidate.name}")
                break
        else:
            print("  ❌ No se encontró archivo de componentes fotónicos válido")
            return False
        
        # Verificar que contiene las clases esperadas
        content = self.source_file.read_text(encoding='utf-8')
        
        all_components = []
        for tier_components in self.components_config.values():
            for file_components in tier_components.values():
                all_components.extend(file_components)
        
        missing_components = []
        for component in all_components:
            if f"class {component}" not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"  ❌ Componentes faltantes: {missing_components}")
            return False
        
        print(f"  ✅ Todos los componentes encontrados: {all_components}")
        return True
    
    def extract_component_code(self, source_content: str, component_name: str) -> Tuple[str, List[str]]:
        """Extraer el código de un componente específico y sus dependencias."""
        lines = source_content.split('\n')
        
        # Encontrar el inicio de la clase
        start_idx = None
        for i, line in enumerate(lines):
            if re.match(rf'^class {component_name}\s*\(', line.strip()):
                start_idx = i
                break
        
        if start_idx is None:
            raise ValueError(f"Componente {component_name} no encontrado")
        
        # Encontrar el final de la clase (siguiente class o final de archivo)
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if line.startswith('class ') and not line.startswith('    '):
                end_idx = i
                break
        
        # Extraer el código de la clase
        class_lines = lines[start_idx:end_idx]
        
        # Limpiar líneas vacías al final
        while class_lines and not class_lines[-1].strip():
            class_lines.pop()
        
        component_code = '\n'.join(class_lines)
        
        # Detectar dependencias (imports dentro del código)
        dependencies = []
        for line in class_lines:
            # Buscar imports de otros componentes del mismo archivo
            for other_component in ["MicroringResonator", "AddDropMRR", "DirectionalCoupler", 
                                  "Photodetector", "PhaseChangeCell", "WDMMultiplexer", "MRRWeightBank"]:
                if other_component != component_name and other_component in line:
                    dependencies.append(other_component)
        
        return component_code, list(set(dependencies))
    
    def create_module_file(self, tier: str, module_name: str, components: List[str], source_content: str):
        """Crear un archivo de módulo con los componentes especificados."""
        print(f"  📝 Creando {tier}/{module_name} con {components}")
        
        # Determinar directorio de destino
        if tier == "layers":
            dest_dir = self.repo_root / "torchonn" / "layers"
        else:  # components
            dest_dir = self.repo_root / "torchonn" / "components"
            dest_dir.mkdir(exist_ok=True)
        
        dest_file = dest_dir / module_name
        
        # Obtener imports base del archivo original
        source_lines = source_content.split('\n')
        import_lines = []
        
        for line in source_lines:
            line = line.strip()
            if (line.startswith('import ') or line.startswith('from ')) and not line.startswith('from advanced'):
                import_lines.append(line)
            elif line.startswith('"""') or line.startswith('class ') or line == '':
                break
        
        # Crear contenido del módulo
        module_content = []
        
        # Header
        module_content.append('"""')
        module_content.append(f'Photonic {tier.title()} for PtONN-TESTS')
        module_content.append('')
        if tier == "layers":
            module_content.append('Basic photonic components for neural network layers.')
        else:
            module_content.append('Specialized photonic components and systems.')
        module_content.append('"""')
        module_content.append('')
        
        # Imports
        module_content.extend(import_lines)
        module_content.append('')
        
        # Extraer y agregar cada componente
        all_dependencies = set()
        for component in components:
            component_code, deps = self.extract_component_code(source_content, component)
            module_content.append(component_code)
            module_content.append('')
            all_dependencies.update(deps)
        
        # Agregar funciones de test si están relacionadas
        if 'test_advanced_components' in source_content and tier == "layers":
            # Crear una versión simplificada del test
            test_lines = [
                'def test_basic_components():',
                '    """Test básico de componentes fotónicos."""',
                '    import torch',
                '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
                '    print("🧪 Testing basic photonic components...")',
                '    ',
                '    # Test MicroringResonator if available',
                '    if "MicroringResonator" in globals():',
                '        mrr = MicroringResonator(device=device)',
                '        wavelengths = torch.linspace(1530e-9, 1570e-9, 8, device=device)',
                '        input_signal = torch.randn(2, 8, device=device)',
                '        output = mrr(input_signal, wavelengths)',
                '        print("  ✅ MicroringResonator working")',
                '    ',
                '    print("✅ Basic components test completed")',
                '    return True'
            ]
            module_content.extend(test_lines)
            module_content.append('')
        
        # Escribir archivo
        with open(dest_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(module_content))
        
        print(f"    ✅ Creado: {dest_file}")
        return all_dependencies
    
    def create_directory_structure(self):
        """Crear la estructura de directorios necesaria."""
        print("📁 Creando estructura de directorios...")
        
        # Crear torchonn/components/ si no existe
        components_dir = self.repo_root / "torchonn" / "components"
        components_dir.mkdir(exist_ok=True)
        print(f"  ✅ {components_dir}")
        
        # Crear __init__.py para components si no existe
        components_init = components_dir / "__init__.py"
        if not components_init.exists():
            init_content = '''"""
Components module for PtONN-TESTS

Specialized photonic components and systems for advanced applications.
"""

from .memory import PhaseChangeCell
from .wdm import WDMMultiplexer, MRRWeightBank

__all__ = [
    "PhaseChangeCell",
    "WDMMultiplexer", 
    "MRRWeightBank",
]'''
            components_init.write_text(init_content, encoding='utf-8')
            print(f"  ✅ {components_init}")
    
    def migrate_components(self):
        """Migrar todos los componentes a sus nuevos módulos."""
        print("🔄 Migrando componentes fotónicos...")
        
        source_content = self.source_file.read_text(encoding='utf-8')
        all_dependencies = set()
        
        # Procesar cada tier
        for tier, modules in self.components_config.items():
            print(f"  📦 Tier: {tier}")
            
            for module_name, components in modules.items():
                deps = self.create_module_file(tier, module_name, components, source_content)
                all_dependencies.update(deps)
        
        print(f"  ✅ Componentes migrados exitosamente")
        return all_dependencies
    
    def update_init_files(self):
        """Actualizar archivos __init__.py con las nuevas importaciones."""
        print("📝 Actualizando archivos __init__.py...")
        
        # Actualizar torchonn/layers/__init__.py
        layers_init = self.repo_root / "torchonn" / "layers" / "__init__.py"
        
        new_layers_content = '''"""
Layers module for PtONN-TESTS

Modern implementation of photonic layers compatible with current PyTorch versions.
"""

from .mzi_layer import MZILayer
from .mzi_block_linear import MZIBlockLinear
from .microring import MicroringResonator, AddDropMRR
from .couplers import DirectionalCoupler
from .detectors import Photodetector

__all__ = [
    "MZILayer",
    "MZIBlockLinear",
    "MicroringResonator",
    "AddDropMRR", 
    "DirectionalCoupler",
    "Photodetector",
]'''
        
        layers_init.write_text(new_layers_content, encoding='utf-8')
        print(f"  ✅ Actualizado: {layers_init}")
        
        # Actualizar torchonn/__init__.py
        main_init = self.repo_root / "torchonn" / "__init__.py"
        current_content = main_init.read_text(encoding='utf-8')
        
        # Agregar import de components si no existe
        if "from . import components" not in current_content:
            # Buscar línea de "from . import ops" y agregar después
            lines = current_content.split('\n')
            new_lines = []
            
            for line in lines:
                new_lines.append(line)
                if line.strip() == "from . import ops":
                    new_lines.append("    from . import components")
            
            # Actualizar __all__ si existe
            for i, line in enumerate(new_lines):
                if '"ops",' in line:
                    new_lines.insert(i + 1, '    "components",')
                    break
            
            main_init.write_text('\n'.join(new_lines), encoding='utf-8')
            print(f"  ✅ Actualizado: {main_init}")
    
    def find_and_update_references(self):
        """Encontrar y actualizar todas las referencias a los componentes migrados."""
        print("🔍 Buscando y actualizando referencias...")
        
        # Patrones de import a buscar y reemplazar
        replacement_patterns = [
            # Imports desde examples
            (
                r'sys\.path\.append\([\'"]examples[\'"]\)',
                '# # sys.path.append("examples")  # No longer needed  # No longer needed'
            ),
            (
                r'# Updated imports:
# from torchonn.layers import (.+)',  # or from torchonn.components import (.+)',
                r'# Updated imports:\n# from torchonn.layers import \1  # or from torchonn.components import \1'
            ),
            (
                r'import sys\s+from pathlib import Path\s+sys\.path\.append\([\'"]examples[\'"]\)',
                '# Removed examples path addition'
            )
        ]
        
        # Archivos a revisar
        files_to_check = [
            "quick_test_adddrop.py",
            "quick_test_adddrop_fixed.py",
            "test_installation.py"
        ]
        
        # Buscar archivos Python adicionales que puedan tener referencias
        for py_file in self.repo_root.rglob("*.py"):
            if (py_file.is_file() and 
                "advanced_photonic_components" in py_file.read_text(encoding='utf-8', errors='ignore')):
                files_to_check.append(str(py_file.relative_to(self.repo_root)))
        
        files_to_check = list(set(files_to_check))  # Eliminar duplicados
        
        for file_path in files_to_check:
            full_path = self.repo_root / file_path
            if not full_path.exists():
                continue
                
            try:
                content = full_path.read_text(encoding='utf-8')
                original_content = content
                
                # Aplicar reemplazos
                for pattern, replacement in replacement_patterns:
                    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                
                # Agregar comentarios de ayuda para migración manual
                if "advanced_photonic_components" in content:
                    migration_comment = '''
# ========================================
# MIGRACIÓN REQUERIDA:
# Los componentes fotónicos han sido movidos:
# 
# ANTES:
# # Updated imports:
# from torchonn.layers import MicroringResonator, AddDropMRR  # or from torchonn.components import MicroringResonator, AddDropMRR
# 
# DESPUÉS:
# from torchonn.layers import MicroringResonator, AddDropMRR, DirectionalCoupler, Photodetector
# from torchonn.components import PhaseChangeCell, WDMMultiplexer, MRRWeightBank
# ========================================
'''
                    content = migration_comment + content
                
                # Escribir si hubo cambios
                if content != original_content:
                    full_path.write_text(content, encoding='utf-8')
                    print(f"  ✅ Actualizado: {file_path}")
                
            except Exception as e:
                print(f"  ⚠️  Error procesando {file_path}: {e}")
    
    def create_migration_test(self):
        """Crear script de test para verificar la migración."""
        test_content = '''#!/usr/bin/env python3
"""
🧪 Test de Verificación de Migración - PtONN-TESTS

Verifica que todos los componentes fotónicos migrados funcionan correctamente.
"""

import sys
import torch

def test_layers_imports():
    """Test de importaciones básicas de layers."""
    print("🧪 Testing torchonn.layers imports...")
    
    try:
        from torchonn.layers import (
            MZILayer, MZIBlockLinear,
            MicroringResonator, AddDropMRR,
            DirectionalCoupler, Photodetector
        )
        print("  ✅ Todas las importaciones de layers exitosas")
        return True
    except Exception as e:
        print(f"  ❌ Error en importaciones de layers: {e}")
        return False

def test_components_imports():
    """Test de importaciones de components."""
    print("🧪 Testing torchonn.components imports...")
    
    try:
        from torchonn.components import (
            PhaseChangeCell, WDMMultiplexer, MRRWeightBank
        )
        print("  ✅ Todas las importaciones de components exitosas")
        return True
    except Exception as e:
        print(f"  ❌ Error en importaciones de components: {e}")
        return False

def test_basic_functionality():
    """Test básico de funcionalidad."""
    print("🧪 Testing basic functionality...")
    
    try:
        from torchonn.layers import MicroringResonator, DirectionalCoupler
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test MicroringResonator
        mrr = MicroringResonator(device=device)
        wavelengths = torch.linspace(1530e-9, 1570e-9, 8, device=device)
        input_signal = torch.randn(2, 8, device=device)
        output = mrr(input_signal, wavelengths)
        
        assert 'through' in output
        assert 'drop' in output
        print("  ✅ MicroringResonator funcionando")
        
        # Test DirectionalCoupler  
        coupler = DirectionalCoupler(device=device)
        input1 = torch.randn(2, 8, device=device)
        input2 = torch.randn(2, 8, device=device)
        out1, out2 = coupler(input1, input2)
        
        assert out1.shape == input1.shape
        assert out2.shape == input2.shape
        print("  ✅ DirectionalCoupler funcionando")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error en test de funcionalidad: {e}")
        return False

def main():
    """Test principal."""
    print("🎯 Test de Verificación de Migración - PtONN-TESTS")
    print("=" * 60)
    
    success = True
    success &= test_layers_imports()
    success &= test_components_imports() 
    success &= test_basic_functionality()
    
    if success:
        print("\\n🎉 ¡Migración verificada exitosamente!")
        print("\\n📋 Nuevas importaciones disponibles:")
        print("   from torchonn.layers import MicroringResonator, AddDropMRR")
        print("   from torchonn.layers import DirectionalCoupler, Photodetector")
        print("   from torchonn.components import PhaseChangeCell, WDMMultiplexer")
        return 0
    else:
        print("\\n❌ Problemas detectados en la migración")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        test_file = self.repo_root / "test_migration.py"
        test_file.write_text(test_content, encoding='utf-8')
        print(f"✅ Creado script de verificación: {test_file}")
    
    def create_updated_examples(self):
        """Crear ejemplos actualizados que usen la nueva estructura."""
        
        example_content = '''#!/usr/bin/env python3
"""
🌟 Ejemplo Actualizado: Componentes Fotónicos - PtONN-TESTS

Demostración de uso de los componentes fotónicos migrados a la estructura principal.
"""

import torch
from torchonn.layers import (
    MZILayer, MZIBlockLinear,
    MicroringResonator, AddDropMRR, 
    DirectionalCoupler, Photodetector
)
from torchonn.components import (
    PhaseChangeCell, WDMMultiplexer, MRRWeightBank
)

def demo_basic_components():
    """Demostración de componentes básicos."""
    print("🔧 Demo: Componentes Básicos")
    print("-" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    n_wavelengths = 8
    
    # Wavelengths de prueba
    wavelengths = torch.linspace(1530e-9, 1570e-9, n_wavelengths, device=device)
    
    print("1️⃣ MicroringResonator:")
    mrr = MicroringResonator(device=device)
    input_signal = torch.randn(batch_size, n_wavelengths, device=device)
    mrr_output = mrr(input_signal, wavelengths)
    print(f"   Through: {mrr_output['through'].shape}")
    print(f"   Drop: {mrr_output['drop'].shape}")
    
    print("\\n2️⃣ Add-Drop MRR:")
    add_drop = AddDropMRR(device=device)
    add_signal = torch.randn(batch_size, n_wavelengths, device=device)
    add_drop_output = add_drop(input_signal, add_signal, wavelengths)
    print(f"   Through: {add_drop_output['through'].shape}")
    print(f"   Drop: {add_drop_output['drop'].shape}")
    
    print("\\n3️⃣ DirectionalCoupler:")
    coupler = DirectionalCoupler(device=device)
    input_1 = torch.randn(batch_size, n_wavelengths, device=device)
    input_2 = torch.randn(batch_size, n_wavelengths, device=device)
    out_1, out_2 = coupler(input_1, input_2)
    print(f"   Output 1: {out_1.shape}")
    print(f"   Output 2: {out_2.shape}")

def demo_specialized_components():
    """Demostración de componentes especializados."""
    print("\\n🔬 Demo: Componentes Especializados") 
    print("-" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    
    print("1️⃣ PhaseChangeCell:")
    pcm = PhaseChangeCell(device=device)
    pcm_input = torch.randn(batch_size, 8, device=device)
    pcm_output = pcm(pcm_input)
    print(f"   PCM state: {pcm.pcm_state.item():.3f}")
    print(f"   Output: {pcm_output.shape}")
    
    print("\\n2️⃣ WDMMultiplexer:")
    wdm_wavelengths = [1530e-9, 1540e-9, 1550e-9, 1560e-9]
    wdm = WDMMultiplexer(wdm_wavelengths, device=device)
    channels = [torch.randn(batch_size, device=device) for _ in range(4)]
    muxed = wdm.multiplex(channels)
    print(f"   Multiplexed: {muxed.shape}")

def main():
    """Función principal."""
    print("🌟 Componentes Fotónicos - PtONN-TESTS")
    print("=" * 60)
    print("✅ Usando nueva estructura de importaciones!")
    print()
    
    try:
        demo_basic_components()
        demo_specialized_components()
        
        print("\\n🎉 ¡Todos los componentes funcionando correctamente!")
        print("\\n📚 Para más información:")
        print("   - Documentación: torchonn.layers y torchonn.components")
        print("   - Tests: python test_migration.py")
        
    except Exception as e:
        print(f"\\n❌ Error durante demo: {e}")
        raise

if __name__ == "__main__":
    main()
'''
        
        example_file = self.repo_root / "examples" / "photonic_components_demo.py"
        example_file.write_text(example_content, encoding='utf-8')
        print(f"✅ Creado ejemplo actualizado: {example_file}")
    
    def run_migration(self):
        """Ejecutar el proceso completo de migración."""
        print("🚀 INICIANDO MIGRACIÓN DE COMPONENTES FOTÓNICOS")
        print("=" * 80)
        
        try:
            # 1. Crear backup
            self.create_backup()
            
            # 2. Verificar archivos fuente
            if not self.verify_source_files():
                print("❌ Error: Archivos fuente no válidos. Abortando.")
                return False
            
            # 3. Crear estructura de directorios
            self.create_directory_structure()
            
            # 4. Migrar componentes
            dependencies = self.migrate_components()
            
            # 5. Actualizar __init__.py
            self.update_init_files()
            
            # 6. Actualizar referencias
            self.find_and_update_references()
            
            # 7. Crear scripts de verificación
            self.create_migration_test()
            
            # 8. Crear ejemplos actualizados
            self.create_updated_examples()
            
            print("\\n✅ MIGRACIÓN COMPLETADA EXITOSAMENTE!")
            print("=" * 80)
            print("\\n📋 Resumen de cambios:")
            print(f"   🔹 Backup creado en: {self.backup_dir}")
            print(f"   🔹 Componentes migrados: {len([c for tier in self.components_config.values() for comps in tier.values() for c in comps])}")
            print(f"   🔹 Nuevos módulos creados: {sum(len(tier) for tier in self.components_config.values())}")
            print(f"   🔹 Referencias actualizadas automáticamente")
            
            print("\\n🎯 Próximos pasos:")
            print("   1. python test_migration.py           # Verificar migración")
            print("   2. python examples/photonic_components_demo.py  # Probar ejemplos")
            print("   3. pytest tests/                      # Ejecutar tests completos")
            print("   4. Actualizar manualmente archivos con comentarios de migración")
            
            print("\\n📚 Nueva estructura de importaciones:")
            print("   from torchonn.layers import MicroringResonator, AddDropMRR")
            print("   from torchonn.layers import DirectionalCoupler, Photodetector") 
            print("   from torchonn.components import PhaseChangeCell, WDMMultiplexer")
            
            return True
            
        except Exception as e:
            print(f"\\n❌ ERROR DURANTE MIGRACIÓN: {e}")
            print(f"📁 Backup disponible en: {self.backup_dir}")
            print("🔄 Para revertir: restaurar archivos desde backup")
            return False

def main():
    """Función principal del script."""
    print("🔧 PtONN-TESTS: Migrador de Componentes Fotónicos")
    print("⚡ Migrando desde examples/ hacia torchonn/")
    print()
    
    # Detectar directorio del repositorio
    repo_root = Path.cwd()
    
    # Verificar que estamos en el directorio correcto
    if not (repo_root / "torchonn").exists():
        print("❌ Error: No se encontró directorio torchonn/")
        print("   Por favor ejecutar desde el directorio raíz del repositorio")
        sys.exit(1)
    
    # Crear y ejecutar migrador
    migrator = PhotonicComponentsMigrator(repo_root)
    
    success = migrator.run_migration()
    
    if success:
        print("\\n🎉 ¡Migración completada exitosamente!")
        sys.exit(0)
    else:
        print("\\n❌ Migración falló")
        sys.exit(1)

if __name__ == "__main__":
    main()