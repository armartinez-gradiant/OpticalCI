#!/usr/bin/env python3
"""
🔍 PtONN-TESTS: Validador Post-Migración

Script completo de validación que verifica que la migración de componentes
fotónicos se completó correctamente y todo funciona como esperado.
"""

import sys
import os
import torch
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

class MigrationValidator:
    """Validador completo de la migración."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.errors = []
        self.warnings = []
        
        # Configuración esperada post-migración
        self.expected_structure = {
            "torchonn/layers/microring.py": ["MicroringResonator", "AddDropMRR"],
            "torchonn/layers/couplers.py": ["DirectionalCoupler"],
            "torchonn/layers/detectors.py": ["Photodetector"],
            "torchonn/components/memory.py": ["PhaseChangeCell"],
            "torchonn/components/wdm.py": ["WDMMultiplexer", "MRRWeightBank"]
        }
        
        print(f"🔍 Validador inicializado en: {self.repo_root}")
    
    def check_file_structure(self) -> bool:
        """Verificar que la estructura de archivos es correcta."""
        print("📁 Validando estructura de archivos...")
        
        success = True
        
        # Verificar que existen los archivos esperados
        for file_path, components in self.expected_structure.items():
            full_path = self.repo_root / file_path
            
            if not full_path.exists():
                self.errors.append(f"Archivo faltante: {file_path}")
                success = False
                continue
            
            # Verificar que el archivo contiene las clases esperadas
            try:
                content = full_path.read_text(encoding='utf-8')
                for component in components:
                    if f"class {component}" not in content:
                        self.errors.append(f"Clase {component} no encontrada en {file_path}")
                        success = False
                    else:
                        print(f"  ✅ {component} encontrado en {file_path}")
            except Exception as e:
                self.errors.append(f"Error leyendo {file_path}: {e}")
                success = False
        
        # Verificar que existe torchonn/components/__init__.py
        components_init = self.repo_root / "torchonn/components/__init__.py"
        if not components_init.exists():
            self.errors.append("Archivo faltante: torchonn/components/__init__.py")
            success = False
        else:
            print(f"  ✅ {components_init} existe")
        
        return success
    
    def check_imports(self) -> bool:
        """Verificar que todas las importaciones funcionan."""
        print("📦 Validando importaciones...")
        
        success = True
        
        # Test imports de layers
        try:
            from torchonn.layers import (
                MZILayer, MZIBlockLinear,
                MicroringResonator, AddDropMRR,
                DirectionalCoupler, Photodetector
            )
            print("  ✅ torchonn.layers - todas las importaciones exitosas")
        except ImportError as e:
            self.errors.append(f"Error importando layers: {e}")
            success = False
        except Exception as e:
            self.errors.append(f"Error inesperado importando layers: {e}")
            success = False
        
        # Test imports de components
        try:
            from torchonn.components import (
                PhaseChangeCell, WDMMultiplexer, MRRWeightBank
            )
            print("  ✅ torchonn.components - todas las importaciones exitosas")
        except ImportError as e:
            self.errors.append(f"Error importando components: {e}")
            success = False
        except Exception as e:
            self.errors.append(f"Error inesperado importando components: {e}")
            success = False
        
        # Test import del módulo principal
        try:
            import torchonn
            if hasattr(torchonn, 'components'):
                print("  ✅ torchonn.components accesible desde módulo principal")
            else:
                self.warnings.append("torchonn.components no accesible desde módulo principal")
        except Exception as e:
            self.errors.append(f"Error importando torchonn principal: {e}")
            success = False
        
        return success
    
    def test_component_instantiation(self) -> bool:
        """Verificar que los componentes se pueden instanciar."""
        print("🔧 Validando instanciación de componentes...")
        
        success = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test de cada componente básico
        test_cases = [
            ("MicroringResonator", "torchonn.layers", {}),
            ("AddDropMRR", "torchonn.layers", {}),
            ("DirectionalCoupler", "torchonn.layers", {}),
            ("Photodetector", "torchonn.layers", {}),
            ("PhaseChangeCell", "torchonn.components", {}),
            ("WDMMultiplexer", "torchonn.components", {"wavelengths": [1530e-9, 1540e-9, 1550e-9]})
        ]
        
        for component_name, module_name, kwargs in test_cases:
            try:
                module = importlib.import_module(module_name)
                component_class = getattr(module, component_name)
                
                # Instanciar con device
                if 'device' not in kwargs:
                    kwargs['device'] = device
                
                instance = component_class(**kwargs)
                print(f"  ✅ {component_name} instanciado correctamente")
                
            except Exception as e:
                self.errors.append(f"Error instanciando {component_name}: {e}")
                success = False
        
        return success
    
    def test_basic_functionality(self) -> bool:
        """Test básico de funcionalidad de componentes."""
        print("⚡ Validando funcionalidad básica...")
        
        success = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Test MicroringResonator
            from torchonn.layers import MicroringResonator
            mrr = MicroringResonator(device=device)
            
            wavelengths = torch.linspace(1530e-9, 1570e-9, 8, device=device)
            input_signal = torch.randn(2, 8, device=device)
            output = mrr(input_signal, wavelengths)
            
            if not isinstance(output, dict):
                self.errors.append("MicroringResonator no retorna dict")
                success = False
            elif 'through' not in output or 'drop' not in output:
                self.errors.append("MicroringResonator output falta 'through' o 'drop'")
                success = False
            else:
                print("  ✅ MicroringResonator funcionalidad básica OK")
            
        except Exception as e:
            self.errors.append(f"Error test MicroringResonator: {e}")
            success = False
        
        try:
            # Test DirectionalCoupler
            from torchonn.layers import DirectionalCoupler
            coupler = DirectionalCoupler(device=device)
            
            input1 = torch.randn(2, 8, device=device)
            input2 = torch.randn(2, 8, device=device)
            out1, out2 = coupler(input1, input2)
            
            if out1.shape != input1.shape or out2.shape != input2.shape:
                self.errors.append("DirectionalCoupler shapes incorrectos")
                success = False
            else:
                print("  ✅ DirectionalCoupler funcionalidad básica OK")
                
        except Exception as e:
            self.errors.append(f"Error test DirectionalCoupler: {e}")
            success = False
        
        try:
            # Test AddDropMRR
            from torchonn.layers import AddDropMRR
            add_drop = AddDropMRR(device=device)
            
            input_signal = torch.randn(2, 8, device=device)
            add_signal = torch.randn(2, 8, device=device)
            wavelengths = torch.linspace(1530e-9, 1570e-9, 8, device=device)
            
            output = add_drop(input_signal, add_signal, wavelengths)
            
            if not isinstance(output, dict):
                self.errors.append("AddDropMRR no retorna dict")
                success = False
            elif 'through' not in output or 'drop' not in output:
                self.errors.append("AddDropMRR output falta 'through' o 'drop'")
                success = False
            else:
                print("  ✅ AddDropMRR funcionalidad básica OK")
                
        except Exception as e:
            self.errors.append(f"Error test AddDropMRR: {e}")
            success = False
        
        return success
    
    def check_backward_compatibility(self) -> bool:
        """Verificar que no se rompió funcionalidad existente."""
        print("🔄 Validando compatibilidad hacia atrás...")
        
        success = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Test componentes originales
            from torchonn.layers import MZILayer, MZIBlockLinear
            
            # Test MZILayer
            mzi = MZILayer(in_features=4, out_features=4, device=device)
            x = torch.randn(2, 4, device=device)
            output = mzi(x)
            
            if output.shape != (2, 4):
                self.errors.append(f"MZILayer shape incorrecto: {output.shape}")
                success = False
            else:
                print("  ✅ MZILayer funciona correctamente")
            
            # Test MZIBlockLinear
            mzi_block = MZIBlockLinear(in_features=4, out_features=3, device=device)
            output = mzi_block(x)
            
            if output.shape != (2, 3):
                self.errors.append(f"MZIBlockLinear shape incorrecto: {output.shape}")
                success = False
            else:
                print("  ✅ MZIBlockLinear funciona correctamente")
                
        except Exception as e:
            self.errors.append(f"Error test compatibilidad: {e}")
            success = False
        
        return success
    
    def check_documentation_consistency(self) -> bool:
        """Verificar consistencia de documentación."""
        print("📚 Validando documentación...")
        
        success = True
        
        # Verificar que __init__.py tienen __all__ consistente
        try:
            from torchonn.layers import __all__ as layers_all
            from torchonn.components import __all__ as components_all
            
            expected_layers = {
                "MZILayer", "MZIBlockLinear", 
                "MicroringResonator", "AddDropMRR",
                "DirectionalCoupler", "Photodetector"
            }
            
            expected_components = {
                "PhaseChangeCell", "WDMMultiplexer", "MRRWeightBank"
            }
            
            if not expected_layers.issubset(set(layers_all)):
                missing = expected_layers - set(layers_all)
                self.warnings.append(f"__all__ layers faltante: {missing}")
            else:
                print("  ✅ torchonn.layers.__all__ consistente")
            
            if not expected_components.issubset(set(components_all)):
                missing = expected_components - set(components_all)
                self.warnings.append(f"__all__ components faltante: {missing}")
            else:
                print("  ✅ torchonn.components.__all__ consistente")
                
        except Exception as e:
            self.warnings.append(f"Error verificando __all__: {e}")
        
        return success
    
    def check_migration_artifacts(self) -> bool:
        """Verificar que no quedan artifacts de migración."""
        print("🧹 Verificando artifacts de migración...")
        
        success = True
        
        # Verificar que examples/advanced_photonic_components.py ya no es usado
        old_file = self.repo_root / "examples" / "advanced_photonic_components.py"
        if old_file.exists():
            self.warnings.append("Archivo original advanced_photonic_components.py aún existe")
        
        # Buscar referencias al archivo viejo
        python_files = list(self.repo_root.rglob("*.py"))
        for py_file in python_files:
            if py_file.name.startswith("advanced_photonic_components"):
                continue  # Skip the source files themselves
                
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if "advanced_photonic_components" in content and "MIGRACIÓN REQUERIDA" not in content:
                    self.warnings.append(f"Referencia a archivo viejo en: {py_file.relative_to(self.repo_root)}")
            except:
                pass  # Skip files that can't be read
        
        return success
    
    def run_comprehensive_validation(self) -> bool:
        """Ejecutar validación completa."""
        print("🎯 VALIDACIÓN COMPLETA POST-MIGRACIÓN")
        print("=" * 80)
        
        all_success = True
        
        # Ejecutar todas las validaciones
        validations = [
            ("Estructura de archivos", self.check_file_structure),
            ("Importaciones", self.check_imports),
            ("Instanciación de componentes", self.test_component_instantiation),
            ("Funcionalidad básica", self.test_basic_functionality),
            ("Compatibilidad hacia atrás", self.check_backward_compatibility),
            ("Consistencia de documentación", self.check_documentation_consistency),
            ("Artifacts de migración", self.check_migration_artifacts)
        ]
        
        for validation_name, validation_func in validations:
            print(f"\n{validation_name}:")
            success = validation_func()
            all_success &= success
            
            if not success:
                print(f"  ❌ {validation_name} FALLÓ")
            else:
                print(f"  ✅ {validation_name} EXITOSO")
        
        # Mostrar resumen
        print("\n" + "=" * 80)
        print("📊 RESUMEN DE VALIDACIÓN")
        print("=" * 80)
        
        if self.errors:
            print(f"❌ ERRORES ENCONTRADOS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  ADVERTENCIAS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        if all_success and not self.errors:
            print("\n🎉 ¡VALIDACIÓN COMPLETAMENTE EXITOSA!")
            print("✅ Todos los componentes migrados funcionan correctamente")
            print("✅ Compatibilidad hacia atrás mantenida")
            print("✅ Estructura de importaciones consistente")
            
            print("\n📚 Importaciones disponibles:")
            print("   from torchonn.layers import MicroringResonator, AddDropMRR")
            print("   from torchonn.layers import DirectionalCoupler, Photodetector")
            print("   from torchonn.components import PhaseChangeCell, WDMMultiplexer")
            
        else:
            print(f"\n❌ VALIDACIÓN FALLÓ")
            print("🔧 Por favor revisar y corregir los errores antes de continuar")
        
        return all_success and not self.errors

def main():
    """Función principal."""
    repo_root = Path.cwd()
    
    # Verificar que estamos en el directorio correcto
    if not (repo_root / "torchonn").exists():
        print("❌ Error: No se encontró directorio torchonn/")
        print("   Por favor ejecutar desde el directorio raíz del repositorio")
        sys.exit(1)
    
    validator = MigrationValidator(repo_root)
    success = validator.run_comprehensive_validation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()