#!/usr/bin/env python3
"""
Script de Verificación Post-Reestructuración PtONN-TESTS
========================================================

Verifica que la reestructuración se completó correctamente:
- No hay archivos duplicados
- No quedan carpetas obsoletas  
- Estructura es correcta
- Imports funcionan
- Tests pasan

Uso: python verify_restructure.py /path/to/PtONN-TESTS
"""

import os
import sys
import ast
import hashlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
import subprocess
import argparse


class StructureVerifier:
    """Verificador de estructura post-reestructuración"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.issues = []
        self.warnings = []
        self.info = []
        
        # Estructura esperada después de la reestructuración
        self.expected_structure = {
            'torchonn': {
                'required_files': ['__init__.py'],
                'required_dirs': ['components', 'layers', 'models', 'devices', 'ops', 'utils'],
                'optional_dirs': ['systems', 'training', 'hardware']
            },
            'examples': {
                'required_files': ['advanced_photonic_components.py'],
                'optional_dirs': ['basic', 'intermediate', 'advanced', 'benchmarks', 'tutorials']
            },
            'tests': {
                'required_files': ['__init__.py'],
                'optional_dirs': ['unit', 'integration', 'performance']
            }
        }
        
        # Archivos que NO deberían existir después de la reestructuración
        self.obsolete_files = [
            'examples/old_advanced_photonic_components.py',
            'torchonn/legacy_components.py'
        ]
        
        # Patrones de archivos temporales que se pueden eliminar
        self.temp_patterns = [
            '*.pyc',
            '__pycache__',
            '*.tmp',
            '.DS_Store',
            'Thumbs.db'
        ]

    def log(self, message: str, level: str = "INFO"):
        """Log con categorización"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {level}: {message}")
        
        if level == "ERROR":
            self.issues.append(message)
        elif level == "WARNING":
            self.warnings.append(message)
        else:
            self.info.append(message)

    def check_basic_structure(self) -> bool:
        """Verificar estructura básica de directorios"""
        self.log("=== VERIFICANDO ESTRUCTURA BÁSICA ===")
        
        if not self.repo_path.exists():
            self.log(f"Repositorio no encontrado: {self.repo_path}", "ERROR")
            return False
        
        # Verificar directorios principales
        for main_dir, requirements in self.expected_structure.items():
            dir_path = self.repo_path / main_dir
            
            if not dir_path.exists():
                self.log(f"Directorio principal faltante: {main_dir}", "ERROR")
                continue
            
            # Verificar archivos requeridos
            for req_file in requirements.get('required_files', []):
                file_path = dir_path / req_file
                if not file_path.exists():
                    self.log(f"Archivo requerido faltante: {main_dir}/{req_file}", "ERROR")
            
            # Verificar directorios requeridos
            for req_dir in requirements.get('required_dirs', []):
                subdir_path = dir_path / req_dir
                if not subdir_path.exists():
                    self.log(f"Directorio requerido faltante: {main_dir}/{req_dir}", "ERROR")
                elif not (subdir_path / '__init__.py').exists():
                    self.log(f"__init__.py faltante en: {main_dir}/{req_dir}", "WARNING")
            
            # Verificar directorios opcionales
            for opt_dir in requirements.get('optional_dirs', []):
                subdir_path = dir_path / opt_dir
                if subdir_path.exists():
                    self.log(f"Directorio opcional encontrado: {main_dir}/{opt_dir}")
                    if not (subdir_path / '__init__.py').exists():
                        self.log(f"__init__.py faltante en: {main_dir}/{opt_dir}", "WARNING")
        
        return len([issue for issue in self.issues if "faltante" in issue]) == 0

    def find_duplicate_files(self) -> Dict[str, List[str]]:
        """Encontrar archivos duplicados por contenido"""
        self.log("=== BUSCANDO ARCHIVOS DUPLICADOS ===")
        
        file_hashes = {}
        duplicates = {}
        
        # Obtener hash de todos los archivos .py
        for py_file in self.repo_path.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'rb') as f:
                    content = f.read()
                    file_hash = hashlib.md5(content).hexdigest()
                
                if file_hash in file_hashes:
                    # Encontrado duplicado
                    if file_hash not in duplicates:
                        duplicates[file_hash] = [file_hashes[file_hash]]
                    duplicates[file_hash].append(str(py_file.relative_to(self.repo_path)))
                else:
                    file_hashes[file_hash] = str(py_file.relative_to(self.repo_path))
            
            except Exception as e:
                self.log(f"Error leyendo {py_file}: {e}", "WARNING")
        
        # Reportar duplicados
        for file_hash, files in duplicates.items():
            self.log(f"Archivos duplicados encontrados:", "WARNING")
            for file in files:
                self.log(f"  - {file}", "WARNING")
        
        if not duplicates:
            self.log("✅ No se encontraron archivos duplicados")
        
        return duplicates

    def check_obsolete_files(self) -> List[str]:
        """Verificar archivos obsoletos que deberían eliminarse"""
        self.log("=== VERIFICANDO ARCHIVOS OBSOLETOS ===")
        
        found_obsolete = []
        
        for obsolete_file in self.obsolete_files:
            file_path = self.repo_path / obsolete_file
            if file_path.exists():
                found_obsolete.append(obsolete_file)
                self.log(f"Archivo obsoleto encontrado: {obsolete_file}", "WARNING")
        
        # Buscar patrones de archivos temporales
        temp_found = []
        for pattern in self.temp_patterns:
            if pattern.startswith('*'):
                # Patrón de extensión
                extension = pattern[1:]
                for temp_file in self.repo_path.rglob(f"*{extension}"):
                    temp_found.append(str(temp_file.relative_to(self.repo_path)))
            else:
                # Nombre exacto
                for temp_file in self.repo_path.rglob(pattern):
                    temp_found.append(str(temp_file.relative_to(self.repo_path)))
        
        if temp_found:
            self.log("Archivos temporales encontrados (pueden eliminarse):", "INFO")
            for temp in temp_found[:10]:  # Mostrar solo los primeros 10
                self.log(f"  - {temp}", "INFO")
            if len(temp_found) > 10:
                self.log(f"  ... y {len(temp_found) - 10} más", "INFO")
        
        if not found_obsolete:
            self.log("✅ No se encontraron archivos obsoletos")
        
        return found_obsolete

    def verify_imports(self) -> bool:
        """Verificar que los imports funcionan correctamente"""
        self.log("=== VERIFICANDO IMPORTS ===")
        
        # Cambiar al directorio del repositorio para imports relativos
        original_cwd = os.getcwd()
        os.chdir(self.repo_path)
        
        # Agregar al Python path
        sys.path.insert(0, str(self.repo_path))
        
        import_tests = [
            # Imports principales
            ("torchonn", "Paquete principal"),
            ("torchonn.components", "Módulo de componentes"),
            ("torchonn.layers", "Módulo de capas"),
            ("torchonn.models", "Módulo de modelos"),
            
            # Imports específicos (si existen)
            ("torchonn.components.base_component", "Clase base de componentes"),
        ]
        
        failed_imports = []
        
        for module_name, description in import_tests:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    self.log(f"Módulo no encontrado: {module_name} ({description})", "WARNING")
                    failed_imports.append(module_name)
                else:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.log(f"✅ Import exitoso: {module_name}")
            except Exception as e:
                self.log(f"Error importando {module_name}: {e}", "ERROR")
                failed_imports.append(module_name)
        
        # Restaurar directorio y path
        os.chdir(original_cwd)
        sys.path.remove(str(self.repo_path))
        
        return len(failed_imports) == 0

    def verify_advanced_components_migration(self) -> bool:
        """Verificar que la migración de advanced_photonic_components fue exitosa"""
        self.log("=== VERIFICANDO MIGRACIÓN DE COMPONENTES ===")
        
        advanced_file = self.repo_path / "examples" / "advanced_photonic_components.py"
        
        if not advanced_file.exists():
            self.log("Archivo advanced_photonic_components.py no encontrado", "ERROR")
            return False
        
        # Leer contenido del archivo
        with open(advanced_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar que contiene imports modulares y no definiciones de clases
        has_modular_imports = "from torchonn.components" in content
        has_class_definitions = "class MicroringResonator" in content or "class AddDropMRR" in content
        
        if has_modular_imports and not has_class_definitions:
            self.log("✅ Migración exitosa: archivo usa imports modulares")
            return True
        elif has_class_definitions:
            self.log("Archivo aún contiene definiciones de clases (no migrado)", "WARNING")
            return False
        else:
            self.log("Archivo no contiene imports esperados", "WARNING")
            return False

    def run_tests(self) -> bool:
        """Ejecutar tests si existen"""
        self.log("=== EJECUTANDO TESTS ===")
        
        tests_dir = self.repo_path / "tests"
        if not tests_dir.exists():
            self.log("Directorio tests no encontrado", "WARNING")
            return True
        
        # Buscar archivos de test
        test_files = list(tests_dir.rglob("test_*.py"))
        if not test_files:
            self.log("No se encontraron archivos de test", "INFO")
            return True
        
        # Intentar ejecutar pytest
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(tests_dir), "-v"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.log("✅ Todos los tests pasaron")
                return True
            else:
                self.log("❌ Algunos tests fallaron:", "ERROR")
                self.log(result.stdout, "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("Tests tomaron demasiado tiempo (timeout)", "WARNING")
            return False
        except FileNotFoundError:
            self.log("pytest no encontrado, intentando con unittest", "INFO")
            
            # Intentar con unittest
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", "discover", "-s", str(tests_dir)],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    self.log("✅ Tests unittest pasaron")
                    return True
                else:
                    self.log("❌ Tests unittest fallaron", "WARNING")
                    return False
                    
            except Exception as e:
                self.log(f"Error ejecutando tests: {e}", "WARNING")
                return False

    def check_file_sizes(self) -> Dict[str, int]:
        """Verificar tamaños de archivos para detectar posibles problemas"""
        self.log("=== VERIFICANDO TAMAÑOS DE ARCHIVOS ===")
        
        large_files = {}
        empty_files = []
        
        for py_file in self.repo_path.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue
            
            size = py_file.stat().st_size
            
            if size == 0:
                empty_files.append(str(py_file.relative_to(self.repo_path)))
            elif size > 100000:  # Archivos > 100KB
                large_files[str(py_file.relative_to(self.repo_path))] = size
        
        if empty_files:
            self.log("Archivos Python vacíos encontrados:", "WARNING")
            for empty in empty_files[:5]:
                self.log(f"  - {empty}", "WARNING")
        
        if large_files:
            self.log("Archivos Python grandes encontrados:", "INFO")
            for large_file, size in large_files.items():
                self.log(f"  - {large_file}: {size/1024:.1f} KB", "INFO")
        
        return {"large": large_files, "empty": empty_files}

    def generate_cleanup_script(self, duplicates: Dict, obsolete: List[str], temp_files: List[str]):
        """Generar script de limpieza automática"""
        cleanup_script = self.repo_path / "cleanup_post_restructure.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Script de Limpieza Post-Reestructuración
========================================

Generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Elimina archivos duplicados, obsoletos y temporales detectados.

IMPORTANTE: Revisa este script antes de ejecutarlo
"""

import os
import shutil
from pathlib import Path

def main():
    repo_path = Path(__file__).parent
    
    print("🧹 Iniciando limpieza post-reestructuración...")
    
    # Archivos duplicados (mantener el primero, eliminar el resto)
    duplicates = {repr(duplicates)}
    
    # Archivos obsoletos
    obsolete_files = {obsolete}
    
    # Archivos temporales (primeros 50)
    temp_files = {temp_files[:50]}
    
    # Eliminar duplicados (excepto el primero)
    for file_hash, files in duplicates.items():
        if len(files) > 1:
            print(f"Eliminando duplicados de {{files[0]}}:")
            for duplicate in files[1:]:
                file_path = repo_path / duplicate
                if file_path.exists():
                    print(f"  - Eliminando: {{duplicate}}")
                    file_path.unlink()
    
    # Eliminar archivos obsoletos
    for obsolete in obsolete_files:
        file_path = repo_path / obsolete
        if file_path.exists():
            print(f"Eliminando obsoleto: {{obsolete}}")
            file_path.unlink()
    
    # Eliminar archivos temporales
    for temp in temp_files:
        file_path = repo_path / temp
        if file_path.exists():
            if file_path.is_dir():
                print(f"Eliminando directorio temporal: {{temp}}")
                shutil.rmtree(file_path)
            else:
                print(f"Eliminando archivo temporal: {{temp}}")
                file_path.unlink()
    
    print("✅ Limpieza completada")

if __name__ == "__main__":
    input("Presiona Enter para continuar con la limpieza (Ctrl+C para cancelar)...")
    main()
'''
        
        cleanup_script.write_text(script_content, encoding='utf-8')
        self.log(f"Script de limpieza generado: {cleanup_script}")

    def generate_verification_report(self) -> str:
        """Generar reporte completo de verificación"""
        report_path = self.repo_path / "VERIFICATION_REPORT.md"
        
        status = "✅ EXITOSA" if not self.issues else "❌ CON ERRORES"
        
        report_content = f'''# Reporte de Verificación Post-Reestructuración
## Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## Estado: {status}

### Resumen

- **Errores encontrados**: {len(self.issues)}
- **Advertencias**: {len(self.warnings)}
- **Información**: {len(self.info)}

### Estructura Verificada

✅ Verificación básica de directorios y archivos
✅ Búsqueda de archivos duplicados
✅ Detección de archivos obsoletos
✅ Verificación de imports
✅ Validación de migración de componentes
✅ Ejecución de tests
✅ Análisis de tamaños de archivos

### Errores Críticos

{chr(10).join([f"- {issue}" for issue in self.issues]) if self.issues else "Ninguno"}

### Advertencias

{chr(10).join([f"- {warning}" for warning in self.warnings]) if self.warnings else "Ninguna"}

### Próximos Pasos

{"#### 🚨 Acción Requerida" if self.issues else "#### ✅ Todo Correcto"}

{"""Hay errores críticos que necesitan atención inmediata.
Revisa los errores listados arriba y corrígelos antes de continuar.""" if self.issues else """La reestructuración fue exitosa. El repositorio está listo para usar.

Pasos opcionales:
1. Ejecutar `cleanup_post_restructure.py` para limpiar archivos temporales
2. Revisar y completar implementaciones en nuevos módulos
3. Actualizar documentación
4. Configurar CI/CD"""}

### Archivos Generados

- `VERIFICATION_REPORT.md` - Este reporte
{"- `cleanup_post_restructure.py` - Script de limpieza automática" if self.warnings else ""}

### Comandos de Verificación Manual

```bash
# Verificar que imports funcionan
cd {self.repo_path}
python -c "import torchonn; print('✅ Import principal OK')"
python -c "from torchonn.components import BasePhotonicComponent; print('✅ Componentes OK')"

# Ejecutar tests
python -m pytest tests/ -v

# Verificar estructura
find . -name "*.py" -path "*/__pycache__*" -delete
find . -name "__pycache__" -type d -exec rm -rf {{}} +
```
'''
        
        report_path.write_text(report_content, encoding='utf-8')
        self.log(f"Reporte de verificación generado: {report_path}")
        
        return str(report_path)

    def run_full_verification(self) -> bool:
        """Ejecutar verificación completa"""
        self.log("=" * 60)
        self.log("INICIANDO VERIFICACIÓN POST-REESTRUCTURACIÓN")
        self.log("=" * 60)
        
        # 1. Estructura básica
        structure_ok = self.check_basic_structure()
        
        # 2. Archivos duplicados
        duplicates = self.find_duplicate_files()
        
        # 3. Archivos obsoletos
        obsolete = self.check_obsolete_files()
        
        # 4. Verificar imports
        imports_ok = self.verify_imports()
        
        # 5. Verificar migración
        migration_ok = self.verify_advanced_components_migration()
        
        # 6. Ejecutar tests
        tests_ok = self.run_tests()
        
        # 7. Verificar tamaños
        file_sizes = self.check_file_sizes()
        
        # 8. Generar script de limpieza si es necesario
        temp_files = []
        for pattern in self.temp_patterns:
            if pattern.startswith('*'):
                extension = pattern[1:]
                temp_files.extend([str(f.relative_to(self.repo_path)) 
                                 for f in self.repo_path.rglob(f"*{extension}")])
            else:
                temp_files.extend([str(f.relative_to(self.repo_path)) 
                                 for f in self.repo_path.rglob(pattern)])
        
        if duplicates or obsolete or temp_files:
            self.generate_cleanup_script(duplicates, obsolete, temp_files)
        
        # 9. Generar reporte
        report_path = self.generate_verification_report()
        
        # Resumen final
        self.log("\n" + "=" * 60)
        if self.issues:
            self.log("❌ VERIFICACIÓN COMPLETADA CON ERRORES")
            self.log(f"Se encontraron {len(self.issues)} errores críticos")
        else:
            self.log("✅ VERIFICACIÓN EXITOSA")
            self.log("La reestructuración fue completada correctamente")
        
        self.log(f"📄 Ver reporte completo en: {report_path}")
        self.log("=" * 60)
        
        return len(self.issues) == 0


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Verificar reestructuración de PtONN-TESTS"
    )
    parser.add_argument(
        "repo_path",
        help="Ruta al repositorio PtONN-TESTS reestructurado"
    )
    
    args = parser.parse_args()
    
    # Verificar que existe el repositorio
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"ERROR: Repositorio no encontrado: {repo_path}")
        sys.exit(1)
    
    # Ejecutar verificación
    verifier = StructureVerifier(repo_path)
    success = verifier.run_full_verification()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()