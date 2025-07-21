#!/usr/bin/env python3
"""
🧹 Limpieza del Repositorio PtONN-TESTS

Elimina archivos innecesarios, temporales, backups y scripts de diagnóstico,
manteniendo solo los archivos esenciales del proyecto.
"""

import os
import shutil
from pathlib import Path
import sys

class RepoCleanup:
    """Limpieza inteligente del repositorio."""
    
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).absolute()
        self.removed_files = []
        self.removed_folders = []
        self.kept_files = []
        self.protected_files = []
        
    def get_files_to_remove(self):
        """Definir qué archivos y carpetas eliminar."""
        
        # Carpetas a eliminar completamente
        folders_to_remove = [
            "examples_legacy_backup",
            "safety_backup_*",  # Cualquier backup con fecha
            "ptonn_tests.egg-info",
            "__pycache__",
            ".pytest_cache",
            "htmlcov",
            ".mypy_cache",
            ".coverage",
            "build",
            "dist",
            "*.egg-info"
        ]
        
        # Archivos específicos a eliminar
        files_to_remove = [
            # Scripts de diagnóstico temporales
            "diagnosis.py.backup",
            "diagnostic_report.json", 
            "repair_report.json",
            "corregir2.py",
            "debugmzi.py", 
            "quick_fix_ptonn.py",
            "fix_*.py",  # Todos los scripts de fix
            "debug_*.py",  # Todos los scripts de debug
            "ptonn_repair_coordinator.py",
            "analyze_ci_errors.py",
            "simple_diagnosis.py",
            "simple_test.py",
            
            # Archivos de cobertura y testing temporales
            "coverage.xml",
            ".coverage",
            "pytest.ini.backup",
            
            # Archivos de análisis temporales  
            "photonic_chip_analysis.png",
            "*.pth",  # Modelos temporales (excepto si son importantes)
            
            # Scripts de setup temporales (después de usar)
            "setup_ptonn.sh",
            
            # Archivos de log y temporales
            "*.log",
            "*.tmp",
            "temp_*",
            "tmp_*",
            
            # Backups de archivos Python
            "*.py.backup",
            "*.py.bak",
            
            # Archivos de configuración temporales
            "ci_fixed.yml",  # Ya aplicado si es necesario
            
            # Reportes JSON temporales
            "*_report.json",
            "diagnostic_*.json",
            "repair_*.json",
        ]
        
        # Archivos esenciales que NUNCA eliminar
        essential_files = [
            "README.md",
            "CHANGELOG.md", 
            "requirements.txt",
            "setup.py",
            "pyproject.toml",
            "LICENSE",
            ".gitignore",
            "test_installation.py",  # Mantener para verificaciones futuras
            "diagnosis.py",  # Si funciona, mantenerlo
        ]
        
        # Carpetas esenciales que NUNCA eliminar
        essential_folders = [
            "torchonn",
            "tests", 
            "examples",
            ".github",
            "docs"
        ]
        
        return folders_to_remove, files_to_remove, essential_files, essential_folders
    
    def is_protected(self, path):
        """Verificar si un archivo/carpeta está protegido."""
        _, _, essential_files, essential_folders = self.get_files_to_remove()
        
        path_obj = Path(path)
        
        # Proteger archivos esenciales
        if path_obj.name in essential_files:
            return True
            
        # Proteger carpetas esenciales  
        for essential in essential_folders:
            if str(path_obj).startswith(essential):
                return True
                
        return False
    
    def matches_pattern(self, filename, pattern):
        """Verificar si un archivo coincide con un patrón."""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    def remove_files_and_folders(self, dry_run=True):
        """Eliminar archivos y carpetas innecesarias."""
        folders_to_remove, files_to_remove, _, _ = self.get_files_to_remove()
        
        print(f"🧹 Limpieza del repositorio{'(DRY RUN)' if dry_run else ''}")
        print(f"📁 Directorio: {self.project_root}")
        print("=" * 60)
        
        # Eliminar carpetas
        print("\n📂 CARPETAS A ELIMINAR:")
        all_items = list(self.project_root.rglob("*"))
        
        for item in all_items:
            if item.is_dir():
                relative_path = item.relative_to(self.project_root)
                
                # Verificar si coincide con patrones de carpetas a eliminar
                for pattern in folders_to_remove:
                    if self.matches_pattern(str(relative_path), pattern) or self.matches_pattern(item.name, pattern):
                        if not self.is_protected(str(relative_path)):
                            if dry_run:
                                print(f"   🗑️  ELIMINARÍA: {relative_path}/")
                            else:
                                try:
                                    shutil.rmtree(item)
                                    self.removed_folders.append(str(relative_path))
                                    print(f"   ✅ ELIMINADO: {relative_path}/")
                                except Exception as e:
                                    print(f"   ❌ ERROR: {relative_path}/ - {e}")
                            break
        
        # Eliminar archivos
        print("\n📄 ARCHIVOS A ELIMINAR:")
        for item in self.project_root.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(self.project_root)
                
                # Verificar si coincide con patrones de archivos a eliminar
                should_remove = False
                for pattern in files_to_remove:
                    if self.matches_pattern(str(relative_path), pattern) or self.matches_pattern(item.name, pattern):
                        should_remove = True
                        break
                
                if should_remove and not self.is_protected(str(relative_path)):
                    if dry_run:
                        print(f"   🗑️  ELIMINARÍA: {relative_path}")
                    else:
                        try:
                            item.unlink()
                            self.removed_files.append(str(relative_path))
                            print(f"   ✅ ELIMINADO: {relative_path}")
                        except Exception as e:
                            print(f"   ❌ ERROR: {relative_path} - {e}")
    
    def show_what_remains(self):
        """Mostrar qué archivos importantes se mantienen."""
        print("\n📋 ARCHIVOS QUE SE MANTIENEN:")
        
        important_files = []
        for item in self.project_root.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(self.project_root)
                
                # Mostrar solo archivos en la raíz y algunos importantes
                if (len(relative_path.parts) == 1 or  # Archivos en raíz
                    str(relative_path).startswith('torchonn/') or
                    str(relative_path).startswith('tests/') or  
                    str(relative_path).startswith('.github/')):
                    important_files.append(str(relative_path))
        
        # Ordenar y mostrar
        important_files.sort()
        for file in important_files[:20]:  # Mostrar primeros 20
            print(f"   ✅ {file}")
        
        if len(important_files) > 20:
            print(f"   ... y {len(important_files) - 20} archivos más")
    
    def update_gitignore(self):
        """Actualizar .gitignore con patrones para evitar futuros archivos innecesarios."""
        gitignore_additions = """
# Archivos de diagnóstico y temporales
diagnostic_report.json
repair_report.json  
*_report.json
debug_*.py
fix_*.py
*.py.backup
*.py.bak

# Archivos de coverage y testing temporales
.coverage
htmlcov/
.pytest_cache/
coverage.xml

# Archivos temporales
*.tmp
*.log
temp_*
tmp_*

# Modelos temporales
*.pth

# Backups de seguridad
safety_backup_*/
*_backup_*/

# Build artifacts
build/
dist/
*.egg-info/
"""
        
        gitignore_path = self.project_root / ".gitignore"
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                current_content = f.read()
                
            # Solo añadir si no existe ya
            if "diagnostic_report.json" not in current_content:
                with open(gitignore_path, 'a') as f:
                    f.write(gitignore_additions)
                print("✅ .gitignore actualizado")
            else:
                print("✅ .gitignore ya contiene las reglas necesarias")
        else:
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_additions)
            print("✅ .gitignore creado")
    
    def run_cleanup(self, dry_run=True):
        """Ejecutar limpieza completa."""
        print("🧹 LIMPIEZA INTELIGENTE DEL REPOSITORIO")
        print("=" * 60)
        
        # Mostrar qué se va a eliminar
        self.remove_files_and_folders(dry_run=dry_run)
        
        if not dry_run:
            # Actualizar .gitignore
            print("\n📝 Actualizando .gitignore...")
            self.update_gitignore()
        
        # Mostrar resumen
        print(f"\n📊 RESUMEN:")
        if dry_run:
            print("   🔍 Esto es una simulación (DRY RUN)")
            print("   📄 Para ejecutar realmente: python cleanup_repo.py --execute")
        else:
            print(f"   🗑️  Carpetas eliminadas: {len(self.removed_folders)}")
            print(f"   📄 Archivos eliminados: {len(self.removed_files)}")
            print("   ✅ Limpieza completada")
        
        # Mostrar archivos importantes que se mantienen
        self.show_what_remains()
        
        if not dry_run:
            print(f"\n🚀 SIGUIENTE PASO:")
            print("   git add -A")
            print("   git commit -m 'Clean: Removed temporary files and diagnostics'")
            print("   git push")

def main():
    """Función principal."""
    
    # Verificar argumentos
    dry_run = True
    if len(sys.argv) > 1 and sys.argv[1] in ["--execute", "-x", "--real"]:
        dry_run = False
    
    print("🧹 Limpieza del Repositorio PtONN-TESTS")
    
    if dry_run:
        print("🔍 MODO SIMULACIÓN - Solo muestra qué se eliminaría")
        print("💡 Para ejecutar realmente: python cleanup_repo.py --execute")
    else:
        print("⚠️  MODO EJECUCIÓN - Se eliminarán archivos realmente")
        
        # Confirmación de seguridad
        response = input("\n¿Estás seguro? (escribir 'SI' para confirmar): ")
        if response != "SI":
            print("❌ Operación cancelada")
            return 1
    
    print()
    
    # Ejecutar limpieza
    cleanup = RepoCleanup()
    cleanup.run_cleanup(dry_run=dry_run)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())