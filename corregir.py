#!/usr/bin/env python3
"""
Script de Corrección de Errores de Sintaxis PtONN-TESTS
======================================================

Corrige automáticamente los errores detectados:
1. Import faltante de MicroringResonator en add_drop_mrr.py
2. Asteriscos dobles (**) que deben ser dobles underscores (__)
3. Otros problemas de sintaxis

Uso: python fix_syntax_errors.py
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

class SyntaxFixer:
    """Corrector automático de errores de sintaxis"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.fixes_applied = []
        
    def log(self, message: str):
        """Log de cambios"""
        print(f"✅ {message}")
        self.fixes_applied.append(message)
    
    def fix_double_asterisks(self, file_path: Path) -> bool:
        """Corregir ** por __ en variables especiales"""
        if not file_path.exists():
            return False
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Patterns to fix
        patterns = [
            (r'\*\*version\*\*', '__version__'),
            (r'\*\*author\*\*', '__author__'),
            (r'\*\*all\*\*', '__all__'),
            (r'\*\*name\*\*', '__name__'),
            (r'\*\*doc\*\*', '__doc__'),
            (r'\*\*file\*\*', '__file__'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                self.log(f"Fixed {pattern} → {replacement} in {file_path}")
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    
    def fix_add_drop_mrr_import(self) -> bool:
        """Corregir import faltante en add_drop_mrr.py"""
        file_path = self.repo_path / "torchonn" / "components" / "add_drop_mrr.py"
        
        if not file_path.exists():
            print(f"⚠️  Archivo no encontrado: {file_path}")
            return False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if MicroringResonator import is missing
        if "from .microring_resonator import MicroringResonator" in content:
            print(f"✅ Import de MicroringResonator ya existe en {file_path}")
            return False  # Already fixed
        
        # Check if MicroringResonator is used but not imported
        if "MicroringResonator(" in content and "from .microring_resonator import" not in content:
            # Find the right place to add the import
            if "from .base_component import BasePhotonicComponent" in content:
                # Add the import after the base_component import
                content = content.replace(
                    "from .base_component import BasePhotonicComponent",
                    "from .base_component import BasePhotonicComponent\nfrom .microring_resonator import MicroringResonator"
                )
            else:
                # Add after the typing import
                if "from typing import" in content:
                    # Find the last typing import
                    lines = content.split('\n')
                    insert_index = -1
                    for i, line in enumerate(lines):
                        if line.strip().startswith("from typing import") or line.strip().startswith("from .base_component import"):
                            insert_index = i
                    
                    if insert_index >= 0:
                        lines.insert(insert_index + 1, "from .microring_resonator import MicroringResonator")
                        content = '\n'.join(lines)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.log(f"Added MicroringResonator import to {file_path}")
            return True
        
        return False
    
    def fix_components_init(self) -> bool:
        """Corregir imports en components/__init__.py"""
        file_path = self.repo_path / "torchonn" / "components" / "__init__.py"
        
        if not file_path.exists():
            return False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Ensure AddDropMRR import is properly handled
        if "AddDropMRR = _safe_import('add_drop_mrr', 'AddDropMRR')" in content:
            return False  # Already handled
        
        # Look for the section where imports are done
        if "_safe_import('add_drop_mrr', 'AddDropMRR')" not in content:
            # Add import handling for AddDropMRR if missing
            mrr_import_block = """
AddDropMRR = _safe_import('add_drop_mrr', 'AddDropMRR')  
if AddDropMRR:
    __all__.append('AddDropMRR')
"""
            
            # Insert after MicroringResonator import
            if "if MicroringResonator:" in content and "__all__.append('MicroringResonator')" in content:
                content = content.replace(
                    "if MicroringResonator:\n    __all__.append('MicroringResonator')",
                    "if MicroringResonator:\n    __all__.append('MicroringResonator')" + mrr_import_block
                )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.log(f"Fixed AddDropMRR import handling in {file_path}")
                return True
        
        return False
    
    def create_backup(self):
        """Crear backup antes de las correcciones"""
        backup_path = self.repo_path / f"backup_syntax_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if (self.repo_path / "torchonn").exists():
                backup_path.mkdir(exist_ok=True)
                shutil.copytree(
                    self.repo_path / "torchonn",
                    backup_path / "torchonn",
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc')
                )
                print(f"📁 Backup creado en: {backup_path}")
                return backup_path
        except Exception as e:
            print(f"⚠️  No se pudo crear backup: {e}")
        
        return None
    
    def scan_for_remaining_issues(self):
        """Buscar problemas restantes"""
        print("\n🔍 Buscando problemas restantes...")
        
        issues_found = []
        
        for py_file in self.repo_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for remaining ** patterns that should be __
                if re.search(r'\*\*(version|author|all|name|doc|file)\*\*', content):
                    issues_found.append(f"⚠️  Patrón ** restante en: {py_file}")
                
                # Check for common import issues
                if "MicroringResonator(" in content and "from .microring_resonator import" not in content and "class MicroringResonator" not in content:
                    issues_found.append(f"⚠️  Uso de MicroringResonator sin import en: {py_file}")
                    
            except Exception as e:
                issues_found.append(f"⚠️  Error leyendo {py_file}: {e}")
        
        if issues_found:
            for issue in issues_found:
                print(issue)
        else:
            print("✅ No se encontraron problemas restantes")
        
        return issues_found
    
    def verify_fixes(self) -> bool:
        """Verificar que las correcciones funcionan"""
        print("\n🧪 Verificando correcciones...")
        
        try:
            # Save current working directory
            original_cwd = os.getcwd()
            
            # Change to repo directory for imports
            os.chdir(self.repo_path)
            
            # Try to import the main package
            import sys
            if str(self.repo_path) not in sys.path:
                sys.path.insert(0, str(self.repo_path))
            
            try:
                import torchonn
                self.log("Import principal funciona")
                
                # Check version
                version = getattr(torchonn, '__version__', None)
                if version:
                    self.log(f"Versión detectada: {version}")
                
            except Exception as e:
                print(f"❌ Error importando torchonn: {e}")
                return False
            
            # Try to import layers
            try:
                from torchonn.layers import MZILayer, MZIBlockLinear
                self.log("Import de layers funciona")
            except ImportError as e:
                print(f"⚠️  Layers import issue: {e}")
            
            # Try to import components if available
            try:
                from torchonn.components import BasePhotonicComponent
                self.log("Import de componentes básicos funciona")
                
                # Try more specific imports
                try:
                    from torchonn.components import MicroringResonator
                    self.log("Import de MicroringResonator funciona")
                except ImportError as e:
                    print(f"⚠️  MicroringResonator import issue: {e}")
                
                try:
                    from torchonn.components import AddDropMRR
                    self.log("Import de AddDropMRR funciona")
                except ImportError as e:
                    print(f"⚠️  AddDropMRR import issue: {e}")
                    
            except ImportError as e:
                print(f"⚠️  Components import issue: {e}")
            
            # Restore working directory
            os.chdir(original_cwd)
            
            return True
            
        except Exception as e:
            print(f"❌ Error durante verificación: {e}")
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
            return False
    
    def fix_all_files(self) -> List[str]:
        """Corregir todos los archivos problemáticos"""
        print("🔧 Iniciando corrección automática de errores...")
        
        # Files to check for ** → __ fixes
        files_to_check = [
            "torchonn/__init__.py",
            "torchonn/layers/__init__.py", 
            "torchonn/models/__init__.py",
            "torchonn/components/__init__.py",
            "torchonn/ops/__init__.py",
            "torchonn/utils/__init__.py",
            "torchonn/devices/__init__.py",
        ]
        
        # Fix double asterisks in all files
        for file_path in files_to_check:
            full_path = self.repo_path / file_path
            if full_path.exists():
                self.fix_double_asterisks(full_path)
            else:
                print(f"⚠️  Archivo no encontrado: {full_path}")
        
        # Fix specific import issues
        self.fix_add_drop_mrr_import()
        self.fix_components_init()
        
        return self.fixes_applied


def main():
    """Función principal"""
    print("🚀 Corrector de Errores de Sintaxis PtONN-TESTS")
    print("=" * 60)
    
    # Check if we're in the right directory
    current_path = Path.cwd()
    if not (current_path / "torchonn").exists():
        print("❌ Error: No se encuentra el directorio 'torchonn'")
        print(f"   Directorio actual: {current_path}")
        print("   Asegúrate de ejecutar este script desde el directorio raíz del proyecto")
        return
    
    fixer = SyntaxFixer(current_path)
    
    # Create backup
    backup_path = fixer.create_backup()
    
    # Apply fixes
    fixes = fixer.fix_all_files()
    
    if fixes:
        print(f"\n✅ Se aplicaron {len(fixes)} correcciones:")
        for fix in fixes:
            print(f"  • {fix}")
    else:
        print("\n✅ No se encontraron problemas que corregir")
    
    # Scan for remaining issues
    remaining_issues = fixer.scan_for_remaining_issues()
    
    # Verify fixes
    verification_ok = fixer.verify_fixes()
    
    print("\n" + "=" * 60)
    if verification_ok and (fixes or not remaining_issues):
        print("🎉 ¡Correcciones aplicadas exitosamente!")
        print("\n📋 Próximos pasos:")
        print("1. Ejecuta: python quick_test.py")
        print("2. Verifica: python -c \"from torchonn.components import AddDropMRR; print('✅ OK')\"")
    elif not fixes and not remaining_issues:
        print("✅ No había errores que corregir")
    else:
        print("⚠️  Hay problemas restantes que requieren atención manual")
        if remaining_issues:
            print("\nProblemas detectados:")
            for issue in remaining_issues[:5]:  # Show first 5
                print(f"  {issue}")
        
    if backup_path:
        print(f"\n📁 Backup disponible en: {backup_path}")
    
    print(f"\n🔍 Comandos para verificar manualmente:")
    print(f"  cd {current_path}")
    print("  python -c \"import torchonn; print('✅ Import OK')\"")
    print("  python -c \"from torchonn.components import AddDropMRR; print('✅ AddDropMRR OK')\"")


if __name__ == "__main__":
    main()