#!/usr/bin/env python3
"""
Suite de tests para verificar el funcionamiento del script de restauración.
Crea entornos simulados y verifica que la restauración funciona correctamente.
"""

import os
import shutil
import tempfile
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import time

class TestRestoreBackup(unittest.TestCase):
    """Tests para el script de restauración"""
    
    def setUp(self):
        """Configuración antes de cada test"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Crear estructura de repositorio simulado
        self.create_mock_repo()
        
    def tearDown(self):
        """Limpieza después de cada test"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
        
    def create_mock_repo(self):
        """Crea un repositorio simulado para testing"""
        # Crear .git (simular repo Git)
        git_dir = self.test_dir / '.git'
        git_dir.mkdir()
        (git_dir / 'config').write_text('[core]\n    repositoryformatversion = 0')
        
        # Crear archivos que deben preservarse
        (self.test_dir / 'README.md').write_text('# Test Repo')
        (self.test_dir / 'LICENSE').write_text('MIT License')
        (self.test_dir / '.gitignore').write_text('__pycache__/')
        
        # Crear carpeta backup con contenido
        backup_dir = self.test_dir / 'backup'
        backup_dir.mkdir()
        
        # Contenido del backup
        (backup_dir / 'main.py').write_text('# Main script from backup')
        (backup_dir / 'utils.py').write_text('# Utils from backup')
        
        backup_subdir = backup_dir / 'subdir'
        backup_subdir.mkdir()
        (backup_subdir / 'file.txt').write_text('Backup file content')
        
        # Crear archivos "nuevos" que deben eliminarse
        (self.test_dir / 'new_file.py').write_text('# New file to be removed')
        (self.test_dir / 'temp_folder').mkdir()
        (self.test_dir / 'temp_folder' / 'temp.txt').write_text('Temporary content')
        
    def test_basic_restore_functionality(self):
        """Test básico de restauración"""
        print("🧪 Test: Funcionalidad básica de restauración")
        
        # Verificar estado inicial
        self.assertTrue((self.test_dir / 'backup').exists())
        self.assertTrue((self.test_dir / 'new_file.py').exists())
        self.assertTrue((self.test_dir / 'temp_folder').exists())
        
        # Importar y ejecutar el script
        sys.path.insert(0, str(self.test_dir.parent))
        
        # Simular el script de restauración
        from simple_restore import restore_from_backup
        
        # Ejecutar restauración
        result = restore_from_backup()
        
        # Verificar resultados
        self.assertTrue(result, "La restauración debería haber sido exitosa")
        
        # Verificar que archivos del backup fueron restaurados
        self.assertTrue((self.test_dir / 'main.py').exists())
        self.assertTrue((self.test_dir / 'utils.py').exists())
        self.assertTrue((self.test_dir / 'subdir' / 'file.txt').exists())
        
        # Verificar que archivos nuevos fueron eliminados
        self.assertFalse((self.test_dir / 'new_file.py').exists())
        self.assertFalse((self.test_dir / 'temp_folder').exists())
        
        # Verificar que archivos preservados siguen ahí
        self.assertTrue((self.test_dir / 'README.md').exists())
        self.assertTrue((self.test_dir / 'LICENSE').exists())
        self.assertTrue((self.test_dir / '.gitignore').exists())
        self.assertTrue((self.test_dir / '.git').exists())
        
        # Verificar que backup fue eliminado
        self.assertFalse((self.test_dir / 'backup').exists())
        
        print("✅ Test básico pasado")
        
    def test_preserve_critical_files(self):
        """Test de preservación de archivos críticos"""
        print("🧪 Test: Preservación de archivos críticos")
        
        # Crear archivos adicionales que deben preservarse
        (self.test_dir / '.gitattributes').write_text('*.py text')
        
        # Contenido original que debe mantenerse
        original_readme = (self.test_dir / 'README.md').read_text()
        original_license = (self.test_dir / 'LICENSE').read_text()
        
        # Ejecutar restauración
        sys.path.insert(0, str(self.test_dir.parent))
        from simple_restore import restore_from_backup
        
        result = restore_from_backup()
        self.assertTrue(result)
        
        # Verificar que archivos críticos mantienen su contenido
        self.assertEqual((self.test_dir / 'README.md').read_text(), original_readme)
        self.assertEqual((self.test_dir / 'LICENSE').read_text(), original_license)
        self.assertTrue((self.test_dir / '.gitattributes').exists())
        
        print("✅ Test de preservación pasado")
        
    def test_safety_backup_creation(self):
        """Test de creación de backup de seguridad"""
        print("🧪 Test: Creación de backup de seguridad")
        
        # Crear contenido único para verificar el backup
        unique_content = "# Unique content for safety test"
        (self.test_dir / 'unique_file.py').write_text(unique_content)
        
        # Ejecutar restauración
        sys.path.insert(0, str(self.test_dir.parent))
        from simple_restore import restore_from_backup
        
        result = restore_from_backup()
        self.assertTrue(result)
        
        # Verificar que se creó backup de seguridad
        safety_backups = [d for d in self.test_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('safety_backup_')]
        
        self.assertEqual(len(safety_backups), 1, "Debe haberse creado exactamente un backup de seguridad")
        
        safety_backup = safety_backups[0]
        
        # Verificar que el backup contiene el archivo único
        unique_file_backup = safety_backup / 'unique_file.py'
        self.assertTrue(unique_file_backup.exists())
        self.assertEqual(unique_file_backup.read_text(), unique_content)
        
        print("✅ Test de backup de seguridad pasado")

def create_test_environment():
    """Crea un entorno de test completo con scripts"""
    
    test_env_dir = Path(tempfile.mkdtemp(prefix='restore_test_'))
    
    print(f"🏗️  Creando entorno de test en: {test_env_dir}")
    
    # Copiar el script de restauración al entorno de test
    script_content = '''#!/usr/bin/env python3
"""
Script simple para restaurar automáticamente desde la carpeta 'backup'
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def restore_from_backup():
    """Restaura automáticamente desde la carpeta 'backup'"""
    
    # Configuración
    repo_path = Path('.').resolve()
    backup_path = repo_path / 'backup'
    
    print("🔄 Restauración automática desde carpeta 'backup'")
    print(f"📂 Repositorio: {repo_path}")
    print(f"📦 Backup: {backup_path}")
    
    # Verificaciones
    if not backup_path.exists():
        print("❌ Error: No se encontró la carpeta 'backup'")
        return False
    
    if not (repo_path / '.git').exists():
        print("❌ Error: No es un repositorio Git")
        return False
    
    # Crear backup de seguridad automáticamente
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safety_backup = repo_path / f"safety_backup_{timestamp}"
    
    print(f"🔐 Creando backup de seguridad: {safety_backup.name}")
    
    try:
        # Copiar todo excepto .git y backup para seguridad
        shutil.copytree(repo_path, safety_backup, 
                       ignore=shutil.ignore_patterns('.git', 'backup', '__pycache__', '*.pyc'))
        
        # Elementos que siempre se preservan
        preserve = {'.git', '.gitignore', '.gitattributes', 'LICENSE', 'README.md', 'backup', safety_backup.name}
        
        print("🗑️  Eliminando archivos actuales...")
        
        # Eliminar todo excepto elementos preservados
        for item in repo_path.iterdir():
            if item.name not in preserve:
                print(f"   🗑️  {item.name}")
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        
        print("📋 Restaurando desde backup...")
        
        # Copiar contenido del backup
        for item in backup_path.iterdir():
            if item.name not in preserve:
                dest = repo_path / item.name
                print(f"   📄 {item.name}")
                
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        
        # Eliminar carpeta backup (ya no se necesita)
        print("🗑️  Eliminando carpeta backup...")
        shutil.rmtree(backup_path)
        
        print("✅ ¡Restauración completada exitosamente!")
        print(f"🔐 Backup de seguridad: {safety_backup.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la restauración: {e}")
        return False

if __name__ == "__main__":
    restore_from_backup()
'''
    
    (test_env_dir / 'simple_restore.py').write_text(script_content)
    
    # Crear repositorio simulado
    create_mock_repository(test_env_dir)
    
    return test_env_dir

def create_mock_repository(base_dir):
    """Crea un repositorio simulado completo"""
    
    # Crear .git
    git_dir = base_dir / '.git'
    git_dir.mkdir()
    (git_dir / 'config').write_text('[core]\n    repositoryformatversion = 0')
    
    # Crear archivos base
    (base_dir / 'README.md').write_text('# PtONN-TESTS\nTest repository')
    (base_dir / 'LICENSE').write_text('MIT License')
    (base_dir / '.gitignore').write_text('__pycache__/\n*.pyc\n.DS_Store')
    
    # Crear estructura "nueva" que debe eliminarse
    new_files = [
        'new_feature.py',
        'experimental_code.py',
        'temp_analysis.py'
    ]
    
    for file in new_files:
        (base_dir / file).write_text(f'# {file} - This should be removed')
    
    # Crear carpetas nuevas
    (base_dir / 'new_folder').mkdir()
    (base_dir / 'new_folder' / 'nested_file.py').write_text('# Nested file')
    
    # Crear carpeta backup con contenido "original"
    backup_dir = base_dir / 'backup'
    backup_dir.mkdir()
    
    # Contenido del backup
    backup_files = {
        'main.py': '# Main script from backup',
        'utils.py': '# Utilities from backup',
        'config.py': '# Configuration from backup'
    }
    
    for file, content in backup_files.items():
        (backup_dir / file).write_text(content)
    
    # Crear subcarpetas en backup
    (backup_dir / 'src').mkdir()
    (backup_dir / 'src' / 'core.py').write_text('# Core module from backup')
    
    (backup_dir / 'tests').mkdir()
    (backup_dir / 'tests' / 'test_main.py').write_text('# Tests from backup')
    
    print("📦 Repositorio simulado creado con:")
    print("   📁 Archivos a preservar: README.md, LICENSE, .gitignore")
    print("   📁 Archivos a eliminar: new_feature.py, experimental_code.py, temp_analysis.py")
    print("   📁 Carpetas a eliminar: new_folder/")
    print("   📁 Backup con: main.py, utils.py, config.py, src/, tests/")

def run_manual_test():
    """Ejecuta un test manual paso a paso"""
    
    print("🧪 EJECUTANDO TEST MANUAL")
    print("=" * 50)
    
    # Crear entorno de test
    test_env = create_test_environment()
    original_cwd = os.getcwd()
    
    try:
        os.chdir(test_env)
        
        print("\n📋 Estado ANTES de la restauración:")
        print("Archivos actuales:")
        for item in sorted(test_env.iterdir()):
            icon = "📁" if item.is_dir() else "📄"
            print(f"   {icon} {item.name}")
        
        print("\n📦 Contenido del backup:")
        backup_dir = test_env / 'backup'
        for item in sorted(backup_dir.rglob('*')):
            if item.is_file():
                rel_path = item.relative_to(backup_dir)
                print(f"   📄 backup/{rel_path}")
        
        # Ejecutar restauración
        print("\n🔄 EJECUTANDO RESTAURACIÓN...")
        print("-" * 30)
        
        result = subprocess.run([sys.executable, 'simple_restore.py'], 
                              capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"\nCódigo de salida: {result.returncode}")
        
        print("\n📋 Estado DESPUÉS de la restauración:")
        print("Archivos resultantes:")
        for item in sorted(test_env.iterdir()):
            icon = "📁" if item.is_dir() else "📄"
            print(f"   {icon} {item.name}")
        
        # Verificar contenido
        print("\n✅ VERIFICACIONES:")
        
        # Verificar que archivos del backup fueron restaurados
        expected_files = ['main.py', 'utils.py', 'config.py']
        for file in expected_files:
            if (test_env / file).exists():
                print(f"   ✅ {file} restaurado correctamente")
            else:
                print(f"   ❌ {file} NO encontrado")
        
        # Verificar que archivos nuevos fueron eliminados
        removed_files = ['new_feature.py', 'experimental_code.py', 'temp_analysis.py']
        for file in removed_files:
            if not (test_env / file).exists():
                print(f"   ✅ {file} eliminado correctamente")
            else:
                print(f"   ❌ {file} NO fue eliminado")
        
        # Verificar que archivos críticos se preservaron
        critical_files = ['README.md', 'LICENSE', '.gitignore', '.git']
        for file in critical_files:
            if (test_env / file).exists():
                print(f"   ✅ {file} preservado correctamente")
            else:
                print(f"   ❌ {file} NO preservado")
        
        # Verificar backup de seguridad
        safety_backups = [d for d in test_env.iterdir() 
                         if d.is_dir() and d.name.startswith('safety_backup_')]
        
        if safety_backups:
            print(f"   ✅ Backup de seguridad creado: {safety_backups[0].name}")
        else:
            print("   ❌ NO se creó backup de seguridad")
        
        print(f"\n🎉 Test completado. Entorno de test en: {test_env}")
        print("💡 Puedes explorar manualmente el resultado")
        
    finally:
        os.chdir(original_cwd)

def main():
    """Función principal para ejecutar tests"""
    
    print("🧪 SUITE DE TESTS - SCRIPT DE RESTAURACIÓN")
    print("=" * 50)
    
    print("Selecciona el tipo de test:")
    print("1. Test automático (unittest)")
    print("2. Test manual paso a paso")
    print("3. Ambos")
    
    choice = input("Opción (1-3): ").strip()
    
    if choice in ['1', '3']:
        print("\n🤖 EJECUTANDO TESTS AUTOMÁTICOS")
        print("-" * 30)
        
        # Ejecutar tests unitarios
        unittest.main(argv=[''], exit=False, verbosity=2)
    
    if choice in ['2', '3']:
        print("\n👨‍💻 EJECUTANDO TEST MANUAL")
        print("-" * 30)
        
        run_manual_test()
    
    print("\n🏁 Tests completados")

if __name__ == "__main__":
    main()