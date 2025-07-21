#!/usr/bin/env python3
"""
Script simple para restaurar automáticamente desde la carpeta 'backup'
Uso: python simple_restore.py
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
        print("💡 Puedes eliminar el backup de seguridad si todo funciona bien")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la restauración: {e}")
        
        # Intentar restaurar desde backup de seguridad
        if safety_backup.exists():
            print("🔄 Restaurando desde backup de seguridad...")
            try:
                for item in safety_backup.iterdir():
                    dest = repo_path / item.name
                    if dest.exists() and item.name not in {'.git'}:
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
                
                print("✅ Restauración desde backup de seguridad completada")
                
            except Exception as restore_error:
                print(f"❌ Error restaurando backup de seguridad: {restore_error}")
        
        return False

def main():
    print("🚀 Restaurador Automático PtONN-TESTS")
    print("=" * 50)
    
    # Verificar que estamos en el lugar correcto
    current_dir = Path('.').resolve()
    if not any(item.name == 'backup' for item in current_dir.iterdir() if item.is_dir()):
        print("❌ No se encontró la carpeta 'backup' en el directorio actual")
        print(f"📂 Directorio actual: {current_dir}")
        print("💡 Ejecuta este script desde la raíz del repositorio")
        return
    
    # Mostrar contenido del backup
    backup_path = current_dir / 'backup'
    print("📋 Contenido del backup:")
    for item in backup_path.iterdir():
        icon = "📁" if item.is_dir() else "📄"
        print(f"   {icon} {item.name}")
    
    print("\n⚠️  ADVERTENCIA: Esto eliminará todos los archivos actuales")
    print("⚠️  y restaurará solo lo que está en 'backup'")
    print("🔐 Se creará un backup de seguridad automáticamente")
    
    # Ejecutar restauración
    print("\n🔄 Iniciando restauración en 3 segundos...")
    import time
    time.sleep(3)
    
    if restore_from_backup():
        print("\n🎉 ¡Listo! Tu repositorio ha sido restaurado al backup")
        print("🔧 Recuerda hacer un commit si estás satisfecho")
    else:
        print("\n❌ La restauración falló")

if __name__ == "__main__":
    main()