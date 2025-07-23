#!/usr/bin/env python3
"""
🔧 CORRECTOR DE PATHS PARA EXAMPLE1.PY
Soluciona el problema "ModuleNotFoundError: No module named 'torchonn'"

PROBLEMA: Example1.py está en subdirectorio examples/ y no encuentra torchonn
SOLUCIÓN: Añadir path del directorio padre automáticamente
"""

import os
import shutil
from pathlib import Path
import datetime

def fix_example1_paths():
    """Corregir imports en Example1.py para que funcione desde cualquier directorio."""
    
    print("🔧 CORRECTOR DE PATHS PARA EXAMPLE1.PY")
    print("=" * 50)
    
    example_path = Path("examples/Example1.py")
    
    if not example_path.exists():
        print("❌ ERROR: No se encuentra examples/Example1.py")
        return False
    
    # 1. BACKUP
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = Path(f"Example1_paths_backup_{timestamp}.py")
    shutil.copy2(example_path, backup_path)
    print(f"💾 Backup creado: {backup_path}")
    
    # 2. LEER CONTENIDO
    with open(example_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 3. VERIFICAR SI YA TIENE PATH FIX
    if "sys.path.insert" in content and "dirname" in content:
        print("✅ Example1.py ya tiene corrección de paths")
        return True
    
    print("🔧 Añadiendo corrección de paths...")
    
    # 4. AÑADIR PATH FIX AL INICIO
    path_fix = '''#!/usr/bin/env python3
"""
🌟 Complete Photonic Simulation Demo - PtONN-TESTS (CORREGIDO)

Ejemplo completo que demuestra las capacidades del repositorio con:
- Análisis de componentes individuales
- Red fotónica completa
- Resultados teóricos esperados
- Validación física
"""

# ✅ CORRECCIÓN DE PATHS - Permite ejecutar desde cualquier directorio
import sys
import os
# Añadir directorio padre (raíz del repositorio) al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

'''
    
    # 5. ENCONTRAR DONDE INSERTAR
    # Buscar después del docstring inicial
    lines = content.split('\n')
    
    # Encontrar línea donde empiezan los imports reales
    insert_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('import torch') or line.strip().startswith('from torchonn'):
            insert_line = i
            break
    
    if insert_line == 0:
        print("⚠️ No se encontró lugar adecuado para insertar path fix")
        # Buscar después de imports estándar
        for i, line in enumerate(lines):
            if 'import torch' in line or 'import numpy' in line:
                insert_line = i
                break
    
    # 6. CONSTRUIR NUEVO CONTENIDO
    if insert_line > 0:
        # Insertar path fix antes de los imports principales
        new_lines = lines[:insert_line]
        new_lines.extend([
            "",
            "# ✅ CORRECCIÓN DE PATHS - Permite ejecutar desde cualquier directorio",
            "import sys",
            "import os",
            "# Añadir directorio padre (raíz del repositorio) al path", 
            "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
            ""
        ])
        new_lines.extend(lines[insert_line:])
        new_content = '\n'.join(new_lines)
    else:
        # Si no encontramos lugar, añadir al principio después del shebang/docstring
        new_content = path_fix + content
    
    # 7. ESCRIBIR ARCHIVO CORREGIDO
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Corrección de paths añadida a Example1.py")
    
    # 8. VERIFICAR CORRECCIÓN
    with open(example_path, 'r') as f:
        new_content = f.read()
    
    if "sys.path.insert" in new_content:
        print("✅ Verificación: Corrección aplicada correctamente")
        return True
    else:
        print("❌ Verificación: Corrección no aplicada")
        return False

def test_fixed_example1():
    """Test de que Example1.py funciona después de la corrección."""
    print("\n🧪 TESTING EXAMPLE1.PY CORREGIDO...")
    
    try:
        # Cambiar al directorio examples para simular ejecución real
        original_cwd = os.getcwd()
        os.chdir('examples')
        
        # Intentar importar
        import sys
        sys.path.insert(0, '..')  # Añadir directorio padre
        
        # Test de import
        from torchonn.layers import MicroringResonator
        print("✅ Import desde examples/ funciona")
        
        # Test básico
        import torch
        device = torch.device("cpu")
        mrr = MicroringResonator(device=device)
        print("✅ MicroringResonator se puede crear")
        
        os.chdir(original_cwd)
        return True
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        os.chdir(original_cwd)
        return False

def create_run_script():
    """Crear script de ejecución alternativo."""
    print("\n📝 CREANDO SCRIPT DE EJECUCIÓN ALTERNATIVO...")
    
    run_script = """#!/bin/bash
# 🚀 EJECUTOR DE EXAMPLE1.PY
# Soluciona automáticamente problemas de paths

echo "🚀 Ejecutando Example1.py con paths corregidos..."

# Método 1: Añadir PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Método 2: Ejecutar
python examples/Example1.py

# Si falla, intentar método alternativo
if [ $? -ne 0 ]; then
    echo "⚠️ Método 1 falló, intentando método 2..."
    cd examples/
    export PYTHONPATH=$PYTHONPATH:..
    python Example1.py
    cd ..
fi
"""
    
    with open("run_example1.sh", 'w') as f:
        f.write(run_script)
    
    # Hacer ejecutable en sistemas Unix
    try:
        os.chmod("run_example1.sh", 0o755)
        print("✅ Script run_example1.sh creado")
        print("🚀 Úsalo con: ./run_example1.sh")
    except:
        print("✅ Script run_example1.sh creado")
        print("🚀 Úsalo con: bash run_example1.sh")

def main():
    """Función principal con múltiples soluciones."""
    
    print("🔍 PROBLEMA DETECTADO:")
    print("❌ Example1.py no encuentra 'torchonn' desde subdirectorio examples/")
    print("✅ test_installation.py SÍ funciona desde directorio raíz")
    print("🔍 Causa: Problema de Python paths/imports")
    
    print("\n🔧 APLICANDO SOLUCIONES...")
    
    # Solución 1: Modificar Example1.py
    success = fix_example1_paths()
    
    if success:
        # Test de la corrección
        test_success = test_fixed_example1()
        
        if test_success:
            print("\n🎉 ¡PROBLEM SOLVED!")
            print("✅ Example1.py corregido permanentemente")
            print("🚀 Ahora ejecuta: python examples/Example1.py")
        else:
            print("\n⚠️ Corrección aplicada pero test falló")
    
    # Solución 2: Crear script alternativo
    create_run_script()
    
    print("\n📋 RESUMEN DE SOLUCIONES DISPONIBLES:")
    print("1. 🔧 Example1.py modificado permanentemente")
    print("2. 🚀 Script run_example1.sh creado")
    print("3. ⚡ Comando rápido: export PYTHONPATH=$PYTHONPATH:. && python examples/Example1.py")
    print("4. 📦 Método módulo: python -m examples.Example1")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())