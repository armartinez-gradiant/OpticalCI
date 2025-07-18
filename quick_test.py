#!/usr/bin/env python3
"""
Diagnóstico Simple PtONN-TESTS
==============================

Script simple para diagnosticar problemas básicos.
"""

import sys
import os
from pathlib import Path
import subprocess

def print_header(title):
    print(f"\n{'='*20} {title} {'='*20}")

def print_step(step_num, title):
    print(f"\n{step_num}️⃣ {title}")
    print("-" * 40)

def run_basic_checks():
    """Verificaciones básicas del sistema"""
    print_header("DIAGNÓSTICO BÁSICO PtONN-TESTS")
    
    # 1. Verificar entorno
    print_step(1, "VERIFICANDO ENTORNO")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in current dir: {list(Path('.').glob('*'))[:10]}")
    
    # 2. Verificar estructura
    print_step(2, "VERIFICANDO ESTRUCTURA")
    key_paths = [
        "torchonn",
        "torchonn/__init__.py", 
        "torchonn/layers",
        "torchonn/layers/__init__.py",
        "torchonn/layers/mzi_layer.py",
        "examples",
        "tests"
    ]
    
    for path in key_paths:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - MISSING")
    
    # 3. Test de import básico
    print_step(3, "TEST DE IMPORT BÁSICO")
    
    # Add current directory to Python path
    current_dir = str(Path.cwd())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Test imports uno por uno
    modules_to_test = [
        "torch",
        "numpy", 
        "torchonn",
        "torchonn.layers",
        "torchonn.models"
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}: OK")
        except ImportError as e:
            print(f"❌ {module}: ImportError - {str(e)[:100]}")
        except Exception as e:
            print(f"⚠️ {module}: Error - {str(e)[:100]}")
    
    # 4. Test específico de clases
    print_step(4, "TEST DE CLASES ESPECÍFICAS")
    
    try:
        from torchonn.layers import MZILayer
        print("✅ MZILayer import: OK")
        
        layer = MZILayer(4, 2)
        print("✅ MZILayer creation: OK")
        
    except Exception as e:
        print(f"❌ MZILayer test failed: {e}")
    
    try:
        from torchonn.layers import MZIBlockLinear
        print("✅ MZIBlockLinear import: OK")
        
        block = MZIBlockLinear(4, 2, mode="usv")
        print("✅ MZIBlockLinear creation: OK")
        
    except Exception as e:
        print(f"❌ MZIBlockLinear test failed: {e}")
    
    # 5. Test forward pass
    print_step(5, "TEST FORWARD PASS")
    
    try:
        import torch
        from torchonn.layers import MZILayer
        
        x = torch.randn(2, 4)
        layer = MZILayer(4, 2)
        output = layer(x)
        
        print(f"✅ Forward pass: {x.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")

def check_specific_files():
    """Verificar archivos específicos que causan problemas"""
    print_step(6, "VERIFICANDO ARCHIVOS PROBLEMÁTICOS")
    
    problematic_files = [
        "torchonn/components/add_drop_mrr.py",
        "torchonn/components/microring_resonator.py", 
        "torchonn/layers/mrr_weight_bank.py",
        "torchonn/systems/wdm_system.py"
    ]
    
    for file_path in problematic_files:
        if Path(file_path).exists():
            try:
                content = Path(file_path).read_text()
                if "TODO: Implementar" in content:
                    print(f"⚠️ {file_path}: STUB FILE (TODO)")
                elif "MicroringResonator" in content and "from .microring_resonator import" not in content:
                    print(f"❌ {file_path}: Missing MicroringResonator import")
                else:
                    print(f"✅ {file_path}: Looks OK")
            except Exception as e:
                print(f"❌ {file_path}: Read error - {e}")
        else:
            print(f"❌ {file_path}: NOT FOUND")

def check_installation():
    """Verificar si el paquete está instalado"""
    print_step(7, "VERIFICANDO INSTALACIÓN")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        
        if "torchonn" in result.stdout or "ptonn" in result.stdout:
            print("✅ Package appears to be installed")
            print("Installed packages containing 'torch' or 'onn':")
            for line in result.stdout.split('\n'):
                if 'torch' in line.lower() or 'onn' in line.lower():
                    print(f"  {line}")
        else:
            print("❌ Package not installed via pip")
            print("💡 Try: pip install -e .")
            
    except Exception as e:
        print(f"❌ Error checking installation: {e}")

def show_file_contents():
    """Mostrar contenido de archivos clave para debug"""
    print_step(8, "CONTENIDO DE ARCHIVOS CLAVE")
    
    key_files = [
        "torchonn/__init__.py",
        "torchonn/layers/__init__.py"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"\n--- {file_path} ---")
            try:
                content = Path(file_path).read_text()
                lines = content.split('\n')
                # Show first 15 lines
                for i, line in enumerate(lines[:15], 1):
                    print(f"{i:2d}: {line}")
                if len(lines) > 15:
                    print(f"... ({len(lines) - 15} more lines)")
            except Exception as e:
                print(f"Error reading file: {e}")

def main():
    """Función principal"""
    try:
        run_basic_checks()
        check_specific_files()
        check_installation()
        show_file_contents()
        
        print_header("DIAGNÓSTICO COMPLETADO")
        print("\n💡 PRÓXIMOS PASOS:")
        print("1. Si hay archivos STUB (TODO), necesitan implementación")
        print("2. Si hay imports faltantes, usar el script de corrección")
        print("3. Si el paquete no está instalado: pip install -e .")
        print("4. Si hay errores de sintaxis, revisar archivos específicos")
        
    except KeyboardInterrupt:
        print("\n\nDiagnóstico interrumpido por el usuario")
    except Exception as e:
        print(f"\nError durante diagnóstico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
