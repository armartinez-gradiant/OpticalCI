#!/usr/bin/env python3
"""
Fix All - Reparación Automática Completa para PtONN-TESTS
=========================================================

Este script hace TODO automáticamente para solucionar el problema
"_C is not defined" de PyTorch y dejar PtONN-TESTS completamente funcional.

¡Solo ejecuta este script y listo!
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
import importlib.util

def run_command(cmd, description="", ignore_errors=False):
    """Ejecutar comando con manejo de errores"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Éxito")
            return True
        else:
            print(f"   ⚠️  Salida: {result.stderr.strip()}")
            if not ignore_errors:
                print(f"   ❌ Error, pero continuando...")
            return False
    except Exception as e:
        print(f"   ❌ Excepción: {e}")
        return False

def clean_python_files():
    """Limpiar archivos Python compilados"""
    print("\n🧹 LIMPIANDO ARCHIVOS PYTHON")
    print("=" * 50)
    
    current_dir = Path.cwd()
    
    # Limpiar __pycache__
    pycache_dirs = list(current_dir.rglob("__pycache__"))
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"   ✅ Eliminado: {pycache_dir.name}")
        except Exception as e:
            print(f"   ⚠️  Error eliminando {pycache_dir}: {e}")
    
    # Limpiar .pyc
    pyc_files = list(current_dir.rglob("*.pyc"))
    for pyc_file in pyc_files:
        try:
            pyc_file.unlink()
            print(f"   ✅ Eliminado: {pyc_file.name}")
        except Exception as e:
            print(f"   ⚠️  Error eliminando {pyc_file}: {e}")
    
    # Limpiar imports en memoria
    modules_to_remove = []
    for module_name in sys.modules:
        if 'torch' in module_name or 'torchonn' in module_name:
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        try:
            del sys.modules[module_name]
            print(f"   ✅ Eliminado de memoria: {module_name}")
        except:
            pass
    
    print("   🎉 Limpieza completada")

def fix_pytorch():
    """Reparar PyTorch completamente"""
    print("\n🔥 REPARANDO PYTORCH")
    print("=" * 50)
    
    # 1. Limpiar cache
    run_command("pip cache purge", "Limpiando cache pip", ignore_errors=True)
    
    # 2. Actualizar pip
    run_command("pip install --upgrade pip", "Actualizando pip", ignore_errors=True)
    
    # 3. Desinstalar PyTorch
    print("\n🗑️  Desinstalando PyTorch actual...")
    run_command("pip uninstall torch -y", "Desinstalando torch", ignore_errors=True)
    run_command("pip uninstall torchvision -y", "Desinstalando torchvision", ignore_errors=True)
    run_command("pip uninstall torchaudio -y", "Desinstalando torchaudio", ignore_errors=True)
    run_command("pip uninstall pytorch -y", "Desinstalando pytorch", ignore_errors=True)
    
    # 4. Instalar NumPy compatible
    print("\n🔢 Instalando NumPy compatible...")
    run_command("pip install 'numpy>=1.19.0,<2.0.0'", "Instalando NumPy < 2.0")
    
    # 5. Reinstalar PyTorch
    print("\n🔥 Reinstalando PyTorch...")
    
    # Detectar plataforma
    if platform.system() == "Linux":
        pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    elif platform.system() == "Windows":
        pytorch_cmd = "pip install torch torchvision torchaudio"
    elif platform.system() == "Darwin":  # macOS
        pytorch_cmd = "pip install torch torchvision torchaudio"
    else:
        pytorch_cmd = "pip install torch torchvision torchaudio"
    
    success = run_command(pytorch_cmd, "Instalando PyTorch")
    
    if not success:
        print("   ⚠️  Instalación estándar falló, intentando alternativa...")
        run_command("pip install torch torchvision torchaudio --no-cache-dir", "Instalación alternativa")
    
    # 6. Instalar dependencias adicionales
    print("\n📦 Instalando dependencias adicionales...")
    run_command("pip install scipy matplotlib pyyaml tqdm", "Instalando dependencias", ignore_errors=True)

def install_ptonn():
    """Instalar PtONN-TESTS"""
    print("\n📦 INSTALANDO PtONN-TESTS")
    print("=" * 50)
    
    # Instalar en modo desarrollo
    run_command("pip install -e .", "Instalando PtONN-TESTS en modo desarrollo")

def verify_installation():
    """Verificar que todo funciona"""
    print("\n✅ VERIFICANDO INSTALACIÓN")
    print("=" * 50)
    
    # Test PyTorch
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        
        # Test tensor
        x = torch.randn(2, 3)
        print(f"   ✅ Tensor creation: {x.shape}")
        
        # Test _C module
        if hasattr(torch, '_C'):
            print(f"   ✅ torch._C: Disponible")
        else:
            print(f"   ⚠️  torch._C: No disponible (pero puede funcionar)")
        
    except Exception as e:
        print(f"   ❌ PyTorch error: {e}")
        return False
    
    # Test TorchONN
    try:
        import torchonn
        print(f"   ✅ TorchONN: {torchonn.__version__}")
        
        from torchonn.layers import MZILayer, MZIBlockLinear
        print(f"   ✅ Layers: Importados correctamente")
        
        from torchonn.models import ONNBaseModel
        print(f"   ✅ Models: Importados correctamente")
        
        # Test funcionalidad
        layer = MZILayer(4, 3)
        x = torch.randn(2, 4)
        output = layer(x)
        print(f"   ✅ Funcionalidad: {x.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"   ❌ TorchONN error: {e}")
        return False
    
    # Test gradientes
    try:
        x = torch.randn(2, 4, requires_grad=True)
        layer = MZILayer(4, 3)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        print(f"   ✅ Gradientes: Funcionan correctamente")
        
    except Exception as e:
        print(f"   ❌ Gradientes error: {e}")
        return False
    
    return True

def create_test_files():
    """Crear archivos de test seguros"""
    print("\n📝 CREANDO ARCHIVOS DE TEST")
    print("=" * 50)
    
    # Crear test simple
    test_simple = '''#!/usr/bin/env python3
"""Test simple para verificar que todo funciona"""
import torch
import torchonn
from torchonn.layers import MZILayer

print("🧪 Test Simple PtONN-TESTS")
print("=" * 40)

# Test PyTorch
x = torch.randn(2, 4)
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ Tensor: {x.shape}")

# Test TorchONN
layer = MZILayer(4, 3)
output = layer(x)
print(f"✅ TorchONN: {torchonn.__version__}")
print(f"✅ MZI Layer: {x.shape} -> {output.shape}")

# Test gradientes
x.requires_grad_(True)
output = layer(x)
loss = output.sum()
loss.backward()
print(f"✅ Gradientes: OK")

print("\\n🎉 ¡Todo funciona correctamente!")
'''
    
    with open("test_simple_final.py", "w") as f:
        f.write(test_simple)
    
    print("   ✅ test_simple_final.py creado")
    
    # Hacer ejecutable
    os.chmod("test_simple_final.py", 0o755)

def main():
    """Función principal - hace todo automáticamente"""
    print("🚀 FIX ALL - REPARACIÓN AUTOMÁTICA COMPLETA")
    print("=" * 70)
    print("\nEste script solucionará AUTOMÁTICAMENTE el problema de PyTorch")
    print("y dejará PtONN-TESTS completamente funcional.")
    print("\n¡Solo relájate y espera! 🍿")
    
    # Información del sistema
    print(f"\n🖥️  Sistema: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Directorio: {os.getcwd()}")
    
    try:
        # Paso 1: Limpieza
        clean_python_files()
        
        # Paso 2: Reparar PyTorch
        fix_pytorch()
        
        # Paso 3: Instalar PtONN-TESTS
        install_ptonn()
        
        # Paso 4: Verificar
        if verify_installation():
            print("\n🎉 ¡REPARACIÓN COMPLETADA CON ÉXITO!")
            
            # Crear archivos de test
            create_test_files()
            
            print("\n" + "=" * 70)
            print("✨ PtONN-TESTS ESTÁ COMPLETAMENTE FUNCIONAL ✨")
            print("=" * 70)
            
            print("\n📝 Para verificar, ejecuta:")
            print("   python test_simple_final.py")
            print("   python test_installation.py")
            print("   pytest tests/ -v")
            
            print("\n🚀 Para empezar a usar:")
            print("   python examples/basic_usage.py")
            print("   python examples/advanced_usage.py")
            print("   python explore.py")
            
            print("\n🎯 ¡Ya puedes construir redes neuronales ópticas!")
            
        else:
            print("\n⚠️  ALGUNOS PROBLEMAS PERSISTEN")
            print("\n🔧 Intenta estos pasos manuales:")
            print("1. Verificar entorno virtual activo")
            print("2. Ejecutar: pip install torch torchvision torchaudio --force-reinstall")
            print("3. Ejecutar: pip install -e .")
            print("4. Ejecutar: python test_simple_final.py")
            
    except KeyboardInterrupt:
        print("\n👋 Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 Solución manual:")
        print("1. pip uninstall torch torchvision torchaudio -y")
        print("2. pip install 'numpy>=1.19.0,<2.0.0'")
        print("3. pip install torch torchvision torchaudio")
        print("4. pip install -e .")

if __name__ == "__main__":
    main()