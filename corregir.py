#!/usr/bin/env python3
"""
Script de Actualización de Dependencias
=======================================

Actualiza todas las dependencias del proyecto para ser compatibles
con las versiones más recientes de PyTorch.

Basado en la matriz de compatibilidad oficial de PyTorch:
https://github.com/pytorch/pytorch/wiki/PyTorch-Versions
"""

import os
import sys
from pathlib import Path

def print_step(step_num, description):
    """Print formatted step"""
    print(f"\n{step_num}️⃣ {description}")
    print("-" * 50)

def update_requirements_txt():
    """Actualizar requirements.txt"""
    print_step(1, "ACTUALIZANDO requirements.txt")
    
    requirements_file = Path("requirements.txt")
    
    # Dependencias actualizadas con versiones compatibles
    new_requirements = """torch>=2.0.0,<2.8.0
torchvision>=0.15.0,<0.23.0
torchaudio>=2.0.0,<2.8.0
numpy>=1.19.0,<2.0.0
scipy>=1.7.0,<1.15.0
matplotlib>=3.3.0,<4.0.0
pyyaml>=5.4.0,<7.0.0
tqdm>=4.60.0,<5.0.0
pytest>=6.0.0,<8.0.0
pytest-cov>=2.0.0,<5.0.0
black>=21.0.0,<25.0.0
flake8>=3.8.0,<7.0.0
"""
    
    try:
        with open(requirements_file, 'w') as f:
            f.write(new_requirements)
        print(f"✅ {requirements_file} actualizado")
        print("   📋 Cambios principales:")
        print("      • torchvision: <0.20.0 → <0.23.0")
        print("      • scipy: <1.13.0 → <1.15.0")
        return True
    except Exception as e:
        print(f"❌ Error actualizando {requirements_file}: {e}")
        return False

def update_setup_py():
    """Actualizar setup.py"""
    print_step(2, "ACTUALIZANDO setup.py")
    
    setup_file = Path("setup.py")
    
    if not setup_file.exists():
        print(f"⚠️  {setup_file} no existe")
        return True
    
    try:
        with open(setup_file, 'r') as f:
            content = f.read()
        
        # Actualizaciones específicas
        replacements = [
            ('torchvision>=0.15.0,<0.20.0', 'torchvision>=0.15.0,<0.23.0'),
            ('scipy>=1.7.0,<1.13.0', 'scipy>=1.7.0,<1.15.0'),
            ('numpy>=1.19.0,<2.0.0', 'numpy>=1.19.0,<2.0.0'),  # Mantener
            ('torch>=2.0.0,<2.8.0', 'torch>=2.0.0,<2.8.0'),    # Mantener
            ('torchaudio>=2.0.0,<2.8.0', 'torchaudio>=2.0.0,<2.8.0'),  # Mantener
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        with open(setup_file, 'w') as f:
            f.write(content)
        
        print(f"✅ {setup_file} actualizado")
        return True
    except Exception as e:
        print(f"❌ Error actualizando {setup_file}: {e}")
        return False

def update_pyproject_toml():
    """Actualizar pyproject.toml"""
    print_step(3, "ACTUALIZANDO pyproject.toml")
    
    pyproject_file = Path("pyproject.toml")
    
    if not pyproject_file.exists():
        print(f"⚠️  {pyproject_file} no existe")
        return True
    
    try:
        with open(pyproject_file, 'r') as f:
            content = f.read()
        
        # Actualizaciones específicas
        replacements = [
            ('torchvision>=0.15.0,<0.20.0', 'torchvision>=0.15.0,<0.23.0'),
            ('scipy>=1.7.0,<1.13.0', 'scipy>=1.7.0,<1.15.0'),
            ('matplotlib>=3.3.0,<4.0.0', 'matplotlib>=3.3.0,<4.0.0'),
            ('pyyaml>=5.4.0,<7.0.0', 'pyyaml>=5.4.0,<7.0.0'),
            ('tqdm>=4.60.0,<5.0.0', 'tqdm>=4.60.0,<5.0.0'),
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        with open(pyproject_file, 'w') as f:
            f.write(content)
        
        print(f"✅ {pyproject_file} actualizado")
        return True
    except Exception as e:
        print(f"❌ Error actualizando {pyproject_file}: {e}")
        return False

def update_pyproyect_toml():
    """Actualizar pyproyect.toml (archivo duplicado)"""
    print_step(4, "ACTUALIZANDO pyproyect.toml")
    
    pyproyect_file = Path("pyproyect.toml")
    
    if not pyproyect_file.exists():
        print(f"⚠️  {pyproyect_file} no existe")
        return True
    
    try:
        with open(pyproyect_file, 'r') as f:
            content = f.read()
        
        # Actualizaciones específicas
        replacements = [
            ('torchvision>=0.15.0,<0.20.0', 'torchvision>=0.15.0,<0.23.0'),
            ('scipy>=1.7.0,<1.13.0', 'scipy>=1.7.0,<1.15.0'),
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        with open(pyproyect_file, 'w') as f:
            f.write(content)
        
        print(f"✅ {pyproyect_file} actualizado")
        return True
    except Exception as e:
        print(f"❌ Error actualizando {pyproyect_file}: {e}")
        return False

def show_compatibility_matrix():
    """Mostrar matriz de compatibilidad"""
    print_step(5, "MATRIZ DE COMPATIBILIDAD PYTORCH")
    
    matrix = [
        ("PyTorch 2.7.x", "torchvision 0.22.x", "torchaudio 2.7.x"),
        ("PyTorch 2.6.x", "torchvision 0.21.x", "torchaudio 2.6.x"),
        ("PyTorch 2.5.x", "torchvision 0.20.x", "torchaudio 2.5.x"),
        ("PyTorch 2.4.x", "torchvision 0.19.x", "torchaudio 2.4.x"),
        ("PyTorch 2.3.x", "torchvision 0.18.x", "torchaudio 2.3.x"),
        ("PyTorch 2.2.x", "torchvision 0.17.x", "torchaudio 2.2.x"),
        ("PyTorch 2.1.x", "torchvision 0.16.x", "torchaudio 2.1.x"),
        ("PyTorch 2.0.x", "torchvision 0.15.x", "torchaudio 2.0.x"),
    ]
    
    print("📋 Matriz de compatibilidad oficial:")
    print("   PyTorch        | torchvision    | torchaudio")
    print("   ---------------|----------------|----------------")
    for pytorch, torchvision, torchaudio in matrix:
        print(f"   {pytorch:<14} | {torchvision:<14} | {torchaudio}")
    
    print("\n🔍 Problema detectado:")
    print("   • Usuario tiene: PyTorch 2.7.1 + torchvision 0.22.1")
    print("   • Proyecto requería: torchvision<0.20.0")
    print("   • Solución: Actualizar límites a torchvision<0.23.0")

def create_install_script():
    """Crear script de instalación compatible"""
    print_step(6, "CREANDO SCRIPT DE INSTALACIÓN")
    
    install_script = """#!/usr/bin/env python3
'''
Script de Instalación Compatible PyTorch
========================================

Instala PyTorch con versiones compatibles automáticamente.
'''

import subprocess
import sys

def install_pytorch_compatible():
    \"\"\"Instalar PyTorch con versiones compatibles\"\"\"
    print("🔥 Instalando PyTorch con versiones compatibles...")
    
    # Para CPU (más estable)
    cmd_cpu = [
        sys.executable, "-m", "pip", "install",
        "torch>=2.0.0,<2.8.0",
        "torchvision>=0.15.0,<0.23.0", 
        "torchaudio>=2.0.0,<2.8.0",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ]
    
    try:
        subprocess.run(cmd_cpu, check=True)
        print("✅ PyTorch instalado exitosamente")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error instalando PyTorch")
        return False

def install_dependencies():
    \"\"\"Instalar dependencias del proyecto\"\"\"
    print("📦 Instalando dependencias del proyecto...")
    
    cmd_deps = [
        sys.executable, "-m", "pip", "install",
        "numpy>=1.19.0,<2.0.0",
        "scipy>=1.7.0,<1.15.0",
        "matplotlib>=3.3.0,<4.0.0",
        "pyyaml>=5.4.0,<7.0.0",
        "tqdm>=4.60.0,<5.0.0"
    ]
    
    try:
        subprocess.run(cmd_deps, check=True)
        print("✅ Dependencias instaladas")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error instalando dependencias")
        return False

def install_project():
    \"\"\"Instalar proyecto en modo desarrollo\"\"\"
    print("🔧 Instalando proyecto en modo desarrollo...")
    
    cmd_proj = [sys.executable, "-m", "pip", "install", "-e", "."]
    
    try:
        subprocess.run(cmd_proj, check=True)
        print("✅ Proyecto instalado")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error instalando proyecto")
        return False

if __name__ == "__main__":
    print("🚀 Instalación Compatible PyTorch + PtONN-TESTS")
    print("=" * 60)
    
    steps = [
        ("PyTorch compatible", install_pytorch_compatible),
        ("Dependencias", install_dependencies),
        ("Proyecto", install_project)
    ]
    
    for step_name, step_func in steps:
        print(f"\\n⏳ {step_name}...")
        if not step_func():
            print(f"❌ Error en: {step_name}")
            sys.exit(1)
    
    print("\\n🎉 ¡Instalación completada!")
    print("\\nPuedes probar con:")
    print("   python -c 'import torch; print(torch.__version__)'")
    print("   python quick_test.py")
"""
    
    try:
        with open("install_compatible.py", 'w') as f:
            f.write(install_script)
        
        # Hacer ejecutable
        os.chmod("install_compatible.py", 0o755)
        
        print("✅ Script creado: install_compatible.py")
        return True
    except Exception as e:
        print(f"❌ Error creando script: {e}")
        return False

def verify_current_versions():
    """Verificar versiones actuales"""
    print_step(7, "VERIFICANDO VERSIONES ACTUALES")
    
    try:
        import torch
        import torchvision
        import torchaudio
        
        print(f"📋 Versiones instaladas:")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   torchvision: {torchvision.__version__}")
        print(f"   torchaudio: {torchaudio.__version__}")
        
        # Verificar compatibilidad
        torch_version = torch.__version__.split('+')[0]  # Remove +cpu suffix
        torchvision_version = torchvision.__version__.split('+')[0]
        
        # Matriz de compatibilidad simplificada
        compatible_pairs = {
            "2.7": "0.22",
            "2.6": "0.21", 
            "2.5": "0.20",
            "2.4": "0.19",
            "2.3": "0.18",
            "2.2": "0.17",
            "2.1": "0.16",
            "2.0": "0.15"
        }
        
        torch_major_minor = '.'.join(torch_version.split('.')[:2])
        torchvision_major_minor = '.'.join(torchvision_version.split('.')[:2])
        
        expected_torchvision = compatible_pairs.get(torch_major_minor, "unknown")
        
        if expected_torchvision != "unknown" and torchvision_major_minor == expected_torchvision:
            print("✅ Versiones compatibles")
        else:
            print("⚠️  Versiones podrían ser incompatibles")
            print(f"   PyTorch {torch_major_minor} esperaba torchvision {expected_torchvision}")
            print(f"   Pero tienes torchvision {torchvision_major_minor}")
        
        return True
    except ImportError as e:
        print(f"❌ Error importando: {e}")
        return False
    except Exception as e:
        print(f"❌ Error verificando: {e}")
        return False

def main():
    """Función principal"""
    print("🔧 ACTUALIZACIÓN DE DEPENDENCIAS PYTORCH")
    print("=" * 60)
    print("Solucionando conflictos de dependencias...")
    print("=" * 60)
    
    # Verificar versiones actuales
    verify_current_versions()
    
    # Actualizar archivos
    steps = [
        ("Actualizar requirements.txt", update_requirements_txt),
        ("Actualizar setup.py", update_setup_py),
        ("Actualizar pyproject.toml", update_pyproject_toml),
        ("Actualizar pyproyect.toml", update_pyproyect_toml),
        ("Mostrar matriz compatibilidad", show_compatibility_matrix),
        ("Crear script instalación", create_install_script),
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        try:
            if step_func():
                success_count += 1
                print(f"✅ {step_name}: ÉXITO")
            else:
                print(f"❌ {step_name}: FALLÓ")
        except Exception as e:
            print(f"❌ {step_name}: ERROR - {e}")
    
    # Resultados
    print(f"\n{'='*60}")
    print("📊 RESULTADOS")
    print(f"{'='*60}")
    print(f"✅ Actualizaciones completadas: {success_count}/{len(steps)}")
    
    if success_count >= 4:  # Al menos archivos principales actualizados
        print("\n🎉 ¡DEPENDENCIAS ACTUALIZADAS EXITOSAMENTE!")
        print("\n🚀 Próximos pasos:")
        print("   1. pip install -r requirements.txt")
        print("   2. pip install -e .")
        print("   3. python quick_test.py")
        print("\n💡 O usa el script automático:")
        print("   python install_compatible.py")
        
        return True
    else:
        print("\n⚠️  ACTUALIZACIONES PARCIALES")
        print("Revisa los errores arriba")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Actualización interrumpida")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        sys.exit(1)