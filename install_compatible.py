#!/usr/bin/env python3
'''
Script de Instalación Compatible PyTorch
========================================

Instala PyTorch con versiones compatibles automáticamente.
'''

import subprocess
import sys

def install_pytorch_compatible():
    """Instalar PyTorch con versiones compatibles"""
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
    """Instalar dependencias del proyecto"""
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
    """Instalar proyecto en modo desarrollo"""
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
        print(f"\n⏳ {step_name}...")
        if not step_func():
            print(f"❌ Error en: {step_name}")
            sys.exit(1)
    
    print("\n🎉 ¡Instalación completada!")
    print("\nPuedes probar con:")
    print("   python -c 'import torch; print(torch.__version__)'")
    print("   python quick_test.py")
