#!/usr/bin/env python3
"""
🔍 Diagnóstico Detallado de Imports - ONNs Module

Script para diagnosticar exactamente qué está fallando en los imports.
Prueba cada import individualmente y reporta errores específicos.
"""

import sys
import traceback
from pathlib import Path

def check_file_existence():
    """Verificar que todos los archivos existen."""
    print("📁 Checking file existence...")
    
    required_files = [
        "torchonn/__init__.py",
        "torchonn/onns/__init__.py", 
        "torchonn/onns/architectures/__init__.py",
        "torchonn/onns/architectures/base_onn.py",
        "torchonn/onns/architectures/coherent_onn.py",
        "torchonn/onns/benchmarks/__init__.py",
        "torchonn/onns/benchmarks/mnist_optical.py",
        "torchonn/onns/training/__init__.py",
        "torchonn/onns/utils/__init__.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ MISSING: {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def test_basic_python_imports():
    """Test imports básicos de Python y PyTorch."""
    print("\n🐍 Testing basic Python/PyTorch imports...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"   ❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"   ✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"   ❌ NumPy import failed: {e}")
        return False
    
    return True

def test_opticalci_core_imports():
    """Test imports del core de OpticalCI."""
    print("\n🔬 Testing OpticalCI core imports...")
    
    # Test 1: Main package
    try:
        import torchonn
        version = torchonn.__version__
        print(f"   ✅ torchonn v{version}")
    except ImportError as e:
        print(f"   ❌ torchonn import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ torchonn error: {e}")
        return False
    
    # Test 2: Core modules
    try:
        from torchonn import layers
        print(f"   ✅ torchonn.layers")
    except ImportError as e:
        print(f"   ❌ torchonn.layers import failed: {e}")
        return False
    
    # Test 3: Specific layers
    try:
        from torchonn.layers import MZILayer
        print(f"   ✅ MZILayer")
    except ImportError as e:
        print(f"   ❌ MZILayer import failed: {e}")
        print(f"       Full error: {e}")
        return False
    
    try:
        from torchonn.layers import MicroringResonator
        print(f"   ✅ MicroringResonator")
    except ImportError as e:
        print(f"   ❌ MicroringResonator import failed: {e}")
        return False
    
    # Test 4: Models
    try:
        from torchonn.models import ONNBaseModel
        print(f"   ✅ ONNBaseModel")
    except ImportError as e:
        print(f"   ❌ ONNBaseModel import failed: {e}")
        return False
    
    return True

def test_onns_module_imports():
    """Test imports del módulo ONNs paso a paso."""
    print("\n🌟 Testing ONNs module imports step by step...")
    
    # Test 1: Main ONNs module
    try:
        from torchonn import onns
        print(f"   ✅ torchonn.onns imported")
    except ImportError as e:
        print(f"   ❌ torchonn.onns import failed: {e}")
        print(f"       This could mean:")
        print(f"       - torchonn/onns/__init__.py missing or has errors")
        print(f"       - Syntax errors in ONNs module files")
        traceback.print_exc()
        return False
    
    # Test 2: Get ONNs info
    try:
        from torchonn.onns import get_onn_info
        info = get_onn_info()
        print(f"   ✅ get_onn_info: {info}")
    except ImportError as e:
        print(f"   ❌ get_onn_info import failed: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"   ❌ get_onn_info execution failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: Architectures module
    try:
        from torchonn.onns import architectures
        print(f"   ✅ torchonn.onns.architectures")
    except ImportError as e:
        print(f"   ❌ architectures import failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 4: BaseONN
    try:
        from torchonn.onns.architectures import BaseONN
        print(f"   ✅ BaseONN")
    except ImportError as e:
        print(f"   ❌ BaseONN import failed: {e}")
        print(f"       Check torchonn/onns/architectures/base_onn.py")
        traceback.print_exc()
        return False
    
    # Test 5: CoherentONN
    try:
        from torchonn.onns.architectures import CoherentONN
        print(f"   ✅ CoherentONN")
    except ImportError as e:
        print(f"   ❌ CoherentONN import failed: {e}")
        print(f"       Check torchonn/onns/architectures/coherent_onn.py")
        traceback.print_exc()
        return False
    
    # Test 6: Benchmarks
    try:
        from torchonn.onns.benchmarks import OpticalMNIST
        print(f"   ✅ OpticalMNIST")
    except ImportError as e:
        print(f"   ❌ OpticalMNIST import failed: {e}")
        print(f"       Check torchonn/onns/benchmarks/mnist_optical.py")
        traceback.print_exc()
        return False
    
    return True

def test_individual_file_syntax():
    """Test syntax de archivos individuales."""
    print("\n📝 Testing individual file syntax...")
    
    files_to_test = [
        "torchonn/onns/__init__.py",
        "torchonn/onns/architectures/__init__.py", 
        "torchonn/onns/architectures/base_onn.py",
        "torchonn/onns/architectures/coherent_onn.py",
        "torchonn/onns/benchmarks/__init__.py",
        "torchonn/onns/benchmarks/mnist_optical.py",
    ]
    
    all_good = True
    
    for file_path in files_to_test:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Try to compile
                compile(content, file_path, 'exec')
                print(f"   ✅ {file_path} - syntax OK")
                
            except SyntaxError as e:
                print(f"   ❌ {file_path} - SYNTAX ERROR:")
                print(f"       Line {e.lineno}: {e.text}")
                print(f"       Error: {e.msg}")
                all_good = False
            except Exception as e:
                print(f"   ❌ {file_path} - ERROR: {e}")
                all_good = False
        else:
            print(f"   ❌ {file_path} - FILE MISSING")
            all_good = False
    
    return all_good

def main():
    """Run diagnostic."""
    print("🔍 ONNs Module Import Diagnostic")
    print("=" * 50)
    
    # Step 1: File existence
    files_ok, missing = check_file_existence()
    if not files_ok:
        print(f"\n❌ CRITICAL: Missing files detected")
        print(f"   Missing: {missing}")
        print(f"\n💡 SOLUTION: Create these files first")
        return 1
    
    # Step 2: Basic imports
    if not test_basic_python_imports():
        print(f"\n❌ CRITICAL: Basic Python/PyTorch imports failed")
        print(f"\n💡 SOLUTION: Check Python environment and PyTorch installation")
        return 1
    
    # Step 3: OpticalCI core
    if not test_opticalci_core_imports():
        print(f"\n❌ CRITICAL: OpticalCI core imports failed")
        print(f"\n💡 SOLUTION: Check OpticalCI base installation")
        return 1
    
    # Step 4: File syntax
    if not test_individual_file_syntax():
        print(f"\n❌ CRITICAL: Syntax errors in ONNs files")
        print(f"\n💡 SOLUTION: Fix syntax errors shown above")
        return 1
    
    # Step 5: ONNs imports
    if not test_onns_module_imports():
        print(f"\n❌ CRITICAL: ONNs module imports failed")
        print(f"\n💡 SOLUTION: Check error messages above for specific issues")
        return 1
    
    print(f"\n✅ ALL DIAGNOSTICS PASSED!")
    print(f"   All files exist and have valid syntax")
    print(f"   All imports working correctly")
    print(f"   ONNs module should be functional")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)