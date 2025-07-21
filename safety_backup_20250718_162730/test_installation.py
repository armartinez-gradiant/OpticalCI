"""
Installation verification test for PtONN-TESTS

This script verifies that all dependencies are installed correctly
and that the PtONN-TESTS package is working properly.

Usage:
    python test_installation.py
    
Exit codes:
    0: All tests passed
    1: Some tests failed
"""

import sys
import importlib
import platform
import traceback
import warnings
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def print_header(title: str, char: str = "=") -> None:
    """Print a formatted header."""
    print(f"\n{char * 20} {title} {char * 20}")

def print_status(status: bool, message: str) -> None:
    """Print status message with appropriate emoji."""
    emoji = "✅" if status else "❌"
    print(f"{emoji} {message}")

def check_python_version() -> bool:
    """Verify Python version compatibility."""
    print_header("Python Version Check")
    
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    print(f"🏗️  Platform: {platform.platform()}")
    print(f"🖥️  Architecture: {platform.machine()}")
    print(f"📁 Python executable: {sys.executable}")
    
    # Check version requirements
    if version.major != 3:
        print_status(False, "Python 3 is required")
        return False
    
    if version.minor < 8:
        print_status(False, "Python 3.8 or higher is required")
        return False
    
    if version.minor >= 13:
        print_status(True, "Python 3.13 detected - may have compatibility issues")
        print("   ⚠️  Consider using Python 3.11 for best compatibility")
    else:
        print_status(True, "Python version is compatible")
    
    return True

def check_module_import(
    module_name: str, 
    package_name: str = None, 
    show_error: bool = True,
    optional: bool = False
) -> bool:
    """Verify module import and get version info."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print_status(True, f"{package_name or module_name}: {version}")
        return True
    except ImportError as e:
        if not optional and show_error:
            print_status(False, f"{package_name or module_name}: Import failed - {e}")
        elif optional:
            print_status(True, f"{package_name or module_name}: Not available (optional)")
        return not optional  # Return True for optional modules even if not found
    except Exception as e:
        if show_error:
            print_status(False, f"{package_name or module_name}: Error - {e}")
        return False

def check_pytorch_installation() -> bool:
    """Verify PyTorch installation and compatibility."""
    print_header("PyTorch Installation")
    
    success = True
    
    # Check PyTorch core
    print("📦 Checking PyTorch core...")
    try:
        import torch
        print_status(True, f"PyTorch: {torch.__version__}")
        
        # Check device availability
        try:
            cuda_available = torch.cuda.is_available()
            cuda_count = torch.cuda.device_count() if cuda_available else 0
            print(f"   🔥 CUDA available: {cuda_available} ({cuda_count} devices)")
        except Exception as e:
            print(f"   ⚠️  CUDA check failed: {e}")
        
        try:
            mps_available = torch.backends.mps.is_available()
            print(f"   🍎 MPS available: {mps_available}")
        except Exception as e:
            print(f"   ⚠️  MPS check failed: {e}")
            
        # Check PyTorch build info
        print(f"   🔧 PyTorch built with CUDA: {torch.version.cuda}")
        print(f"   🔧 PyTorch debug: {torch.version.debug}")
        
    except ImportError as e:
        print_status(False, f"PyTorch: Import failed - {e}")
        success = False
    except Exception as e:
        print_status(False, f"PyTorch: Error - {e}")
        success = False
    
    # Check TorchVision
    print("📦 Checking TorchVision...")
    try:
        import torchvision
        print_status(True, f"TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print_status(False, f"TorchVision: Import failed - {e}")
        success = False
    except Exception as e:
        print_status(False, f"TorchVision: Error - {e}")
        success = False
    
    # Check TorchAudio
    print("📦 Checking TorchAudio...")
    try:
        import torchaudio
        print_status(True, f"TorchAudio: {torchaudio.__version__}")
    except ImportError as e:
        print_status(False, f"TorchAudio: Import failed - {e}")
        success = False
    except Exception as e:
        print_status(False, f"TorchAudio: Error - {e}")
        success = False
    
    return success

def check_scientific_libraries() -> bool:
    """Verify scientific computing libraries."""
    print_header("Scientific Libraries")
    
    success = True
    libraries = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
    ]
    
    for module_name, package_name in libraries:
        if not check_module_import(module_name, package_name):
            success = False
    
    # Special check for NumPy compatibility
    try:
        import numpy as np
        if hasattr(np, '__version__'):
            version_parts = np.__version__.split('.')
            major_version = int(version_parts[0])
            if major_version >= 2:
                print("   ⚠️  NumPy 2.x detected - may cause compatibility issues")
                print("   💡 Consider downgrading: pip install 'numpy<2.0'")
    except:
        pass
    
    return success

def check_ptonn_installation() -> bool:
    """Verify PtONN-TESTS installation."""
    print_header("PtONN-TESTS Installation")
    
    # Check main module
    try:
        import torchonn
        version = getattr(torchonn, '__version__', 'unknown')
        print_status(True, f"TorchONN main module: {version}")
    except ImportError as e:
        print_status(False, f"TorchONN: {e}")
        print("💡 Make sure you installed the package: pip install -e .")
        return False
    except Exception as e:
        print_status(False, f"TorchONN: Error - {e}")
        return False
    
    # Check submodules
    submodules = [
        ('torchonn.layers', 'Layers'),
        ('torchonn.models', 'Models'),
        ('torchonn.devices', 'Devices'),
        ('torchonn.ops', 'Operations'),
        ('torchonn.utils', 'Utils'),
    ]
    
    success = True
    for module_name, display_name in submodules:
        if not check_module_import(module_name, display_name):
            success = False
    
    # Check specific layer imports
    try:
        from torchonn.layers import MZILayer, MZIBlockLinear
        print_status(True, "Layer classes imported successfully")
    except ImportError as e:
        print_status(False, f"Layer import failed: {e}")
        success = False
    except Exception as e:
        print_status(False, f"Layer import error: {e}")
        success = False
    
    # Check model base class
    try:
        from torchonn.models import ONNBaseModel
        print_status(True, "Model base class imported successfully")
    except ImportError as e:
        print_status(False, f"Model import failed: {e}")
        success = False
    except Exception as e:
        print_status(False, f"Model import error: {e}")
        success = False
    
    return success

def check_test_environment() -> bool:
    """Verify testing environment."""
    print_header("Test Environment")
    
    success = True
    
    # Check pytest
    if not check_module_import('pytest', 'pytest'):
        success = False
    
    # Check optional testing tools
    check_module_import('pytest_cov', 'pytest-cov', optional=True)
    check_module_import('pytest_xdist', 'pytest-xdist', optional=True)
    
    return success

def run_basic_functionality_test() -> bool:
    """Run basic functionality tests."""
    print_header("Basic Functionality Test")
    
    try:
        print("🧪 Testing tensor operations...")
        import torch
        x = torch.randn(2, 4)
        print_status(True, f"Tensor created: {x.shape}")
        
        print("🧪 Testing MZILayer...")
        from torchonn.layers import MZILayer
        layer = MZILayer(in_features=4, out_features=2)
        output = layer(x)
        print_status(True, f"MZILayer forward pass: {output.shape}")
        
        print("🧪 Testing MZIBlockLinear...")
        from torchonn.layers import MZIBlockLinear
        block_layer = MZIBlockLinear(
            in_features=4,
            out_features=2,
            miniblock=2,
            mode="usv"
        )
        block_output = block_layer(x)
        print_status(True, f"MZIBlockLinear forward pass: {block_output.shape}")
        
        print("🧪 Testing ONNBaseModel...")
        from torchonn.models import ONNBaseModel
        
        class SimpleModel(ONNBaseModel):
            def __init__(self):
                super().__init__()
                self.layer = MZILayer(4, 2)
            
            def forward(self, x):
                return self.layer(x)
        
        model = SimpleModel()
        model_output = model(x)
        print_status(True, f"Simple model forward pass: {model_output.shape}")
        
        print("🧪 Testing gradient computation...")
        model_output.sum().backward()
        print_status(True, "Gradient computation successful")
        
        print("🧪 Testing device operations...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_device = x.to(device)
        model_device = model.to(device)
        device_output = model_device(x_device)
        print_status(True, f"Device operations successful on {device}")
        
        return True
        
    except Exception as e:
        print_status(False, f"Basic functionality test failed: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return False

def run_package_info_test() -> bool:
    """Test package information and utilities."""
    print_header("Package Information")
    
    try:
        from torchonn.utils import get_package_info, check_torch_version
        
        # Package info
        info = get_package_info()
        print(f"📦 Package version: {info.get('version', 'unknown')}")
        print(f"👨‍💻 Author: {info.get('author', 'unknown')}")
        
        # PyTorch compatibility
        torch_info = check_torch_version()
        compatible = torch_info.get('version_compatible', False)
        print_status(compatible, f"PyTorch compatibility check")
        
        return True
        
    except Exception as e:
        print_status(False, f"Package info test failed: {e}")
        return False

def generate_system_report() -> Dict[str, Any]:
    """Generate comprehensive system report."""
    print_header("System Report")
    
    report = {
        "timestamp": str(platform.datetime.now()) if hasattr(platform, 'datetime') else "unknown",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
        },
        "packages": {},
        "devices": {},
    }
    
    # Check installed packages
    try:
        import pkg_resources
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        report["packages"] = {
            key: installed_packages.get(key, "not installed") 
            for key in ["torch", "torchvision", "torchaudio", "numpy", "scipy", "matplotlib"]
        }
    except:
        pass
    
    # Check devices
    try:
        import torch
        report["devices"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        }
    except:
        pass
    
    print("📊 System report generated successfully")
    return report

def main() -> int:
    """Main function to run all tests."""
    print("🚀 PtONN-TESTS Installation Verification")
    print("=" * 60)
    
    # Run all checks
    checks = [
        ("Python version", check_python_version),
        ("PyTorch installation", check_pytorch_installation),
        ("Scientific libraries", check_scientific_libraries),
        ("PtONN-TESTS installation", check_ptonn_installation),
        ("Test environment", check_test_environment),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_status(False, f"Error during {check_name}: {e}")
            traceback.print_exc()
            results.append((check_name, False))
    
    # Run functionality tests only if basic checks pass
    basic_checks_passed = all(result for _, result in results)
    if basic_checks_passed:
        try:
            functionality_result = run_basic_functionality_test()
            results.append(("Basic functionality", functionality_result))
        except Exception as e:
            print_status(False, f"Functionality test error: {e}")
            results.append(("Basic functionality", False))
        
        try:
            package_info_result = run_package_info_test()
            results.append(("Package information", package_info_result))
        except Exception as e:
            print_status(False, f"Package info test error: {e}")
            results.append(("Package information", False))
    else:
        print_header("Skipping Advanced Tests")
        print("⚠️  Basic checks failed, skipping functionality tests")
    
    # Generate system report
    try:
        system_report = generate_system_report()
    except Exception as e:
        print(f"⚠️  Could not generate system report: {e}")
    
    # Final summary
    print_header("FINAL SUMMARY", "=")
    passed = 0
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    success_rate = (passed / len(results)) * 100 if results else 0
    print(f"\n🎯 Result: {passed}/{len(results)} checks passed ({success_rate:.1f}%)")
    
    if passed == len(results):
        print("🎉 All checks passed! PtONN-TESTS is ready to use.")
        print("\n📚 Next steps:")
        print("   • Run pytest tests/ to verify full functionality")
        print("   • Check examples in the documentation")
        print("   • Start building your photonic neural networks!")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   • Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   • Check NumPy version compatibility: pip install 'numpy<2.0'")
        print("   • Verify PyTorch installation: pip install torch torchvision torchaudio")
        print("   • Install package in development mode: pip install -e .")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Installation test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)