"""
PtONN-TESTS - A modern, updated PyTorch Library for Photonic Integrated Circuit Simulation

This is an updated and improved version of pytorch-onn that works with modern PyTorch versions.

🌟 NEW: ONNs Module - Optical Neural Networks
📚 Architectures: CoherentONN, IncoherentONN (coming soon)
🎯 Benchmarks: OpticalMNIST, classification tasks
✅ Physics: Energy conservation, unitarity validation

IMPORTANT: This update only ADDS new functionality, doesn't modify existing code.
""" 

__version__ = "1.1.0"  # Incremented for ONNs module
__author__ = "Anxo Rodríguez Martínez"
__email__ = "anxo.martinez@email.com"

# ============================================================================
# EXISTING IMPORTS (NO CHANGES)
# ============================================================================

# Importar módulos principales existentes
try:
    from . import layers
    from . import models
    from . import devices
    from . import ops
    from . import components
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import some core modules: {e}")
    CORE_MODULES_AVAILABLE = False

# ============================================================================
# NEW: ONNs MODULE IMPORTS
# ============================================================================

# Importar nuevo módulo ONNs (solo si está disponible)
try:
    from . import onns
    ONNS_MODULE_AVAILABLE = True
    print("🌟 ONNs module loaded successfully!")
except ImportError as e:
    ONNS_MODULE_AVAILABLE = False
    # No warning - es normal durante desarrollo incremental

# ============================================================================
# EXISTING CONFIGURATION (NO CHANGES)
# ============================================================================

# Configuración por defecto
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = "float32"

def get_version():
    """Obtener versión del paquete."""
    return __version__

def get_device():
    """Obtener dispositivo por defecto."""
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return DEFAULT_DEVICE

# ============================================================================
# NEW: ONNs FUNCTIONALITY
# ============================================================================

def get_onns_info():
    """
    Obtener información del módulo ONNs.
    
    Returns:
        Dict con información del módulo ONNs o None si no está disponible
    """
    if not ONNS_MODULE_AVAILABLE:
        return {
            "available": False,
            "message": "ONNs module not available. Check implementation."
        }
    
    try:
        return onns.get_onn_info()
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

def list_onn_architectures():
    """
    Listar arquitecturas ONNs disponibles.
    
    Returns:
        Dict con arquitecturas disponibles
    """
    if not ONNS_MODULE_AVAILABLE:
        return {}
    
    try:
        from .onns.architectures import list_available_architectures
        return list_available_architectures()
    except Exception:
        return {}

def create_onn(architecture: str, **kwargs):
    """
    Factory function para crear ONNs.
    
    Args:
        architecture: Nombre de la arquitectura ("CoherentONN", etc.)
        **kwargs: Argumentos para el constructor
        
    Returns:
        Instancia de ONN o raises ValueError si no está disponible
    """
    if not ONNS_MODULE_AVAILABLE:
        raise ImportError("ONNs module not available")
    
    try:
        from .onns.architectures import create_onn
        return create_onn(architecture, **kwargs)
    except Exception as e:
        raise ValueError(f"Could not create ONN '{architecture}': {e}")

def run_onn_demo(demo_type: str = "quick", **kwargs):
    """
    Ejecutar demo de ONNs.
    
    Args:
        demo_type: Tipo de demo ("quick", "full")
        **kwargs: Argumentos específicos del demo
        
    Returns:
        Resultados del demo o None si falla
    """
    if not ONNS_MODULE_AVAILABLE:
        print("❌ ONNs module not available for demo")
        return None
    
    try:
        from .onns.benchmarks import run_quick_demo
        
        if demo_type == "quick":
            return run_quick_demo(
                image_size=kwargs.get("image_size", 4),
                n_epochs=kwargs.get("n_epochs", 3)
            )
        else:
            print(f"Demo type '{demo_type}' not implemented yet")
            return None
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None

# ============================================================================
# UPDATED __all__ (CONSERVATIVE)
# ============================================================================

# Información del paquete base (sin cambios)
__all__ = [
    "get_version",
    "get_device",
    "__version__",
]

# Agregar módulos core si están disponibles
if CORE_MODULES_AVAILABLE:
    __all__.extend([
        "layers",
        "models", 
        "devices",
        "ops",
        "components",
    ])

# Agregar funcionalidad ONNs si está disponible
if ONNS_MODULE_AVAILABLE:
    __all__.extend([
        "onns",
        "get_onns_info",
        "list_onn_architectures", 
        "create_onn",
        "run_onn_demo"
    ])

# ============================================================================
# INFORMATIONAL FUNCTIONS
# ============================================================================

def print_package_info():
    """Imprimir información completa del paquete."""
    print(f"📦 PtONN-TESTS v{__version__}")
    print(f"👤 Author: {__author__}")
    print("=" * 50)
    
    # Core modules
    print("🔧 Core Modules:")
    if CORE_MODULES_AVAILABLE:
        core_modules = ["layers", "models", "devices", "ops", "components"]
        for module in core_modules:
            print(f"   ✅ {module}")
    else:
        print("   ❌ Core modules not available")
    
    # ONNs module
    print("\n🌟 ONNs Module:")
    if ONNS_MODULE_AVAILABLE:
        print("   ✅ onns (Optical Neural Networks)")
        
        # ONNs info
        try:
            onns_info = get_onns_info()
            if onns_info.get("available", False):
                architectures = onns_info.get("architectures_available", [])
                print(f"   📐 Architectures: {architectures}")
            else:
                print(f"   ⚠️ {onns_info.get('message', 'Unknown issue')}")
        except Exception as e:
            print(f"   ⚠️ Error getting ONNs info: {e}")
    else:
        print("   ❌ ONNs module not available")
        print("       Run: pip install -e . to ensure proper installation")
    
    # Device info
    print(f"\n💻 Device: {get_device()}")
    
    # Usage examples
    print(f"\n🚀 Quick Start:")
    print(f"   import torchonn")
    if ONNS_MODULE_AVAILABLE:
        print(f"   # Run ONNs demo:")
        print(f"   torchonn.run_onn_demo('quick')")
        print(f"   # Create CoherentONN:")
        print(f"   onn = torchonn.create_onn('CoherentONN', layer_sizes=[4,8,3])")
    print(f"   # Use existing components:")
    print(f"   from torchonn.layers import MZILayer")

def check_installation():
    """Verificar que la instalación está completa."""
    issues = []
    
    # Check core modules
    if not CORE_MODULES_AVAILABLE:
        issues.append("Core modules not available")
    
    # Check ONNs module
    if not ONNS_MODULE_AVAILABLE:
        issues.append("ONNs module not available")
    
    # Check PyTorch
    try:
        import torch
        torch_version = torch.__version__
        if not torch_version.startswith('2.'):
            issues.append(f"PyTorch version {torch_version} may not be supported (recommend 2.0+)")
    except ImportError:
        issues.append("PyTorch not available")
    
    # Report
    if not issues:
        print("✅ Installation check: All components available")
        return True
    else:
        print("⚠️ Installation issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False