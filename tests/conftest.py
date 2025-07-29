"""
Configuración común para tests de OpticalCI

Este archivo contiene:
- Fixtures comunes para todos los tests
- Configuración de pytest
- Funciones helper para testing
- Marcadores personalizados
- Setup y teardown globales
"""

import pytest
import torch
import numpy as np
import warnings
import time
import gc
from typing import Dict, Any, List, Optional

# Suprimir warnings de deprecation durante tests
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

def pytest_configure(config):
    """Configuración global de pytest."""
    # Marcadores personalizados
    config.addinivalue_line(
        "markers", "slow: marca tests como lentos (usar -m 'not slow' para excluir)"
    )
    config.addinivalue_line(
        "markers", "gpu: marca tests que requieren GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marca tests de integración"
    )
    config.addinivalue_line(
        "markers", "physics: marca tests de validación física"
    )
    config.addinivalue_line(
        "markers", "performance: marca tests de performance"
    )


def pytest_collection_modifyitems(config, items):
    """Modificar colección de tests."""
    # Auto-marcar tests lentos basado en nombre
    slow_keywords = ["large_batch", "multiple_forward", "performance", "complex"]
    
    for item in items:
        # Marcar tests lentos automáticamente
        if any(keyword in item.name.lower() for keyword in slow_keywords):
            item.add_marker(pytest.mark.slow)
        
        # Marcar tests de GPU automáticamente
        if "gpu" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Marcar tests de integración
        if "integration" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)


# ============================================================================
# FIXTURES GLOBALES
# ============================================================================

@pytest.fixture(scope="session")
def torch_device():
    """Fixture global para device de PyTorch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}")
    
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


@pytest.fixture(scope="session")
def torch_settings():
    """Configuración global de PyTorch para tests."""
    # Configuración determinística para reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuración de PyTorch
    original_num_threads = torch.get_num_threads()
    torch.set_num_threads(2)  # Limitar threads para tests
    
    # Guardar configuración original
    original_settings = {
        'num_threads': original_num_threads,
    }
    
    yield original_settings
    
    # Restaurar configuración
    torch.set_num_threads(original_settings['num_threads'])


@pytest.fixture
def clean_gpu_memory(torch_device):
    """Fixture para limpiar memoria GPU antes y después de tests."""
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    yield
    
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()


@pytest.fixture
def performance_timer():
    """Fixture para medir tiempo de ejecución."""
    start_time = time.time()
    
    def get_elapsed():
        return time.time() - start_time
    
    yield get_elapsed


# ============================================================================
# FIXTURES PARA DATOS DE TEST
# ============================================================================

@pytest.fixture
def standard_wavelengths(torch_device):
    """Wavelengths estándar para tests."""
    return torch.linspace(1530e-9, 1570e-9, 100, device=torch_device, dtype=torch.float32)


@pytest.fixture
def telecom_wavelengths(torch_device):
    """Wavelengths de telecomunicaciones."""
    return torch.tensor([
        1530e-9, 1535e-9, 1540e-9, 1545e-9, 1550e-9, 
        1555e-9, 1560e-9, 1565e-9, 1570e-9
    ], device=torch_device, dtype=torch.float32)


@pytest.fixture
def random_optical_signal(torch_device):
    """Señal óptica aleatoria para tests."""
    def _generate(batch_size=8, n_wavelengths=50, power_range=(0.1, 1.0)):
        # Generar señal con potencia realista
        signal = torch.rand(batch_size, n_wavelengths, device=torch_device, dtype=torch.float32)
        signal = signal * (power_range[1] - power_range[0]) + power_range[0]
        return signal
    
    return _generate


@pytest.fixture
def coherent_optical_signal(torch_device):
    """Señal óptica coherente para tests."""
    def _generate(batch_size=8, n_wavelengths=50, amplitude=1.0, phase=0.0):
        # Generar señal coherente con fase controlada
        signal = torch.ones(batch_size, n_wavelengths, device=torch_device, dtype=torch.complex64)
        signal = signal * amplitude * torch.exp(1j * phase)
        return signal
    
    return _generate


# ============================================================================
# FIXTURES PARA PARÁMETROS FÍSICOS
# ============================================================================

@pytest.fixture
def realistic_microring_params():
    """Parámetros realistas para microring resonators."""
    return {
        "radius": 10e-6,  # 10 μm
        "q_factor": 5000,  # Q realista para demos
        "center_wavelength": 1550e-9,  # Telecom band
        "coupling_mode": "critical"
    }


@pytest.fixture
def wdm_channel_params():
    """Parámetros para sistemas WDM."""
    return {
        "wavelengths": [1530e-9, 1540e-9, 1550e-9, 1560e-9],
        "channel_spacing": 10e-9,  # 10 nm spacing
        "n_channels": 4
    }


@pytest.fixture
def mzi_standard_params():
    """Parámetros estándar para MZI."""
    return {
        "dimensions": [(4, 4), (6, 6), (8, 8)],
        "modes": ["usv", "weight", "phase"]
    }


# ============================================================================
# FUNCIONES HELPER
# ============================================================================

def assert_energy_conservation(input_signal, output_signal, tolerance=1e-3, name=""):
    """Helper para verificar conservación de energía."""
    input_energy = torch.sum(torch.abs(input_signal)**2, dim=1)
    output_energy = torch.sum(torch.abs(output_signal)**2, dim=1)
    
    energy_ratio = output_energy / torch.clamp(input_energy, min=1e-10)
    energy_conservation = torch.mean(energy_ratio)
    
    assert abs(energy_conservation - 1.0) < tolerance, (
        f"Energy not conserved {name}: {energy_conservation:.6f} "
        f"(tolerance: {tolerance})"
    )


def assert_physical_range(tensor, min_val=0.0, max_val=1.0, name="tensor"):
    """Helper para verificar rango físico."""
    min_actual = torch.min(tensor).item()
    max_actual = torch.max(tensor).item()
    
    assert min_actual >= min_val, (
        f"{name} below physical range: {min_actual:.6f} < {min_val}"
    )
    assert max_actual <= max_val, (
        f"{name} above physical range: {max_actual:.6f} > {max_val}"
    )


def assert_no_nan_inf(tensor, name="tensor"):
    """Helper para verificar ausencia de NaN/Inf."""
    assert not torch.any(torch.isnan(tensor)), f"NaN detected in {name}"
    assert not torch.any(torch.isinf(tensor)), f"Inf detected in {name}"


def assert_realistic_extinction_ratio(er_db, q_factor=None, tolerance_db=10):
    """Helper para verificar extinction ratio realista."""
    # Rango realista basado en Q factor
    if q_factor is not None:
        expected_min = max(5, 0.5 * np.log10(q_factor))
        expected_max = min(30, 3 * np.log10(q_factor))
    else:
        expected_min, expected_max = 5, 30
    
    assert expected_min <= er_db <= expected_max, (
        f"Extinction ratio not realistic: {er_db:.1f} dB "
        f"(expected: {expected_min:.1f}-{expected_max:.1f} dB)"
    )


def measure_forward_time(network, input_data, n_runs=10):
    """Helper para medir tiempo de forward pass."""
    # Warm up
    for _ in range(3):
        _ = network(input_data)
    
    # Measure
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        _ = network(input_data)
        times.append(time.time() - start_time)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def create_test_wavelength_range(center_wl=1550e-9, range_pm=1000, n_points=100, device=None):
    """Helper para crear rangos de wavelength para tests."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    half_range = range_pm * 1e-12 / 2  # pm to m
    return torch.linspace(
        center_wl - half_range,
        center_wl + half_range,
        n_points,
        device=device,
        dtype=torch.float32
    )


# ============================================================================
# PARAMETRIZACIONES COMUNES
# ============================================================================

# Batch sizes para tests parametrizados
BATCH_SIZES = [1, 4, 16]

# Q factors para tests de microring
Q_FACTORS = [500, 1000, 2000, 5000]

# Dimensiones para tests de MZI
MZI_DIMENSIONS = [(2, 2), (4, 4), (6, 6), (8, 8)]

# Modos para MZIBlockLinear
MZI_MODES = ["usv", "weight", "phase"]

# Coupling modes para microring
COUPLING_MODES = ["critical", "under", "over"]


# ============================================================================
# SKIP CONDITIONS
# ============================================================================

def requires_gpu():
    """Decorator para tests que requieren GPU."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )


def requires_large_memory():
    """Decorator para tests que requieren mucha memoria."""
    return pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 4e9,
        reason="Insufficient GPU memory (< 4GB)"
    )


# ============================================================================
# REPORTES PERSONALIZADOS
# ============================================================================

def pytest_runtest_teardown(item, nextitem):
    """Cleanup después de cada test."""
    # Limpiar memoria GPU si está disponible
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Garbage collection
    gc.collect()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Resumen personalizado al final de los tests."""
    if hasattr(terminalreporter, 'stats'):
        passed = len(terminalreporter.stats.get('passed', []))
        failed = len(terminalreporter.stats.get('failed', []))
        skipped = len(terminalreporter.stats.get('skipped', []))
        
        print(f"\n🎯 PtONN-TESTS Summary:")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏭️  Skipped: {skipped}")
        
        if torch.cuda.is_available():
            print(f"   🔧 GPU Tests: Enabled")
            print(f"   💾 GPU Memory: {torch.cuda.memory_allocated()/1e6:.1f} MB used")
        else:
            print(f"   🔧 GPU Tests: Disabled (CPU only)")