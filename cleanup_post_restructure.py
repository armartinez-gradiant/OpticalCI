#!/usr/bin/env python3
"""
Script de Limpieza Post-Reestructuración
========================================

Generado automáticamente el 2025-07-17 15:01:48
Elimina archivos duplicados, obsoletos y temporales detectados.

IMPORTANTE: Revisa este script antes de ejecutarlo
"""

import os
import shutil
from pathlib import Path

def main():
    repo_path = Path(__file__).parent
    
    print("🧹 Iniciando limpieza post-reestructuración...")
    
    # Archivos duplicados (mantener el primero, eliminar el resto)
    duplicates = {'4e3e7cc37dcd67debc0836b4b7281b78': ['test_installation.py', 'backup_20250717_144328/test_installation.py'], 'f780158df17531e8a1cda44046ae7943': ['restructure_ptonn.py', 'backup_20250717_144328/restructure_ptonn.py'], '6e746d8f807ae6155caa2018d67538f4': ['benchmark.py', 'backup_20250717_144328/benchmark.py'], 'd75da94be6d7c1fded70d29c1e27f6cb': ['setup.py', 'backup_20250717_144328/setup.py'], '1f5b3d3e745046b5b822d91ed57318ee': ['explore.py', 'backup_20250717_144328/explore.py'], '7826a218a1f25f508ba3c89442405b5f': ['examples/realistic_mz_network.py', 'backup_20250717_144328/examples/realistic_mz_network.py'], 'b36833a2aec472df5217a8cb62d87553': ['examples/__init__.py', 'backup_20250717_144328/examples/__init__.py'], 'a19ec2ff63346d367564a1d4ff347ff7': ['examples/complete_photonic_system.py', 'backup_20250717_144328/examples/complete_photonic_system.py'], 'df52c06821174b5108345a2007a94a37': ['examples/advanced_usage.py', 'backup_20250717_144328/examples/advanced_usage.py'], '45846d69e5cfc940f7d597feb49b940e': ['examples/example_basic_usage.py', 'backup_20250717_144328/examples/example_basic_usage.py'], '40ce6c50e0add379c64f9d9396124246': ['examples/reck_architecture_chip.py', 'backup_20250717_144328/examples/reck_architecture_chip.py'], 'e12a89155b78f1e22c0364435d429157': ['examples/example_training.py', 'backup_20250717_144328/examples/example_training.py'], '4b004935f1172a59ba9618066809fd07': ['examples/basic_usage.py', 'backup_20250717_144328/examples/basic_usage.py'], '5b87c4620ed9fb08cb780eac3e8de1c4': ['tests/__init__.py', 'backup_20250717_144328/tests/__init__.py'], '06bc64b04ad7cdad0ae2d7ec36693c89': ['tests/test_layers.py', 'backup_20250717_144328/tests/test_layers.py'], '0d74267b932b0d965223d81ef3841497': ['tests/test_integration.py', 'backup_20250717_144328/tests/test_integration.py'], 'd73025de82ba4a8e2f362360a2c570c2': ['tests/test_models.py', 'backup_20250717_144328/tests/test_models.py'], '803e0935f55139083f590fab2d5ca660': ['backup_20250717_144328/examples_legacy_backup/core/builder.py', 'examples_legacy_backup/core/builder.py'], '04046609d0d7af28278a64980178fa03': ['backup_20250717_144328/examples_legacy_backup/core/models/mzi_cnn.py', 'examples_legacy_backup/core/models/mzi_cnn.py'], '059ef25e77aebb42e2ea242507ad44c6': ['backup_20250717_144328/torchonn/utils/__init__.py', 'torchonn/utils/__init__.py'], '5485b2bfbbffc0322970bf7d7c995510': ['backup_20250717_144328/torchonn/utils/helpers.py', 'torchonn/utils/helpers.py'], '8ddfd07cd49ebf3a2f71e7b22cfeaebf': ['backup_20250717_144328/torchonn/devices/device_configs.py', 'torchonn/devices/device_configs.py'], 'fabde27a19318df51904c679fd679c82': ['backup_20250717_144328/torchonn/devices/__init__.py', 'torchonn/devices/__init__.py'], 'cafb0df6ff48e3f69e255a37dec66790': ['backup_20250717_144328/torchonn/layers/mzi_layer.py', 'torchonn/layers/mzi_layer.py'], 'c96718c0be0223d11354c7649c140181': ['backup_20250717_144328/torchonn/layers/mzi_block_linear.py', 'torchonn/layers/mzi_block_linear.py'], 'a1a90802f415b2f1b2365c1bcd90bd5f': ['backup_20250717_144328/torchonn/models/base_model.py', 'torchonn/models/base_model.py'], 'f02716189dc2db90f2dc82e8e9ce461a': ['backup_20250717_144328/torchonn/ops/__init__.py', 'torchonn/ops/__init__.py'], 'c40a006d0927c82a806531f609bf98ea': ['backup_20250717_144328/torchonn/ops/operations.py', 'torchonn/ops/operations.py'], '179f3279ae69daa8a46d5b2265209c0e': ['configs/hardware/__init__.py', 'torchonn/hardware/__init__.py']}
    
    # Archivos obsoletos
    obsolete_files = []
    
    # Archivos temporales (primeros 50)
    temp_files = ['tests/__pycache__/__init__.cpython-312.pyc', 'tests/__pycache__/test_models.cpython-312-pytest-7.4.4.pyc', 'tests/__pycache__/test_layers.cpython-312-pytest-7.4.4.pyc', 'tests/__pycache__/test_integration.cpython-312-pytest-7.4.4.pyc', 'tests/performance/__pycache__/test_speed.cpython-312-pytest-7.4.4.pyc', 'tests/performance/__pycache__/test_accuracy.cpython-312-pytest-7.4.4.pyc', 'tests/performance/__pycache__/__init__.cpython-312.pyc', 'tests/performance/__pycache__/test_memory_usage.cpython-312-pytest-7.4.4.pyc', 'tests/unit/__pycache__/test_systems.cpython-312-pytest-7.4.4.pyc', 'tests/unit/__pycache__/test_components.cpython-312-pytest-7.4.4.pyc', 'tests/unit/__pycache__/__init__.cpython-312.pyc', 'tests/integration/__pycache__/__init__.cpython-312.pyc', 'tests/integration/__pycache__/test_benchmarks.cpython-312-pytest-7.4.4.pyc', 'tests/integration/__pycache__/test_training_loops.cpython-312-pytest-7.4.4.pyc', 'torchonn/__pycache__/__init__.cpython-312.pyc', 'torchonn/utils/__pycache__/__init__.cpython-312.pyc', 'torchonn/utils/__pycache__/helpers.cpython-312.pyc', 'torchonn/components/__pycache__/microring_resonator.cpython-312.pyc', 'torchonn/components/__pycache__/__init__.cpython-312.pyc', 'torchonn/components/__pycache__/photodetector.cpython-312.pyc', 'torchonn/components/__pycache__/base_component.cpython-312.pyc', 'torchonn/components/__pycache__/laser_source.cpython-312.pyc', 'torchonn/components/__pycache__/waveguide.cpython-312.pyc', 'torchonn/components/__pycache__/directional_coupler.cpython-312.pyc', 'torchonn/components/__pycache__/add_drop_mrr.cpython-312.pyc', 'torchonn/components/__pycache__/phase_change_cell.cpython-312.pyc', 'torchonn/systems/__pycache__/__init__.cpython-312.pyc', 'torchonn/devices/__pycache__/__init__.cpython-312.pyc', 'torchonn/devices/__pycache__/device_configs.cpython-312.pyc', 'torchonn/layers/__pycache__/photonic_linear.cpython-312.pyc', 'torchonn/layers/__pycache__/__init__.cpython-312.pyc', 'torchonn/layers/__pycache__/mzi_block_linear.cpython-312.pyc', 'torchonn/layers/__pycache__/photonic_conv2d.cpython-312.pyc', 'torchonn/layers/__pycache__/mzi_layer.cpython-312.pyc', 'torchonn/layers/__pycache__/mrr_weight_bank.cpython-312.pyc', 'torchonn/models/__pycache__/__init__.cpython-312.pyc', 'torchonn/models/__pycache__/base_model.cpython-312.pyc', 'torchonn/ops/__pycache__/operations.cpython-312.pyc', 'torchonn/ops/__pycache__/__init__.cpython-312.pyc', 'tests/__pycache__', 'torchonn/__pycache__', 'tests/performance/__pycache__', 'tests/unit/__pycache__', 'tests/integration/__pycache__', 'torchonn/utils/__pycache__', 'torchonn/components/__pycache__', 'torchonn/systems/__pycache__', 'torchonn/devices/__pycache__', 'torchonn/layers/__pycache__', 'torchonn/models/__pycache__']
    
    # Eliminar duplicados (excepto el primero)
    for file_hash, files in duplicates.items():
        if len(files) > 1:
            print(f"Eliminando duplicados de {files[0]}:")
            for duplicate in files[1:]:
                file_path = repo_path / duplicate
                if file_path.exists():
                    print(f"  - Eliminando: {duplicate}")
                    file_path.unlink()
    
    # Eliminar archivos obsoletos
    for obsolete in obsolete_files:
        file_path = repo_path / obsolete
        if file_path.exists():
            print(f"Eliminando obsoleto: {obsolete}")
            file_path.unlink()
    
    # Eliminar archivos temporales
    for temp in temp_files:
        file_path = repo_path / temp
        if file_path.exists():
            if file_path.is_dir():
                print(f"Eliminando directorio temporal: {temp}")
                shutil.rmtree(file_path)
            else:
                print(f"Eliminando archivo temporal: {temp}")
                file_path.unlink()
    
    print("✅ Limpieza completada")

if __name__ == "__main__":
    input("Presiona Enter para continuar con la limpieza (Ctrl+C para cancelar)...")
    main()
