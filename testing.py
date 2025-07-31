#!/usr/bin/env python3
"""
🎯 PLAN DE ACCIÓN INMEDIATO - POST FIX DE AYER
=============================================

Script para verificar rápidamente el estado actual de tu implementación
después de los cambios que hiciste ayer con fix.py.
"""

import os
import torch
import time
import sys
from pathlib import Path

def quick_diagnostic():
    """Diagnóstico rápido del estado actual."""
    print("🔬 DIAGNÓSTICO RÁPIDO")
    print("=" * 20)
    
    results = {}
    
    # 1. Check archivos
    main_file = "torchonn/onns/architectures/coherent_onn.py"
    if os.path.exists(main_file):
        stat = os.stat(main_file)
        mod_time = time.ctime(stat.st_mtime)
        print(f"✅ coherent_onn.py encontrado (mod: {mod_time})")
        results['file_exists'] = True
    else:
        print("❌ coherent_onn.py NO encontrado")
        results['file_exists'] = False
        return results
    
    # 2. Test imports básicos
    try:
        print("🔍 Testing imports...")
        from torchonn.onns.benchmarks.mnist_optical import OpticalMNIST
        from torchonn.onns.architectures.coherent_onn import CoherentONN
        print("✅ Imports OK")
        results['imports_ok'] = True
    except Exception as e:
        print(f"❌ Import error: {e}")
        results['imports_ok'] = False
        return results
    
    # 3. Test creación ONN
    try:
        print("🔍 Testing ONN creation...")
        benchmark = OpticalMNIST(image_size=4, n_classes=3)
        onn = benchmark.create_coherent_onn()
        print("✅ ONN creation OK")
        results['creation_ok'] = True
    except Exception as e:
        print(f"❌ ONN creation error: {e}")
        results['creation_ok'] = False
        return results
    
    # 4. Test forward pass crítico
    try:
        print("🔍 Testing forward pass...")
        with torch.no_grad():
            x = torch.randn(4, 16) * 0.2  # Conservative input
            output = onn(x)
            
            has_nan = torch.any(torch.isnan(output)).item()
            has_inf = torch.any(torch.isinf(output)).item()
            output_mean = torch.mean(torch.abs(output)).item()
            
            print(f"   📊 Output shape: {output.shape}")
            print(f"   📊 Output mean abs: {output_mean:.4f}")
            
            if has_nan:
                print("   ❌ CRITICAL: NaN detected!")
                results['nan_issue'] = True
            elif has_inf:
                print("   ❌ CRITICAL: Inf detected!")
                results['inf_issue'] = True
            else:
                print("   ✅ Forward pass OK (no NaN/Inf)")
                results['forward_ok'] = True
                
    except Exception as e:
        print(f"   ❌ Forward pass error: {e}")
        results['forward_error'] = str(e)
    
    # 5. Check tipo de datos
    try:
        print("🔍 Checking training data...")
        train_batch = next(iter(benchmark.train_loader))
        x_data, y_data = train_batch
        
        data_std = torch.std(x_data).item()
        n_train = len(benchmark.train_loader.dataset)
        
        print(f"   📊 Training samples: {n_train}")
        print(f"   📊 Data std: {data_std:.4f}")
        
        if data_std > 0.4:
            print("   ⚠️ Likely synthetic random data")
            results['data_type'] = 'synthetic'
        elif data_std > 0.2:
            print("   🎯 Likely structured data")
            results['data_type'] = 'structured'
        else:
            print("   📚 Likely real MNIST")
            results['data_type'] = 'real'
            
    except Exception as e:
        print(f"   ❌ Data check error: {e}")
        results['data_error'] = str(e)
    
    return results

def recommend_action(results):
    """Recomendar acción basándose en resultados."""
    print("\n🎯 RECOMENDACIÓN:")
    print("=" * 15)
    
    if not results.get('file_exists'):
        print("❌ CRÍTICO: Archivo principal perdido")
        print("🔧 ACCIÓN: Restaurar desde backup")
        return "restore_backup"
    
    if not results.get('imports_ok'):
        print("❌ CRÍTICO: Problemas de import")
        print("🔧 ACCIÓN: Verificar paths y imports")
        return "fix_imports"
    
    if not results.get('creation_ok'):
        print("❌ CRÍTICO: No se pode crear ONN")
        print("🔧 ACCIÓN: Revisar arquitectura")
        return "fix_architecture"
    
    if results.get('nan_issue') or results.get('inf_issue'):
        print("❌ CRÍTICO: Problemas NaN/Inf persisten")
        print("🔧 ACCIÓN: Ejecutar fix.py o restaurar backup estable")
        return "fix_nan_issues"
    
    if results.get('forward_ok'):
        print("✅ BÁSICO: Forward pass funciona")
        
        if results.get('data_type') == 'synthetic':
            print("🎯 SIGUIENTE: Implementar MNIST real (mejorará accuracy)")
            return "implement_mnist"
        elif results.get('data_type') == 'structured':
            print("🎯 SIGUIENTE: Optimizar entrenamiento (fix epochs + init)")
            return "optimize_training"
        else:
            print("🎯 SIGUIENTE: Test completo de entrenamiento")
            return "full_training_test"
    
    print("❓ ESTADO: Incierto, hacer diagnóstico completo")
    return "full_diagnostic"

def provide_specific_commands(action):
    """Proporcionar comandos específicos."""
    print(f"\n🚀 COMANDOS ESPECÍFICOS:")
    print("=" * 22)
    
    commands = {
        'restore_backup': [
            "# Lista backups disponibles:",
            "ls -la torchonn/onns/architectures/coherent_onn.py.*",
            "",
            "# Restaurar el más reciente estable:",
            "cp torchonn/onns/architectures/coherent_onn.py.backup_20250730_114855 torchonn/onns/architectures/coherent_onn.py"
        ],
        
        'fix_nan_issues': [
            "# Opción 1: Ejecutar tu fix.py",
            "python fix.py",
            "",
            "# Opción 2: Restaurar backup + aplicar mis fixes",
            "# (Te doy el código específico)"
        ],
        
        'implement_mnist': [
            "# Test actual funcionamiento:",
            "python demos/demo_onn.py --quick",
            "",
            "# Si funciona pero accuracy ~10%:",
            "# Implementar MNIST real (te doy código específico)"
        ],
        
        'optimize_training': [
            "# Test entrenamiento completo:",
            "python demos/demo_onn.py --size 8 --epochs 10",
            "",
            "# Fix display epochs + inicialización"
        ],
        
        'full_training_test': [
            "# Test completo:",
            "python demos/demo_onn.py --size 8 --epochs 10",
            "",
            "# Monitorear accuracy y convergencia"
        ]
    }
    
    if action in commands:
        for cmd in commands[action]:
            if cmd.startswith('#'):
                print(f"   💡 {cmd}")
            elif cmd == "":
                print()
            else:
                print(f"   $ {cmd}")
    else:
        print("   💡 Hacer diagnóstico completo manual")

def main():
    """Main execution."""
    print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    print("🌟  DIAGNÓSTICO POST-FIX DE AYER  🌟")
    print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    
    # Run diagnostic
    results = quick_diagnostic()
    
    # Get recommendation
    action = recommend_action(results)
    
    # Provide specific commands
    provide_specific_commands(action)
    
    print(f"\n💡 CONTEXTO:")
    print("   Tu fix.py de ayer fue muy inteligente")
    print("   Detecta automáticamente problemas y aplica fixes")
    print("   Si persisten issues, tengo fixes científicos específicos")
    
    print(f"\n🎯 OBJETIVO FINAL:")
    print("   CoherentONN → 40-60% accuracy (matrices unitarias)")
    print("   IncoherentONN → 70-80% accuracy (sin restricciones)")
    print("   HybridONN → 80-90% accuracy (combinar ambos)")

if __name__ == "__main__":
    main()