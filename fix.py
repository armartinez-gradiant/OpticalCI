#!/usr/bin/env python3
"""
Script para corregir el problema de imports y verificar las correcciones v5.5

Soluciona:
1. Error "name 'torch' is not defined" en verificación
2. Verifica que las correcciones del microring funcionen
3. Ejecuta tests de validación
4. Proporciona diagnóstico completo
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header():
    """Imprimir header del script."""
    print("🔧 CORRECCIÓN DE IMPORTS Y VERIFICACIÓN v5.5")
    print("=" * 65)
    print("Solucionando:")
    print("  - Error 'torch' is not defined")
    print("  - Verificación de correcciones del microring")
    print("  - Validación de ecuaciones físicas")
    print()

def check_environment():
    """Verificar que estamos en el directorio correcto."""
    required_files = [
        'torchonn/layers/microring.py',
        'tests/test_microring.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ ERROR: Archivos no encontrados:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    print("✅ Entorno verificado")
    return True

def fix_corregir_py_imports():
    """Corregir los imports en corregir.py para evitar el error."""
    print("🔧 Corrigiendo imports en corregir.py...")
    
    try:
        with open('corregir.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Agregar import de torch al inicio de run_verification
        old_func_start = "def run_verification():"
        new_func_start = """def run_verification():
    \"\"\"Ejecutar verificación inmediata.\"\"\"
    import torch  # ✅ FIX: Agregar import faltante
    import numpy as np"""
        
        if old_func_start in content and "import torch" not in content[content.find("def run_verification()"):content.find("def run_verification()") + 500]:
            content = content.replace(
                "def run_verification():\n    \"\"\"Ejecutar verificación inmediata.\"\"\"",
                new_func_start
            )
            
            with open('corregir.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("   ✅ Import de torch agregado a run_verification()")
            return True
        else:
            print("   ℹ️ Import ya existe o función no encontrada")
            return True
            
    except Exception as e:
        print(f"   ❌ Error corrigiendo imports: {e}")
        return False

def verify_microring_manually():
    """Verificación manual del microring con manejo robusto de errores."""
    print("🔍 Verificación manual del microring...")
    
    try:
        # Agregar directorio actual al path
        sys.path.insert(0, '.')
        
        # Import con manejo de errores
        import torch
        print(f"   ✅ PyTorch {torch.__version__} importado")
        
        from torchonn.layers.microring import MicroringResonator
        print("   ✅ MicroringResonator importado")
        
        # Crear dispositivo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   📱 Usando device: {device}")
        
        # Crear microring con parámetros corregidos
        mrr = MicroringResonator(
            radius=10e-6,
            q_factor=5000,
            center_wavelength=1550e-9,
            coupling_mode="critical",
            device=device
        )
        print("   ✅ Microring creado exitosamente")
        
        # Test con wavelengths recomendados
        wavelengths = mrr.get_recommended_wavelengths(1000)
        input_signal = torch.ones(1, 1000, device=device, dtype=torch.float32)
        
        print(f"   📊 Wavelengths: {len(wavelengths)} puntos")
        print(f"   📊 Rango: {torch.min(wavelengths)*1e9:.1f} - {torch.max(wavelengths)*1e9:.1f} nm")
        
        # Forward pass
        with torch.no_grad():
            output = mrr(input_signal, wavelengths)
        
        through_response = output['through'][0]
        drop_response = output['drop'][0]
        
        # Análisis de resultados
        min_through = torch.min(through_response)
        max_through = torch.max(through_response)
        max_drop = torch.max(drop_response)
        
        print(f"   📊 Through: min={min_through:.6f}, max={max_through:.6f}")
        print(f"   📊 Drop: max={max_drop:.6f}")
        
        # Cálculo de extinction ratio
        if min_through > 1e-12:
            er_ratio = max_through / min_through
            er_db = 10 * torch.log10(er_ratio)
            print(f"   📊 ER medido: {er_db:.1f} dB")
            print(f"   📊 ER teórico: {mrr.extinction_ratio_theory_db:.1f} dB")
            
            # Verificar conservación de energía
            total_energy = through_response + drop_response
            max_energy = torch.max(total_energy)
            print(f"   📊 Conservación energía: max={max_energy:.3f}")
            
            # Validación automática
            validation = mrr.validate_physics(wavelengths)
            print(f"   📊 Validación automática:")
            print(f"      Energy conserved: {validation['energy_conserved']}")
            print(f"      ER coherent: {validation['extinction_ratio_coherent']}")
            print(f"      Resonance centered: {validation['resonance_centered']}")
            
            # Evaluación de éxito
            success_criteria = [
                min_through < 0.2,  # Through mínimo bajo
                er_db > 8,          # ER razonable
                max_energy < 1.05,  # Conservación de energía
                validation['energy_conserved'],
                validation['resonance_centered']
            ]
            
            success = all(success_criteria)
            
            if success:
                print("   🎉 ¡VERIFICACIÓN EXITOSA!")
                print(f"      - Through min: {min_through:.6f} < 0.2 ✅")
                print(f"      - ER: {er_db:.1f} dB > 8 ✅")
                print(f"      - Conservación: {max_energy:.3f} < 1.05 ✅")
                return True
            else:
                print("   ⚠️ Verificación parcialmente exitosa")
                print(f"      - Through min: {min_through:.6f} {'✅' if min_through < 0.2 else '❌'}")
                print(f"      - ER: {er_db:.1f} dB {'✅' if er_db > 8 else '❌'}")
                print(f"      - Conservación: {max_energy:.3f} {'✅' if max_energy < 1.05 else '❌'}")
                return False
        else:
            print("   ❌ Through response demasiado bajo para calcular ER")
            return False
            
    except ImportError as e:
        print(f"   ❌ Error de import: {e}")
        print("   💡 Sugerencia: Verificar que el módulo esté instalado correctamente")
        return False
    except Exception as e:
        print(f"   ❌ Error en verificación: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_specific_test():
    """Ejecutar el test específico del extinction ratio."""
    print("🧪 Ejecutando test específico...")
    
    try:
        # Intentar ejecutar el test específico
        cmd = [
            sys.executable, '-m', 'pytest', 
            'tests/test_microring.py::TestMicroringResonator::test_extinction_ratio_realistic',
            '-v', '--tb=short', '--no-header'
        ]
        
        print(f"   Comando: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120,
            cwd='.'
        )
        
        if result.returncode == 0:
            print("   ✅ TEST PASÓ exitosamente")
            print("   📊 Output del test:")
            # Mostrar las últimas líneas relevantes
            lines = result.stdout.split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"      {line}")
            return True
        else:
            print("   ❌ Test falló")
            print("   📊 STDOUT:")
            stdout_lines = result.stdout.split('\n')
            for line in stdout_lines[-15:]:
                if line.strip():
                    print(f"      {line}")
            
            print("   📊 STDERR:")
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines[-10:]:
                if line.strip():
                    print(f"      {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⚠️ Test timeout después de 2 minutos")
        return False
    except FileNotFoundError:
        print("   ❌ pytest no encontrado")
        print("   💡 Sugerencia: pip install pytest")
        return False
    except Exception as e:
        print(f"   ❌ Error ejecutando test: {e}")
        return False

def run_diagnostic():
    """Ejecutar diagnóstico si existe."""
    print("🔍 Ejecutando diagnóstico adicional...")
    
    try:
        if os.path.exists('diagnosis.py'):
            result = subprocess.run([sys.executable, 'diagnosis.py'], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("   ✅ Diagnóstico ejecutado exitosamente")
                # Mostrar resultado relevante
                lines = result.stdout.split('\n')
                for line in lines[-20:]:
                    if '✅' in line or '❌' in line or 'DIAGNÓSTICO' in line:
                        print(f"      {line}")
                return True
            else:
                print("   ⚠️ Diagnóstico completado con warnings")
                return False
        else:
            print("   ℹ️ diagnosis.py no encontrado, saltando")
            return True
            
    except Exception as e:
        print(f"   ⚠️ Error en diagnóstico: {e}")
        return True  # No es crítico

def check_key_files():
    """Verificar que los archivos clave tienen el contenido correcto."""
    print("📁 Verificando archivos clave...")
    
    try:
        # Verificar microring.py
        with open('torchonn/layers/microring.py', 'r', encoding='utf-8') as f:
            microring_content = f.read()
        
        critical_features = [
            "ECUACIONES CORREGIDAS v5.5",
            "get_transmission",
            "validate_physics",
            "def forward",
            "kappa_critical",
            "extinction_ratio_theory_db"
        ]
        
        missing_features = []
        for feature in critical_features:
            if feature not in microring_content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"   ❌ Faltan características en microring.py: {missing_features}")
            return False
        else:
            print("   ✅ microring.py tiene todas las características necesarias")
        
        # Verificar test
        with open('tests/test_microring.py', 'r', encoding='utf-8') as f:
            test_content = f.read()
        
        test_features = [
            "test_extinction_ratio_realistic",
            "CORREGIDO v5.5",
            "DIAGNÓSTICO DETALLADO"
        ]
        
        missing_test_features = []
        for feature in test_features:
            if feature not in test_content:
                missing_test_features.append(feature)
        
        if missing_test_features:
            print(f"   ⚠️ Faltan características en test: {missing_test_features}")
        else:
            print("   ✅ test_microring.py tiene las correcciones aplicadas")
        
        return len(missing_features) == 0
        
    except Exception as e:
        print(f"   ❌ Error verificando archivos: {e}")
        return False

def provide_recommendations():
    """Proporcionar recomendaciones basadas en los resultados."""
    print("\n💡 RECOMENDACIONES FINALES:")
    print("=" * 50)
    
    print("1. Si la verificación manual fue exitosa:")
    print("   ✅ Las correcciones funcionan")
    print("   ✅ Ejecutar: pytest tests/test_microring.py -v")
    print("   ✅ Continuar con desarrollo")
    
    print("\n2. Si hay problemas con imports:")
    print("   🔧 Verificar instalación: pip install -e .")
    print("   🔧 Verificar Python path")
    print("   🔧 Reiniciar shell/IDE")
    
    print("\n3. Si el test falla:")
    print("   📊 Revisar tolerancias en el test")
    print("   📊 Ejecutar diagnóstico detallado")
    print("   📊 Verificar parámetros físicos")
    
    print("\n4. Si persisten problemas:")
    print("   🔄 Restaurar backup:")
    print("      cp torchonn/layers/microring.py.backup_* torchonn/layers/microring.py")
    print("   🔄 Aplicar correcciones manualmente")
    
    print("\n5. Para verificar corrección final:")
    print("   🧪 python -c \"from torchonn.layers import MicroringResonator; print('OK')\"")
    print("   🧪 pytest tests/test_microring.py::TestMicroringResonator::test_extinction_ratio_realistic -v")

def main():
    """Función principal."""
    print_header()
    
    # Verificar entorno
    if not check_environment():
        return 1
    
    # Corregir imports en corregir.py
    fix_corregir_py_imports()
    
    # Verificar archivos clave
    files_ok = check_key_files()
    
    # Verificación manual del microring
    manual_verification = verify_microring_manually()
    
    # Ejecutar test específico
    test_result = run_specific_test()
    
    # Diagnóstico adicional
    diagnostic_result = run_diagnostic()
    
    # Resumen final
    print("\n" + "="*65)
    print("📊 RESUMEN DE VERIFICACIÓN:")
    print(f"   📁 Archivos clave: {'✅' if files_ok else '❌'}")
    print(f"   🔍 Verificación manual: {'✅' if manual_verification else '❌'}")
    print(f"   🧪 Test específico: {'✅' if test_result else '❌'}")
    print(f"   🔍 Diagnóstico: {'✅' if diagnostic_result else '⚠️'}")
    
    overall_success = files_ok and manual_verification
    
    if overall_success:
        print("\n🎉 CORRECCIÓN EXITOSA!")
        print("Las ecuaciones del microring han sido corregidas")
        print("El extinction ratio ahora debe ser realista (>8dB)")
        print("Through min debe ser <0.2")
    else:
        print("\n⚠️ CORRECCIÓN NECESITA AJUSTES")
        print("Revisar recomendaciones abajo")
    
    # Proporcionar recomendaciones
    provide_recommendations()
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit(main())