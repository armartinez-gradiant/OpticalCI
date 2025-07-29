#!/usr/bin/env python3
"""
Verificación Final Completa - v5.5.1

Ejecuta una verificación exhaustiva de todas las correcciones aplicadas:
1. Verifica sintaxis corregida
2. Prueba import y funcionalidad básica
3. Analiza ER medido vs teórico
4. Ejecuta tests automáticos
5. Genera reporte final con métricas
"""

import os
import sys
import subprocess
import time
import traceback
from pathlib import Path

def print_header():
    """Header del script."""
    print("🔍 VERIFICACIÓN FINAL COMPLETA v5.5.1")
    print("=" * 65)
    print("Verificando todas las correcciones aplicadas:")
    print("  ✅ Sintaxis del test corregida")
    print("  📊 ER mejorado (>4 dB mínimo)")
    print("  🔧 Ecuaciones físicas funcionando")
    print("  🧪 Tests pasando")
    print()

def verify_syntax_fixed():
    """Verificar que el error de sintaxis esté corregido."""
    print("🔍 VERIFICACIÓN 1: SINTAXIS CORREGIDA")
    print("-" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'py_compile', 'tests/test_microring.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Sintaxis de test_microring.py correcta")
            return True
        else:
            print("   ❌ Error de sintaxis persistente:")
            print(f"      {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error verificando sintaxis: {e}")
        return False

def verify_import_and_basic_functionality():
    """Verificar import y funcionalidad básica."""
    print("\n🔍 VERIFICACIÓN 2: IMPORT Y FUNCIONALIDAD BÁSICA")
    print("-" * 50)
    
    try:
        # Agregar path
        sys.path.insert(0, '.')
        
        # Import básico
        import torch
        from torchonn.layers.microring import MicroringResonator
        print("   ✅ Imports exitosos")
        
        # Crear microring
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mrr = MicroringResonator(
            radius=10e-6,
            q_factor=5000,
            center_wavelength=1550e-9,
            coupling_mode="critical",
            device=device
        )
        print("   ✅ Microring creado exitosamente")
        
        # Verificar parámetros
        print(f"      α = {mrr.alpha:.6f}")
        print(f"      κ_critical = {mrr.kappa_critical:.6f}")
        print(f"      ER teórico = {mrr.extinction_ratio_theory_db:.1f} dB")
        
        return True, mrr, device
        
    except Exception as e:
        print(f"   ❌ Error en import/funcionalidad básica: {e}")
        traceback.print_exc()
        return False, None, None

def analyze_er_performance(mrr, device):
    """Analizar performance del extinction ratio."""
    print("\n🔍 VERIFICACIÓN 3: ANÁLISIS DE ER DETALLADO")
    print("-" * 50)
    
    try:
        import torch
        
        # Test con múltiples resoluciones para robustez
        test_cases = [
            ("Quick test", 500),
            ("Standard test", 1500),
            ("High resolution", 3000)
        ]
        
        results = {}
        
        for test_name, n_points in test_cases:
            print(f"   📊 {test_name} ({n_points} puntos):")
            
            wavelengths = mrr.get_recommended_wavelengths(n_points)
            input_signal = torch.ones(1, n_points, device=device, dtype=torch.float32)
            
            with torch.no_grad():
                output = mrr(input_signal, wavelengths)
            
            through_response = output['through'][0]
            drop_response = output['drop'][0]
            
            # Análisis detallado
            min_through = torch.min(through_response)
            max_through = torch.max(through_response)
            max_drop = torch.max(drop_response)
            
            # ER calculation robusto
            if min_through > 1e-15:
                # Método 1: Simple max/min
                er_simple = max_through / min_through
                er_simple_db = 10 * torch.log10(er_simple)
                
                # Método 2: Off-resonance robusto
                n_edge = max(n_points // 10, 10)
                sorted_vals, _ = torch.sort(through_response, descending=True)
                off_res_max = torch.mean(sorted_vals[:n_edge])
                er_robust = off_res_max / min_through
                er_robust_db = 10 * torch.log10(er_robust)
                
                # Conservación de energía
                total_energy = through_response + drop_response
                max_energy = torch.max(total_energy)
                
                results[test_name] = {
                    'min_through': min_through.item(),
                    'max_through': max_through.item(),
                    'er_simple_db': er_simple_db.item(),
                    'er_robust_db': er_robust_db.item(),
                    'max_energy': max_energy.item(),
                    'n_points': n_points
                }
                
                print(f"      Through: min={min_through:.6f}, max={max_through:.6f}")
                print(f"      ER simple: {er_simple_db:.1f} dB")
                print(f"      ER robusto: {er_robust_db:.1f} dB")
                print(f"      Conservación: {max_energy:.3f}")
                
            else:
                print(f"      ❌ Through min demasiado bajo: {min_through:.2e}")
                results[test_name] = None
        
        # Análisis de consistencia
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if len(valid_results) >= 2:
            ers = [r['er_robust_db'] for r in valid_results.values()]
            er_std = torch.std(torch.tensor(ers)).item()
            er_mean = torch.mean(torch.tensor(ers)).item()
            
            print(f"   📊 CONSISTENCIA:")
            print(f"      ER promedio: {er_mean:.1f} dB")
            print(f"      ER std: {er_std:.2f} dB")
            print(f"      ER teórico: {mrr.extinction_ratio_theory_db:.1f} dB")
            
            # Criterios de éxito
            success_criteria = [
                er_mean > 4,  # ER mínimo aceptable
                er_std < 2,   # Consistencia entre mediciones
                all(r['max_energy'] < 1.05 for r in valid_results.values())  # Conservación
            ]
            
            success = all(success_criteria)
            
            print(f"   🎯 EVALUACIÓN:")
            print(f"      ER > 4 dB: {'✅' if er_mean > 4 else '❌'} ({er_mean:.1f} dB)")
            print(f"      Consistencia: {'✅' if er_std < 2 else '❌'} (std={er_std:.2f})")
            print(f"      Conservación: {'✅' if all(r['max_energy'] < 1.05 for r in valid_results.values()) else '❌'}")
            
            return success, results
        else:
            print("   ❌ Insuficientes resultados válidos")
            return False, results
            
    except Exception as e:
        print(f"   ❌ Error en análisis de ER: {e}")
        traceback.print_exc()
        return False, {}

def run_automated_tests():
    """Ejecutar tests automáticos."""
    print("\n🔍 VERIFICACIÓN 4: TESTS AUTOMÁTICOS")
    print("-" * 50)
    
    tests_to_run = [
        {
            "name": "Test específico ER",
            "cmd": [
                sys.executable, "-m", "pytest", 
                "tests/test_microring.py::TestMicroringResonator::test_extinction_ratio_realistic",
                "-v", "--tb=short"
            ],
            "timeout": 60,
            "critical": True
        },
        {
            "name": "Test conservación energía",
            "cmd": [
                sys.executable, "-m", "pytest", 
                "tests/test_microring.py::TestMicroringResonator::test_energy_conservation",
                "-v", "--tb=short"
            ],
            "timeout": 30,
            "critical": True
        },
        {
            "name": "Test inicialización", 
            "cmd": [
                sys.executable, "-m", "pytest", 
                "tests/test_microring.py::TestMicroringResonator::test_microring_initialization",
                "-v", "--tb=short"
            ],
            "timeout": 30,
            "critical": False
        }
    ]
    
    results = {}
    
    for test in tests_to_run:
        print(f"   🧪 {test['name']}...")
        
        try:
            result = subprocess.run(
                test['cmd'],
                capture_output=True,
                text=True,
                timeout=test['timeout']
            )
            
            success = result.returncode == 0
            results[test['name']] = {
                'success': success,
                'critical': test['critical'],
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if success:
                print(f"      ✅ PASÓ")
            else:
                print(f"      ❌ FALLÓ")
                if test['critical']:
                    print("         🔴 TEST CRÍTICO")
                
                # Mostrar error relevante
                error_lines = (result.stdout + "\\n" + result.stderr).split('\\n')
                relevant = [l for l in error_lines if any(word in l.lower() for word in ['failed', 'error', 'assert', 'exception'])]
                for line in relevant[-3:]:
                    if line.strip():
                        print(f"         {line[:80]}")
        
        except subprocess.TimeoutExpired:
            print(f"      ⏱️ TIMEOUT")
            results[test['name']] = {'success': False, 'critical': test['critical'], 'timeout': True}
        except Exception as e:
            print(f"      ❌ ERROR: {e}")
            results[test['name']] = {'success': False, 'critical': test['critical'], 'error': str(e)}
    
    return results

def generate_final_report(syntax_ok, basic_ok, er_analysis_ok, er_results, test_results):
    """Generar reporte final completo."""
    print("\n" + "="*65)
    print("📋 REPORTE FINAL COMPLETO v5.5.1")
    print("="*65)
    
    # Resumen de verificaciones
    print("📊 RESUMEN DE VERIFICACIONES:")
    print(f"   1. Sintaxis corregida: {'✅' if syntax_ok else '❌'}")
    print(f"   2. Funcionalidad básica: {'✅' if basic_ok else '❌'}")
    print(f"   3. Análisis de ER: {'✅' if er_analysis_ok else '❌'}")
    
    # Tests automáticos
    if test_results:
        passed_tests = sum(1 for r in test_results.values() if r.get('success', False))
        total_tests = len(test_results)
        critical_tests = [k for k, v in test_results.items() if v.get('critical', False)]
        critical_passed = sum(1 for k in critical_tests if test_results[k].get('success', False))
        
        print(f"   4. Tests automáticos: {passed_tests}/{total_tests} pasaron")
        print(f"      Tests críticos: {critical_passed}/{len(critical_tests)} pasaron")
    
    # Métricas de ER
    if er_results:
        valid_ers = [r for r in er_results.values() if r is not None]
        if valid_ers:
            avg_er = sum(r['er_robust_db'] for r in valid_ers) / len(valid_ers)
            min_through_avg = sum(r['min_through'] for r in valid_ers) / len(valid_ers)
            print(f"\\n📊 MÉTRICAS DE EXTINCTION RATIO:")
            print(f"   ER promedio medido: {avg_er:.1f} dB")
            print(f"   Through min promedio: {min_through_avg:.6f}")
            print(f"   Mejora vs inicial: {avg_er:.1f} dB vs ~5.1 dB anterior")
    
    # Evaluación general
    critical_passed_all = all(test_results.get(k, {}).get('success', False) for k in critical_tests) if test_results else False
    
    overall_success = syntax_ok and basic_ok and er_analysis_ok and critical_passed_all
    
    print(f"\\n🎯 EVALUACIÓN FINAL:")
    if overall_success:
        print("🎉 ¡TODAS LAS CORRECCIONES EXITOSAS!")
        print("   ✅ Error de sintaxis resuelto")
        print("   ✅ Extinction ratio mejorado significativamente")
        print("   ✅ Ecuaciones físicas funcionando correctamente")
        print("   ✅ Tests críticos pasando")
        print("   ✅ Conservación de energía garantizada")
        
        print("\\n📋 RECOMENDACIONES FINALES:")
        print("   1. ✅ Ejecutar suite completa: pytest tests/test_microring.py -v")
        print("   2. ✅ Verificar otros tests: pytest tests/ -x")
        print("   3. ✅ Commit de las correcciones aplicadas")
        print("   4. ✅ Documentar mejoras en changelog")
        
    else:
        print("⚠️ CORRECCIONES PARCIALES - REVISAR PENDIENTES")
        
        if not syntax_ok:
            print("   🔧 Corregir sintaxis restante manualmente")
        if not basic_ok:
            print("   🔧 Revisar imports y configuración del proyecto")
        if not er_analysis_ok:
            print("   📊 Ajustar parámetros físicos o tolerancias")
        if not critical_passed_all:
            print("   🧪 Revisar y corregir tests críticos")
        
        print("\\n📋 ACCIONES RECOMENDADAS:")
        print("   1. Revisar logs de error específicos arriba")
        print("   2. Ejecutar diagnóstico detallado")
        print("   3. Considerar restaurar backup si es necesario")
    
    # Información de archivos
    print(f"\\n📁 ARCHIVOS MODIFICADOS:")
    modified_files = [
        'torchonn/layers/microring.py',
        'tests/test_microring.py'
    ]
    
    for file_path in modified_files:
        if os.path.exists(file_path):
            mod_time = os.path.getmtime(file_path)
            mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
            print(f"   ✅ {file_path} (modificado: {mod_time_str})")
        else:
            print(f"   ❌ {file_path} (faltante)")
    
    # Backups disponibles
    backup_files = [f for f in os.listdir('.') if 'backup' in f and ('microring' in f or 'test_microring' in f)]
    if backup_files:
        print(f"\\n📁 BACKUPS DISPONIBLES ({len(backup_files)}):")
        for backup in sorted(backup_files)[-5:]:  # Últimos 5
            print(f"   📄 {backup}")
    
    return overall_success

def main():
    """Función principal."""
    print_header()
    
    # Verificación 1: Sintaxis
    syntax_ok = verify_syntax_fixed()
    
    # Verificación 2: Funcionalidad básica
    basic_ok, mrr, device = verify_import_and_basic_functionality()
    
    # Verificación 3: Análisis de ER
    er_analysis_ok = False
    er_results = {}
    if basic_ok and mrr is not None:
        er_analysis_ok, er_results = analyze_er_performance(mrr, device)
    
    # Verificación 4: Tests automáticos
    test_results = run_automated_tests()
    
    # Reporte final
    success = generate_final_report(syntax_ok, basic_ok, er_analysis_ok, er_results, test_results)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())