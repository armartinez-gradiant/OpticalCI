#!/usr/bin/env python3
"""
🌟 Demo Comparativo: CoherentONN vs IncoherentONN

Demostración completa de ambas arquitecturas de Redes Neuronales Ópticas.
Muestra las diferencias fundamentales, ventajas y casos de uso de cada una.

🎯 ARQUITECTURAS DEMOSTRADAS:
- CoherentONN: MZI mesh + matrices unitarias + conservación perfecta de energía
- IncoherentONN: Microring arrays + WDM + robustez a fabricación

🔬 COMPARACIONES:
- Principios físicos fundamentales
- Escalabilidad con número de wavelengths
- Robustez a variaciones de fabricación
- Eficiencia energética y velocidad
- Casos de uso recomendados

📚 Basado en: Shen et al. (2017) [Coherent] + Hughes et al. [Incoherent]

USO:
    python demo_incoherent_vs_coherent.py [--quick] [--wavelengths WL] [--size SIZE]
    
EJEMPLOS:
    python demo_incoherent_vs_coherent.py --quick              # Demo rápido
    python demo_incoherent_vs_coherent.py --wavelengths 8      # Escalar WDM
    python demo_incoherent_vs_coherent.py --size 16            # Problema más grande
"""

import argparse
import sys
import torch
import numpy as np
import time
import warnings
from typing import Dict, Any, List, Tuple

# Configurar warnings
warnings.filterwarnings("ignore", category=UserWarning)

def check_requirements():
    """Verificar que ambas arquitecturas están disponibles."""
    print("🔧 Checking requirements for comparative demo...")
    
    issues = []
    available_architectures = []
    
    # Check CoherentONN
    try:
        from torchonn.onns.architectures import CoherentONN
        available_architectures.append("CoherentONN")
        print(f"   ✅ CoherentONN available")
    except ImportError as e:
        issues.append(f"CoherentONN not available: {e}")
    
    # Check IncoherentONN  
    try:
        from torchonn.onns.architectures import IncoherentONN
        available_architectures.append("IncoherentONN") 
        print(f"   ✅ IncoherentONN available")
    except ImportError as e:
        issues.append(f"IncoherentONN not available: {e}")
    
    # Check PyTorch
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
    except ImportError:
        issues.append("PyTorch not available")
    
    if len(available_architectures) < 2:
        issues.append("Need both CoherentONN and IncoherentONN for comparison")
    
    if issues:
        print(f"\n❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False, available_architectures
    else:
        print(f"   ✅ All requirements satisfied for comparative demo")
        return True, available_architectures


def demo_architecture_principles():
    """Demo 1: Principios fundamentales de cada arquitectura."""
    print("\n" + "="*60)
    print("🔬 DEMO 1: Architectural Principles Comparison")
    print("="*60)
    
    try:
        from torchonn.onns.architectures import CoherentONN, IncoherentONN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🖥️ Device: {device}")
        
        # Crear arquitecturas para comparación
        layer_sizes = [8, 8, 4]  # Usar cuadradas para CoherentONN
        
        print(f"\n🏗️ Creating architectures with layers: {layer_sizes}")
        
        # CoherentONN
        print(f"\n🌟 CoherentONN Principles:")
        coherent_onn = CoherentONN(layer_sizes=layer_sizes, device=device)
        
        print(f"   📐 Architecture: Unitary matrices via MZI mesh")
        print(f"   ⚡ Operation: Phase-sensitive interferometry")
        print(f"   🔋 Energy: Perfect conservation (unitarity)")
        print(f"   🎯 Precision: High (complex-valued)")
        print(f"   🔧 Fabrication: Requires precise phase control")
        
        # IncoherentONN  
        print(f"\n💫 IncoherentONN Principles:")
        incoherent_onn = IncoherentONN(
            layer_sizes=layer_sizes, 
            n_wavelengths=4, 
            device=device
        )
        
        print(f"   📐 Architecture: Microring resonator arrays")
        print(f"   ⚡ Operation: Intensity-based (incoherent)")
        print(f"   🔋 Energy: Allows realistic losses")
        print(f"   🎯 Precision: Moderate (positive weights)")
        print(f"   🔧 Fabrication: More tolerant to variations")
        print(f"   🌈 WDM: {incoherent_onn.n_wavelengths} wavelength channels")
        
        # Física validation comparison
        print(f"\n🔬 Physics Validation:")
        
        coherent_physics = coherent_onn.validate_unitarity()
        incoherent_physics = incoherent_onn.validate_physics()
        
        print(f"   CoherentONN - Unitarity check: {'✅' if coherent_physics.get('overall_valid', False) else '❌'}")
        print(f"   IncoherentONN - Transmission check: {'✅' if incoherent_physics.get('valid_transmissions', False) else '❌'}")
        print(f"   IncoherentONN - Energy conservation: {incoherent_physics.get('energy_conservation_type', 'N/A')}")
        print(f"   IncoherentONN - Allows losses: {'✅' if incoherent_physics.get('allows_energy_loss', False) else '❌'}")
        
        return {
            "coherent_created": True,
            "incoherent_created": True,
            "coherent_physics": coherent_physics,
            "incoherent_physics": incoherent_physics
        }
        
    except Exception as e:
        print(f"❌ Architecture principles demo failed: {e}")
        return {"error": str(e)}


def demo_forward_pass_comparison():
    """Demo 2: Comparación de forward passes."""
    print("\n" + "="*60)
    print("🚀 DEMO 2: Forward Pass & Performance Comparison")
    print("="*60)
    
    try:
        from torchonn.onns.architectures import CoherentONN, IncoherentONN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configuración
        layer_sizes = [6, 6, 4]
        batch_size = 16
        n_wavelengths = 4
        
        print(f"📊 Test Configuration:")
        print(f"   Input size: {layer_sizes[0]}")
        print(f"   Output size: {layer_sizes[-1]}")
        print(f"   Batch size: {batch_size}")
        print(f"   Wavelengths (IncoherentONN): {n_wavelengths}")
        
        # Crear modelos
        coherent_onn = CoherentONN(layer_sizes=layer_sizes, device=device)
        incoherent_onn = IncoherentONN(
            layer_sizes=layer_sizes, 
            n_wavelengths=n_wavelengths, 
            device=device
        )
        
        # Datos de entrada idénticos
        x = torch.randn(batch_size, layer_sizes[0], device=device) * 0.5
        
        print(f"\n⚡ Forward Pass Performance:")
        
        # CoherentONN timing
        start_time = time.time()
        with torch.no_grad():
            y_coherent = coherent_onn(x)
        coherent_time = time.time() - start_time
        
        print(f"   CoherentONN: {coherent_time*1000:.2f}ms")
        print(f"   - Output shape: {y_coherent.shape}")
        print(f"   - Output range: [{torch.min(y_coherent):.3f}, {torch.max(y_coherent):.3f}]")
        
        # IncoherentONN timing
        start_time = time.time()
        with torch.no_grad():
            y_incoherent = incoherent_onn(x)
        incoherent_time = time.time() - start_time
        
        print(f"   IncoherentONN: {incoherent_time*1000:.2f}ms")
        print(f"   - Output shape: {y_incoherent.shape}")
        print(f"   - Output range: [{torch.min(y_incoherent):.3f}, {torch.max(y_incoherent):.3f}]")
        
        # Análisis de diferencias
        if y_coherent.shape == y_incoherent.shape:
            diff = torch.norm(y_coherent - y_incoherent)
            print(f"   Output difference (L2): {diff:.6f}")
            print(f"   → Confirm different physics: {'✅' if diff > 1e-6 else '❌'}")
        
        # Métricas de eficiencia
        print(f"\n📈 Efficiency Metrics:")
        
        coherent_eff = coherent_onn.get_optical_efficiency()
        incoherent_eff = incoherent_onn.get_optical_efficiency_metrics()
        
        print(f"   CoherentONN optical fraction: {coherent_eff.get('optical_fraction', 0):.3f}")
        print(f"   IncoherentONN optical fraction: {incoherent_eff.get('optical_fraction', 0):.3f}")
        print(f"   CoherentONN theoretical speedup: {coherent_eff.get('theoretical_speedup', 1):.1f}x")
        print(f"   IncoherentONN theoretical speedup: {incoherent_eff.get('theoretical_speedup', 1):.1f}x")
        
        # Capacidades específicas
        print(f"\n🔍 Architecture-Specific Features:")
        print(f"   CoherentONN - Unitary operations: {'✅' if coherent_eff.get('unitary_operations', 0) > 0 else '❌'}")
        print(f"   IncoherentONN - WDM channels: {incoherent_eff.get('wavelength_channels', 0)}")
        print(f"   IncoherentONN - Parallel operations: {incoherent_eff.get('parallel_operations', 0)}")
        print(f"   IncoherentONN - Microring count: {incoherent_eff.get('total_microrings', 0)}")
        
        return {
            "coherent_time": coherent_time,
            "incoherent_time": incoherent_time,
            "coherent_efficiency": coherent_eff,
            "incoherent_efficiency": incoherent_eff,
            "output_difference": diff.item() if 'diff' in locals() else None
        }
        
    except Exception as e:
        print(f"❌ Forward pass comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def demo_wavelength_scaling():
    """Demo 3: Escalabilidad con wavelengths (específico de IncoherentONN)."""
    print("\n" + "="*60)
    print("🌈 DEMO 3: Wavelength Division Multiplexing Scaling")
    print("="*60)
    
    try:
        from torchonn.onns.architectures import IncoherentONN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layer_sizes = [4, 6, 3]
        batch_size = 8
        wavelength_counts = [1, 2, 4, 8, 16]
        
        print(f"🎯 Testing IncoherentONN scalability with WDM channels")
        print(f"   Architecture: {layer_sizes}")
        print(f"   Wavelength counts: {wavelength_counts}")
        
        results = []
        
        for n_wl in wavelength_counts:
            print(f"\n📡 Testing {n_wl} wavelength{'s' if n_wl > 1 else ''}:")
            
            try:
                # Crear IncoherentONN con diferentes números de wavelengths
                onn = IncoherentONN(
                    layer_sizes=layer_sizes,
                    n_wavelengths=n_wl,
                    device=device
                )
                
                # Test forward pass
                x = torch.randn(batch_size, layer_sizes[0], device=device)
                
                start_time = time.time()
                with torch.no_grad():
                    y = onn(x)
                forward_time = time.time() - start_time
                
                # Obtener métricas específicas
                efficiency = onn.get_optical_efficiency_metrics()
                
                print(f"   ✅ Forward time: {forward_time*1000:.2f}ms")
                print(f"   📊 Theoretical speedup: {efficiency.get('theoretical_speedup', 1):.1f}x")
                print(f"   🔄 Parallel operations: {efficiency.get('parallel_operations', 0)}")
                print(f"   💻 Microring count: {efficiency.get('total_microrings', 0)}")
                
                results.append({
                    "wavelengths": n_wl,
                    "forward_time": forward_time,
                    "speedup": efficiency.get('theoretical_speedup', 1),
                    "parallel_ops": efficiency.get('parallel_operations', 0),
                    "efficiency": efficiency
                })
                
            except Exception as e:
                print(f"   ❌ Failed for {n_wl} wavelengths: {e}")
                results.append({
                    "wavelengths": n_wl,
                    "error": str(e)
                })
        
        # Análisis de escalabilidad
        print(f"\n📈 SCALABILITY ANALYSIS:")
        
        valid_results = [r for r in results if "error" not in r]
        if len(valid_results) >= 2:
            speedups = [r["speedup"] for r in valid_results]
            wavelengths = [r["wavelengths"] for r in valid_results]
            
            print(f"   Wavelengths: {wavelengths}")
            print(f"   Speedups: {[f'{s:.1f}x' for s in speedups]}")
            
            # Verificar tendencia de escalabilidad
            is_scaling = all(speedups[i] <= speedups[i+1] for i in range(len(speedups)-1))
            print(f"   Scaling behavior: {'✅ Improving' if is_scaling else '⚠️ Not monotonic'}")
            
            # Mostrar ventaja de WDM
            if len(valid_results) > 1:
                max_speedup = max(speedups)
                min_speedup = min(speedups)
                wdm_advantage = max_speedup / min_speedup
                print(f"   WDM advantage: {wdm_advantage:.1f}x improvement from 1 to {max(wavelengths)} wavelengths")
        
        print(f"\n💡 IncoherentONN WDM Benefits:")
        print(f"   🌈 Parallel processing across wavelength channels")
        print(f"   📡 Natural scalability with telecom infrastructure")
        print(f"   🔧 Each wavelength can carry independent data")
        print(f"   ⚡ Theoretical speedup scales with channel count")
        
        return {"results": results, "valid_results": valid_results}
        
    except Exception as e:
        print(f"❌ Wavelength scaling demo failed: {e}")
        return {"error": str(e)}


def demo_robustness_comparison():
    """Demo 4: Comparación de robustez a variaciones."""
    print("\n" + "="*60)
    print("🛡️ DEMO 4: Fabrication Tolerance & Robustness")
    print("="*60)
    
    try:
        from torchonn.onns.architectures import CoherentONN, IncoherentONN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔬 Testing robustness to parameter variations")
        
        layer_sizes = [4, 4, 2]
        n_tests = 5
        noise_levels = [0.0, 0.01, 0.05, 0.1]  # Noise levels for testing
        
        coherent_results = []
        incoherent_results = []
        
        for noise_level in noise_levels:
            print(f"\n📊 Noise level: {noise_level*100:.1f}%")
            
            coherent_scores = []
            incoherent_scores = []
            
            for test_i in range(n_tests):
                try:
                    # Test CoherentONN
                    coherent_onn = CoherentONN(layer_sizes=layer_sizes, device=device)
                    
                    # Add noise to parameters (simulating fabrication variations)
                    if noise_level > 0:
                        with torch.no_grad():
                            for param in coherent_onn.parameters():
                                noise = torch.randn_like(param) * noise_level
                                param.add_(noise)
                    
                    # Test IncoherentONN
                    incoherent_onn = IncoherentONN(
                        layer_sizes=layer_sizes,
                        n_wavelengths=4,
                        device=device
                    )
                    
                    if noise_level > 0:
                        with torch.no_grad():
                            for param in incoherent_onn.parameters():
                                noise = torch.randn_like(param) * noise_level
                                param.add_(noise)
                    
                    # Test forward passes
                    x = torch.randn(4, layer_sizes[0], device=device)
                    
                    with torch.no_grad():
                        y_coherent = coherent_onn(x)
                        y_incoherent = incoherent_onn(x)
                    
                    # Simple scoring: check for NaN/Inf and reasonable output range
                    coherent_score = 1.0
                    if torch.any(torch.isnan(y_coherent)) or torch.any(torch.isinf(y_coherent)):
                        coherent_score = 0.0
                    elif torch.max(torch.abs(y_coherent)) > 100:  # Unreasonably large
                        coherent_score = 0.5
                    
                    incoherent_score = 1.0
                    if torch.any(torch.isnan(y_incoherent)) or torch.any(torch.isinf(y_incoherent)):
                        incoherent_score = 0.0
                    elif torch.max(torch.abs(y_incoherent)) > 100:
                        incoherent_score = 0.5
                    
                    coherent_scores.append(coherent_score)
                    incoherent_scores.append(incoherent_score)
                    
                except Exception as e:
                    coherent_scores.append(0.0)
                    incoherent_scores.append(0.0)
            
            # Calculate average scores
            coherent_avg = np.mean(coherent_scores)
            incoherent_avg = np.mean(incoherent_scores)
            
            coherent_results.append(coherent_avg)
            incoherent_results.append(incoherent_avg)
            
            print(f"   CoherentONN reliability: {coherent_avg*100:.1f}%")
            print(f"   IncoherentONN reliability: {incoherent_avg*100:.1f}%")
        
        # Analysis
        print(f"\n🏆 ROBUSTNESS SUMMARY:")
        print(f"   Noise Level | CoherentONN | IncoherentONN")
        print(f"   ------------|-------------|---------------")
        for i, noise in enumerate(noise_levels):
            coh_score = coherent_results[i] * 100
            inc_score = incoherent_results[i] * 100
            winner = "CoherentONN" if coh_score > inc_score else "IncoherentONN" if inc_score > coh_score else "Tie"
            print(f"   {noise*100:8.1f}%   |   {coh_score:6.1f}%   |    {inc_score:6.1f}%   | {winner}")
        
        # Theoretical advantages
        print(f"\n💡 Theoretical Robustness Factors:")
        print(f"   CoherentONN (MZI-based):")
        print(f"   + Mathematically elegant (unitary)")
        print(f"   - Sensitive to phase variations")
        print(f"   - Requires precise fabrication")
        print(f"   - Path length matching critical")
        
        print(f"\n   IncoherentONN (Microring-based):")
        print(f"   + Intensity-based (phase insensitive)")
        print(f"   + More tolerant to fabrication variations")
        print(f"   + Individual component tuning possible")
        print(f"   - Lower precision (positive weights only)")
        
        return {
            "noise_levels": noise_levels,
            "coherent_reliability": coherent_results,
            "incoherent_reliability": incoherent_results
        }
        
    except Exception as e:
        print(f"❌ Robustness comparison failed: {e}")
        return {"error": str(e)}


def demo_use_case_recommendations():
    """Demo 5: Recomendaciones de casos de uso."""
    print("\n" + "="*60)
    print("🎯 DEMO 5: Architecture Selection Guidelines")
    print("="*60)
    
    print(f"🌟 COHERENTONN - Recommended Use Cases:")
    print(f"   ✅ High-precision applications")
    print(f"   ✅ Research environments with precise control")
    print(f"   ✅ Applications requiring unitary operations")
    print(f"   ✅ Quantum-inspired classical computing")
    print(f"   ✅ Small to medium scale problems")
    print(f"   ✅ Phase-sensitive signal processing")
    
    print(f"\n   ❌ Less suitable for:")
    print(f"   ❌ Large-scale deployment (fabrication challenges)")
    print(f"   ❌ Commercial products (complexity)")
    print(f"   ❌ Applications requiring fabrication tolerance")
    print(f"   ❌ High-volume manufacturing")
    
    print(f"\n💫 INCOHERENTONN - Recommended Use Cases:")
    print(f"   ✅ Large-scale neural networks")
    print(f"   ✅ Commercial photonic products")
    print(f"   ✅ Telecom-integrated AI systems")
    print(f"   ✅ Applications requiring scalability")
    print(f"   ✅ Edge computing with optical links")
    print(f"   ✅ Robust manufacturing environments")
    print(f"   ✅ WDM-based parallel processing")
    
    print(f"\n   ❌ Less suitable for:")
    print(f"   ❌ Applications requiring negative weights")
    print(f"   ❌ Highest precision requirements")
    print(f"   ❌ Research requiring perfect unitarity")
    print(f"   ❌ Small problems (overkill)")
    
    print(f"\n📊 PERFORMANCE COMPARISON MATRIX:")
    print(f"   Criterion          | CoherentONN | IncoherentONN")
    print(f"   -------------------|-------------|---------------")
    print(f"   Precision          |     High    |    Medium")
    print(f"   Scalability        |     Low     |     High")
    print(f"   Fabrication Tol.   |     Low     |     High")
    print(f"   Energy Efficiency  |     High    |    Medium")
    print(f"   Commercial Ready   |     Low     |     High")
    print(f"   Research Value     |     High    |    Medium")
    print(f"   Implementation     |   Complex   |   Moderate")
    
    print(f"\n🏭 TECHNOLOGY READINESS LEVELS:")
    print(f"   CoherentONN:   TRL 3-4 (Proof of concept)")
    print(f"   IncoherentONN: TRL 5-6 (Technology demonstration)")
    
    print(f"\n🔮 FUTURE ROADMAP:")
    print(f"   Near-term (1-2 years):")
    print(f"   - IncoherentONN commercialization")
    print(f"   - CoherentONN research breakthroughs")
    print(f"   - Hybrid architectures exploration")
    
    print(f"   Long-term (3-5 years):")
    print(f"   - CoherentONN fabrication solutions")
    print(f"   - Advanced IncoherentONN capabilities")
    print(f"   - Integration with quantum photonics")
    
    return {
        "recommendation_generated": True,
        "coherent_trl": 3.5,
        "incoherent_trl": 5.5
    }


def generate_final_report(results: Dict[str, Any]):
    """Generar reporte final de la demo comparativa."""
    print("\n" + "🌟"*20)
    print("🌟  FINAL COMPARATIVE REPORT  🌟")
    print("🌟"*20)
    
    successful_demos = sum(1 for result in results.values() if "error" not in result)
    total_demos = len(results)
    
    print(f"\n📊 DEMO COMPLETION SUMMARY:")
    print(f"   Successfully completed: {successful_demos}/{total_demos} demos")
    
    for demo_name, result in results.items():
        status = "✅ PASS" if "error" not in result else "❌ FAIL"
        print(f"   {status} {demo_name.replace('_', ' ').title()}")
        
        if "error" in result:
            print(f"     Error: {result['error']}")
    
    print(f"\n🏆 KEY FINDINGS:")
    
    if "forward_pass_comparison" in results and "error" not in results["forward_pass_comparison"]:
        fp_results = results["forward_pass_comparison"]
        
        if "coherent_time" in fp_results and "incoherent_time" in fp_results:
            coh_time = fp_results["coherent_time"] * 1000
            inc_time = fp_results["incoherent_time"] * 1000
            faster = "CoherentONN" if coh_time < inc_time else "IncoherentONN"
            print(f"   ⚡ Performance: {faster} is faster ({min(coh_time, inc_time):.1f}ms vs {max(coh_time, inc_time):.1f}ms)")
        
        if "output_difference" in fp_results and fp_results["output_difference"]:
            diff = fp_results["output_difference"]
            print(f"   🔬 Physics: Architectures produce different outputs (diff: {diff:.6f})")
    
    if "wavelength_scaling" in results and "error" not in results["wavelength_scaling"]:
        wdm_results = results["wavelength_scaling"]
        if "valid_results" in wdm_results and len(wdm_results["valid_results"]) > 1:
            max_speedup = max(r["speedup"] for r in wdm_results["valid_results"])
            print(f"   🌈 WDM Scaling: IncoherentONN achieves up to {max_speedup:.1f}x theoretical speedup")
    
    print(f"\n🎯 ARCHITECTURE SELECTION GUIDE:")
    print(f"   Choose CoherentONN when:")
    print(f"   - Highest precision is critical")
    print(f"   - Research environment with controlled fabrication")
    print(f"   - Mathematical elegance is valued")
    print(f"   - Small to medium scale problems")
    
    print(f"\n   Choose IncoherentONN when:")
    print(f"   - Scalability is priority")
    print(f"   - Commercial deployment is the goal")
    print(f"   - Fabrication tolerance is needed")
    print(f"   - WDM infrastructure is available")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Try both architectures on your specific problem")
    print(f"   2. Consider hybrid approaches")
    print(f"   3. Evaluate fabrication constraints")
    print(f"   4. Plan for scalability requirements")
    
    if successful_demos == total_demos:
        print(f"\n🎉 EXCELLENT: All comparative demos completed!")
        print(f"✅ Both architectures are functional and ready for research")
        return 0
    elif successful_demos >= total_demos * 0.75:
        print(f"\n✅ GOOD: Most demos completed successfully")
        return 0
    else:
        print(f"\n⚠️ ISSUES: Several demos failed")
        return 1


def main():
    """Función principal de la demo comparativa."""
    parser = argparse.ArgumentParser(description="Comparative Demo: CoherentONN vs IncoherentONN")
    parser.add_argument("--quick", action="store_true", help="Run quick demo")
    parser.add_argument("--wavelengths", type=int, default=8, help="Max wavelengths for scaling demo")
    parser.add_argument("--size", type=int, default=6, help="Problem size")
    
    args = parser.parse_args()
    
    # Banner
    print("🌟" * 25)
    print("🌟  COHERENT vs INCOHERENT ONN DEMO  🌟")
    print("🌟" * 25)
    print(f"🔬 Comparative analysis of optical neural network architectures")
    print(f"📚 CoherentONN: MZI-based unitary processing")
    print(f"💫 IncoherentONN: Microring-based WDM processing")
    
    # Check requirements
    requirements_ok, available = check_requirements()
    if not requirements_ok:
        print(f"\n❌ Requirements not satisfied. Cannot run comparative demo.")
        return 1
    
    print(f"\n🎯 Available architectures: {available}")
    
    # Configuration
    if args.quick:
        print(f"\n⚡ Quick mode enabled")
    
    # Run demos
    all_results = {}
    
    try:
        # Demo 1: Architecture principles
        all_results["architecture_principles"] = demo_architecture_principles()
        
        # Demo 2: Forward pass comparison
        all_results["forward_pass_comparison"] = demo_forward_pass_comparison()
        
        # Demo 3: Wavelength scaling (IncoherentONN specific)
        all_results["wavelength_scaling"] = demo_wavelength_scaling()
        
        # Demo 4: Robustness comparison
        if not args.quick:  # Skip in quick mode as it's time-consuming
            all_results["robustness_comparison"] = demo_robustness_comparison()
        else:
            print(f"\n⏭️ Skipping robustness comparison in quick mode")
        
        # Demo 5: Use case recommendations
        all_results["use_case_recommendations"] = demo_use_case_recommendations()
        
        # Final report
        return generate_final_report(all_results)
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Demo failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    print(f"\n🎉 Comparative demo completed!")
    sys.exit(exit_code)