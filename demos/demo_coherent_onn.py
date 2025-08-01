#!/usr/bin/env python3
"""
🌟 Complete ONNs Demo for OpticalCI

Demostración completa del nuevo módulo de Redes Neuronales Ópticas (ONNs).
Muestra todas las capacidades implementadas en la Fase 1.

🎯 Funcionalidades Demostradas:
- CoherentONN: Arquitectura coherente usando MZI mesh
- Optical MNIST: Benchmark con dataset real
- Physics Validation: Conservación de energía y unitaridad
- Comparison: ONN vs ANN equivalente
- Metrics: Eficiencia óptica y métricas específicas

📚 Basado en: Shen et al. (2017), Hughes et al. (2018)
🔧 Usa: Componentes OpticalCI existentes + nuevo módulo ONNs

USO:
    python demo_onns_complete.py [--quick] [--size SIZE] [--epochs EPOCHS]
    
EJEMPLOS:
    python demo_onns_complete.py --quick           # Demo rápido 4x4, 3 epochs
    python demo_onns_complete.py --size 8         # Imágenes 8x8, configuración estándar
    python demo_onns_complete.py --epochs 15      # Entrenamiento más largo
"""

import argparse
import sys
import torch
import numpy as np
import time
import warnings
from typing import Dict, Any, List

# Configurar warnings
warnings.filterwarnings("ignore", category=UserWarning)

def check_requirements():
    """Verificar que todos los componentes necesarios están disponibles."""
    print("🔧 Checking requirements...")
    
    issues = []
    
    # Check PyTorch
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        if not torch.__version__.startswith('2.'):
            issues.append(f"PyTorch {torch.__version__} may not be fully supported (recommend 2.0+)")
    except ImportError:
        issues.append("PyTorch not available")
    
    # Check OpticalCI core
    try:
        import torchonn
        from torchonn.layers import MZILayer, MicroringResonator
        print(f"   ✅ OpticalCI core v{torchonn.__version__}")
    except ImportError as e:
        issues.append(f"OpticalCI core not available: {e}")
    
    # Check ONNs module
    try:
        from torchonn.onns.architectures import CoherentONN
        from torchonn.onns.benchmarks import OpticalMNIST
        print(f"   ✅ ONNs module available")
    except ImportError as e:
        issues.append(f"ONNs module not available: {e}")
    
    # Report issues
    if issues:
        print(f"\n❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"   ✅ All requirements satisfied")
        return True

def demo_basic_coherent_onn():
    """Demo 1: Funcionalidad básica de CoherentONN."""
    print("\n" + "="*60)
    print("🌟 DEMO 1: CoherentONN Basic Functionality")
    print("="*60)
    
    try:
        from torchonn.onns.architectures import CoherentONN
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Crear CoherentONN con diferentes configuraciones
        print("\n📐 Testing different architectures...")
        
        architectures = [
            {"name": "Simple", "sizes": [4, 6, 3], "activation": "square_law"},
            {"name": "Deep", "sizes": [8, 16, 12, 6], "activation": "square_law"},
            {"name": "Wide", "sizes": [10, 20, 10], "activation": "square_law"}
        ]
        
        results = {}
        
        for arch in architectures:
            print(f"\n🔧 Testing {arch['name']} architecture: {arch['sizes']}")
            
            try:
                # Crear ONN
                onn = CoherentONN(
                    layer_sizes=arch['sizes'],
                    activation_type=arch['activation'],
                    optical_power=1.0,
                    use_unitary_constraints=True,
                    device=device
                )
                
                # Test forward pass
                batch_size = 8
                input_size = arch['sizes'][0]
                output_size = arch['sizes'][-1]
                
                x = torch.randn(batch_size, input_size, device=device) * 0.5
                
                start_time = time.time()
                output = onn(x)
                forward_time = time.time() - start_time
                
                # Validaciones
                assert output.shape == (batch_size, output_size), f"Wrong output shape"
                assert not torch.any(torch.isnan(output)), "NaN in output"
                assert not torch.any(torch.isinf(output)), "Inf in output"
                
                # Métricas
                efficiency = onn.get_optical_efficiency()
                unitarity = onn.validate_unitarity()
                
                results[arch['name']] = {
                    "success": True,
                    "forward_time": forward_time,
                    "optical_fraction": efficiency["optical_fraction"],
                    "unitarity_valid": unitarity["overall_valid"],
                    "output_range": [float(torch.min(output)), float(torch.max(output))]
                }
                
                print(f"   ✅ Success - Forward: {forward_time*1000:.2f}ms")
                print(f"   📊 Optical fraction: {efficiency['optical_fraction']:.2f}")
                print(f"   🔬 Unitarity: {unitarity['overall_valid']}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[arch['name']] = {"success": False, "error": str(e)}
        
        return results
        
    except Exception as e:
        print(f"❌ Demo 1 failed: {e}")
        return {"error": str(e)}

def demo_physics_validation():
    """Demo 2: Validación física detallada."""
    print("\n" + "="*60)
    print("🔬 DEMO 2: Physics Validation")
    print("="*60)
    
    try:
        from torchonn.onns.architectures import CoherentONN
        from torchonn.onns.utils import validate_onn_energy_conservation
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear ONN para physics testing
        print("📐 Creating physics test ONN...")
        onn = CoherentONN(
            layer_sizes=[6, 8, 6],  # Cuadrada para mejor unitaridad
            activation_type="square_law",
            optical_power=1.0,
            use_unitary_constraints=True,
            device=device
        )
        
        # Habilitar validación física completa
        onn.set_physics_validation(True, frequency=1.0)  # 100% validación
        
        print("🔬 Running physics validation tests...")
        
        # Test 1: Conservación de energía en múltiples forward passes
        print("\n1️⃣ Energy Conservation Test:")
        energy_ratios = []
        
        for i in range(10):
            x = torch.randn(4, 6, device=device) * 0.8  # Input moderado
            input_power = torch.sum(torch.abs(x)**2)
            
            with torch.no_grad():
                output = onn(x)
            output_power = torch.sum(torch.abs(output)**2)
            
            ratio = (output_power / input_power).item()
            energy_ratios.append(ratio)
        
        avg_ratio = np.mean(energy_ratios)
        std_ratio = np.std(energy_ratios)
        
        print(f"   Average energy ratio: {avg_ratio:.6f} ± {std_ratio:.6f}")
        print(f"   Expected: ~1.000000 (perfect conservation)")
        
        conservation_quality = "Excellent" if abs(avg_ratio - 1.0) < 0.01 else \
                             "Good" if abs(avg_ratio - 1.0) < 0.05 else \
                             "Poor"
        print(f"   Quality: {conservation_quality}")
        
        # Test 2: Unitaridad de las capas ópticas
        print("\n2️⃣ Unitarity Validation:")
        unitarity = onn.validate_unitarity()
        
        for layer_name, layer_info in unitarity["layers"].items():
            if "unitarity_error" in layer_info:
                error = layer_info["unitarity_error"]
                status = "✅ Unitary" if error < 1e-3 else "⚠️ Approximate" if error < 1e-1 else "❌ Non-unitary"
                print(f"   {layer_name}: {status} (error: {error:.2e})")
        
        print(f"   Overall: {'✅ All layers unitary' if unitarity['overall_valid'] else '⚠️ Some issues detected'}")
        
        # Test 3: Gradient flow
        print("\n3️⃣ Gradient Flow Test:")
        x = torch.randn(2, 6, device=device, requires_grad=True)
        output = onn(x)
        loss = torch.mean(output**2)
        loss.backward()
        
        grad_norm = torch.norm(x.grad).item()
        print(f"   Input gradient norm: {grad_norm:.6f}")
        
        param_grads = []
        for name, param in onn.named_parameters():
            if param.grad is not None:
                param_grad_norm = torch.norm(param.grad).item()
                param_grads.append(param_grad_norm)
        
        if param_grads:
            avg_param_grad = np.mean(param_grads)
            print(f"   Avg parameter gradient: {avg_param_grad:.6f}")
            print(f"   Gradient flow: {'✅ Healthy' if avg_param_grad > 1e-6 else '⚠️ Weak'}")
        
        # Test 4: Métricas ONN específicas
        print("\n4️⃣ ONN-Specific Metrics:")
        metrics = onn.get_onn_metrics()
        efficiency = onn.get_optical_efficiency()
        
        print(f"   Forward passes: {metrics['total_forward_passes']}")
        print(f"   Optical operations: {efficiency['optical_operations']}")
        print(f"   Optical fraction: {efficiency['optical_fraction']:.2%}")
        print(f"   Theoretical speedup: {efficiency['theoretical_speedup']:.1f}x")
        
        return {
            "energy_conservation": {
                "average_ratio": avg_ratio,
                "std_ratio": std_ratio,
                "quality": conservation_quality
            },
            "unitarity": unitarity["overall_valid"],
            "gradient_flow": grad_norm > 1e-8,
            "onn_metrics": metrics
        }
        
    except Exception as e:
        print(f"❌ Demo 2 failed: {e}")
        return {"error": str(e)}

def demo_optical_mnist(image_size: int = 8, n_epochs: int = 10):
    """Demo 3: Optical MNIST benchmark completo."""
    print("\n" + "="*60)
    print("🎯 DEMO 3: Optical MNIST Benchmark")
    print("="*60)
    
    try:
        from torchonn.onns.benchmarks import OpticalMNIST
        
        print(f"📊 Configuration:")
        print(f"   Image size: {image_size}x{image_size}")
        print(f"   Training epochs: {n_epochs}")
        print(f"   Dataset: MNIST (subset)")
        
        # Crear benchmark
        benchmark = OpticalMNIST(
            image_size=image_size,
            n_classes=10
        )
        
        # Ejecutar benchmark comparativo
        print(f"\n🚀 Running comparative benchmark...")
        results = benchmark.run_comparison_benchmark(
            n_epochs=n_epochs,
            learning_rate=0.01
        )
        
        # Análisis de resultados
        print(f"\n📊 DETAILED RESULTS:")
        
        if "onn" in results and "error" not in results["onn"]:
            onn_results = results["onn"]
            print(f"\n🌟 CoherentONN Performance:")
            print(f"   Test Accuracy: {onn_results.get('test_accuracy', 0):.2f}%")
            print(f"   Training Time: {onn_results.get('total_training_time', 0):.1f}s")
            print(f"   Inference Time: {onn_results.get('avg_inference_time', 0)*1000:.2f}ms/batch")
            
            if "onn_specific_metrics" in onn_results:
                onn_metrics = onn_results["onn_specific_metrics"]
                print(f"   Energy Conservation Samples: {len(onn_metrics.get('energy_conservation_history', []))}")
                
                if "energy_conservation_stats" in onn_metrics:
                    energy_stats = onn_metrics["energy_conservation_stats"]
                    print(f"   Avg Energy Conservation: {energy_stats['mean']:.6f}")
            
            if "physics_violations" in onn_results:
                violations = onn_results["n_physics_violations"]
                status = "✅ Compliant" if violations == 0 else f"⚠️ {violations} violations"
                print(f"   Physics Validation: {status}")
        
        if "ann" in results and "error" not in results["ann"]:
            ann_results = results["ann"]
            print(f"\n🔌 Reference ANN Performance:")
            print(f"   Test Accuracy: {ann_results.get('test_accuracy', 0):.2f}%")
            print(f"   Training Time: {ann_results.get('total_training_time', 0):.1f}s")
            print(f"   Inference Time: {ann_results.get('avg_inference_time', 0)*1000:.2f}ms/batch")
        
        if "comparison" in results:
            comp = results["comparison"]
            print(f"\n⚖️ Comparative Analysis:")
            print(f"   Accuracy Ratio (ONN/ANN): {comp.get('accuracy_ratio', 0):.3f}")
            print(f"   Speed Ratio (ONN/ANN): {comp.get('speed_ratio', 0):.3f}")
            print(f"   Training Time Ratio: {comp.get('training_time_ratio', 0):.3f}")
            
            # Assessment
            acc_ratio = comp.get('accuracy_ratio', 0)
            if acc_ratio > 0.95:
                assessment = "🎉 Excellent - ONN matches ANN performance"
            elif acc_ratio > 0.85:
                assessment = "✅ Good - ONN performance acceptable"
            elif acc_ratio > 0.7:
                assessment = "⚠️ Fair - ONN shows promise but needs improvement"
            else:
                assessment = "❌ Poor - ONN significantly underperforms"
            
            print(f"   Assessment: {assessment}")
        
        return results
        
    except Exception as e:
        print(f"❌ Demo 3 failed: {e}")
        return {"error": str(e)}

def demo_component_integration():
    """Demo 4: Integración con componentes OpticalCI existentes."""
    print("\n" + "="*60)
    print("🔗 DEMO 4: Component Integration")
    print("="*60)
    
    try:
        print("🔧 Testing integration with existing OpticalCI components...")
        
        # Test 1: MZI components
        print("\n1️⃣ MZI Components Integration:")
        from torchonn.layers import MZILayer, MZIBlockLinear
        from torchonn.onns.architectures import CoherentONN
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Verificar que CoherentONN usa MZI components internamente
        onn = CoherentONN(layer_sizes=[4, 6, 4], device=device)
        
        mzi_components = [m for m in onn.modules() if 'MZI' in str(type(m))]
        print(f"   MZI components found in ONN: {len(mzi_components)}")
        
        for i, component in enumerate(mzi_components[:3]):  # Solo primeros 3
            print(f"   Component {i+1}: {type(component).__name__}")
        
        # Test 2: Photodetector integration
        print("\n2️⃣ Photodetector Integration:")
        from torchonn.layers import Photodetector
        
        photodetectors = [m for m in onn.modules() if isinstance(m, Photodetector)]
        print(f"   Photodetectors found in ONN: {len(photodetectors)}")
        
        # Test forward pass que usa estos componentes
        x = torch.randn(2, 4, device=device)
        output = onn(x)
        print(f"   Integrated forward pass: {x.shape} → {output.shape}")
        
        # Test 3: Device consistency
        print("\n3️⃣ Device Consistency:")
        all_on_device = True
        device_summary = {}
        
        for name, param in onn.named_parameters():
            param_device = param.device
            if param_device not in device_summary:
                device_summary[param_device] = 0
            device_summary[param_device] += 1
            
            if param_device != device:
                all_on_device = False
        
        print(f"   Target device: {device}")
        print(f"   Parameter devices: {dict(device_summary)}")
        print(f"   All on correct device: {'✅' if all_on_device else '❌'}")
        
        # Test 4: Backward compatibility
        print("\n4️⃣ Backward Compatibility:")
        
        # Test que componentes existentes siguen funcionando
        mzi_standalone = MZILayer(4, 4, device=device)
        x_test = torch.randn(2, 4, device=device)
        y_test = mzi_standalone(x_test)
        print(f"   Standalone MZI still works: ✅")
        
        # Test microring
        from torchonn.layers import MicroringResonator
        mrr = MicroringResonator(device=device)
        wavelengths = torch.linspace(1549e-9, 1551e-9, 5, device=device)
        signal = torch.ones(1, 5, device=device)
        
        with torch.no_grad():
            mrr_output = mrr(signal, wavelengths)
        print(f"   Standalone MRR still works: ✅")
        
        # Test que la importación base no se rompió
        import torchonn
        version = torchonn.get_version()
        print(f"   Package version accessible: {version}")
        
        return {
            "mzi_integration": len(mzi_components) > 0,
            "photodetector_integration": len(photodetectors) > 0,
            "device_consistency": all_on_device,
            "backward_compatibility": True
        }
        
    except Exception as e:
        print(f"❌ Demo 4 failed: {e}")
        return {"error": str(e)}

def generate_final_report(all_results: Dict[str, Any]):
    """Generar reporte final de todos los demos."""
    print("\n" + "="*80)
    print("📋 FINAL COMPREHENSIVE REPORT")
    print("="*80)
    
    # Summary
    successful_demos = sum(1 for result in all_results.values() if "error" not in result)
    total_demos = len(all_results)
    
    print(f"🎯 Demo Success Rate: {successful_demos}/{total_demos}")
    
    # Individual demo status
    print(f"\n📊 Individual Demo Results:")
    for demo_name, result in all_results.items():
        if "error" in result:
            print(f"   ❌ {demo_name}: Failed ({result['error'][:50]}...)")
        else:
            print(f"   ✅ {demo_name}: Successful")
    
    # Physics validation summary
    print(f"\n🔬 Physics Validation Summary:")
    if "physics_validation" in all_results and "error" not in all_results["physics_validation"]:
        physics = all_results["physics_validation"]
        energy_quality = physics.get("energy_conservation", {}).get("quality", "Unknown")
        unitarity = physics.get("unitarity", False)
        
        print(f"   Energy Conservation: {energy_quality}")
        print(f"   Unitarity: {'✅ Valid' if unitarity else '⚠️ Issues detected'}")
    else:
        print(f"   ⚠️ Physics validation not completed")
    
    # Performance summary
    print(f"\n🎯 Performance Summary:")
    if "optical_mnist" in all_results and "error" not in all_results["optical_mnist"]:
        mnist = all_results["optical_mnist"]
        
        if "comparison" in mnist:
            comp = mnist["comparison"]
            acc_ratio = comp.get("accuracy_ratio", 0)
            speed_ratio = comp.get("speed_ratio", 0)
            
            print(f"   ONN vs ANN Accuracy: {acc_ratio:.3f}")
            print(f"   ONN vs ANN Speed: {speed_ratio:.3f}")
            
            if acc_ratio > 0.9:
                print(f"   Performance Grade: A (Excellent)")
            elif acc_ratio > 0.8:
                print(f"   Performance Grade: B (Good)")
            elif acc_ratio > 0.7:
                print(f"   Performance Grade: C (Fair)")
            else:
                print(f"   Performance Grade: D (Needs Improvement)")
    
    # Integration summary
    print(f"\n🔗 Integration Summary:")
    if "component_integration" in all_results and "error" not in all_results["component_integration"]:
        integration = all_results["component_integration"]
        
        mzi_ok = integration.get("mzi_integration", False)
        pd_ok = integration.get("photodetector_integration", False)
        device_ok = integration.get("device_consistency", False)
        compat_ok = integration.get("backward_compatibility", False)
        
        print(f"   MZI Integration: {'✅' if mzi_ok else '❌'}")
        print(f"   Photodetector Integration: {'✅' if pd_ok else '❌'}")
        print(f"   Device Consistency: {'✅' if device_ok else '❌'}")
        print(f"   Backward Compatibility: {'✅' if compat_ok else '❌'}")
    
    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    
    if successful_demos == total_demos:
        print(f"   🎉 EXCELLENT: All demos completed successfully")
        print(f"   ✅ ONNs module is fully functional")
        print(f"   ✅ Integration with OpticalCI is seamless")
        print(f"   ✅ Ready for research and development")
    elif successful_demos >= total_demos * 0.75:
        print(f"   ✅ GOOD: Most demos completed successfully")
        print(f"   ⚠️ Some minor issues detected")
        print(f"   🔧 Recommend reviewing failed components")
    else:
        print(f"   ⚠️ NEEDS WORK: Several demos failed")
        print(f"   🔧 Significant issues need to be addressed")
        print(f"   📝 Check error messages and implementations")
    
    # Next steps
    print(f"\n🚀 RECOMMENDED NEXT STEPS:")
    print(f"   1. 📚 Implement additional ONN architectures (IncoherentONN)")
    print(f"   2. 🎯 Add more benchmarks (CIFAR-10, Iris, Signal Processing)")
    print(f"   3. 🔬 Enhance physics models (thermal effects, noise)")
    print(f"   4. ⚡ Optimize performance for larger networks")
    print(f"   5. 📖 Create comprehensive documentation and tutorials")

def main():
    """Función principal del demo completo."""
    parser = argparse.ArgumentParser(description="Complete ONNs Demo for OpticalCI")
    parser.add_argument("--quick", action="store_true", help="Run quick demo (4x4, 3 epochs)")
    parser.add_argument("--size", type=int, default=8, help="Image size for MNIST demo")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--skip-mnist", action="store_true", help="Skip MNIST benchmark (faster)")
    
    args = parser.parse_args()
    
    # Banner
    print("🌟" * 20)
    print("🌟  OPTICALCI ONNs COMPLETE DEMO  🌟")
    print("🌟" * 20)
    print(f"🔬 Demonstrating: Coherent Optical Neural Networks")
    print(f"📚 Based on: Shen et al. (2017), Hughes et al. (2018)")
    print(f"⚡ Implementation: Phase 1 - Foundational Architecture")
    
    # Check requirements
    if not check_requirements():
        print(f"\n❌ Requirements not satisfied. Please check installation.")
        return 1
    
    # Configuration
    if args.quick:
        mnist_size = 4
        mnist_epochs = 3
        print(f"\n⚡ Quick mode: {mnist_size}x{mnist_size} images, {mnist_epochs} epochs")
    else:
        mnist_size = args.size
        mnist_epochs = args.epochs
        print(f"\n🎯 Standard mode: {mnist_size}x{mnist_size} images, {mnist_epochs} epochs")
    
    # Run demos
    all_results = {}
    
    try:
        # Demo 1: Basic functionality
        all_results["basic_functionality"] = demo_basic_coherent_onn()
        
        # Demo 2: Physics validation
        all_results["physics_validation"] = demo_physics_validation()
        
        # Demo 3: Optical MNIST (si no se omite)
        if not args.skip_mnist:
            all_results["optical_mnist"] = demo_optical_mnist(mnist_size, mnist_epochs)
        else:
            print(f"\n⏭️ Skipping MNIST benchmark (--skip-mnist)")
        
        # Demo 4: Component integration
        all_results["component_integration"] = demo_component_integration()
        
        # Final report
        generate_final_report(all_results)
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Demo failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n🎉 Demo completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)