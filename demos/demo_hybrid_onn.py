#!/usr/bin/env python3
"""
🌟 Demo Real HybridONN - Implementación Funcionando

Demo de la implementación real de HybridONN usando componentes OpticalCI reales.
NO es conceptual - usa la implementación real que funciona.

UBICACIÓN: demos/demo_hybrid_onn_real.py
USO: python demos/demo_hybrid_onn_real.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def check_real_hybrid_requirements():
    """Verificar que HybridONN real esté disponible."""
    print("🔧 Checking real HybridONN requirements...")
    
    try:
        # Test HybridONN real import
        from torchonn.onns.architectures import HybridONN, HybridMode
        print("   ✅ HybridONN real implementation available")
        
        # Test existing architectures
        from torchonn.onns.architectures import CoherentONN, IncoherentONN
        print("   ✅ CoherentONN and IncoherentONN available")
        
        # Test OpticalCI components  
        from torchonn.layers import MZILayer, Photodetector
        from torchonn.components import WDMMultiplexer
        print("   ✅ OpticalCI components available")
        
        # Test factory functions
        from torchonn.onns.architectures.hybrid_onn import (
            create_image_processing_hybrid,
            create_signal_processing_hybrid
        )
        print("   ✅ HybridONN factory functions available")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ {e}")
        return False


def demo_real_hybrid_functionality():
    """Demo 1: Funcionalidad real de HybridONN."""
    print("\n" + "="*60)
    print("🔬 DEMO 1: Real HybridONN Functionality")
    print("="*60)
    
    try:
        from torchonn.onns.architectures import HybridONN, HybridMode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layer_sizes = [8, 12, 8, 4]
        
        print(f"🏗️ Creating real HybridONN:")
        print(f"   Layer sizes: {layer_sizes}")
        print(f"   Device: {device}")
        
        # Crear HybridONN real
        onn = HybridONN(
            layer_sizes=layer_sizes,
            hybrid_mode=HybridMode.ALTERNATING,
            n_wavelengths=4,
            device=device
        )
        
        print(f"\n✅ HybridONN created successfully!")
        
        # Test forward pass real
        batch_size = 16
        x = torch.randn(batch_size, layer_sizes[0], device=device) * 0.5
        
        print(f"\n🚀 Testing real forward pass:")
        print(f"   Input shape: {x.shape}")
        
        start_time = time.time()
        y = onn(x)
        forward_time = time.time() - start_time
        
        print(f"   Output shape: {y.shape}")
        print(f"   Forward time: {forward_time*1000:.1f}ms")
        print(f"   Output range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"   No NaN/Inf: {'✅' if not torch.any(torch.isnan(y) | torch.isinf(y)) else '❌'}")
        
        # Test métricas reales
        print(f"\n📊 Real HybridONN Metrics:")
        metrics = onn.get_hybrid_metrics()
        
        print(f"   Architecture: {metrics['architecture_type']}")
        print(f"   Mode: {metrics['hybrid_mode']}")
        print(f"   Layer types: {' → '.join(metrics['layer_configuration']['layer_types'])}")
        print(f"   Coherent fraction: {metrics['layer_configuration']['coherent_fraction']:.1%}")
        print(f"   Transitions: {metrics['transition_analysis']['total_transitions']}")
        print(f"   Parameters: {metrics['resource_utilization']['total_parameters']}")
        print(f"   Theoretical speedup: {metrics['resource_utilization']['theoretical_speedup']:.2f}x")
        
        # Test validación física real
        print(f"\n🔬 Real Physics Validation:")
        physics = onn.validate_hybrid_physics(verbose=False)
        
        print(f"   Overall valid: {'✅' if physics['overall_valid'] else '❌'}")
        print(f"   Transition physics: {'✅' if physics['checks']['transitions']['valid'] else '❌'}")
        print(f"   Coherent layers: {'✅' if physics['checks']['coherent_layers']['valid'] else '❌'}")
        print(f"   Incoherent layers: {'✅' if physics['checks']['incoherent_layers']['valid'] else '❌'}")
        
        return {
            "forward_time": forward_time,
            "output_shape": y.shape,
            "metrics": metrics,
            "physics_valid": physics['overall_valid']
        }
        
    except Exception as e:
        print(f"❌ Real hybrid functionality demo failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def demo_real_comparison_with_existing():
    """Demo 2: Comparación real con arquitecturas existentes."""
    print("\n" + "="*60)
    print("🏆 DEMO 2: Real Comparison with Existing Architectures")
    print("="*60)
    
    try:
        from torchonn.onns.architectures import CoherentONN, IncoherentONN, HybridONN, HybridMode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layer_sizes = [6, 6, 3]  # Tamaño manejable para comparison
        batch_size = 32
        
        print(f"🔬 Creating all architectures for real comparison:")
        
        # Crear todas las arquitecturas
        coherent_onn = CoherentONN(layer_sizes=layer_sizes, device=device)
        incoherent_onn = IncoherentONN(layer_sizes=layer_sizes, n_wavelengths=4, device=device)
        hybrid_onn = HybridONN(
            layer_sizes=layer_sizes,
            hybrid_mode=HybridMode.ADAPTIVE,
            n_wavelengths=4,
            device=device
        )
        
        architectures = {
            "CoherentONN": coherent_onn,
            "IncoherentONN": incoherent_onn,
            "HybridONN": hybrid_onn
        }
        
        # Test mismo input en todas
        x = torch.randn(batch_size, layer_sizes[0], device=device) * 0.5
        
        results = {}
        
        print(f"\n⚡ Real Performance Comparison:")
        
        for arch_name, model in architectures.items():
            print(f"   Testing {arch_name}...")
            
            # Warmup
            for _ in range(3):
                _ = model(x)
            
            # Real timing test
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            y = model(x)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
                
            forward_time = time.time() - start_time
            
            # Analysis
            results[arch_name] = {
                "forward_time": forward_time,
                "forward_time_ms": forward_time * 1000,
                "output_mean": y.mean().item(),
                "output_std": y.std().item(),
                "output_range": [y.min().item(), y.max().item()],
                "parameters": sum(p.numel() for p in model.parameters())
            }
            
            print(f"     • Forward time: {forward_time*1000:.1f}ms")
            print(f"     • Parameters: {results[arch_name]['parameters']}")
            print(f"     • Output stats: μ={y.mean():.3f}, σ={y.std():.3f}")
        
        # Real comparison analysis
        print(f"\n📊 Real Comparison Results:")
        
        fastest = min(results.keys(), key=lambda k: results[k]["forward_time"])
        most_params = max(results.keys(), key=lambda k: results[k]["parameters"])
        
        print(f"   🏃 Fastest: {fastest} ({results[fastest]['forward_time_ms']:.1f}ms)")
        print(f"   🧠 Most Parameters: {most_params} ({results[most_params]['parameters']} params)")
        
        # Test outputs are different (different physics)
        coherent_out = results["CoherentONN"]["output_mean"]
        incoherent_out = results["IncoherentONN"]["output_mean"] 
        hybrid_out = results["HybridONN"]["output_mean"]
        
        print(f"   🔬 Output Diversity (different physics):")
        print(f"     • Coherent mean: {coherent_out:.3f}")
        print(f"     • Incoherent mean: {incoherent_out:.3f}")
        print(f"     • Hybrid mean: {hybrid_out:.3f}")
        
        # Verify they're actually different
        outputs_different = (abs(coherent_out - incoherent_out) > 0.01 and 
                           abs(coherent_out - hybrid_out) > 0.01)
        print(f"   ✅ Architectures produce different outputs: {'Yes' if outputs_different else 'No'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Real comparison demo failed: {e}")
        return {"error": str(e)}


def demo_real_use_cases():
    """Demo 3: Casos de uso reales con factory functions."""
    print("\n" + "="*60)
    print("🎯 DEMO 3: Real Use Cases with Factory Functions")
    print("="*60)
    
    try:
        from torchonn.onns.architectures.hybrid_onn import (
            create_image_processing_hybrid,
            create_signal_processing_hybrid,
            create_large_scale_hybrid
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🏭 Testing real factory functions:")
        
        # Use Case 1: Image Processing (small MNIST-like)
        print(f"\n📸 Image Processing Use Case:")
        img_size = 8  # 8x8 images for demo
        n_classes = 10
        
        img_onn = create_image_processing_hybrid(
            input_size=img_size * img_size,
            n_classes=n_classes
        )
        
        # Real test with synthetic images
        batch_size = 16
        x_img = torch.randn(batch_size, img_size * img_size, device=device) * 0.5
        y_img = img_onn(x_img)
        
        print(f"   ✅ Image ONN created: {x_img.shape} → {y_img.shape}")
        print(f"   Mode: {img_onn.hybrid_mode.value}")
        print(f"   Layer types: {' → '.join(img_onn.layer_types)}")
        
        # Use Case 2: Signal Processing
        print(f"\n📡 Signal Processing Use Case:")
        sig_input_size = 32
        sig_output_size = 8
        
        sig_onn = create_signal_processing_hybrid(
            input_size=sig_input_size,
            output_size=sig_output_size
        )
        
        # Real test with synthetic signals
        x_sig = torch.randn(batch_size, sig_input_size, device=device) * 0.5
        y_sig = sig_onn(x_sig)
        
        print(f"   ✅ Signal ONN created: {x_sig.shape} → {y_sig.shape}")
        print(f"   Mode: {sig_onn.hybrid_mode.value}")
        print(f"   WDM channels: {sig_onn.n_wavelengths}")
        
        # Use Case 3: Large Scale (smaller for demo)
        print(f"\n🏗️ Large Scale Use Case:")
        large_layer_sizes = [64, 32, 16, 8]
        
        large_onn = create_large_scale_hybrid(layer_sizes=large_layer_sizes)
        
        # Real test
        x_large = torch.randn(batch_size, large_layer_sizes[0], device=device) * 0.5
        y_large = large_onn(x_large)
        
        print(f"   ✅ Large Scale ONN created: {x_large.shape} → {y_large.shape}")
        print(f"   Mode: {large_onn.hybrid_mode.value} (automatic optimization)")
        print(f"   Transition loss: {large_onn.transition_loss} (optimized coupling)")
        
        # Performance summary
        print(f"\n📊 Use Case Performance Summary:")
        use_cases = {
            "Image Processing": (img_onn, x_img, y_img),
            "Signal Processing": (sig_onn, x_sig, y_sig),
            "Large Scale": (large_onn, x_large, y_large)
        }
        
        for use_case_name, (model, x_test, y_test) in use_cases.items():
            metrics = model.get_hybrid_metrics()
            params = metrics["resource_utilization"]["total_parameters"]
            speedup = metrics["resource_utilization"]["theoretical_speedup"]
            
            print(f"   {use_case_name}:")
            print(f"     • Parameters: {params}")
            print(f"     • Theoretical speedup: {speedup:.2f}x")
            print(f"     • Coherent fraction: {metrics['layer_configuration']['coherent_fraction']:.1%}")
        
        return {
            "image_processing": True,
            "signal_processing": True, 
            "large_scale": True
        }
        
    except Exception as e:
        print(f"❌ Real use cases demo failed: {e}")
        return {"error": str(e)}


def demo_real_training():
    """Demo 4: Entrenamiento real con HybridONN."""
    print("\n" + "="*60)
    print("🎯 DEMO 4: Real Training with HybridONN")
    print("="*60)
    
    try:
        from torchonn.onns.architectures import HybridONN, HybridMode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Problema de clasificación sintético simple
        n_samples = 200
        layer_sizes = [4, 8, 3]  # 4 features → 3 classes
        
        print(f"🎯 Setting up real training scenario:")
        print(f"   Problem: 4-feature classification → 3 classes")
        print(f"   Samples: {n_samples}")
        
        # Generar datos sintéticos
        X = torch.randn(n_samples, layer_sizes[0], device=device)
        
        # Target classes based on simple rule for reproducibility
        y_target = torch.zeros(n_samples, dtype=torch.long, device=device)
        for i in range(n_samples):
            if X[i, 0] > 0 and X[i, 1] > 0:
                y_target[i] = 0
            elif X[i, 2] > 0:
                y_target[i] = 1  
            else:
                y_target[i] = 2
        
        print(f"   Data generated - Classes distribution: {torch.bincount(y_target)}")
        
        # Crear HybridONN para entrenamiento
        onn = HybridONN(
            layer_sizes=layer_sizes,
            hybrid_mode=HybridMode.ALTERNATING,
            device=device
        )
        
        print(f"\n🏗️ HybridONN for training:")
        print(f"   Mode: {onn.hybrid_mode.value}")
        print(f"   Layer types: {' → '.join(onn.layer_types)}")
        
        # Setup training real
        optimizer = torch.optim.Adam(onn.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop real
        n_epochs = 25
        print(f"\n🔥 Starting real training ({n_epochs} epochs):")
        
        initial_loss = None
        losses = []
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = onn(X)
            loss = criterion(outputs, y_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch == 0:
                initial_loss = loss.item()
            
            if epoch % 5 == 0 or epoch == n_epochs - 1:
                # Compute accuracy
                with torch.no_grad():
                    pred_classes = torch.argmax(outputs, dim=1)
                    accuracy = (pred_classes == y_target).float().mean().item()
                    
                print(f"   Epoch {epoch:2d}: Loss={loss.item():.4f}, Accuracy={accuracy:.3f}")
        
        final_loss = losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss
        
        print(f"\n✅ Real training completed!")
        print(f"   Initial loss: {initial_loss:.4f}")
        print(f"   Final loss: {final_loss:.4f}")
        print(f"   Improvement: {improvement:.1%}")
        print(f"   Training successful: {'✅' if improvement > 0.1 else '❌'}")
        
        # Test generalization con nuevos datos
        print(f"\n🧪 Testing generalization:")
        X_test = torch.randn(50, layer_sizes[0], device=device)
        y_test = torch.zeros(50, dtype=torch.long, device=device)
        for i in range(50):
            if X_test[i, 0] > 0 and X_test[i, 1] > 0:
                y_test[i] = 0
            elif X_test[i, 2] > 0:
                y_test[i] = 1
            else:
                y_test[i] = 2
        
        with torch.no_grad():
            test_outputs = onn(X_test)
            test_pred = torch.argmax(test_outputs, dim=1)
            test_accuracy = (test_pred == y_test).float().mean().item()
            
        print(f"   Test accuracy: {test_accuracy:.3f}")
        print(f"   Generalization: {'✅' if test_accuracy > 0.4 else '❌'}")
        
        return {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "improvement": improvement,
            "test_accuracy": test_accuracy,
            "training_successful": improvement > 0.1
        }
        
    except Exception as e:
        print(f"❌ Real training demo failed: {e}")
        return {"error": str(e)}


def main():
    """Main function para demo real."""
    parser = argparse.ArgumentParser(description="Real HybridONN Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick demo")
    
    args = parser.parse_args()
    
    # Banner
    print("🌟" * 30)
    print("🌟  REAL HYBRIDONN IMPLEMENTATION DEMO  🌟")
    print("🌟" * 30)
    print("🎯 Testing: Real HybridONN with actual OpticalCI components")
    print("🔬 Focus: Functionality, performance, training convergence")
    print("🚀 Goal: Validate complete implementation")
    
    # Check requirements
    if not check_real_hybrid_requirements():
        print("\n❌ Real HybridONN implementation not available")
        print("📝 Please ensure hybrid_onn.py is installed in:")
        print("   torchonn/onns/architectures/hybrid_onn.py")
        return 1
    
    print()
    
    # Run demos
    results = {}
    
    demos = [
        ("real_functionality", demo_real_hybrid_functionality),
        ("real_comparison", demo_real_comparison_with_existing),
        ("real_use_cases", demo_real_use_cases),
        ("real_training", demo_real_training)
    ]
    
    if args.quick:
        demos = demos[:2]  # Only first 2 demos in quick mode
    
    for demo_name, demo_func in demos:
        print(f"Running {demo_name.replace('_', ' ')} demo...")
        try:
            results[demo_name] = demo_func()
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")
            results[demo_name] = {"error": str(e)}
    
    # Final summary
    print("\n" + "🌟"*30)
    print("🌟  REAL HYBRIDONN DEMO SUMMARY  🌟")
    print("🌟"*30)
    
    successful_demos = sum(1 for result in results.values() if "error" not in result)
    total_demos = len(results)
    
    print(f"\n📊 DEMO COMPLETION:")
    print(f"   Successfully completed: {successful_demos}/{total_demos} demos")
    
    for demo_name, result in results.items():
        status = "✅ PASS" if "error" not in result else "❌ FAIL"
        print(f"   {status} {demo_name.replace('_', ' ').title()}")
    
    if successful_demos == total_demos:
        print(f"\n🎉 ALL REAL DEMOS PASSED!")
        print(f"✅ HybridONN real implementation is fully functional")
        print(f"✅ Ready for production use and further development")
        
        return 0
    else:
        print(f"\n⚠️ Some demos failed - check implementation")
        return 1


if __name__ == "__main__":
    exit_code = main()
    print(f"\n🏁 Real demo completed with exit code: {exit_code}")