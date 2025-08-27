#!/usr/bin/env python3
"""
🚀 Quick WDM Optimization Demo

UBICACIÓN: demos/quick_demo_wdm.py

Demo rápido para verificar que las optimizaciones WDM funcionan correctamente.
Ideal para testing inicial y validación rápida.

USAGE:
    python demos/quick_demo_wdm.py
    python demos/quick_demo_wdm.py --wavelengths 16 --batch 64
"""

import argparse
import time
import torch
import numpy as np
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    issues = []
    
    # Check PyTorch
    try:
        print(f"   ✅ PyTorch {torch.__version__}")
        if tuple(map(int, torch.__version__.split('.')[:2])) < (1, 9):
            issues.append("PyTorch version too old (need ≥1.9)")
    except Exception as e:
        issues.append(f"PyTorch issue: {e}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"   {'✅' if cuda_available else '⚠️'} CUDA: {'Available' if cuda_available else 'Not available (CPU only)'}")
    
    # Check NumPy
    try:
        print(f"   ✅ NumPy {np.__version__}")
    except Exception as e:
        issues.append(f"NumPy issue: {e}")
    
    # Check IncoherentONN
    try:
        from torchonn.onns.architectures import IncoherentONN
        print("   ✅ IncoherentONN available")
    except ImportError as e:
        issues.append(f"IncoherentONN not available: {e}")
    
    # Check WDM optimizations
    try:
        from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
        print("   ✅ WDM Optimizations available")
        optimizations_available = True
    except ImportError as e:
        print(f"   ⚠️ WDM Optimizations not available: {e}")
        optimizations_available = False
    
    if issues:
        print("\n❌ Dependency issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False, optimizations_available
    
    print("   ✅ All dependencies OK!")
    return True, optimizations_available

def demo_basic_functionality(device):
    """Demo 1: Basic functionality test."""
    print("\n" + "="*50)
    print("🧪 DEMO 1: Basic Functionality Test")
    print("="*50)
    
    try:
        from torchonn.onns.architectures import IncoherentONN
        
        # Simple test configuration
        layer_sizes = [4, 6, 3]
        n_wavelengths = 4
        batch_size = 8
        
        print(f"📊 Configuration:")
        print(f"   Layers: {layer_sizes}")
        print(f"   Wavelengths: {n_wavelengths}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {device}")
        
        # Test original implementation
        print(f"\n🔧 Testing original implementation...")
        onn_original = IncoherentONN(
            layer_sizes, 
            n_wavelengths=n_wavelengths,
            enable_wdm_optimization=False,
            device=device
        )
        
        x = torch.randn(batch_size, layer_sizes[0], device=device)
        
        start_time = time.time()
        y_original = onn_original(x)
        time_original = time.time() - start_time
        
        print(f"   ✅ Forward pass: {x.shape} → {y_original.shape}")
        print(f"   ⏱️ Time: {time_original*1000:.2f}ms")
        
        metrics_original = onn_original.get_optical_efficiency_metrics()
        print(f"   📊 Efficiency: {metrics_original.get('parallel_efficiency', 0):.1f}%")
        print(f"   ⚡ Theoretical speedup: {metrics_original.get('theoretical_speedup', 1.0):.2f}x")
        
        # Test optimized implementation
        print(f"\n🚀 Testing optimized implementation...")
        onn_optimized = IncoherentONN(
            layer_sizes,
            n_wavelengths=n_wavelengths,
            enable_wdm_optimization=True,
            device=device
        )
        
        start_time = time.time()
        y_optimized = onn_optimized(x)
        time_optimized = time.time() - start_time
        
        print(f"   ✅ Forward pass: {x.shape} → {y_optimized.shape}")
        print(f"   ⏱️ Time: {time_optimized*1000:.2f}ms")
        
        metrics_optimized = onn_optimized.get_optical_efficiency_metrics()
        print(f"   📊 Efficiency: {metrics_optimized.get('parallel_efficiency', 0):.1f}%")
        print(f"   ⚡ Theoretical speedup: {metrics_optimized.get('theoretical_speedup', 1.0):.2f}x")
        
        # Compare results
        speedup_actual = time_original / time_optimized if time_optimized > 0 else 1.0
        efficiency_improvement = (metrics_optimized.get('parallel_efficiency', 0) / 
                                max(metrics_original.get('parallel_efficiency', 1), 1))
        
        print(f"\n📈 Comparison:")
        print(f"   ⚡ Actual speedup: {speedup_actual:.2f}x")
        print(f"   📊 Efficiency improvement: {efficiency_improvement:.2f}x")
        print(f"   🎯 Output similarity: {torch.nn.functional.cosine_similarity(y_original.flatten(), y_optimized.flatten(), dim=0).item():.3f}")
        
        return {
            "success": True,
            "speedup": speedup_actual,
            "efficiency_improvement": efficiency_improvement,
            "original_efficiency": metrics_original.get('parallel_efficiency', 0),
            "optimized_efficiency": metrics_optimized.get('parallel_efficiency', 0)
        }
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def demo_wdm_scaling(device, max_wavelengths=16):
    """Demo 2: WDM scaling test."""
    print("\n" + "="*50)
    print("🌈 DEMO 2: WDM Scaling Test")
    print("="*50)
    
    try:
        from torchonn.onns.architectures import IncoherentONN
        
        layer_sizes = [8, 12, 6]
        batch_size = 16
        wavelength_counts = [1, 2, 4, 8]
        
        if max_wavelengths >= 16:
            wavelength_counts.append(16)
        
        print(f"📊 Testing wavelength scaling:")
        print(f"   Layers: {layer_sizes}")
        print(f"   Batch size: {batch_size}")
        print(f"   Wavelength counts: {wavelength_counts}")
        
        results = []
        
        for n_wl in wavelength_counts:
            print(f"\n📡 Testing {n_wl} wavelengths:")
            
            try:
                # Test with optimization enabled
                onn = IncoherentONN(
                    layer_sizes,
                    n_wavelengths=n_wl,
                    enable_wdm_optimization=True,
                    device=device
                )
                
                x = torch.randn(batch_size, layer_sizes[0], device=device)
                
                # Time forward pass
                start_time = time.time()
                y = onn(x)
                forward_time = time.time() - start_time
                
                # Get metrics
                metrics = onn.get_optical_efficiency_metrics()
                
                efficiency = metrics.get('parallel_efficiency', 0)
                theoretical_speedup = metrics.get('theoretical_speedup', 1.0)
                microrings = metrics.get('total_microrings', 0)
                
                print(f"   ⏱️ Time: {forward_time*1000:.2f}ms")
                print(f"   📊 Efficiency: {efficiency:.1f}%")
                print(f"   ⚡ Theoretical speedup: {theoretical_speedup:.2f}x")
                print(f"   💍 Microrings: {microrings:,}")
                
                results.append({
                    "wavelengths": n_wl,
                    "time_ms": forward_time * 1000,
                    "efficiency": efficiency,
                    "theoretical_speedup": theoretical_speedup,
                    "microrings": microrings,
                    "success": True
                })
                
                # Success indicator
                if efficiency > 70:
                    print(f"   ✅ Excellent efficiency!")
                elif efficiency > 50:
                    print(f"   ⚠️ Good efficiency")
                elif efficiency > 30:
                    print(f"   ⚠️ Moderate efficiency")
                else:
                    print(f"   ❌ Poor efficiency")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results.append({
                    "wavelengths": n_wl,
                    "error": str(e),
                    "success": False
                })
        
        # Analysis
        print(f"\n📈 WDM Scaling Analysis:")
        
        successful_results = [r for r in results if r.get("success", False)]
        
        if len(successful_results) >= 2:
            efficiencies = [r["efficiency"] for r in successful_results]
            wavelengths = [r["wavelengths"] for r in successful_results]
            speedups = [r["theoretical_speedup"] for r in successful_results]
            
            min_efficiency = min(efficiencies)
            max_efficiency = max(efficiencies)
            final_efficiency = successful_results[-1]["efficiency"]
            
            print(f"   📊 Efficiency range: {min_efficiency:.1f}% - {max_efficiency:.1f}%")
            print(f"   🌈 Wavelength range: {min(wavelengths)} - {max(wavelengths)}")
            print(f"   🎯 Final efficiency ({max(wavelengths)}wl): {final_efficiency:.1f}%")
            
            # Goal assessment
            if final_efficiency > 50.0:
                print(f"   ✅ SUCCESS: Efficiency >50% achieved at high wavelength count!")
            elif final_efficiency > 30.0:
                print(f"   ⚠️ GOOD: Significant improvement over original ~15%")
            else:
                print(f"   ❌ NEEDS WORK: Efficiency still low at high wavelength count")
            
            # Scaling behavior
            efficiency_drop = max_efficiency - min_efficiency
            if efficiency_drop < 20:
                print(f"   ✅ Good scaling: efficiency drop <20%")
            else:
                print(f"   ⚠️ Moderate scaling: efficiency drop {efficiency_drop:.1f}%")
        
        else:
            print(f"   ❌ Insufficient results for analysis")
        
        return {
            "success": len(successful_results) > 0,
            "results": results,
            "successful_results": successful_results
        }
        
    except Exception as e:
        print(f"❌ WDM scaling test failed: {e}")
        return {"success": False, "error": str(e)}

def demo_performance_comparison(device, batch_size=32):
    """Demo 3: Performance comparison."""
    print("\n" + "="*50)
    print("⚡ DEMO 3: Performance Comparison")
    print("="*50)
    
    try:
        from torchonn.onns.architectures import IncoherentONN
        
        # Test configuration
        layer_sizes = [16, 24, 16, 8]
        n_wavelengths = 8
        n_trials = 5
        
        print(f"📊 Performance benchmark:")
        print(f"   Layers: {layer_sizes}")
        print(f"   Wavelengths: {n_wavelengths}")
        print(f"   Batch size: {batch_size}")
        print(f"   Trials: {n_trials}")
        
        # Test original implementation
        print(f"\n🔧 Benchmarking original implementation...")
        onn_original = IncoherentONN(
            layer_sizes,
            n_wavelengths=n_wavelengths,
            enable_wdm_optimization=False,
            device=device
        )
        
        x = torch.randn(batch_size, layer_sizes[0], device=device)
        
        # Warmup
        with torch.no_grad():
            _ = onn_original(x)
        
        # Timing
        times_original = []
        for _ in range(n_trials):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            with torch.no_grad():
                y_original = onn_original(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times_original.append(time.time() - start)
        
        avg_time_original = np.mean(times_original) * 1000
        std_time_original = np.std(times_original) * 1000
        throughput_original = batch_size / (avg_time_original / 1000)
        
        metrics_original = onn_original.get_optical_efficiency_metrics()
        
        print(f"   ⏱️ Time: {avg_time_original:.2f}±{std_time_original:.2f}ms")
        print(f"   🚀 Throughput: {throughput_original:.1f} samples/sec")
        print(f"   📊 Efficiency: {metrics_original.get('parallel_efficiency', 0):.1f}%")
        
        # Test optimized implementation
        print(f"\n🚀 Benchmarking optimized implementation...")
        onn_optimized = IncoherentONN(
            layer_sizes,
            n_wavelengths=n_wavelengths,
            enable_wdm_optimization=True,
            device=device
        )
        
        # Warmup
        with torch.no_grad():
            _ = onn_optimized(x)
        
        # Timing
        times_optimized = []
        for _ in range(n_trials):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            with torch.no_grad():
                y_optimized = onn_optimized(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times_optimized.append(time.time() - start)
        
        avg_time_optimized = np.mean(times_optimized) * 1000
        std_time_optimized = np.std(times_optimized) * 1000
        throughput_optimized = batch_size / (avg_time_optimized / 1000)
        
        metrics_optimized = onn_optimized.get_optical_efficiency_metrics()
        
        print(f"   ⏱️ Time: {avg_time_optimized:.2f}±{std_time_optimized:.2f}ms")
        print(f"   🚀 Throughput: {throughput_optimized:.1f} samples/sec")
        print(f"   📊 Efficiency: {metrics_optimized.get('parallel_efficiency', 0):.1f}%")
        
        # Comparison
        speedup_actual = avg_time_original / avg_time_optimized if avg_time_optimized > 0 else 1.0
        throughput_improvement = throughput_optimized / throughput_original if throughput_original > 0 else 1.0
        efficiency_improvement = (metrics_optimized.get('parallel_efficiency', 0) / 
                                max(metrics_original.get('parallel_efficiency', 1), 1))
        
        print(f"\n📈 Performance Improvement:")
        print(f"   ⚡ Speed improvement: {speedup_actual:.2f}x")
        print(f"   🚀 Throughput improvement: {throughput_improvement:.2f}x")
        print(f"   📊 Efficiency improvement: {efficiency_improvement:.2f}x")
        
        # Assessment
        if speedup_actual > 2.0:
            print(f"   ✅ Excellent speed improvement!")
        elif speedup_actual > 1.5:
            print(f"   ⚠️ Good speed improvement")
        elif speedup_actual > 1.2:
            print(f"   ⚠️ Moderate speed improvement")
        else:
            print(f"   ❌ Limited speed improvement")
        
        return {
            "success": True,
            "speedup": speedup_actual,
            "throughput_improvement": throughput_improvement,
            "efficiency_improvement": efficiency_improvement,
            "original_time_ms": avg_time_original,
            "optimized_time_ms": avg_time_optimized,
            "original_efficiency": metrics_original.get('parallel_efficiency', 0),
            "optimized_efficiency": metrics_optimized.get('parallel_efficiency', 0)
        }
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return {"success": False, "error": str(e)}

def demo_direct_optimization_api(device):
    """Demo 4: Direct optimization API test."""
    print("\n" + "="*50)
    print("🔧 DEMO 4: Direct Optimization API")
    print("="*50)
    
    try:
        from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
        
        # Test direct usage of optimized API
        layer_sizes = [12, 16, 8]
        n_wavelengths = 8
        batch_size = 24
        
        print(f"📊 Testing direct OptimizedIncoherentONN:")
        print(f"   Layers: {layer_sizes}")
        print(f"   Wavelengths: {n_wavelengths}")
        print(f"   Batch size: {batch_size}")
        
        # Create optimized model directly
        onn = OptimizedIncoherentONN(layer_sizes, n_wavelengths, device)
        
        x = torch.randn(batch_size, layer_sizes[0], device=device)
        
        # Test forward pass
        start_time = time.time()
        y = onn(x)
        forward_time = time.time() - start_time
        
        print(f"   ✅ Forward pass: {x.shape} → {y.shape}")
        print(f"   ⏱️ Time: {forward_time*1000:.2f}ms")
        
        # Test advanced metrics
        metrics = onn.get_wdm_efficiency_metrics()
        
        print(f"\n📊 Advanced WDM Metrics:")
        print(f"   🌈 Wavelengths: {metrics['n_wavelengths']}")
        print(f"   💍 Total microrings: {metrics['total_microrings']:,}")
        print(f"   📷 Total photodetectors: {metrics['total_photodetectors']:,}")
        print(f"   🔧 Total parameters: {metrics['total_parameters']:,}")
        print(f"   📊 Parallel efficiency: {metrics['parallel_efficiency']:.1f}%")
        print(f"   ⚡ Theoretical speedup: {metrics['theoretical_speedup']:.2f}x")
        print(f"   💾 Memory efficiency: {metrics['memory_efficiency']:.2f}")
        print(f"   ⚙️ Compute efficiency: {metrics['compute_efficiency']:.2f}")
        print(f"   🎯 WDM scaling factor: {metrics['wdm_scaling_factor']:.2f}")
        print(f"   🔬 Optical fraction: {metrics['optical_fraction']:.3f}")
        
        # Test physics validation
        physics = onn.validate_physics()
        
        print(f"\n🔬 Physics Validation:")
        print(f"   ✅ Valid transmissions: {physics['valid_transmissions']}")
        print(f"   ⚖️ Energy conservation: {physics['energy_conservation']}")
        print(f"   ➕ Positive powers: {physics['positive_powers']}")
        print(f"   🔗 Realistic coupling: {physics['realistic_coupling']}")
        
        # Assessment
        efficiency = metrics['parallel_efficiency']
        if efficiency > 70:
            print(f"\n   ✅ EXCELLENT: High efficiency achieved!")
        elif efficiency > 50:
            print(f"\n   ⚠️ GOOD: Target efficiency achieved")
        elif efficiency > 30:
            print(f"\n   ⚠️ MODERATE: Reasonable efficiency")
        else:
            print(f"\n   ❌ POOR: Low efficiency")
        
        return {
            "success": True,
            "efficiency": efficiency,
            "theoretical_speedup": metrics['theoretical_speedup'],
            "total_microrings": metrics['total_microrings'],
            "forward_time_ms": forward_time * 1000,
            "physics_valid": all([
                physics['valid_transmissions'],
                physics['energy_conservation'],
                physics['positive_powers'],
                physics['realistic_coupling']
            ])
        }
        
    except ImportError:
        print("⚠️ Direct optimization API not available")
        print("   This is expected if optimizations are not installed")
        return {"success": False, "reason": "optimization_api_not_available"}
    except Exception as e:
        print(f"❌ Direct optimization API test failed: {e}")
        return {"success": False, "error": str(e)}

def generate_summary(results):
    """Generate final summary of all results."""
    print("\n" + "="*50)
    print("📋 FINAL SUMMARY")
    print("="*50)
    
    print("🎯 Demo Results:")
    
    # Demo 1: Basic functionality
    demo1 = results.get("demo1", {})
    if demo1.get("success", False):
        print(f"   ✅ Basic Functionality: {demo1.get('speedup', 1.0):.2f}x speedup, {demo1.get('efficiency_improvement', 1.0):.2f}x efficiency")
    else:
        print(f"   ❌ Basic Functionality: Failed")
    
    # Demo 2: WDM scaling
    demo2 = results.get("demo2", {})
    if demo2.get("success", False):
        successful_results = demo2.get("successful_results", [])
        if successful_results:
            final_result = successful_results[-1]
            final_efficiency = final_result.get("efficiency", 0)
            max_wavelengths = final_result.get("wavelengths", 0)
            print(f"   ✅ WDM Scaling: {final_efficiency:.1f}% efficiency at {max_wavelengths} wavelengths")
        else:
            print(f"   ⚠️ WDM Scaling: Limited results")
    else:
        print(f"   ❌ WDM Scaling: Failed")
    
    # Demo 3: Performance comparison
    demo3 = results.get("demo3", {})
    if demo3.get("success", False):
        print(f"   ✅ Performance: {demo3.get('speedup', 1.0):.2f}x faster, {demo3.get('efficiency_improvement', 1.0):.2f}x more efficient")
    else:
        print(f"   ❌ Performance: Failed")
    
    # Demo 4: Direct API
    demo4 = results.get("demo4", {})
    if demo4.get("success", False):
        print(f"   ✅ Direct API: {demo4.get('efficiency', 0):.1f}% efficiency, physics valid: {demo4.get('physics_valid', False)}")
    elif demo4.get("reason") == "optimization_api_not_available":
        print(f"   ⚠️ Direct API: Not available (expected)")
    else:
        print(f"   ❌ Direct API: Failed")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    
    successful_demos = sum([
        demo1.get("success", False),
        demo2.get("success", False),
        demo3.get("success", False),
        demo4.get("success", False) or demo4.get("reason") == "optimization_api_not_available"
    ])
    
    if successful_demos >= 3:
        print(f"   ✅ EXCELLENT: {successful_demos}/4 demos successful")
        
        # Check if we achieved main goals
        goals_achieved = []
        
        # Goal 1: WDM scaling >50% efficiency
        if demo2.get("success", False):
            successful_results = demo2.get("successful_results", [])
            if successful_results:
                final_efficiency = successful_results[-1].get("efficiency", 0)
                if final_efficiency > 50.0:
                    goals_achieved.append("WDM scaling >50%")
        
        # Goal 2: Significant speedup
        if demo3.get("success", False):
            speedup = demo3.get("speedup", 1.0)
            if speedup > 1.5:
                goals_achieved.append("Speed improvement >1.5x")
        
        # Goal 3: High efficiency in direct API
        if demo4.get("success", False):
            efficiency = demo4.get("efficiency", 0)
            if efficiency > 50.0:
                goals_achieved.append("Direct API >50% efficiency")
        
        if len(goals_achieved) >= 2:
            print(f"   🎉 MAJOR GOALS ACHIEVED: {', '.join(goals_achieved)}")
        elif len(goals_achieved) >= 1:
            print(f"   ⚠️ PARTIAL SUCCESS: {', '.join(goals_achieved)}")
        else:
            print(f"   ❌ GOALS NOT MET: Performance targets not achieved")
    
    elif successful_demos >= 2:
        print(f"   ⚠️ GOOD: {successful_demos}/4 demos successful")
        print(f"   📈 Partial optimization success")
    
    else:
        print(f"   ❌ POOR: Only {successful_demos}/4 demos successful")
        print(f"   🔧 Significant work needed")
    
    print(f"\n💡 Recommendations:")
    
    recommendations = []
    
    # Specific recommendations based on results
    if not demo1.get("success", False):
        recommendations.append("Fix basic IncoherentONN integration issues")
    
    if demo2.get("success", False):
        successful_results = demo2.get("successful_results", [])
        if successful_results:
            final_efficiency = successful_results[-1].get("efficiency", 0)
            if final_efficiency < 50.0:
                recommendations.append("Improve WDM scaling efficiency (target: >50%)")
    
    if demo3.get("success", False):
        speedup = demo3.get("speedup", 1.0)
        if speedup < 1.5:
            recommendations.append("Optimize for better speed improvements")
    
    if not demo4.get("success", False) and demo4.get("reason") != "optimization_api_not_available":
        recommendations.append("Fix direct optimization API issues")
    
    if not recommendations:
        recommendations.append("Continue monitoring and incremental improvements")
        recommendations.append("Consider deployment testing with real applications")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n✅ Quick demo completed!")

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Quick WDM Optimization Demo")
    parser.add_argument("--wavelengths", type=int, default=16, help="Max wavelengths to test")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for performance test")
    parser.add_argument("--skip-perf", action="store_true", help="Skip performance comparison")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only mode")
    
    args = parser.parse_args()
    
    print("🚀" * 25)
    print("🚀  QUICK WDM OPTIMIZATION DEMO  🚀")
    print("🚀" * 25)
    print("🔬 Quick validation of WDM optimization improvements")
    print("⚡ Focus: Verify that optimizations work as expected")
    print("🎯 Target: >50% WDM efficiency even at high wavelength counts")
    
    # Check dependencies
    deps_ok, optimizations_available = check_dependencies()
    if not deps_ok:
        print("\n❌ Dependency check failed. Cannot continue.")
        return 1
    
    if not optimizations_available:
        print("\n⚠️ WDM optimizations not available - testing compatibility mode only")
    
    # Setup device
    if args.cpu_only:
        device = torch.device("cpu")
        print("🖥️ Using CPU (forced)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {device}")
    
    # Run demos
    results = {}
    
    try:
        # Demo 1: Basic functionality
        results["demo1"] = demo_basic_functionality(device)
        
        # Demo 2: WDM scaling  
        results["demo2"] = demo_wdm_scaling(device, args.wavelengths)
        
        # Demo 3: Performance comparison (optional)
        if not args.skip_perf:
            results["demo3"] = demo_performance_comparison(device, args.batch)
        else:
            print("\n⏭️ Skipping performance comparison (--skip-perf)")
            results["demo3"] = {"success": False, "skipped": True}
        
        # Demo 4: Direct optimization API (if available)
        results["demo4"] = demo_direct_optimization_api(device)
        
        # Generate summary
        generate_summary(results)
        
        # Return success code
        successful_demos = sum([
            results["demo1"].get("success", False),
            results["demo2"].get("success", False),
            results["demo3"].get("success", False) or results["demo3"].get("skipped", False),
            results["demo4"].get("success", False) or results["demo4"].get("reason") == "optimization_api_not_available"
        ])
        
        return 0 if successful_demos >= 3 else 1
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Demo failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n🏁 Demo finished with exit code: {exit_code}")
    sys.exit(exit_code)

# ========================================
# SUMMARY OF QUICK DEMO
# ========================================

"""
🚀 QUICK WDM OPTIMIZATION DEMO SUMMARY:

PURPOSE:
✅ Rapid validation that WDM optimizations work correctly
✅ Easy-to-run verification of key improvements  
✅ Dependency checking and compatibility testing
✅ Performance target validation (>50% efficiency goal)

DEMO STRUCTURE:
1️⃣ Basic Functionality: Original vs Optimized comparison
2️⃣ WDM Scaling: Efficiency across wavelength counts (key test)
3️⃣ Performance: Speed and throughput improvements
4️⃣ Direct API: Advanced optimization API testing

KEY FEATURES:
🔍 Automatic dependency checking
⚙️ Configurable test parameters via command line
📊 Clear success/failure indicators for each demo
🎯 Target-oriented assessment (>50% efficiency goal)
💡 Specific recommendations based on results
⏭️ Optional performance testing (--skip-perf)

SUCCESS CRITERIA:
✅ Demo 2 (WDM Scaling): >50% efficiency at 16 wavelengths
✅ Demo 3 (Performance): >1.5x speedup improvement
✅ Demo 1 (Basic): Successful integration and forward pass
✅ Overall: ≥3/4 demos successful = EXCELLENT

USAGE EXAMPLES:
python demos/quick_demo_wdm.py                    # Standard run
python demos/quick_demo_wdm.py --wavelengths 8    # Test up to 8 wavelengths
python demos/quick_demo_wdm.py --batch 64         # Larger batch size
python demos/quick_demo_wdm.py --skip-perf        # Skip performance test
python demos/quick_demo_wdm.py --cpu-only         # Force CPU mode

EXPECTED RESULTS:
📊 WDM efficiency should stay >50% even at 16 wavelengths
⚡ Speed improvements of 1.5-3x over original implementation
🔧 All physics validations should pass
✅ 3-4/4 demos should succeed for overall success

This demo provides a quick way to validate that the WDM optimizations
are working correctly and achieving the target performance improvements.
"""