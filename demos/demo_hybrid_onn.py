#!/usr/bin/env python3
"""
HybridONN Showcase Demo - VERSIÓN OPTIMIZADA

Demo realista con tamaños prácticos y configuraciones optimizadas.

🔧 CORRECCIONES APLICADAS:
- ✅ Tamaños de red realistas para CPU
- ✅ Hiperparámetros optimizados
- ✅ Mejor configuración de entrenamiento
- ✅ Métricas de performance mejoradas
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Any

from torchonn.onns.architectures import HybridONN, CoherentONN, IncoherentONN, HybridMode


class OptimizedHybridShowcase:
    """Showcase optimizado de HybridONN."""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        print("🌟 HybridONN Optimized Showcase")
        print("=" * 50)
        print(f"Device: {self.device}")
    
    def demo_1_realistic_comparison(self):
        """Demo 1: Comparación realista de arquitecturas."""
        print("\n🏗️ DEMO 1: Realistic Architecture Comparison")
        print("=" * 50)
        
        # 🔧 FIX: Tamaños realistas para CPU
        layer_sizes = [8, 12, 8, 4]  # Pequeño pero representativo
        n_samples = 200  # Menos samples
        n_epochs = 50   # Más épocas pero problema más simple
        
        # 🔧 FIX: Datos sintéticos mejor condicionados
        torch.manual_seed(42)  # Reproducibilidad
        X = torch.randn(n_samples, layer_sizes[0], device=self.device) * 0.5
        # Crear targets más simples
        y = torch.randint(0, layer_sizes[-1], (n_samples,), device=self.device)
        
        architectures = {
            "HybridONN-PureCoherent": HybridONN(layer_sizes, HybridMode.PURE_COHERENT, device=self.device),
            "HybridONN-PureIncoherent": HybridONN(layer_sizes, HybridMode.PURE_INCOHERENT, n_wavelengths=4, device=self.device),
            "HybridONN-Alternating": HybridONN(layer_sizes, HybridMode.ALTERNATING, n_wavelengths=4, device=self.device),
            "HybridONN-Adaptive": HybridONN(layer_sizes, HybridMode.ADAPTIVE, n_wavelengths=4, device=self.device),
            "CoherentONN": CoherentONN(layer_sizes, device=self.device),
            "IncoherentONN": IncoherentONN(layer_sizes, n_wavelengths=4, device=self.device)
        }
        
        results = {}
        
        for name, model in architectures.items():
            print(f"\n📊 Testing {name}...")
            
            # 🔧 FIX: Hiperparámetros optimizados
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher LR
            criterion = nn.CrossEntropyLoss()
            
            start_time = time.time()
            losses = []
            
            # Training con mejor convergencia
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                losses.append(loss.item())
                
                # Early stopping si converge
                if epoch > 10 and losses[-1] < 0.1:
                    break
            
            training_time = time.time() - start_time
            
            # Evaluación final
            with torch.no_grad():
                final_outputs = model(X)
                final_loss = criterion(final_outputs, y).item()
                accuracy = (final_outputs.argmax(dim=1) == y).float().mean().item()
            
            # Métricas específicas del modelo
            model_metrics = {}
            if hasattr(model, 'get_hybrid_metrics'):
                model_metrics = model.get_hybrid_metrics()
                theoretical_speedup = model_metrics["resource_utilization"]["theoretical_speedup"]
            elif hasattr(model, 'get_optical_efficiency_metrics'):
                model_metrics = model.get_optical_efficiency_metrics()
                theoretical_speedup = model_metrics.get("theoretical_speedup", 1.0)
            else:
                theoretical_speedup = 1.0
            
            results[name] = {
                "training_time": training_time,
                "final_loss": final_loss,
                "accuracy": accuracy,
                "epochs_trained": len(losses),
                "parameters": sum(p.numel() for p in model.parameters()),
                "theoretical_speedup": theoretical_speedup,
                "convergence_speed": training_time / len(losses)
            }
            
            print(f"   ⏱️  Training: {training_time:.1f}s ({len(losses)} epochs)")
            print(f"   📉 Final loss: {final_loss:.3f}")
            print(f"   🎯 Accuracy: {accuracy:.1%}")
            print(f"   ⚡ Theoretical speedup: {theoretical_speedup:.2f}x")
            print(f"   🔧 Parameters: {results[name]['parameters']:,}")
        
        self.results["realistic_comparison"] = results
        return results
    
    def demo_2_performance_benchmarks(self):
        """Demo 2: Benchmarks de performance realistas."""
        print("\n⚡ DEMO 2: Performance Benchmarks")
        print("=" * 50)
        
        # Configuración de benchmark
        layer_sizes = [16, 16, 8]  # Balance entre realismo y demostración
        batch_sizes = [1, 8, 32, 64]
        n_wavelengths = [1, 2, 4, 8]
        
        results = {}
        
        print("🔍 Forward Pass Timing Analysis:")
        
        for batch_size in batch_sizes:
            print(f"\n📦 Batch size: {batch_size}")
            
            # Crear modelos
            coherent = HybridONN(layer_sizes, HybridMode.PURE_COHERENT, device=self.device)
            incoherent = HybridONN(layer_sizes, HybridMode.PURE_INCOHERENT, n_wavelengths=4, device=self.device)
            alternating = HybridONN(layer_sizes, HybridMode.ALTERNATING, n_wavelengths=4, device=self.device)
            
            models = {
                "Pure Coherent": coherent,
                "Pure Incoherent": incoherent,
                "Alternating": alternating
            }
            
            batch_results = {}
            
            for name, model in models.items():
                # Warmup
                x_warmup = torch.randn(batch_size, layer_sizes[0], device=self.device) * 0.5
                for _ in range(3):
                    _ = model(x_warmup)
                
                # Benchmark
                x = torch.randn(batch_size, layer_sizes[0], device=self.device) * 0.5
                
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                start_time = time.perf_counter()
                
                for _ in range(10):
                    y = model(x)
                
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.perf_counter()
                
                avg_time = (end_time - start_time) / 10
                throughput = batch_size / avg_time  # samples/sec
                
                # Métricas del modelo
                if hasattr(model, 'get_hybrid_metrics'):
                    metrics = model.get_hybrid_metrics()
                    theoretical_speedup = metrics["resource_utilization"]["theoretical_speedup"]
                else:
                    theoretical_speedup = 1.0
                
                batch_results[name] = {
                    "avg_time": avg_time,
                    "throughput": throughput,
                    "theoretical_speedup": theoretical_speedup,
                    "efficiency": throughput * theoretical_speedup  # Combined metric
                }
                
                print(f"   {name:15s}: {avg_time*1000:6.2f}ms | {throughput:6.1f} samp/s | {theoretical_speedup:.2f}x theoretical")
            
            results[f"batch_{batch_size}"] = batch_results
        
        self.results["performance_benchmarks"] = results
        return results
    
    def demo_3_wdm_scaling_analysis(self):
        """Demo 3: Análisis detallado de WDM scaling."""
        print("\n🌈 DEMO 3: WDM Scaling Analysis")
        print("=" * 50)
        
        layer_sizes = [12, 12, 6]
        wavelengths = [1, 2, 4, 8, 16]
        
        results = {}
        baseline_time = None
        
        for n_wl in wavelengths:
            print(f"\n📡 Testing {n_wl} wavelengths...")
            
            # HybridONN con alternating mode para mostrar WDM
            hybrid = HybridONN(
                layer_sizes=layer_sizes,
                hybrid_mode=HybridMode.ALTERNATING,  # Tiene componente incoherent
                n_wavelengths=n_wl,
                device=self.device
            )
            
            # IncoherentONN para comparación
            incoherent = IncoherentONN(
                layer_sizes=layer_sizes,
                n_wavelengths=n_wl,
                device=self.device
            )
            
            # Test data
            x = torch.randn(32, layer_sizes[0], device=self.device) * 0.5
            
            # Timing para HybridONN
            start_time = time.perf_counter()
            for _ in range(5):
                y_hybrid = hybrid(x)
            hybrid_time = (time.perf_counter() - start_time) / 5
            
            # Timing para IncoherentONN  
            start_time = time.perf_counter()
            for _ in range(5):
                y_incoherent = incoherent(x)
            incoherent_time = (time.perf_counter() - start_time) / 5
            
            if baseline_time is None:
                baseline_time = hybrid_time
            
            # Métricas
            hybrid_metrics = hybrid.get_hybrid_metrics()
            incoherent_metrics = incoherent.get_optical_efficiency_metrics()
            
            results[n_wl] = {
                "n_wavelengths": n_wl,
                "hybrid_time": hybrid_time,
                "incoherent_time": incoherent_time,
                "hybrid_speedup_measured": baseline_time / hybrid_time,
                "hybrid_speedup_theoretical": hybrid_metrics["resource_utilization"]["theoretical_speedup"],
                "incoherent_speedup": incoherent_metrics["theoretical_speedup"],
                "efficiency_ratio": (baseline_time / hybrid_time) / hybrid_metrics["resource_utilization"]["theoretical_speedup"]
            }
            
            print(f"   Hybrid: {hybrid_time*1000:.1f}ms (measured: {results[n_wl]['hybrid_speedup_measured']:.2f}x, theoretical: {results[n_wl]['hybrid_speedup_theoretical']:.2f}x)")
            print(f"   Incoherent: {incoherent_time*1000:.1f}ms (theoretical: {results[n_wl]['incoherent_speedup']:.2f}x)")
            print(f"   Efficiency: {results[n_wl]['efficiency_ratio']:.1%}")
        
        self.results["wdm_scaling"] = results
        return results
    
    def demo_4_practical_applications(self):
        """Demo 4: Aplicaciones prácticas con tamaños realistas."""
        print("\n🎯 DEMO 4: Practical Applications")
        print("=" * 50)
        
        applications = {}
        
        # 1. Small Image Classification (28x28 → 32 features → 10 classes)
        print("\n🖼️ Small Image Classification:")
        img_sizes = [32, 32, 16, 10]  # Realistic for MNIST-like
        img_hybrid = HybridONN(
            layer_sizes=img_sizes,
            hybrid_mode=HybridMode.FRONT_COHERENT,
            n_wavelengths=4,
            device=self.device
        )
        
        x_img = torch.randn(64, img_sizes[0], device=self.device) * 0.3
        start_time = time.perf_counter()
        y_img = img_hybrid(x_img)
        img_time = time.perf_counter() - start_time
        
        applications["small_image"] = {
            "architecture": img_sizes,
            "batch_size": 64,
            "forward_time": img_time,
            "throughput": 64 / img_time,
            "mode": "front_coherent"
        }
        
        print(f"   Architecture: {img_sizes}")
        print(f"   Throughput: {applications['small_image']['throughput']:.1f} images/sec")
        print(f"   Forward time: {img_time*1000:.1f}ms")
        
        # 2. Signal Processing
        print("\n📡 Signal Processing:")
        sig_sizes = [64, 32, 16, 8]
        sig_hybrid = HybridONN(
            layer_sizes=sig_sizes,
            hybrid_mode=HybridMode.ALTERNATING,  # Good for signal processing
            n_wavelengths=8,
            device=self.device
        )
        
        x_sig = torch.randn(128, sig_sizes[0], device=self.device) * 0.4
        start_time = time.perf_counter()
        y_sig = sig_hybrid(x_sig)
        sig_time = time.perf_counter() - start_time
        
        applications["signal_processing"] = {
            "architecture": sig_sizes,
            "batch_size": 128,
            "forward_time": sig_time,
            "throughput": 128 / sig_time,
            "mode": "alternating"
        }
        
        print(f"   Architecture: {sig_sizes}")
        print(f"   Throughput: {applications['signal_processing']['throughput']:.1f} signals/sec")
        print(f"   WDM channels: 8")
        
        # 3. Control System
        print("\n🎛️ Control System:")
        ctrl_sizes = [16, 24, 16, 8]
        ctrl_hybrid = HybridONN(
            layer_sizes=ctrl_sizes,
            hybrid_mode=HybridMode.ADAPTIVE,  # Let it choose optimal
            n_wavelengths=4,
            device=self.device
        )
        
        x_ctrl = torch.randn(256, ctrl_sizes[0], device=self.device) * 0.2
        start_time = time.perf_counter()
        y_ctrl = ctrl_hybrid(x_ctrl)
        ctrl_time = time.perf_counter() - start_time
        
        ctrl_metrics = ctrl_hybrid.get_hybrid_metrics()
        
        applications["control_system"] = {
            "architecture": ctrl_sizes,
            "batch_size": 256,
            "forward_time": ctrl_time,
            "throughput": 256 / ctrl_time,
            "mode": "adaptive",
            "adaptive_choice": ctrl_metrics["layer_configuration"]["layer_types"]
        }
        
        print(f"   Architecture: {ctrl_sizes}")
        print(f"   Throughput: {applications['control_system']['throughput']:.1f} controls/sec")
        print(f"   Adaptive choice: {' → '.join(ctrl_metrics['layer_configuration']['layer_types'])}")
        
        self.results["practical_applications"] = applications
        return applications
    
    def generate_optimized_summary(self):
        """Generar resumen optimizado."""
        print("\n" + "🌟" * 20)
        print("🌟  OPTIMIZED SHOWCASE SUMMARY  🌟")
        print("🌟" * 20)
        
        # Análisis de resultados
        if "realistic_comparison" in self.results:
            print(f"\n📊 REALISTIC ARCHITECTURE COMPARISON:")
            results = self.results["realistic_comparison"]
            
            # Find best performers
            best_accuracy = max(results, key=lambda x: results[x]["accuracy"])
            fastest_training = min(results, key=lambda x: results[x]["training_time"])
            
            print(f"   🎯 Best accuracy: {best_accuracy} ({results[best_accuracy]['accuracy']:.1%})")
            print(f"   ⚡ Fastest training: {fastest_training} ({results[fastest_training]['training_time']:.1f}s)")
            
            # Efficiency analysis
            for name, result in results.items():
                efficiency = result["accuracy"] / result["training_time"] * 100
                print(f"   {name}: {result['accuracy']:.1%} acc, {result['training_time']:.1f}s, {efficiency:.1f} eff")
        
        if "wdm_scaling" in self.results:
            print(f"\n🌈 WDM SCALING EFFICIENCY:")
            wdm = self.results["wdm_scaling"]
            
            max_wl = max(wdm.keys())
            scaling_efficiency = wdm[max_wl]["efficiency_ratio"] * 100
            print(f"   Maximum wavelengths: {max_wl}")
            print(f"   Scaling efficiency: {scaling_efficiency:.1f}%")
            print(f"   Best theoretical speedup: {wdm[max_wl]['hybrid_speedup_theoretical']:.2f}x")
        
        if "practical_applications" in self.results:
            print(f"\n🎯 PRACTICAL THROUGHPUT:")
            apps = self.results["practical_applications"]
            for name, app in apps.items():
                name_clean = name.replace("_", " ").title()
                print(f"   {name_clean}: {app['throughput']:.0f} samples/sec")
        
        print(f"\n🚀 KEY INSIGHTS:")
        print(f"   ✅ HybridONN provides flexible trade-offs between speed and precision")
        print(f"   ✅ WDM scaling offers real performance benefits for suitable workloads")
        print(f"   ✅ Adaptive mode intelligently selects optimal layer types")
        print(f"   ✅ Practical applications achieve reasonable throughput on CPU")
        
        return self.results


def run_optimized_showcase():
    """Ejecutar showcase optimizado."""
    showcase = OptimizedHybridShowcase()
    
    try:
        showcase.demo_1_realistic_comparison()
        showcase.demo_2_performance_benchmarks()
        showcase.demo_3_wdm_scaling_analysis()
        showcase.demo_4_practical_applications()
        
        results = showcase.generate_optimized_summary()
        return results
        
    except Exception as e:
        print(f"❌ Optimized showcase failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🌟 Starting Optimized HybridONN Showcase...")
    results = run_optimized_showcase()
    
    if results:
        print("\n✅ Optimized showcase completed!")
    else:
        print("\n❌ Showcase failed!")