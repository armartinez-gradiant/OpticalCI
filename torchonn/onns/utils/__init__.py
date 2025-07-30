"""
Utils Module for ONNs

Utilidades específicas para redes neuronales ópticas.
Incluye herramientas para análisis, conversión de datos, y validación física.

Utilidades Disponibles:
- Matrix decomposition tools para convertir pesos a operaciones ópticas
- Métricas específicas para ONNs
- Validadores de propiedades físicas
- Conversores de formato

Funciones Helper:
✅ Análisis de eficiencia óptica
✅ Validación de conservación de energía
✅ Métricas de performance específicas para fotónica
✅ Tools para debugging de ONNs
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Utilidades básicas para ONNs
def analyze_onn_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analizar performance de una ONN desde resultados de entrenamiento.
    
    Args:
        results: Resultados de benchmark o entrenamiento
        
    Returns:
        Análisis detallado de performance
    """
    analysis = {}
    
    # Accuracy analysis
    if "test_accuracy" in results:
        acc = results["test_accuracy"]
        if acc > 90:
            analysis["accuracy_grade"] = "Excellent"
        elif acc > 80:
            analysis["accuracy_grade"] = "Good"
        elif acc > 70:
            analysis["accuracy_grade"] = "Fair"
        else:
            analysis["accuracy_grade"] = "Poor"
        
        analysis["accuracy_value"] = acc
    
    # Physics validation
    if "physics_violations" in results:
        violations = results["n_physics_violations"]
        analysis["physics_compliant"] = violations == 0
        analysis["physics_violations"] = violations
    
    # Efficiency metrics
    if "optical_efficiency" in results:
        eff = results["optical_efficiency"]
        analysis["optical_fraction"] = eff.get("optical_fraction", 0)
        analysis["optical_operations"] = eff.get("optical_operations", 0)
    
    return analysis

def validate_onn_energy_conservation(
    input_power: torch.Tensor,
    output_power: torch.Tensor,
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Validar conservación de energía en ONN.
    
    Args:
        input_power: Potencia de entrada
        output_power: Potencia de salida
        tolerance: Tolerancia permitida
        
    Returns:
        Resultados de validación
    """
    with torch.no_grad():
        total_input = torch.sum(input_power)
        total_output = torch.sum(output_power)
        
        if total_input > 1e-10:
            energy_ratio = total_output / total_input
            conservation_error = abs(energy_ratio - 1.0)
            
            return {
                "energy_conserved": conservation_error < tolerance,
                "energy_ratio": energy_ratio.item(),
                "conservation_error": conservation_error.item(),
                "tolerance_used": tolerance
            }
        else:
            return {
                "energy_conserved": False,
                "energy_ratio": 0.0,
                "conservation_error": float('inf'),
                "error": "Input power too low"
            }

def compare_onn_vs_ann(onn_results: Dict, ann_results: Dict) -> Dict[str, Any]:
    """
    Comparar resultados de ONN vs ANN.
    
    Args:
        onn_results: Resultados de ONN
        ann_results: Resultados de ANN
        
    Returns:
        Análisis comparativo
    """
    comparison = {}
    
    # Accuracy comparison
    onn_acc = onn_results.get("test_accuracy", 0)
    ann_acc = ann_results.get("test_accuracy", 0)
    
    comparison["accuracy_parity"] = onn_acc / max(ann_acc, 1e-6)
    comparison["accuracy_difference"] = onn_acc - ann_acc
    
    # Speed comparison  
    onn_time = onn_results.get("avg_inference_time", float('inf'))
    ann_time = ann_results.get("avg_inference_time", float('inf'))
    comparison["speedup"] = ann_time / max(onn_time, 1e-9)
    
    # Training efficiency
    onn_train_time = onn_results.get("total_training_time", float('inf'))
    ann_train_time = ann_results.get("total_training_time", float('inf'))
    comparison["training_efficiency"] = ann_train_time / max(onn_train_time, 1e-6)
    
    # Overall assessment
    if comparison["accuracy_parity"] > 0.95:
        comparison["performance_assessment"] = "ONN matches ANN performance"
    elif comparison["accuracy_parity"] > 0.85:
        comparison["performance_assessment"] = "ONN performance acceptable" 
    else:
        comparison["performance_assessment"] = "ONN underperforms ANN"
    
    return comparison

def calculate_optical_efficiency_metrics(model) -> Dict[str, Any]:
    """
    Calcular métricas de eficiencia óptica para un modelo.
    
    Args:
        model: Modelo ONN
        
    Returns:
        Métricas de eficiencia
    """
    metrics = {}
    
    if hasattr(model, 'get_optical_efficiency'):
        # Usar método del modelo si está disponible
        metrics = model.get_optical_efficiency()
    else:
        # Calcular métricas básicas
        total_params = sum(p.numel() for p in model.parameters())
        metrics["total_parameters"] = total_params
        
        # Estimar operaciones ópticas vs eléctricas
        optical_layers = [m for m in model.modules() 
                         if hasattr(m, 'get_unitary_matrix') or 'MZI' in str(type(m))]
        
        metrics["n_optical_layers"] = len(optical_layers)
        metrics["has_optical_components"] = len(optical_layers) > 0
    
    return metrics

def generate_onn_report(results: Dict[str, Any]) -> str:
    """
    Generar reporte textual de resultados ONN.
    
    Args:
        results: Resultados de benchmark
        
    Returns:
        Reporte formateado como string
    """
    report = []
    report.append("📊 ONN Performance Report")
    report.append("=" * 50)
    
    # Configuration
    if "config" in results:
        config = results["config"]
        report.append(f"🔧 Configuration:")
        report.append(f"   Image size: {config.get('image_size', 'N/A')}")
        report.append(f"   Epochs: {config.get('n_epochs', 'N/A')}")
        report.append(f"   Device: {config.get('device', 'N/A')}")
        report.append("")
    
    # ONN Results
    if "onn" in results and "error" not in results["onn"]:
        onn = results["onn"]
        report.append(f"🌟 Coherent ONN Results:")
        report.append(f"   Test Accuracy: {onn.get('test_accuracy', 0):.2f}%")
        report.append(f"   Training Time: {onn.get('total_training_time', 0):.1f}s")
        
        if "physics_violations" in onn:
            violations = onn["n_physics_violations"]
            status = "✅ Compliant" if violations == 0 else f"⚠️ {violations} violations"
            report.append(f"   Physics Validation: {status}")
        
        report.append("")
    
    # ANN Comparison
    if "ann" in results and "error" not in results["ann"]:
        ann = results["ann"]
        report.append(f"🔌 Reference ANN Results:")
        report.append(f"   Test Accuracy: {ann.get('test_accuracy', 0):.2f}%")
        report.append(f"   Training Time: {ann.get('total_training_time', 0):.1f}s")
        report.append("")
    
    # Comparison
    if "comparison" in results:
        comp = results["comparison"]
        report.append(f"⚖️ Comparison Analysis:")
        report.append(f"   Accuracy Ratio: {comp.get('accuracy_ratio', 0):.3f}")
        report.append(f"   Speed Ratio: {comp.get('speed_ratio', 0):.3f}")
        
        if comp.get('accuracy_ratio', 0) > 0.9:
            report.append(f"   Status: ✅ ONN performance competitive")
        else:
            report.append(f"   Status: ⚠️ ONN needs improvement")
    
    return "\n".join(report)

# Funciones de análisis disponibles
__all__ = [
    "analyze_onn_performance",
    "validate_onn_energy_conservation", 
    "compare_onn_vs_ann",
    "calculate_optical_efficiency_metrics",
    "generate_onn_report"
]