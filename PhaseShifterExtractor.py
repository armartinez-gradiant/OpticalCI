"""
PhaseShifterExtractor.py
========================

Módulo reutilizable para extraer valores de phase shifters de modelos ONN entrenados.
Copia este archivo a tu proyecto y úsalo con cualquier modelo.

Autor: Para uso con OpticalCI
Fecha: 2025
"""

import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class PhaseShifterExtractor:
    """
    Extractor reutilizable de valores de phase shifters para modelos ONN.
    
    Uso básico:
        extractor = PhaseShifterExtractor()
        values = extractor.extract(model)
        extractor.save_to_file(values, "my_model_weights.json")
    """
    
    def __init__(self, verbose: bool = True):
        """
        Inicializar extractor.
        
        Args:
            verbose: Si mostrar información detallada
        """
        self.verbose = verbose
    
    def extract(self, model: nn.Module) -> Dict[str, Any]:
        """
        Extraer TODOS los valores de phase shifters del modelo.
        
        Args:
            model: Modelo ONN entrenado
            
        Returns:
            Diccionario completo con todos los valores
        """
        if self.verbose:
            print("🔍 Extrayendo phase shifters del modelo...")
        
        phase_shifters = {
            'model_info': {
                'model_class': model.__class__.__name__,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'device': str(next(model.parameters()).device),
            },
            'layers': {}
        }
        
        layer_count = 0
        total_phase_shifters = 0
        
        for name, module in model.named_modules():
            # Buscar capas MZI
            if self._is_mzi_layer(module):
                layer_name = f"layer_{layer_count:02d}" + (f"_{name}" if name else "")
                
                layer_info = self._extract_layer_info(module)
                phase_shifters['layers'][layer_name] = layer_info
                
                # Contar phase shifters
                if 'mzis' in layer_info:
                    total_phase_shifters += len(layer_info['mzis']) * 2
                elif 'phases' in layer_info:
                    total_phase_shifters += len(layer_info['phases'])
                
                layer_count += 1
        
        phase_shifters['summary'] = {
            'total_layers': layer_count,
            'total_phase_shifters': total_phase_shifters
        }
        
        if self.verbose:
            print(f"   ✅ Encontradas {layer_count} capas con {total_phase_shifters} phase shifters")
        
        return phase_shifters
    
    def _is_mzi_layer(self, module) -> bool:
        """Verificar si el módulo es una capa MZI."""
        return (
            hasattr(module, 'theta') and hasattr(module, 'phi')
        ) or (
            hasattr(module, 'phases')
        ) or (
            'MZI' in str(type(module))
        )
    
    def _extract_layer_info(self, module) -> Dict[str, Any]:
        """Extraer información detallada de una capa MZI."""
        layer_info = {
            'type': type(module).__name__,
            'in_features': getattr(module, 'in_features', None),
            'out_features': getattr(module, 'out_features', None),
        }
        
        # Caso 1: MZI con theta y phi (más común)
        if hasattr(module, 'theta') and hasattr(module, 'phi'):
            layer_info.update(self._extract_theta_phi(module))
        
        # Caso 2: MZI con phases generales
        elif hasattr(module, 'phases'):
            layer_info.update(self._extract_phases(module))
        
        return layer_info
    
    def _extract_theta_phi(self, module) -> Dict[str, Any]:
        """Extraer valores theta y phi específicos."""
        with torch.no_grad():
            theta_values = module.theta.detach().cpu().numpy()
            phi_values = module.phi.detach().cpu().numpy()
        
        mzis = {}
        for i in range(len(theta_values)):
            mzi_id = f"MZI_{i:02d}"
            mzis[mzi_id] = {
                'theta': {
                    'radians': float(theta_values[i]),
                    'degrees': float(np.degrees(theta_values[i])),
                    'normalized': float(theta_values[i] / (2 * np.pi))
                },
                'phi': {
                    'radians': float(phi_values[i]),
                    'degrees': float(np.degrees(phi_values[i])),
                    'normalized': float(phi_values[i] / (2 * np.pi))
                }
            }
        
        return {
            'n_mzis': len(theta_values),
            'mzis': mzis
        }
    
    def _extract_phases(self, module) -> Dict[str, Any]:
        """Extraer phases generales."""
        with torch.no_grad():
            phases = module.phases.detach().cpu().numpy()
        
        phase_dict = {}
        for i, phase_val in enumerate(phases):
            phase_dict[f"phase_{i:02d}"] = {
                'radians': float(phase_val),
                'degrees': float(np.degrees(phase_val)),
                'normalized': float(phase_val / (2 * np.pi))
            }
        
        return {
            'mode': getattr(module, 'mode', 'unknown'),
            'n_phases': len(phases),
            'phases': phase_dict
        }
    
    def save_to_file(self, phase_shifters: Dict[str, Any], filepath: str):
        """
        Guardar valores a archivo JSON.
        
        Args:
            phase_shifters: Valores extraídos
            filepath: Ruta del archivo
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(phase_shifters, f, indent=2)
        
        if self.verbose:
            print(f"💾 Guardado en: {filepath}")
    
    def load_from_file(self, filepath: str) -> Dict[str, Any]:
        """
        Cargar valores desde archivo JSON.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Valores cargados
        """
        with open(filepath, 'r') as f:
            phase_shifters = json.load(f)
        
        if self.verbose:
            print(f"📁 Cargado desde: {filepath}")
        
        return phase_shifters
    
    def print_summary(self, phase_shifters: Dict[str, Any]):
        """
        Imprimir resumen legible de los valores.
        
        Args:
            phase_shifters: Valores extraídos
        """
        print("\n" + "="*70)
        print("🔬 RESUMEN DE PHASE SHIFTERS")
        print("="*70)
        
        # Info del modelo
        model_info = phase_shifters.get('model_info', {})
        print(f"📱 Modelo: {model_info.get('model_class', 'Unknown')}")
        print(f"🔢 Parámetros totales: {model_info.get('total_parameters', 0):,}")
        print(f"💻 Device: {model_info.get('device', 'Unknown')}")
        
        # Info de capas
        layers = phase_shifters.get('layers', {})
        summary = phase_shifters.get('summary', {})
        
        print(f"🏗️ Capas totales: {summary.get('total_layers', 0)}")
        print(f"🔧 Phase shifters totales: {summary.get('total_phase_shifters', 0)}")
        
        # Detalles por capa
        for layer_name, layer_info in layers.items():
            print(f"\n📐 {layer_name.upper()}")
            print(f"   Tipo: {layer_info['type']}")
            print(f"   Dimensión: {layer_info.get('in_features', '?')}→{layer_info.get('out_features', '?')}")
            
            # Mostrar MZIs
            if 'mzis' in layer_info:
                mzis = layer_info['mzis']
                print(f"   MZIs: {len(mzis)}")
                
                # Mostrar primeros 3 MZIs
                for i, (mzi_name, mzi_data) in enumerate(mzis.items()):
                    if i >= 3:  # Limitar salida
                        print(f"      ... y {len(mzis) - 3} MZIs más")
                        break
                    
                    theta_deg = mzi_data['theta']['degrees']
                    phi_deg = mzi_data['phi']['degrees']
                    print(f"      {mzi_name}: θ={theta_deg:6.1f}°, φ={phi_deg:6.1f}°")
            
            # Mostrar phases
            elif 'phases' in layer_info:
                phases = layer_info['phases']
                print(f"   Phases: {len(phases)} (modo: {layer_info.get('mode', '?')})")
        
        print("="*70)
    
    def apply_to_model(self, model: nn.Module, phase_shifters: Dict[str, Any]):
        """
        Aplicar valores cargados a un modelo (para inferencia).
        
        Args:
            model: Modelo objetivo
            phase_shifters: Valores a aplicar
        """
        if self.verbose:
            print("🎯 Aplicando phase shifters al modelo...")
        
        layers = phase_shifters.get('layers', {})
        layer_count = 0
        applied_count = 0
        
        for name, module in model.named_modules():
            if self._is_mzi_layer(module):
                layer_name = f"layer_{layer_count:02d}" + (f"_{name}" if name else "")
                
                if layer_name in layers:
                    layer_data = layers[layer_name]
                    
                    # Aplicar theta y phi
                    if 'mzis' in layer_data and hasattr(module, 'theta') and hasattr(module, 'phi'):
                        mzis = layer_data['mzis']
                        
                        theta_values = []
                        phi_values = []
                        
                        for mzi_name in sorted(mzis.keys()):
                            mzi_data = mzis[mzi_name]
                            theta_values.append(mzi_data['theta']['radians'])
                            phi_values.append(mzi_data['phi']['radians'])
                        
                        with torch.no_grad():
                            module.theta.copy_(torch.tensor(theta_values, device=module.device))
                            module.phi.copy_(torch.tensor(phi_values, device=module.device))
                        
                        applied_count += 1
                
                layer_count += 1
        
        if self.verbose:
            print(f"   ✅ Aplicado a {applied_count}/{layer_count} capas")
    
    def get_unitary_matrix(self, model: nn.Module, layer_index: int = 0) -> np.ndarray:
        """
        Obtener la matriz unitaria de una capa específica.
        
        Args:
            model: Modelo
            layer_index: Índice de la capa
            
        Returns:
            Matriz unitaria como numpy array
        """
        layer_count = 0
        for name, module in model.named_modules():
            if self._is_mzi_layer(module):
                if layer_count == layer_index:
                    if hasattr(module, 'get_unitary_matrix'):
                        with torch.no_grad():
                            U = module.get_unitary_matrix()
                            return U.detach().cpu().numpy()
                    elif hasattr(module, '_construct_unitary_matrix'):
                        with torch.no_grad():
                            U = module._construct_unitary_matrix()
                            return U.detach().cpu().numpy()
                layer_count += 1
        
        raise ValueError(f"No se encontró capa MZI en índice {layer_index}")

# FUNCIONES DE CONVENIENCIA PARA COPIAR Y PEGAR
def quick_extract(model: nn.Module, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Función rápida para extraer phase shifters.
    
    USO:
        values = quick_extract(my_model, "weights.json")
    """
    extractor = PhaseShifterExtractor()
    values = extractor.extract(model)
    extractor.print_summary(values)
    
    if save_path:
        extractor.save_to_file(values, save_path)
    
    return values

def quick_apply(model: nn.Module, weights_file: str):
    """
    Función rápida para aplicar phase shifters.
    
    USO:
        quick_apply(my_model, "weights.json")
    """
    extractor = PhaseShifterExtractor()
    values = extractor.load_from_file(weights_file)
    extractor.apply_to_model(model, values)

# EJEMPLO DE USO
if __name__ == "__main__":
    print("🧪 PhaseShifterExtractor - Módulo de prueba")
    print("Para usar este módulo, impórtalo en tu código:")
    print()
    print("from PhaseShifterExtractor import quick_extract, PhaseShifterExtractor")
    print()
    print("# Extraer valores")
    print("values = quick_extract(my_trained_model, 'my_weights.json')")
    print()
    print("# Aplicar valores")  
    print("quick_apply(my_inference_model, 'my_weights.json')")