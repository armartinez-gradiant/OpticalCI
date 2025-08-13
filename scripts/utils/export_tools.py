"""
Herramientas de Exportación - OpticalCI Utils
=============================================

Utilidades para exportar resultados a diferentes formatos.
"""

import json
import csv
import numpy as np
from pathlib import Path

def export_phase_shifters_to_csv(phase_shifters_dict, output_path):
    """Exportar phase shifters a formato CSV."""
    output_path = Path(output_path)
    
    # Preparar datos para CSV
    csv_data = []
    
    layers = phase_shifters_dict.get('layers', {})
    for layer_name, layer_info in layers.items():
        mzis = layer_info.get('mzis', {})
        
        for mzi_name, mzi_data in mzis.items():
            csv_data.append({
                'layer': layer_name,
                'mzi': mzi_name,
                'theta_rad': mzi_data['theta']['radians'],
                'theta_deg': mzi_data['theta']['degrees'],
                'phi_rad': mzi_data['phi']['radians'], 
                'phi_deg': mzi_data['phi']['degrees']
            })
    
    # Escribir CSV
    with open(output_path, 'w', newline='') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
    
    print(f"📄 CSV exportado: {output_path}")

def export_for_hardware(phase_shifters_dict, output_path, voltage_conversion=0.5):
    """Exportar configuración para implementación en hardware."""
    output_path = Path(output_path)
    
    hardware_config = {
        'metadata': {
            'format_version': '1.0',
            'voltage_conversion_factor': voltage_conversion,
            'units': {
                'phase': 'radians',
                'voltage': 'volts'
            }
        },
        'phase_shifters': []
    }
    
    layers = phase_shifters_dict.get('layers', {})
    for layer_name, layer_info in layers.items():
        mzis = layer_info.get('mzis', {})
        
        for mzi_name, mzi_data in mzis.items():
            # Convertir fases a voltajes (ejemplo)
            theta_voltage = mzi_data['theta']['radians'] * voltage_conversion
            phi_voltage = mzi_data['phi']['radians'] * voltage_conversion
            
            hardware_config['phase_shifters'].append({
                'id': f"{layer_name}_{mzi_name}",
                'layer': layer_name,
                'mzi': mzi_name,
                'upper_arm_voltage': float(theta_voltage),
                'lower_arm_voltage': float(phi_voltage),
                'phase_rad': {
                    'theta': mzi_data['theta']['radians'],
                    'phi': mzi_data['phi']['radians']
                }
            })
    
    # Guardar como JSON
    with open(output_path, 'w') as f:
        json.dump(hardware_config, f, indent=2)
    
    print(f"🔧 Configuración de hardware exportada: {output_path}")