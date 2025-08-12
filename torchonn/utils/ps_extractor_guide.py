"""
GUÍA DE USO RÁPIDO - COPY & PASTE
=================================

Copia y pega estos fragmentos de código en tu proyecto para:
1. Entrenar cualquier modelo ONN
2. Extraer TODOS los valores theta y phi
3. Guardar y cargar estos valores
4. Usar para inferencia

REQUIERE: OpticalCI instalado
"""

# ============================================================================
# 1. IMPORTACIONES NECESARIAS (copiar siempre)
# ============================================================================
import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, Any

# Importaciones de OpticalCI
from torchonn.layers import MZILayer, MZIBlockLinear
from torchonn.onns.architectures import CoherentONN

# ============================================================================
# 2. FUNCIÓN RÁPIDA PARA EXTRAER VALORES (copiar esta función)
# ============================================================================
def extract_all_phase_shifters(model: nn.Module) -> Dict[str, Any]:
    """
    FUNCIÓN PRINCIPAL: Extrae TODOS los valores theta y phi del modelo.
    
    USO:
        values = extract_all_phase_shifters(my_trained_model)
    """
    result = {'layers': {}}
    layer_count = 0
    
    for name, module in model.named_modules():
        # Buscar capas MZI
        if hasattr(module, 'theta') and hasattr(module, 'phi'):
            layer_name = f"layer_{layer_count:02d}"
            
            # Extraer valores theta y phi
            with torch.no_grad():
                theta_vals = module.theta.detach().cpu().numpy()
                phi_vals = module.phi.detach().cpu().numpy()
            
            # Organizar por MZI
            mzis = {}
            for i in range(len(theta_vals)):
                mzis[f"MZI_{i:02d}"] = {
                    'theta_rad': float(theta_vals[i]),
                    'phi_rad': float(phi_vals[i]),
                    'theta_deg': float(np.degrees(theta_vals[i])),
                    'phi_deg': float(np.degrees(phi_vals[i]))
                }
            
            result['layers'][layer_name] = {
                'type': type(module).__name__,
                'dimensions': f"{getattr(module, 'in_features', '?')}x{getattr(module, 'out_features', '?')}",
                'n_mzis': len(theta_vals),
                'mzis': mzis
            }
            layer_count += 1
    
    result['total_layers'] = layer_count
    result['total_phase_shifters'] = sum(len(layer['mzis']) * 2 for layer in result['layers'].values())
    
    return result

# ============================================================================
# 3. FUNCIÓN PARA GUARDAR Y CARGAR (copiar estas funciones)
# ============================================================================
def save_phase_shifters(values: Dict[str, Any], filepath: str):
    """Guardar valores a archivo JSON."""
    with open(filepath, 'w') as f:
        json.dump(values, f, indent=2)
    print(f"💾 Guardado en: {filepath}")

def load_phase_shifters(filepath: str) -> Dict[str, Any]:
    """Cargar valores desde archivo JSON."""
    with open(filepath, 'r') as f:
        values = json.load(f)
    print(f"📁 Cargado desde: {filepath}")
    return values

def apply_phase_shifters(model: nn.Module, values: Dict[str, Any]):
    """Aplicar valores cargados a un modelo."""
    layers = values['layers']
    layer_count = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'theta') and hasattr(module, 'phi'):
            layer_name = f"layer_{layer_count:02d}"
            
            if layer_name in layers:
                layer_data = layers[layer_name]
                mzis = layer_data['mzis']
                
                # Extraer valores en orden
                theta_list = []
                phi_list = []
                for mzi_name in sorted(mzis.keys()):
                    theta_list.append(mzis[mzi_name]['theta_rad'])
                    phi_list.append(mzis[mzi_name]['phi_rad'])
                
                # Aplicar al modelo
                with torch.no_grad():
                    module.theta.copy_(torch.tensor(theta_list, device=module.device))
                    module.phi.copy_(torch.tensor(phi_list, device=module.device))
            
            layer_count += 1
    
    print(f"✅ Phase shifters aplicados al modelo")

# ============================================================================
# 4. EJEMPLO WORKFLOW COMPLETO (copiar y modificar según tu caso)
# ============================================================================
def mi_workflow_completo():
    """
    EJEMPLO DE WORKFLOW COMPLETO - Modifica según tu caso.
    """
    print("🚀 INICIANDO WORKFLOW COMPLETO")
    
    # PASO 1: Crear tu modelo (CAMBIAR SEGÚN TU ARQUITECTURA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Opción A: Modelo simple MZI
    modelo = MZILayer(in_features=4, out_features=4, device=device)
    
    # Opción B: Red más compleja (descomenta si usas CoherentONN)
    # modelo = CoherentONN(layer_sizes=[10, 20, 10], device=device)
    
    print(f"✅ Modelo creado: {type(modelo).__name__}")
    
    # PASO 2: [AQUÍ VA TU ENTRENAMIENTO]
    # - Carga tus datos
    # - Define pérdida y optimizador  
    # - Ejecuta bucle de entrenamiento
    print("📚 [AQUÍ ENTRENARÍAS TU MODELO]")
    print("   modelo.train()")
    print("   for epoch in range(epochs):")
    print("       # tu código de entrenamiento")
    print("✅ Modelo entrenado (simulado)")
    
    # PASO 3: Extraer valores de phase shifters
    print("\n🔍 Extrayendo phase shifters...")
    valores = extract_all_phase_shifters(modelo)
    
    # PASO 4: Mostrar resumen
    print(f"\n📊 RESUMEN:")
    print(f"   Capas con MZI: {valores['total_layers']}")  
    print(f"   Phase shifters totales: {valores['total_phase_shifters']}")
    
    # Mostrar valores de primera capa como ejemplo
    if 'layer_00' in valores['layers']:
        primera_capa = valores['layers']['layer_00']
        print(f"   Primera capa: {primera_capa['dimensions']}")
        
        mzis = primera_capa['mzis']
        for mzi_name, mzi_data in list(mzis.items())[:3]:  # Solo primeros 3
            theta_deg = mzi_data['theta_deg'] 
            phi_deg = mzi_data['phi_deg']
            print(f"      {mzi_name}: θ={theta_deg:6.1f}°, φ={phi_deg:6.1f}°")
    
    # PASO 5: Guardar valores
    nombre_archivo = "mi_modelo_entrenado_weights.json"
    save_phase_shifters(valores, nombre_archivo)
    
    # PASO 6: Demostrar carga en nuevo modelo (para inferencia)
    print(f"\n🔄 Probando inferencia con modelo nuevo...")
    modelo_inferencia = MZILayer(in_features=4, out_features=4, device=device)
    
    # Cargar valores guardados
    valores_cargados = load_phase_shifters(nombre_archivo)
    apply_phase_shifters(modelo_inferencia, valores_cargados)
    
    # Verificar que funcionan igual
    entrada_test = torch.randn(1, 4, device=device)
    salida_original = modelo(entrada_test)
    salida_cargada = modelo_inferencia(entrada_test)
    
    diferencia = torch.max(torch.abs(salida_original - salida_cargada))
    print(f"   Diferencia entre modelos: {diferencia:.2e}")
    print(f"   ✅ Inferencia funcionando: {diferencia < 1e-6}")
    
    print(f"\n🎉 ¡WORKFLOW COMPLETADO!")
    return valores

# ============================================================================
# 5. FUNCIONES ONE-LINER PARA USO RÁPIDO (copiar estas)
# ============================================================================

# Extraer y guardar en una línea
def quick_save_weights(model, filename="weights.json"):
    """USO: quick_save_weights(my_model, "my_weights.json")"""
    values = extract_all_phase_shifters(model)
    save_phase_shifters(values, filename)
    return values

# Cargar y aplicar en una línea  
def quick_load_weights(model, filename="weights.json"):
    """USO: quick_load_weights(my_model, "my_weights.json")"""
    values = load_phase_shifters(filename)
    apply_phase_shifters(model, values)
    return values

# Ver resumen rápido
def quick_summary(model):
    """USO: quick_summary(my_model)"""
    values = extract_all_phase_shifters(model)
    print(f"🔧 Capas: {values['total_layers']}, Phase shifters: {values['total_phase_shifters']}")
    return values

# ============================================================================
# 6. EJEMPLO SIMPLE 3x3 (el que pediste específicamente)
# ============================================================================
def ejemplo_3x3_simple():
    """
    EJEMPLO ESPECÍFICO: Red 3x3 con 3 MZIs.
    Esto es exactamente lo que pediste.
    """
    print("🎯 EJEMPLO 3x3 - 3 MZIs - 6 Phase Shifters")
    
    # Crear modelo 3x3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo_3x3 = MZILayer(in_features=3, out_features=3, device=device)
    
    print(f"📐 Modelo 3x3 creado:")
    print(f"   MZIs físicos: {modelo_3x3.n_mzis}")  # Será 3
    print(f"   Phase shifters: {modelo_3x3.get_phase_shifter_count()}")  # Será 6
    
    # Definir entrada específica
    entrada = torch.tensor([[1.0, 0.5, 0.2]], device=device)
    print(f"\n📥 Entrada: {entrada.squeeze().numpy()}")
    
    # Obtener salida
    salida = modelo_3x3(entrada)
    print(f"📤 Salida: {salida.squeeze().detach().cpu().numpy()}")
    
    # Extraer TODOS los valores theta y phi
    valores = extract_all_phase_shifters(modelo_3x3)
    
    # Mostrar valores específicos
    print(f"\n🔧 VALORES DE PHASE SHIFTERS:")
    mzis = valores['layers']['layer_00']['mzis']
    for mzi_name, mzi_data in mzis.items():
        theta_deg = mzi_data['theta_deg']
        phi_deg = mzi_data['phi_deg'] 
        theta_rad = mzi_data['theta_rad']
        phi_rad = mzi_data['phi_rad']
        print(f"   {mzi_name}: θ={theta_deg:6.1f}° ({theta_rad:.4f} rad), φ={phi_deg:6.1f}° ({phi_rad:.4f} rad)")
    
    # Verificar que es multiplicación matricial unitaria
    U = modelo_3x3.get_unitary_matrix().detach().cpu().numpy()
    entrada_np = entrada.squeeze().detach().cpu().numpy() 
    salida_manual = U @ entrada_np
    salida_modelo = salida.squeeze().detach().cpu().numpy()
    
    print(f"\n🧮 VERIFICACIÓN MULTIPLICACIÓN UNITARIA:")
    print(f"   Salida modelo: {salida_modelo}")
    print(f"   U × entrada:   {salida_manual}")
    print(f"   Son iguales:   {np.allclose(salida_modelo, salida_manual)}")
    
    # Guardar para uso posterior
    save_phase_shifters(valores, "ejemplo_3x3.json")
    
    return modelo_3x3, valores

# ============================================================================
# 7. FUNCIÓN PRINCIPAL DE PRUEBA
# ============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("🧪 PRUEBA DE FUNCIONES - GUÍA RÁPIDA")
    print("=" * 50)
    
    # Opción 1: Workflow completo
    print("1️⃣ Ejecutando workflow completo...")
    try:
        mi_workflow_completo()
    except Exception as e:
        print(f"⚠️ Error en workflow: {e}")
    
    print("\n" + "-" * 50)
    
    # Opción 2: Ejemplo 3x3 específico  
    print("2️⃣ Ejecutando ejemplo 3x3...")
    try:
        ejemplo_3x3_simple()
    except Exception as e:
        print(f"⚠️ Error en 3x3: {e}")
    
    print(f"\n🎉 Pruebas completadas!")

"""
============================================================================
INSTRUCCIONES DE USO:

1. COPIA LAS FUNCIONES que necesites (extract_all_phase_shifters es la principal)

2. PARA TU CASO DE USO:
   - Entrena tu modelo normalmente
   - Llama a: values = extract_all_phase_shifters(tu_modelo) 
   - Guarda: save_phase_shifters(values, "mi_archivo.json")
   - Carga: values = load_phase_shifters("mi_archivo.json")
   - Aplica: apply_phase_shifters(nuevo_modelo, values)

3. ONE-LINERS RÁPIDOS:
   - quick_save_weights(modelo, "archivo.json")  
   - quick_load_weights(modelo, "archivo.json")
   - quick_summary(modelo)

4. EJEMPLO 3x3:
   - Ejecuta ejemplo_3x3_simple() para ver exactamente lo que pediste

¡Con esto tienes TODO lo que necesitas para extraer y usar los valores theta y phi!
============================================================================
"""