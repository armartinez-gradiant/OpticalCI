"""
Ejemplo Completo: Red Óptica 3x3 con 3 MZIs
===========================================

Este ejemplo demuestra:
1. Como crear una red 3x3 con exactamente 3 MZIs
2. Ver los valores theta y phi de cada phase shifter
3. Entender la multiplicación matricial unitaria
4. Extraer y usar estos valores

Copia este código para entender el funcionamiento básico.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Importar OpticalCI (ajusta la ruta según tu instalación)
try:
    from torchonn.layers import MZILayer
    from PhaseShifterExtractor import PhaseShifterExtractor, quick_extract
except ImportError:
    print("⚠️ Asegúrate de tener OpticalCI instalado y el PhaseShifterExtractor en tu directorio")
    sys.exit(1)

class Simple3x3ONN(nn.Module):
    """
    Red óptica simple 3x3.
    
    Arquitectura:
    - 1 capa MZI de 3x3 
    - 3 MZIs físicos internos
    - 6 phase shifters totales (2 por MZI)
    """
    
    def __init__(self, device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Capa MZI principal 3x3
        self.mzi_layer = MZILayer(
            in_features=3,
            out_features=3, 
            device=device
        )
        
        # Mostrar información de construcción
        print(f"🏗️ Red 3x3 creada:")
        print(f"   📐 Dimensiones: 3×3")
        print(f"   🔗 MZIs físicos: {self.mzi_layer.n_mzis}")  
        print(f"   🌊 Phase shifters: {self.mzi_layer.get_phase_shifter_count()}")
        print(f"   💻 Device: {device}")
    
    def forward(self, x):
        """Forward pass a través de la capa MZI."""
        return self.mzi_layer(x)
    
    def get_unitary_matrix(self):
        """Obtener la matriz unitaria actual."""
        return self.mzi_layer.get_unitary_matrix()
    
    def set_phase_shifters(self, theta_values, phi_values):
        """
        Establecer valores específicos de phase shifters.
        
        Args:
            theta_values: Lista de 3 valores theta [rad]
            phi_values: Lista de 3 valores phi [rad] 
        """
        if len(theta_values) != 3 or len(phi_values) != 3:
            raise ValueError("Necesitas exactamente 3 valores theta y 3 phi")
        
        with torch.no_grad():
            self.mzi_layer.theta.copy_(torch.tensor(theta_values, device=self.device))
            self.mzi_layer.phi.copy_(torch.tensor(phi_values, device=self.device))
        
        print(f"✅ Phase shifters actualizados:")
        for i in range(3):
            print(f"   MZI_{i}: θ={np.degrees(theta_values[i]):6.1f}°, φ={np.degrees(phi_values[i]):6.1f}°")

def demo_multiplicacion_unitaria():
    """
    Demostración de multiplicación matricial unitaria.
    """
    print("\n" + "="*70)
    print("🧮 DEMO: MULTIPLICACIÓN MATRICIAL UNITARIA")
    print("="*70)
    
    # Crear modelo
    model = Simple3x3ONN()
    
    # Definir entrada específica
    entrada = torch.tensor([[1.0, 0.5, 0.2]], device=model.device)
    print(f"\n📥 Entrada definida:")
    print(f"   {entrada.squeeze().numpy()}")
    
    # Obtener matriz unitaria actual
    U = model.get_unitary_matrix()
    U_np = U.detach().cpu().numpy()
    
    print(f"\n📊 Matriz Unitaria U (3×3):")
    print(f"   Valores complejos actuales:")
    for i in range(3):
        row = [f"{U_np[i,j].real:+.3f}{U_np[i,j].imag:+.3f}j" for j in range(3)]
        print(f"   [{', '.join(row)}]")
    
    # Forward pass a través del modelo
    salida_modelo = model(entrada)
    salida_np = salida_modelo.squeeze().detach().cpu().numpy()
    
    # Multiplicación matricial manual  
    entrada_np = entrada.squeeze().detach().cpu().numpy()
    U_entrada = U_np @ entrada_np  # Multiplicación matricial
    
    print(f"\n🎯 Resultados:")
    print(f"   Salida modelo:    {salida_np}")
    print(f"   U × entrada:      {U_entrada}")
    print(f"   Diferencia:       {np.abs(salida_np - U_entrada)}")
    
    # Verificar que son iguales
    son_iguales = np.allclose(salida_np, U_entrada, atol=1e-6)
    print(f"   ✅ Son iguales: {son_iguales}")
    
    return model, entrada, salida_modelo

def demo_phase_shifters_especificos():
    """
    Demo con valores específicos de phase shifters.
    """
    print("\n" + "="*70)
    print("🔧 DEMO: PHASE SHIFTERS ESPECÍFICOS")
    print("="*70)
    
    # Crear modelo
    model = Simple3x3ONN()
    
    # Establecer valores específicos (ejemplo)
    theta_values = [np.pi/4, np.pi/2, np.pi/6]      # 45°, 90°, 30°
    phi_values = [np.pi/3, np.pi/4, np.pi/2]        # 60°, 45°, 90°
    
    print(f"\n🎚️ Estableciendo valores específicos:")
    model.set_phase_shifters(theta_values, phi_values)
    
    # Probar con diferentes entradas
    test_inputs = [
        [1.0, 0.0, 0.0],  # Solo primer elemento
        [0.0, 1.0, 0.0],  # Solo segundo elemento  
        [0.0, 0.0, 1.0],  # Solo tercer elemento
        [1.0, 1.0, 1.0],  # Todos iguales
        [0.5, -0.3, 0.8], # Valores mixtos
    ]
    
    print(f"\n📋 Pruebas con diferentes entradas:")
    print(f"{'Entrada':<15} -> {'Salida':<25}")
    print("-" * 45)
    
    for entrada_list in test_inputs:
        entrada = torch.tensor([entrada_list], device=model.device, dtype=torch.float32)
        salida = model(entrada)
        salida_np = salida.squeeze().detach().cpu().numpy()
        
        entrada_str = str([f"{x:+.1f}" for x in entrada_list]).replace("'", "")
        salida_str = str([f"{x:+.3f}" for x in salida_np]).replace("'", "")
        print(f"{entrada_str:<15} -> {salida_str:<25}")
    
    return model

def demo_extraccion_completa():
    """
    Demo completo de extracción de phase shifters.
    """
    print("\n" + "="*70)
    print("💾 DEMO: EXTRACCIÓN COMPLETA DE VALORES")
    print("="*70)
    
    # Crear y configurar modelo
    model = Simple3x3ONN()
    
    # Establecer valores conocidos para la demo
    theta_values = [0.7854, 1.5708, 0.5236]  # π/4, π/2, π/6 en radianes
    phi_values = [1.0472, 0.7854, 1.5708]    # π/3, π/4, π/2 en radianes
    model.set_phase_shifters(theta_values, phi_values)
    
    # Extraer valores usando nuestro módulo
    print(f"\n🔍 Extrayendo todos los valores...")
    extractor = PhaseShifterExtractor(verbose=True)
    valores_extraidos = extractor.extract(model)
    
    # Mostrar resumen
    extractor.print_summary(valores_extraidos)
    
    # Guardar a archivo
    extractor.save_to_file(valores_extraidos, "ejemplo_3x3_weights.json")
    
    # Crear nuevo modelo y cargar valores
    print(f"\n🔄 Probando carga en modelo nuevo...")
    modelo_nuevo = Simple3x3ONN()
    extractor.apply_to_model(modelo_nuevo, valores_extraidos)
    
    # Verificar que son iguales
    entrada_test = torch.tensor([[0.5, -0.2, 0.8]], device=model.device)
    
    salida_original = model(entrada_test)
    salida_cargada = modelo_nuevo(entrada_test)
    
    diferencia = torch.max(torch.abs(salida_original - salida_cargada))
    print(f"   📊 Diferencia entre modelos: {diferencia:.2e}")
    print(f"   ✅ Carga exitosa: {diferencia < 1e-6}")
    
    return valores_extraidos

def demo_analisis_matriz_unitaria():
    """
    Análisis detallado de propiedades de la matriz unitaria.
    """
    print("\n" + "="*70)
    print("🔬 DEMO: ANÁLISIS DE MATRIZ UNITARIA")
    print("="*70)
    
    model = Simple3x3ONN()
    
    # Obtener matriz
    U = model.get_unitary_matrix()
    U_np = U.detach().cpu().numpy()
    
    # Test 1: U @ U^† = I (identidad)
    U_dagger = U_np.conj().T  # Conjugado transpuesto
    producto = U_np @ U_dagger
    identidad = np.eye(3)
    error_unitario = np.max(np.abs(producto - identidad))
    
    print(f"📊 Test 1 - Propiedad unitaria (U @ U^† = I):")
    print(f"   Error máximo: {error_unitario:.2e}")
    print(f"   ✅ Es unitaria: {error_unitario < 1e-6}")
    
    # Test 2: |det(U)| = 1
    determinante = np.linalg.det(U_np)
    det_magnitud = np.abs(determinante)
    error_det = abs(det_magnitud - 1.0)
    
    print(f"\n📊 Test 2 - Determinante:")
    print(f"   det(U) = {determinante:.6f}")
    print(f"   |det(U)| = {det_magnitud:.6f}")
    print(f"   Error: {error_det:.2e}")
    print(f"   ✅ Determinante correcto: {error_det < 1e-6}")
    
    # Test 3: Conservación de energía
    entrada_test = torch.randn(5, 3, device=model.device)
    salida_test = model(entrada_test)
    
    energia_entrada = torch.sum(entrada_test**2, dim=1)
    energia_salida = torch.sum(salida_test**2, dim=1)
    ratio_energia = energia_salida / energia_entrada
    error_energia = torch.max(torch.abs(ratio_energia - 1.0))
    
    print(f"\n📊 Test 3 - Conservación de energía:")
    print(f"   Ratio promedio: {torch.mean(ratio_energia):.6f}")
    print(f"   Error máximo: {error_energia:.2e}")
    print(f"   ✅ Energía conservada: {error_energia < 1e-6}")

def main():
    """
    Función principal que ejecuta todos los demos.
    """
    print("🌟" * 35)
    print("🌟  EJEMPLO COMPLETO: RED 3×3 CON 3 MZIs  🌟")
    print("🌟" * 35)
    print(f"📚 Demostrando multiplicación matricial unitaria")
    print(f"🔧 Extracción de valores theta y phi")
    print(f"💾 Guardado y carga de parámetros")
    
    try:
        # Demo 1: Multiplicación matricial unitaria
        model, entrada, salida = demo_multiplicacion_unitaria()
        
        # Demo 2: Phase shifters específicos
        model = demo_phase_shifters_especificos()
        
        # Demo 3: Extracción completa
        valores = demo_extraccion_completa()
        
        # Demo 4: Análisis de propiedades unitarias
        demo_analisis_matriz_unitaria()
        
        print(f"\n🎉 ¡TODOS LOS DEMOS COMPLETADOS!")
        print(f"   📁 Archivo generado: ejemplo_3x3_weights.json")
        print(f"   🔬 La red 3×3 funciona correctamente")
        print(f"   ⚖️ Multiplicación matricial unitaria verificada")
        
        # Resumen final
        print(f"\n📋 RESUMEN TÉCNICO:")
        print(f"   • Red: 3 entradas → 3 salidas")
        print(f"   • MZIs físicos: 3")
        print(f"   • Phase shifters: 6 (2 por MZI)")
        print(f"   • Operación: salida = U × entrada")
        print(f"   • U es matriz unitaria 3×3")
        
    except Exception as e:
        print(f"❌ Error en la demostración: {e}")
        print(f"   Verifica que OpticalCI esté instalado correctamente")

if __name__ == "__main__":
    main()