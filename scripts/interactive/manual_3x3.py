"""
Manual ONN Trainer - Script Interactivo
=======================================

Script para introducir datos manualmente y entrenar redes ópticas.
Ubicación: /workspaces/OpticalCI/scripts/interactive/manual_3x3.py

CARACTERÍSTICAS:
- Introducir arrays de entrada y salida manualmente
- Configurar arquitectura interactivamente  
- Ver evolución de θ y φ en tiempo real
- Exportar resultados automáticamente

USO:
    cd /workspaces/OpticalCI
    python scripts/interactive/manual_3x3.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from pathlib import Path
import sys

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from torchonn.layers import MZILayer
    from torchonn.utils.phase_shifter_extractor import PhaseShifterExtractor, quick_extract
    print("🌟 OpticalCI cargado exitosamente!")
except ImportError as e:
    print(f"❌ Error importando OpticalCI: {e}")
    print("Asegúrate de estar en el directorio raíz y tener OpticalCI instalado")
    sys.exit(1)

class InteractiveONNTrainer:
    """Entrenador interactivo de redes ópticas."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.training_data = None
        self.target_data = None
        self.results_dir = project_root / "results" / "trained_models"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("🚀 Interactive ONN Trainer iniciado")
        print(f"💻 Device: {self.device}")
    
    def input_network_architecture(self):
        """Configurar la arquitectura de la red interactivamente."""
        print("\n" + "="*60)
        print("🏗️ CONFIGURACIÓN DE ARQUITECTURA")
        print("="*60)
        
        while True:
            try:
                print("\nDimensiones de la red:")
                input_size = int(input("  📥 Número de entradas: "))
                output_size = int(input("  📤 Número de salidas: "))
                
                if input_size > 0 and output_size > 0:
                    if input_size != output_size:
                        print(f"  ⚠️ Advertencia: {input_size}x{output_size} no es cuadrada")
                        print(f"     Se usará matriz {max(input_size, output_size)}x{max(input_size, output_size)}")
                    break
                else:
                    print("  ❌ Por favor introduce números positivos")
                    
            except ValueError:
                print("  ❌ Por favor introduce números enteros válidos")
        
        # Crear modelo
        print(f"\n🔧 Creando red {input_size}x{output_size}...")
        self.model = MZILayer(
            in_features=input_size,
            out_features=output_size,
            device=self.device
        )
        
        print(f"✅ Red creada:")
        print(f"   📐 Dimensiones: {input_size}×{output_size}")
        print(f"   🔗 MZIs físicos: {self.model.n_mzis}")
        print(f"   🌊 Phase shifters: {self.model.get_phase_shifter_count()}")
        
        return input_size, output_size
    
    def input_training_data(self, input_size, output_size):
        """Introducir datos de entrenamiento manualmente."""
        print("\n" + "="*60)
        print("📊 INTRODUCIR DATOS DE ENTRENAMIENTO")
        print("="*60)
        
        print("Puedes introducir datos de las siguientes maneras:")
        print("1. Manualmente (vector por vector)")
        print("2. Usando patrones predefinidos") 
        print("3. Datos aleatorios para pruebas")
        
        while True:
            choice = input("\nElige opción (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("❌ Por favor elige 1, 2 o 3")
        
        if choice == '1':
            return self._input_manual_data(input_size, output_size)
        elif choice == '2':
            return self._input_predefined_patterns(input_size, output_size)
        else:
            return self._generate_random_data(input_size, output_size)
    
    def _input_manual_data(self, input_size, output_size):
        """Introducir datos completamente a mano."""
        print(f"\n📝 Introducir datos manualmente:")
        print(f"Formato: separar números con espacios o comas")
        print(f"Ejemplo entrada ({input_size} números): 1.0 0.5 -0.2")
        
        inputs = []
        outputs = []
        
        while True:
            try:
                n_pairs = int(input(f"\n¿Cuántos pares entrada-salida quieres introducir? (min 1): "))
                if n_pairs > 0:
                    break
                print("❌ Introduce al menos 1 par")
            except ValueError:
                print("❌ Introduce un número válido")
        
        for i in range(n_pairs):
            print(f"\n--- Par {i+1}/{n_pairs} ---")
            
            # Entrada
            while True:
                entrada_str = input(f"📥 Entrada ({input_size} números): ").strip()
                try:
                    entrada_nums = self._parse_numbers(entrada_str)
                    if len(entrada_nums) == input_size:
                        inputs.append(entrada_nums)
                        break
                    else:
                        print(f"❌ Necesitas exactamente {input_size} números, tienes {len(entrada_nums)}")
                except ValueError as e:
                    print(f"❌ Error parseando números: {e}")
            
            # Salida
            while True:
                salida_str = input(f"📤 Salida deseada ({output_size} números): ").strip()
                try:
                    salida_nums = self._parse_numbers(salida_str)
                    if len(salida_nums) == output_size:
                        outputs.append(salida_nums)
                        break
                    else:
                        print(f"❌ Necesitas exactamente {output_size} números, tienes {len(salida_nums)}")
                except ValueError as e:
                    print(f"❌ Error parseando números: {e}")
            
            print(f"✅ Par {i+1}: {inputs[-1]} → {outputs[-1]}")
        
        # Convertir a tensors
        self.training_data = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        self.target_data = torch.tensor(outputs, dtype=torch.float32, device=self.device)
        
        print(f"\n✅ Datos cargados: {len(inputs)} pares entrada-salida")
        return self.training_data, self.target_data
    
    def _input_predefined_patterns(self, input_size, output_size):
        """Usar patrones predefinidos."""
        print(f"\n🎨 Patrones predefinidos disponibles:")
        
        patterns = {
            '1': 'Identidad (entrada = salida)',
            '2': 'Inversión (entrada = -salida)', 
            '3': 'Rotación 45° (2D/3D)',
            '4': 'Permutación circular',
            '5': 'Escalado (entrada × 0.5)'
        }
        
        for key, desc in patterns.items():
            print(f"  {key}. {desc}")
        
        while True:
            choice = input("\nElige patrón (1-5): ").strip()
            if choice in patterns:
                break
            print("❌ Elige un número del 1 al 5")
        
        # Generar datos según el patrón
        n_samples = 10  # Número fijo de muestras
        inputs = torch.randn(n_samples, input_size) * 2  # Entradas aleatorias variadas
        
        if choice == '1':  # Identidad
            outputs = inputs[:, :output_size] if output_size <= input_size else torch.cat([inputs, torch.zeros(n_samples, output_size - input_size)], dim=1)
        elif choice == '2':  # Inversión
            outputs = -inputs[:, :output_size] if output_size <= input_size else torch.cat([-inputs, torch.zeros(n_samples, output_size - input_size)], dim=1)
        elif choice == '3':  # Rotación 45°
            theta = np.pi / 4
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            if input_size >= 2 and output_size >= 2:
                outputs = inputs.clone()
                outputs[:, 0] = cos_t * inputs[:, 0] - sin_t * inputs[:, 1]
                outputs[:, 1] = sin_t * inputs[:, 0] + cos_t * inputs[:, 1]
            else:
                outputs = inputs * 0.707  # Aproximación para 1D
        elif choice == '4':  # Permutación circular
            outputs = torch.roll(inputs, 1, dims=1)[:, :output_size]
        else:  # Escalado
            outputs = inputs[:, :output_size] * 0.5
        
        self.training_data = inputs.to(self.device)
        self.target_data = outputs.to(self.device)
        
        print(f"✅ Patrón '{patterns[choice]}' generado con {n_samples} muestras")
        return self.training_data, self.target_data
    
    def _generate_random_data(self, input_size, output_size):
        """Generar datos aleatorios para pruebas."""
        n_samples = 15
        
        # Datos de entrada aleatorios
        inputs = torch.randn(n_samples, input_size) * 1.5
        
        # Salidas: transformación aleatoria pero consistente
        transform_matrix = torch.randn(output_size, input_size) * 0.5
        outputs = torch.matmul(inputs, transform_matrix.t())
        
        self.training_data = inputs.to(self.device)
        self.target_data = outputs.to(self.device)
        
        print(f"✅ Datos aleatorios generados: {n_samples} muestras")
        print(f"   Transformación aplicada: matriz {output_size}x{input_size}")
        return self.training_data, self.target_data
    
    def _parse_numbers(self, text):
        """Parsear números desde texto."""
        # Reemplazar comas por espacios y dividir
        numbers = text.replace(',', ' ').split()
        return [float(x) for x in numbers if x.strip()]
    
    def configure_training(self):
        """Configurar parámetros de entrenamiento."""
        print("\n" + "="*60)
        print("⚙️ CONFIGURACIÓN DE ENTRENAMIENTO")
        print("="*60)
        
        # Learning rate
        while True:
            try:
                lr_input = input("📈 Learning rate (recomendado 0.01): ").strip()
                learning_rate = float(lr_input) if lr_input else 0.01
                if 0 < learning_rate < 1:
                    break
                print("❌ Learning rate debe estar entre 0 y 1")
            except ValueError:
                print("❌ Introduce un número válido")
        
        # Épocas
        while True:
            try:
                epochs_input = input("🔄 Número de épocas (recomendado 100): ").strip()
                epochs = int(epochs_input) if epochs_input else 100
                if epochs > 0:
                    break
                print("❌ Número de épocas debe ser positivo")
            except ValueError:
                print("❌ Introduce un número entero válido")
        
        # Mostrar progreso
        show_progress = input("📊 ¿Mostrar progreso durante entrenamiento? (s/n, default s): ").strip().lower()
        verbose = show_progress != 'n'
        
        return learning_rate, epochs, verbose
    
    def train_network(self, learning_rate, epochs, verbose=True):
        """Entrenar la red con los datos introducidos."""
        print("\n" + "="*60)
        print("🔥 ENTRENANDO RED NEURONAL ÓPTICA")
        print("="*60)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.8)
        
        # Historial para análisis
        history = {
            'epoch': [],
            'loss': [],
            'theta_values': [],
            'phi_values': []
        }
        
        if verbose:
            print(f"🚀 Iniciando entrenamiento:")
            print(f"   Learning rate: {learning_rate}")
            print(f"   Épocas: {epochs}")
            print(f"   Datos: {len(self.training_data)} pares entrada-salida")
            print(f"   Optimizer: Adam")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            
            # Forward pass
            predictions = self.model(self.training_data)
            loss = criterion(predictions, self.target_data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # Guardar historial
            with torch.no_grad():
                theta_vals = self.model.theta.detach().cpu().numpy().copy()
                phi_vals = self.model.phi.detach().cpu().numpy().copy()
                
                history['epoch'].append(epoch)
                history['loss'].append(loss.item())
                history['theta_values'].append(theta_vals)
                history['phi_values'].append(phi_vals)
            
            # Mostrar progreso
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1 or epoch < 5):
                lr_current = optimizer.param_groups[0]['lr']
                print(f"   Época {epoch:3d}: Loss={loss.item():.6f}, LR={lr_current:.6f}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
            elif loss.item() > best_loss * 10:  # Si diverge mucho
                print(f"   ⚠️ Posible divergencia detectada en época {epoch}")
                break
        
        final_loss = history['loss'][-1]
        print(f"\n✅ Entrenamiento completado!")
        print(f"   Loss final: {final_loss:.6f}")
        print(f"   Épocas entrenadas: {len(history['epoch'])}")
        print(f"   Convergencia: {'Excelente' if final_loss < 0.01 else 'Buena' if final_loss < 0.1 else 'Aceptable'}")
        
        return history
    
    def analyze_results(self, history):
        """Analizar y mostrar resultados del entrenamiento."""
        print("\n" + "="*60)
        print("🔬 ANÁLISIS DE RESULTADOS")
        print("="*60)
        
        # Extraer valores finales
        extractor = PhaseShifterExtractor(verbose=False)
        final_values = extractor.extract(self.model)
        
        print("🎯 PHASE SHIFTERS FINALES ENTRENADOS:")
        layer_data = final_values['layers']['layer_00']
        mzis = layer_data['mzis']
        
        for mzi_name, mzi_data in mzis.items():
            theta_deg = mzi_data['theta']['degrees']
            phi_deg = mzi_data['phi']['degrees']
            print(f"   {mzi_name}: θ={theta_deg:6.1f}°, φ={phi_deg:6.1f}°")
        
        # Test con datos de entrenamiento
        print(f"\n📊 VERIFICACIÓN CON DATOS DE ENTRENAMIENTO:")
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.training_data)
            
            for i in range(min(5, len(self.training_data))):  # Mostrar máximo 5 ejemplos
                entrada = self.training_data[i].cpu().numpy()
                target = self.target_data[i].cpu().numpy()
                pred = predictions[i].cpu().numpy()
                error = np.abs(pred - target)
                
                print(f"   Ejemplo {i+1}:")
                print(f"     Entrada:  [{', '.join(f'{x:+.3f}' for x in entrada)}]")
                print(f"     Objetivo: [{', '.join(f'{x:+.3f}' for x in target)}]")
                print(f"     Predicho: [{', '.join(f'{x:+.3f}' for x in pred)}]")
                print(f"     Error:    [{', '.join(f'{x:.3f}' for x in error)}]")
        
        return final_values
    
    def save_results(self, history, final_values, session_name=None):
        """Guardar resultados del entrenamiento."""
        if session_name is None:
            from datetime import datetime
            session_name = f"manual_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_dir = self.results_dir / session_name
        session_dir.mkdir(exist_ok=True)
        
        # Guardar valores de phase shifters
        extractor = PhaseShifterExtractor(verbose=False)
        extractor.save_to_file(final_values, str(session_dir / "phase_shifters.json"))
        
        # Guardar datos de entrenamiento
        training_data = {
            'inputs': self.training_data.cpu().numpy().tolist(),
            'targets': self.target_data.cpu().numpy().tolist(),
            'final_loss': history['loss'][-1],
            'epochs_trained': len(history['epoch']),
            'architecture': {
                'input_size': self.model.in_features,
                'output_size': self.model.out_features,
                'n_mzis': self.model.n_mzis,
                'phase_shifters': self.model.get_phase_shifter_count()
            }
        }
        
        with open(session_dir / "training_data.json", 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Guardar historial
        history_simple = {
            'epochs': history['epoch'],
            'loss': history['loss']
        }
        
        with open(session_dir / "training_history.json", 'w') as f:
            json.dump(history_simple, f, indent=2)
        
        print(f"\n💾 Resultados guardados en:")
        print(f"   📁 {session_dir}/")
        print(f"   📄 phase_shifters.json - Valores θ y φ finales")
        print(f"   📄 training_data.json - Datos y configuración")
        print(f"   📄 training_history.json - Historial de entrenamiento")
        
        return session_dir
    
    def run_interactive_session(self):
        """Ejecutar sesión interactiva completa."""
        print("🌟" * 30)
        print("🌟  MANUAL ONN TRAINER  🌟")
        print("🌟" * 30)
        print("Script interactivo para entrenar redes ópticas con tus datos")
        
        try:
            # 1. Configurar arquitectura
            input_size, output_size = self.input_network_architecture()
            
            # 2. Introducir datos
            self.input_training_data(input_size, output_size)
            
            # 3. Configurar entrenamiento
            learning_rate, epochs, verbose = self.configure_training()
            
            # 4. Entrenar
            history = self.train_network(learning_rate, epochs, verbose)
            
            # 5. Analizar resultados
            final_values = self.analyze_results(history)
            
            # 6. Guardar resultados
            session_name = input("\n💾 Nombre para esta sesión (opcional): ").strip()
            session_dir = self.save_results(history, final_values, session_name or None)
            
            # 7. Opciones finales
            print(f"\n🎉 ¡SESIÓN COMPLETADA EXITOSAMENTE!")
            print(f"✅ Red entrenada con tus datos específicos")
            print(f"✅ Phase shifters optimizados y guardados")
            print(f"✅ Listo para usar en inferencia o hardware")
            
            print(f"\n📋 PRÓXIMOS PASOS:")
            print(f"1. Usar los valores guardados para inferencia")
            print(f"2. Implementar en hardware real")
            print(f"3. Probar con nuevos datos de entrada")
            
            return session_dir
            
        except KeyboardInterrupt:
            print(f"\n⚠️ Sesión interrumpida por el usuario")
            return None
        except Exception as e:
            print(f"\n❌ Error durante la sesión: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Función principal."""
    trainer = InteractiveONNTrainer()
    session_dir = trainer.run_interactive_session()
    
    if session_dir:
        print(f"\n🔗 Para usar los resultados:")
        print(f"   from torchonn.utils.phase_shifter_extractor import quick_apply")
        print(f"   quick_apply(modelo, '{session_dir}/phase_shifters.json')")

if __name__ == "__main__":
    main()