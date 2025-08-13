"""
Configurable ONN Experiment Runner
==================================

Script avanzado para ejecutar experimentos personalizados con redes ópticas.
Ubicación: /workspaces/OpticalCI/scripts/experiments/network_designer.py

CARACTERÍSTICAS:
- Configuración mediante archivo JSON o interactiva
- Múltiples arquitecturas (single layer, multi-layer, custom)
- Diferentes tipos de datos (sintéticos, archivos, funciones)
- Comparación de múltiples configuraciones
- Exportación automática de resultados

USO:
    python scripts/experiments/network_designer.py --config my_config.json
    python scripts/experiments/network_designer.py --interactive
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import argparse
from pathlib import Path
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# Añadir path del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from torchonn.layers import MZILayer, MZIBlockLinear
    from torchonn.utils.phase_shifter_extractor import PhaseShifterExtractor
    print("🌟 OpticalCI cargado exitosamente!")
except ImportError as e:
    print(f"❌ Error importando OpticalCI: {e}")
    sys.exit(1)

class ConfigurableONN(nn.Module):
    """Red óptica configurable con múltiples arquitecturas."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Construir arquitectura según configuración
        self.layers = nn.ModuleList()
        self._build_architecture()
        
        self.to(self.device)
    
    def _build_architecture(self):
        """Construir arquitectura según configuración."""
        arch_type = self.config['architecture']['type']
        
        if arch_type == 'single_layer':
            self._build_single_layer()
        elif arch_type == 'multi_layer':
            self._build_multi_layer()
        elif arch_type == 'custom':
            self._build_custom()
        else:
            raise ValueError(f"Tipo de arquitectura no soportado: {arch_type}")
    
    def _build_single_layer(self):
        """Construir arquitectura de una sola capa."""
        config = self.config['architecture']
        
        input_size = config['input_size']
        output_size = config['output_size']
        layer_type = config.get('layer_type', 'MZI')
        
        if layer_type == 'MZI':
            self.layers.append(MZILayer(input_size, output_size, device=self.device))
        elif layer_type == 'MZIBlock':
            mode = config.get('mode', 'usv')
            self.layers.append(MZIBlockLinear(input_size, output_size, mode=mode, device=self.device))
    
    def _build_multi_layer(self):
        """Construir arquitectura multicapa."""
        config = self.config['architecture']
        layer_sizes = config['layer_sizes']
        
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            # Capa óptica
            self.layers.append(MZILayer(in_size, out_size, device=self.device))
            
            # Activación no-lineal si no es la última capa
            if i < len(layer_sizes) - 2:
                activation = config.get('activation', 'relu')
                if activation == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                elif activation == 'tanh':
                    self.layers.append(nn.Tanh())
    
    def _build_custom(self):
        """Construir arquitectura personalizada."""
        # Para arquitecturas muy específicas
        config = self.config['architecture']
        layers_config = config['layers']
        
        for layer_config in layers_config:
            layer_type = layer_config['type']
            
            if layer_type == 'MZI':
                self.layers.append(MZILayer(
                    layer_config['input_size'],
                    layer_config['output_size'],
                    device=self.device
                ))
            elif layer_type == 'Linear':
                self.layers.append(nn.Linear(
                    layer_config['input_size'],
                    layer_config['output_size']
                ))
            elif layer_type == 'ReLU':
                self.layers.append(nn.ReLU())
            # Añadir más tipos según necesidad
    
    def forward(self, x):
        """Forward pass a través de todas las capas."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_optical_layers(self):
        """Obtener solo las capas ópticas."""
        return [layer for layer in self.layers if isinstance(layer, (MZILayer, MZIBlockLinear))]
    
    def get_architecture_summary(self):
        """Resumen de la arquitectura."""
        optical_layers = self.get_optical_layers()
        
        return {
            'total_layers': len(self.layers),
            'optical_layers': len(optical_layers),
            'total_mzis': sum(getattr(layer, 'n_mzis', 0) for layer in optical_layers),
            'total_phase_shifters': sum(getattr(layer, 'get_phase_shifter_count', lambda: 0)() for layer in optical_layers),
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

class DataGenerator:
    """Generador de datos configurables."""
    
    @staticmethod
    def generate_from_config(config):
        """Generar datos según configuración."""
        data_config = config['data']
        data_type = data_config['type']
        
        if data_type == 'synthetic_function':
            return DataGenerator._generate_function_data(data_config)
        elif data_type == 'synthetic_classification':
            return DataGenerator._generate_classification_data(data_config)
        elif data_type == 'synthetic_regression':
            return DataGenerator._generate_regression_data(data_config)
        elif data_type == 'from_file':
            return DataGenerator._load_from_file(data_config)
        else:
            raise ValueError(f"Tipo de datos no soportado: {data_type}")
    
    @staticmethod
    def _generate_function_data(config):
        """Generar datos de una función matemática."""
        func_name = config['function']
        n_samples = config.get('n_samples', 1000)
        input_size = config['input_size']
        noise_level = config.get('noise_level', 0.1)
        
        # Generar entradas aleatorias
        X = torch.randn(n_samples, input_size) * config.get('input_scale', 1.0)
        
        # Aplicar función
        if func_name == 'polynomial':
            degree = config.get('degree', 2)
            Y = torch.sum(X ** degree, dim=1, keepdim=True)
        elif func_name == 'sine':
            freq = config.get('frequency', 1.0)
            Y = torch.sin(freq * torch.sum(X, dim=1, keepdim=True))
        elif func_name == 'linear_combination':
            weights = torch.tensor(config.get('weights', [1.0] * input_size))
            Y = torch.matmul(X, weights.unsqueeze(1))
        elif func_name == 'quadratic_form':
            # Y = X^T A X donde A es una matriz
            A = torch.randn(input_size, input_size) * 0.5
            Y = torch.sum(X * torch.matmul(X, A), dim=1, keepdim=True)
        else:
            raise ValueError(f"Función no soportada: {func_name}")
        
        # Añadir ruido
        if noise_level > 0:
            Y += torch.randn_like(Y) * noise_level
        
        return X, Y
    
    @staticmethod
    def _generate_classification_data(config):
        """Generar datos de clasificación."""
        n_samples = config.get('n_samples', 1000)
        n_classes = config['n_classes']
        input_size = config['input_size']
        
        # Generar centros de clases
        class_centers = torch.randn(n_classes, input_size) * 2.0
        
        # Generar muestras alrededor de cada centro
        samples_per_class = n_samples // n_classes
        X_list = []
        Y_list = []
        
        for class_id in range(n_classes):
            center = class_centers[class_id]
            # Muestras gaussianas alrededor del centro
            class_samples = center + torch.randn(samples_per_class, input_size) * 0.5
            class_labels = torch.full((samples_per_class,), class_id, dtype=torch.long)
            
            X_list.append(class_samples)
            Y_list.append(class_labels)
        
        X = torch.cat(X_list, dim=0)
        Y = torch.cat(Y_list, dim=0)
        
        # Mezclar datos
        perm = torch.randperm(len(X))
        X = X[perm]
        Y = Y[perm]
        
        return X, Y
    
    @staticmethod
    def _generate_regression_data(config):
        """Generar datos de regresión."""
        n_samples = config.get('n_samples', 1000)
        input_size = config['input_size']
        output_size = config['output_size']
        
        # Generar matriz de transformación aleatoria
        transform_matrix = torch.randn(output_size, input_size) * config.get('transform_scale', 1.0)
        
        # Generar entradas
        X = torch.randn(n_samples, input_size)
        
        # Aplicar transformación
        Y = torch.matmul(X, transform_matrix.t())
        
        # Añadir no-linealidad opcional
        if config.get('add_nonlinearity', False):
            Y = torch.tanh(Y)
        
        return X, Y

class ExperimentRunner:
    """Ejecutor de experimentos configurables."""
    
    def __init__(self, config_path=None, config_dict=None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Debe proporcionar config_path o config_dict")
        
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.results_dir = Path(self.config.get('results_dir', 'results/experiments'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Experiment Runner inicializado")
        print(f"💻 Device: {self.device}")
        print(f"📁 Results dir: {self.results_dir}")
    
    def run_experiment(self, experiment_name=None):
        """Ejecutar experimento completo."""
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n🔬 EJECUTANDO EXPERIMENTO: {experiment_name}")
        print("=" * 60)
        
        # 1. Crear modelo
        print("🏗️ Creando modelo...")
        model = ConfigurableONN(self.config)
        arch_summary = model.get_architecture_summary()
        
        print(f"✅ Modelo creado:")
        for key, value in arch_summary.items():
            print(f"   {key}: {value}")
        
        # 2. Generar datos
        print("\n📊 Generando datos...")
        X, Y = DataGenerator.generate_from_config(self.config)
        X, Y = X.to(self.device), Y.to(self.device)
        
        print(f"✅ Datos generados:")
        print(f"   Entradas: {X.shape}")
        print(f"   Salidas: {Y.shape}")
        
        # 3. Dividir datos
        train_ratio = self.config.get('train_ratio', 0.8)
        n_train = int(len(X) * train_ratio)
        
        # Mezclar datos antes de dividir
        perm = torch.randperm(len(X))
        X, Y = X[perm], Y[perm]
        
        X_train, Y_train = X[:n_train], Y[:n_train]
        X_test, Y_test = X[n_train:], Y[n_train:]
        
        print(f"   Train: {X_train.shape[0]} muestras")
        print(f"   Test: {X_test.shape[0]} muestras")
        
        # 4. Entrenar
        print("\n🔥 Entrenando modelo...")
        history = self._train_model(model, X_train, Y_train, X_test, Y_test)
        
        # 5. Evaluar
        print("\n📊 Evaluando modelo...")
        evaluation = self._evaluate_model(model, X_test, Y_test)
        
        # 6. Extraer phase shifters
        print("\n🔧 Extrayendo phase shifters...")
        extractor = PhaseShifterExtractor(verbose=False)
        phase_shifters = extractor.extract(model)
        
        # 7. Guardar resultados
        experiment_dir = self._save_results(
            experiment_name, model, history, evaluation, 
            phase_shifters, arch_summary
        )
        
        print(f"\n🎉 Experimento completado:")
        print(f"   📁 Resultados en: {experiment_dir}")
        print(f"   📊 Loss final: {history['train_loss'][-1]:.6f}")
        print(f"   🎯 Precisión test: {evaluation.get('accuracy', 'N/A')}")
        
        return experiment_dir, history, evaluation
    
    def _train_model(self, model, X_train, Y_train, X_test, Y_test):
        """Entrenar el modelo."""
        training_config = self.config['training']
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=training_config.get('learning_rate', 0.01)
        )
        
        # Determinar función de pérdida
        task_type = self.config['data'].get('task_type', 'regression')
        if task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.8
        )
        
        epochs = training_config.get('epochs', 100)
        
        history = {
            'epoch': [],
            'train_loss': [],
            'test_loss': [],
            'learning_rate': []
        }
        
        for epoch in range(epochs):
            # Entrenamiento
            model.train()
            train_predictions = model(X_train)
            train_loss = criterion(train_predictions, Y_train)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Evaluación
            model.eval()
            with torch.no_grad():
                test_predictions = model(X_test)
                test_loss = criterion(test_predictions, Y_test)
            
            scheduler.step(test_loss)
            
            # Guardar historial
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss.item())
            history['test_loss'].append(test_loss.item())
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Mostrar progreso
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"   Época {epoch:3d}: Train={train_loss.item():.6f}, Test={test_loss.item():.6f}, LR={lr:.6f}")
        
        return history
    
    def _evaluate_model(self, model, X_test, Y_test):
        """Evaluar el modelo."""
        model.eval()
        evaluation = {}
        
        with torch.no_grad():
            predictions = model(X_test)
            
            # MSE
            mse = torch.mean((predictions - Y_test) ** 2).item()
            evaluation['mse'] = mse
            
            # RMSE
            evaluation['rmse'] = np.sqrt(mse)
            
            # Accuracy para clasificación
            task_type = self.config['data'].get('task_type', 'regression')
            if task_type == 'classification':
                _, predicted_classes = torch.max(predictions, 1)
                accuracy = (predicted_classes == Y_test).float().mean().item() * 100
                evaluation['accuracy'] = f"{accuracy:.2f}%"
            else:
                # R² score para regresión
                y_mean = torch.mean(Y_test)
                ss_tot = torch.sum((Y_test - y_mean) ** 2)
                ss_res = torch.sum((Y_test - predictions) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                evaluation['r2_score'] = r2.item()
        
        return evaluation
    
    def _save_results(self, experiment_name, model, history, evaluation, phase_shifters, arch_summary):
        """Guardar todos los resultados."""
        experiment_dir = self.results_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        # Guardar configuración
        with open(experiment_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Guardar phase shifters
        extractor = PhaseShifterExtractor(verbose=False)
        extractor.save_to_file(phase_shifters, str(experiment_dir / 'phase_shifters.json'))
        
        # Guardar historial y evaluación
        results = {
            'architecture_summary': arch_summary,
            'training_history': history,
            'evaluation': evaluation,
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(experiment_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Guardar gráfico de entrenamiento
        self._plot_training_history(history, experiment_dir / 'training_plot.png')
        
        # Guardar modelo
        torch.save(model.state_dict(), experiment_dir / 'model_weights.pth')
        
        return experiment_dir
    
    def _plot_training_history(self, history, save_path):
        """Crear gráfico del entrenamiento."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = history['epoch']
        
        # Loss
        ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['test_loss'], label='Test Loss', linewidth=2)
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.set_title('Evolución del Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate
        ax2.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Evolución del Learning Rate')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def create_interactive_config():
    """Crear configuración interactivamente."""
    print("🎛️ CONFIGURACIÓN INTERACTIVA")
    print("=" * 40)
    
    config = {}
    
    # Arquitectura
    print("\n🏗️ Arquitectura:")
    arch_type = input("Tipo (single_layer/multi_layer): ").strip() or "single_layer"
    
    if arch_type == "single_layer":
        input_size = int(input("Tamaño entrada: ") or "4")
        output_size = int(input("Tamaño salida: ") or "4")
        
        config['architecture'] = {
            'type': 'single_layer',
            'input_size': input_size,
            'output_size': output_size,
            'layer_type': 'MZI'
        }
    else:
        sizes_str = input("Tamaños capas (ej: 4,8,4): ") or "4,8,4"
        layer_sizes = [int(x.strip()) for x in sizes_str.split(',')]
        
        config['architecture'] = {
            'type': 'multi_layer',
            'layer_sizes': layer_sizes,
            'activation': 'relu'
        }
    
    # Datos
    print("\n📊 Datos:")
    data_type = input("Tipo (synthetic_function/synthetic_regression): ").strip() or "synthetic_function"
    n_samples = int(input("Número de muestras: ") or "1000")
    
    if data_type == "synthetic_function":
        func_name = input("Función (polynomial/sine/linear_combination): ").strip() or "polynomial"
        config['data'] = {
            'type': 'synthetic_function',
            'function': func_name,
            'n_samples': n_samples,
            'input_size': config['architecture'].get('input_size', layer_sizes[0] if 'layer_sizes' in locals() else 4),
            'noise_level': 0.1
        }
    
    # Entrenamiento
    print("\n🔥 Entrenamiento:")
    lr = float(input("Learning rate: ") or "0.01")
    epochs = int(input("Épocas: ") or "100")
    
    config['training'] = {
        'learning_rate': lr,
        'epochs': epochs
    }
    
    config['train_ratio'] = 0.8
    config['device'] = 'auto'
    
    return config

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Configurable ONN Experiment Runner')
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración JSON')
    parser.add_argument('--interactive', action='store_true', help='Modo interactivo')
    parser.add_argument('--name', type=str, help='Nombre del experimento')
    
    args = parser.parse_args()
    
    print("🌟" * 30)
    print("🌟  CONFIGURABLE ONN EXPERIMENTS  🌟")
    print("🌟" * 30)
    
    if args.interactive:
        config = create_interactive_config()
        runner = ExperimentRunner(config_dict=config)
    elif args.config:
        runner = ExperimentRunner(config_path=args.config)
    else:
        print("❌ Debe proporcionar --config o usar --interactive")
        return
    
    # Ejecutar experimento
    experiment_dir, history, evaluation = runner.run_experiment(args.name)
    
    print(f"\n🎯 EXPERIMENTO COMPLETADO:")
    print(f"   📁 Resultados: {experiment_dir}")
    print(f"   📊 Archivos generados:")
    print(f"      - config.json (configuración)")
    print(f"      - phase_shifters.json (valores θ,φ)")
    print(f"      - results.json (métricas)")
    print(f"      - training_plot.png (gráfico)")
    print(f"      - model_weights.pth (pesos del modelo)")

if __name__ == "__main__":
    main()