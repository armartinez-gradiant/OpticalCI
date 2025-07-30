"""
Optical MNIST Benchmark for CoherentONN

Benchmark estándar para evaluar redes neuronales ópticas usando MNIST.
Implementa versiones escalables desde 4x4 hasta 28x28.

🎯 Objetivos:
- Demostrar capacidades de CoherentONN 
- Comparar con redes electrónicas equivalentes
- Validar conservación de propiedades físicas durante entrenamiento
- Medir métricas específicas para ONNs

📚 Inspirado en: Shen et al. (2017), Hughes et al. (2018)
🔧 Usa: CoherentONN + componentes OpticalCI existentes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Import de la arquitectura ONN
try:
    from ..architectures import CoherentONN, create_simple_coherent_onn
    COHERENT_AVAILABLE = True
except ImportError:
    COHERENT_AVAILABLE = False
    warnings.warn("CoherentONN not available - check implementation")


class OpticalMNIST:
    """
    Benchmark de MNIST para redes neuronales ópticas.
    
    Características:
    - Múltiples resoluciones (4x4, 8x8, 14x14, 28x28)
    - Comparación ONN vs ANN
    - Métricas específicas para fotónica
    - Validación física durante entrenamiento
    """
    
    def __init__(
        self,
        image_size: int = 8,
        n_classes: int = 10,
        device: Optional[torch.device] = None,
        data_path: str = "./data"
    ):
        """
        Inicializar benchmark MNIST óptico.
        
        Args:
            image_size: Tamaño de imagen (4, 8, 14, 28)
            n_classes: Número de clases (típicamente 10)
            device: Device para computación
            data_path: Path para datos MNIST
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        if not COHERENT_AVAILABLE:
            raise ImportError("CoherentONN not available - check imports")
        
        self.image_size = image_size
        self.n_classes = n_classes
        self.data_path = data_path
        
        # Configuración del experimento
        self.input_size = image_size * image_size
        self.results = {}
        
        print(f"🔬 Optical MNIST Benchmark")
        print(f"   Image size: {image_size}x{image_size} ({self.input_size} pixels)")
        print(f"   Classes: {n_classes}")
        print(f"   Device: {device}")
        
        # Preparar datos
        self.train_loader, self.test_loader = self._prepare_data()
        
    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Preparar dataset MNIST con el tamaño apropiado."""
        
        # Transformaciones para redimensionar
        if self.image_size == 28:
            # MNIST original
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            # MNIST redimensionado
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        # Cargar MNIST
        try:
            train_dataset = torchvision.datasets.MNIST(
                root=self.data_path,
                train=True,
                download=True,
                transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root=self.data_path,
                train=False,
                download=True,
                transform=transform
            )
        except Exception as e:
            # Fallback: crear datos sintéticos para testing
            warnings.warn(f"MNIST download failed: {e}. Using synthetic data.")
            return self._create_synthetic_data()
        
        # Para empezar con datasets más pequeños
        if self.image_size <= 8:
            # Usar subset para pruebas rápidas
            train_size = min(1000, len(train_dataset))
            test_size = min(200, len(test_dataset))
            
            train_indices = torch.randperm(len(train_dataset))[:train_size]
            test_indices = torch.randperm(len(test_dataset))[:test_size]
            
            train_subset = torch.utils.data.Subset(train_dataset, train_indices)
            test_subset = torch.utils.data.Subset(test_dataset, test_indices)
            
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
        else:
            # Dataset completo para imágenes grandes
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")
        
        return train_loader, test_loader
    
    def _create_synthetic_data(self) -> Tuple[DataLoader, DataLoader]:
        """Crear datos sintéticos para testing cuando MNIST no está disponible."""
        n_train = 500
        n_test = 100
        
        # Generar imágenes sintéticas
        train_data = torch.randn(n_train, self.input_size)
        train_labels = torch.randint(0, self.n_classes, (n_train,))
        
        test_data = torch.randn(n_test, self.input_size)
        test_labels = torch.randint(0, self.n_classes, (n_test,))
        
        # Normalizar a [0, 1]
        train_data = torch.sigmoid(train_data)
        test_data = torch.sigmoid(test_data)
        
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        print("   Using synthetic data (MNIST unavailable)")
        
        return train_loader, test_loader
    
    def create_coherent_onn(
        self,
        hidden_sizes: List[int] = None,
        activation_type: str = "square_law"
    ) -> CoherentONN:
        """
        Crear CoherentONN para MNIST.
        
        Args:
            hidden_sizes: Tamaños de capas ocultas
            activation_type: Tipo de activación óptica
            
        Returns:
            CoherentONN configurada para MNIST
        """
        if hidden_sizes is None:
            # Arquitectura por defecto basada en tamaño de imagen
            if self.input_size <= 16:  # 4x4
                hidden_sizes = [8]
            elif self.input_size <= 64:  # 8x8
                hidden_sizes = [32, 16]
            elif self.input_size <= 196:  # 14x14
                hidden_sizes = [64, 32]
            else:  # 28x28
                hidden_sizes = [128, 64, 32]
        
        layer_sizes = [self.input_size] + hidden_sizes + [self.n_classes]
        
        onn = CoherentONN(
            layer_sizes=layer_sizes,
            activation_type=activation_type,
            optical_power=1.0,
            use_unitary_constraints=True,
            device=self.device
        )
        
        print(f"   CoherentONN: {' → '.join(map(str, layer_sizes))}")
        return onn
    
    def create_reference_ann(self, hidden_sizes: List[int] = None) -> nn.Module:
        """
        Crear ANN de referencia para comparación.
        
        Args:
            hidden_sizes: Tamaños de capas ocultas (mismos que ONN)
            
        Returns:
            Red neuronal electrónica equivalente
        """
        if hidden_sizes is None:
            # Usar misma arquitectura que ONN
            if self.input_size <= 16:
                hidden_sizes = [8]
            elif self.input_size <= 64:
                hidden_sizes = [32, 16]
            elif self.input_size <= 196:
                hidden_sizes = [64, 32]
            else:
                hidden_sizes = [128, 64, 32]
        
        layers = []
        layer_sizes = [self.input_size] + hidden_sizes + [self.n_classes]
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation en última capa
                layers.append(nn.ReLU())
        
        ann = nn.Sequential(*layers).to(self.device)
        
        print(f"   Reference ANN: {' → '.join(map(str, layer_sizes))}")
        return ann
    
    def train_model(
        self,
        model: nn.Module,
        n_epochs: int = 10,
        learning_rate: float = 0.01,
        model_type: str = "ONN"
    ) -> Dict[str, Any]:
        """
        Entrenar modelo (ONN o ANN).
        
        Args:
            model: Modelo a entrenar
            n_epochs: Número de épocas
            learning_rate: Learning rate
            model_type: "ONN" o "ANN"
            
        Returns:
            Resultados del entrenamiento
        """
        print(f"\n🚀 Training {model_type} ({n_epochs} epochs, lr={learning_rate})")
        
        # Configurar entrenamiento
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Métricas
        train_losses = []
        train_accuracies = []
        epoch_times = []
        physics_violations = [] if model_type == "ONN" else None
        
        model.train()
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Preparar datos
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Flatten para ONN/ANN
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Estadísticas
                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Validación física para ONN
                if model_type == "ONN" and hasattr(model, 'validate_unitarity'):
                    if batch_idx % 10 == 0:  # Cada 10 batches
                        try:
                            unitarity = model.validate_unitarity()
                            if not unitarity.get("overall_valid", True):
                                physics_violations.append({
                                    "epoch": epoch,
                                    "batch": batch_idx,
                                    "unitarity": unitarity
                                })
                        except Exception as e:
                            warnings.warn(f"Physics validation failed: {e}")
            
            # Métricas de época
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(self.train_loader)
            accuracy = 100. * correct / total
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            epoch_times.append(epoch_time)
            
            if epoch % max(1, n_epochs // 5) == 0:  # Print cada 20%
                print(f"   Epoch {epoch+1}/{n_epochs}: "
                      f"Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, "
                      f"Time={epoch_time:.2f}s")
        
        # Resultados finales
        results = {
            "model_type": model_type,
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "epoch_times": epoch_times,
            "final_train_accuracy": train_accuracies[-1],
            "avg_epoch_time": np.mean(epoch_times),
            "total_training_time": sum(epoch_times)
        }
        
        if physics_violations is not None:
            results["physics_violations"] = physics_violations
            results["n_physics_violations"] = len(physics_violations)
        
        return results
    
    def evaluate_model(self, model: nn.Module, model_type: str = "ONN") -> Dict[str, Any]:
        """Evaluar modelo en test set."""
        print(f"\n📊 Evaluating {model_type}")
        
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        inference_times = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                # Medir tiempo de inferencia
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(self.test_loader)
        avg_inference_time = np.mean(inference_times)
        
        results = {
            "test_accuracy": accuracy,
            "test_loss": avg_loss,
            "avg_inference_time": avg_inference_time,
            "total_test_samples": total
        }
        
        print(f"   Test Accuracy: {accuracy:.2f}%")
        print(f"   Test Loss: {avg_loss:.4f}")
        print(f"   Avg Inference Time: {avg_inference_time*1000:.2f} ms/batch")
        
        return results
    
    def run_comparison_benchmark(
        self,
        n_epochs: int = 10,
        learning_rate: float = 0.01
    ) -> Dict[str, Any]:
        """
        Ejecutar benchmark comparativo ONN vs ANN.
        
        Args:
            n_epochs: Épocas de entrenamiento
            learning_rate: Learning rate
            
        Returns:
            Resultados completos del benchmark
        """
        print(f"\n🏁 Starting Optical MNIST Benchmark")
        print(f"   Image size: {self.image_size}x{self.image_size}")
        print(f"   Epochs: {n_epochs}, LR: {learning_rate}")
        
        results = {
            "config": {
                "image_size": self.image_size,
                "input_size": self.input_size,
                "n_classes": self.n_classes,
                "n_epochs": n_epochs,
                "learning_rate": learning_rate,
                "device": str(self.device)
            }
        }
        
        # 1. Crear y entrenar CoherentONN
        print("\n" + "="*50)
        print("1️⃣ COHERENT ONN")
        print("="*50)
        
        try:
            onn = self.create_coherent_onn()
            onn_train_results = self.train_model(onn, n_epochs, learning_rate, "ONN")
            onn_test_results = self.evaluate_model(onn, "ONN")
            
            # Métricas específicas ONN
            if hasattr(onn, 'get_optical_efficiency'):
                efficiency = onn.get_optical_efficiency()
                onn_test_results["optical_efficiency"] = efficiency
            
            if hasattr(onn, 'get_onn_metrics'):
                onn_metrics = onn.get_onn_metrics()
                onn_test_results["onn_specific_metrics"] = onn_metrics
            
            results["onn"] = {**onn_train_results, **onn_test_results}
            
        except Exception as e:
            print(f"❌ ONN training failed: {e}")
            results["onn"] = {"error": str(e)}
        
        # 2. Crear y entrenar ANN de referencia
        print("\n" + "="*50)
        print("2️⃣ REFERENCE ANN")
        print("="*50)
        
        try:
            ann = self.create_reference_ann()
            ann_train_results = self.train_model(ann, n_epochs, learning_rate, "ANN")
            ann_test_results = self.evaluate_model(ann, "ANN")
            
            results["ann"] = {**ann_train_results, **ann_test_results}
            
        except Exception as e:
            print(f"❌ ANN training failed: {e}")
            results["ann"] = {"error": str(e)}
        
        # 3. Análisis comparativo
        print("\n" + "="*50)
        print("📊 COMPARISON ANALYSIS")
        print("="*50)
        
        if "onn" in results and "ann" in results and "error" not in results["onn"]:
            comparison = self._analyze_comparison(results["onn"], results["ann"])
            results["comparison"] = comparison
        
        # Guardar resultados
        self.results = results
        
        return results
    
    def _analyze_comparison(self, onn_results: Dict, ann_results: Dict) -> Dict[str, Any]:
        """Analizar resultados comparativos."""
        comparison = {}
        
        # Accuracy comparison
        onn_acc = onn_results.get("test_accuracy", 0)
        ann_acc = ann_results.get("test_accuracy", 0)
        comparison["accuracy_ratio"] = onn_acc / max(ann_acc, 1e-6)
        comparison["accuracy_difference"] = onn_acc - ann_acc
        
        # Speed comparison
        onn_time = onn_results.get("avg_inference_time", float('inf'))
        ann_time = ann_results.get("avg_inference_time", float('inf'))
        comparison["speed_ratio"] = ann_time / max(onn_time, 1e-9)  # ONN/ANN
        
        # Training comparison
        onn_train_time = onn_results.get("total_training_time", float('inf'))
        ann_train_time = ann_results.get("total_training_time", float('inf'))
        comparison["training_time_ratio"] = onn_train_time / max(ann_train_time, 1e-6)
        
        # Physics validation
        if "physics_violations" in onn_results:
            comparison["physics_violations"] = onn_results["n_physics_violations"]
        
        # Summary
        print(f"   ONN Accuracy: {onn_acc:.2f}% vs ANN: {ann_acc:.2f}%")
        print(f"   Accuracy Ratio: {comparison['accuracy_ratio']:.3f}")
        print(f"   Speed Ratio (ONN/ANN): {comparison['speed_ratio']:.3f}")
        print(f"   Training Time Ratio: {comparison['training_time_ratio']:.3f}")
        
        if comparison["accuracy_ratio"] > 0.9:
            print("   ✅ ONN performance comparable to ANN")
        else:
            print("   ⚠️ ONN performance below ANN")
        
        return comparison


# Función helper para ejecutar demo rápido
def run_quick_demo(image_size: int = 4, n_epochs: int = 5) -> Dict[str, Any]:
    """
    Ejecutar demo rápido de Optical MNIST.
    
    Args:
        image_size: Tamaño de imagen (4 para demo rápido)
        n_epochs: Épocas (5 para demo rápido)
        
    Returns:
        Resultados del benchmark
    """
    if not COHERENT_AVAILABLE:
        print("❌ CoherentONN no disponible - revisar implementación")
        return {}
    
    print("🚀 Quick Optical MNIST Demo")
    
    # Crear benchmark
    benchmark = OpticalMNIST(image_size=image_size)
    
    # Ejecutar comparación
    results = benchmark.run_comparison_benchmark(n_epochs=n_epochs, learning_rate=0.01)
    
    return results


if __name__ == "__main__":
    # Demo básico
    results = run_quick_demo(image_size=4, n_epochs=3)
    print("\n🎉 Demo completed!")