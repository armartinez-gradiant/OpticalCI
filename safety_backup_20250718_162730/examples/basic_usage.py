"""
Basic usage example for PtONN-TESTS
"""

import torch
from torchonn.layers import MZILayer, MZIBlockLinear
from torchonn.models import ONNBaseModel

def basic_layer_example():
    """Example of using basic MZI layers."""
    print("🔬 Basic MZI Layer Example")
    
    # Create MZI layer
    layer = MZILayer(in_features=10, out_features=5)
    
    # Forward pass
    x = torch.randn(32, 10)  # Batch of 32 samples
    output = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    return output

def block_linear_example():
    """Example of using MZI Block Linear layers."""
    print("\n🔬 MZI Block Linear Example")
    
    # Create block linear layer
    layer = MZIBlockLinear(
        in_features=16,
        out_features=8,
        miniblock=4,
        mode="usv"
    )
    
    # Forward pass
    x = torch.randn(16, 16)
    output = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Mode: {layer.mode}")
    return output

class SimpleONN(ONNBaseModel):
    """Simple ONN model example."""
    
    def __init__(self):
        super().__init__()
        self.layer1 = MZIBlockLinear(20, 16, miniblock=4, mode="usv")
        self.layer2 = MZILayer(16, 10)
        self.layer3 = MZILayer(10, 4)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.activation(x)
        
        x = self.layer3(x)
        return x

def model_example():
    """Example of creating and using a complete ONN model."""
    print("\n🔬 Complete ONN Model Example")
    
    # Create model
    model = SimpleONN()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Forward pass
    x = torch.randn(8, 20)  # Batch of 8 samples
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    return model, output

def training_example():
    """Example of training an ONN model."""
    print("\n🔬 Training Example")
    
    # Create model and data
    model = SimpleONN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy training data
    x_train = torch.randn(100, 20)
    y_train = torch.randint(0, 4, (100,))
    
    # Training loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print("Training completed!")
    return model

if __name__ == "__main__":
    print("🚀 PtONN-TESTS Examples")
    print("=" * 40)
    
    # Run examples
    basic_layer_example()
    block_linear_example()
    model, output = model_example()
    trained_model = training_example()
    
    print("\n✅ All examples completed successfully!")