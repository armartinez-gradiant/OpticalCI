"""
Helper functions for PtONN-TESTS
"""

import torch
import numpy as np
import logging
import sys
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

def check_torch_version() -> Dict[str, Any]:
    """
    Check PyTorch version and compatibility.
    
    Returns:
        Dictionary with version information
    """
    info = {
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "mps_available": torch.backends.mps.is_available(),
    }
    
    # Check minimum requirements
    torch_version = tuple(map(int, torch.__version__.split('+')[0].split('.')))
    min_version = (2, 0, 0)
    info["version_compatible"] = torch_version >= min_version
    info["minimum_version"] = ".".join(map(str, min_version))
    
    return info

def get_memory_info() -> Dict[str, Any]:
    """
    Get memory information for available devices.
    
    Returns:
        Dictionary with memory information
    """
    info = {}
    
    # CPU memory (approximate)
    try:
        import psutil
        info["cpu_memory"] = {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "used": psutil.virtual_memory().used,
            "percent": psutil.virtual_memory().percent,
        }
    except ImportError:
        info["cpu_memory"] = {"error": "psutil not available"}
    
    # GPU memory
    if torch.cuda.is_available():
        info["gpu_memory"] = {}
        for i in range(torch.cuda.device_count()):
            info["gpu_memory"][f"cuda:{i}"] = {
                "total": torch.cuda.get_device_properties(i).total_memory,
                "allocated": torch.cuda.memory_allocated(i),
                "cached": torch.cuda.memory_reserved(i),
            }
    
    return info

def validate_tensor_shape(
    tensor: torch.Tensor, 
    expected_shape: Tuple[int, ...],
    name: str = "tensor"
) -> bool:
    """
    Validate tensor shape.
    
    Args:
        tensor: Input tensor
        expected_shape: Expected shape (use -1 for any dimension)
        name: Name for error messages
        
    Returns:
        True if shape is valid
        
    Raises:
        ValueError: If shape is invalid
    """
    actual_shape = tensor.shape
    
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"{name} shape mismatch: expected {len(expected_shape)} dimensions, "
            f"got {len(actual_shape)} dimensions"
        )
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{name} shape mismatch at dimension {i}: expected {expected}, got {actual}"
            )
    
    return True

def convert_to_tensor(
    data: Union[np.ndarray, list, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert data to torch tensor.
    
    Args:
        data: Input data
        dtype: Target dtype
        device: Target device
        
    Returns:
        Converted tensor
    """
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)
    
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    
    if device is not None:
        tensor = tensor.to(device=device)
    
    return tensor

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    filepath: Union[str, Path],
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        additional_info: Additional information to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "torch_version": torch.__version__,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into
        device: Device to load tensors to
        
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint

def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]) -> None:
    """
    Print model summary.
    
    Args:
        model: Model to summarize
        input_size: Input size for the model
    """
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model structure
    print("\nModel Structure:")
    print(model)
    
    # Memory estimation
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_size, device=device)
        
        # Forward pass to get output size
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\nInput size: {tuple(dummy_input.shape)}")
        print(f"Output size: {tuple(output.shape)}")
        
        # Memory usage (approximate)
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        print(f"Parameter memory: {param_size / 1024**2:.2f} MB")
        print(f"Buffer memory: {buffer_size / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"Could not estimate memory usage: {e}")
    
    print("=" * 60)

def setup_logging(
    level: Union[int, str] = logging.INFO,
    format_string: Optional[str] = None,
    filename: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        format_string: Custom format string
        filename: Log file name
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        filename=filename,
        filemode='a' if filename else None
    )
    
    # Create logger for this package
    logger = logging.getLogger("torchonn")
    logger.setLevel(level)
    
    # Add console handler if logging to file
    if filename:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(format_string)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_package_info() -> Dict[str, Any]:
    """
    Get package information.
    
    Returns:
        Package information dictionary
    """
    try:
        from .. import __version__, __author__, __email__
        info = {
            "version": __version__,
            "author": __author__,
            "email": __email__,
        }
    except ImportError:
        info = {
            "version": "unknown",
            "author": "unknown",
            "email": "unknown",
        }
    
    # Add system info
    info.update(check_torch_version())
    
    return info

def benchmark_function(func, *args, num_runs: int = 100, **kwargs) -> Dict[str, float]:
    """
    Benchmark a function.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        num_runs: Number of runs
        **kwargs: Function keyword arguments
        
    Returns:
        Benchmark results
    """
    import time
    
    # Warm up
    for _ in range(min(10, num_runs)):
        func(*args, **kwargs)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "median": float(np.median(times)),
        "num_runs": num_runs,
    }