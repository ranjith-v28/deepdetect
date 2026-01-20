"""
Model Optimization Module
Implements various optimization techniques for faster inference.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Optimizer for deepfake detection models.
    Implements quantization, pruning, and other optimizations.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        
    def quantize_model(self, quantization_type: str = 'dynamic'):
        """
        Quantize model for faster inference and smaller size.
        
        Args:
            quantization_type: Type of quantization ('dynamic', 'static', 'qat')
        
        Returns:
            quantized_model: Quantized model
        """
        logger.info(f"Applying {quantization_type} quantization...")
        
        self.model.eval()
        
        if quantization_type == 'dynamic':
            # Dynamic quantization (no calibration data needed)
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.LSTM, nn.GRU, nn.Conv1d, nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Dynamic quantization applied successfully")
            
        elif quantization_type == 'static':
            # Static quantization (requires calibration data)
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            quantized_model = torch.quantization.prepare(self.model, inplace=True)
            
            # Calibration would happen here with representative data
            logger.warning("Static quantization prepared. Calibration data needed for quantize()")
            
        elif quantization_type == 'qat':
            # Quantization-aware training (should be done during training)
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            quantized_model = torch.quantization.prepare_qat(self.model, inplace=True)
            logger.warning("QAT model prepared. Fine-tuning needed for best results")
            
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        return quantized_model
    
    def prune_model(self, pruning_amount: float = 0.2):
        """
        Prune model to reduce size and potentially improve generalization.
        
        Args:
            pruning_amount: Fraction of parameters to prune (0-1)
        """
        logger.info(f"Applying pruning with amount={pruning_amount}...")
        
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # Global pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_amount
        )
        
        # Make pruning permanent
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
        
        logger.info(f"Pruned {len(parameters_to_prune)} layers")
        
    def optimize_for_inference(self):
        """
        Apply various optimizations for inference.
        """
        logger.info("Optimizing model for inference...")
        
        # Set to eval mode
        self.model.eval()
        
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Fuse operations (Conv+BN+ReLU, Linear+ReLU, etc.)
        self.model = torch.quantization.fuse_modules(
            self.model,
            [['conv1', 'bn1', 'relu1']],
            inplace=True
        )
        
        logger.info("Model optimized for inference")
        
    def get_model_size(self) -> dict:
        """
        Get model size statistics.
        
        Returns:
            size_info: Dictionary with size information
        """
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        size_info = {
            'total_size_mb': total_size / (1024 ** 2),
            'param_size_mb': param_size / (1024 ** 2),
            'buffer_size_mb': buffer_size / (1024 ** 2),
            'total_params': sum(p.nelement() for p in self.model.parameters())
        }
        
        return size_info
    
    def benchmark_inference(self, dummy_input: torch.Tensor, n_runs: int = 100) -> dict:
        """
        Benchmark inference speed.
        
        Args:
            dummy_input: Dummy input tensor
            n_runs: Number of inference runs
        
        Returns:
            benchmark_results: Dictionary with benchmark results
        """
        import time
        
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy_input)
        
        # Synchronize CUDA if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / n_runs
        
        benchmark_results = {
            'total_time': total_time,
            'avg_time_per_inference': avg_time,
            'inferences_per_second': 1.0 / avg_time if avg_time > 0 else 0,
            'n_runs': n_runs
        }
        
        logger.info(f"Benchmark: {avg_time*1000:.2f}ms per inference, {benchmark_results['inferences_per_second']:.2f} inf/s")
        
        return benchmark_results


class ModelCache:
    """
    Cache for storing preprocessed data and model outputs.
    """
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
        
    def get(self, key: str) -> Optional[any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
        
        Returns:
            item: Cached item or None
        """
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: any):
        """
        Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = value
        self.access_order.append(key)
        
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()
        
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


def optimize_model_for_deployment(
    model: nn.Module,
    quantize: bool = True,
    prune: bool = False,
    pruning_amount: float = 0.2
) -> nn.Module:
    """
    Optimize model for deployment.
    
    Args:
        model: Model to optimize
        quantize: Whether to apply quantization
        prune: Whether to apply pruning
        pruning_amount: Pruning amount
    
    Returns:
        optimized_model: Optimized model
    """
    optimizer = ModelOptimizer(model)
    
    if prune:
        optimizer.prune_model(pruning_amount)
    
    if quantize:
        model = optimizer.quantize_model('dynamic')
    
    optimizer.optimize_for_inference()
    
    return model