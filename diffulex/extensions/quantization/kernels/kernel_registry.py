"""
Kernel Registry

Abstract base classes and registry for all quantization kernels.
Provides unified interface for vLLM kernels and custom Triton kernels.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, List
import warnings


class BaseKernel(ABC):
    """Abstract base class for all quantization kernels."""
    
    name: str = "base"
    description: str = "Base kernel"
    
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this kernel is available on the current system."""
        pass
    
    @classmethod
    def get_missing_reason(cls) -> Optional[str]:
        """Return reason why kernel is not available, or None if available."""
        if cls.is_available():
            return None
        return f"{cls.name} kernel not available"
    
    def __init__(self):
        if not self.is_available():
            reason = self.get_missing_reason()
            raise RuntimeError(f"Cannot instantiate {self.name}: {reason}")


class LinearKernel(BaseKernel):
    """Base class for linear layer kernels."""
    
    @abstractmethod
    def forward(self, x: Any, weight: Any, **kwargs) -> Any:
        """Execute linear transformation."""
        pass


class KVCacheKernel(BaseKernel):
    """Base class for KV cache operation kernels."""
    
    @abstractmethod
    def attention(
        self,
        q: Any,
        k_cache: Any,
        v_cache: Any,
        **kwargs
    ) -> Any:
        """Compute attention with KV cache."""
        pass


# Global kernel registry
_kernel_registry: Dict[str, Type[BaseKernel]] = {}


def register_kernel(name: str, kernel_cls: Type[BaseKernel] = None):
    """
    Register a kernel class.
    
    Can be used as a decorator:
        @register_kernel("my_kernel")
        class MyKernel(BaseKernel): ...
    
    Or as a function:
        register_kernel("my_kernel", MyKernel)
    """
    def decorator(cls: Type[BaseKernel]):
        _kernel_registry[name] = cls
        return cls
    
    if kernel_cls is not None:
        # Called as function: register_kernel("name", Class)
        return decorator(kernel_cls)
    else:
        # Called as decorator: @register_kernel("name")
        return decorator


def get_kernel(name: str) -> Optional[Type[BaseKernel]]:
    """Get a kernel class by name."""
    return _kernel_registry.get(name)


def list_available_kernels() -> List[str]:
    """List all registered kernel names."""
    return list(_kernel_registry.keys())


def list_working_kernels() -> Dict[str, bool]:
    """List all kernels and their availability status."""
    return {
        name: kernel_cls.is_available()
        for name, kernel_cls in _kernel_registry.items()
    }


class KernelRegistry:
    """Convenient interface for kernel management."""
    
    @staticmethod
    def register(name: str, kernel_cls: Type[BaseKernel]):
        """Register a kernel."""
        return register_kernel(name, kernel_cls)
    
    @staticmethod
    def get(name: str) -> Optional[Type[BaseKernel]]:
        """Get kernel by name."""
        return get_kernel(name)
    
    @staticmethod
    def list_all() -> List[str]:
        """List all registered kernels."""
        return list_available_kernels()
    
    @staticmethod
    def list_available() -> List[str]:
        """List only available kernels."""
        return [
            name for name, cls in _kernel_registry.items()
            if cls.is_available()
        ]
    
    @staticmethod
    def get_status() -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all kernels."""
        status = {}
        for name, kernel_cls in _kernel_registry.items():
            available = kernel_cls.is_available()
            status[name] = {
                "available": available,
                "description": getattr(kernel_cls, 'description', 'No description'),
                "reason": None if available else kernel_cls.get_missing_reason(),
            }
        return status
