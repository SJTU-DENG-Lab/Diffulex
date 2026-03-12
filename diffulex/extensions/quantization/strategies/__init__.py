"""
Quantization Strategy Implementations

All concrete strategy implementations.
Importing this module registers all strategies.
"""

# Import all strategies to register them
# Order matters - avoid circular imports

# KV Cache strategies
from .kv_cache_bf16 import BF16KVCacheStrategy
from .kv_cache_fp8_running_max import FP8E4M3KVCacheStrategy, FP8E5M2KVCacheStrategy

# BF16 strategy (base)
from .linear_bf16 import BF16LinearStrategy
from .linear_no_quantization import NoQuantizationStrategy

# FP8 strategies
from .linear_fp8_w8a8 import FP8E4M3W8A8LinearStrategy, FP8E5M2W8A8LinearStrategy
from .linear_fp8_w8a16 import FP8E4M3W8A16LinearStrategy, FP8E5M2W8A16LinearStrategy

# INT8 strategies  
from .linear_int8_w8a8 import INT8W8A8LinearStrategy
from .linear_int8_w8a16 import INT8W8A16LinearStrategy

# GPTQ strategies (2, 3, 4, 8 bit) - unified implementation
from .linear_gptq_wxa16 import (
    GPTQW2A16LinearStrategy,
    GPTQW3A16LinearStrategy,
    GPTQW4A16LinearStrategy,
    GPTQW8A16LinearStrategy,
)

# GPTQ + Marlin strategies (4, 8 bit)
from .linear_gptq_marlin_w4a16 import GPTQMarlinW4A16LinearStrategy
from .linear_gptq_marlin_w8a16 import GPTQMarlinW8A16LinearStrategy

# AWQ strategies
from .linear_awq_w4a16 import AWQW4A16LinearStrategy
from .linear_awq_marlin_w4a16 import AWQMarlinW4A16LinearStrategy

# CUTLASS W4A8 (Hopper+)
from .linear_w4a8_cutlass import CutlassW4A8LinearStrategy

__all__ = [
    # KV Cache
    "BF16KVCacheStrategy",
    "FP8E4M3KVCacheStrategy",
    "FP8E5M2KVCacheStrategy",
    # Linear
    "BF16LinearStrategy",
    "NoQuantizationStrategy",
    # FP8
    "FP8E4M3W8A8LinearStrategy",
    "FP8E5M2W8A8LinearStrategy",
    "FP8E4M3W8A16LinearStrategy",
    "FP8E5M2W8A16LinearStrategy",
    # INT8
    "INT8W8A8LinearStrategy",
    "INT8W8A16LinearStrategy",
    # GPTQ (2, 3, 4, 8 bit)
    "GPTQW2A16LinearStrategy",
    "GPTQW3A16LinearStrategy",
    "GPTQW4A16LinearStrategy",
    "GPTQW8A16LinearStrategy",
    # GPTQ + Marlin (4, 8 bit)
    "GPTQMarlinW4A16LinearStrategy",
    "GPTQMarlinW8A16LinearStrategy",
    # AWQ
    "AWQW4A16LinearStrategy",
    "AWQMarlinW4A16LinearStrategy",
    # CUTLASS W4A8 (Hopper+)
    "CutlassW4A8LinearStrategy",
]
