"""Diffulex strategy package that imports built-in strategies to trigger registration."""
from __future__ import annotations
import importlib
from pathlib import Path

# Import built-in strategies so their registrations run at import time.
# Automatically import all subdirectory packages in the current directory
_excluded_dirs = {"__pycache__", "__init__"}
_strategy_modules = []

_current_dir = Path(__file__).parent
for item in _current_dir.iterdir():
    if item.is_dir() and not item.name.startswith("_") and item.name not in _excluded_dirs:
        # Check if it's a Python package (has __init__.py)
        init_file = item / "__init__.py"
        if init_file.exists():
            try:
                importlib.import_module(f".{item.name}", __name__)
                _strategy_modules.append(item.name)
            except Exception as e:
                # Skip packages that fail to import
                import warnings
                warnings.warn(f"Failed to import strategy {item.name}: {e}", ImportWarning)

__all__ = _strategy_modules.copy()

DECODING_STRATEGY = None

def fetch_decoding_strategy() -> str | None:
    return DECODING_STRATEGY

def set_decoding_strategy(strategy: str) -> None:
    global DECODING_STRATEGY
    DECODING_STRATEGY = strategy

def reset_decoding_strategy() -> None:
    global DECODING_STRATEGY
    DECODING_STRATEGY = None