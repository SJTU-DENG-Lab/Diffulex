#!/usr/bin/env python3
"""Quick test to verify monkey patch works"""

from test.python.engine.dummy_attn_with_validation import install_validation_hook

print("Before patch:")
from diffulex.attention.attn_impl import Attention
print(f"  Attention class: {Attention}")

install_validation_hook()

print("\nAfter patch:")
# Force reimport
import sys
if 'diffulex.attention.attn_impl' in sys.modules:
    del sys.modules['diffulex.attention.attn_impl']
from diffulex.attention.attn_impl import Attention
print(f"  Attention class: {Attention}")
print(f"  Has validation: {hasattr(Attention, 'validation_enabled')}")
