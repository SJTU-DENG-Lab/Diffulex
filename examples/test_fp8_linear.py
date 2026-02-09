"""
End-to-end test for FP8 Linear quantization.

This script tests FP8 Linear strategies in a complete inference pipeline.
Note: This is a basic smoke test. For full model inference, see the main
test scripts in the examples directory.
"""

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import create_linear_strategy
from diffulex.utils.quantization.context import get_quantization_context


def test_fp8_w8a16_e2e():
    """End-to-end test for FP8 W8A16 strategy."""
    print("Testing FP8 W8A16 (e4m3) strategy...")
    
    # Create strategy
    strategy = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
    ctx = get_quantization_context()
    ctx.set_linear_strategy("attn", strategy)
    
    # Simulate a small attention projection
    M, K, N = 32, 512, 256  # batch_size=32, hidden_size=512, num_heads*head_dim=256
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.randn(N, K, dtype=torch.bfloat16, device=x.device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=x.device)
    
    # Reference output
    ref_out = F.linear(x, weight, bias)
    
    # FP8 quantized output
    fp8_out = strategy.linear_forward(x, weight, bias, quant_kind="attn")
    
    # Check output
    assert fp8_out.shape == ref_out.shape
    assert fp8_out.dtype == torch.bfloat16
    
    # Compute error metrics
    max_error = torch.abs(fp8_out - ref_out).max().item()
    mean_error = torch.abs(fp8_out - ref_out).mean().item()
    relative_error = (torch.abs(fp8_out - ref_out) / (ref_out.abs() + 1e-8)).mean().item()
    
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    print(f"  Mean relative error: {relative_error:.6f}")
    print(f"  Output range: [{fp8_out.min().item():.3f}, {fp8_out.max().item():.3f}]")
    print("  ✓ FP8 W8A16 test passed")
    
    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "relative_error": relative_error,
    }


def test_fp8_w8a8_e2e():
    """End-to-end test for FP8 W8A8 strategy."""
    print("Testing FP8 W8A8 (e4m3) strategy...")
    
    # Create strategy
    strategy = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="fp8_e4m3")
    ctx = get_quantization_context()
    ctx.set_linear_strategy("attn", strategy)
    
    # Simulate a small attention projection
    M, K, N = 32, 512, 256
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.randn(N, K, dtype=torch.bfloat16, device=x.device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=x.device)
    
    # Reference output
    ref_out = F.linear(x, weight, bias)
    
    # FP8 quantized output
    fp8_out = strategy.linear_forward(x, weight, bias, quant_kind="attn")
    
    # Check output
    assert fp8_out.shape == ref_out.shape
    assert fp8_out.dtype == torch.bfloat16
    
    # Compute error metrics
    max_error = torch.abs(fp8_out - ref_out).max().item()
    mean_error = torch.abs(fp8_out - ref_out).mean().item()
    relative_error = (torch.abs(fp8_out - ref_out) / (ref_out.abs() + 1e-8)).mean().item()
    
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    print(f"  Mean relative error: {relative_error:.6f}")
    print(f"  Output range: [{fp8_out.min().item():.3f}, {fp8_out.max().item():.3f}]")
    print("  ✓ FP8 W8A8 test passed")
    
    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "relative_error": relative_error,
    }


def test_memory_usage():
    """Test memory usage comparison (basic check)."""
    print("Testing memory usage...")
    
    if not torch.cuda.is_available():
        print("  Skipping memory test (CUDA not available)")
        return
    
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # BF16 baseline
    M, K, N = 32, 512, 256
    weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    mem_bf16 = torch.cuda.memory_allocated()
    
    # FP8 quantized
    strategy = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
    weight_fp8, scales = strategy.quantize_weight_for_kernel(weight_bf16, device=device)
    mem_fp8 = torch.cuda.memory_allocated()
    
    # Memory reduction
    weight_size_bf16 = weight_bf16.numel() * 2  # bf16 = 2 bytes
    weight_size_fp8 = weight_fp8.numel() * 1 + scales.numel() * 4  # uint8 = 1 byte, float32 = 4 bytes
    reduction = (1 - weight_size_fp8 / weight_size_bf16) * 100
    
    print(f"  BF16 weight size: {weight_size_bf16 / 1024:.2f} KB")
    print(f"  FP8 weight size: {weight_size_fp8 / 1024:.2f} KB")
    print(f"  Memory reduction: {reduction:.1f}%")
    print("  ✓ Memory test passed")


def main():
    """Run all end-to-end tests."""
    print("=" * 60)
    print("FP8 Linear Quantization End-to-End Tests")
    print("=" * 60)
    print()
    
    try:
        # Test FP8 W8A16
        w8a16_metrics = test_fp8_w8a16_e2e()
        print()
        
        # Test FP8 W8A8
        w8a8_metrics = test_fp8_w8a8_e2e()
        print()
        
        # Test memory usage
        test_memory_usage()
        print()
        
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        print()
        print("Summary:")
        print(f"  FP8 W8A16 - Max error: {w8a16_metrics['max_error']:.6f}")
        print(f"  FP8 W8A8  - Max error: {w8a8_metrics['max_error']:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

