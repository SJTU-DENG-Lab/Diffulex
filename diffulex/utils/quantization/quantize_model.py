#!/usr/bin/env python3
"""离线量化脚本：将模型权重量化为 GPTQ/AWQ 格式

支持两种量化格式：
- GPTQ: Groupwise quantization with optional g_idx
- AWQ: Groupwise quantization (no g_idx)

使用方法:
    python -m diffulex.utils.quantization.quantize_model \
        --model-path /path/to/model \
        --output-path /path/to/output \
        --quant-format gptq \
        --group-size 128 \
        --bits 4
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors.torch import save_file

# Import model loading utilities
import sys
from pathlib import Path as PathLib

# Add project root to path
_REPO_ROOT = PathLib(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoConfig, AutoModelForCausalLM
from safetensors import safe_open
from glob import glob


def _pack_int4_to_int8(int4_tensor: torch.Tensor) -> torch.Tensor:
    """Pack int4 tensor into int8 format.
    
    Args:
        int4_tensor: int8 tensor [N, K] with values in [-8, 7]
        
    Returns:
        packed: int8 tensor [N, (K + 1) // 2] with 2 int4 values per byte
    """
    out_features, in_features = int4_tensor.shape
    
    # Clamp to int4 range [-8, 7]
    int4_tensor = int4_tensor.clamp(-8, 7)
    
    # Convert to unsigned: [-8, 7] -> [0, 15]
    uint8_tensor = (int4_tensor + 8).to(torch.uint8)
    
    # Pad to even number of columns if needed
    if in_features % 2 != 0:
        pad_size = 1
        padding = torch.zeros(out_features, pad_size, dtype=torch.uint8, device=uint8_tensor.device) + 8
        uint8_tensor = torch.cat([uint8_tensor, padding], dim=1)
        padded_in_features = in_features + pad_size
    else:
        padded_in_features = in_features
    
    # Reshape to [N, K//2, 2] where first column is even indices, second is odd indices
    reshaped = uint8_tensor.view(out_features, padded_in_features // 2, 2)
    
    # Pack: lower 4 bits = even columns, upper 4 bits = odd columns
    packed = reshaped[:, :, 0] | (reshaped[:, :, 1] << 4)
    return packed.to(torch.int8)


def _quantize_gptq_groupwise(
    weight: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
    g_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Quantize weight using GPTQ groupwise quantization.
    
    Args:
        weight: float32 tensor [out_features, in_features]
        group_size: Group size for quantization (default: 128)
        bits: Number of bits per weight (default: 4)
        g_idx: Optional int32 tensor [out_features] mapping each output channel to its group.
               If None, uses sequential grouping: group_id = out_idx // group_size
    
    Returns:
        qweight: int8 packed int4 weights [out_features, (in_features + 1) // 2]
        qzeros: int8 packed int4 zeros [num_groups, (in_features + 1) // 2]
        scales: float32 per-group scales [num_groups, in_features]
        g_idx: int32 tensor [out_features] group indices (always returned, even if input was None)
    """
    out_features, in_features = weight.shape
    device = weight.device
    
    # Determine group assignments
    if g_idx is None:
        # Sequential grouping: group_id = out_idx // group_size
        group_ids = torch.arange(out_features, device=device) // group_size
    else:
        # Use provided g_idx
        if g_idx.shape != (out_features,):
            raise ValueError(f"g_idx shape mismatch: got {g_idx.shape}, expected ({out_features},)")
        group_ids = g_idx.to(device=device).to(torch.int64)
    
    num_groups = int(group_ids.max().item() + 1)
    
    # Quantize per group
    qweight_list = []
    qzeros_list = []
    scales_list = []
    
    for g in range(num_groups):
        # Get output channels in this group
        group_mask = (group_ids == g)
        group_indices = torch.where(group_mask)[0]
        
        if len(group_indices) == 0:
            continue
            
        group_weight = weight[group_indices]  # [group_out_size, in_features]
        group_out_size = group_weight.shape[0]
        
        # Compute scale and zero point per input feature (per-channel within group)
        # For GPTQ, we use per-channel quantization within each group
        abs_max = torch.abs(group_weight).max(dim=0, keepdim=True)[0]  # [1, in_features]
        scales_group = (abs_max.clamp(min=1e-8) / (2 ** (bits - 1) - 1)).squeeze(0)  # [in_features]
        
        # Compute zero point: mean of group (per-channel)
        zeros_group = group_weight.mean(dim=0)  # [in_features]
        
        # Quantize: (weight - zero) / scale
        quantized_group = ((group_weight - zeros_group.unsqueeze(0)) / scales_group.unsqueeze(0).clamp(min=1e-8))
        quantized_group = quantized_group.round().clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1).to(torch.int8)
        
        # Pack quantized weights
        packed_group = _pack_int4_to_int8(quantized_group)  # [group_out_size, (in_features + 1) // 2]
        qweight_list.append(packed_group)
        
        # Quantize and pack zeros
        zeros_quantized = (zeros_group / scales_group.clamp(min=1e-8)).round().clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1).to(torch.int8)
        zeros_packed = _pack_int4_to_int8(zeros_quantized.unsqueeze(0))  # [1, (in_features + 1) // 2]
        qzeros_list.append(zeros_packed)
        
        # Store scales
        scales_list.append(scales_group.unsqueeze(0))  # [1, in_features]
    
    # Concatenate all groups
    qweight = torch.cat(qweight_list, dim=0)  # [out_features, (in_features + 1) // 2]
    qzeros = torch.cat(qzeros_list, dim=0)  # [num_groups, (in_features + 1) // 2]
    scales = torch.cat(scales_list, dim=0)  # [num_groups, in_features]
    
    # Ensure g_idx is returned (create if was None)
    if g_idx is None:
        g_idx = group_ids.to(torch.int32)
    else:
        g_idx = g_idx.to(torch.int32)
    
    return qweight, qzeros, scales, g_idx


def _quantize_awq_groupwise(
    weight: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weight using AWQ groupwise quantization.
    
    Args:
        weight: float32 tensor [out_features, in_features]
        group_size: Group size for quantization (default: 128)
        bits: Number of bits per weight (default: 4)
    
    Returns:
        qweight: int8 packed int4 weights [out_features, (in_features + 1) // 2]
        qzeros: int8 packed int4 zeros [num_groups, (in_features + 1) // 2]
        scales: float32 per-group scales [num_groups, in_features] or [num_groups]
    """
    out_features, in_features = weight.shape
    device = weight.device
    
    num_groups = (out_features + group_size - 1) // group_size
    
    # Quantize per group (sequential grouping)
    qweight_list = []
    qzeros_list = []
    scales_list = []
    
    for g in range(num_groups):
        start_idx = g * group_size
        end_idx = min((g + 1) * group_size, out_features)
        group_weight = weight[start_idx:end_idx]  # [group_size (or remainder), in_features]
        group_out_size = group_weight.shape[0]
        
        # AWQ: Compute scale per group (can be scalar or per-channel)
        # For simplicity, use per-channel scales within group
        abs_max = torch.abs(group_weight).max(dim=0, keepdim=True)[0]  # [1, in_features]
        scales_group = (abs_max.clamp(min=1e-8) / (2 ** (bits - 1) - 1)).squeeze(0)  # [in_features]
        
        # AWQ: Compute zero point per input channel (per-channel)
        # Use minimum value for better quantization range
        zeros_group = group_weight.min(dim=0)[0]  # [in_features]
        
        # Quantize: (weight - zero) / scale
        quantized_group = ((group_weight - zeros_group.unsqueeze(0)) / scales_group.unsqueeze(0).clamp(min=1e-8))
        quantized_group = quantized_group.round().clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1).to(torch.int8)
        
        # Pack quantized weights
        packed_group = _pack_int4_to_int8(quantized_group)  # [group_out_size, (in_features + 1) // 2]
        qweight_list.append(packed_group)
        
        # Quantize and pack zeros
        zeros_quantized = (zeros_group / scales_group.clamp(min=1e-8)).round().clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1).to(torch.int8)
        zeros_packed = _pack_int4_to_int8(zeros_quantized.unsqueeze(0))  # [1, (in_features + 1) // 2]
        qzeros_list.append(zeros_packed)
        
        # Store scales
        scales_list.append(scales_group.unsqueeze(0))  # [1, in_features]
    
    # Concatenate all groups
    qweight = torch.cat(qweight_list, dim=0)  # [out_features, (in_features + 1) // 2]
    qzeros = torch.cat(qzeros_list, dim=0)  # [num_groups, (in_features + 1) // 2]
    scales = torch.cat(scales_list, dim=0)  # [num_groups, in_features]
    
    return qweight, qzeros, scales


def quantize_model(
    model_path: str,
    output_path: str,
    quant_format: str = "gptq",
    group_size: int = 128,
    bits: int = 4,
    target_modules: Optional[list[str]] = None,
    device: str = "cpu",
) -> None:
    """Quantize model weights to GPTQ/AWQ format.
    
    Args:
        model_path: Path to input model directory (containing safetensors files)
        output_path: Path to output directory (will create if not exists)
        quant_format: "gptq" or "awq"
        group_size: Group size for quantization (default: 128)
        bits: Number of bits per weight (default: 4)
        target_modules: List of module name patterns to quantize (e.g., ["q_proj", "k_proj"]).
                       If None, quantizes all linear layers.
        device: Device to use for quantization ("cpu" or "cuda")
    """
    if quant_format not in ["gptq", "awq"]:
        raise ValueError(f"Unsupported quant_format: {quant_format}. Must be 'gptq' or 'awq'")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model weights from safetensors files
    safetensors_files = list(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")
    
    print(f"Found {len(safetensors_files)} safetensors files")
    
    # Collect all weight names
    all_weight_keys = []
    for file in safetensors_files:
        with safe_open(file, "pt", device) as f:
            all_weight_keys.extend(f.keys())
    
    # Filter to linear layer weights only (exclude biases and non-linear layers)
    linear_weight_keys = []
    for key in all_weight_keys:
        # Skip biases, layer norms, embeddings, etc.
        # Note: lm_head is excluded because ParallelLMHead doesn't support offline quantization yet
        if any(skip in key for skip in [".bias", ".norm", ".embed", ".lm_head"]):
            continue
        # Only process weight parameters
        if not key.endswith(".weight"):
            continue
        # Check if target_modules filter applies
        if target_modules:
            if not any(target in key for target in target_modules):
                continue
        linear_weight_keys.append(key)
    
    print(f"Found {len(linear_weight_keys)} linear layer weights to quantize")
    
    # Quantize each linear layer
    quantized_weights = {}
    metadata = {
        "quant_format": quant_format,
        "group_size": group_size,
        "bits": bits,
        "quantized_modules": [],
    }
    
    for key in tqdm(linear_weight_keys, desc="Quantizing weights"):
        # Load weight from safetensors
        weight = None
        source_file = None
        for file in safetensors_files:
            with safe_open(file, "pt", device) as f:
                if key in f.keys():
                    weight = f.get_tensor(key)
                    source_file = file
                    break
        
        if weight is None:
            print(f"Warning: Could not load weight for {key}")
            continue
        
        # Skip if weight is not 2D (not a linear layer weight)
        if weight.dim() != 2:
            print(f"Skipping {key}: not a 2D weight (shape: {weight.shape})")
            continue
        
        out_features, in_features = weight.shape
        
        # Convert to float32 for quantization
        weight_fp32 = weight.to(torch.float32).to(device)
        
        # Quantize
        if quant_format == "gptq":
            qweight, qzeros, scales, g_idx = _quantize_gptq_groupwise(
                weight_fp32, group_size=group_size, bits=bits, g_idx=None
            )
            # Save quantized weights with module prefix
            prefix = key[:-7]  # Remove ".weight"
            quantized_weights[f"{prefix}.qweight"] = qweight.cpu()
            quantized_weights[f"{prefix}.qzeros"] = qzeros.cpu()
            quantized_weights[f"{prefix}.scales"] = scales.cpu()
            quantized_weights[f"{prefix}.g_idx"] = g_idx.cpu()
            quantized_weights[f"{prefix}.group_size"] = torch.tensor(group_size, dtype=torch.int32)
            quantized_weights[f"{prefix}.bits"] = torch.tensor(bits, dtype=torch.int32)
        else:  # awq
            qweight, qzeros, scales = _quantize_awq_groupwise(
                weight_fp32, group_size=group_size, bits=bits
            )
            # Save quantized weights with module prefix
            prefix = key[:-7]  # Remove ".weight"
            quantized_weights[f"{prefix}.qweight"] = qweight.cpu()
            quantized_weights[f"{prefix}.qzeros"] = qzeros.cpu()
            quantized_weights[f"{prefix}.scales"] = scales.cpu()
            quantized_weights[f"{prefix}.group_size"] = torch.tensor(group_size, dtype=torch.int32)
            quantized_weights[f"{prefix}.bits"] = torch.tensor(bits, dtype=torch.int32)
        
        metadata["quantized_modules"].append({
            "name": prefix,
            "out_features": int(out_features),
            "in_features": int(in_features),
            "group_size": group_size,
            "bits": bits,
        })
        
        # Clear GPU cache if using CUDA
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Copy all model files (config, tokenizer, etc.) to output directory
    import shutil
    print(f"\nCopying model files to {output_path}...")
    model_path_obj = Path(model_path)
    
    # First, copy original safetensors files (for non-quantized layers like lm_head, embeddings, etc.)
    print("  Copying original safetensors files (for non-quantized layers)...")
    for file in model_path_obj.glob("*.safetensors"):
        dest_file = output_path / file.name
        shutil.copy2(file, dest_file)
        print(f"    Copied {file.name}")
    
    # Copy other non-safetensors files
    for file in model_path_obj.iterdir():
        if file.is_file() and not file.name.endswith('.safetensors'):
            dest_file = output_path / file.name
            shutil.copy2(file, dest_file)
            print(f"  Copied {file.name}")
    
    # Save quantized weights to safetensors (this will add quantized weights to the directory)
    output_file = output_path / f"model_quantized_{quant_format}.safetensors"
    print(f"\nSaving quantized weights to {output_file}...")
    save_file(quantized_weights, output_file)
    
    # Save metadata
    metadata_file = output_path / f"quantization_metadata_{quant_format}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Quantization complete!")
    print(f"  - Quantized {len(metadata['quantized_modules'])} modules")
    print(f"  - Output directory: {output_path}")
    print(f"  - Quantized weights file: {output_file}")
    print(f"  - Metadata file: {metadata_file}")
    print(f"\n  You can now use this directory directly as model path:")
    print(f"    --model-path {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="离线量化模型权重为 GPTQ/AWQ 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, required=True, help="输入模型路径")
    parser.add_argument("--output-path", type=str, required=True, help="输出路径")
    parser.add_argument("--quant-format", type=str, choices=["gptq", "awq"], default="gptq", help="量化格式: gptq 或 awq")
    parser.add_argument("--group-size", type=int, default=128, help="量化组大小 (默认: 128)")
    parser.add_argument("--bits", type=int, default=4, help="每个权重的位数 (默认: 4)")
    parser.add_argument("--target-modules", type=str, help="要量化的模块名称模式（逗号分隔），例如: q_proj,k_proj,v_proj")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="量化设备 (默认: cpu)")
    
    args = parser.parse_args()
    
    target_modules = None
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",")]
    
    quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        quant_format=args.quant_format,
        group_size=args.group_size,
        bits=args.bits,
        target_modules=target_modules,
        device=args.device,
    )


if __name__ == "__main__":
    main()
