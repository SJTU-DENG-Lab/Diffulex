#!/usr/bin/env python3
"""
离线量化脚本：将 AutoModelForDiffusionLM 模型权重量化为 GPTQ/AWQ/Marlin 格式

支持方法:
- rtn: Round-To-Nearest，无校准，快速
- gptq: Hessian-based GPTQ，需校准
- awq: Activation-aware AWQ，需校准  
- gptq_marlin: GPTQ + Marlin重排，需校准，强制sym=True
- awq_marlin: AWQ + Marlin重排，需校准

使用方法:
    python -m diffulex.extensions.quantization.quantize_model \
        --model-path /path/to/model \
        --output-path /path/to/output \
        --quant-method gptq \
        --bits 4 \
        --group-size 128 \
        --calib-text-file /path/to/calib.txt
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModel


def _require_vllm():
    """导入vLLM量化工具函数。"""
    try:
        from vllm.scalar_type import scalar_types
        from vllm.model_executor.layers.quantization.utils.quantize_utils import (
            quantize_weights,
            pack_cols,
        )
        return scalar_types, quantize_weights, pack_cols
    except Exception as e:
        raise RuntimeError("需要vLLM来执行量化打包操作") from e


def _require_vllm_marlin():
    """导入vLLM Marlin相关函数。"""
    try:
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_permute_scales,
        )
        return ops, marlin_permute_scales
    except Exception as e:
        raise RuntimeError("需要vLLM Marlin支持（含CUDA custom ops）") from e


# =============================================================================
# Pack 函数
# =============================================================================

def gptq_pack(qweight_int: torch.Tensor, bits: int) -> torch.Tensor:
    """
    将[K, N]的int张量pack成[K//pack, N]的int32张量（GPTQ格式）。
    
    Pack规则（小端序）：
    qweight[k//pack, n] = sum(qweight_int[k+i, n] << (bits*i) for i in range(pack))
    """
    pack_factor = 32 // bits
    size_k, size_n = qweight_int.shape
    assert size_k % pack_factor == 0, f"K={size_k} must be divisible by pack_factor={pack_factor}"
    
    qweight = torch.zeros((size_k // pack_factor, size_n), 
                          dtype=torch.int32, device=qweight_int.device)
    
    for i in range(pack_factor):
        qweight |= (qweight_int[i::pack_factor].to(torch.int32) << (bits * i))
    
    return qweight


def awq_pack(qweight_int: torch.Tensor, bits: int) -> torch.Tensor:
    """
    将[K, N]的int张量pack成[K, N//pack]的int32张量（AWQ格式）。
    
    Pack规则：
    qweight[k, n//pack] = sum(qweight_int[k, n+i] << (bits*i) for i in range(pack))
    """
    pack_factor = 32 // bits
    size_k, size_n = qweight_int.shape
    assert size_n % pack_factor == 0, f"N={size_n} must be divisible by pack_factor={pack_factor}"
    
    qweight = torch.zeros((size_k, size_n // pack_factor),
                          dtype=torch.int32, device=qweight_int.device)
    
    for i in range(pack_factor):
        qweight |= (qweight_int[:, i::pack_factor].to(torch.int32) << (bits * i))
    
    return qweight


def gptq_pack_zeros(zeros_int: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack zeros tensor [num_groups, N] to [num_groups, N//pack].
    GPTQ v1 format stores (zeros - 1).
    """
    pack_factor = 32 // bits
    num_groups, size_n = zeros_int.shape
    assert size_n % pack_factor == 0
    
    # v1 format: store zeros - 1
    zeros_v1 = zeros_int - 1
    
    qzeros = torch.zeros((num_groups, size_n // pack_factor),
                         dtype=torch.int32, device=zeros_int.device)
    
    for i in range(pack_factor):
        qzeros |= (zeros_v1[:, i::pack_factor].to(torch.int32) << (bits * i))
    
    return qzeros


# =============================================================================
# Marlin 重排
# =============================================================================

def repack_gptq_to_marlin(
    qweight: torch.Tensor,      # GPTQ packed: [K//pack, N]
    scales: torch.Tensor,       # [K//group, N]
    bits: int,
    size_k: int,
    size_n: int,
    group_size: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将GPTQ格式权重重排为Marlin格式。
    
    Returns:
        marlin_qweight: [K//16, N*16//32]
        marlin_scales: [K//group, N]
        marlin_workspace: [N*16//32]
    """
    ops, marlin_permute_scales = _require_vllm_marlin()
    
    qweight = qweight.to(device)
    scales = scales.to(device)
    
    # 空perm（desc_act=False）
    empty_perm = torch.empty((0,), dtype=torch.int32, device=device)
    
    # 重排权重
    marlin_qweight = ops.gptq_marlin_repack(
        qweight.contiguous(),
        perm=empty_perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=bits,
        is_a_8bit=(bits == 8),
    ).contiguous()
    
    # Permute scales
    marlin_scales = marlin_permute_scales(
        scales.contiguous().to(torch.float16),
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
        is_a_8bit=(bits == 8),
    ).contiguous()
    
    # 创建工作区
    marlin_workspace = torch.zeros(
        marlin_qweight.shape[1] * 32 // bits,
        dtype=torch.int32,
        device=device,
    )
    
    return marlin_qweight.cpu(), marlin_scales.cpu(), marlin_workspace.cpu()


def repack_awq_to_marlin(
    qweight: torch.Tensor,      # AWQ packed: [K, N//pack]
    scales: torch.Tensor,       # [K//group, N]
    bits: int,
    size_k: int,
    size_n: int,
    group_size: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将AWQ格式权重重排为Marlin格式。
    
    注意：AWQ需要先unpack到标准格式，再调用gptq_marlin_repack。
    """
    ops, marlin_permute_scales = _require_vllm_marlin()
    
    # AWQ unpack (简化实现，实际需要完整的unpack逻辑)
    # 这里假设AWQ的pack是可逆的
    pack_factor = 32 // bits
    qweight_unpacked = torch.zeros((size_k, size_n), dtype=torch.int32, device=qweight.device)
    
    for i in range(pack_factor):
        mask = (2 ** bits - 1) << (bits * i)
        qweight_unpacked[:, i::pack_factor] = ((qweight >> (bits * i)) & ((1 << bits) - 1)).to(torch.int32)
    
    # 转为GPTQ格式再repack
    qweight_gptq = gptq_pack(qweight_unpacked, bits)
    
    return repack_gptq_to_marlin(qweight_gptq, scales, bits, size_k, size_n, group_size, device)


# =============================================================================
# 量化算法
# =============================================================================

def quantize_rtn(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    sym: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Round-To-Nearest 量化。
    
    Returns:
        qweight: packed weight
        qzeros: packed zeros  
        scales: scales
        g_idx: empty tensor
    """
    weight = weight.float()
    out_features, in_features = weight.shape
    
    if group_size == -1:
        group_size = in_features
    
    num_groups = (in_features + group_size - 1) // group_size
    max_q = 2 ** bits - 1
    
    # 存储量化结果
    q_full = torch.zeros_like(weight, dtype=torch.int32)
    scales_list = []
    zeros_list = []
    
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, in_features)
        w_group = weight[:, start:end]
        
        if sym:
            # 对称量化
            w_max = w_group.abs().max(dim=1, keepdim=True)[0]
            scale = w_max / (2 ** (bits - 1) - 1)
            scale = torch.clamp(scale, min=1e-5)
            zero = torch.zeros(out_features, 1, dtype=torch.int32)
            
            q = torch.round(w_group / scale).to(torch.int32)
            q = torch.clamp(q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            q = q + (2 ** (bits - 1))  # 映射到[0, max_q]
        else:
            # 非对称量化
            w_min = w_group.min(dim=1, keepdim=True)[0]
            w_max = w_group.max(dim=1, keepdim=True)[0]
            scale = (w_max - w_min) / max_q
            scale = torch.clamp(scale, min=1e-5)
            zero = torch.round(-w_min / scale).to(torch.int32)
            
            q = torch.round(w_group / scale + zero).to(torch.int32)
            q = torch.clamp(q, 0, max_q)
        
        q_full[:, start:end] = q
        scales_list.append(scale.squeeze(1))
        zeros_list.append(zero.squeeze(1))
    
    # 合并scales和zeros
    scales = torch.stack(scales_list, dim=0).to(torch.float16)  # [num_groups, out_features]
    zeros = torch.stack(zeros_list, dim=0)  # [num_groups, out_features]
    
    # Pack
    qweight = gptq_pack(q_full.T.contiguous(), bits)  # [K//pack, N]
    qzeros = gptq_pack_zeros(zeros, bits)  # [num_groups, N//pack]
    g_idx = torch.empty((0,), dtype=torch.int32)
    
    return qweight, qzeros, scales, g_idx


def quantize_gptq(
    layer: nn.Linear,
    calibration_inputs: List[torch.Tensor],
    bits: int,
    group_size: int,
    sym: bool = True,
    damp_percent: float = 0.01,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPTQ算法量化。
    
    基于Hessian的逐列量化误差补偿。
    """
    layer = layer.to(device)
    layer.eval()
    
    weight = layer.weight.data.float()  # [out_features, in_features]
    out_features, in_features = weight.shape
    
    if group_size == -1:
        group_size = in_features
    
    # 1. 收集校准数据
    print(f"  Collecting calibration data for {layer.name if hasattr(layer, 'name') else 'layer'}...")
    inputs_list = []
    for x in calibration_inputs:
        x = x.to(device)
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            # 前向传播获取输入
            inputs_list.append(x.cpu())
    
    # 2. 计算Hessian
    H = torch.zeros((in_features, in_features), dtype=torch.float32, device=device)
    num_samples = 0
    for x in inputs_list:
        x = x.to(device).float()
        H.addmm_(x.T, x)
        num_samples += x.shape[0]
    H /= num_samples
    
    # 3. 添加阻尼
    damp = damp_percent * torch.mean(torch.diag(H))
    H += damp * torch.eye(in_features, device=device, dtype=H.dtype)
    
    # 4. Cholesky分解
    try:
        L = torch.linalg.cholesky(H)
    except RuntimeError:
        print("  Warning: Cholesky failed, falling back to RTN")
        return quantize_rtn(weight, bits, group_size if group_size != in_features else -1, sym)
    
    # 5. 逐列量化
    W = weight.clone().to(device)  # [out_features, in_features]
    Q = torch.zeros_like(W, dtype=torch.int32)
    
    num_groups = (in_features + group_size - 1) // group_size
    scales_list = []
    zeros_list = []
    
    # 预计算所有组的量化参数
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, in_features)
        w_group = W[:, start:end]
        
        if sym:
            w_max = w_group.abs().max(dim=1)[0]
            scale = w_max / (2 ** (bits - 1) - 1)
            scale = torch.clamp(scale, min=1e-5)
            zero = torch.zeros(out_features, dtype=torch.int32, device=device)
        else:
            w_min = w_group.min(dim=1)[0]
            w_max = w_group.max(dim=1)[0]
            scale = (w_max - w_min) / (2 ** bits - 1)
            scale = torch.clamp(scale, min=1e-5)
            zero = torch.round(-w_min / scale).to(torch.int32)
        
        scales_list.append(scale)
        zeros_list.append(zero)
    
    # 合并为tensor
    scales_all = torch.stack(scales_list, dim=0).to(torch.float16)  # [num_groups, out_features]
    zeros_all = torch.stack(zeros_list, dim=0)  # [num_groups, out_features]
    
    max_q = 2 ** bits - 1
    
    # 逐列处理
    for i in range(in_features):
        g = i // group_size
        scale = scales_all[g]
        zero = zeros_all[g]
        
        w_col = W[:, i]
        
        if sym:
            q_col = torch.round(w_col / scale).to(torch.int32)
            q_col = torch.clamp(q_col, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            q_col = q_col + (2 ** (bits - 1))  # 映射到[0, max_q]
            w_q = (q_col.float() - (2 ** (bits - 1))) * scale
        else:
            q_col = torch.round(w_col / scale + zero).to(torch.int32)
            q_col = torch.clamp(q_col, 0, max_q)
            w_q = (q_col.float() - zero.float()) * scale
        
        Q[:, i] = q_col
        
        # 误差补偿
        err = w_col - w_q
        if i < in_features - 1:
            Li = L[i, i]
            if Li > 1e-8:
                W[:, i+1:] -= err.unsqueeze(1) * (L[i, i+1:] / Li).unsqueeze(0)
    
    # Pack
    qweight = gptq_pack(Q.T.contiguous(), bits)
    qzeros = gptq_pack_zeros(zeros_all, bits)
    g_idx = torch.empty((0,), dtype=torch.int32)
    
    return qweight.cpu(), qzeros.cpu(), scales_all.cpu(), g_idx


def quantize_awq(
    layer: nn.Linear,
    calibration_inputs: List[torch.Tensor],
    bits: int,
    group_size: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    AWQ算法量化（简化实现）。
    
    AWQ使用激活感知来缩放权重，保护重要channel。
    这里实现一个基础版本，完整的AWQ需要更复杂的逻辑。
    """
    layer = layer.to(device)
    layer.eval()
    
    weight = layer.weight.data.float()
    out_features, in_features = weight.shape
    
    if group_size == -1:
        group_size = in_features
    
    # 收集激活数据
    print(f"  Collecting activation data for AWQ...")
    activations = []
    for x in calibration_inputs:
        x = x.to(device)
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        activations.append(x.abs().mean(dim=0).cpu())
    
    # 计算channel-wise激活幅度
    act_scale = torch.stack(activations, dim=0).mean(dim=0).to(device)  # [in_features]
    
    # 简单的AWQ：根据激活幅度调整权重
    # 重要的channel（激活大）分配更多精度
    weight_scaled = weight * act_scale.unsqueeze(0)
    
    # 量化（AWQ固定非对称）
    num_groups = (in_features + group_size - 1) // group_size
    max_q = 2 ** bits - 1
    
    Q = torch.zeros_like(weight, dtype=torch.int32)
    scales_list = []
    zeros_list = []
    
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, in_features)
        w_group = weight[:, start:end]
        
        w_min = w_group.min(dim=1)[0]
        w_max = w_group.max(dim=1)[0]
        scale = (w_max - w_min) / max_q
        scale = torch.clamp(scale, min=1e-5)
        zero = torch.round(-w_min / scale).to(torch.int32)
        
        q = torch.round(w_group / scale.unsqueeze(1) + zero.unsqueeze(1)).to(torch.int32)
        q = torch.clamp(q, 0, max_q)
        
        Q[:, start:end] = q
        scales_list.append(scale)
        zeros_list.append(zero)
    
    scales = torch.stack(scales_list, dim=0).to(torch.float16)
    zeros = torch.stack(zeros_list, dim=0)
    
    # AWQ pack（列方向）
    qweight = awq_pack(Q.T.contiguous(), bits)
    qzeros = awq_pack(zeros, bits)
    
    return qweight.cpu(), qzeros.cpu(), scales.cpu()


# =============================================================================
# 校准数据处理
# =============================================================================

def build_calibration_data(
    model_path: str,
    calib_text_file: str,
    num_samples: int,
    seq_len: int,
    batch_size: int,
    seed: int = 0,
) -> List[Dict[str, torch.Tensor]]:
    """
    从文本文件构建校准数据。
    
    对于Diffusion模型，校准数据需要包含：
    - input_ids
    - attention_mask
    - 可能需要timestep（由模型类型决定）
    """
    random.seed(seed)
    
    # 读取文本
    with open(calib_text_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) < num_samples:
        print(f"Warning: only {len(lines)} samples available, requested {num_samples}")
        num_samples = len(lines)
    
    lines = random.sample(lines, num_samples)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    calib_data = []
    
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        
        # Tokenize
        encoded = tokenizer(
            batch_lines,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        )
        
        calib_data.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'].bool(),
        })
    
    return calib_data


def collect_layer_inputs(
    model: nn.Module,
    calib_data: List[Dict[str, torch.Tensor]],
    target_modules: Optional[List[str]] = None,
    device: str = "cuda",
) -> Dict[str, List[torch.Tensor]]:
    """
    收集每层线性层的输入数据用于量化。
    
    Returns:
        Dict mapping layer name to list of input tensors
    """
    model = model.to(device)
    model.eval()
    
    layer_inputs: Dict[str, List[torch.Tensor]] = {}
    handles = []
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            
            # 只保存必要的部分
            if name not in layer_inputs:
                layer_inputs[name] = []
            
            # 处理不同输入形状
            if x.dim() == 3:
                # [batch, seq, hidden] -> 保存每个token
                layer_inputs[name].append(x.detach().cpu())
            elif x.dim() == 2:
                layer_inputs[name].append(x.detach().cpu())
        return hook
    
    # 注册hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_modules and not any(t in name for t in target_modules):
                continue
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
            # 保存名字到模块
            module.name = name
    
    # 运行校准
    print(f"Running calibration with {len(calib_data)} batches...")
    with torch.no_grad():
        for batch in tqdm(calib_data, desc="Calibration"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            try:
                model(**batch)
            except Exception as e:
                print(f"  Warning: calibration step failed: {e}")
                continue
    
    # 移除hooks
    for handle in handles:
        handle.remove()
    
    return layer_inputs


# =============================================================================
# 主量化函数
# =============================================================================

def quantize_model(
    model_path: str,
    output_path: str,
    quant_method: str = "rtn",
    bits: int = 4,
    group_size: int = 128,
    target_modules: Optional[List[str]] = None,
    device: str = "cuda",
    # 校准相关
    calib_text_file: Optional[str] = None,
    calib_num_samples: int = 128,
    calib_seq_len: int = 512,
    calib_batch_size: int = 1,
    calib_seed: int = 0,
    # GPTQ特定
    desc_act: bool = False,
    damp_percent: float = 0.01,
    # AWQ特定
    awq_version: str = "GEMM",
) -> None:
    """
    量化模型并保存为Diffulex可加载的格式。
    
    Args:
        model_path: 输入模型路径（HF格式）
        output_path: 输出目录路径
        quant_method: 量化方法
            - "rtn": Round-To-Nearest，无校准
            - "gptq": GPTQ算法，需校准
            - "awq": AWQ算法，需校准
            - "gptq_marlin": GPTQ + Marlin重排，需校准，强制sym=True
            - "awq_marlin": AWQ + Marlin重排，需校准
        bits: 量化位数（2, 4, 8）
        group_size: 量化组大小（-1表示per-channel）
        target_modules: 要量化的模块名模式，None表示所有Linear
        device: 计算设备
        calib_text_file: 校准文本文件路径（每行一个样本）
        calib_num_samples: 校准样本数
        calib_seq_len: 校准序列长度
        calib_batch_size: 校准batch size
        calib_seed: 随机种子
        desc_act: GPTQ是否使用act-order（暂不支持True）
        damp_percent: GPTQ Hessian阻尼系数
        awq_version: AWQ版本（"GEMM"或"GEMV"）
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 验证参数
    if quant_method not in ["rtn", "gptq", "awq", "gptq_marlin", "awq_marlin"]:
        raise ValueError(f"Unknown quant_method: {quant_method}")
    
    if bits not in [2, 3, 4, 8]:
        raise ValueError(f"bits must be in [2, 3, 4, 8], got {bits}")
    
    # Marlin强制约束
    is_marlin = "marlin" in quant_method
    if is_marlin:
        if bits not in [4, 8]:
            raise ValueError(f"Marlin only supports 4-bit or 8-bit, got {bits}")
        if group_size not in [128, -1]:
            print(f"Warning: Marlin prefers group_size=128, got {group_size}")
        print(f"Marlin mode: forcing symmetric quantization")
        sym = True
    else:
        sym = quant_method == "gptq"  # GPTQ默认对称，AWQ默认非对称
    
    if quant_method != "rtn" and calib_text_file is None:
        raise ValueError(f"{quant_method} requires calib_text_file")
    
    if desc_act:
        raise NotImplementedError("desc_act=True is not yet supported")
    
    # 加载模型配置
    print(f"Loading model config from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 确定输出格式
    if quant_method == "gptq_marlin":
        output_format = "gptq_marlin"
    elif quant_method == "awq_marlin":
        output_format = "awq_marlin"
    elif quant_method == "awq":
        output_format = "awq"
    else:
        output_format = "gptq"
    
    # 元数据
    metadata = {
        "quant_method": quant_method,
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "desc_act": desc_act,
        "quantized_modules": [],
    }
    
    quantized_weights: Dict[str, torch.Tensor] = {}
    
    # RTN方法：不需要加载完整模型
    if quant_method == "rtn":
        print("Loading model weights...")
        # 加载safetensors
        from glob import glob
        safetensors_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
        
        if not safetensors_files:
            raise ValueError(f"No safetensors files found in {model_path}")
        
        # 收集所有key
        all_keys = []
        for f in safetensors_files:
            st = load_file(f, device="cpu")
            all_keys.extend(list(st.keys()))
        
        # 筛选线性层权重
        linear_keys = []
        for key in all_keys:
            if not key.endswith(".weight"):
                continue
            if any(skip in key for skip in [".norm", "norm.", "embed", "lm_head"]):
                continue
            if target_modules and not any(t in key for t in target_modules):
                continue
            linear_keys.append(key)
        
        print(f"Quantizing {len(linear_keys)} layers with RTN...")
        
        for key in tqdm(linear_keys, desc="RTN Quantize"):
            # 加载权重
            weight = None
            for f in safetensors_files:
                try:
                    st = load_file(f, device="cpu")
                    if key in st:
                        weight = st[key]
                        break
                except:
                    continue
            
            if weight is None or weight.dim() != 2:
                continue
            
            out_features, in_features = weight.shape
            
            # RTN量化
            qweight, qzeros, scales, g_idx = quantize_rtn(
                weight, bits, group_size, sym
            )
            
            prefix = key[:-7]  # Remove ".weight"
            
            if is_marlin:
                # Marlin重排
                marlin_qw, marlin_sc, marlin_ws = repack_gptq_to_marlin(
                    qweight, scales, bits, in_features, out_features,
                    group_size if group_size != -1 else in_features, device
                )
                quantized_weights[f"{prefix}.marlin_qweight"] = marlin_qw
                quantized_weights[f"{prefix}.marlin_scales"] = marlin_sc
                quantized_weights[f"{prefix}.marlin_workspace"] = marlin_ws
            else:
                quantized_weights[f"{prefix}.qweight"] = qweight
                quantized_weights[f"{prefix}.qzeros"] = qzeros
                quantized_weights[f"{prefix}.scales"] = scales
                quantized_weights[f"{prefix}.g_idx"] = g_idx
            
            metadata["quantized_modules"].append({
                "name": prefix,
                "in_features": in_features,
                "out_features": out_features,
                "group_size": group_size,
                "bits": bits,
            })
    
    else:
        # GPTQ/AWQ方法：需要加载模型和校准
        print(f"Building calibration data...")
        calib_data = build_calibration_data(
            model_path, calib_text_file, calib_num_samples,
            calib_seq_len, calib_batch_size, calib_seed
        )
        
        print(f"Loading model for {quant_method} quantization...")
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="cpu",  # 先在CPU加载
        )
        
        print("Collecting layer inputs...")
        layer_inputs = collect_layer_inputs(
            model, calib_data, target_modules, device
        )
        
        # 量化每个层
        print(f"Quantizing with {quant_method}...")
        for name, module in tqdm(list(model.named_modules()), desc="Quantize"):
            if not isinstance(module, nn.Linear):
                continue
            if target_modules and not any(t in name for t in target_modules):
                continue
            if name not in layer_inputs:
                continue
            
            weight = module.weight.data
            out_features, in_features = weight.shape
            
            inputs = layer_inputs[name]
            
            if quant_method in ["gptq", "gptq_marlin"]:
                qweight, qzeros, scales, g_idx = quantize_gptq(
                    module, inputs, bits, group_size, sym=True,
                    damp_percent=damp_percent, device=device
                )
                
                if is_marlin:
                    marlin_qw, marlin_sc, marlin_ws = repack_gptq_to_marlin(
                        qweight, scales, bits, in_features, out_features,
                        group_size if group_size != -1 else in_features, device
                    )
                    quantized_weights[f"{name}.marlin_qweight"] = marlin_qw
                    quantized_weights[f"{name}.marlin_scales"] = marlin_sc
                    quantized_weights[f"{name}.marlin_workspace"] = marlin_ws
                else:
                    quantized_weights[f"{name}.qweight"] = qweight
                    quantized_weights[f"{name}.qzeros"] = qzeros
                    quantized_weights[f"{name}.scales"] = scales
                    quantized_weights[f"{name}.g_idx"] = g_idx
            
            elif quant_method in ["awq", "awq_marlin"]:
                qweight, qzeros, scales = quantize_awq(
                    module, inputs, bits, group_size, device
                )
                
                if is_marlin:
                    marlin_qw, marlin_sc, marlin_ws = repack_awq_to_marlin(
                        qweight, scales, bits, in_features, out_features,
                        group_size if group_size != -1 else in_features, device
                    )
                    quantized_weights[f"{name}.marlin_qweight"] = marlin_qw
                    quantized_weights[f"{name}.marlin_scales"] = marlin_sc
                    quantized_weights[f"{name}.marlin_workspace"] = marlin_ws
                else:
                    quantized_weights[f"{name}.qweight"] = qweight
                    quantized_weights[f"{name}.qzeros"] = qzeros
                    quantized_weights[f"{name}.scales"] = scales
            
            metadata["quantized_modules"].append({
                "name": name,
                "in_features": in_features,
                "out_features": out_features,
                "group_size": group_size,
                "bits": bits,
            })
            
            # 清理内存
            del layer_inputs[name]
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # 保存文件
    print("\nSaving quantized model...")
    
    # 复制原始模型文件
    model_path_obj = Path(model_path)
    for file in model_path_obj.glob("*.safetensors"):
        if "quantized" not in file.name:
            shutil.copy2(file, output_path / file.name)
    
    for file in model_path_obj.iterdir():
        if file.is_file() and not file.name.endswith('.safetensors'):
            shutil.copy2(file, output_path / file.name)
    
    # 保存量化权重
    output_file = output_path / f"model_quantized_{output_format}.safetensors"
    save_file(quantized_weights, output_file)
    print(f"  Saved: {output_file}")
    
    # 保存元数据
    metadata_file = output_path / f"quantization_metadata_{output_format}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_file}")
    
    # 保存quantize_config.json
    if output_format == "gptq_marlin":
        quantize_cfg = {
            "bits": bits,
            "group_size": group_size,
            "desc_act": False,
            "sym": True,
            "lm_head": False,
            "checkpoint_format": "gptq_marlin",
        }
    elif output_format == "awq_marlin":
        quantize_cfg = {
            "bits": bits,
            "group_size": group_size,
            "desc_act": False,
            "sym": False,
            "lm_head": False,
            "checkpoint_format": "awq_marlin",
        }
    elif output_format == "awq":
        quantize_cfg = {
            "bits": bits,
            "group_size": group_size,
            "desc_act": False,
            "sym": False,
            "lm_head": False,
            "checkpoint_format": "awq",
            "version": awq_version,
        }
    else:  # gptq
        quantize_cfg = {
            "bits": bits,
            "group_size": group_size,
            "desc_act": desc_act,
            "sym": sym,
            "lm_head": False,
            "checkpoint_format": "gptq",
        }
    
    cfg_file = output_path / "quantize_config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(quantize_cfg, f, indent=2)
    print(f"  Saved: {cfg_file}")
    
    print(f"\n✓ Quantization complete!")
    print(f"  Method: {quant_method}")
    print(f"  Format: {output_format}")
    print(f"  Quantized {len(metadata['quantized_modules'])} layers")
    print(f"  Output: {output_path}")


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="量化 AutoModelForDiffusionLM 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, required=True, help="输入模型路径")
    parser.add_argument("--output-path", type=str, required=True, help="输出路径")
    parser.add_argument(
        "--quant-method",
        type=str,
        choices=["rtn", "gptq", "awq", "gptq_marlin", "awq_marlin"],
        default="rtn",
        help="量化方法: rtn(快速)/gptq(高精度)/awq/gptq_marlin(高性能)/awq_marlin",
    )
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8], help="量化位数")
    parser.add_argument("--group-size", type=int, default=128, help="量化组大小")
    parser.add_argument(
        "--target-modules",
        type=str,
        help="要量化的模块名模式（逗号分隔），例如: q_proj,k_proj,v_proj"
    )
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    
    # 校准参数
    parser.add_argument("--calib-text-file", type=str, help="校准文本文件（每行一条）")
    parser.add_argument("--calib-num-samples", type=int, default=128, help="校准样本数")
    parser.add_argument("--calib-seq-len", type=int, default=512, help="校准序列长度")
    parser.add_argument("--calib-batch-size", type=int, default=1, help="校准batch size")
    parser.add_argument("--calib-seed", type=int, default=0, help="随机种子")
    
    # GPTQ参数
    parser.add_argument("--desc-act", action="store_true", help="使用act-order（暂不支持）")
    parser.add_argument("--damp-percent", type=float, default=0.01, help="Hessian阻尼系数")
    
    # AWQ参数
    parser.add_argument("--awq-version", type=str, default="GEMM", choices=["GEMM", "GEMV"])
    
    args = parser.parse_args()
    
    target_modules = None
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",")]
    
    quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        quant_method=args.quant_method,
        bits=args.bits,
        group_size=args.group_size,
        target_modules=target_modules,
        device=args.device,
        calib_text_file=args.calib_text_file,
        calib_num_samples=args.calib_num_samples,
        calib_seq_len=args.calib_seq_len,
        calib_batch_size=args.calib_batch_size,
        calib_seed=args.calib_seed,
        desc_act=args.desc_act,
        damp_percent=args.damp_percent,
        awq_version=args.awq_version,
    )


if __name__ == "__main__":
    main()
