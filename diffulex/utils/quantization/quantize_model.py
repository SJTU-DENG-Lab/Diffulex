#!/usr/bin/env python3
"""离线量化脚本：将模型权重量化为 vLLM 标准 GPTQ/AWQ 格式

支持两种量化格式（对齐 vLLM 权重格式）：
- GPTQ: qweight/qzeros 为 int32 packed，scales 为 fp16，g_idx 可选（常见 desc_act=False 时为空）
- GPTQ_MARLIN: 导出 Marlin-ready 的 GPTQ 权重布局（qweight 已 repack，scales 已 permute，zp 为空）
- AWQ : qweight/qzeros 为 int32 packed，scales 为 fp16

使用方法:
    python -m diffulex.utils.quantization.quantize_model \
        --model-path /path/to/model \
        --output-path /path/to/output \
        --quant-format gptq_marlin \
        --group-size 128 \
        --bits 4 \
        --quant-method auto \
        --calib-text-file /path/to/calib.txt \
        --calib-num-samples 128 \
        --calib-seq-len 512

说明:
- `quant-method=simple`：沿用当前“直接分组量化/舍入”的旧实现（不需要校准数据，不是真 GPTQ/AWQ）。
- `quant-method=auto`：使用 `auto-gptq` / `awq(autoawq)` 做真正的校准量化，然后导出为 vLLM/Diffulex 可加载的权重格式。
"""

from __future__ import annotations

import argparse
import os
import json
import random
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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from glob import glob


def _require_vllm():
    try:
        from vllm.scalar_type import scalar_types  # type: ignore
        from vllm.model_executor.layers.quantization.utils.quant_utils import (  # type: ignore
            awq_pack,
            gptq_pack,
            pack_cols,
            quantize_weights,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "离线 GPTQ/AWQ 打包已切换到 vLLM 标准格式，需要可 import 的 vLLM。"
        ) from e
    return scalar_types, quantize_weights, gptq_pack, awq_pack, pack_cols


def _require_vllm_marlin():
    # Marlin 预处理依赖 CUDA custom ops
    try:
        from vllm import _custom_ops as ops  # type: ignore
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # type: ignore
            marlin_permute_scales,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "导出 gptq_marlin 格式需要可 import 的 vLLM Marlin（含 CUDA custom ops）。"
        ) from e
    return ops, marlin_permute_scales


def _require_auto_gptq():
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未能导入 auto-gptq。请确认已在当前 .venv 安装（例如：BUILD_CUDA_EXT=0 pip install auto-gptq）。"
        ) from e
    return AutoGPTQForCausalLM, BaseQuantizeConfig


def _require_awq():
    try:
        from awq import AutoAWQForCausalLM  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未能导入 awq（autoawq 的导入名是 `awq`）。"
        ) from e
    return AutoAWQForCausalLM


def _load_calib_texts(
    calib_text_file: str, *, num_samples: int, seed: int
) -> list[str]:
    p = Path(calib_text_file)
    if not p.exists():
        raise FileNotFoundError(f"calib_text_file 不存在: {calib_text_file}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"calib_text_file 为空: {calib_text_file}")
    if num_samples <= 0:
        raise ValueError(f"calib_num_samples 必须 > 0, got {num_samples}")
    if len(lines) <= num_samples:
        return lines[:num_samples]
    rng = random.Random(seed)
    return rng.sample(lines, k=num_samples)


def _build_autogptq_examples(
    tokenizer, texts: list[str], *, seq_len: int
) -> list[dict[str, torch.Tensor]]:
    if seq_len <= 0:
        raise ValueError(f"calib_seq_len 必须 > 0, got {seq_len}")

    # AutoGPTQ 会自行 collate/pad；这里用 fixed max_length 保持输入一致。
    examples: list[dict[str, torch.Tensor]] = []
    for t in texts:
        enc = tokenizer(
            t,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        examples.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc.get("attention_mask", torch.ones_like(enc["input_ids"])),
            }
        )
    return examples


def _quantize_to_vllm_gptq(
    weight: torch.Tensor, *, group_size: int, bits: int, use_v2_format: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize and pack weights into vLLM GPTQ checkpoint format.

    Input:
      weight: fp32 [N, K] (PyTorch Linear weight)
    Output (vLLM format):
      qweight: int32 [K/pack, N]
      qzeros : int32 [K/group, N/pack]   (GPTQ v1 stores (zeros - 1); v2 stores zeros)
      scales : fp16  [K/group, N]
      g_idx  : int32 empty tensor (desc_act=False)
    """
    scalar_types, quantize_weights, gptq_pack, _, pack_cols = _require_vllm()
    # vLLM GPTQConfig mentions 2/3/4/8, but the standard vLLM int32 packing
    # used by `gptq_pack/pack_cols` requires 32 % bits == 0.
    # So we support 2/4/8 here; 3-bit would need a different packing scheme.
    if bits not in (2, 4, 8):
        raise ValueError(
            f"GPTQ bits 仅支持 2/4/8（vLLM 标准 int32 pack 要求 32%bits==0），当前 bits={bits}"
        )

    # vLLM operates on (K, N)
    w = weight.T.contiguous()
    size_k, size_n = w.shape
    group_size_norm = size_k if group_size == -1 else group_size
    if group_size_norm <= 0 or size_k % group_size_norm != 0:
        raise ValueError(f"Invalid group_size={group_size} for in_features={size_k}")

    if bits == 2:
        quant_type = scalar_types.uint2b2
    elif bits == 4:
        quant_type = scalar_types.uint4b8
    else:  # bits == 8
        quant_type = scalar_types.uint8b128

    _, w_q, w_s, _ = quantize_weights(w, quant_type, group_size_norm, zero_points=False)

    pack_factor = 32 // bits
    qweight = gptq_pack(w_q, bits, size_k, size_n).contiguous()  # [K/pack, N]

    num_groups = size_k // group_size_norm
    zeros = torch.full(
        (num_groups, size_n),
        int(getattr(quant_type, "bias", 0)),
        dtype=torch.int32,
        device=w.device,
    )
    # GPTQ v1 stores zeros-1 in the checkpoint.
    zeros_to_store = zeros if use_v2_format else (zeros - 1)
    qzeros = pack_cols(zeros_to_store, bits, num_groups, size_n).contiguous()  # [K/group, N/pack]

    scales = w_s.to(torch.float16).contiguous()  # [K/group, N]
    g_idx = torch.empty((0,), dtype=torch.int32, device=w.device)
    return qweight, qzeros, scales, g_idx


def _quantize_to_vllm_gptq_marlin(
    weight: torch.Tensor, *, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weights and export marlin-ready GPTQ layout.

    该导出格式对齐 vLLM `MarlinLinearKernel.process_weights_after_loading` 的结果：
    - qweight: 已执行 `gptq_marlin_repack`
    - scales : 已执行 `marlin_permute_scales`
    - qzeros : 置空（Marlin GPTQ symmetric 路径不使用 runtime zp）
    - g_idx  : 空（desc_act=False）

    注意：需要在 CUDA 上执行（`gptq_marlin_repack` 为 CUDA op）。
    """
    if weight.device.type != "cuda":
        raise ValueError("gptq_marlin 导出需要 device=cuda（Marlin repack 为 CUDA op）")

    ops, marlin_permute_scales = _require_vllm_marlin()

    # 先按 vLLM 标准 GPTQ（symmetric, zero_points=False）量化并打包
    qweight, _qzeros, scales, g_idx = _quantize_to_vllm_gptq(
        weight, group_size=group_size, bits=bits, use_v2_format=False
    )

    # vLLM GPTQ packing 的 shape 基于 w=(K,N)；这里 size_k=in_features, size_n=out_features
    size_k = weight.shape[1]
    size_n = weight.shape[0]
    group_size_norm = size_k if group_size == -1 else group_size

    # desc_act=False 时 perm 为空
    empty_perm = torch.empty((0,), dtype=torch.int32, device=weight.device)

    marlin_qweight = ops.gptq_marlin_repack(
        qweight.contiguous(),
        perm=empty_perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=bits,
        is_a_8bit=False,
    ).contiguous()

    marlin_scales = marlin_permute_scales(
        scales.contiguous(),
        size_k=size_k,
        size_n=size_n,
        group_size=group_size_norm,
        is_a_8bit=False,
    ).contiguous()

    # Marlin GPTQ symmetric 不使用 runtime zero points，导出空 qzeros 保持一致性
    marlin_qzeros = torch.empty((0,), dtype=torch.int32, device=weight.device)
    marlin_g_idx = g_idx  # already empty

    return marlin_qweight, marlin_qzeros, marlin_scales, marlin_g_idx


def _quantize_to_vllm_awq(
    weight: torch.Tensor, *, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize and pack weights into vLLM AWQ checkpoint format.

    Input:
      weight: fp32 [N, K]
    Output (vLLM format):
      qweight: int32 [K, N/pack]
      qzeros : int32 [K/group, N/pack]
      scales : fp16  [K/group, N]
    """
    scalar_types, quantize_weights, _, awq_pack, _ = _require_vllm()
    if bits != 4:
        raise ValueError(f"AWQ 目前仅支持 4-bit，当前 bits={bits}")

    w = weight.T.contiguous()
    size_k, size_n = w.shape
    group_size_norm = size_k if group_size == -1 else group_size
    if group_size_norm <= 0 or size_k % group_size_norm != 0:
        raise ValueError(f"Invalid group_size={group_size} for in_features={size_k}")

    quant_type = scalar_types.uint4
    _, w_q, w_s, w_zp = quantize_weights(w, quant_type, group_size_norm, zero_points=True)
    if w_zp is None:
        raise RuntimeError("AWQ zero_points=True 但未生成 zero points，vLLM 量化返回异常。")

    qweight = awq_pack(w_q, bits, size_k, size_n).contiguous()  # [K, N/pack]
    num_groups = size_k // group_size_norm
    qzeros = awq_pack(w_zp.to(torch.int32), bits, num_groups, size_n).contiguous()  # [K/group, N/pack]
    scales = w_s.to(torch.float16).contiguous()  # [K/group, N]
    return qweight, qzeros, scales


@torch.inference_mode()
def _export_autogptq_to_vllm_weights(
    *,
    gptq_base_model: nn.Module,
    quant_format: str,
    target_modules: Optional[list[str]],
    desc_act: bool,
    bits: int,
    group_size: int,
) -> dict[str, torch.Tensor]:
    """
    从 auto-gptq 的量化后模型中抽取 qweight/qzeros/scales/g_idx，并按 vLLM/Diffulex 的命名导出。
    - quant_format == "gptq": 直接导出 QuantLinear 的 buffers。
    - quant_format == "gptq_marlin": 在导出前使用 vLLM Marlin 的 repack/permute，且导出空 qzeros/g_idx。
    """
    quantized_weights: dict[str, torch.Tensor] = {}

    if quant_format not in ("gptq", "gptq_marlin"):
        raise ValueError(f"Unexpected quant_format for auto-gptq export: {quant_format}")

    if quant_format == "gptq_marlin":
        if not torch.cuda.is_available():
            raise RuntimeError("导出 gptq_marlin 需要 CUDA（vLLM Marlin repack 为 CUDA op）。")
        ops, marlin_permute_scales = _require_vllm_marlin()

    for module_name, module in gptq_base_model.named_modules():
        # AutoGPTQ 的 QuantLinear（triton/cuda）会有这些 buffer
        if not (hasattr(module, "qweight") and hasattr(module, "qzeros") and hasattr(module, "scales")):
            continue

        # 过滤：保持和旧脚本一致，默认不量化 lm_head
        if "lm_head" in module_name:
            continue
        if target_modules and not any(t in module_name for t in target_modules):
            continue

        qweight = getattr(module, "qweight")
        qzeros = getattr(module, "qzeros")
        scales = getattr(module, "scales")
        g_idx = getattr(module, "g_idx", None)

        if not isinstance(qweight, torch.Tensor) or not isinstance(qzeros, torch.Tensor) or not isinstance(scales, torch.Tensor):
            continue

        if quant_format == "gptq":
            quantized_weights[f"{module_name}.qweight"] = qweight.detach().cpu().contiguous()
            quantized_weights[f"{module_name}.qzeros"] = qzeros.detach().cpu().contiguous()
            quantized_weights[f"{module_name}.scales"] = scales.detach().cpu().contiguous()
            if desc_act and isinstance(g_idx, torch.Tensor) and g_idx.numel() > 0:
                quantized_weights[f"{module_name}.g_idx"] = g_idx.detach().to(dtype=torch.int32).cpu().contiguous()
            else:
                quantized_weights[f"{module_name}.g_idx"] = torch.empty((0,), dtype=torch.int32)
            continue

        # gptq_marlin 导出：用 vLLM 的 repack/permute 变成 Marlin-ready layout
        in_features = int(getattr(module, "infeatures", 0))
        out_features = int(getattr(module, "outfeatures", 0))
        if in_features <= 0 or out_features <= 0:
            # fallback：从张量形状推断（qweight shape: [K/pack, N]）
            out_features = int(qweight.shape[1])
            pack = 32 // bits
            in_features = int(qweight.shape[0] * pack)

        group_size_norm = in_features if group_size == -1 else group_size
        empty_perm = torch.empty((0,), dtype=torch.int32, device="cuda")

        qweight_cuda = qweight.contiguous().to(device="cuda")
        scales_cuda = scales.contiguous().to(device="cuda", dtype=torch.float16)

        marlin_qweight = ops.gptq_marlin_repack(
            qweight_cuda,
            perm=empty_perm,
            size_k=in_features,
            size_n=out_features,
            num_bits=bits,
            is_a_8bit=(bits == 8),
        ).contiguous()
        marlin_scales = marlin_permute_scales(
            scales_cuda,
            size_k=in_features,
            size_n=out_features,
            group_size=group_size_norm,
            is_a_8bit=(bits == 8),
        ).contiguous()

        quantized_weights[f"{module_name}.qweight"] = marlin_qweight.detach().cpu().contiguous()
        quantized_weights[f"{module_name}.qzeros"] = torch.empty((0,), dtype=torch.int32)
        quantized_weights[f"{module_name}.scales"] = marlin_scales.detach().cpu().contiguous()
        quantized_weights[f"{module_name}.g_idx"] = torch.empty((0,), dtype=torch.int32)

    return quantized_weights


@torch.inference_mode()
def _export_awq_to_vllm_weights(
    *,
    awq_base_model: nn.Module,
    target_modules: Optional[list[str]],
) -> dict[str, torch.Tensor]:
    """
    从 awq(pack 后)模型中抽取 qweight/qzeros/scales，并按 vLLM/Diffulex 的命名导出。
    """
    quantized_weights: dict[str, torch.Tensor] = {}
    for module_name, module in awq_base_model.named_modules():
        if not (hasattr(module, "qweight") and hasattr(module, "qzeros") and hasattr(module, "scales")):
            continue
        if "lm_head" in module_name:
            continue
        if target_modules and not any(t in module_name for t in target_modules):
            continue

        qweight = getattr(module, "qweight")
        qzeros = getattr(module, "qzeros")
        scales = getattr(module, "scales")
        if not isinstance(qweight, torch.Tensor) or not isinstance(qzeros, torch.Tensor) or not isinstance(scales, torch.Tensor):
            continue

        quantized_weights[f"{module_name}.qweight"] = qweight.detach().cpu().contiguous()
        quantized_weights[f"{module_name}.qzeros"] = qzeros.detach().cpu().contiguous()
        quantized_weights[f"{module_name}.scales"] = scales.detach().cpu().contiguous()
    return quantized_weights


def quantize_model(
    model_path: str,
    output_path: str,
    quant_format: str = "gptq",
    group_size: int = 128,
    bits: int = 4,
    target_modules: Optional[list[str]] = None,
    device: str = "cpu",
    quant_method: str = "auto",
    calib_text_file: Optional[str] = None,
    calib_num_samples: int = 128,
    calib_seq_len: int = 512,
    calib_batch_size: int = 1,
    calib_seed: int = 0,
    # GPTQ config
    desc_act: bool = False,
    sym: bool = True,
    damp_percent: float = 0.01,
    true_sequential: bool = True,
    use_triton: bool = True,
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
        quant_method: "auto"（真 GPTQ/AWQ，需校准数据）或 "simple"（旧实现，无校准）
        calib_text_file: 校准文本文件（每行一条样本）
    """
    if quant_format not in ["gptq", "gptq_marlin", "awq"]:
        raise ValueError(
            f"Unsupported quant_format: {quant_format}. Must be 'gptq', 'gptq_marlin' or 'awq'"
        )
    if quant_method not in ["auto", "simple"]:
        raise ValueError("quant_method must be 'auto' or 'simple'")

    # Marlin GPTQ 强约束：对称量化 + 不使用 act-order
    if quant_format == "gptq_marlin":
        desc_act = False
        sym = True
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model config (for tokenizer special tokens, etc.)
    _ = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    quantized_weights: dict[str, torch.Tensor] = {}
    metadata = {
        "quant_format": quant_format,
        "quant_method": quant_method,
        "group_size": group_size,
        "bits": bits,
        "quantized_modules": [],
    }

    # ----------------------------
    # 真 GPTQ/AWQ（需要校准数据）
    # ----------------------------
    if quant_method == "auto":
        if calib_text_file is None:
            raise ValueError("quant_method=auto 需要提供 --calib-text-file")

        texts = _load_calib_texts(calib_text_file, num_samples=calib_num_samples, seed=calib_seed)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if quant_format in ("gptq", "gptq_marlin"):
            if quant_format == "gptq_marlin" and device != "cuda":
                raise ValueError("导出 gptq_marlin 需要 --device cuda")

            AutoGPTQForCausalLM, BaseQuantizeConfig = _require_auto_gptq()
            examples = _build_autogptq_examples(tokenizer, texts, seq_len=calib_seq_len)

            qcfg = BaseQuantizeConfig(
                bits=int(bits),
                group_size=int(group_size),
                damp_percent=float(damp_percent),
                desc_act=bool(desc_act),
                sym=bool(sym),
                true_sequential=bool(true_sequential),
            )

            model_init_kwargs = {
                "trust_remote_code": True,
            }
            # 让 AutoGPTQ 自己用 accelerate 做 device_map；CPU 模式下走默认加载。
            if device == "cuda":
                model_init_kwargs["device_map"] = "auto"
                model_init_kwargs["torch_dtype"] = torch.float16

            gptq_model = AutoGPTQForCausalLM.from_pretrained(
                model_path,
                qcfg,
                **model_init_kwargs,
            )
            gptq_model.quantize(
                examples,
                batch_size=int(calib_batch_size),
                use_triton=bool(use_triton),
                cache_examples_on_gpu=(device == "cuda"),
            )

            quantized_weights = _export_autogptq_to_vllm_weights(
                gptq_base_model=gptq_model.model,
                quant_format=quant_format,
                target_modules=target_modules,
                desc_act=bool(desc_act),
                bits=int(bits),
                group_size=int(group_size),
            )

        else:  # awq
            if bits != 4:
                raise ValueError(f"AWQ 目前仅支持 4-bit，当前 bits={bits}")
            AutoAWQForCausalLM = _require_awq()

            awq_model = AutoAWQForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                safetensors=True,
                device_map="auto" if device == "cuda" else None,
                torch_dtype="auto",
            )

            awq_model.quantize(
                tokenizer=tokenizer,
                quant_config={
                    "zero_point": True,
                    "q_group_size": int(group_size),
                    "w_bit": int(bits),
                    "version": "GEMM",
                },
                calib_data=texts,
                max_calib_samples=int(calib_num_samples),
                max_calib_seq_len=int(calib_seq_len),
            )
            awq_model.pack()

            quantized_weights = _export_awq_to_vllm_weights(
                awq_base_model=awq_model.model,
                target_modules=target_modules,
            )

    # ----------------------------
    # 旧实现（无校准，不是真 GPTQ/AWQ）
    # ----------------------------
    else:
        safetensors_files = list(glob(os.path.join(model_path, "*.safetensors")))
        if not safetensors_files:
            raise ValueError(f"No safetensors files found in {model_path}")

        print(f"Found {len(safetensors_files)} safetensors files")

        all_weight_keys: list[str] = []
        for file in safetensors_files:
            with safe_open(file, "pt", device) as f:
                all_weight_keys.extend(f.keys())

        linear_weight_keys: list[str] = []
        for key in all_weight_keys:
            if any(skip in key for skip in [".bias", ".norm", ".embed", ".lm_head"]):
                continue
            if not key.endswith(".weight"):
                continue
            if target_modules and not any(target in key for target in target_modules):
                continue
            linear_weight_keys.append(key)

        print(f"Found {len(linear_weight_keys)} linear layer weights to quantize")

        for key in tqdm(linear_weight_keys, desc="Quantizing weights (simple)"):
            weight = None
            for file in safetensors_files:
                with safe_open(file, "pt", device) as f:
                    if key in f.keys():
                        weight = f.get_tensor(key)
                        break

            if weight is None:
                print(f"Warning: Could not load weight for {key}")
                continue
            if weight.dim() != 2:
                print(f"Skipping {key}: not a 2D weight (shape: {weight.shape})")
                continue

            out_features, in_features = weight.shape
            weight_fp32 = weight.to(torch.float32).to(device)
            prefix = key[:-7]  # Remove ".weight"

            if quant_format == "gptq":
                qweight, qzeros, scales, g_idx = _quantize_to_vllm_gptq(
                    weight_fp32, group_size=group_size, bits=bits, use_v2_format=False
                )
                quantized_weights[f"{prefix}.qweight"] = qweight.cpu()
                quantized_weights[f"{prefix}.qzeros"] = qzeros.cpu()
                quantized_weights[f"{prefix}.scales"] = scales.cpu()
                quantized_weights[f"{prefix}.g_idx"] = g_idx.cpu()

            elif quant_format == "gptq_marlin":
                qweight, qzeros, scales, g_idx = _quantize_to_vllm_gptq_marlin(
                    weight_fp32, group_size=group_size, bits=bits
                )
                quantized_weights[f"{prefix}.qweight"] = qweight.cpu()
                quantized_weights[f"{prefix}.qzeros"] = qzeros.cpu()
                quantized_weights[f"{prefix}.scales"] = scales.cpu()
                quantized_weights[f"{prefix}.g_idx"] = g_idx.cpu()

            else:  # awq
                qweight, qzeros, scales = _quantize_to_vllm_awq(
                    weight_fp32, group_size=group_size, bits=bits
                )
                quantized_weights[f"{prefix}.qweight"] = qweight.cpu()
                quantized_weights[f"{prefix}.qzeros"] = qzeros.cpu()
                quantized_weights[f"{prefix}.scales"] = scales.cpu()

            metadata["quantized_modules"].append(
                {
                    "name": prefix,
                    "out_features": int(out_features),
                    "in_features": int(in_features),
                    "group_size": group_size,
                    "bits": bits,
                }
            )

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

    # vLLM/Diffulex 会读取 quantize_config.json 识别量化类型与超参
    if quant_format in ("gptq", "gptq_marlin", "awq"):
        if quant_format == "gptq_marlin":
            cfg_desc_act = False
            cfg_sym = True
            cfg_ckpt = "gptq_marlin"
        elif quant_format == "gptq":
            cfg_desc_act = bool(desc_act)
            cfg_sym = bool(sym)
            cfg_ckpt = "gptq"
        else:  # awq
            cfg_desc_act = False
            cfg_sym = False
            cfg_ckpt = "awq"

        quantize_cfg = {
            "bits": int(bits),
            "group_size": int(group_size),
            "desc_act": bool(cfg_desc_act),
            "sym": bool(cfg_sym),
            "lm_head": False,
            "checkpoint_format": cfg_ckpt,
        }
        with open(output_path / "quantize_config.json", "w", encoding="utf-8") as f:
            json.dump(quantize_cfg, f, indent=2)
    
    print(f"\n✓ Quantization complete!")
    print(f"  - Quant method: {quant_method}")
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
    parser.add_argument(
        "--quant-format",
        type=str,
        choices=["gptq", "gptq_marlin", "awq"],
        default="gptq",
        help="量化格式: gptq / gptq_marlin / awq",
    )
    parser.add_argument("--group-size", type=int, default=128, help="量化组大小 (默认: 128)")
    parser.add_argument("--bits", type=int, default=4, help="每个权重的位数 (默认: 4)")
    parser.add_argument("--target-modules", type=str, help="要量化的模块名称模式（逗号分隔），例如: q_proj,k_proj,v_proj")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="量化设备 (默认: cpu)")
    parser.add_argument(
        "--quant-method",
        type=str,
        choices=["auto", "simple"],
        default="auto",
        help="量化方法: auto(真 GPTQ/AWQ, 需要校准数据) / simple(旧实现, 无校准)",
    )
    parser.add_argument("--calib-text-file", type=str, default=None, help="校准文本文件（每行一条样本）")
    parser.add_argument("--calib-num-samples", type=int, default=128, help="校准样本数 (默认: 128)")
    parser.add_argument("--calib-seq-len", type=int, default=512, help="校准序列长度 (默认: 512)")
    parser.add_argument("--calib-batch-size", type=int, default=1, help="校准 batch size (默认: 1)")
    parser.add_argument("--calib-seed", type=int, default=0, help="校准采样随机种子 (默认: 0)")
    parser.add_argument("--desc-act", action="store_true", help="GPTQ act-order(desc_act) (默认: False)")
    parser.add_argument("--sym", dest="sym", action="store_true", default=True, help="GPTQ symmetric quant (默认: True)")
    parser.add_argument("--no-sym", dest="sym", action="store_false", help="关闭 GPTQ symmetric quant")
    parser.add_argument("--damp-percent", type=float, default=0.01, help="GPTQ damp_percent (默认: 0.01)")
    parser.add_argument(
        "--true-sequential",
        dest="true_sequential",
        action="store_true",
        default=True,
        help="GPTQ true_sequential (默认: True)",
    )
    parser.add_argument(
        "--no-true-sequential",
        dest="true_sequential",
        action="store_false",
        help="关闭 GPTQ true_sequential",
    )
    parser.add_argument(
        "--use-triton",
        dest="use_triton",
        action="store_true",
        default=True,
        help="AutoGPTQ 使用 Triton backend (默认: True)",
    )
    parser.add_argument(
        "--no-triton",
        dest="use_triton",
        action="store_false",
        help="关闭 AutoGPTQ Triton backend（可能回退到 CUDA extension）",
    )
    
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
        quant_method=args.quant_method,
        calib_text_file=args.calib_text_file,
        calib_num_samples=args.calib_num_samples,
        calib_seq_len=args.calib_seq_len,
        calib_batch_size=args.calib_batch_size,
        calib_seed=args.calib_seed,
        desc_act=bool(args.desc_act),
        sym=bool(args.sym),
        damp_percent=float(args.damp_percent),
        true_sequential=bool(args.true_sequential),
        use_triton=bool(args.use_triton),
    )


if __name__ == "__main__":
    main()
