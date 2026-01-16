from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from diffulex.utils.quantization.context import get_linear_strategy


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LoRAMixin:
    """Mixin class to add LoRA support to existing linear layers."""
    def __init_lora__(self, r: int = 0, lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        if r > 0:
            self.r = r
            self.lora_alpha = lora_alpha
            self.scaling = lora_alpha / r
            
            # Initialize LoRA parameters
            if hasattr(self, 'output_size_per_partition'):
                out_features = self.output_size_per_partition
            else:
                out_features = self.output_size
            
            if hasattr(self, 'input_size_per_partition'):
                in_features = self.input_size_per_partition
            else:
                in_features = self.input_size
            
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
            self.merged = False
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.r = 0
            self.merged = True
    
    def merge_lora(self):
        """Merge LoRA weights into base weight."""
        if not (hasattr(self, 'r') and self.r > 0 and not self.merged):
            return
        # If base weight is missing (e.g., quantized linear removed bf16 weight Parameter),
        # we cannot merge in-place. Keep LoRA unmerged and apply via lora_forward.
        weight = getattr(self, "weight", None)
        if weight is None or not hasattr(weight, "data"):
            return
        self.weight.data += self.scaling * torch.mm(self.lora_B, self.lora_A)
        self.merged = True
    
    def lora_forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply LoRA forward pass."""
        if not hasattr(self, 'r') or self.r == 0 or self.merged:
            return base_output
        
        lora_out = F.linear(self.lora_dropout(x), self.lora_A.T)
        lora_out = F.linear(lora_out, self.lora_B.T)
        return base_output + lora_out * self.scaling


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
        quant_kind: str = "other",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.quant_kind = (quant_kind or "other").strip().lower() or "other"
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # Quantized weight storage (W8A16 etc.). Empty by default.
        # NOTE: We keep these as buffers so they move with the module and do not appear as Parameters.
        self.register_buffer("quant_weight_int8", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("quant_scales", torch.empty(0, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("_weight_is_quantized", torch.tensor(False, dtype=torch.bool), persistent=False)
        
        # GPTQ/AWQ offline quantized weight storage (W4A16).
        # GPTQ: qweight (packed int4), qzeros (packed int4), scales (per-group), g_idx (optional)
        # AWQ: qweight (packed int4), qzeros (packed int4), scales (per-group)
        self.register_buffer("gptq_qweight", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("gptq_qzeros", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("gptq_scales", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("gptq_g_idx", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("awq_qweight", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("awq_qzeros", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("awq_scales", torch.empty(0, dtype=torch.float32), persistent=False)
        # Metadata for offline quantized weights
        self.register_buffer("_offline_quant_format", torch.empty(0, dtype=torch.int8), persistent=False)  # 0=none, 1=gptq, 2=awq
        self.register_buffer("_offline_quant_group_size", torch.tensor(128, dtype=torch.int32), persistent=False)
        self.register_buffer("_offline_quant_out_features", torch.tensor(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_offline_quant_in_features", torch.tensor(0, dtype=torch.int32), persistent=False)

    def has_quantized_weight(self) -> bool:
        return bool(self._weight_is_quantized.item()) and self.quant_weight_int8.numel() > 0 and self.quant_scales.numel() > 0

    def has_offline_quantized_weight(self) -> bool:
        """Check if offline quantized weights (GPTQ/AWQ) are present."""
        format_val = int(self._offline_quant_format.item()) if self._offline_quant_format.numel() > 0 else 0
        if format_val == 1:  # GPTQ
            return (
                self.gptq_qweight.numel() > 0
                and self.gptq_qzeros.numel() > 0
                and self.gptq_scales.numel() > 0
            )
        elif format_val == 2:  # AWQ
            return (
                self.awq_qweight.numel() > 0
                and self.awq_qzeros.numel() > 0
                and self.awq_scales.numel() > 0
            )
        return False

    def set_offline_quantized_weight(
        self,
        format: str,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        *,
        out_features: int,
        in_features: int,
        group_size: int = 128,
        g_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """Set offline quantized weights (GPTQ or AWQ format).

        Args:
            format: "gptq" or "awq"
            qweight: int8 packed int4 weights [out_features, (in_features + 1) // 2]
            qzeros: int8 packed int4 zeros [num_groups, (in_features + 1) // 2]
            scales: float32 per-group scales [num_groups, in_features] or [num_groups]
            out_features: Output features (N)
            in_features: Input features (K)
            group_size: Group size for quantization (default: 128)
            g_idx: Optional int32 tensor [out_features] for GPTQ group indices (GPTQ only)
        """
        format = format.strip().lower()
        if format not in ("gptq", "awq"):
            raise ValueError(f"Unsupported offline quant format: {format}. Supported: 'gptq', 'awq'")

        if qweight.dtype != torch.int8:
            raise TypeError(f"qweight must be int8, got {qweight.dtype}")
        if qzeros.dtype != torch.int8:
            raise TypeError(f"qzeros must be int8, got {qzeros.dtype}")
        if scales.dtype != torch.float32:
            scales = scales.to(dtype=torch.float32)

        num_groups = (out_features + group_size - 1) // group_size
        expected_qweight_shape = (out_features, (in_features + 1) // 2)
        expected_qzeros_shape = (num_groups, (in_features + 1) // 2)

        if qweight.shape != expected_qweight_shape:
            raise ValueError(
                f"qweight shape mismatch: got {qweight.shape}, expected {expected_qweight_shape}"
            )
        if qzeros.shape != expected_qzeros_shape:
            raise ValueError(
                f"qzeros shape mismatch: got {qzeros.shape}, expected {expected_qzeros_shape}"
            )

        if format == "gptq":
            self.gptq_qweight = qweight
            self.gptq_qzeros = qzeros
            self.gptq_scales = scales
            if g_idx is not None:
                if g_idx.shape != (out_features,):
                    raise ValueError(
                        f"g_idx shape mismatch: got {g_idx.shape}, expected ({out_features},)"
                    )
                if g_idx.dtype != torch.int32:
                    g_idx = g_idx.to(dtype=torch.int32)
                self.gptq_g_idx = g_idx
            else:
                # Clear g_idx if not provided
                self.gptq_g_idx = torch.empty(0, dtype=torch.int32)
            self._offline_quant_format = torch.tensor(1, dtype=torch.int8)
        else:  # AWQ
            self.awq_qweight = qweight
            self.awq_qzeros = qzeros
            self.awq_scales = scales
            # AWQ doesn't use g_idx, clear it
            self.gptq_qweight = torch.empty(0, dtype=torch.int8)
            self.gptq_qzeros = torch.empty(0, dtype=torch.int8)
            self.gptq_scales = torch.empty(0, dtype=torch.float32)
            self.gptq_g_idx = torch.empty(0, dtype=torch.int32)
            self._offline_quant_format = torch.tensor(2, dtype=torch.int8)

        self._offline_quant_group_size = torch.tensor(group_size, dtype=torch.int32)
        self._offline_quant_out_features = torch.tensor(out_features, dtype=torch.int32)
        self._offline_quant_in_features = torch.tensor(in_features, dtype=torch.int32)

        # Drop bf16 weight Parameter if present (to free memory)
        if "weight" in self._parameters:
            self._parameters.pop("weight", None)
            setattr(self, "weight", None)

    def set_quantized_weight(self, quant_weight_int8: torch.Tensor, quant_scales: torch.Tensor) -> None:
        # Support both int8 (for int8/int4 quantization) and uint8 (for FP8 quantization)
        if quant_weight_int8.dtype not in (torch.int8, torch.uint8):
            raise TypeError(f"quant_weight_int8 must be int8 or uint8, got {quant_weight_int8.dtype}")
        # Store scales dtype depends on strategy:
        # - W8A16/W4A16 kernels currently take bf16 scales.
        # - W8A8/W4A8 paths are more sensitive to scale precision; keep scales at fp16.
        # - FP8 W8A16 uses float32 scales.
        # - FP8 W8A8 uses float16 scales.
        try:
            strategy = get_linear_strategy(self.quant_kind)
        except Exception:
            strategy = None
        scale_dtype = torch.bfloat16
        if strategy is not None:
            weight_format = getattr(strategy, "linear_weight_format", None)
            act_format = getattr(strategy, "linear_act_format", None)
            # FP8 W8A16 uses float32 scales
            if weight_format in ("fp8_e4m3", "fp8_e5m2") and act_format == "bf16":
                scale_dtype = torch.float32
            # FP8 W8A8 and int8 W8A8 use float16 scales
            elif act_format in ("int8", "fp8_e4m3", "fp8_e5m2"):
                scale_dtype = torch.float16
        if quant_scales.dtype != scale_dtype:
            quant_scales = quant_scales.to(dtype=scale_dtype)
        self.quant_weight_int8 = quant_weight_int8
        self.quant_scales = quant_scales
        self._weight_is_quantized.fill_(True)

    def _maybe_quantize_loaded_weight_param(
        self,
        param: nn.Parameter,
        *,
        loaded_shard_id: object = None,
        expected_shard_ids: set[object] | None = None,
    ) -> None:
        """If current Linear is configured for quantization, quantize the loaded bf16 weight and drop the bf16 Parameter.

        This is called at the end of weight_loader(), after the shard copy is done.
        Supports int8 (W8A16/W8A8), int4 (W4A16/W4A8), and FP8 (FP8 W8A16/FP8 W8A8) quantization.
        """
        # Only process the real weight Parameter (ignore bias).
        current_weight = self._parameters.get("weight", None)
        if current_weight is None or current_weight is not param:
            return

        # Some modules load the same weight parameter in multiple shards (e.g., QKV / merged linears).
        # In that case, we must wait until all shards are loaded before quantizing/removing the bf16 Parameter,
        # otherwise subsequent shard loads would fail (model.get_parameter can't find it).
        if expected_shard_ids is not None:
            if not hasattr(self, "_loaded_weight_shard_ids"):
                self._loaded_weight_shard_ids: set[object] = set()
            self._loaded_weight_shard_ids.add(loaded_shard_id)
            if self._loaded_weight_shard_ids != expected_shard_ids:
                return

        # Get strategy for this kind; default bf16 strategy should not trigger quantization.
        strategy = get_linear_strategy(self.quant_kind)
        if strategy is None:
            return
        weight_format = getattr(strategy, "linear_weight_format", None)
        # NOTE: We intentionally do NOT require act_format == "bf16" here.
        # For W8A8/W4A8/FP8 W8A8 we still want to quantize+drop the bf16 weight Parameter at load-time.
        # But we must avoid doing this for the generic stub strategy (unsupported combos),
        # otherwise we'd drop weights and then raise NotImplementedError at runtime.
        if getattr(strategy, "name", "").startswith("linear_stub"):
            return
        
        # Support int8/int4/FP8 weight formats (W8A16/W8A8, W4A16/W4A8, FP8 W8A16/FP8 W8A8).
        if weight_format not in ("int8", "int4", "fp8_e4m3", "fp8_e5m2"):
            return

        # Quantize on the same device as the loaded param (typically CUDA).
        qweight, scales = strategy.quantize_weight_for_kernel(param.data, device=param.data.device)
        self.set_quantized_weight(qweight, scales)

        # Drop bf16 weight Parameter to free GPU memory.
        self._parameters.pop("weight", None)
        # Keep attribute for compatibility, but ensure forward uses quant buffers.
        setattr(self, "weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase, LoRAMixin):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        LinearBase.__init__(self, input_size, output_size, None, quant_kind)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        
        self.__init_lora__(r, lora_alpha, lora_dropout)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)
        self._maybe_quantize_loaded_weight_param(param, loaded_shard_id=None, expected_shard_ids={None})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        strategy = get_linear_strategy(self.quant_kind)
        
        # Check for offline quantized weights (GPTQ/AWQ) first
        if self.has_offline_quantized_weight():
            if strategy is None:
                raise RuntimeError("Offline quantized weight is present but no linear strategy is configured.")
            format_val = int(self._offline_quant_format.item())
            out_features = int(self._offline_quant_out_features.item())
            in_features = int(self._offline_quant_in_features.item())
            group_size = int(self._offline_quant_group_size.item())
            
            kwargs = {
                "out_features": out_features,
                "in_features": in_features,
                "group_size": group_size,
            }
            
            if format_val == 1:  # GPTQ
                kwargs.update({
                    "gptq_qweight": self.gptq_qweight,
                    "gptq_qzeros": self.gptq_qzeros,
                    "gptq_scales": self.gptq_scales,
                    "gptq_group_size": group_size,
                })
                if self.gptq_g_idx.numel() > 0:
                    kwargs["gptq_g_idx"] = self.gptq_g_idx
            elif format_val == 2:  # AWQ
                kwargs.update({
                    "awq_qweight": self.awq_qweight,
                    "awq_qzeros": self.awq_qzeros,
                    "awq_scales": self.awq_scales,
                    "awq_group_size": group_size,
                })
            
            base_out = strategy.linear_forward(
                x,
                None,  # weight not used for offline quantized weights
                self.bias,
                quant_kind=self.quant_kind,
                **kwargs,
            )
        elif self.has_quantized_weight():
            if strategy is None:
                raise RuntimeError("Quantized weight is present but no linear strategy is configured.")
            # For int4 (W4A16), we need to pass original_in_features
            weight_format = getattr(strategy, "linear_weight_format", None)
            kwargs = {"quant_scales": self.quant_scales}
            if weight_format == "int4":
                # For int4, packed weight shape is [out_features, (in_features + 1) // 2]
                # We use x.shape[1] as the source of truth (it's the actual K dimension)
                kwargs["original_in_features"] = x.shape[1]
            base_out = strategy.linear_forward(
                x,
                self.quant_weight_int8,
                self.bias,
                quant_kind=self.quant_kind,
                **kwargs,
            )
        elif strategy is None:
            base_out = F.linear(x, self.weight, self.bias)
        else:
            # For int4 strategies (W4A16/W4A8), we need to pass original_in_features even when weight is not quantized yet
            weight_format = getattr(strategy, "linear_weight_format", None)
            kwargs = {}
            if weight_format == "int4":
                kwargs["original_in_features"] = x.shape[1]
            base_out = strategy.linear_forward(x, self.weight, self.bias, quant_kind=self.quant_kind, **kwargs)
        return self.lora_forward(x, base_out)


class ColumnParallelLinear(LinearBase, LoRAMixin):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        LinearBase.__init__(self, input_size, output_size, 0, quant_kind)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        
        self.__init_lora__(r, lora_alpha, lora_dropout)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
        self._maybe_quantize_loaded_weight_param(param, loaded_shard_id=None, expected_shard_ids={None})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        strategy = get_linear_strategy(self.quant_kind)
        
        # Check for offline quantized weights (GPTQ/AWQ) first
        if self.has_offline_quantized_weight():
            if strategy is None:
                raise RuntimeError("Offline quantized weight is present but no linear strategy is configured.")
            format_val = int(self._offline_quant_format.item())
            out_features = int(self._offline_quant_out_features.item())
            in_features = int(self._offline_quant_in_features.item())
            group_size = int(self._offline_quant_group_size.item())
            
            kwargs = {
                "out_features": out_features,
                "in_features": in_features,
                "group_size": group_size,
            }
            
            if format_val == 1:  # GPTQ
                kwargs.update({
                    "gptq_qweight": self.gptq_qweight,
                    "gptq_qzeros": self.gptq_qzeros,
                    "gptq_scales": self.gptq_scales,
                    "gptq_group_size": group_size,
                })
                if self.gptq_g_idx.numel() > 0:
                    kwargs["gptq_g_idx"] = self.gptq_g_idx
            elif format_val == 2:  # AWQ
                kwargs.update({
                    "awq_qweight": self.awq_qweight,
                    "awq_qzeros": self.awq_qzeros,
                    "awq_scales": self.awq_scales,
                    "awq_group_size": group_size,
                })
            
            base_out = strategy.linear_forward(
                x,
                None,  # weight not used for offline quantized weights
                self.bias,
                quant_kind=self.quant_kind,
                **kwargs,
            )
        elif self.has_quantized_weight():
            if strategy is None:
                raise RuntimeError("Quantized weight is present but no linear strategy is configured.")
            # For int4 (W4A16), we need to pass original_in_features
            weight_format = getattr(strategy, "linear_weight_format", None)
            kwargs = {"quant_scales": self.quant_scales}
            if weight_format == "int4":
                # For int4, packed weight shape is [out_features, (in_features + 1) // 2]
                # We use x.shape[1] as the source of truth (it's the actual K dimension)
                kwargs["original_in_features"] = x.shape[1]
            base_out = strategy.linear_forward(
                x,
                self.quant_weight_int8,
                self.bias,
                quant_kind=self.quant_kind,
                **kwargs,
            )
        elif strategy is None:
            base_out = F.linear(x, self.weight, self.bias)
        else:
            # For int4 strategies (W4A16/W4A8), we need to pass original_in_features even when weight is not quantized yet
            weight_format = getattr(strategy, "linear_weight_format", None)
            kwargs = {}
            if weight_format == "int4":
                kwargs["original_in_features"] = x.shape[1]
            base_out = strategy.linear_forward(x, self.weight, self.bias, quant_kind=self.quant_kind, **kwargs)
        return self.lora_forward(x, base_out)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            sum(output_sizes),
            bias=bias,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quant_kind=quant_kind,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
        expected = set(range(len(self.output_sizes)))
        self._maybe_quantize_loaded_weight_param(param, loaded_shard_id=loaded_shard_id, expected_shard_ids=expected)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "attn",
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias, r, lora_alpha, lora_dropout, quant_kind=quant_kind)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
        self._maybe_quantize_loaded_weight_param(param, loaded_shard_id=loaded_shard_id, expected_shard_ids={"q", "k", "v"})


class RowParallelLinear(LinearBase, LoRAMixin):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        quant_kind: str = "other",
    ):
        LinearBase.__init__(self, input_size, output_size, 1, quant_kind)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        
        self.__init_lora__(r, lora_alpha, lora_dropout)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
        self._maybe_quantize_loaded_weight_param(param, loaded_shard_id=None, expected_shard_ids={None})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        strategy = get_linear_strategy(self.quant_kind)
        
        # Check for offline quantized weights (GPTQ/AWQ) first
        if self.has_offline_quantized_weight():
            if strategy is None:
                raise RuntimeError("Offline quantized weight is present but no linear strategy is configured.")
            format_val = int(self._offline_quant_format.item())
            out_features = int(self._offline_quant_out_features.item())
            in_features = int(self._offline_quant_in_features.item())
            group_size = int(self._offline_quant_group_size.item())
            
            kwargs = {
                "out_features": out_features,
                "in_features": in_features,
                "group_size": group_size,
            }
            
            if format_val == 1:  # GPTQ
                kwargs.update({
                    "gptq_qweight": self.gptq_qweight,
                    "gptq_qzeros": self.gptq_qzeros,
                    "gptq_scales": self.gptq_scales,
                    "gptq_group_size": group_size,
                })
                if self.gptq_g_idx.numel() > 0:
                    kwargs["gptq_g_idx"] = self.gptq_g_idx
            elif format_val == 2:  # AWQ
                kwargs.update({
                    "awq_qweight": self.awq_qweight,
                    "awq_qzeros": self.awq_qzeros,
                    "awq_scales": self.awq_scales,
                    "awq_group_size": group_size,
                })
            
            y = strategy.linear_forward(
                x,
                None,  # weight not used for offline quantized weights
                bias,
                quant_kind=self.quant_kind,
                **kwargs,
            )
        elif self.has_quantized_weight():
            if strategy is None:
                raise RuntimeError("Quantized weight is present but no linear strategy is configured.")
            # For int4 (W4A16), we must pass original_in_features to disambiguate packed K.
            weight_format = getattr(strategy, "linear_weight_format", None)
            kwargs = {"quant_scales": self.quant_scales}
            if weight_format == "int4":
                # Use activation K as the source of truth (it's the actual K dimension).
                kwargs["original_in_features"] = x.shape[1]
            y = strategy.linear_forward(
                x,
                self.quant_weight_int8,
                bias,
                quant_kind=self.quant_kind,
                **kwargs,
            )
        elif strategy is None:
            y = F.linear(x, self.weight, bias)
        else:
            # For int4 strategies (W4A16/W4A8), we need to pass original_in_features even when weight is not quantized yet
            weight_format = getattr(strategy, "linear_weight_format", None)
            kwargs = {}
            if weight_format == "int4":
                kwargs["original_in_features"] = x.shape[1]
            y = strategy.linear_forward(x, self.weight, bias, quant_kind=self.quant_kind, **kwargs)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return self.lora_forward(x, y)
