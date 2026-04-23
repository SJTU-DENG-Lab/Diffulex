import os
import json
import torch
import torch.nn as nn

from tqdm import tqdm
from glob import glob
from functools import partial
from safetensors import safe_open
from diffulex.config import Config
from diffulex.logger import get_logger
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight

logger = get_logger(__name__)


def load_lora_config(lora_path: str) -> dict:
    """Load LoRA configuration from adapter_config.json."""
    config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def enable_lora_for_model(
    model: nn.Module,
    lora_config: dict,
    packed_modules_mapping: dict | None = None,
):
    """Enable LoRA for existing linear layers in the model.

    `target_modules` from PEFT adapter_config refers to the *checkpoint* leaf
    names (e.g. `attn_out`). When the local model class re-names a layer (e.g.
    LLaDA's `attn_out` is implemented as `self_attn.o_proj`), the mapping is
    declared in `packed_modules_mapping` as `{ckpt_leaf: (local_dotted_name, _)}`.
    We must consult that mapping here, otherwise renamed targets silently miss
    `__init_lora__` and the loaded LoRA tensors get dropped at apply time.
    """
    r = lora_config.get("r", 16)
    lora_alpha = lora_config.get("lora_alpha", 32.0)
    lora_dropout = lora_config.get("lora_dropout", 0.0)
    target_modules = lora_config.get("target_modules", [])
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    rev_mapping = {}
    if packed_modules_mapping:
        for ckpt_leaf, (local_dotted, _) in packed_modules_mapping.items():
            local_leaf = local_dotted.split(".")[-1]
            rev_mapping[local_leaf] = ckpt_leaf

    for name, module in model.named_modules():
        if hasattr(module, "__init_lora__"):
            should_apply = True
            if target_modules:
                leaf = name.split(".")[-1] if name else name
                effective = rev_mapping.get(leaf, leaf)
                should_apply = any(target == effective for target in target_modules)
            if should_apply:
                module.__init_lora__(r, lora_alpha, lora_dropout)
    return model


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def resolve_weight_spec(
    model: nn.Module,
    weight_name: str,
    *,
    config: Config,
    named_modules: dict[str, nn.Module] | None = None,
) -> ResolvedWeight | None:
    if named_modules is None:
        named_modules = dict(model.named_modules())

    ctx = LoadContext(config=config, full_name=weight_name)
    parts = weight_name.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        module = named_modules.get(prefix)
        if module is None:
            continue

        resolver = getattr(module, "resolve_checkpoint_weight", None)
        if resolver is None:
            continue

        suffix = ".".join(parts[i:])
        spec = resolver(suffix, ctx)
        if spec is not None:
            return spec

    root_resolver = getattr(model, "resolve_checkpoint_weight", None)
    if root_resolver is not None:
        return root_resolver(weight_name, ctx)
    return None


def apply_resolved_weight(spec: ResolvedWeight, loaded_weight: torch.Tensor):
    if spec.skip:
        return

    if spec.transform is not None:
        loaded_weight = spec.transform(loaded_weight)

    if spec.loader is not None:
        spec.loader(loaded_weight)
        return

    if spec.param is not None:
        weight_loader = getattr(spec.param, "weight_loader", default_weight_loader)
        if spec.shard_id is None:
            weight_loader(spec.param, loaded_weight)
        else:
            weight_loader(spec.param, loaded_weight, spec.shard_id)
        return

    if spec.buffer is not None:
        spec.buffer.copy_(loaded_weight)
        return

    raise ValueError("ResolvedWeight must specify loader, param, buffer, or skip.")


def try_load_direct(model: nn.Module, weight_name: str, loaded_weight: torch.Tensor) -> bool:
    try:
        param = model.get_parameter(weight_name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return True
    except (AttributeError, KeyError):
        pass

    try:
        buffer = model.get_buffer(weight_name)
        buffer.copy_(loaded_weight)
        return True
    except (AttributeError, KeyError):
        return False


def try_load_via_packed_mapping(
    model: nn.Module,
    packed_modules_mapping: dict,
    weight_name: str,
    loaded_weight: torch.Tensor,
    config: Config,
) -> bool:
    for k in packed_modules_mapping:
        if k not in weight_name:
            continue

        if config.model_name == "llada" and k == "ff_out" and "transformer.ff_out" in weight_name:
            continue
        elif config.model_name == "llada" and k == "transformer.ff_out":
            v, shard_id = packed_modules_mapping[k]
            assert v == "lm_head"
            param_name = "lm_head.weight"
        else:
            v, shard_id = packed_modules_mapping[k]
            param_name = weight_name.replace(k, v)

        if "layernorm" in param_name:
            try:
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            except (AttributeError, KeyError):
                try:
                    buffer = model.get_buffer(param_name)
                    buffer.copy_(loaded_weight)
                except (AttributeError, KeyError):
                    pass
        else:
            try:
                param = model.get_parameter(param_name)
                weight_loader = partial(
                    getattr(param, "weight_loader"),
                    param,
                    loaded_weight,
                )
                if shard_id is None:
                    weight_loader()
                else:
                    weight_loader(shard_id)
            except (AttributeError, KeyError):
                pass
        return True

    return False


def load_model(model: nn.Module, config: Config):
    """Load model weights and optionally LoRA weights."""
    # Enable LoRA for linear layers if LoRA is enabled
    if config.use_lora and config.lora_path:
        lora_config = load_lora_config(config.lora_path)
        packed_modules_mapping_for_lora = getattr(model, "packed_modules_mapping", None)
        if lora_config:
            logger.info(f"LoRA Config Loaded: {lora_config}")
            model = enable_lora_for_model(model, lora_config, packed_modules_mapping_for_lora)
        else:
            logger.info("No adapter_config.json found, using default LoRA parameters")
            default_config = {"r": 16, "lora_alpha": 32.0, "lora_dropout": 0.0}
            model = enable_lora_for_model(model, default_config, packed_modules_mapping_for_lora)

    # Load base model weights
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    named_modules = dict(model.named_modules())
    for file in tqdm(glob(os.path.join(config.model, "*.safetensors")), desc="Loading base model"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)

                spec = resolve_weight_spec(
                    model,
                    weight_name,
                    config=config,
                    named_modules=named_modules,
                )
                if spec is not None:
                    apply_resolved_weight(spec, loaded_weight)
                    continue

                if try_load_via_packed_mapping(model, packed_modules_mapping, weight_name, loaded_weight, config):
                    continue

                try_load_direct(model, weight_name, loaded_weight)

    # Load LoRA weights if enabled
    if config.use_lora and config.lora_path:
        if os.path.exists(config.lora_path):
            logger.info(f"Loading LoRA weights from {config.lora_path}")
            model = load_lora_weights(
                model,
                config.lora_path,
                packed_modules_mapping=packed_modules_mapping if config.model_name == "llada" else None,
                pre_merge_lora=getattr(config, "pre_merge_lora", False),
            )
        else:
            logger.warning(f"LoRA path {config.lora_path} does not exist, skipping LoRA loading")

    return model


def load_lora_weights(
    model: nn.Module,
    lora_path: str,
    packed_modules_mapping: dict | None = None,
    pre_merge_lora: bool = False,
):
    """Load LoRA weights into LoRA-enabled layers.

    Args:
        model: The model with LoRA-enabled linear layers.
        lora_path: Path to LoRA checkpoint.
        packed_modules_mapping: Optional mapping for packed modules (e.g. llada lm_head).
        pre_merge_lora: If True, merge LoRA into base weights after loading so that
            forward does not need to run LoRA computation each time. If False, keep
            LoRA separate and apply it in lora_forward during each forward pass.
    """
    try:
        lora_config = load_lora_config(lora_path)
        target_modules = lora_config.get("target_modules", [])

        lora_weights = {}

        for file in tqdm(glob(os.path.join(lora_path, "*.safetensors")), desc="Loading LoRA"):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    lora_weights[weight_name] = f.get_tensor(weight_name)

        applied_count = 0

        modified_modules = None
        if packed_modules_mapping is not None:
            modified_modules = [v for k, (v, _) in packed_modules_mapping.items() if k in target_modules]
            rev_mapping = {v: k for k, (v, _) in packed_modules_mapping.items()}

        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                should_apply = True

                if modified_modules is not None:
                    modified_module_type = ".".join(name.split(".")[-2:])
                    org_module_type = rev_mapping[modified_module_type]
                    org_name = name.replace(modified_module_type, org_module_type)
                    should_apply = any(target in modified_module_type for target in modified_modules)
                elif target_modules:
                    module_type = name.split(".")[-1] if "." in name else name
                    should_apply = any(target in module_type for target in target_modules)

                if not should_apply:
                    continue

                base_patterns = (
                    [
                        name,
                        f"base_model.model.{name}",
                        f"model.{name}",
                    ]
                    if modified_modules is None
                    else [
                        org_name,
                        f"base_model.model.{org_name}",
                        f"model.{org_name}",
                    ]
                )

                found_a = found_b = None
                for base_name in base_patterns:
                    lora_a_keys = [
                        f"{base_name}.lora_A.weight",
                        f"{base_name}.lora_A.default.weight",
                        f"{base_name}.lora_A",
                    ]
                    lora_b_keys = [
                        f"{base_name}.lora_B.weight",
                        f"{base_name}.lora_B.default.weight",
                        f"{base_name}.lora_B",
                    ]

                    for key in lora_a_keys:
                        if key in lora_weights:
                            found_a = lora_weights[key]
                            break
                    for key in lora_b_keys:
                        if key in lora_weights:
                            found_b = lora_weights[key]
                            break

                    if found_a is not None and found_b is not None:
                        break

                if found_a is not None and found_b is not None:
                    if hasattr(module, "tp_size") and module.tp_size > 1:
                        if hasattr(module, "tp_dim") and module.tp_dim == 0:
                            shard_size = found_b.size(0) // module.tp_size
                            start_idx = module.tp_rank * shard_size
                            found_b = found_b[start_idx : start_idx + shard_size]
                        elif hasattr(module, "tp_dim") and module.tp_dim == 1:
                            shard_size = found_a.size(1) // module.tp_size
                            start_idx = module.tp_rank * shard_size
                            found_a = found_a[:, start_idx : start_idx + shard_size]

                    try:
                        module.lora_A.data.copy_(found_a)
                        module.lora_B.data.copy_(found_b)
                        applied_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load LoRA weights for {name}: {e}")

        if pre_merge_lora:
            mergeable_modules = [module for module in model.modules() if hasattr(module, "merge_lora")]
            for module in tqdm(mergeable_modules, desc="Merging LoRA"):
                module.merge_lora()
            logger.info(f"LoRA weights applied to {applied_count} layers and merged into base")
        else:
            logger.info(f"LoRA weights applied to {applied_count} layers (unmerged, applied per forward)")

    except Exception as e:
        logger.error(f"Error loading LoRA weights: {e}")
        logger.warning("Continuing with base model only")

    return model
