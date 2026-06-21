# diffulex.moe

`diffulex.moe` contains Mixture-of-Experts configuration helpers, router
metadata, token dispatchers, top-k routing, and fused expert execution layers.
Model code should use the package-level builders instead of selecting MoE
implementations directly.

| Module | Role |
| --- | --- |
| `diffulex.moe.config` | Reads MoE-related attributes from model configs. |
| `diffulex.moe.dispatcher` | Token dispatcher implementations and dispatcher factory. |
| `diffulex.moe.layer` | Fused MoE layer implementations and layer factory. |
| `diffulex.moe.metadata` | Router, dispatcher, and expert execution metadata dataclasses. |
| `diffulex.moe.mode` | DeepEP mode enum and normalization helpers. |
| `diffulex.moe.topk` | Top-k router implementations and router factory. |

## diffulex.moe.config

This module normalizes model-config differences so MoE code can ask for common
concepts such as expert count, experts-per-token, sparse-layer placement, and
intermediate size.

| Symbol | Purpose |
| --- | --- |
| `get_num_experts` | Reads total expert count. |
| `get_num_experts_per_tok` | Reads top-k experts per token. |
| `get_moe_intermediate_size` | Reads MoE hidden size. |
| `is_moe_layer` | Determines whether a layer index should use MoE. |

Use these helpers rather than reading raw HF config attributes directly.

## diffulex.moe.dispatcher

This package moves tokens between ranks for expert execution. The dispatcher
factory chooses an implementation based on config.

| Symbol | Purpose |
| --- | --- |
| `TokenDispatcher` | Abstract dispatcher contract. |
| `DispatcherOutput` | Output structure returned by dispatchers. |
| `build_token_dispatcher` | Factory for the configured dispatcher backend. |
| `NaiveA2ADispatcher` | Reference all-to-all dispatcher. |
| `DeepEPDispatcher` | DeepEP-backed dispatcher for supported environments. |

The current validated config path keeps dispatcher use conservative; treat
non-standard backends as experimental unless explicitly enabled and tested.

## diffulex.moe.layer

This package executes expert MLPs after routing. It provides naive, tensor
parallel, expert parallel, and optional vLLM-backed implementations behind a
factory function.

| Symbol | Purpose |
| --- | --- |
| `build_moe_block` | Factory for MoE blocks. |
| `FusedMoE` | Base fused MoE layer contract. |
| `SharedExpertMLP` | Shared expert MLP helper. |
| `NaiveFusedMoE` | Reference fused MoE implementation. |
| `TPFusedMoE` | Tensor-parallel fused MoE implementation. |
| `EPFusedMoE` | Expert-parallel fused MoE implementation. |

Model layers should call the factory rather than instantiate implementation
classes directly.

## diffulex.moe.metadata

This module defines structured metadata passed between routers, dispatchers, and
expert execution layers.

| Symbol | Purpose |
| --- | --- |
| `RouterMetadata` | Router output metadata. |
| `DispatchMetadata` | Base dispatcher metadata. |
| `ExpertExecutionMetadata` | Metadata needed while executing experts. |
| `DeepEPDispatchMetadata` | DeepEP-specific dispatcher metadata. |
| `DispatcherStage` | Dispatcher lifecycle stage enum. |

Use these dataclasses to keep dispatcher and expert-layer contracts explicit.

## diffulex.moe.mode

This module defines supported DeepEP mode values and string normalization.

| Symbol | Purpose |
| --- | --- |
| `DeepEPMode` | Enum for `normal`, `low_latency`, and `auto` dispatcher modes. |

## diffulex.moe.topk

This package selects experts for each token. It provides top-k router
implementations and a factory used by MoE layers.

| Symbol | Purpose |
| --- | --- |
| `TopKRouter` | Base router contract. |
| `TopKOutput` | Router output dataclass. |
| `build_topk_router` | Factory for configured router behavior. |
| `NaiveTopKRouter` | Standard top-k router. |
| `GroupLimitedTopKRouter` | Router with group-limited expert selection. |
