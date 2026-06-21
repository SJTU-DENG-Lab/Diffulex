# diffulex.distributed

`diffulex.distributed` builds and stores the distributed topology used by tensor
parallel, data parallel, expert parallel, and sequence-parallel style groupings.
The rest of the engine reads this state instead of reconstructing process groups
locally.

| Module | Role |
| --- | --- |
| `diffulex.distributed.parallel_state` | Validates topology inputs, initializes torch distributed groups, and exposes the current parallel state. |

## diffulex.distributed.parallel_state

This module centralizes distributed process-group setup. It computes world size
from configured parallel dimensions, resolves topology, creates rank groups, and
stores a `ParallelState` object for later use by layers, MoE code, and model
runners.

| Symbol | Purpose |
| --- | --- |
| `WorldMesh` | Frozen description of global rank layout. |
| `BaseModelParallelLayout` | Tensor/data/sequence-parallel rank grouping for dense model execution. |
| `MoEParallelLayout` | Expert-parallel rank grouping for MoE paths. |
| `ParallelState` | Aggregates rank, world size, process groups, and parallel-layout metadata. |
| `get_world_size` | Computes effective world size from configured parallel dimensions. |
| `init_parallel_state` | Creates and stores the runtime `ParallelState`. |
| `fetch_parallel_state` | Returns the active parallel state. |
| `reset_parallel_state` | Clears global parallel state, mainly for tests or process teardown. |
| `build_parallel_state_for_test` | Builds a state object without full runtime initialization. |

Callers should fetch the existing state rather than deriving rank groups from
configuration again. That keeps layer code and MoE code aligned with the engine
worker topology.
