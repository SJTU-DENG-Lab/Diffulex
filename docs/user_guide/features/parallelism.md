# Model and Data Parallelism

Diffulex exposes tensor, data, and expert parallel dimensions. The effective
world size is computed from these values, and the number of visible CUDA devices
must be large enough for the requested topology.

## Tensor Parallelism

`tensor_parallel_size` partitions model compute across devices.

Set it to an integer from `1` to `8`. The core `Config` default is `2`, while
the server and benchmark CLIs default to `1` so a fresh run can start on a
single GPU.

Increase tensor parallelism when one model replica does not fit or when a model
family expects tensor-parallel execution. Use `1` for initial debugging.

## Data Parallelism

`data_parallel_size` runs independent request-processing groups.

Set it to an integer from `1` to `1024`. The default is `1`, which means a
single request-processing group.

Data parallelism is useful for serving throughput when each group can own a
model-parallel worker set. It increases the required CUDA device count.

## Expert Parallelism

`expert_parallel_size` is reserved for MoE execution topology.

The config accepts integer values from `1` to `32768`, and defaults to `1`.
In the current engine path, keep it at `1`.

Current config validation requires `expert_parallel_size == 1` and
`moe_dispatcher_backend == "standard"` because MoE all-to-all backends are
currently unsupported in this path.

## Device Selection

Use `device_ids` or `--device-ids` to select logical CUDA device IDs. When
`CUDA_VISIBLE_DEVICES` is set, PyTorch remaps visible physical GPUs to logical
IDs starting at `0`.

## Related Arguments

| Surface | Names | Notes |
| --- | --- | --- |
| Python/config | `tensor_parallel_size`, `data_parallel_size`, `expert_parallel_size`, `device_ids` | Use these when constructing `Config` or editing YAML. |
| CLI | `--tensor-parallel-size`, `--data-parallel-size`, `--expert-parallel-size`, `--device-ids` | Use these when launching server or benchmark commands. |
