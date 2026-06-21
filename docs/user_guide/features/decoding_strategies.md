# Decoding Strategies

Diffulex selects strategy-specific request, scheduler, KV cache manager, model
runner, and attention metadata components through registries. The strategy is
chosen by `decoding_strategy`.

## Decoding Strategy

Set `decoding_strategy` to one of `d2f`, `multi_bd`, or `dmax`.

Benchmark config input also normalizes older aliases `multi_block_diffusion`,
`block_diffusion`, and `fast_dllm` to `multi_bd`.

The choice changes more than the sampler name:

| Strategy | Behavior |
| --- | --- |
| `d2f` | Forces full-prefix multi-block behavior and disables prefix caching. |
| `multi_bd` | Uses multi-block behavior with prefix caching enabled when compatible. |
| `dmax` | Enables DMax-style token merging on supported edit-sampling models. |

## Decoding Thresholds

Thresholds tune when a strategy adds, releases, accepts, edits, or remasks
tokens and blocks.

| Key | How to set it | What it does |
| --- | --- | --- |
| `add_block_threshold` | Start from the default `0.1`; tune as a float for block-add behavior. | Controls when another decoding block can be added. |
| `semi_complete_threshold` | Start from the default `0.9`; tune as a float for block advancement. | Controls when semi-complete block state can advance. |
| `accept_threshold` | Use a confidence value from `0` to `1`. The default is `0.9`. | Accepts mask-to-token updates once confidence is high enough. |
| `edit_threshold` | Use a confidence value from `0` to `1`. The default is `0.0`. | Accepts token-to-token edits in edit-style decoding. |
| `remask_threshold` | Use a confidence value from `0` to `1`. The default is `0.4`. | Remasks filled tokens that fall below the confidence threshold. |
| `token_stability_threshold` | Use a stability ratio from `0` to `1`. The default is `0.0`. | Controls DMax-style edit-block progress. |

Keep thresholds in YAML when comparing experiments. Use CLI overrides for short
ad hoc runs.

## Sampling Mode

Set `sampling_mode` to `naive` for the standard sampler path or `edit` for
edit-style decoding.

`sampling_mode="edit"` is restricted to edit-sampling model names:

| Compatible `model_name` |
| --- |
| `llada2` |
| `llada2_moe` |
| `llada2_mini` |
| `llada2dot1_mini` |
| `llada2_mini_dmax` |

`decoding_strategy="dmax"` requires `sampling_mode="edit"` and one of the
compatible model names.

## Related Arguments

| Surface | Names | Notes |
| --- | --- | --- |
| Python/config | `decoding_strategy`, `sampling_mode`, `decoding_thresholds` | Use these in `Config`, YAML, or Python construction. |
| CLI | `--decoding-strategy`, `--sampling-mode`, `--add-block-threshold`, `--semi-complete-threshold`, `--accept-threshold`, `--edit-threshold`, `--remask-threshold`, `--token-stability-threshold` | Use CLI overrides for short experiment runs. |
