# diffulex.strategy.templates

`diffulex.strategy.templates` contains reusable strategy building blocks. A
concrete strategy package usually subclasses or directly reuses one of these
templates, then registers request, scheduler, KV cache manager, model runner,
and attention metadata classes under a strategy name.

The templates are organized by decoding layout:

| Template package | Use it when | Core classes |
| --- | --- | --- |
| `diffulex.strategy.templates.dual_cache` | A strategy needs the multi-block lifecycle but wants a separate cache-layout variant. | Dual-cache extension points. |
| `diffulex.mixin.multi_block` | A strategy decodes with a rolling buffer of fixed-size diffusion blocks. | Multi-block request, scheduler, KV cache, runner, attention metadata, and full-static runner templates. |
| `diffulex.strategy.templates.token_merge` | A strategy is multi-block and also passes token-merge distributions into attention. | Token-merge request, scheduler, KV cache, runner, and attention metadata templates. |

For extension work, start from the smallest template that already matches the
decoding lifecycle. Most new block-diffusion strategies should begin with the
multi-block template; only use token-merge templates when attention truly needs
per-token merge descriptors.

## dual_cache

The dual-cache template package is currently a set of named extension points for
future strategies that need separate cache views or cache lifecycles. The
planned Dual Cache mechanism is tracked separately from the standard multi-block
runtime.

## multi_block

The multi-block template represents each request as a chain of `DllmBlock`
objects, keeps a rolling block buffer, schedules prefill/decode work against the
KV cache, and prepares attention metadata for paged attention.

The request template owns most lifecycle semantics: waiting, prefilling,
decoding, completed, preempted, EOS, `max_tokens`, `max_model_len`, `max_nfe`,
and `max_repetition_run`.

The scheduler template remains narrow: it chooses which requests run this step,
asks the cache manager whether pages are available, and applies sampler writes.

## token_merge

Token-merge templates extend multi-block decoding with per-position
token-merge metadata. The DMax strategy uses this family to pass top-k token
IDs, top-k probabilities, residual probabilities, merge mode, weight, and
renormalization settings into attention metadata.
