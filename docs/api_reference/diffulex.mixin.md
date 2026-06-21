# diffulex.mixin

`diffulex.mixin` contains reusable behavior shared across requests, schedulers,
samplers, and serving paths. Mixins keep cross-cutting behavior out of concrete
strategy classes while still allowing each strategy to compose only the pieces it
needs.

| Module | Role |
| --- | --- |
| `diffulex.mixin.async_serving` | Async serving state and engine-facing async helpers. |
| `diffulex.mixin.edit` | Edit-decoding request, scheduler, and sampler behavior. |
| `diffulex.mixin.request_state` | Common request-state helpers shared by request classes. |
| `diffulex.mixin.token_merge` | Token-merge sampler behavior for DMax-style decoding. |

## diffulex.mixin.async_serving

This package supports request tracking for the HTTP serving path. It keeps
serving-specific state separate from the core offline engine loop.

| Symbol | Purpose |
| --- | --- |
| `ServingRequestState` | Tracks one serving request's lifecycle state. |
| `ServingState` | Stores serving-side request state shared by async helpers. |
| `DiffulexAsyncEngineMixin` | Adds async-serving operations to the engine implementation. |

Use this path when changing how online serving submits, aborts, or observes
engine requests.

## diffulex.mixin.edit

This package groups edit-decoding behavior used by compatible LLaDA2-style
samplers and strategies.

| Symbol | Purpose |
| --- | --- |
| `DllmBlockEditMixin` | Adds edit-specific block state. |
| `EditSamplerMixin` | Adds token edit/remask behavior to sampler implementations. |
| `EditSchedulerMixin` | Adds scheduler helpers for edit-style progress. |

Use these mixins instead of duplicating edit-specific logic inside a concrete
strategy package.

## diffulex.mixin.request_state

This module provides request-state helpers that are independent of a concrete
strategy.

| Symbol | Purpose |
| --- | --- |
| `ReqStateMixin` | Shared request-state behavior used by `DllmReq` and strategy request classes. |

Add general request helpers here only when they apply across strategies.

## diffulex.mixin.token_merge

This package supports token-merge decoding behavior used by DMax-style paths.

| Symbol | Purpose |
| --- | --- |
| `TokenMergeSamplerMixin` | Adds token-merge update behavior to compatible sampler implementations. |

Keep token-merge sampler mechanics here; strategy scheduling and cache behavior
belong in `diffulex.strategy.templates.token_merge` or a concrete strategy.
