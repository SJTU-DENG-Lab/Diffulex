# Extending the Engine

Start here when adding model support, decoding behavior, or kernel
implementations. Keep each extension focused: add the component required for the
new behavior, then verify it with a targeted test or smoke run before expanding
the scope.

If the goal is a new dLLM inference algorithm, read
[Research Engine](research_engine.md) first. It maps block-level algorithms to
Diffulex's Block Buffer backend and lists the files a code agent should cover.

## Add New Models

See [Add New Models](add_new_models.md) for the model, sampler, config, CLI, and
verification workflow.

## Add New Decoding Strategies

See [Add a Decoding Strategy](add_decoding_strategy.md) for the detailed
strategy component walkthrough.

:::{toctree}
:maxdepth: 1

add_new_models
add_decoding_strategy
add_new_kernel
research_engine
:::

Strategy work usually touches these components:

- a request class for per-request state;
- a scheduler for lifecycle policy;
- a KV cache manager for page allocation and append rules;
- attention metadata for kernel inputs;
- a model runner for tensor preparation and execution.

Use a strategy template when possible. Override only the methods whose behavior
is different.

## Add New Kernel

See [Add New Kernel](add_new_kernel.md) for kernel placement, reference testing,
integration, and profiling guidance.

## Verification Checklist

For model and strategy extensions:

- import the package that should trigger registration;
- inspect available registry keys;
- construct a `Config`;
- run a tiny offline generation;
- run a limited benchmark or server smoke test;
- add focused regression tests for the behavior you changed.
