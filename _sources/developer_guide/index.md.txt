---
orphan: true
---

# Developer Guide

These pages explain how to extend Diffulex internally to adapt new requirements.
They focus on engine boundaries, extension points, test strategy, and debugging
workflow rather than user-facing command examples.

- [Research Engine](research_engine.md)
- [The Design](the_design.md)
- [Extending the Engine](extending_the_engine.md)
- [Testing](testing.md)
- [Developer Troubleshooting](developer_troubleshooting.md)
- [Profiling](profiling.md)

## Suggested Reading Order

Read [Research Engine](research_engine.md) first when using Diffulex as a
backend for a new dLLM algorithm. Read [The Design](the_design.md) before
changing scheduler, cache, runner, or strategy code. Use
[Extending the Engine](extending_the_engine.md) when adding a model, sampler,
strategy, or kernel. Use [Testing](testing.md) and
[Developer Troubleshooting](developer_troubleshooting.md) while iterating on a
change.

Run focused tests first. Expand to full-suite or GPU-heavy checks only after the
smallest relevant verification passes.
