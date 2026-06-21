---
orphan: true
---

# API Reference

- [diffulex](diffulex.md)
- [diffulex_bench](diffulex_bench.md)
- [diffulex_kernel](diffulex_kernel.md)

These pages describe the stable package-level entry points and the shallow
source package map used by developers. They are intentionally higher level than
generated autodoc because several modules depend on GPU libraries, strategy
imports, or optional kernel dependencies.

Use `diffulex` for in-process inference, `diffulex_bench` for evaluation
configuration and lm-eval integration, and `diffulex_kernel` only for focused
kernel tests or low-level integration work.
