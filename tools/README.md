# Tools

Development and debugging scripts for terrain-maker: flow validation, diagnostics, data
downloads and one-off plots. They are not usage examples; see [`examples/`](../examples/)
for those.

Run them from the repository root, as they read and write paths such as `examples/output/`:

```bash
uv run python tools/validate_flow_complete.py --bigness small
```

| Prefix | Purpose |
| --- | --- |
| `validate_flow_*` | End-to-end checks of flow accumulation against real data |
| `diagnose_*` | Inspect a specific failure mode (outlets, lakes, spillways, SNODAS coverage) |
| `test_*`, `quick_flow_test`, `debug_*` | Ad-hoc checks written while fixing a bug (not part of the pytest suite) |
| `download_*` | Fetch sample data |
| `plot_*`, `visualize_*` | One-off diagnostic plots |
| `check_numba_threading` | Report Numba threading configuration |

Like the examples, most of these process real DEMs and can take minutes and a lot of memory.
