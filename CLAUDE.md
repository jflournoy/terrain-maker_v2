# CLAUDE.md — terrain-maker

> **Note to Claude:** This file holds the rules for all work in this project. It is a vendored
> copy of the shared config in `~/code/claude-config`; take upstream changes with
> `node install.js --update <this repo> --apply` from there, and local edits survive. For
> task-specific guidance, consult the guides in `.claude/guides/` listed at the bottom. Load
> them based on what you're working on—don't assume you need everything.

## Critical Rules (Always Apply)

**These rules are non-negotiable and apply to all work.**

## Running commands

**CRITICAL: Never include comments in bash command blocks. Run commands without inline or preceding comments.**

Doing so makes it hard to approve or deny commands in settings.json

### NO SILENT FALLBACKS — THIS IS A HARD RULE

Silent fallbacks are among the most dangerous patterns in software. They mask bugs, produce incorrect results quietly, and make debugging nearly impossible.

- **Never write code that silently falls back to a different code path when the primary path fails.**
- If data is missing → **error loudly** with a clear, actionable message.
- If a file is not found → **stop and tell the user** what file was expected and where.
- If a parameter is wrong → **throw an error**, do not substitute a default silently.
- If a model is not available → **fail**, do not substitute a different model silently.
- **Try-catch used to silently swallow errors is forbidden** unless the user has explicitly asked for fallback behavior over your explicit objection.
- **Optional parameters that silently change behavior are forbidden.** If a parameter controls which code path runs, its absence must cause an error or explicit warning — not silent substitution.

The only acceptable fallback pattern is one where:
1. The user has explicitly requested it in this conversation, AND
2. You have raised your objection to it on the record, AND
3. The fallback produces a **visible, logged warning** every time it fires.

### TEST INTEGRITY — NEVER MAKE A TEST PASS BY WEAKENING IT

**Never change a test solely to make it pass.** A failing test is information. Suppressing it
destroys the information and leaves the bug.

When a test fails after a change:

1. **Stop and report it.** Say which test, and what the failure actually says.
2. **Explain why it is failing** — did the change break behavior, or did it correctly change
   behavior the test still encodes?
3. **Ask which it is.** That judgment is the user's, not yours.

**Allowed without asking:** adding tests for new behavior; renaming or reorganizing tests
without changing what they assert; fixing a test that is itself provably wrong, saying so.

**Forbidden without explicit approval:** changing an expected value; loosening an assertion;
adding a tolerance to make a comparison pass; marking a test skipped, pending, or `.only`
elsewhere; deleting a failing test; wrapping a failing call so the error is swallowed.

This applies with full force when the number is the deliverable. A weakened assertion in
analysis code is a published wrong result with a green check mark next to it.

### ENFORCEMENT BEATS DOCUMENTATION

A rule with no mechanism behind it does not happen — it just looks like it does. If a rule
here matters, prefer a hook, a test, or a script that enforces it over a paragraph asking for
it. And **the enforcement mechanism is production code**: it gets a test like anything else.
An unverified gate is worse than no gate, because people stop checking the thing themselves.

### ALWAYS use `date` command for dates

Never assume or guess dates. Always run `date "+%Y-%m-%d"` when you need the current date for documentation, commits, or any other purpose.

### AI Integrity Principles

**Always provide honest, objective recommendations based on technical merit, not user bias.**

- **Never agree with users by default** - evaluate each suggestion independently
- **Challenge bad ideas directly** - if something is technically wrong, say so clearly
- **Recommend best practices** even if they contradict user preferences
- **Explain trade-offs honestly** - don't hide downsides of approaches
- **Prioritize code quality** over convenience when they conflict
- **Question requirements** that seem technically unsound
- **Suggest alternatives** when user's first approach has issues
- **Disagree when necessary** — silence is complicity. If you spot a bug, design flaw, security issue, or bad pattern, name it.

Examples of honest responses:
- "That approach would work but has significant performance implications..."
- "I'd recommend against that pattern because..."
- "While that's possible, a better approach would be..."
- "That's technically feasible but violates [principle] because..."
- "I'm concerned about [issue]. Let me explain why this won't work as written..."

## Commands

- `/commit` — atomic commits with quality checks
- `/push` — push, after checking CI is not already red
- `/hygiene` — project health: detects R, Python or Node and uses that project's runner
- `/next` — priorities, via the `next-priorities` agent
- `/refactor` — refactoring analysis for code a human reads
- `/refactor-verified` — refactoring analysis for code nobody reads, where checks replace review

Claude Code now covers natively what the rest of this repo's commands used to do:
transcripts and `--resume` replace session capture, the memory system replaces learning
capture, `TodoWrite` and `gh` replace todo management, `/loop` and `/schedule` replace
monitoring, and `/code-review` and `/simplify` replace the quality commands. They were
removed rather than maintained in parallel.

See [.claude/guides/workflow.md](.claude/guides/workflow.md) for full collaboration guidelines.

## Never run the example or render scripts unless asked

The scripts in `examples/` — `detroit_combined_render.py`, `detroit_elevation_real.py`,
`san_diego_flow_demo.py` and friends — process large DEMs, run Blender renders that take
minutes to hours, and consume serious CPU, GPU and memory. **Do not execute them, and do not
run any rendering operation, unless the user explicitly asks.**

Reading them, editing them to fix bugs or add features, suggesting parameter changes, and
writing test scripts that avoid the expensive paths are all fine. So is suggesting a command
for the user to run themselves. `--no-render` exercises the data pipeline alone, with their
approval.

## Know what resolution your data is at

This is the project's main way to lose an afternoon. The pipeline carries three resolutions at
once: the **original DEM** at source resolution, potentially 10K×10K or more; **flow
computation**, downsampled to target_vertices, typically 1–2K on a side; and **mesh vertices**,
downsampled again for rendering.

Expensive operations on full-resolution data will OOM. `scipy.ndimage.distance_transform_edt()`
allocates arrays matching the input plus index arrays; `scipy.ndimage.morphological_*` builds
large intermediates; so does convolution with a big kernel, or anything allocating a temporary
the size of its input.

```python
distances, indices = distance_transform_edt(~stream_mask, return_indices=True)
# 10K×10K: distances (floats) + 2 index arrays (ints) = ~2GB+, and it OOMs
```

Work at flow resolution, which is already downsampled; or downsample before the expensive step;
or check the size and refuse, loudly, with the library's memory guard:

```python
from terrain_maker.terrain._memory import EDT_DISTANCES_AND_INDICES, check_memory

check_memory(stream_mask.size, EDT_DISTANCES_AND_INDICES, "variable-width line expansion")
distances, indices = distance_transform_edt(~stream_mask, return_indices=True)
```

`check_memory` raises `ArrayTooLargeError` (a `MemoryError`) naming the operation and the fix
instead of letting the OS kill the process. The budget is half of available RAM, or
`TERRAIN_MAKER_MEMORY_LIMIT_GB` if set.

When you add a feature: check the input resolution first, prefer flow or mesh resolution, guard
it with `check_memory` if it must run at full resolution, and document which resolution it
operates on.

## Use the library; it has a grammar

terrain-maker is well-designed and the point is to use it fully rather than reimplement around
it. The `Terrain` class is the primary interface — it holds the data layers, manages coordinate
systems and transforms, reprojects and resamples automatically, and builds meshes for Blender.

The central pattern is the data layer, and it works for *any* geographic raster — elevation,
score grids, rasterized roads, trails, land cover, computed slopes:

```python
terrain.add_data_layer(
    name="my_layer",
    data=my_grid,             # 2D numpy array
    transform=my_transform,   # affine in source CRS
    crs="EPSG:4326",
    target_layer="dem",       # align to the DEM's transformed state
)
```

`target_layer="dem"` is what makes it work: the library reprojects source CRS → DEM CRS,
resamples to the DEM's shape (this is where downsampling is handled for you), and aligns the
coordinates. Add the layer once, then read it back aligned rather than doing your own
resampling.

The library also covers transforms (`reproject_raster`, `downsample_raster`, `flip_raster`,
`scale_elevation`, `smooth_raster`, and custom ones via `add_transform`), analysis
(`detect_water_highres`, slope, aspect, hillshade), coloring (`elevation_colormap`,
`slope_colormap`, `set_color_mapping`, `set_blended_color_mapping`, `compute_proximity_mask`),
mesh generation (`create_mesh`, with automatic downsampling to a target vertex count, boundary
extension and centering), and Blender (`position_camera_relative`, `setup_light`,
`apply_vertex_colors`, `render_scene_to_file`, `create_background_plane`).

**Write examples library-first.** Use `Terrain` methods, `ScoreCombiner` and the score config
classes, `GriddedDataLoader`, the helpers in `color_mapping.py`, and public functions from
`terrain/`. Do not call the Blender API directly, reimplement scoring, hand-roll mesh
generation, or write custom DEM loading — each of those has a library path, and examples that
use it survive refactors and double as templates.

Suggest a *new* library function when you implement the same processing twice, when examples
repeat a loading/transforming/coloring pattern, or when vertex-coloring or mesh logic recurs.
Do not suggest one for a one-off, for something under ten straightforward lines, or for logic
specific to a single visualization.

## Scripts that render through Blender

Import `bpy` at module level, like any other dependency — no try/except wrapper. These are
ordinary Python scripts run with ordinary Python, not "Blender scripts". Say so in the
docstring, list the requirement, and take arguments with argparse.

```bash
uv pip install bpy          # or: uv sync --extra blender
uv run python -c "import bpy; print(f'Blender {bpy.app.version_string}')"
```

Needs Python 3.11+ to match Blender's, and about 400MB. Route all Blender work through
terrain-maker's `Terrain` class rather than low-level `bpy` calls — your script's job is the
data pipeline. `examples/detroit_dual_render.py` is the pattern to copy.

## Colormaps

Perceptually uniform and colorblind-safe, always. Equal steps in the data should be equal
perceptual steps in color, and it should survive conversion to grayscale.

- **Elevation/terrain**: `michigan` (the custom Great Lakes blue → forest green → upland meadow
  → sand dune ramp, preferred for the Michigan and Detroit examples), then `turbo`, `cividis`.
  `gist_earth` and `terrain` are intuitive but less uniform.
- **Scores**: the viridis family — `viridis`, `plasma`, `inferno`, `magma`, `cividis`.
- **Snow and ice**: `boreal_mako`, the custom boreal green → mako blue → pale mint ramp with the
  edge effect.
- **Diverging data**: a center-neutral ramp, `coolwarm` or `RdBu`, anchored at the midpoint.

**Never use a rainbow colormap** — `jet`, `hsv` — they are perceptually non-uniform and invent
structure that is not in the data. For publication prefer the viridis family; for a
presentation, the terrain-specific ramps read faster. Dual colormaps pair a terrain-like base
(`michigan`, `gist_earth`) with a viridis-family overlay.

## State tracking

`TodoWrite` is the primary, user-visible task list. `.claude-current-status` holds the notes
that do not fit a todo item — timestamps, decisions, file references, the context that makes a
session resumable. Start with `TodoWrite`, always add notes to `.claude-current-status`, and
prune stale ones as you go.

## Documentation

Sphinx, built through npm scripts, with `uv` for Python deps:

```bash
uv sync --extra docs        # first time
npm run docs:build          # or docs:build:clean, docs:serve (:8080), docs:check
npm run docs:images         # regenerates from REAL data, 2-5 minutes
npm run docs:build:full     # images + build
```

Sources live in `docs/source/` — `api/` (19 modules, 100% coverage), `examples/`, `guides/` —
and the build lands in `docs/build/html/`, which is not in git. A new module needs an RST file
in `docs/source/api/` and an entry in `docs/source/index.rst`.

Documentation images are generated from real DEM and SNODAS data, so the files must be present
in `data/` before running. They land under `docs/images/` in numbered stages: `01_raw/`,
`02_slope_stats/`, `03_slope_penalties/`, `04_score_components/`, `05_final/`. To add one:
have the example script write into the right subdirectory, add its invocation to
`scripts/generate-docs-images.sh`, and reference it relatively
(`../../images/subdir/image.png`).

Simple Analytics is configured in `docs/source/conf.py` via `html_js_files`, so Sphinx includes
it in every generated page.

## When to Consult Each Guide

### 🔴 Load for Feature/Bug Work

- [.claude/guides/tdd.md](.claude/guides/tdd.md) — When implementing features, fixing bugs, or refactoring
  - Defines how to write tests first, then code
  - Required for any non-trivial code change

### 📋 Load for General Development

- [.claude/guides/standards.md](.claude/guides/standards.md) — Code quality expectations, testing strategy
  - Consult when: running tests, committing code, reviewing architecture
  - Covers: complexity limits, test standards, markdown validation, architecture principles

### 📊 Load for Statistical / Modeling Work

- [.claude/guides/bayesian-production.md](.claude/guides/bayesian-production.md) — When working
  on Bayesian models, Stan code, MCMC diagnostics, or time-series inference
  - Covers: Kalman filters, Pathfinder, reparameterization, correlation-matrix priors,
    regularized horseshoe, warm-starting, R̂/ESS thresholds
  - Every Stan fence is compiled by `node check-stan.js`; the R (cmdstanr) and Python
    (cmdstanpy) calls differ

## Review Agents

Two fire automatically. A `PostToolUse` hook (`hooks/reviewer-dispatch.cjs`) routes an
edited file to its reviewer:

| Edited | Agent |
|---|---|
| `*.stan` | `stan-reviewer` — silent-wrong-answer bugs, geometry, wasted cycles |
| `*.R` `*.Rmd` `*.qmd` | `r-analysis-reviewer` — joins, coercion, non-determinism, claims |

Three are on demand — ask for them by name:

- `statistical-analysis-reviewer` — skeptical peer review of a finished analysis, before it
  is shared. Design, assumptions, inference, and whether the conclusion is supported.
- `determinism-reviewer` — finds work done by model reasoning that tested code could do.
- `voice-authenticator` — checks prose against [.claude/guides/voice.md](.claude/guides/voice.md).

All of them report findings and never edit. Silence them for a session with
`CLAUDE_REVIEWER_DISPATCH=0`. Adding a file type is one entry in `RULES` plus a test.

### 📐 Load for R Work

- [.claude/guides/r-development.md](.claude/guides/r-development.md) — Writing R:
  data.table over tidyverse, targets pipelines, testthat with reference semantics
  - Covers: approved packages, the tidyverse→data.table substitution table, parquet I/O,
    joins that lose rows and identifiers that lose zeros, large-data and parallelism
    gotchas, who owns the object `:=` just modified, model output as data.tables, Quarto
    reports on top of targets, and running a pipeline on a scheduler or a cloud
  - Includes the cmdstanr/posterior rule: the rstan-backed brms accessors can hard-crash
    R with SIGABRT in a container where rstan is broken

### ✍️ Load for Reader-Facing Prose

- [.claude/guides/voice.md](.claude/guides/voice.md) — Before writing or editing any prose a reader will see.
  The guide is the craft; each project declares in its own `CLAUDE.md` which files it covers
  and where they sit on the register dial
  - Report `.qmd` files, figure captions, README and docs pages, supplements
  - Defines the register dial (paper / commentary / conversational) and the rules for each