"""
Pin the outputs of examples/detroit_combined_render.py so it can be refactored safely.

Runs main() with --mock-data --no-render (seeded; no real data, no Blender render) for a
set of flag variants, each in its own process, and records what the run produced: the
terrain mesh (vertices, faces, colors, data layers, model parameters), the Blender scene
objects, and the pixels of every image written. A refactor is behavior-preserving only if
`check` reports every variant unchanged.

With --real, runs on your real data instead (no --mock-data; needs data/dem/detroit, the
score files and network for lakes/roads). Real-data pins depend on your local data, so they
live in tools/pins/local/ (not committed): the first `check --real` records a baseline, later
runs report what changed, and `record --real` accepts the current outputs as the new baseline.

Usage (from the repository root):
    uv run python tools/pin_combined_render.py record   # write tools/pins/combined_render.json
    uv run python tools/pin_combined_render.py check    # compare against the pins
    uv run python tools/pin_combined_render.py check default wavelet   # selected variants
    uv run python tools/pin_combined_render.py check --real    # real data vs your local baseline
    uv run python tools/pin_combined_render.py record --real   # accept real-data outputs
"""

import hashlib
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PINS = Path(__file__).parent / "pins" / "combined_render.json"
REAL_PINS = Path(__file__).parent / "pins" / "local" / "combined_render_real.json"

# Real-data variants: the default preview and the saved preset (roads, lakes, upscaling)
REAL_VARIANTS = {
    "real_default": [],
    "real_preset_skiing_overhead_dark": ["@examples/presets/skiing_overhead_dark.args"],
}

VARIANTS = {
    "default": [],
    "skiing_base": ["--base-scores", "skiing"],
    "normalize_gamma": ["--normalize-scores", "--gamma", "0.7"],
    "score_floor": ["--score-floor", "0.5"],
    "no_water": ["--no-water"],
    "print_colors_no_purple": ["--print-colors", "--no-purple"],
    "elev_saturation": ["--elev-saturation", "0.5", "--elev-value", "0.3"],
    "smooth_despeckle": ["--smooth", "--despeckle-dem"],
    "wavelet": ["--wavelet-denoise"],
    "remove_bumps": ["--remove-bumps", "3"],
    "smooth_scores": ["--smooth-scores"],
    "two_tier": ["--two-tier-edge", "--edge-spacing", "1.0"],
    "two_tier_catmull": ["--two-tier-edge", "--use-catmull-rom"],
    "hdri_background": ["--hdri-lighting", "--background"],
    "adaptive": ["--adaptive-smooth"],
    "adaptive_bumps": ["--adaptive-smooth", "--remove-bumps", "3"],
}


def _digest(array):
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.floating):
        array = np.round(array.astype(np.float64), 5)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _file_digest(path):
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        from PIL import Image

        return _digest(np.asarray(Image.open(path)))  # pixels only: metadata embeds times
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_one(flags, real=False):
    """Run main() once in this process and summarize what it produced."""
    import bpy

    sys.path[:0] = [".", "examples"]
    import terrain_maker.terrain.core as core

    meshes = []
    create_mesh = core.Terrain.create_mesh

    def recording_create_mesh(self, *args, **kwargs):
        obj = create_mesh(self, *args, **kwargs)
        meshes.append(
            {
                "vertices": _digest(self.vertices),
                "faces": _digest(np.concatenate([np.asarray(f) for f in self.faces])),
                "colors": _digest(self.colors),
                "boundary_colors": (
                    None if self.boundary_colors is None else _digest(self.boundary_colors)
                ),
                "layers": {
                    name: _digest(info.get("transformed_data", info["data"]))
                    for name, info in sorted(self.data_layers.items())
                },
                "model_params": {k: str(v) for k, v in sorted(self.model_params.items())},
            }
        )
        return obj

    core.Terrain.create_mesh = recording_create_mesh
    bpy.ops.wm.read_factory_settings(use_empty=True)
    np.random.seed(0)  # the mock DEM is drawn with np.random
    out = Path(tempfile.mkdtemp())
    import detroit_combined_render

    argv = ([] if real else ["--mock-data"]) + ["--no-render", "--output-dir", str(out), *flags]
    code = detroit_combined_render.main(argv)
    assert code in (0, None), f"main() returned {code}"
    return {
        "meshes": meshes,
        "scene": sorted(
            [o.name, o.type, [round(v, 4) for v in o.location]] for o in bpy.data.objects
        ),
        "files": {
            str(p.relative_to(out)): _file_digest(p) for p in sorted(out.rglob("*")) if p.is_file()
        },
    }


def run_all(names, variants, real):
    """Run each variant in a fresh process (main() mutates global matplotlib/Blender state)."""
    results = {}
    for name in names:
        proc = subprocess.run(
            [sys.executable, __file__, "_one_real" if real else "_one", json.dumps(variants[name])],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            results[name] = {"error": proc.stderr.strip().splitlines()[-1][:300]}
        else:
            results[name] = json.loads(proc.stdout.strip().splitlines()[-1])
        print(f"  {name}: {'ERROR ' + results[name]['error'] if 'error' in results[name] else 'ran'}",
              file=sys.stderr)
    return results


def main():
    args = sys.argv[1:]
    real = "--real" in args
    args = [a for a in args if a != "--real"]
    command, names = args[0], args[1:]
    if command in ("_one", "_one_real"):
        logging.disable(logging.CRITICAL)
        print(json.dumps(run_one(json.loads(names[0]), real=command == "_one_real")))
        return 0

    variants, pins_path = (REAL_VARIANTS, REAL_PINS) if real else (VARIANTS, PINS)
    names = names or list(variants)
    results = run_all(names, variants, real)
    errors = [n for n in names if "error" in results[n]]

    if command == "record" or (real and not pins_path.exists()):
        if errors:
            print(f"Not recording: {', '.join(errors)} failed")
            return 1
        pins_path.parent.mkdir(parents=True, exist_ok=True)
        pinned = json.loads(pins_path.read_text()) if pins_path.exists() else {}
        pinned.update(results)
        pins_path.write_text(json.dumps(pinned, indent=1, sort_keys=True) + "\n")
        what = "Recorded" if command == "record" else "No baseline yet; recorded"
        print(f"{what} {len(results)} variants to {pins_path}")
        return 0

    pinned = json.loads(pins_path.read_text())
    changed = [n for n in names if results[n] != pinned.get(n)]
    for name in changed:
        print(f"CHANGED: {name}")
        old, new = pinned.get(name, {}), results[name]
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                print(f"  {key}: {json.dumps(old.get(key))[:200]}\n    -> {json.dumps(new.get(key))[:200]}")
    print(f"{len(names) - len(changed)}/{len(names)} variants unchanged")
    if changed and real:
        print("If these changes are intended, accept them with: "
              "uv run python tools/pin_combined_render.py record --real")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
