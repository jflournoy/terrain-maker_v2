"""
Pin the outputs of examples/detroit_combined_render.py so it can be refactored safely.

Runs main() with --mock-data --no-render (seeded; no real data, no Blender render) for a
set of flag variants, each in its own process, and records what the run produced: the
terrain mesh (vertices, faces, colors, data layers, model parameters), the Blender scene
objects, and the pixels of every image written. A refactor is behavior-preserving only if
`check` reports every variant unchanged.

Pins must hold across machines: the same code on another CPU, BLAS or compiler differs in
the last bits of its floats. Each array is pinned by a digest plus, for mock runs, the array
itself (tools/pins/arrays/<digest>.npz, shared between variants). When a digest differs, the
arrays are compared within the tolerances below, and every match that is not exact prints
how large the difference was, so a pass within tolerance is never silent.

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
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PINS = Path(__file__).parent / "pins" / "combined_render.json"
REAL_PINS = Path(__file__).parent / "pins" / "local" / "combined_render_real.json"
ARRAYS = Path(__file__).parent / "pins" / "arrays"
FORMAT = 2  # bump when the shape of a pin changes; old pin files are refused, not misread

# Tolerances for comparing across machines. Each is set from what float noise can do, far
# below what a change in behavior does:
# - Floats: last-bit differences grow through smoothing and resampling, but stay near 1e-7
#   of the array's range in float32. A wrong computation moves values by percent.
FLOAT_RTOL = 1e-5  # max |new - ref| allowed, as a fraction of max |ref|
# - 8-bit colors come from a 256-entry colormap lookup, so a float nudged across a bin edge
#   jumps a whole entry: up to 2.7 codes in viridis, 24 in boreal_mako's purple band. Noise
#   can only do that to the rare values sitting within ~1e-7 of an edge (2 of 460k in the
#   Detroit mock); a change in color logic moves a large share (330k of 460k for a
#   different smoothing backend). So colors are judged by how many values change, not by how far.
COLOR_MAX_CHANGED_FRACTION = 1e-4
UINT8_MAX_STEP = 1  # per-pixel change in a PNG that antialiasing noise can make
# - Images (histogram PNGs) are drawn from those floats: a value crossing a bin edge moves a
#   bar by a fraction of a pixel and re-antialiases its edge. Any visible change to the plot
#   (data, labels, layout) touches far more pixels than that.
IMAGE_MAX_CHANGED_FRACTION = 1e-3  # pixels differing by more than UINT8_MAX_STEP

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


class ArrayPinner:
    """Pins arrays by digest, and saves each one under its digest when given a directory."""

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def __call__(self, array, kind="array"):
        array = np.asarray(array)
        digest = _digest(array)
        # Other dtypes (faces, indices) must match exactly, so their digest is the whole pin
        if self.save_dir is not None and _has_tolerance(array.dtype):
            path = Path(self.save_dir) / f"{digest}.npz"
            if not path.exists():
                np.savez_compressed(path, array=array)
        return {"sha": digest, "kind": kind, "shape": list(array.shape), "dtype": str(array.dtype)}


def _has_tolerance(dtype):
    return np.issubdtype(dtype, np.floating) or dtype == np.uint8


def _file_pin(path, pin_array):
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        from PIL import Image

        return pin_array(np.asarray(Image.open(path)), kind="image")  # pixels: metadata has times
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def compare_arrays(ref, new, kind):
    """(ok, detail): whether new matches ref within the tolerance for its kind."""
    if ref.shape != new.shape or ref.dtype != new.dtype:
        return False, f"{ref.dtype}{list(ref.shape)} -> {new.dtype}{list(new.shape)}"
    if np.issubdtype(ref.dtype, np.floating):
        ref_nan, new_nan = np.isnan(ref), np.isnan(new)
        if not np.array_equal(ref_nan, new_nan):
            return False, f"NaN cells changed: {int(np.sum(ref_nan != new_nan))}"
        finite = ~ref_nan
        scale = float(np.max(np.abs(ref[finite]), initial=0.0))
        diff = float(np.max(np.abs(new[finite].astype(np.float64) - ref[finite]), initial=0.0))
        if scale == 0.0:
            return diff == 0.0, f"max diff {diff:.3g} on an all-zero reference"
        rel = diff / scale
        return rel <= FLOAT_RTOL, f"max diff {rel:.2g} of range (allowed {FLOAT_RTOL:g})"
    if ref.dtype == np.uint8:
        step = np.abs(new.astype(np.int16) - ref.astype(np.int16))
        if kind == "image":
            pixels = np.any(step > UINT8_MAX_STEP, axis=-1) if step.ndim == 3 else step > UINT8_MAX_STEP
            frac = float(np.mean(pixels))
            return frac <= IMAGE_MAX_CHANGED_FRACTION, (
                f"{frac:.2%} of pixels changed by more than {UINT8_MAX_STEP} "
                f"(allowed {IMAGE_MAX_CHANGED_FRACTION:.2%})"
            )
        changed = int(np.sum(step > 0))
        frac = changed / max(step.size, 1)
        return frac <= COLOR_MAX_CHANGED_FRACTION, (
            f"{changed} of {step.size} values changed ({frac:.2g}, allowed "
            f"{COLOR_MAX_CHANGED_FRACTION:g}), max step {int(step.max(initial=0))}"
        )
    equal = np.array_equal(ref, new)
    return equal, "identical" if equal else f"{int(np.sum(ref != new))} values differ (exact match required)"


def _is_array_pin(value):
    return isinstance(value, dict) and "sha" in value and "kind" in value


def compare(old, new, ref_dir, new_dir, path=""):
    """Walk two pins; return (failures, within_tolerance) as lists of 'path: detail' lines."""
    failures, tolerated = [], []
    if _is_array_pin(old) and _is_array_pin(new):
        if old["sha"] == new["sha"]:
            return failures, tolerated
        ref_file = None if ref_dir is None else Path(ref_dir) / f"{old['sha']}.npz"
        new_file = None if new_dir is None else Path(new_dir) / f"{new['sha']}.npz"
        if not _has_tolerance(np.dtype(old["dtype"])):
            failures.append(f"{path}: digest {old['sha']} -> {new['sha']} (exact match required)")
            return failures, tolerated
        if ref_file is None or new_file is None or not ref_file.exists():
            failures.append(f"{path}: digest {old['sha']} -> {new['sha']} (no reference array to compare)")
            return failures, tolerated
        ok, detail = compare_arrays(np.load(ref_file)["array"], np.load(new_file)["array"], new["kind"])
        (tolerated if ok else failures).append(f"{path}: {detail}")
    elif isinstance(old, dict) and isinstance(new, dict) and not _is_array_pin(old) and not _is_array_pin(new):
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                failures.append(f"{path}/{key}: {'added' if key in new else 'removed'}")
                continue
            f, t = compare(old[key], new[key], ref_dir, new_dir, f"{path}/{key}")
            failures += f
            tolerated += t
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            failures.append(f"{path}: {len(old)} items -> {len(new)}")
            return failures, tolerated
        for i, (a, b) in enumerate(zip(old, new)):
            f, t = compare(a, b, ref_dir, new_dir, f"{path}[{i}]")
            failures += f
            tolerated += t
    elif _is_number(old) and _is_number(new):
        if old != new:
            ok = math.isclose(old, new, rel_tol=FLOAT_RTOL, abs_tol=1e-9)
            (tolerated if ok else failures).append(f"{path}: {old!r} -> {new!r}")
    elif old != new:
        failures.append(f"{path}: {json.dumps(old)[:150]} -> {json.dumps(new)[:150]}")
    return failures, tolerated


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _param(value):
    """A model parameter as JSON: numbers (and lists of them) stay numbers, so they compare
    within tolerance; anything else is pinned by its string form."""
    if _is_number(value) or isinstance(value, np.number):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)) and all(
        _is_number(v) or isinstance(v, np.number) for v in value
    ):
        return [float(v) for v in value]
    return str(value)


def run_one(flags, real=False, save_dir=None):
    """Run main() once in this process and summarize what it produced."""
    import bpy

    pin = ArrayPinner(save_dir)

    sys.path[:0] = [".", "examples"]
    import terrain_maker.terrain.core as core

    meshes = []
    create_mesh = core.Terrain.create_mesh

    def recording_create_mesh(self, *args, **kwargs):
        obj = create_mesh(self, *args, **kwargs)
        meshes.append(
            {
                "vertices": pin(self.vertices),
                "faces": pin(np.concatenate([np.asarray(f) for f in self.faces])),
                "colors": pin(self.colors),
                "boundary_colors": (
                    None if self.boundary_colors is None else pin(self.boundary_colors)
                ),
                "layers": {
                    name: pin(info.get("transformed_data", info["data"]))
                    for name, info in sorted(self.data_layers.items())
                },
                "model_params": {k: _param(v) for k, v in sorted(self.model_params.items())},
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
            [o.name, o.type, [float(v) for v in o.location]] for o in bpy.data.objects
        ),
        "files": {
            str(p.relative_to(out)): _file_pin(p, pin) for p in sorted(out.rglob("*")) if p.is_file()
        },
    }


def run_all(names, variants, real, save_dir):
    """Run each variant in a fresh process (main() mutates global matplotlib/Blender state)."""
    results = {}
    for name in names:
        proc = subprocess.run(
            [sys.executable, __file__, "_one_real" if real else "_one", json.dumps(variants[name]),
             "" if save_dir is None else str(save_dir)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            results[name] = {"error": proc.stderr.strip().splitlines()[-1][:300]}
        else:
            results[name] = json.loads(proc.stdout.strip().splitlines()[-1])
        print(f"  {name}: {'ERROR ' + results[name]['error'] if 'error' in results[name] else 'ran'}",
              file=sys.stderr)
    return results


def _array_shas(value):
    """Digests of the arrays a pin compares within tolerance, which need a saved reference."""
    if _is_array_pin(value):
        return {value["sha"]} if _has_tolerance(np.dtype(value["dtype"])) else set()
    children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    return set().union(*(_array_shas(v) for v in children))


def _load_pins(pins_path):
    pinned = json.loads(pins_path.read_text())
    if pinned.get("format") != FORMAT:
        raise SystemExit(
            f"{pins_path} is pin format {pinned.get('format', 1)}; this tool reads format {FORMAT}. "
            "Re-record it on a commit you trust: "
            f"uv run python tools/pin_combined_render.py record{' --real' if pins_path == REAL_PINS else ''}"
        )
    return pinned


def _record(pins_path, results, new_dir, real, every_variant):
    """Write pins; a partial record merges into the existing file, a full one replaces it."""
    pinned = _load_pins(pins_path) if pins_path.exists() and not every_variant else {"format": FORMAT}
    pinned["variants"] = {**pinned.get("variants", {}), **results}
    pins_path.parent.mkdir(parents=True, exist_ok=True)
    pins_path.write_text(json.dumps(pinned, indent=1, sort_keys=True) + "\n")
    if not real:
        ARRAYS.mkdir(parents=True, exist_ok=True)
        keep = _array_shas(pinned["variants"])
        for sha in keep:
            src = Path(new_dir) / f"{sha}.npz"
            if src.exists():
                shutil.copyfile(src, ARRAYS / f"{sha}.npz")
        missing = [sha for sha in keep if not (ARRAYS / f"{sha}.npz").exists()]
        if missing:
            raise SystemExit(f"Recorded pins reference arrays that were never saved: {missing[:5]}")
        for stale in ARRAYS.glob("*.npz"):
            if stale.stem not in keep:
                stale.unlink()


def main():
    args = sys.argv[1:]
    real = "--real" in args
    args = [a for a in args if a != "--real"]
    command, names = args[0], args[1:]
    if command in ("_one", "_one_real"):
        logging.disable(logging.CRITICAL)
        save_dir = names[1] or None
        print(json.dumps(run_one(json.loads(names[0]), real=command == "_one_real", save_dir=save_dir)))
        return 0

    variants, pins_path = (REAL_VARIANTS, REAL_PINS) if real else (VARIANTS, PINS)
    every_variant = not names
    names = names or list(variants)
    unknown = [n for n in names if n not in variants]
    if unknown:
        raise SystemExit(f"Unknown variants {unknown}; choose from {list(variants)}")
    # Real-data arrays are large and the baseline is per machine, so real pins are digests only
    # and must match exactly.
    with tempfile.TemporaryDirectory() as tmp:
        new_dir = None if real else tmp
        results = run_all(names, variants, real, new_dir)
        errors = [n for n in names if "error" in results[n]]

        if command == "record" or (real and not pins_path.exists()):
            if errors:
                print(f"Not recording: {', '.join(errors)} failed")
                return 1
            _record(pins_path, results, new_dir, real, every_variant)
            what = "Recorded" if command == "record" else "No baseline yet; recorded"
            print(f"{what} {len(results)} variants to {pins_path}")
            return 0

        pinned = _load_pins(pins_path)["variants"]
        changed = []
        for name in names:
            if name not in pinned:
                failures, tolerated = [f"no pin for {name}; record it first"], []
            else:
                failures, tolerated = compare(pinned[name], results[name], None if real else ARRAYS, new_dir)
            if failures:
                changed.append(name)
                print(f"CHANGED: {name}")
                for line in failures:
                    print(f"  {line}")
            if tolerated:
                print(f"{'  ' if failures else ''}WITHIN TOLERANCE: {name}")
                for line in tolerated:
                    print(f"  {line}")
    print(f"{len(names) - len(changed)}/{len(names)} variants unchanged")
    if changed and real:
        print("If these changes are intended, accept them with: "
              "uv run python tools/pin_combined_render.py record --real")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
