#!/usr/bin/env python3
"""
XC Skiing Temporal Analysis

Plots component scores and combined XC skiing suitability through the winter season,
comparing individual years against the multi-year median with interquartile bands.

Data: NOAA SNODAS daily snow depth (2010–2024)
Region: Great Lakes / Michigan region (lon -89 to -78, lat 37 to 47)

Daily scores computed per pixel, then aggregated:
  depth_score     – Mean of per-pixel depth scores (trapezoidal 100-400mm optimal)
  coverage_score  – Fraction of region with depth ≥ 50mm
  combined_score  – Weighted sum: (0.30*depth + 0.60*coverage) / 0.90
  regional_frac   – Fraction of region with depth ≥ 50mm

The plot shows four panels:
  1. Depth component score (0–1), one trace per year + multi-year median
  2. Coverage component score (0–1), one trace per year + multi-year median
  3. Combined XC skiing score (0–1), one trace per year + multi-year median
  4. Regional coverage (%), one trace per year + multi-year median

Results are cached so that the ~2,500-file scan only happens once.

Usage:
    python examples/xc_skiing_temporal.py
    python examples/xc_skiing_temporal.py --snodas-dir data/snodas_data
    python examples/xc_skiing_temporal.py --force-recompute
    python examples/xc_skiing_temporal.py --output docs/images/xc_skiing/temporal.png
"""

import argparse
import gzip
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Study area – matches DEM extent from detroit_xc_skiing.py
# ---------------------------------------------------------------------------
STUDY_BBOX = (-89.0, 37.0, -78.0, 47.0)  # (minx, miny, maxx, maxy) WGS84

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SNODAS_DIR = ROOT / "data" / "snodas_data"
DEFAULT_CACHE = ROOT / "examples" / "output" / "xc_skiing" / "temporal_timeseries.npz"
DEFAULT_OUTPUT = ROOT / "docs" / "images" / "xc_skiing" / "temporal_analysis.png"

# Winter season months (October = 10 through May = 5)
SEASON_MONTHS = frozenset({10, 11, 12, 1, 2, 3, 4, 5})

# Trapezoidal depth score thresholds (mm) – from src/scoring/configs/xc_skiing.py
RAMP_LOW = 50      # minimum usable depth
SWEET_LOW = 100    # ideal range starts
SWEET_HIGH = 400   # ideal range ends
RAMP_HIGH = 800    # maximum usable depth


# ===========================================================================
# SNODAS header & grid helpers
# ===========================================================================

def _read_header_gz(header_gz: Path) -> dict:
    """Parse SNODAS metadata directly from .txt.gz without writing to disk."""
    float_fields = {
        "data_slope", "data_intercept", "no_data_value",
        "minimum_x-axis_coordinate", "maximum_x-axis_coordinate",
        "minimum_y-axis_coordinate", "maximum_y-axis_coordinate",
        "x-axis_resolution", "y-axis_resolution",
    }
    int_fields = {"number_of_columns", "number_of_rows"}

    meta: dict = {}
    with gzip.open(header_gz, "rt") as f:
        for line in f:
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip().lower().replace(" ", "_")
            val = parts[1].strip()
            if val in ("Not applicable", ""):
                continue
            if key in float_fields:
                try:
                    meta[key] = float(val)
                except ValueError:
                    pass
            elif key in int_fields:
                try:
                    meta[key] = int(val)
                except ValueError:
                    pass

    meta.setdefault("number_of_columns", 8192)
    meta.setdefault("number_of_rows", 4096)
    meta["width"] = meta["number_of_columns"]
    meta["height"] = meta["number_of_rows"]
    return meta


def _bbox_to_slices(
    bbox: Tuple[float, float, float, float],
    meta: dict,
) -> Tuple[int, int, int, int]:
    """
    Convert a geographic bbox to row/col slices in the SNODAS native grid.

    SNODAS rows are stored top-to-bottom (row 0 = northernmost latitude).

    Returns: (row_start, row_end, col_start, col_end)
    """
    xmin_g = meta["minimum_x-axis_coordinate"]
    ymax_g = meta["maximum_y-axis_coordinate"]
    xres = meta.get("x-axis_resolution", 1 / 120)
    yres = meta.get("y-axis_resolution", 1 / 120)
    ncols = meta["width"]
    nrows = meta["height"]

    minx, miny, maxx, maxy = bbox

    col_start = max(0, int((minx - xmin_g) / xres))
    col_end = min(ncols, int((maxx - xmin_g) / xres) + 1)
    row_start = max(0, int((ymax_g - maxy) / yres))
    row_end = min(nrows, int((ymax_g - miny) / yres) + 1)

    return row_start, row_end, col_start, col_end


# ===========================================================================
# Fast partial-read of a single .dat[.gz] file
# ===========================================================================

def _study_area_stats(
    dat_gz: Path,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    meta: dict,
) -> Tuple[float, float, float, float, float]:
    """
    Read only the study-area rows and return five metrics:
      (mean_depth_mm, frac_above_50mm, frac_above_100mm, mean_depth_score, coverage_fraction)

    Prefers the uncompressed .dat sibling (direct seek, much faster).
    Falls back to streaming .dat.gz if no uncompressed copy exists.

    SNODAS snow-depth: slope=1, intercept=0 → raw int16 values are in mm.

    Returns (nan, nan, nan, nan, nan) on error or if all pixels are nodata.
    """
    ncols = meta["width"]
    slope = float(meta.get("data_slope", 1.0))
    intercept = float(meta.get("data_intercept", 0.0))
    nodata_int = int(round(meta.get("no_data_value", -9999)))
    nrows_crop = row_end - row_start
    bpp = 2  # bytes per int16

    nan5 = np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        dat_plain = dat_gz.with_suffix("")  # strip .gz → .dat
        n_bytes = nrows_crop * ncols * bpp

        if dat_plain.exists():
            with open(dat_plain, "rb") as f:
                f.seek(row_start * ncols * bpp)
                raw_bytes = f.read(n_bytes)
        else:
            with gzip.open(dat_gz, "rb") as fz:
                remaining = row_start * ncols * bpp
                chunk = 1 << 16  # 64 KB
                while remaining > 0:
                    buf = fz.read(min(chunk, remaining))
                    if not buf:
                        return nan5
                    remaining -= len(buf)
                raw_bytes = fz.read(n_bytes)

        if len(raw_bytes) < n_bytes:
            return nan5

        arr = np.frombuffer(raw_bytes, dtype=">i2").reshape(nrows_crop, ncols)
        crop = arr[:, col_start:col_end]

        valid = crop != nodata_int
        if not valid.any():
            return nan5

        values = crop[valid].astype(np.float32) * slope + intercept
        values = values[values >= 0]  # discard negative artefacts
        if len(values) == 0:
            return nan5

        mean_depth = float(np.mean(values))
        frac_50mm  = float(np.mean(values >= RAMP_LOW))
        frac_100mm = float(np.mean(values >= SWEET_LOW))

        # Compute per-pixel depth component score, then take mean
        # (Mean gives a smoother, more representative score than median)
        depth_scores = _trapezoidal_score(values)
        median_depth_score = float(np.mean(depth_scores))

        # Compute regional coverage: fraction of pixels with depth ≥ 50mm
        # This gives a continuous 0-1 metric representing how much of the region has skiing-quality snow
        median_coverage_score = float(np.mean(values >= RAMP_LOW))

        return mean_depth, frac_50mm, frac_100mm, median_depth_score, median_coverage_score

    except Exception as exc:
        logger.debug("Error reading %s: %s", dat_gz, exc)
        return nan5


def _process_batch(
    jobs: List[Tuple[str, "date", Path]],
    row_start: int, row_end: int,
    col_start: int, col_end: int,
    meta: dict,
) -> dict:
    """Process a batch of SNODAS files sequentially.  Called once per worker."""
    results = {}
    for season, d, dat_path in jobs:
        stats = _study_area_stats(dat_path, row_start, row_end, col_start, col_end, meta)
        results[(season, d)] = stats
    return results


# ===========================================================================
# File discovery
# ===========================================================================

def _find_winter_files(
    snodas_dir: Path,
) -> Dict[str, List[Tuple[date, Path]]]:
    """
    Discover all SNODAS .dat.gz files for winter months (Oct–May).

    Returns a dict keyed by season string, e.g. "2020-2021", where each value
    is a chronologically sorted list of (date, Path) tuples.
    """
    seasons: Dict[str, List[Tuple[date, Path]]] = {}

    # Resolve symlinks so rglob traverses into symlinked directories
    snodas_dir = snodas_dir.resolve()

    for dat in sorted(snodas_dir.rglob("snow_depth_*.dat.gz")):
        # dat.name = "snow_depth_YYYYMMDD.dat.gz"; strip both extensions
        date_str = dat.name.replace("snow_depth_", "").replace(".dat.gz", "")
        if len(date_str) != 8:
            continue
        try:
            d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        except ValueError:
            continue

        if d.month not in SEASON_MONTHS:
            continue

        season_key = (
            f"{d.year}-{d.year + 1}" if d.month >= 10
            else f"{d.year - 1}-{d.year}"
        )
        seasons.setdefault(season_key, []).append((d, dat))

    for key in seasons:
        seasons[key].sort()

    return seasons


def _days_since_oct1(d: date) -> int:
    """Number of days since October 1 of the current winter season."""
    if d.month >= 10:
        oct1 = date(d.year, 10, 1)
    else:
        oct1 = date(d.year - 1, 10, 1)
    return (d - oct1).days


# ===========================================================================
# Timeseries computation and caching
# ===========================================================================

def _save_cache(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seasons = data["seasons"]
    arrays: dict = {"seasons": np.array(seasons)}
    for s in seasons:
        k = s.replace("-", "_")
        arrays[f"dates_{k}"]  = np.array(
            [d.toordinal() for d in data["dates"][s]], dtype=np.int32
        )
        arrays[f"depths_{k}"] = data["depths"][s]
        arrays[f"frac50_{k}"] = data["frac50"][s]
        arrays[f"frac100_{k}"] = data["frac100"][s]
        arrays[f"median_depth_score_{k}"] = data["median_depth_score"][s]
        arrays[f"median_coverage_score_{k}"] = data["median_coverage_score"][s]
        arrays[f"doy_{k}"]    = data["doy"][s]
    np.savez_compressed(path, **arrays)


def _load_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as npz:
        seasons = [str(s) for s in npz["seasons"]]
        out: dict = {
            "seasons": seasons,
            "dates": {}, "depths": {}, "frac50": {}, "frac100": {},
            "median_depth_score": {}, "median_coverage_score": {}, "doy": {},
        }
        for s in seasons:
            k = s.replace("-", "_")
            out["dates"][s]   = [date.fromordinal(int(o)) for o in npz[f"dates_{k}"]]
            out["depths"][s]  = npz[f"depths_{k}"]
            out["frac50"][s]  = npz[f"frac50_{k}"]
            out["frac100"][s] = npz[f"frac100_{k}"]
            out["median_depth_score"][s] = npz[f"median_depth_score_{k}"]
            out["median_coverage_score"][s] = npz[f"median_coverage_score_{k}"]
            out["doy"][s]     = npz[f"doy_{k}"]
    return out


def build_timeseries(
    snodas_dir: Path,
    cache_file: Path,
    force: bool = False,
) -> dict:
    """
    Build (or load) the day-by-day snow stats timeseries per season.

    Returns a dict with keys:
        seasons  – sorted list of season strings
        dates    – {season: [date, ...]}
        depths   – {season: float32 array of area-mean depths in mm}
        frac50   – {season: float32 array of fraction of pixels with depth≥50mm}
        doy      – {season: int32 array of day-of-season (0 = Oct 1)}
    """
    if not force and cache_file.exists():
        logger.info("Loading cached timeseries: %s", cache_file)
        try:
            return _load_cache(cache_file)
        except Exception as e:
            logger.warning(
                "Cache at %s is incompatible with the current format (%s). "
                "Delete it or pass --force-recompute to rebuild.",
                cache_file, e,
            )
            raise SystemExit(1)

    logger.info("Scanning SNODAS files in %s …", snodas_dir)
    seasons_files = _find_winter_files(snodas_dir)
    if not seasons_files:
        raise FileNotFoundError(f"No winter SNODAS .dat.gz files found in {snodas_dir}")

    # Read grid info from the first available header (grid is fixed every day)
    first_season = sorted(seasons_files)[0]
    _, first_dat = seasons_files[first_season][0]
    date_str = first_dat.name.replace("snow_depth_", "").replace(".dat.gz", "")
    ref_header = first_dat.parent / f"snow_depth_metadata_{date_str}.txt.gz"
    if not ref_header.exists():
        raise FileNotFoundError(f"Reference header not found: {ref_header}")

    logger.info("Reading reference header: %s", ref_header)
    meta = _read_header_gz(ref_header)

    row_start, row_end, col_start, col_end = _bbox_to_slices(STUDY_BBOX, meta)
    n_pixels = (row_end - row_start) * (col_end - col_start)
    logger.info(
        "Study-area slice: rows %d:%d, cols %d:%d  (%s pixels)",
        row_start, row_end, col_start, col_end, f"{n_pixels:,}",
    )

    result: dict = {
        "seasons": [], "dates": {}, "depths": {}, "frac50": {}, "frac100": {},
        "median_depth_score": {}, "median_coverage_score": {}, "doy": {},
    }

    # Flatten all (season, date, path) for parallel batch processing
    all_jobs: List[Tuple[str, date, Path]] = []
    for season in sorted(seasons_files):
        for d, dat_path in seasons_files[season]:
            all_jobs.append((season, d, dat_path))

    # Shuffle so each batch gets a mix of large (winter) and small (shoulder)
    # files — avoids one batch getting all the 12 MB Dec–Mar files
    import random
    random.Random(42).shuffle(all_jobs)

    total = len(all_jobs)
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = min(os.cpu_count() or 1, 4)
    n_batches = n_workers * 8  # 32 small batches → smooth progress
    logger.info(
        "Reading %d SNODAS files with %d workers (%d batches)",
        total, n_workers, n_batches,
    )

    def _chunked(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    chunks = _chunked(all_jobs, n_batches)

    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=total, desc="Reading SNODAS", unit="file")
    except ImportError:
        class _NullPbar:
            def update(self, n=1): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        pbar = _NullPbar()

    with pbar, ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                _process_batch, chunk,
                row_start, row_end, col_start, col_end, meta,
            )
            for chunk in chunks
        ]
        stats_by_key: dict = {}
        for future in as_completed(futures):
            batch_results = future.result()
            stats_by_key.update(batch_results)
            pbar.update(len(batch_results))

    # Reassemble results by season (sorted by date within each season)
    for season in sorted(seasons_files):
        dates_s = [d for d, _ in seasons_files[season]]
        stats = [stats_by_key[(season, d)] for d in dates_s]

        result["seasons"].append(season)
        result["dates"][season] = dates_s
        result["depths"][season] = np.array([s[0] for s in stats], dtype=np.float32)
        result["frac50"][season] = np.array([s[1] for s in stats], dtype=np.float32)
        result["frac100"][season] = np.array([s[2] for s in stats], dtype=np.float32)
        result["median_depth_score"][season] = np.array([s[3] for s in stats], dtype=np.float32)
        result["median_coverage_score"][season] = np.array([s[4] for s in stats], dtype=np.float32)
        result["doy"][season] = np.array(
            [_days_since_oct1(d) for d in dates_s], dtype=np.int32
        )

    _save_cache(result, cache_file)
    logger.info("Timeseries cached → %s", cache_file)
    return result


# ===========================================================================
# Score helper
# ===========================================================================

def _trapezoidal_score(depth_mm: np.ndarray) -> np.ndarray:
    """Depth-component trapezoidal score (0–1), matching xc_skiing.py config."""
    d = np.asarray(depth_mm, dtype=np.float32)
    s = np.zeros_like(d)
    s = np.where((d >= RAMP_LOW)   & (d < SWEET_LOW),
                 (d - RAMP_LOW) / (SWEET_LOW - RAMP_LOW), s)
    s = np.where((d >= SWEET_LOW)  & (d <= SWEET_HIGH), 1.0, s)
    s = np.where((d > SWEET_HIGH)  & (d <= RAMP_HIGH),
                 1.0 - (d - SWEET_HIGH) / (RAMP_HIGH - SWEET_HIGH), s)
    return s


# ===========================================================================
# Plotting
# ===========================================================================

def _interp_season(doy: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a season's daily values onto a common doy grid."""
    valid = ~np.isnan(values)
    if valid.sum() < 5:
        return np.full(len(grid), np.nan)
    return np.interp(grid, doy[valid].astype(float), values[valid], left=np.nan, right=np.nan)


def _band(mat: np.ndarray):
    """Return (p50, p25, p75) along axis=0, ignoring NaNs."""
    return (
        np.nanpercentile(mat, 50, axis=0),
        np.nanpercentile(mat, 25, axis=0),
        np.nanpercentile(mat, 75, axis=0),
    )


# ---------------------------------------------------------------------------
# Helpers for 3D temporal landscape sculpture
# ---------------------------------------------------------------------------


def build_combined_matrix(
    data: dict, n_days: int = 243
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the (n_seasons x n_days) XC skiing combined score matrix + summary.

    Parameters
    ----------
    data : dict
        Output of ``build_timeseries()``.  Must contain keys ``"seasons"``,
        ``"doy"``, ``"median_depth_score"``, ``"median_coverage_score"``.
    n_days : int
        Length of the common day-of-season grid (default 243, Oct 1 → May 31).

    Returns
    -------
    season_matrix : np.ndarray
        float32 (n_seasons, n_days).  NaN replaced with 0.
    seasons : list[str]
        Season labels sorted chronologically.
    median_scores : np.ndarray
        float32 (n_days,).  Median (50th percentile) across seasons.
    q1_scores : np.ndarray
        float32 (n_days,).  25th percentile across seasons.
    q3_scores : np.ndarray
        float32 (n_days,).  75th percentile across seasons.
    """
    seasons = sorted(data["seasons"])
    doy_grid = np.arange(0, n_days)
    season_matrix = np.full((len(seasons), n_days), np.nan, dtype=np.float32)

    for i, s in enumerate(seasons):
        doy = data["doy"][s].astype(float)
        depth = _interp_season(doy, data["median_depth_score"][s], doy_grid)
        coverage = _interp_season(doy, data["median_coverage_score"][s], doy_grid)
        season_matrix[i] = np.clip(
            (0.30 * depth + 0.60 * coverage) / 0.90, 0, 1
        )

    median_scores = np.nanmedian(season_matrix, axis=0)
    q1_scores = np.nanpercentile(season_matrix, 25, axis=0).astype(np.float32)
    q3_scores = np.nanpercentile(season_matrix, 75, axis=0).astype(np.float32)

    # Replace NaN with 0 (missing data = no snow)
    np.nan_to_num(season_matrix, nan=0.0, copy=False)
    np.nan_to_num(median_scores, nan=0.0, copy=False)
    np.nan_to_num(q1_scores, nan=0.0, copy=False)
    np.nan_to_num(q3_scores, nan=0.0, copy=False)

    return season_matrix, seasons, median_scores, q1_scores, q3_scores


def build_ridge_dem(
    season_matrix: np.ndarray,
    median_scores: np.ndarray,
    q1_scores: np.ndarray | None = None,
    q3_scores: np.ndarray | None = None,
    ridge_rows: int = 7,
    gap_rows: int = 3,
    median_width_mult: int = 2,
    median_height_scale: float = 1.5,
    tail_rows: int = 0,
    shear: float = 0.0,
    smooth_sigma: float | None = None,
    summary_gap_rows: int = 6,
    row_scale: int = 1,
    summary_color_scale: float = 1.0,
    col_scale: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a composite ridge DEM and color array for the temporal sculpture.

    When ``q1_scores`` and ``q3_scores`` are provided, uses a **two-group
    layout**: all individual seasons in one group, then a gap, then a
    summary group of Q1 / median / Q3.  Otherwise falls back to the legacy
    layout with the median embedded among the seasons.

    When ``shear>0``, each column shifts northward proportional to its score,
    creating an emergent cuesta shape via differential shifting.

    Parameters
    ----------
    season_matrix : np.ndarray
        (n_seasons, n_days) score matrix from ``build_combined_matrix()``.
    median_scores : np.ndarray
        (n_days,) median scores from ``build_combined_matrix()``.
    q1_scores : np.ndarray or None
        (n_days,) 25th-percentile scores.  When provided together with
        ``q3_scores``, enables the two-group layout.
    q3_scores : np.ndarray or None
        (n_days,) 75th-percentile scores.
    ridge_rows : int
        Number of rows for the core rise of each ridge cross-section.
    gap_rows : int
        Number of zero-height rows between ridges (before tail overlap).
    median_width_mult : int
        Multiplier for the median ridge width (in ridge_rows).
    median_height_scale : float
        Multiplier for median ridge Z height relative to regular ridges.
    tail_rows : int
        Extra rows of exponential decay tail per ridge.  0 gives the
        original symmetric hogback; >0 gives cascading asymmetric cuestas.
    shear : float
        Maximum northward shift (in rows) for the highest-score columns.
        Each column shifts north by ``(score / global_max) * shear`` rows.
        0 means no shift; 6 means the global-max column shifts 6 rows north.
    smooth_sigma : float or None
        Gaussian smoothing sigma along the N-S axis after ridge construction.
        ``None`` (default) means no smoothing — float-precision shifts produce
        inherently smooth ramps.  Set to a positive value for extra smoothing
        if desired (e.g. ``1.0``).
    summary_gap_rows : int
        Number of zero-height rows separating the season group from the
        summary group (two-group layout only).
    row_scale : int
        Multiplier for all row-related dimensions (ridge_rows, gap_rows,
        shear, summary_gap_rows).  Higher values give finer Y resolution
        so float scores are not quantized to coarse integer row positions.
        1 (default) keeps original resolution; 8 gives 8× finer grid.
    summary_color_scale : float
        Multiplier for the IQR summary ridge color values.  Values > 1
        make the summary ridge more vivid relative to per-season ridges.
        1.0 (default) = no boost.
    col_scale : int
        Column (E-W / day-of-season) upscaling factor.  Score arrays are
        linearly interpolated along the day axis *before* building the
        ridge DEM, so every intermediate column gets its own properly
        computed ramp profile.  This preserves line-graph fidelity (no
        peak-sag from interpolating 2D profiles with different shear
        offsets).  1 (default) = no upsampling.

    Returns
    -------
    dem_ridges : np.ndarray
        float32 (total_rows, n_days) — Z height = score * profile envelope.
    color_ridges : np.ndarray
        float32 (total_rows, n_days) — flat score for colormap (no bell).
    """
    n_seasons, n_days = season_matrix.shape
    has_summary = q1_scores is not None and q3_scores is not None

    # Upsample score arrays along day axis BEFORE building ridges.
    # This ensures every intermediate column gets its own ramp profile
    # computed from the interpolated score, preserving line-graph peaks.
    # Uses np.interp with exact placement: original day i → column i*col_scale.
    if col_scale > 1:
        x_old = np.arange(n_days)
        n_out = (n_days - 1) * col_scale + 1
        x_new = np.arange(n_out) / col_scale  # 0, 1/cs, 2/cs, ..., n_days-1
        upsampled = np.zeros((n_seasons, n_out), dtype=np.float32)
        for i in range(n_seasons):
            upsampled[i] = np.interp(x_new, x_old, season_matrix[i])
        season_matrix = upsampled
        median_scores = np.interp(x_new, x_old, median_scores).astype(np.float32)
        if q1_scores is not None:
            q1_scores = np.interp(x_new, x_old, q1_scores).astype(np.float32)
        if q3_scores is not None:
            q3_scores = np.interp(x_new, x_old, q3_scores).astype(np.float32)
        n_days = n_out

    # Scale all row dimensions for finer Y resolution
    if row_scale > 1:
        ridge_rows *= row_scale
        gap_rows *= row_scale
        shear *= row_scale
        summary_gap_rows *= row_scale

    if tail_rows > 0:
        return _build_cuesta_dem(
            season_matrix, median_scores, q1_scores, q3_scores,
            ridge_rows, gap_rows,
            median_width_mult, median_height_scale,
            shear, smooth_sigma, summary_gap_rows,
            summary_color_scale,
        )

    # ----- Legacy hogback path (tail_rows=0, no shear) -----
    median_ridge_rows = ridge_rows * median_width_mult
    first_half = n_seasons // 2
    base_rows = (
        n_seasons * (ridge_rows + gap_rows)
        + median_width_mult * ridge_rows
        + gap_rows
    )
    total_rows = base_rows
    dem_ridges = np.zeros((total_rows, n_days), dtype=np.float32)
    color_ridges = np.zeros((total_rows, n_days), dtype=np.float32)

    def _hogback(n):
        mid = (n - 1) / 2.0
        return np.clip(
            1.0 - np.abs(np.arange(n) - mid) / mid, 0, 1,
        ) if mid > 0 else np.ones(n)

    profile_season = _hogback(ridge_rows)
    profile_median = _hogback(median_ridge_rows)
    row = 0

    def _write_ridge_hog(start_row, score_row, profile, height_scale=1.0):
        n = len(profile)
        for dy in range(n):
            r = start_row + dy
            if 0 <= r < total_rows:
                dem_ridges[r] += score_row * profile[dy] * height_scale
                color_ridges[r] = score_row * profile[dy]

    for i in range(first_half):
        _write_ridge_hog(row, season_matrix[i], profile_season)
        row += ridge_rows + gap_rows

    _write_ridge_hog(row, median_scores, profile_median, median_height_scale)
    row += median_ridge_rows + gap_rows

    for i in range(first_half, n_seasons):
        _write_ridge_hog(row, season_matrix[i], profile_season)
        row += ridge_rows + gap_rows

    return dem_ridges, color_ridges


def _build_cuesta_dem(
    season_matrix: np.ndarray,
    median_scores: np.ndarray,
    q1_scores: np.ndarray | None,
    q3_scores: np.ndarray | None,
    ridge_rows: int,
    gap_rows: int,
    median_width_mult: int,
    median_height_scale: float,
    shear: float,
    smooth_sigma: float | None,
    summary_gap_rows: int,
    summary_color_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Emergent cuesta path (tail_rows>0).  Extracted for clarity."""
    n_seasons, n_days = season_matrix.shape
    has_summary = q1_scores is not None and q3_scores is not None
    shear_pad = int(np.ceil(shear)) if shear > 0 else 0

    if has_summary:
        # Two-group layout (DEM row order, before the [::-1] flip):
        #   [shear_pad] [summary: single IQR ridge] [summary_gap + shear_pad] [seasons]
        #
        # The IQR ridge uses median for height/color; Q1/Q3 define the
        # N-S footprint (where the ridge contacts the background).
        #
        # After the [::-1] row-flip in temporal_sculpture.py:
        #   scene-north (near main terrain): all seasons
        #   scene-south (below):             IQR summary ridge
        summary_block = ridge_rows + gap_rows  # single ridge slot
        season_block = n_seasons * (ridge_rows + gap_rows)
        total_rows = shear_pad + summary_block + summary_gap_rows + shear_pad + season_block
    else:
        # Legacy embedded-median layout
        median_ridge_rows = ridge_rows * median_width_mult
        base_rows = (
            n_seasons * (ridge_rows + gap_rows)
            + median_width_mult * ridge_rows
            + gap_rows
        )
        total_rows = shear_pad + base_rows

    dem_ridges = np.zeros((total_rows, n_days), dtype=np.float32)
    color_ridges = np.zeros((total_rows, n_days), dtype=np.float32)
    # Track per-cell max z contribution for height-based color ownership.
    # When ridges overlap (tall peaks leaning into adjacent territory),
    # whichever ridge contributes the MOST height at a cell owns its color.
    _max_z_grid = np.zeros((total_rows, n_days), dtype=np.float32)

    # Global max across all data for consistent shear normalization
    all_maxes = [float(np.max(season_matrix)), float(np.max(median_scores))]
    if has_summary:
        all_maxes.extend([float(np.max(q1_scores)), float(np.max(q3_scores))])
    global_max = max(all_maxes)

    def _write_ridge(nom_row, score_row, height_scale=1.0):
        """Write one cuesta ridge with float-precision shear."""
        normed = (score_row / global_max) if global_max > 0 else score_row
        shifts_float = normed * shear
        safe_shifts = np.maximum(shifts_float, 1e-6)
        max_shift = int(np.ceil(shear))

        for k in range(max_shift + 1):
            r = nom_row - k
            if r < 0 or r >= total_rows:
                continue
            active_w = np.clip(shifts_float - k + 1, 0, 1)
            active = active_w > 0
            frac = np.minimum(k / safe_shifts, 1.0) * active_w
            z_vals = (score_row * frac * height_scale).astype(np.float32)
            new_z = np.where(active, z_vals, 0).astype(np.float32)
            # Max (not sum): each cuesta keeps its own consistent slope profile.
            # Overlapping tails don't artificially inflate height.
            dem_ridges[r] = np.maximum(dem_ridges[r], new_z)
            owns = new_z > _max_z_grid[r]
            _max_z_grid[r] = np.maximum(_max_z_grid[r], new_z)
            color_val = (score_row * frac).astype(np.float32)
            color_ridges[r] = np.where(owns, color_val, color_ridges[r])

    def _write_iqr_ridge(nom_row):
        """Write a single IQR-envelope summary ridge.

        Height and color driven by median_scores.  The N-S footprint
        (where the ridge contacts the background) is set by Q1 (south
        contact) and Q3 (north contact).  Between Q1 and median the
        surface ramps up; between median and Q3 it ramps back down.
        """
        q1_shifts = (q1_scores / global_max * shear) if global_max > 0 else np.zeros(n_days)
        med_shifts = (median_scores / global_max * shear) if global_max > 0 else np.zeros(n_days)
        q3_shifts = (q3_scores / global_max * shear) if global_max > 0 else np.zeros(n_days)

        # Ensure ordering even with noisy quantile estimates
        q1_shifts = np.minimum(q1_shifts, med_shifts)
        q3_shifts = np.maximum(q3_shifts, med_shifts)

        south_range = np.maximum(med_shifts - q1_shifts, 1e-6)
        north_range = np.maximum(q3_shifts - med_shifts, 1e-6)

        max_k = int(np.ceil(shear)) + 1

        for k in range(max_k):
            r = nom_row - k
            if r < 0 or r >= total_rows:
                continue

            kf = float(k)

            # Ramp fraction: 0 at Q1/Q3 contacts, 1 at median peak
            south_frac = np.clip((kf - q1_shifts) / south_range, 0, 1)
            north_frac = np.clip(1.0 - (kf - med_shifts) / north_range, 0, 1)
            below_med = kf <= med_shifts
            frac = np.where(below_med, south_frac, north_frac)

            # Soft sub-pixel edges at Q1 and Q3 boundaries
            active_south = np.clip(kf - q1_shifts + 1, 0, 1)
            active_north = np.clip(q3_shifts - kf + 1, 0, 1)
            active_w = np.minimum(active_south, active_north)
            active = active_w > 0

            z_vals = (median_scores * frac * active_w).astype(np.float32)
            new_z = np.where(active, z_vals, 0).astype(np.float32)
            dem_ridges[r] = np.maximum(dem_ridges[r], new_z)
            owns = new_z > _max_z_grid[r]
            _max_z_grid[r] = np.maximum(_max_z_grid[r], new_z)
            color_val = (median_scores * frac * active_w * summary_color_scale).astype(np.float32)
            color_ridges[r] = np.where(
                owns, color_val, color_ridges[r],
            )

    if has_summary:
        # --- Group 1: single IQR summary ridge at north end of DEM ---
        row = shear_pad
        _write_iqr_ridge(row)
        row += ridge_rows + gap_rows

        # --- Gap + shear headroom for seasons ---
        row += summary_gap_rows + shear_pad - gap_rows

        # --- Group 2: all seasons (oldest → newest) ---
        for i in range(n_seasons):
            _write_ridge(row, season_matrix[i])
            row += ridge_rows + gap_rows
    else:
        # Legacy layout: first_half + median + second_half
        first_half = n_seasons // 2
        row = shear_pad

        for i in range(first_half):
            _write_ridge(row, season_matrix[i])
            row += ridge_rows + gap_rows

        median_ridge_rows = ridge_rows * median_width_mult
        _write_ridge(row, median_scores, median_height_scale)
        row += median_ridge_rows + gap_rows

        for i in range(first_half, n_seasons):
            _write_ridge(row, season_matrix[i])
            row += ridge_rows + gap_rows

    # Optional smoothing
    if smooth_sigma is not None and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        dem_ridges = gaussian_filter1d(dem_ridges, sigma=smooth_sigma, axis=0)
        color_ridges = gaussian_filter1d(color_ridges, sigma=smooth_sigma, axis=0)

    return dem_ridges, color_ridges


def plot_temporal(
    data: dict,
    output: Optional[Path] = None,
) -> None:
    """
    Four-panel temporal figure showing component and combined scores (50th-percentile
    central line + IQR band):
      Panel 1 – Median depth component score (0–1) across region
      Panel 2 – Median coverage score (0–1) across region
      Panel 3 – Combined XC skiing score (0–1) from components
      Panel 4 – Regional coverage with skiing-quality snow (%)

    Individual years coloured by season start year (plasma: older=dark,
    newer=bright).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    seasons = sorted(data["seasons"])
    n = len(seasons)

    # Interpolate every season onto a 243-day common grid (day 0=Oct 1)
    doy_grid = np.arange(0, 243)

    cmap = cm.plasma
    colors = {s: cmap(i / max(1, n - 1)) for i, s in enumerate(seasons)}

    median_depth_score_interp: Dict[str, np.ndarray] = {}
    median_coverage_score_interp: Dict[str, np.ndarray] = {}
    frac50_interp: Dict[str, np.ndarray] = {}

    for s in seasons:
        doy = data["doy"][s].astype(float)
        median_depth_score_interp[s] = _interp_season(
            doy, data["median_depth_score"][s], doy_grid
        )
        median_coverage_score_interp[s] = _interp_season(
            doy, data["median_coverage_score"][s], doy_grid
        )
        frac50_interp[s] = _interp_season(doy, data["frac50"][s], doy_grid)

    valid_seasons = sorted(median_depth_score_interp)
    depth_score_mat = np.array([median_depth_score_interp[s] for s in valid_seasons])
    coverage_score_mat = np.array([median_coverage_score_interp[s] for s in valid_seasons])
    frac50_mat = np.array([frac50_interp[s] for s in valid_seasons])

    # Combined XC skiing score: weighted sum of depth (30%) and coverage (60%)
    # Normalized to 0-1: (0.30*depth + 0.60*coverage) / 0.90
    combined_mat = (0.30 * depth_score_mat + 0.60 * coverage_score_mat) / 0.90
    combined_mat = np.clip(combined_mat, 0, 1)  # clip to 0-1 range

    p50_ds, p25_ds, p75_ds = _band(depth_score_mat)
    p50_cs, p25_cs, p75_cs = _band(coverage_score_mat)
    p50_comb, p25_comb, p75_comb = _band(combined_mat)
    p50_f50, p25_f50, p75_f50 = _band(frac50_mat)

    # Month tick positions (non-leap reference year)
    month_info = [
        (date(2023, 10, 1), "Oct"), (date(2023, 11, 1), "Nov"),
        (date(2023, 12, 1), "Dec"), (date(2024, 1, 1), "Jan"),
        (date(2024, 2, 1), "Feb"),  (date(2024, 3, 1), "Mar"),
        (date(2024, 4, 1), "Apr"),  (date(2024, 5, 1), "May"),
    ]
    tick_pos = [_days_since_oct1(d) for d, _ in month_info]
    tick_lab = [lbl for _, lbl in month_info]

    # ---- Figure ---------------------------------------------------------------
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(13, 14), sharex=True, constrained_layout=True
    )
    fig.suptitle(
        f"XC Skiing Score Components – Great Lakes Region\n"
        f"Median scores from SNODAS daily snow depth  ({seasons[0]} → {seasons[-1]})",
        fontsize=13, fontweight="bold",
    )

    line_kw = dict(linewidth=0.85, alpha=0.30, zorder=1)
    band_kw = dict(alpha=0.22, color="#2457a0", zorder=2)
    med_kw = dict(color="#0d2b70", linewidth=2.5, zorder=3, label="Median (50th pct)")

    # ---- Panel 1: Depth component score ----------------------------------------
    for s in valid_seasons:
        ax1.plot(doy_grid, median_depth_score_interp[s], color=colors[s], **line_kw)
    ax1.fill_between(doy_grid, p25_ds, p75_ds, label="IQR (25–75th pct)", **band_kw)
    ax1.plot(doy_grid, p50_ds, **med_kw)

    ax1.axhline(1.0, color="#165016", linestyle=":", linewidth=1.1, alpha=0.7)
    ax1.set_ylabel("Depth component score (0–1)", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # ---- Panel 2: Coverage component score (median coverage across pixels) ------
    for s in valid_seasons:
        ax2.plot(doy_grid, median_coverage_score_interp[s], color=colors[s], **line_kw)
    ax2.fill_between(doy_grid, p25_cs, p75_cs, **band_kw)
    ax2.plot(doy_grid, p50_cs, **med_kw)

    ax2.axhline(1.0, color="#165016", linestyle=":", linewidth=1.1, alpha=0.7)
    ax2.set_ylabel("Coverage component score (0–1)", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # ---- Panel 3: Combined XC skiing score -------------------------------------
    for s in valid_seasons:
        combined_season = (
            0.30 * median_depth_score_interp[s] + 0.60 * median_coverage_score_interp[s]
        ) / 0.90
        combined_season = np.clip(combined_season, 0, 1)
        ax3.plot(doy_grid, combined_season, color=colors[s], **line_kw)
    ax3.fill_between(doy_grid, p25_comb, p75_comb, **band_kw)
    ax3.plot(doy_grid, p50_comb, **med_kw)

    ax3.axhline(1.0, color="#165016", linestyle=":", linewidth=1.1, alpha=0.7)
    ax3.set_ylabel("XC skiing score (0–1)", fontsize=10)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
    ax3.grid(True, alpha=0.3, linestyle="--")

    # ---- Panel 4: Regional coverage at 50mm threshold ---------------------------
    for s in valid_seasons:
        ax4.plot(doy_grid, frac50_interp[s] * 100, color=colors[s], **line_kw)

    ax4.fill_between(doy_grid, p25_f50 * 100, p75_f50 * 100,
                     label=f"IQR ≥{RAMP_LOW} mm", **band_kw)
    ax4.plot(doy_grid, p50_f50 * 100, color="#0d2b70", linewidth=2.5,
             zorder=3, label=f"Median ≥{RAMP_LOW} mm")

    ax4.set_ylabel("Region with skiing-quality snow (%)", fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
    ax4.grid(True, alpha=0.3, linestyle="--")

    ax4.set_xticks(tick_pos)
    ax4.set_xticklabels(tick_lab, fontsize=10)
    ax4.set_xlim(0, 242)

    # ---- Colorbar ---------------------------------------------------------------
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(int(seasons[0][:4]), int(seasons[-1][:4])),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3, ax4], orientation="vertical",
                        fraction=0.015, pad=0.02, shrink=0.5)
    cbar.set_label("Season start year", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", output)
    else:
        plt.show()

    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================

def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="XC skiing temporal analysis – snow depth by season",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--snodas-dir", type=Path, default=DEFAULT_SNODAS_DIR,
        help="Directory containing SNODAS .dat.gz files (default: data/snodas_data)",
    )
    p.add_argument(
        "--cache-file", type=Path, default=DEFAULT_CACHE,
        help="Path to .npz cache file (created on first run)",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Save figure to this path instead of showing interactively",
    )
    p.add_argument(
        "--force-recompute", action="store_true",
        help="Ignore cache and re-read all SNODAS files",
    )
    args = p.parse_args(argv)

    data = build_timeseries(
        snodas_dir=args.snodas_dir,
        cache_file=args.cache_file,
        force=args.force_recompute,
    )

    n_seasons = len(data["seasons"])
    n_days = sum(len(v) for v in data["depths"].values())
    logger.info("Plotting %d seasons, %d day-records", n_seasons, n_days)

    plot_temporal(data, output=args.output)


if __name__ == "__main__":
    main()
