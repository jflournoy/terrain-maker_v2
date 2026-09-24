"""
Characterization tests for compute_flow_with_basins.

Pins the current outputs on a small synthetic landscape (slope to an ocean
edge, a closed bowl, a lake, precipitation) so the two historical copies of
this function can be merged without changing behavior.
"""

import hashlib

import numpy as np
import pytest
from rasterio import Affine

from src.terrain import flow_accumulation, flow_pipeline

SHAPE = (60, 60)
TRANSFORM = Affine(0.001, 0, -117.0, 0, -0.001, 33.0)


def _landscape():
    """Deterministic DEM, lake mask/outlets, and precipitation."""
    rng = np.random.default_rng(42)
    rows, cols = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]

    # Slope falling south toward an ocean strip along the last rows
    dem = 120.0 - 2.0 * rows + rng.normal(0, 0.3, SHAPE)
    dem[57:, :] = -1.0

    # Closed bowl (endorheic basin) on the west side
    bowl = (rows - 25) ** 2 + (cols - 12) ** 2 < 36
    dem[bowl] -= 15.0

    # Flat lake in the east with an outlet on its south edge
    lake_mask = np.zeros(SHAPE, dtype=np.uint8)
    lake_mask[18:24, 38:46] = 1
    dem[lake_mask == 1] = dem[23, 38:46].min()
    lake_outlets = np.zeros(SHAPE, dtype=bool)
    lake_outlets[23, 42] = True

    precipitation = (500.0 + 5.0 * cols).astype(np.float32)
    return dem.astype(np.float32), lake_mask, lake_outlets, precipitation


def _run(func, config):
    dem, lake_mask, lake_outlets, precipitation = _landscape()
    kwargs = dict(
        detect_basins=True,
        min_basin_size=50,
        min_basin_depth=1.0,
        verbose=False,
    )
    if config in ("lakes_precip", "coarse_precip"):
        kwargs.update(lake_mask=lake_mask, lake_outlets=lake_outlets)
    if config == "lakes_precip":
        kwargs.update(precipitation=precipitation, precip_transform=TRANSFORM)
    if config == "coarse_precip":
        coarse = precipitation[::2, ::2].copy()
        kwargs.update(
            precipitation=coarse,
            precip_transform=TRANSFORM * Affine.scale(2),
            upscale_precip=True,
            upscale_factor=2,
            upscale_method="bicubic",
        )
    return func(dem, TRANSFORM, **kwargs)


def _digest(arr):
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


CONFIGS = ["dem_only", "lakes_precip", "coarse_precip"]


@pytest.mark.parametrize("config", CONFIGS)
def test_both_copies_agree(config):
    """flow_pipeline and flow_accumulation give identical results."""
    a = _run(flow_pipeline.compute_flow_with_basins, config)
    b = _run(flow_accumulation.compute_flow_with_basins, config)

    assert a.keys() == b.keys()
    for key in a:
        if a[key] is None or b[key] is None:
            assert a[key] is None and b[key] is None, key
        else:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


# Summary of current outputs; update only for an intended behavior change.
EXPECTED = {
    "dem_only": {
        "flow_direction": "77af52429127a1e7",
        "drainage_area_max": 266.0,
        "basin_cells": 60,
        "ocean_cells": 180,
        "lake_inlets": 0,
        "rainfall_sum": None,
    },
    "lakes_precip": {
        "flow_direction": "b071817f802c9186",
        "drainage_area_max": 328.0,
        "basin_cells": 60,
        "ocean_cells": 180,
        "lake_inlets": 7,
        "rainfall_sum": 59957368.0,
    },
    "coarse_precip": {
        "flow_direction": "b071817f802c9186",
        "drainage_area_max": 328.0,
        "basin_cells": 60,
        "ocean_cells": 180,
        "lake_inlets": 7,
        "rainfall_sum": 59713000.0,
    },
}


@pytest.mark.parametrize("config", CONFIGS)
def test_outputs_are_pinned(config):
    result = _run(flow_accumulation.compute_flow_with_basins, config)
    summary = {
        "flow_direction": _digest(result["flow_direction"]),
        "drainage_area_max": float(np.nanmax(result["drainage_area"])),
        "basin_cells": int(np.sum(result["basin_mask"])) if result["basin_mask"] is not None else 0,
        "ocean_cells": int(np.sum(result["ocean_mask"])),
        "lake_inlets": int(np.sum(result["lake_inlets"])) if result["lake_inlets"] is not None else 0,
        "rainfall_sum": (
            round(float(np.nansum(result["upstream_rainfall"])), 1)
            if result["upstream_rainfall"] is not None
            else None
        ),
    }
    expected = dict(EXPECTED[config])
    # Float sums can shift in the last digits across numpy/scipy versions
    assert summary.pop("rainfall_sum") == pytest.approx(expected.pop("rainfall_sum"), rel=1e-5)
    assert summary == expected
