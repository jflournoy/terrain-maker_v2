"""
Characterization tests for flow_accumulation (the file-based entry point).

Pins digests of every output array for each backend and major option, so
the 900-line function can be split into steps without changing results.
Update EXPECTED only for an intended behavior change.
"""

import hashlib

import numpy as np
import pytest
import rasterio
from rasterio import Affine

from src.terrain.flow_accumulation import flow_accumulation
from tests.test_flow_with_basins_characterization import TRANSFORM, _landscape


def _write(path, data, transform):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return str(path)


def _digest(arr, decimals=4):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr.astype(np.float64), decimals)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _count(mask):
    return None if mask is None else int(np.sum(np.asarray(mask) > 0))


CONFIGS = {
    "spec_default": dict(),
    "spec_basins_lakes": dict(detect_basins=True, min_basin_size=50, lakes=True),
    "spec_downsampled": dict(max_cells=900),
    "legacy": dict(backend="legacy"),
    "pysheds": dict(backend="pysheds"),
    "spec_cached": dict(cache=True),
}


def _run(name, tmp_path):
    dem, lake_mask, lake_outlets, precipitation = _landscape()
    dem_path = _write(tmp_path / "dem.tif", dem, TRANSFORM)
    precip_path = _write(
        tmp_path / "precip.tif", precipitation[::2, ::2].copy(), TRANSFORM * Affine.scale(2)
    )
    kwargs = dict(CONFIGS[name])
    if kwargs.pop("lakes", False):
        kwargs.update(lake_mask=lake_mask, lake_outlets=lake_outlets)
    if kwargs.get("cache"):
        kwargs["cache_dir"] = str(tmp_path / "cache")

    result = flow_accumulation(dem_path, precip_path, output_dir=str(tmp_path / "out"), **kwargs)
    if kwargs.get("cache"):  # second call must reproduce the same result from cache
        cached = flow_accumulation(
            dem_path, precip_path, output_dir=str(tmp_path / "out2"), **kwargs
        )
        assert cached["metadata"].get("cache_hit") is True
        for key in ("flow_direction", "drainage_area", "upstream_rainfall"):
            np.testing.assert_array_equal(cached[key], result[key])

    meta = result["metadata"]
    return {
        "shape": tuple(result["flow_direction"].shape),
        "flow_direction": _digest(result["flow_direction"]),
        "drainage_area": _digest(result["drainage_area"]),
        "upstream_rainfall": _digest(result["upstream_rainfall"], decimals=2),
        "conditioned_dem": _digest(result["conditioned_dem"]),
        "basin_cells": _count(result["basin_mask"]),
        "ocean_cells": _count(result["ocean_mask"]),
        "lake_inlets": _count(result["lake_inlets"]),
        "downsampled": meta["downsampling_applied"],
        "cell_size_m": round(float(meta["cell_size_m"]), 3),
    }


EXPECTED = {
    "legacy": {
        "basin_cells": None,
        "cell_size_m": 101.605,
        "conditioned_dem": "824f3a57a274fd02",
        "downsampled": False,
        "drainage_area": "06be8f96277aa6d3",
        "flow_direction": "1a598066347c9eb5",
        "lake_inlets": None,
        "ocean_cells": 180,
        "shape": (60, 60),
        "upstream_rainfall": "22b39b7a522019eb",
    },
    "pysheds": {
        "basin_cells": None,
        "cell_size_m": 101.605,
        "conditioned_dem": "f115a9fb98d78359",
        "downsampled": False,
        "drainage_area": "cac39f7cb3b6271b",
        "flow_direction": "e2511e31b66176a0",
        "lake_inlets": None,
        "ocean_cells": 180,
        "shape": (60, 60),
        "upstream_rainfall": "458d180016b4c462",
    },
    "spec_basins_lakes": {
        "basin_cells": 60,
        "cell_size_m": 101.605,
        "conditioned_dem": "4505d809004f42a3",
        "downsampled": False,
        "drainage_area": "0b229a63a9502039",
        "flow_direction": "c1eb9ef4582adc2a",
        "lake_inlets": 7,
        "ocean_cells": 180,
        "shape": (60, 60),
        "upstream_rainfall": "cd84cee866292c46",
    },
    "spec_cached": {
        "basin_cells": None,
        "cell_size_m": 101.605,
        "conditioned_dem": "30d3cea826398517",
        "downsampled": False,
        "drainage_area": "aaa9d4f44154ca44",
        "flow_direction": "b52b2c5d8053d72d",
        "lake_inlets": None,
        "ocean_cells": 180,
        "shape": (60, 60),
        "upstream_rainfall": "6a8e598874fb5002",
    },
    "spec_default": {
        "basin_cells": None,
        "cell_size_m": 101.605,
        "conditioned_dem": "30d3cea826398517",
        "downsampled": False,
        "drainage_area": "aaa9d4f44154ca44",
        "flow_direction": "b52b2c5d8053d72d",
        "lake_inlets": None,
        "ocean_cells": 180,
        "shape": (60, 60),
        "upstream_rainfall": "6a8e598874fb5002",
    },
    "spec_downsampled": {
        "basin_cells": None,
        "cell_size_m": 203.21,
        "conditioned_dem": "f09b85f6097d1e35",
        "downsampled": True,
        "drainage_area": "5ac161f5a44e2deb",
        "flow_direction": "0e0d83e0c0398466",
        "lake_inlets": None,
        "ocean_cells": 30,
        "shape": (30, 30),
        "upstream_rainfall": "83460dfbc1512713",
    },
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_flow_accumulation_outputs_are_pinned(name, tmp_path):
    summary = _run(name, tmp_path)
    assert summary == EXPECTED[name]
