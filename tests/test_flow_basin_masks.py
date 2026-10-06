"""Ocean is kept out of endorheic-basin detection, and inland basins are found.

These were source-text checks on san_diego_flow_demo.py ("exclude_mask=ocean_mask" in the
demo); basin detection moved inside flow_accumulation(), so the behavior is tested here,
on the library function that does it.
"""

from unittest.mock import patch

import numpy as np

from terrain_maker.terrain.hydrology import conditioning
from terrain_maker.terrain.hydrology.flow import _build_conditioning_masks


def _coast_with_inland_basin():
    dem = np.full((40, 40), 100.0, dtype=np.float32)
    dem[:, :6] = -5.0  # ocean along the west edge, below sea level and border-connected
    dem[15:25, 20:30] = 40.0  # closed inland depression, 60 m deep
    return dem


def _masks(dem, lake_mask=None):
    return _build_conditioning_masks(
        mask_ocean=True, dem_data=dem, ocean_elevation_threshold=0.0, detect_basins=True,
        backend="spec", min_basin_size=10, min_basin_depth=5.0, lake_mask=lake_mask,
    )


def test_inland_basin_found_and_ocean_not_a_basin():
    dem = _coast_with_inland_basin()
    basin_mask, _conditioning, _flow, ocean_mask = _masks(dem)
    assert ocean_mask[:, :6].all() and not ocean_mask[:, 6:].any()
    assert basin_mask[15:25, 20:30].all()
    assert not (basin_mask & ocean_mask).any()


def test_basin_detection_is_given_the_ocean_to_exclude():
    dem = _coast_with_inland_basin()
    with patch.object(conditioning, "detect_endorheic_basins",
                      wraps=conditioning.detect_endorheic_basins) as detect, \
            patch("terrain_maker.terrain.hydrology.flow.detect_endorheic_basins", detect):
        _basin, _conditioning, _flow, ocean_mask = _masks(dem)
    excluded = detect.call_args.kwargs["exclude_mask"]
    assert np.array_equal(excluded, ocean_mask)


def _dem_with_pits(n=100):
    """A 100x100 plateau (10,000 cells) with a 4x4 pit (16 cells) and a 40x40 pit (1,600)."""
    dem = np.full((n, n), 100.0, dtype=np.float32)
    dem[10:14, 10:14] = 50.0
    dem[50:90, 50:90] = 50.0
    return dem


def test_none_min_basin_size_is_adaptive_one_per_thousand_cells():
    # 1/1000 of 10,000 cells = 10: both pits qualify. Make the grid 4x bigger (40,000 cells,
    # threshold 40) and the 16-cell pit must drop out while the 1,600-cell one stays.
    small = _build_conditioning_masks(
        mask_ocean=False, dem_data=_dem_with_pits(), ocean_elevation_threshold=0.0,
        detect_basins=True, backend="spec", min_basin_size=None, min_basin_depth=5.0, lake_mask=None)[0]
    assert small[10:14, 10:14].all() and small[50:90, 50:90].all()

    big_dem = np.kron(_dem_with_pits(), np.ones((2, 2), dtype=np.float32))  # pits become 64 and 6,400
    big_dem[40:44, 150:154] = 50.0  # a 16-cell pit on open plateau, 40,000-cell grid: below 40
    big = _build_conditioning_masks(
        mask_ocean=False, dem_data=big_dem, ocean_elevation_threshold=0.0,
        detect_basins=True, backend="spec", min_basin_size=None, min_basin_depth=5.0, lake_mask=None)[0]
    assert not big[40:44, 150:154].any()
    assert big[100:180, 100:180].sum() >= 6400 - 16


def test_detect_endorheic_basins_refuses_none():
    import pytest

    with pytest.raises(ValueError, match="min_size"):
        conditioning.detect_endorheic_basins(_dem_with_pits(), min_size=None)


def test_compute_flow_with_basins_honours_an_explicit_5000(monkeypatch):
    from terrain_maker.terrain.hydrology import flow

    seen = {}

    def spy(dem, min_size, **kw):
        seen["min_size"] = min_size
        raise RuntimeError("stop after basin detection")

    monkeypatch.setattr(flow, "detect_endorheic_basins", spy)
    import pytest

    with pytest.raises(RuntimeError, match="stop"):
        flow.compute_flow_with_basins(_dem_with_pits(), None, min_basin_size=5000, verbose=False)
    assert seen["min_size"] == 5000
    with pytest.raises(RuntimeError, match="stop"):
        flow.compute_flow_with_basins(_dem_with_pits(), None, min_basin_size=None, verbose=False)
    assert seen["min_size"] == 10  # 1/1000 of 10,000 cells
