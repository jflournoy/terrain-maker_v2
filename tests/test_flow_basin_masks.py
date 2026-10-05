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
