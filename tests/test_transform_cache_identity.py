"""The transform cache returns a result only for the same input and the same transforms.

Keys used to be the layer name plus the transforms' __name__s, and apply_transforms read the
cache even with cache=False. So a different DEM, or a transform with a different parameter
behind the same name, silently loaded another run's result from terrain_cache/.
"""

import numpy as np
import pytest
from rasterio.transform import from_origin

from terrain_maker.terrain.core import Terrain


def _named(name, fn):
    def transform(data, transform=None):
        return fn(data), transform, None

    transform.__name__ = name
    return transform


def add(k):
    return _named("add", lambda d: d + k)  # same __name__ for every k


def scale(k):
    t = _named("scale", lambda d: d * k)
    t._elevation_scale_factor = k
    return t


def _run(cache_dir, dem, transforms, cache):
    terrain = Terrain(dem, from_origin(-83.0, 42.0, 0.01, 0.01), cache_dir=str(cache_dir))
    for t in transforms:
        terrain.add_transform(t)
    terrain.apply_transforms(cache=cache)
    return terrain


@pytest.fixture
def dem():
    return np.arange(20, dtype=np.float32).reshape(4, 5)


def test_cache_false_never_reads_the_cache(tmp_path, dem):
    _run(tmp_path, dem, [add(1)], cache=True)
    other = dem + 100
    result = _run(tmp_path, other, [add(1)], cache=False)
    np.testing.assert_array_equal(result.data_layers["dem"]["transformed_data"], other + 1)


def test_different_dem_is_a_miss(tmp_path, dem):
    _run(tmp_path, dem, [add(1)], cache=True)
    other = dem + 100
    result = _run(tmp_path, other, [add(1)], cache=True)
    np.testing.assert_array_equal(result.data_layers["dem"]["transformed_data"], other + 1)


def test_different_transform_parameter_same_name_is_a_miss(tmp_path, dem):
    _run(tmp_path, dem, [add(1)], cache=True)
    result = _run(tmp_path, dem, [add(5)], cache=True)
    np.testing.assert_array_equal(result.data_layers["dem"]["transformed_data"], dem + 5)


def test_same_input_and_transforms_is_a_hit(tmp_path, dem):
    _run(tmp_path, dem, [add(1)], cache=True)
    files_after_first = sorted(p.name for p in tmp_path.iterdir())
    result = _run(tmp_path, dem, [add(1)], cache=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == files_after_first  # nothing new written
    np.testing.assert_array_equal(result.data_layers["dem"]["transformed_data"], dem + 1)


def test_cache_hit_keeps_elevation_scale(tmp_path, dem):
    first = _run(tmp_path, dem, [scale(0.5)], cache=True)
    second = _run(tmp_path, dem, [scale(0.5)], cache=True)
    assert first.transform_metadata["elevation_scale"] == 0.5
    assert second.transform_metadata["elevation_scale"] == 0.5


def test_library_transform_keys_are_stable_across_processes():
    """A key that changed every run would make the cache never hit (safe, but useless)."""
    import subprocess
    import sys

    code = (
        "from terrain_maker.terrain._fingerprint import callable_fingerprint as f\n"
        "from terrain_maker.terrain import transforms as T\n"
        "print([f(t) for t in (T.reproject_raster('EPSG:4326','EPSG:32617'), T.flip_raster('horizontal'),"
        " T.scale_elevation(0.0001), T.downsample_then_reproject(downsample_zoom_factor=0.2),"
        " T.feature_preserving_smooth(3.0), T.despeckle_dem(kernel_size=3), T.smooth_raster(5),"
        " T.remove_bumps(3))])\n"
    )
    runs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
            for _ in range(2)]
    assert runs[0] == runs[1]


def test_transform_parameters_change_the_key():
    from terrain_maker.terrain import transforms as T
    from terrain_maker.terrain._fingerprint import callable_fingerprint

    assert callable_fingerprint(T.feature_preserving_smooth(3.0, sigma_intensity=10.0)) != \
        callable_fingerprint(T.feature_preserving_smooth(3.0, sigma_intensity=20.0))
