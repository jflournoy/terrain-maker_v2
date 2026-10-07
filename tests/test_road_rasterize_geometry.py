"""Road rasterization: square pixels in meters, no phantom chords, malformed features raise.

The N-S pixel size used the E-W (cos latitude) conversion, so '30 m' pixels were ~40 m tall
at 42N; points outside the bbox were dropped and the rest joined, drawing a straight chord
where a road left and re-entered the area; malformed features were skipped at debug level.
"""

import numpy as np
import pytest

from terrain_maker.terrain.roads import rasterize_roads_to_layer

BBOX = (42.0, -83.5, 42.1, -83.4)  # south, west, north, east


def _road(coords, kind="primary"):
    return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"highway": kind}}


def _fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


def test_pixels_are_the_requested_size_in_both_directions():
    _, tf = rasterize_roads_to_layer(_fc(), BBOX, resolution=30.0)
    lat = np.radians(42.05)
    assert abs(tf.a) * 111_000 * np.cos(lat) == pytest.approx(30.0, rel=1e-3)
    assert abs(tf.e) * 111_000 == pytest.approx(30.0, rel=1e-3)


def test_road_leaving_and_reentering_draws_no_chord():
    # Runs along the south edge, dips out of the bbox, comes back further east
    road = _road([[-83.49, 42.01], [-83.47, 42.01], [-83.46, 41.90], [-83.43, 41.90], [-83.42, 42.01],
                  [-83.41, 42.01]])
    grid, tf = rasterize_roads_to_layer(_fc(road), BBOX, resolution=30.0, road_width_pixels=1)
    row = int((42.1 - 42.01) / abs(tf.e))
    col_mid = int((-83.445 - -83.5) / tf.a)  # between the exit and re-entry points
    assert grid[row - 2:row + 3, col_mid].max() == 0


def test_road_crossing_the_edge_is_drawn_to_the_edge():
    road = _road([[-83.45, 42.05], [-83.30, 42.05]])  # ends well east of the bbox
    grid, tf = rasterize_roads_to_layer(_fc(road), BBOX, resolution=30.0, road_width_pixels=1)
    row = int((42.1 - 42.05) / abs(tf.e))
    assert grid[row, -1] > 0


def test_malformed_feature_raises():
    with pytest.raises(ValueError, match="feature 1"):
        rasterize_roads_to_layer(_fc(_road([[-83.45, 42.05]])), BBOX)
    point = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-83.45, 42.05]},
             "properties": {}}
    with pytest.raises(ValueError, match="Point"):
        rasterize_roads_to_layer(_fc(point), BBOX)


def test_multilinestring_parts_are_drawn():
    multi = {"type": "Feature", "properties": {"highway": "motorway"},
             "geometry": {"type": "MultiLineString",
                          "coordinates": [[[-83.49, 42.02], [-83.47, 42.02]], [[-83.43, 42.08], [-83.41, 42.08]]]}}
    grid, _ = rasterize_roads_to_layer(_fc(multi), BBOX, resolution=30.0, road_width_pixels=1)
    assert (grid == 4).sum() > 20
