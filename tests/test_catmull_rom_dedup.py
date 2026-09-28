"""The Catmull-Rom duplicate filter keeps first occurrences under np.allclose(atol=1e-6).

Pins the new linear-time filter against the original quadratic one (kept here as the
reference) on inputs with exact and near duplicates, and bounds its running time.
"""

import time

import numpy as np

from terrain_maker.terrain.mesh.boundary import _drop_near_duplicates


def _reference(points):
    kept = []
    for pt in points:
        if not any(np.allclose(pt, prev, atol=1e-6) for prev in kept):
            kept.append(pt)
    return kept


def _points_with_duplicates(rng, n, scale):
    base = rng.uniform(0, scale, size=(n, 2))
    dup_idx = rng.integers(0, n, size=n // 4)
    exact = base[dup_idx]
    rtol_edge = base[rng.integers(0, n, size=n // 4)]
    near = rtol_edge * (1 + rng.choice([-1, 1], size=rtol_edge.shape) * rng.uniform(0, 2e-5, rtol_edge.shape))
    pts = np.vstack([base, exact, near])
    return list(pts[rng.permutation(len(pts))])


def test_matches_quadratic_reference():
    rng = np.random.default_rng(0)
    for scale in (1.0, 50.0, 3000.0):
        points = _points_with_duplicates(rng, 400, scale)
        expected = _reference(points)
        got = _drop_near_duplicates(points)
        assert len(got) == len(expected)
        np.testing.assert_array_equal(np.array(got), np.array(expected))


def test_is_fast_on_mesh_sized_boundaries():
    rng = np.random.default_rng(1)
    points = list(rng.uniform(0, 2000, size=(40_000, 2)))
    start = time.perf_counter()
    _drop_near_duplicates(points)
    assert time.perf_counter() - start < 5.0
