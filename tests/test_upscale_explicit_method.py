"""upscale_scores runs the method it is given, or raises.

It used to accept method="auto" (ESRGAN -> bilinear -> nearest) and drop ANY failing method,
an explicitly requested one included, down to nearest-neighbour, with a log line at most.
Upscaled precipitation feeds upstream-rainfall numbers, so the method is part of the result.
"""

import sys

import numpy as np
import pytest
from scipy.ndimage import zoom

from terrain_maker.terrain.transforms import upscale_scores


@pytest.fixture
def scores():
    return np.random.default_rng(0).uniform(0.2, 0.9, (8, 6)).astype(np.float32)


def test_default_is_bilinear(scores):
    expected = np.clip(zoom((scores - scores.min()) / (scores.max() - scores.min()), 4, order=1,
                            mode="reflect") * (scores.max() - scores.min()) + scores.min(),
                       scores.min(), scores.max())
    np.testing.assert_allclose(upscale_scores(scores, scale=4), expected, rtol=1e-6)


def test_auto_is_gone(scores):
    with pytest.raises(ValueError, match="auto"):
        upscale_scores(scores, scale=2, method="auto")


def test_unknown_method_raises(scores):
    with pytest.raises(ValueError, match="lanczos"):
        upscale_scores(scores, scale=2, method="lanczos")


def test_requested_esrgan_unavailable_raises(scores, monkeypatch):
    monkeypatch.setitem(sys.modules, "realesrgan", None)
    with pytest.raises(ImportError, match="upscale"):
        upscale_scores(scores, scale=2, method="esrgan")


def test_failing_method_raises_instead_of_nearest(scores, monkeypatch):
    import terrain_maker.terrain.transforms as T

    def broken(normalized, scale):
        raise RuntimeError("bilateral exploded")

    monkeypatch.setattr(T, "_upscale_bilateral", broken)
    with pytest.raises(RuntimeError, match="bilateral exploded"):
        upscale_scores(scores, scale=2, method="bilateral")
