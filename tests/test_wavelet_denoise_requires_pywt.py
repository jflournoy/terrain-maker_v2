"""wavelet_denoise_dem runs wavelets or raises; it never runs a median filter under its name.

Without PyWavelets it used to return a 3x3 median filter, still named wavelet_denoise(...),
so --wavelet-denoise silently produced (and cached) median-filtered terrain.
"""

import sys

import numpy as np
import pytest
from scipy.ndimage import median_filter

from terrain_maker.terrain.transforms import wavelet_denoise_dem


@pytest.fixture
def dem():
    return np.random.default_rng(0).normal(200, 5, (64, 64)).astype(np.float32)


def test_without_pywt_raises(dem, monkeypatch):
    monkeypatch.setitem(sys.modules, "pywt", None)
    with pytest.raises(ImportError, match="PyWavelets"):
        wavelet_denoise_dem()(dem)


def test_with_pywt_is_not_the_median_filter(dem):
    out, _, _ = wavelet_denoise_dem()(dem)
    assert not np.allclose(out, median_filter(dem, size=3))
