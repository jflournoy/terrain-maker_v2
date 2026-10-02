"""Bilateral smoothing has one backend, OpenCV, and refuses to run without it.

It used to fall back silently to skimage and then to pure numpy. Those are different
algorithms: on the Detroit mock scores, skimage differs from OpenCV by 3% of the score
range, so the same command gave different terrain on a machine where `import cv2` failed.
"""

import sys

import numpy as np
import pytest

from terrain_maker.terrain.transforms import feature_preserving_smooth, smooth_score_data


@pytest.fixture
def scores():
    data = np.random.default_rng(0).uniform(0.4, 0.8, (40, 30)).astype(np.float32)
    data[0, :3] = np.nan
    return data


@pytest.fixture
def no_cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)  # makes `import cv2` raise ImportError


def test_score_smoothing_without_opencv_raises(no_cv2, scores):
    with pytest.raises(ImportError, match="opencv-python-headless"):
        smooth_score_data(scores, sigma_spatial=3.0)


def test_dem_smoothing_without_opencv_raises(no_cv2, scores):
    with pytest.raises(ImportError, match="opencv-python-headless"):
        feature_preserving_smooth(sigma_spatial=3.0)(scores * 1000.0, None)


def test_score_smoothing_with_opencv_keeps_nan_and_range(scores):
    smoothed = smooth_score_data(scores, sigma_spatial=3.0)
    assert np.array_equal(np.isnan(smoothed), np.isnan(scores))
    assert np.nanmin(smoothed) >= 0.0 and np.nanmax(smoothed) <= 1.0
