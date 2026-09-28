"""Units extracted from examples/detroit_combined_render.py, and its no-silent-fallback rules.

The full script is pinned end to end by tools/pin_combined_render.py (mock data, no render).
"""

import numpy as np
import pytest

pytest.importorskip("bpy")

import examples.detroit_combined_render as script  # noqa: E402


# ----- one definition of the score -> colormap transform ---------------------


def _old_formula(score, score_max, min_nonzero, stretch, gamma):
    normalized = score / score_max
    if stretch:
        norm_min = min_nonzero / score_max
        if 1.0 > norm_min:
            normalized = np.clip((normalized - norm_min) / (1.0 - norm_min), 0.0, 1.0)
    return np.power(normalized, gamma)


@pytest.mark.parametrize("stretch", [False, True])
@pytest.mark.parametrize("gamma", [1.0, 0.6])
def test_score_normalization_matches_the_formula_it_replaces(stretch, gamma):
    scores = np.array([np.nan, 0.0, 0.05, 0.2, 0.7, 1.4])
    norm = script.ScoreNormalization(max=1.4, min_nonzero=0.05, stretch=stretch, gamma=gamma)
    np.testing.assert_allclose(
        norm.apply(scores), _old_formula(scores, 1.4, 0.05, stretch, gamma), equal_nan=True
    )


def test_score_normalization_is_measured_on_rendered_valid_non_water_pixels():
    dem = np.array([[1.0, 1.0, np.nan], [1.0, 1.0, 1.0]])
    scores = np.array([[0.2, 9.0, 5.0], [0.0, 0.4, np.nan]])
    water = np.array([[False, True, False], [False, False, False]])
    norm = script.ScoreNormalization.from_rendered_region(scores, dem, water, stretch=True, gamma=1)
    assert norm.max == 0.4  # 9.0 is water, 5.0 is outside the DEM
    assert norm.min_nonzero == 0.2


def test_score_normalization_refuses_an_empty_rendered_region():
    dem = np.full((2, 2), np.nan)
    with pytest.raises(ValueError, match="no rendered"):
        script.ScoreNormalization.from_rendered_region(
            np.ones((2, 2)), dem, None, stretch=False, gamma=1
        )


# ----- small pure helpers ------------------------------------------------------


@pytest.mark.parametrize(
    "ratio, factor", [(0.5, 1), (1.0, 1), (1.5, 2), (2.0, 2), (3.9, 4), (7.0, 8), (40.0, 16)]
)
def test_upscale_factor_is_smallest_power_of_two_covering_ratio_capped_at_16(ratio, factor):
    assert script.upscale_factor_for_ratio(ratio) == factor


def test_parse_rgb_accepts_three_unit_floats_and_rejects_the_rest():
    assert script.parse_rgb("0.1, 0.2,0.3") == (0.1, 0.2, 0.3)
    for bad in ["0.1,0.2", "a,b,c", "0.1,0.2,1.5"]:
        with pytest.raises(ValueError):
            script.parse_rgb(bad)


# ----- no silent fallbacks -------------------------------------------------------


def _parse(*flags):
    return script.parse_args(["--mock-data", *flags])


def test_invalid_park_ring_color_is_an_argument_error():
    with pytest.raises(SystemExit):
        script.parse_args(["--park-ring-color", "red"])


def test_missing_dem_directory_is_an_error_not_a_random_dem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no data/dem/detroit here
    with pytest.raises(FileNotFoundError, match="data/dem/detroit"):
        script.load_dem(script.parse_args([]))


def test_dem_cache_key_changes_with_downsample_method():
    a = script.transform_cache_params(_parse("--downsample-method", "average"), 100_000)
    b = script.transform_cache_params(_parse("--downsample-method", "lanczos"), 100_000)
    assert a != b
