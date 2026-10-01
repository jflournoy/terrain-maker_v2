"""The cross-machine comparison in tools/pin_combined_render.py.

The pin tool is the gate for refactors of detroit_combined_render.py, so its comparison is
production code: float noise must pass (and be reported), real changes must fail.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pin_combined_render as pins  # noqa: E402


def _pin_pair(tmp_path, ref, new, kind="array"):
    ref_dir, new_dir = tmp_path / "ref", tmp_path / "new"
    ref_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    return (pins.ArrayPinner(ref_dir)(ref, kind), pins.ArrayPinner(new_dir)(new, kind), ref_dir, new_dir)


def _compare(tmp_path, ref, new, kind="array"):
    old_pin, new_pin, ref_dir, new_dir = _pin_pair(tmp_path, ref, new, kind)
    return pins.compare({"x": old_pin}, {"x": new_pin}, ref_dir, new_dir)


@pytest.fixture
def floats():
    return np.random.default_rng(0).uniform(150, 250, (40, 30)).astype(np.float32)


def test_identical_arrays_report_nothing(tmp_path, floats):
    assert _compare(tmp_path, floats, floats.copy()) == ([], [])


def test_last_bit_float_noise_passes_and_is_reported(tmp_path, floats):
    noisy = np.nextafter(floats, np.float32(np.inf))
    failures, tolerated = _compare(tmp_path, floats, noisy)
    assert failures == []
    assert len(tolerated) == 1 and "of range" in tolerated[0]


def test_float_change_beyond_tolerance_fails(tmp_path, floats):
    changed = floats.copy()
    changed[3, 4] *= 1.001
    failures, _ = _compare(tmp_path, floats, changed)
    assert len(failures) == 1


def test_nan_mask_change_fails(tmp_path, floats):
    changed = floats.copy()
    changed[0, 0] = np.nan
    failures, _ = _compare(tmp_path, floats, changed)
    assert failures and "NaN" in failures[0]


def test_shape_change_fails(tmp_path, floats):
    failures, _ = _compare(tmp_path, floats, floats[:-1])
    assert len(failures) == 1


def test_uint8_one_step_passes_two_steps_fail(tmp_path):
    colors = np.full((10, 10, 4), 100, dtype=np.uint8)
    one, two = colors.copy(), colors.copy()
    one[2, 2, 0] = 101
    two[2, 2, 0] = 102
    assert _compare(tmp_path / "a", colors, one)[0] == []
    assert len(_compare(tmp_path / "b", colors, two)[0]) == 1


def test_image_tolerates_a_few_pixels_not_a_visible_change(tmp_path):
    image = np.zeros((100, 100, 4), dtype=np.uint8)
    few, many = image.copy(), image.copy()
    few[0, :5] = 200  # 5 of 10,000 pixels: 0.05%
    many[:, :10] = 200  # 10% of pixels
    assert _compare(tmp_path / "a", image, few, kind="image")[0] == []
    assert len(_compare(tmp_path / "b", image, many, kind="image")[0]) == 1


def test_integer_arrays_must_match_exactly(tmp_path):
    faces = np.arange(12, dtype=np.int64)
    changed = faces.copy()
    changed[5] = 6
    failures, tolerated = _compare(tmp_path, faces, changed)
    assert len(failures) == 1 and "exact" in failures[0] and tolerated == []


def test_missing_reference_array_fails(tmp_path, floats):
    old_pin, new_pin, ref_dir, new_dir = _pin_pair(tmp_path, floats, floats * 2)
    (ref_dir / f"{old_pin['sha']}.npz").unlink()
    failures, _ = pins.compare(old_pin, new_pin, ref_dir, new_dir)
    assert failures and "no reference" in failures[0]


def test_numbers_compare_within_tolerance_strings_exactly():
    failures, tolerated = pins.compare(
        {"z": 2.0951, "offset": [1.0, 2.0], "name": "clay"},
        {"z": 2.0951 * (1 + 1e-9), "offset": [1.0, 2.1], "name": "clay"},
        None, None,
    )
    assert tolerated == ["/z: 2.0951 -> 2.0951000020951"]
    assert failures == ["/offset[1]: 2.0 -> 2.1"]


def test_added_or_removed_keys_fail():
    failures, _ = pins.compare({"a": 1, "b": 2}, {"a": 1, "c": 2}, None, None)
    assert sorted(failures) == ["/b: removed", "/c: added"]


def test_old_pin_format_is_refused(tmp_path):
    old = tmp_path / "old.json"
    old.write_text('{"default": {}}')
    with pytest.raises(SystemExit, match="format 1"):
        pins._load_pins(old)
