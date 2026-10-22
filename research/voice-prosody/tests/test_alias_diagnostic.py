import numpy as np
from phenometrix_prosody.alias_diagnostic import intervention, summary


def fixture():
    x = np.arange(57, dtype=np.float32)[None]
    x[0, :48].reshape(6, 8)[:, 0] = 1
    return {"features": x, "candidate_hz": np.array([[100, 150, 200, 250, 300, 400]], np.float64),
            "candidate_mask": np.ones((1, 6), bool)}


def test_removal_repacks_consistent_slots_without_mutating_input():
    original = fixture(); changed = intervention(original, {}, "remove:1")
    assert changed["candidate_hz"][0, :5].tolist() == [100, 200, 250, 300, 400]
    assert np.isnan(changed["candidate_hz"][0, 5])
    assert changed["candidate_mask"][0].tolist() == [True] * 5 + [False]
    np.testing.assert_array_equal(changed["features"][0, 40:48], np.zeros(8))
    assert changed["features"][0, 55] == np.float32(5 / 6)
    assert original["candidate_hz"][0, 1] == 150


def test_mask_and_swap_are_explicit_distinct_counterfactuals():
    original = fixture(); masked = intervention(original, {}, "mask:1")
    np.testing.assert_array_equal(masked["features"], original["features"])
    assert not masked["candidate_mask"][0, 1]
    swapped = intervention(original, {}, "swap:1")
    assert swapped["candidate_hz"][0, 1:3].tolist() == [200, 150]
    np.testing.assert_array_equal(swapped["features"][0, 8:16], original["features"][0, 16:24])


def test_unvoiced_rows_remain_failures_in_known_voiced_summary():
    rows = [{"f0_hz": 300, "slot": 1}, {"f0_hz": None, "slot": None}, {"f0_hz": 150, "slot": 0}]
    result = summary(rows, 300, [True, True, True])
    assert result["success50"] == 1 / 3
    assert result["voiced"] == 2
    assert result["octave_error"] == 1 / 3
