import pytest
from phenometrix_prosody.support_diagnostic import summarize


def test_signed_slope_lag_is_not_redefined_as_correctness():
    rows = [{"estimate_hz": 196, "signed_cents": -35, "slope_hz_per_second": 200,
             "equivalent_linear_slope_delay_ms": 20},
            {"estimate_hz": None, "signed_cents": None, "slope_hz_per_second": 200,
             "equivalent_linear_slope_delay_ms": None}]
    result = summarize(rows)
    assert result["pitch_coverage"] == .5
    assert result["correct_50c_fraction_all"] == .5
    assert result["median_equivalent_linear_slope_delay_ms"] == 20
    assert result["expected_lag_sign_fraction"] == 1


def test_no_available_pitch_stays_explicit():
    result = summarize([])
    assert result["pitch_coverage"] is None
    assert result["median_signed_cents"] is None
