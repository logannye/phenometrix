import math
import pytest
from phenometrix_prosody.metrics import Reference, Prediction, score, oracle_coverage, error_coverage_curve


def test_independent_known_answer_and_denominators():
    refs = [Reference(0, 100, True), Reference(1, 100, True), Reference(2, 100, True),
            Reference(3, None, False), Reference(4, None, False), Reference(5, None, False),
            Reference(6, -1, True, valid=False), Reference(7, 100, True, eligible=False)]
    pred = [Prediction(0, "voiced", 100, .8), Prediction(1, "voiced", 200, .9),
            Prediction(2, "unvoiced", None), Prediction(3, "voiced", 100, .7),
            Prediction(4, "unvoiced", None)]
    r = score(refs, pred)
    assert r["eligible_valid_frames"] == 6
    assert r["joint_error"] == 4 / 6
    assert r["reference_voiced_success_rate"] == 1 / 3
    assert r["voicing_false_positive_rate"] == 1 / 3
    assert r["voicing_false_negative_or_unavailable_rate"] == 1 / 3
    assert r["measurement_coverage"] == 5 / 6
    assert r["jointly_voiced_coverage"] == 2 / 3
    assert r["octave_error_count"] == 1
    assert r["brier_score"] == pytest.approx((.04 + .81 + .49) / 3)
    assert r["excluded_invalid_reference"] == r["excluded_boundary"] == 1


def test_abstain_all_never_looks_accurate_and_bounds_not_excluded():
    r = score([Reference(0, 800, True), Reference(1, None, False)], [])
    assert r["joint_error"] == 1
    assert r["measurement_coverage"] == 0
    assert r["reference_outside_supported_bounds"] == 1
    assert r["median_jointly_voiced_cents"] is None


def test_strict_grid_null_and_duplicate_contract():
    refs = [Reference(10, 100, True)]
    with pytest.raises(ValueError, match="exact reference grid"):
        score(refs, [Prediction(11, "voiced", 100)])
    with pytest.raises(ValueError, match="Duplicate"):
        score(refs, [Prediction(10, "voiced", 100)] * 2)
    with pytest.raises(ValueError, match="null"):
        score(refs, [Prediction(10, "unavailable", 0)])
    with pytest.raises(ValueError, match="positive"):
        score(refs, [Prediction(10, "voiced", math.nan)])


def test_oracle_and_coverage_use_unconditional_denominators():
    refs = [Reference(0, 100, True), Reference(1, 200, True), Reference(2, None, False)]
    assert oracle_coverage(refs, {0: [50, 100], 1: [100]})["coverage"] == .5
    pred = [Prediction(0, "voiced", 100, .8), Prediction(1, "voiced", 400, .2),
            Prediction(2, "unvoiced", None)]
    curve = error_coverage_curve(refs, pred, [0, .5, .9])
    assert [r["joint_error"] for r in curve] == [1/3, 1/3, 2/3]
    assert [r["measurement_coverage"] for r in curve] == [1, 2/3, 1/3]
