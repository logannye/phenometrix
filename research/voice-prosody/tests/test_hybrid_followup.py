import numpy as np
import pytest
from phenometrix_prosody.hybrid_followup import hybrid_predictions, fresh_records


def fixture():
    f = {"reference_sample": np.array([2208, 2688, 3168]), "features": np.zeros((3, 57), np.float32),
         "sample_rate_hz": np.array(48000), "pcm_sample_count": np.array(4800)}
    model = [{"sample": int(e - 960), "voicing_probability": p, "state": "unavailable", "f0_hz": None}
             for e, p in zip(f["reference_sample"], [.4999, .5, .9])]
    yin = [{"sample": int(e), "raw_f0_hz": hz, "window_rms": 0., "normalized_lag_correlation": 0.}
           for e, hz in zip(f["reference_sample"], [300., 301., float("nan")])]
    return f, model, yin


def test_voicing_threshold_raw_pitch_and_unavailable_are_separate():
    f, model, yin = fixture(); rows = hybrid_predictions(f, model, yin)
    assert [r["sample"] for r in rows] == [1248, 1728, 2208]
    assert [r["state"] for r in rows] == ["unvoiced", "voiced", "unavailable"]
    assert rows[1]["f0_hz"] == 301.
    assert all(r["correctness_probability"] is None for r in rows)
    # Previously unavailable candidate output and zero RMS/correlation do not gate raw YIN.
    assert rows[1]["voicing_probability"] == .5


def test_rejects_misaligned_or_invalid_probability_evidence():
    f, model, yin = fixture(); yin[0]["sample"] -= 960
    with pytest.raises(ValueError, match="sample"):
        hybrid_predictions(f, model, yin)
    f, model, yin = fixture(); model[0]["voicing_probability"] = float("nan")
    with pytest.raises(ValueError, match="probability"):
        hybrid_predictions(f, model, yin)
    with pytest.raises(ValueError, match="length"):
        hybrid_predictions(f, model[:-1], yin)


def test_known_truth_is_not_an_inference_input():
    f, model, yin = fixture(); original = hybrid_predictions(f, model, yin)
    f["scorer_only_reference_hz"] = np.array([123., 999., 0.])
    assert hybrid_predictions(f, model, yin) == original


def test_fresh_challenge_is_fixed_and_includes_all_required_source_types():
    records = fresh_records()
    assert len(records) == 32 and len({r["id"] for r in records}) == 32
    assert len({r["seed"] for r in records}) == 32
    assert sum(r["latent_voiced"] for r in records) == 28
    assert len([r for r in records if r["kind"] == "step"]) == 8
    assert len([r for r in records if r["kind"] == "glide"]) == 8
    assert fresh_records() == records
    for r in records:
        if r["latent_voiced"] and r["harmonic_amplitudes"][0] == 0:
            assert r["harmonic_amplitudes"][1] > 0 and r["harmonic_amplitudes"][2] > 0
