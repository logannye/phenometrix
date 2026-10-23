import numpy as np
import pytest
from phenometrix_prosody import evaluate_synthetic as syn


def test_fixed_observation_changes_preserve_identifiable_analytic_fundamental():
    assert len(syn.OBSERVATION_SCENARIOS) == 3
    for scenario in syn.OBSERVATION_SCENARIOS:
        pcm = syn.observation_pcm(scenario)
        assert pcm.shape == (76800,) and pcm.dtype == np.float32 and np.isfinite(pcm).all()
        assert syn.analytic_frequency(scenario, .4) == syn.analytic_frequency(scenario, 1.)
        # A 0.1-second postchange interval has integer periods at every fixed F0.
        spectrum = np.abs(np.fft.rfft(pcm[48000:52800]))
        bin1 = scenario["initialHz"] // 10
        assert spectrum[bin1] < 1e-4
        assert spectrum[2 * bin1] > 100 and spectrum[3 * bin1] > 50
        assert syn.region_at(scenario, 28800) == "observation-boundary"
        assert syn.region_at(scenario, 31200) == "after-observation-change"
    with pytest.raises(ValueError):
        syn.observation_pcm({**syn.OBSERVATION_SCENARIOS[0], "initialHz": 150})


def test_step_response_retains_negative_target_lag_and_positive_availability():
    scenario = {"changeStartSeconds": .6, "finalHz": 200}
    rows = [{"reference_sample": int(t * 48), "target_ms": t, "availability_ms": t + 20, "estimate_hz": 200.}
            for t in [586, 596, 606]]
    response = syn.step_response(rows, scenario)
    assert response["target_time_first_of_three_settling_lag_ms"] == -14
    assert response["earliest_availability_first_of_three_delay_ms"] == 6
    rows[1]["estimate_hz"] = None
    assert syn.step_response(rows, scenario)["settled"] is False


def test_glide_delay_and_distortion_use_declared_target_not_availability():
    scenario = {"id": "known", "kind": "glide", "initialHz": 100., "finalHz": 300.,
                "changeStartSeconds": .25, "changeEndSeconds": 1.25, "slopeHzPerSecond": 200.}
    ends = [14880, 15360, 15840]  # 310,320,330ms availability, target290,300,310ms
    rows = [{"sample": e - 960, "state": "voiced", "f0_hz": syn.analytic_frequency(scenario, (e - 960) / 48000) - 2.}
            for e in ends]
    raw = {"availability_sample": ends, "predictions": {method: rows for method in syn.METHODS}}
    scores, traces = syn.score_predictions(scenario, raw)
    assert scores["gru"]["regions"]["glide-interior"]["median_signed_equivalent_slope_delay_ms"] == 10
    assert all(r["availability_ms"] - r["target_ms"] == 20 for r in traces)


def gate_fixture():
    scenarios = []
    for index, kind in enumerate(["step"] * 2 + ["glide"] * 6):
        name = f"{kind}-{index}"
        metrics = {method: {"step_response": {"target_time_first_of_three_settling_lag_ms": 0.},
                            "regions": {"glide-interior": {"median_signed_equivalent_slope_delay_ms": 0.}}}
                   for method in syn.METHODS}
        scenarios.append({"scenario": {"id": name, "kind": kind}, "methods": metrics})
    return scenarios


def test_transition_gate_requires_all_eight_and_both_subgroup_constraints():
    scenarios = gate_fixture()
    assert syn.transition_gate(scenarios)["model_components"]["gru"]["transition_lag_component_pass"]
    for row in scenarios[:2]:
        row["methods"]["gru"]["step_response"]["target_time_first_of_three_settling_lag_ms"] = 20.
    gate = syn.transition_gate(scenarios)
    assert gate["model_components"]["gru"]["extra_median_target_delay_vs_delayed_dsp_ms"]["median_signed_target_delay_ms"] == 0
    assert not gate["model_components"]["gru"]["transition_lag_component_pass"]
    assert gate["model_components"]["mlp"]["transition_lag_component_pass"]
    scenarios[-1]["methods"]["mlp"]["regions"]["glide-interior"]["median_signed_equivalent_slope_delay_ms"] = None
    assert not syn.transition_gate(scenarios)["model_components"]["mlp"]["transition_lag_component_pass"]


def test_absence_is_a_failure_in_coverage_not_a_zero_pitch_or_omitted_frame():
    summary = syn.summarize([{"signed_cents": None, "equivalent_slope_delay_ms": None},
                             {"signed_cents": 0., "equivalent_slope_delay_ms": 0.}])
    assert summary["frames"] == 2 and summary["voiced_outputs"] == 1
    assert summary["pitch_coverage"] == .5 and summary["success_within_50c_all_frames"] == .5
