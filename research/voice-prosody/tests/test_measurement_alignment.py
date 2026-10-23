import numpy as np
import pytest
from phenometrix_prosody.measurement_alignment import align_feature_timing


def feature_fixture():
    return {"reference_sample": np.array([2208, 2688, 3168, 3648, 4128, 4608]),
            "features": np.zeros((6, 57), np.float32), "sample_rate_hz": np.array(48000),
            "pcm_sample_count": np.array(4800)}


def test_twenty_ms_delay_changes_target_but_never_input_availability():
    feature = feature_fixture()
    timing = align_feature_timing(feature, measurement_delay_ms=20)
    assert timing.support_start_sample.tolist() == [288, 768, 1248, 1728, 2208, 2688]
    assert timing.reference_sample.tolist() == [1248, 1728, 2208, 2688, 3168, 3648]
    assert timing.availability_sample.tolist() == feature["reference_sample"].tolist()
    assert timing.last_required_input_sample[0] == 2207
    # First common eligible target is 66 ms, available at 86 ms. At 76 ms target,
    # the centered comparator's 32 ms future support exceeds this 100 ms file.
    assert timing.common_eligible.tolist() == [False, False, False, False, True, False]
    assert timing.reset_before_frame.tolist() == [True, False, False, False, False, False]
    assert timing.metadata()["lookahead_relative_to_reference_ms"] == 20


def test_original_endpoint_contract_is_preserved_and_arrays_are_not_mutated():
    feature = feature_fixture()
    original = feature["reference_sample"].copy()
    endpoint = align_feature_timing(feature, measurement_delay_ms=0)
    assert np.array_equal(endpoint.reference_sample, original)
    assert endpoint.common_eligible.tolist() == [False, False, True, False, False, False]
    delayed = align_feature_timing(feature, measurement_delay_ms=20)
    assert np.array_equal(feature["reference_sample"], original)
    with pytest.raises(ValueError):
        delayed.reference_sample[0] = 0


def test_label_intervention_cannot_change_input_only_time_mapping():
    feature = feature_fixture()
    feature["scorer_only_f0"] = np.array([100] * 6)
    first = align_feature_timing(feature, measurement_delay_ms=20)
    feature["scorer_only_f0"][:] = 300
    second = align_feature_timing(feature, measurement_delay_ms=20)
    assert np.array_equal(first.reference_sample, second.reference_sample)
    assert np.array_equal(first.common_eligible, second.common_eligible)


@pytest.mark.parametrize("endpoints", [
    [2208, 2688, 3648, 4128, 4608, 5088],
    [2208, 2208, 3168, 3648, 4128, 4608],
    [2209, 2689, 3169, 3649, 4129, 4609],
])
def test_rejects_gaps_duplicates_and_silent_phase_changes(endpoints):
    feature = feature_fixture()
    feature["reference_sample"] = np.array(endpoints)
    feature["pcm_sample_count"] = np.array(10000)
    with pytest.raises(ValueError):
        align_feature_timing(feature, measurement_delay_ms=20)


def test_rejects_unfrozen_delay_search_and_incomplete_support():
    feature = feature_fixture()
    with pytest.raises(ValueError, match="frozen"):
        align_feature_timing(feature, measurement_delay_ms=10)
    feature["reference_sample"][0] = 1728
    with pytest.raises(ValueError, match="complete support"):
        align_feature_timing(feature, measurement_delay_ms=20)
