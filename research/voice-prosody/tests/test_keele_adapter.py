"""Synthetic-only KEELE adapter checks; no corpus/media dependency."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
from phenometrix_prosody.keele_adapter import (
    intended_timing, join_support_ends, rational_time_to_sample,
    read_reference_npy, reference_from_arrays, resample_microphone, timing_for_targets, validate_metadata_binding,
)


def fixture(pitch=None, **changes):
    raw = np.array([100.] * 11 if pitch is None else pitch)
    args = dict(source_id="synthetic", reference_source_id="synthetic",
        time_seconds=np.arange(len(raw)) / 100, pitch_hz=raw,
        pcm_sample_count=6000, expected_reference_rows=len(raw))
    args.update(changes)
    return reference_from_arrays(**args)


def test_exact_rational_grid_and_first_eligible_target():
    assert rational_time_to_sample(7, 100) == 3360
    timing = intended_timing(6000)
    assert timing.target_sample[timing.complete_support][0] == 960
    assert timing.support_end_sample[timing.complete_support][0] == 1920
    assert timing.target_sample[timing.common_eligible].tolist() == [3360, 3840, 4320]
    assert not timing.target_sample.flags.writeable


@pytest.mark.parametrize("num,den", [(16, 1000), (1, 48000), (1, 3), (-1, 100), (0, 0), (True, 100)])
def test_no_rounding_to_nearest_provider_frame(num, den):
    with pytest.raises(ValueError): rational_time_to_sample(num, den)


def test_end_masks_use_exact_finite_duration_and_integer_inequalities():
    target = np.array([3360])
    assert timing_for_targets(target, 4896).common_eligible.tolist() == [True]
    assert timing_for_targets(target, 4895).common_eligible.tolist() == [False]
    with pytest.raises(ValueError, match="outside microphone"):
        timing_for_targets(np.array([4800]), 4800)


def test_classification_preserves_uncertainty_invalid_and_out_of_range():
    ref = fixture([0, -1, -20000, -125, float("nan"), float("inf"), -float("inf"), 49, 50, 700, 701])
    assert ref.status.tolist() == ["unvoiced", "uncertain", "uncertain", "uncertain", "invalid", "invalid", "invalid", "voiced", "voiced", "voiced", "voiced"]
    assert ref.out_of_range.tolist() == [False] * 7 + [True, False, False, True]
    assert ref.scoring_eligible.tolist() == [False] * 7 + [True, True, True, False]
    assert ref.counts() == dict(voiced=4, unvoiced=1, uncertain=3, invalid=3,
        total=11, boundary=8, out_of_range=2, eligible_reference=3)


def test_zero_eligible_reference_denominator_stays_explicit():
    ref = fixture([-20000.] * 11)
    assert ref.counts()["eligible_reference"] == 0
    assert ref.counts()["uncertain"] == 11
    assert not ref.scoring_eligible.any()


@pytest.mark.parametrize("changes", [
    {"reference_source_id": "other"}, {"source_id": ""},
    {"expected_reference_rows": 10}, {"expected_reference_rows": 0},
    {"time_seconds": np.arange(11) / 100 + .016},
    {"time_seconds": np.arange(11) / 100 + 1e-14},
    {"time_seconds": np.arange(11).astype(np.float32) * np.float32(.01)},
    {"pitch_hz": [[100] * 11]}, {"pitch_hz": ["100"] * 11},
    {"pcm_sample_count": 4800},
])
def test_malformed_and_mismatched_references_fail_closed(changes):
    with pytest.raises(ValueError): fixture(**changes)


def test_converter_period_sentinel_transformation_is_not_voicing():
    periods = np.array([200, -200, -1, 0])
    converted = np.array([0 if p == 0 else 1 / (int(p) * 1 / 20000) for p in periods])
    assert converted.tolist() == [100, -100, -20000, 0]
    assert fixture(converted).status.tolist() == ["voiced", "uncertain", "uncertain", "unvoiced"]


def test_exact_source_origin_and_restart_join():
    joined = join_support_ends(np.array([2400, 2880]), source_offset=480,
        first_admitted_block_start=480, pcm_sample_count=6000)
    assert joined.target_sample.tolist() == [960, 1440]
    assert joined.target_sample.tolist() != [1920, 2400]  # endpoint index is two hops ahead
    restarted = join_support_ends(np.array([3840]), source_offset=480,
        first_admitted_block_start=1920, pcm_sample_count=6000)
    assert restarted.target_sample.tolist() == [2400]
    for first, end in [(481, 2401), (1920, 3360), (480, 2401)]:
        with pytest.raises(ValueError):
            join_support_ends(np.array([end]), source_offset=480,
                first_admitted_block_start=first, pcm_sample_count=6000)


@pytest.mark.parametrize("targets", [[480, 480], [960, 480], [481], [-480]])
def test_duplicate_backward_offgrid_targets_fail(targets):
    with pytest.raises(ValueError): timing_for_targets(np.array(targets), 6000)


def test_structured_file_digest_and_schema(tmp_path):
    path = tmp_path / "synthetic.npy"
    array = np.rec.fromarrays([np.arange(11) / 100, np.ones(11) * 100], names=["time", "pitch"])
    np.save(path, array)
    args = dict(source_id="synthetic", reference_source_id="synthetic", pcm_sample_count=6000,
        expected_reference_rows=11, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert read_reference_npy(path, **args).counts()["voiced"] == 11
    with pytest.raises(ValueError, match="digest"):
        read_reference_npy(path, **{**args, "expected_sha256": "0" * 64})
    for malformed in [np.ones((11, 2)), np.array([{}], dtype=object)]:
        np.save(path, malformed)
        with pytest.raises(ValueError):
            read_reference_npy(path, **{**args, "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    path.write_bytes(b"not an npy file")
    with pytest.raises(ValueError):
        read_reference_npy(path, **{**args, "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def test_impulse_origin_interior_index_and_ceil_endpoint():
    for index in (0, 5, 25):
        pcm = np.zeros(51); pcm[index] = 1
        result, receipt = resample_microphone(pcm)
        assert len(result) == 123
        assert np.argmax(result) == index * 12 // 5
        assert result.dtype == np.float32
        assert receipt["source_zero_output_sample"] == 0
        assert receipt["filter"]["taps"] == 481
        assert hashlib.sha256(result.tobytes()).hexdigest() == receipt["output_f32le_sha256"]


def test_resampling_matches_independent_fir_sum_on_index_fixture():
    # Independent direct convolution/index arithmetic, no resample_poly call here.
    from scipy.signal import firwin
    pcm = np.arange(53, dtype=np.float64) / 53
    actual, _ = resample_microphone(pcm)
    h = firwin(481, 1 / 12, window=("kaiser", 5.0))
    expected = []
    for m in range((len(pcm) * 12 + 4) // 5):
        total = sum(x * 12 * h[m * 5 - n * 12 + 240]
            for n, x in enumerate(pcm) if 0 <= m * 5 - n * 12 + 240 < len(h))
        expected.append(total)
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float32), atol=1e-7, rtol=1e-7)
    repeated, receipt = resample_microphone(pcm)
    np.testing.assert_array_equal(actual, repeated)
    assert receipt["offline_preparation"] and not receipt["clipping_or_normalization"]


@pytest.mark.parametrize("pcm,rate", [([],20000), ([[0,1]],20000), ([np.nan],20000), ([1],48000)])
def test_invalid_microphone_contract(pcm, rate):
    with pytest.raises(ValueError): resample_microphone(pcm, source_rate_hz=rate)


def test_pinned_converter_division_not_float_multiplication():
    ref = fixture([100.] * 101, pcm_sample_count=60000)
    assert ref.timing.target_sample[35] == 16800
    with pytest.raises(ValueError, match="timestamps"):
        fixture([100.] * 101, pcm_sample_count=60000, time_seconds=np.arange(101) * .01)


def test_metadata_only_binding_and_corruption():
    item = dict(SRC="synthetic.pes", BEG="0", SAM="20000", END="2000")
    signal = dict(samplerate=20000, _filename="KEELE/synthetic/signal.wav")
    pitch = {"SRC ": "synthetic.pes", "BEG ": "0", "SAM ": "100", "END ": "10",
             "samplerate": 100, "_filename": "KEELE/synthetic/pitch.npy"}
    args = dict(source_id="synthetic", item_metadata=item, signal_metadata=signal, pitch_metadata=pitch)
    receipt = validate_metadata_binding(**args)
    assert receipt["declared_reference_rows"] == 10
    assert not receipt["data_headers_verified"] and not receipt["external_assessment_admitted"]
    for bad in ({**pitch, "SRC ": "other.pes"}, {**pitch, "END ": "12"},
                {**pitch, "BEG ": "1"}, {**pitch, "SAM ": "200"},
                {**pitch, "SRC": "synthetic.pes"}):
        with pytest.raises(ValueError): validate_metadata_binding(**{**args, "pitch_metadata": bad})
