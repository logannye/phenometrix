"""Fixed analytic fixtures only; no training, held-out inputs or model outcomes."""
import math
import numpy as np
import pytest

from phenometrix_prosody.periodicity_evidence import (
    periodicity_evidence as kernel, evidence_from_pcm as pcm_kernel, evidence_contract,
    PeriodicityEvidence,
    WINDOW_SAMPLES, SAMPLE_RATE, ENERGY_FLOOR,
)


def _padded(hz, mask):
    hz, mask = np.asarray(hz), np.asarray(mask)
    k = hz.shape[-1]
    if k >= 6:
        return hz, mask, k
    widths = [(0, 0)] * (hz.ndim - 1) + [(0, 6 - k)]
    return np.pad(hz.astype(float), widths, constant_values=np.nan), np.pad(mask, widths), k


def _slice(result, k):
    return PeriodicityEvidence(result.features[..., :k, :], result.correlations[..., :k, :],
        result.valid[..., :k], result.period_samples[..., :k], result.common_length[..., :k])


def periodicity_evidence(windows, hz, mask, **kwargs):
    # Small scalar fixtures still enter the real kernel with its declared K6/7.
    hz, mask, k = _padded(hz, mask)
    return _slice(kernel(windows, hz, mask, **kwargs), k)


def evidence_from_pcm(pcm, ends, hz, mask, **kwargs):
    hz, mask, k = _padded(hz, mask)
    return _slice(pcm_kernel(pcm, ends, hz, mask, **kwargs), k)


def tone(f=200., amplitude=.1, phase=.13, length=1920):
    t = np.arange(length, dtype=np.float64) / SAMPLE_RATE
    return (amplitude * (np.sin(2 * np.pi * f * t + phase)
                        + .5 * np.sin(4 * np.pi * f * t + .4))).astype(np.float32)


def scalar_reference(window, f):
    # Separate scalar loop/f-sum arithmetic, no production helper or labels.
    x = [float(v) for v in window]
    mean = math.fsum(x) / len(x)
    x = [v - mean for v in x]
    period = 48000 / float(f)
    length = 1920 - math.floor(period) - 1
    a = x[:length]
    energy = math.fsum(v * v for v in a)
    values, energies = [], [energy]
    for divisor in (1., 2., 3., 4.):
        lag = period / divisor
        offset, fraction = math.floor(lag), lag - math.floor(lag)
        b = [(1 - fraction) * x[i + offset] + fraction * x[i + offset + 1] for i in range(length)]
        other = math.fsum(v * v for v in b)
        energies.append(other)
        values.append(max(-1., min(1., math.fsum(u * v for u, v in zip(a, b)) / math.sqrt(energy * other)))
                      if energy * other > 0 else 0.)
    valid = all(e >= ENERGY_FLOOR for e in energies)
    c = np.array(values if valid else [0.] * 4)
    return c, np.array([c[0], (c[0] - max(c[1:])) / 2, float(valid)], np.float32), valid, period, length


@pytest.mark.parametrize('frequency', [50., 100., 137.25, 200., 333.123456789, 700.])
def test_fractional_period_matches_independent_scalar_equation(frequency):
    window = tone(193.75)
    actual = periodicity_evidence(window, [frequency], [True])
    c, features, valid, period, length = scalar_reference(window, frequency)
    np.testing.assert_allclose(actual.correlations[0], c, rtol=0, atol=2e-13)
    np.testing.assert_allclose(actual.features[0], features, rtol=0, atol=1e-7)
    assert actual.valid[0] == valid
    assert actual.period_samples[0] == period and actual.common_length[0] == length
    assert actual.features.dtype == np.float32 and actual.correlations.dtype == np.float64


def test_primitive_period_contrast_distinguishes_longer_repeated_period():
    # This is a known-answer harmonic fixture, not a scientific repair claim.
    evidence = periodicity_evidence(tone(200), [100., 200.], [True, True])
    assert evidence.correlations[:, 0].min() > .99999
    assert abs(evidence.features[0, 1]) < 1e-6
    assert evidence.features[1, 1] > .2


@pytest.mark.parametrize('constant', [0., .25, -1.])
def test_silence_and_dc_return_explicit_invalid_zero_evidence(constant):
    result = periodicity_evidence(np.full(1920, constant, np.float32), [200., np.nan], [True, False])
    assert not result.valid.any() and not result.features.any() and not result.correlations.any()
    assert result.period_samples.tolist() == [240., 0.]
    assert result.common_length.tolist() == [1679, 0]


def test_low_energy_threshold_and_scale_dc_invariance_above_it():
    assert not periodicity_evidence(tone(amplitude=1e-8), [200.], [1]).valid.item()
    assert periodicity_evidence(tone(amplitude=1e-7), [200.], [1]).valid.item()
    window = tone()
    baseline = periodicity_evidence(window, [100., 200., 350.], [1, 1, 1])
    changed = periodicity_evidence(window * 2 + np.float32(.25), [100., 200., 350.], [1, 1, 1])
    np.testing.assert_allclose(changed.correlations, baseline.correlations, rtol=0, atol=2e-7)
    np.testing.assert_array_equal(changed.valid, baseline.valid)


def test_anchor_energy_failure_zeroes_all_four_comparisons():
    window = np.zeros(1920, np.float32)
    window[-2:] = [1., -1.]
    out = periodicity_evidence(window, [50.], [True])
    assert not out.valid.item() and not out.features.any() and not out.correlations.any()


def test_permutation_chunking_batch_order_and_input_immutability():
    windows = np.stack([tone(200), tone(250), tone(175)])
    hz = np.array([[100., 200., 400., np.nan], [125., 250., np.nan, np.nan], [50., 175., 350., 700.]])
    mask = np.isfinite(hz)
    before = [a.copy() for a in (windows, hz, mask)]
    out = periodicity_evidence(windows, hz, mask, chunk_frames=1)
    other = periodicity_evidence(windows, hz, mask, chunk_frames=64)
    for name in out.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(out, name), getattr(other, name))
    perm = [3, 1, 0, 2]
    shuffled = periodicity_evidence(windows[[2, 0, 1]], hz[[2, 0, 1]][:, perm], mask[[2, 0, 1]][:, perm])
    for name in out.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(shuffled, name), getattr(out, name)[[2, 0, 1]][:, perm])
    for a, b in zip((windows, hz, mask), before): np.testing.assert_array_equal(a, b)


def test_empty_sets_padding_and_duplicate_hz_are_exact():
    empty = periodicity_evidence(tone(), np.empty(0), np.empty(0, bool))
    assert empty.features.shape == (0, 3)
    out = periodicity_evidence(tone(), [200., 200., np.nan], [True, True, False])
    np.testing.assert_array_equal(out.features[0], out.features[1])
    for name in out.__dataclass_fields__: assert not getattr(out, name)[2].any()
    rows = periodicity_evidence(np.empty((0, 1920), np.float32), np.empty((0, 7)), np.empty((0, 7), bool))
    assert rows.features.shape == (0, 7, 3)


def test_exact_support_excludes_future_even_nonfinite_future_and_preserves_order():
    pcm = tone(length=4800)
    ends = np.array([2880, 1920, 3840])
    hz = np.full((3, 1), 200.)
    actual = evidence_from_pcm(pcm, ends, hz, np.ones_like(hz, bool), chunk_frames=2)
    direct = periodicity_evidence(np.stack([pcm[e - 1920:e] for e in ends]), hz, np.ones_like(hz, bool))
    for name in actual.__dataclass_fields__: np.testing.assert_array_equal(getattr(actual, name), getattr(direct, name))
    modified = pcm.copy(); modified[1920:] = np.nan
    limited = evidence_from_pcm(modified, [1920], [[200.]], [[True]])
    np.testing.assert_array_equal(limited.features[0], actual.features[1])
    # A changed observed sample CAN affect the result; this is not a constant stub.
    changed = pcm.copy(); changed[1000] += .2
    assert not np.array_equal(evidence_from_pcm(changed, [1920], [[200.]], [[True]]).correlations,
                              limited.correlations)


def test_pre_support_is_unavailable_without_fabricated_padding():
    pcm = tone(length=1920)
    result = evidence_from_pcm(pcm, [768, 1248, 1728, 1920],
        [[np.nan], [np.nan], [np.nan], [200.]], [[False], [False], [False], [True]])
    assert not result.features[:3].any() and result.valid[3, 0]
    with pytest.raises(ValueError, match='complete acquired'):
        evidence_from_pcm(pcm, [1728], [[200.]], [[True]])


@pytest.mark.parametrize('hz,mask', [([49.], [1]), ([701.], [1]), ([np.nan], [1]),
                                    ([200.], [2]), ([200.], [np.nan]), ([200.] * 8, [1] * 8)])
def test_invalid_candidate_inputs_refused(hz, mask):
    with pytest.raises(ValueError): periodicity_evidence(tone(), hz, mask)


@pytest.mark.parametrize('window', [np.zeros(1919), np.full(1920, np.nan), np.full(1920, np.inf)])
def test_bad_window_refused(window):
    with pytest.raises(ValueError): periodicity_evidence(window, [200.], [True])


@pytest.mark.parametrize('ends', [[1920.], [-1], [1921]])
def test_bad_endpoint_refused(ends):
    with pytest.raises(ValueError): evidence_from_pcm(tone(), ends, [[200.]], [[True]])


@pytest.mark.parametrize('chunk', [0, -1, 1025, 1.5, True])
def test_chunk_bound_refused(chunk):
    with pytest.raises(ValueError): periodicity_evidence(tone(), [200.], [True], chunk_frames=chunk)


def test_contract_reports_operation_bound_not_timing_claim():
    contract = evidence_contract()
    assert contract['maximum_interpolated_pairs_per_frame'] == 53760
    assert contract['extra_lookahead_samples'] == contract['extra_measurement_delay_ms'] == 0


@pytest.mark.parametrize('k', [0, 1, 5, 8])
def test_numeric_kernel_requires_six_or_seven_slots(k):
    with pytest.raises(ValueError):
        kernel(tone(), np.full(k, 200.), np.ones(k, bool))
