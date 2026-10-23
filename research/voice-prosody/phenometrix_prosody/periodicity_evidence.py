"""Input-only primitive-period evidence on the existing 40-ms acquired window.

Frequencies MUST be the grouped selector's emitted member frequencies, not raw
slots subsequently merged by a source maximum. This module receives no labels,
predictions, hidden state, or additional samples. Correlations are evidence,
not calibrated correctness probabilities or a hard pitch veto.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np

SCHEMA = 'phenometrix.candidate-periodicity-evidence.v1'
SAMPLE_RATE = 48_000
WINDOW_SAMPLES = 1_920
MAX_CANDIDATES = 7
ENERGY_FLOOR = 1e-12
PERIOD_FACTORS = (1.0, .5, 1.0 / 3.0, .25)
FEATURE_NAMES = ('period_correlation', 'half_primitive_period_gap', 'periodicity_valid')
CORRELATION_NAMES = ('period', 'half_period', 'third_period', 'quarter_period')
DEFAULT_CHUNK_FRAMES = 64


@dataclass(frozen=True)
class PeriodicityEvidence:
    features: np.ndarray
    correlations: np.ndarray
    valid: np.ndarray
    period_samples: np.ndarray
    common_length: np.ndarray

    @property
    def channels(self):
        return self.features

    @property
    def anchor_length(self):
        return self.common_length


def _mask(value):
    value = np.asarray(value)
    if not (value.dtype == np.bool_ or np.all((value == 0) | (value == 1))):
        raise ValueError('Candidate mask must contain only boolean/zero/one values')
    return value.astype(bool, copy=True)


def _empty(shape):
    return PeriodicityEvidence(np.zeros((*shape, 3), np.float32),
        np.zeros((*shape, 4), np.float64), np.zeros(shape, bool),
        np.zeros(shape, np.float64), np.zeros(shape, np.int32))


def periodicity_evidence(windows, candidate_hz, candidate_mask, *,
                         chunk_frames=DEFAULT_CHUNK_FRAMES) -> PeriodicityEvidence:
    """Compute [...,K,3] channels from [...,1920] Float32 windows and grouped Hz.

    K is bounded to 6 or 7, matching the study model; empty sets use false masks.
    T=48000/f; L=1920-floor(T)-1. For every q in {1,1/2,1/3,1/4}, use
    x[0:L] and linearly interpolated x[i+qT], after removing the FULL window
    mean in Float64. All four comparisons use identical anchor indices. Each
    correlation is sum(a*b)/sqrt(sum(a*a)*sum(b*b)), clipped to [-1,1].
    Anchor and ALL FOUR shifted energies must be >=1e-12. Otherwise validity,
    correlations, and model channels are zero; period/length metadata remains
    available for an active candidate. Padding has zero metadata as well.
    Channels are Float32 [C(T), (C(T)-max(C(T/2),C(T/3),C(T/4)))/2, valid].

    Vectorization only changes summation order versus the independent scalar
    browser equation; every sum/interpolation remains Float64. No FFT, fitted
    scaling, frequency-specific rule, extra support or candidate is introduced.
    """
    if not isinstance(chunk_frames, int) or isinstance(chunk_frames, bool) or chunk_frames < 1 or chunk_frames > 1024:
        raise ValueError('chunk_frames must be an integer in [1,1024]')
    pcm = np.asarray(windows, dtype=np.float32)
    hz = np.asarray(candidate_hz, dtype=np.float64)
    mask = _mask(candidate_mask)
    if (pcm.ndim < 1 or pcm.shape[-1] != WINDOW_SAMPLES or hz.ndim != pcm.ndim
            or hz.shape[:-1] != pcm.shape[:-1] or hz.shape[-1] not in (6, 7)
            or mask.shape != hz.shape):
        raise ValueError('Expected windows [...,1920] and matching candidates/mask [...,6 or 7]')
    if not np.isfinite(pcm).all():
        raise ValueError('Window PCM must be finite')
    if not np.isfinite(hz[mask]).all() or np.any((hz[mask] < 50) | (hz[mask] > 700)):
        raise ValueError('Active candidate Hz must be finite and within [50,700]')
    result = _empty(hz.shape)
    k = hz.shape[-1]
    if not mask.any():
        return result
    flat_pcm = pcm.reshape(-1, WINDOW_SAMPLES)
    flat_hz, flat_mask = hz.reshape(-1, k), mask.reshape(-1, k)
    features = result.features.reshape(-1, k, 3)
    correlations = result.correlations.reshape(-1, k, 4)
    valid = result.valid.reshape(-1, k)
    periods = result.period_samples.reshape(-1, k)
    lengths = result.common_length.reshape(-1, k)
    for start in range(0, len(flat_pcm), chunk_frames):
        stop = min(start + chunk_frames, len(flat_pcm))
        rows, slots = np.nonzero(flat_mask[start:stop])
        if not len(rows):
            continue
        x = flat_pcm[start:stop].astype(np.float64)
        x -= x.mean(axis=-1, keepdims=True)
        t = SAMPLE_RATE / flat_hz[start + rows, slots]
        length = WINDOW_SAMPLES - np.floor(t).astype(np.int32) - 1
        # Fixed reduction extent keeps arithmetic independent of chunk/batch peers.
        index = np.arange(WINDOW_SAMPLES, dtype=np.int32)
        included = index[None, :] < length[:, None]
        anchor = x[rows[:, None], index[None, :]]
        anchor = np.where(included, anchor, 0.0)
        anchor_energy = np.sum(anchor * anchor, axis=-1, dtype=np.float64)
        energy_ok = anchor_energy >= ENERGY_FLOOR
        corr = np.zeros((len(rows), 4), np.float64)
        for j, factor in enumerate(PERIOD_FACTORS):
            lag = t * factor
            floor_lag = np.floor(lag).astype(np.int32)
            fraction = lag - floor_lag
            shifted_index = index[None, :] + floor_lag[:, None]
            # Clipping affects only unused padded positions. Every included
            # interpolation pair lies strictly within this acquired window.
            left = x[rows[:, None], np.minimum(shifted_index, WINDOW_SAMPLES - 1)]
            right = x[rows[:, None], np.minimum(shifted_index + 1, WINDOW_SAMPLES - 1)]
            shifted = (1.0 - fraction[:, None]) * left + fraction[:, None] * right
            shifted = np.where(included, shifted, 0.0)
            shifted_energy = np.sum(shifted * shifted, axis=-1, dtype=np.float64)
            energy_ok &= shifted_energy >= ENERGY_FLOOR
            denominator = np.sqrt(anchor_energy * shifted_energy)
            numerator = np.sum(anchor * shifted, axis=-1, dtype=np.float64)
            np.divide(numerator, denominator, out=corr[:, j], where=denominator > 0)
        corr = np.clip(corr, -1.0, 1.0)
        corr[~energy_ok] = 0.0
        values = np.column_stack((corr[:, 0], (corr[:, 0] - corr[:, 1:].max(axis=-1)) / 2.0,
                                  energy_ok.astype(np.float64))).astype(np.float32)
        correlations[start + rows, slots] = corr
        features[start + rows, slots] = values
        valid[start + rows, slots] = energy_ok
        periods[start + rows, slots] = t
        lengths[start + rows, slots] = length
    return result


def evidence_from_pcm(pcm, end_samples, candidate_hz, candidate_mask, *,
                      chunk_frames=DEFAULT_CHUNK_FRAMES) -> PeriodicityEvidence:
    """Exact [end-1920,end) extraction, preserving supplied frame order.

    Pre-support frames are permitted only with no active candidates and return
    all-zero evidence. PCM after a requested endpoint is not inspected by that
    frame. No padding fabricates support and endpoints cannot exceed the PCM.
    """
    samples = np.asarray(pcm, dtype=np.float32)
    ends = np.asarray(end_samples)
    hz = np.asarray(candidate_hz, dtype=np.float64)
    mask = _mask(candidate_mask)
    if (samples.ndim != 1 or ends.ndim != 1 or ends.dtype.kind not in 'iu'
            or hz.ndim != 2 or hz.shape[0] != len(ends) or hz.shape[-1] not in (6, 7)
            or mask.shape != hz.shape or np.any(ends < 0) or np.any(ends > len(samples))):
        raise ValueError('Expected 1-D PCM, integer endpoints, and aligned [frames,6 or 7] candidates')
    if not isinstance(chunk_frames, int) or isinstance(chunk_frames, bool) or not 1 <= chunk_frames <= 1024:
        raise ValueError('chunk_frames must be an integer in [1,1024]')
    supported = ends >= WINDOW_SAMPLES
    if mask[~supported].any():
        raise ValueError('Active candidate requires a complete acquired 1920-sample window')
    result = _empty(hz.shape)
    rows = np.flatnonzero(supported)
    for start in range(0, len(rows), chunk_frames):
        selected = rows[start:start + chunk_frames]
        windows = np.stack([samples[int(ends[i]) - WINDOW_SAMPLES:int(ends[i])] for i in selected])
        block = periodicity_evidence(windows, hz[selected], mask[selected], chunk_frames=chunk_frames)
        for name in result.__dataclass_fields__:
            getattr(result, name)[selected] = getattr(block, name)
    return result


def evidence_contract():
    return dict(schema=SCHEMA, sample_rate=SAMPLE_RATE, window_samples=WINDOW_SAMPLES,
        feature_names=list(FEATURE_NAMES), correlation_names=list(CORRELATION_NAMES),
        period_factors=list(PERIOD_FACTORS), energy_floor=ENERGY_FLOOR,
        candidate_frequency='actual grouped emitted member Hz, 50..700 inclusive',
        common_length='1920-floor(48000/f)-1', centering='full 1920-sample Float64 mean',
        interpolation='(1-fraction)*x[index]+fraction*x[index+1]',
        arithmetic='Float32 acquisition; Float64 interpolation/normalization; Float32 model channels',
        invalid='zero correlations/channels, explicit false validity; active period/length retained',
        extra_lookahead_samples=0, extra_measurement_delay_ms=0,
        maximum_interpolated_pairs_per_frame=MAX_CANDIDATES * 4 * WINDOW_SAMPLES,
        feature_scaling='fixed bounded channels; no learned scaling')


def source_hashes():
    path = Path(__file__).resolve()
    return {str(path.relative_to(path.parents[3])): hashlib.sha256(path.read_bytes()).hexdigest()}
