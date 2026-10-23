"""Add one admitted full40 peak without evicting any original raw hypothesis.

This is a separate research feature contract, not v2 and not an input to the
frozen original voicing model. Cached evidence is permissible only when its
outer acquisition/provenance identities have been verified by the run owner.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import numpy as np

from . import candidate_coverage_features as previous
from .model import FEATURE_NAMES_V2

SCHEMA = 'phenometrix.voice-pitch-features.expanded7.v1'
POLICY = 'add-shortest-admitted-full40-v1'
CANDIDATE_COUNT = 7
FEATURE_COUNT = 65
COUNT_REFERENCE = 6
FEATURE_NAMES_EXPANDED = tuple(
    name for i in range(CANDIDATE_COUNT) for name in (
        f'candidate_{i}_valid', f'candidate_{i}_log2_hz_over_50',
        f'candidate_{i}_autocorrelation_score', f'candidate_{i}_amdf_score',
        f'candidate_{i}_zero_crossing_score', f'candidate_{i}_acf40_refined_score',
        f'candidate_{i}_acf20_refined_score', f'candidate_{i}_maximum_score',
    )
) + FEATURE_NAMES_V2[-9:-2] + ('valid_candidate_count_over_six', FEATURE_NAMES_V2[-1])
FEATURE_NAMES = FEATURE_NAMES_EXPANDED
CoveragePeakWorker = previous.CoveragePeakWorker


def source_hashes():
    result = previous.source_hashes()
    result[str(Path(__file__).relative_to(previous.REPO))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def _evidence(feature, evidence):
    x, hz, mask, ends, count = previous._validate(feature)
    blocks = x[:, :48].reshape(-1, 6, 8)
    if not np.allclose(blocks[:, :, 1][mask], np.log2(hz[mask] / 50), rtol=0, atol=1e-6):
        raise ValueError('Original frequency evidence and candidate Hz disagree')
    if (np.asarray(evidence.get('pcm_sample_count')).shape != ()
            or np.asarray(evidence.get('pcm_sample_count')).dtype.kind not in 'iu'
            or int(evidence['pcm_sample_count']) != count
            or not np.array_equal(evidence.get('ends'), ends)):
        raise ValueError('PCM/cache acquisition identity differs')
    sha = evidence.get('pcm_sha256')
    if not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None:
        raise ValueError('Bound PCM SHA256 is required')
    values = {}
    for label in ('old', 'new'):
        f = np.asarray(evidence[label + '_hz'])
        c = np.asarray(evidence[label + '_confidence'])
        if f.shape != (len(x),) or c.shape != (len(x),) or f.dtype != np.float64 or c.dtype != np.float64:
            raise ValueError('Float64 per-frame peak evidence is required')
        present = np.isfinite(f)
        if (not np.all(present | np.isnan(f)) or np.any((f[present] < 50) | (f[present] > 700))
                or not np.isfinite(c).all() or np.any((c[present] < .35) | (c[present] > 1))
                or np.any(c[~present] != 0)):
            raise ValueError('Invalid admitted peak evidence')
        values[label + '_hz'], values[label + '_confidence'] = f, c
    old, new = values['old_hz'], values['new_hz']
    if not np.array_equal(np.isnan(old), np.isnan(new)):
        raise ValueError('Peak policies must have identical availability')
    # Both policies select from the exact same admitted list. Shortest lag
    # cannot have lower Hz or a greater score than the strongest-score pick.
    present = np.isfinite(old)
    if (np.any(new[present] < old[present])
            or np.any(values['new_confidence'][present] > values['old_confidence'][present])
            or np.any((new[present] == old[present]) &
                      (values['new_confidence'][present] != values['old_confidence'][present]))):
        raise ValueError('Strongest/shortest admitted peak ordering is inconsistent')
    for row, b in enumerate(blocks):
        indices = np.flatnonzero(b[:, 5] > 0)
        if np.isnan(old[row]):
            if len(indices):
                raise ValueError('Cached old full40 evidence does not match original v2')
        elif (len(indices) != 1 or hz[row, indices[0]] != old[row]
                or b[indices[0], 5] != np.float32(values['old_confidence'][row])):
            raise ValueError('Cached old full40 evidence does not match original v2')
    return x, hz, mask, values


def additive_from_evidence(feature, evidence):
    """Return new 65-column feature dict and input-only additive provenance.

    ``evidence`` is the normalized CoveragePeakWorker.inspect result. This
    function checks all frame/peak invariants, but cannot authenticate an outer
    run manifest: callers must bind cached provenance and its PCM/source hashes.
    All old raw Hz remain; only an existing candidate's full40 source channel
    and derived maximum may increase. Newly inserted log-Hz uses NumPy float64
    log2 followed by Float32 conversion in both live and cached routes.
    """
    x, hz, mask, peaks = _evidence(feature, evidence)
    n = len(x)
    expanded = np.zeros((n, FEATURE_COUNT), np.float32)
    expanded[:, 56:] = x[:, 48:]
    output_hz = np.full((n, CANDIDATE_COUNT), np.nan, np.float64)
    output_mask = np.zeros((n, CANDIDATE_COUNT), bool)
    old_to_new = np.full((n, 6), -1, np.int64)
    added = np.zeros(n, bool)
    enhanced = np.zeros(n, bool)
    original_blocks = x[:, :48].reshape(n, 6, 8)
    for row, blocks in enumerate(original_blocks):
        candidates = {float(hz[row, j]): blocks[j].copy() for j in np.flatnonzero(mask[row])}
        new_f, new_c = peaks['new_hz'][row], peaks['new_confidence'][row]
        if np.isfinite(new_f):
            if new_f not in candidates:
                b = np.zeros(8, np.float32)
                b[:2] = [1, np.log2(new_f / 50)]
                candidates[new_f] = b
                added[row] = True
            else:
                enhanced[row] = np.float32(new_c) > candidates[new_f][5]
            candidates[new_f][5] = max(candidates[new_f][5], np.float32(new_c))
        if len(candidates) > CANDIDATE_COUNT:
            raise ValueError('Expanded seven-candidate bound exceeded')
        for j, (f, b) in enumerate(sorted(candidates.items())):
            b[7] = b[2:7].max()
            expanded[row, j * 8:(j + 1) * 8] = b
            output_hz[row, j], output_mask[row, j] = f, True
            original_indices = np.flatnonzero(mask[row] & (hz[row] == f))
            old_to_new[row, original_indices] = j
        expanded[row, 63] = np.float32(len(candidates) / COUNT_REFERENCE)
    result = {key: np.asarray(value).copy() for key, value in feature.items()}
    result.update(features=expanded, candidate_hz=output_hz, candidate_mask=output_mask,
                  feature_names=np.array(FEATURE_NAMES_EXPANDED))
    provenance = {f'{label}_full40_{field}': peaks[f'{label}_{field}'].copy()
                  for label in ('old', 'new') for field in ('hz', 'confidence')}
    provenance.update(candidate_policy=np.array(POLICY), feature_schema=np.array(SCHEMA),
        pcm_sha256=np.array(evidence['pcm_sha256']), original_to_expanded_slot=old_to_new,
        old_candidate_count=mask.sum(axis=1), new_candidate_count=output_mask.sum(axis=1),
        new_frequency_added=added, existing_full40_evidence_enhanced=enhanced,
        old_frequency_evicted=np.zeros(n, bool),
        worker_rss_bytes=np.array(evidence.get('rss_bytes', 0), np.int64))
    return result, provenance


def expanded_feature(feature, pcm_path, worker):
    """Live input-only extraction; preserves the original v2 dict unchanged."""
    _, _, _, ends, _ = previous._validate(feature)
    return additive_from_evidence(feature, worker.inspect(pcm_path, ends))


def expanded_from_provenance(feature, provenance):
    """Reuse completed old-policy evidence after caller verifies its outer hashes.

    The old evidence does not carry endpoints; those come from the original
    cache, whose identity must be matched to the completed prior run by caller.
    No scalar metadata or reference label is used to choose a candidate.
    """
    if (np.asarray(provenance.get('candidate_policy')).shape != ()
            or str(np.asarray(provenance.get('candidate_policy')).item()) != previous.POLICY):
        raise ValueError('Expected historical shortest-admitted provenance policy')
    pcm_sha = np.asarray(provenance.get('pcm_sha256'))
    if pcm_sha.shape != ():
        raise ValueError('Scalar PCM identity required')
    evidence = dict(pcm_sha256=str(pcm_sha.item()),
                    pcm_sample_count=feature['pcm_sample_count'], ends=feature['reference_sample'],
                    rss_bytes=int(np.asarray(provenance.get('worker_rss_bytes', 0))))
    for label in ('old', 'new'):
        for field in ('hz', 'confidence'):
            evidence[f'{label}_{field}'] = np.asarray(provenance[f'{label}_full40_{field}'])
    return additive_from_evidence(feature, evidence)
