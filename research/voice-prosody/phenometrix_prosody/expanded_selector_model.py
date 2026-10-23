"""Same 3,009-parameter harmonic scorer for six or seven candidate inputs.

Only the candidate-axis extent changes. Count remains expressed in original
six-slot units; scaling is fitted exclusively on original six-candidate TRAIN
inputs or restored from that model. Original MLP voicing remains external.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

from .harmonic_selector_model import HarmonicSelector, harmonic_selector_metadata, LOG2_FREQUENCY_RANGE
from .selector_model import (
    SelectorBatch, REPRESENTATIONS, GROUP_TOLERANCE_CENTS, MASKED_LOGIT,
    CANDIDATE_WIDTH, GLOBAL_COUNT, _numpy, prepare_selector_batch,
)
from .expanded_candidate_features import SCHEMA as FEATURE_SCHEMA

SCHEMA = 'phenometrix.expanded-harmonic-selector.v1'
CANDIDATE_COUNTS = (6, 7)
COUNT_REFERENCE = 6


def prepare_expanded_selector_batch(features, candidate_hz, candidate_mask, *,
                                    representation='shared-candidates') -> SelectorBatch:
    """Input-only old grouping algebra, bounded to K=6 or K=7.

    Raw Hz and source evidence are preserved as metadata. Groups use sorted
    greedy complete-span <=5 cents; source maxima, highest-evidence actual
    member, then lower-Hz tie. Adding a candidate can change groups and actual
    representatives, so raw retention does not guarantee grouped coverage.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError('Unknown selector representation')
    raw = _numpy(features, np.float32).copy()
    hz = _numpy(candidate_hz, np.float64).copy()
    mask = _numpy(candidate_mask, np.bool_).copy()
    if raw.ndim < 2 or hz.ndim != raw.ndim or hz.shape[-1] not in CANDIDATE_COUNTS:
        raise ValueError('Expected six or seven candidates with matching leading axes')
    k = hz.shape[-1]
    width = k * CANDIDATE_WIDTH
    if raw.shape[-1] != width + GLOBAL_COUNT or hz.shape != raw.shape[:-1] + (k,) or mask.shape != hz.shape:
        raise ValueError('Expected features [...,8*K+9] and candidates/mask [...,K]')
    if (not np.isfinite(raw).all() or not np.isfinite(hz[mask]).all()
            or np.any((hz[mask] < 50) | (hz[mask] > 700))):
        raise ValueError('Nonfinite evidence or out-of-range candidate frequency')
    blocks = raw[..., :width].reshape(*raw.shape[:-1], k, CANDIDATE_WIDTH)
    if not np.array_equal(blocks[..., 0], mask.astype(np.float32)):
        raise ValueError('Feature validity and candidate mask disagree')
    if not np.allclose(blocks[..., 1][mask], np.log2(hz[mask] / 50), rtol=0, atol=1e-6):
        raise ValueError('Frequency evidence and candidate Hz disagree')
    if (np.any(blocks[~mask] != 0) or np.any((blocks[..., 2:7] < 0) | (blocks[..., 2:7] > 1))
            or np.any(mask & ~np.any(blocks[..., 2:7] > 0, axis=-1))):
        raise ValueError('Invalid padding or candidate source evidence')
    if not np.array_equal(blocks[..., 7], blocks[..., 2:7].max(axis=-1)):
        raise ValueError('Maximum evidence is inconsistent with source scores')
    if not np.array_equal(raw[..., width + 7], (mask.sum(axis=-1) / COUNT_REFERENCE).astype(np.float32)):
        raise ValueError('Candidate count must remain expressed in six-slot units')
    # Dispatching the unchanged function makes the six-slot control exact,
    # including its deterministic representative and Float32 log conversion.
    if k == 6:
        return prepare_selector_batch(raw, hz, mask, representation=representation)
    hz[~mask] = np.nan
    output, out_hz, out_mask = raw.copy(), hz.copy(), mask.copy()
    members = np.zeros((*hz.shape, k), bool)
    flat_blocks = blocks.reshape(-1, k, CANDIDATE_WIDTH)
    flat_hz, flat_mask = hz.reshape(-1, k), mask.reshape(-1, k)
    flat_out = output.reshape(-1, width + GLOBAL_COUNT)
    flat_out_hz, flat_out_mask = out_hz.reshape(-1, k), out_mask.reshape(-1, k)
    flat_members = members.reshape(-1, k, k)
    if representation == 'fixed-slots':
        diagonal = np.arange(k)
        flat_members[:, diagonal, diagonal] = flat_mask
    for row in range(len(flat_blocks) if representation == 'shared-candidates' else 0):
        ordered = sorted(np.flatnonzero(flat_mask[row]), key=lambda i: flat_hz[row, i])
        groups = []
        for index in ordered:
            if not groups or 1200 * np.log2(flat_hz[row, index] / flat_hz[row, groups[-1][0]]) > GROUP_TOLERANCE_CENTS:
                groups.append([int(index)])
            else:
                groups[-1].append(int(index))
        flat_out[row, :width] = 0
        flat_out_hz[row], flat_out_mask[row] = np.nan, False
        for group_index, group in enumerate(groups):
            representative = min(group, key=lambda i: (-float(flat_blocks[row, i, 7]), flat_hz[row, i]))
            scores = flat_blocks[row, group, 2:7].max(axis=0)
            b = flat_out[row, group_index * 8:(group_index + 1) * 8]
            b[:] = [1, np.log2(flat_hz[row, representative] / 50), *scores, float(scores.max())]
            flat_out_hz[row, group_index], flat_out_mask[row, group_index] = flat_hz[row, representative], True
            flat_members[row, group_index, group] = True
        flat_out[row, width + 7] = len(groups) / COUNT_REFERENCE
    return SelectorBatch(torch.from_numpy(output), torch.from_numpy(out_hz), torch.from_numpy(out_mask),
        torch.from_numpy(members), torch.from_numpy(hz), torch.from_numpy(mask), torch.from_numpy(raw), representation)


# Short public alias for batch-building callers; not a replacement for the old API.
prepare = prepare_expanded_selector_batch


class ExpandedHarmonicSelector(HarmonicSelector):
    """Same tensor keys, shapes, initialization, and equations; bounded axis K."""
    def __init__(self, context_mode='relative', objective='acceptable-set'):
        super().__init__(context_mode, objective)

    def _check_batch(self, batch: SelectorBatch):
        if batch.representation != self.config.representation:
            raise ValueError('Batch representation differs from model contract')
        if batch.features.ndim < 2 or batch.candidate_mask.ndim != batch.features.ndim:
            raise ValueError('Malformed selector batch')
        k = batch.candidate_mask.shape[-1]
        if (k not in CANDIDATE_COUNTS or batch.features.shape[-1] != 8 * k + 9
                or batch.candidate_mask.shape != batch.features.shape[:-1] + (k,)
                or batch.candidate_mask.dtype != torch.bool):
            raise ValueError('Malformed six/seven-candidate selector batch')

    def fit_scaling(self, batch: SelectorBatch, *, split: str):
        self._check_batch(batch)
        if batch.candidate_mask.shape[-1] != 6:
            raise ValueError('Scaling must use original six-candidate TRAIN features')
        super().fit_scaling(batch, split=split)

    def _standardized_candidates(self, batch: SelectorBatch) -> Tensor:
        self._check_batch(batch)
        k = batch.candidate_mask.shape[-1]
        values = batch.features[..., :8 * k].reshape(*batch.features.shape[:-1], k, 8)
        return (values - self.candidate_mean) / self.candidate_scale

    def relative_log_ratios(self, batch: SelectorBatch) -> Tensor:
        self._check_batch(batch)
        k = batch.candidate_mask.shape[-1]
        raw = batch.features[..., :8 * k].reshape(*batch.features.shape[:-1], k, 8)
        log_hz = raw[..., 1]
        ratio = (log_hz.unsqueeze(-2) - log_hz.unsqueeze(-1)) / LOG2_FREQUENCY_RANGE
        edges = batch.candidate_mask.unsqueeze(-1) & batch.candidate_mask.unsqueeze(-2)
        return torch.where(edges, ratio, torch.zeros_like(ratio))

    def message_inputs(self, batch: SelectorBatch) -> Tensor:
        candidates = self._standardized_candidates(batch)
        k = batch.candidate_mask.shape[-1]
        neighbors = candidates.unsqueeze(-3).expand(*candidates.shape[:-2], k, k, 8)
        ratio = self.relative_log_ratios(batch)
        if self.context_mode == 'absolute':
            ratio = torch.zeros_like(ratio)
        values = torch.cat((neighbors, ratio.unsqueeze(-1)), dim=-1)
        edges = batch.candidate_mask.unsqueeze(-1) & batch.candidate_mask.unsqueeze(-2)
        return torch.where(edges.unsqueeze(-1), values, torch.zeros_like(values))

    def forward(self, batch: SelectorBatch) -> Tensor:
        candidates = self._standardized_candidates(batch)
        k = batch.candidate_mask.shape[-1]
        embedding = torch.tanh(self.candidate_encoder(candidates))
        context = self.relation_context(batch)
        globals_ = ((batch.features[..., 8 * k:] - self.global_mean) / self.global_scale).unsqueeze(-2)
        globals_ = globals_.expand(*batch.features.shape[:-1], k, GLOBAL_COUNT)
        hidden = torch.tanh(self.shared_hidden(torch.cat((embedding, context, globals_), dim=-1)))
        logits = self.shared_head(hidden).squeeze(-1)
        return torch.where(batch.candidate_mask, logits, torch.full_like(logits, MASKED_LOGIT))


def expanded_selector_metadata(model: ExpandedHarmonicSelector):
    metadata = harmonic_selector_metadata(model)
    metadata.update(schema=SCHEMA, feature_schema=FEATURE_SCHEMA, candidate_counts=list(CANDIDATE_COUNTS),
        feature_counts=[57, 65], group_global_count='groupcount/6 for BOTH widths; may exceed one',
        scaling='original six-candidate TRAIN only, same shared candidate/global buffers across arms',
        original_voicing_unchanged=True, expanded_features_permitted_for_voicing=False,
        raw_pool_preservation='all original raw candidates and source scores retained; grouped coverage audited separately',
        acquisition_information_added=True, parameter_count=sum(p.numel() for p in model.parameters()))
    return metadata


def source_hashes():
    from .expanded_candidate_features import source_hashes as feature_source_hashes
    result = feature_source_hashes()
    root = Path(__file__).resolve().parents[3]
    for name in ('selector_model.py', 'harmonic_selector_model.py', Path(__file__).name):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
