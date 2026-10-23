"""Matched-capacity, zero-initialized periodicity residual for the fixed scorer."""
from __future__ import annotations

import hashlib
from pathlib import Path
import torch
from torch import Tensor, nn

from .expanded_selector_model import ExpandedHarmonicSelector, expanded_selector_metadata
from .selector_model import SelectorBatch, MASKED_LOGIT
from .periodicity_evidence import evidence_contract

SCHEMA = 'phenometrix.periodicity-evidence-selector.v1'
EVIDENCE_MODES = ('off', 'periodicity')


class EvidenceHarmonicSelector(ExpandedHarmonicSelector):
    """The original 3,009 weights plus three shared bias-free residual weights.

    Both arms allocate the same tensors. The off arm feeds zeros through the
    branch regardless of supplied evidence. The periodicity arm receives only
    [C(T), half primitive-period gap, validity] at each grouped member frequency.
    All three weights initialize to zero. Base tensors AND caller CPU RNG state
    match constructing the old scorer with the same seed. No new scaling,
    candidate, voicing policy, label input or temporal state is introduced.
    """
    def __init__(self, evidence_mode='off'):
        if evidence_mode not in EVIDENCE_MODES:
            raise ValueError('Unknown periodicity evidence mode')
        super().__init__(context_mode='relative', objective='acceptable-set')
        self.evidence_mode = evidence_mode
        # Creating a zero branch must not perturb the historical base/control
        # initialization or the caller's subsequent random-number stream.
        with torch.random.fork_rng(devices=[]):
            self.evidence_head = nn.Linear(3, 1, bias=False)
            nn.init.zeros_(self.evidence_head.weight)

    def forward(self, batch: SelectorBatch, evidence: Tensor) -> Tensor:
        self._check_batch(batch)
        if (not isinstance(evidence, Tensor) or evidence.shape != batch.candidate_mask.shape + (3,)
                or evidence.dtype != batch.features.dtype or evidence.device != batch.features.device):
            raise ValueError('Expected aligned floating evidence [...,K,3] on the model-input device')
        if not torch.isfinite(evidence).all():
            raise ValueError('Periodicity evidence must be finite')
        validity = evidence[..., 2]
        if (torch.any((validity != 0) & (validity != 1)) or torch.any(evidence[..., :2].abs() > 1)
                or torch.any(evidence[~batch.candidate_mask] != 0)
                or torch.any(evidence[..., :2][validity == 0] != 0)):
            raise ValueError('Invalid periodicity bounds, validity or masked/low-energy padding')
        inputs = evidence if self.evidence_mode == 'periodicity' else torch.zeros_like(evidence)
        logits = super().forward(batch) + self.evidence_head(inputs).squeeze(-1)
        return torch.where(batch.candidate_mask, logits, torch.full_like(logits, MASKED_LOGIT))


def evidence_selector_metadata(model: EvidenceHarmonicSelector):
    result = expanded_selector_metadata(model)
    result.update(schema=SCHEMA, evidence_mode=model.evidence_mode,
        evidence_contract=evidence_contract(),
        evidence_branch='shared bias-free Linear(3,1) added to each unmasked base logit',
        evidence_initialization='three exact zeros; original base initialization and CPU RNG stream unchanged',
        evidence_control='same branch/parameter shapes; off arm inputs exactly zero',
        evidence_scaling='none; analytically bounded [-1,1],[-1,1],{0,1}',
        parameter_count=sum(p.numel() for p in model.parameters()),
        fitted_voicing_unchanged=True, temporal_state=False)
    return result


def source_hashes():
    from .expanded_selector_model import source_hashes as base_sources
    from .periodicity_evidence import source_hashes as evidence_sources
    result = base_sources(); result.update(evidence_sources())
    path = Path(__file__).resolve()
    result[str(path.relative_to(path.parents[3]))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
