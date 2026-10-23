"""Frozen objective-only alias experiment; inference and candidate support unchanged.

References enter through training PitchTargets only. The alias relation compares
actual emitted frequencies with ANY acceptable emitted candidate, not the raw
reference or an unselected member of a near-duplicate group. Alias and all-margin
arms use different eligible-row averages as well as different negative sets;
this is an explicit part of the experiment, not an isolated factor comparison.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .selector_model import PitchTargets, SelectorBatch, selector_loss

ARMS = ("set", "all-margin", "alias-margin")
MARGIN = 1.0
MARGIN_WEIGHT = 0.5
ALIAS_FACTORS = (2.0, 3.0, 4.0, 0.5, 1.0 / 3.0, 0.25)
ALIAS_TOLERANCE_CENTS = 50.0
SCHEMA = "phenometrix.subharmonic-objective.v1"


def _validate_types(candidate_mask: Tensor, targets: PitchTargets) -> None:
    if candidate_mask.dtype != torch.bool or targets.acceptable.dtype != torch.bool or targets.covered.dtype != torch.bool:
        raise ValueError("Candidate, acceptable and covered masks must be boolean")
    if targets.nearest.dtype != torch.int64:
        raise ValueError("Nearest target indices must be int64")
    if torch.any((~targets.covered) & (targets.nearest != -1)):
        raise ValueError("Uncovered rows require the explicit -1 nearest sentinel")


def build_negative_masks(batch: SelectorBatch, targets: PitchTargets) -> dict[str, Tensor]:
    """Build detached CPU bool masks using CPU float64 emitted-Hz metadata.

    The symmetric factor relation is inclusive at 50 cents in float64 arithmetic.
    All negatives must be available, nonacceptable and on a covered row. A row
    with no acceptable candidate has no margin target. No reference is invented
    for an uncovered, unvoiced or invalid row.
    """
    hz = batch.candidate_hz.detach().cpu().double()
    mask = batch.candidate_mask.detach().cpu()
    labels = targets.to("cpu")
    _validate_types(mask, labels)
    if hz.shape != mask.shape:
        raise ValueError("Emitted frequencies and candidate mask disagree")
    if torch.any(mask & (~torch.isfinite(hz) | (hz <= 0))):
        raise ValueError("Available emitted frequencies must be finite and positive")
    # Reuse the frozen objective's shape, support and nearest-target checks.
    selector_loss(torch.zeros_like(hz), mask, labels, objective="acceptable-set")
    negative = mask & ~labels.acceptable & labels.covered.unsqueeze(-1)
    safe_hz = torch.where(mask, hz, torch.ones_like(hz))
    cents = 1200.0 * torch.log2(safe_hz.unsqueeze(-1) / safe_hz.unsqueeze(-2))
    # Axis -2 is the potential negative, axis -1 the acceptable anchor.
    pairs = negative.unsqueeze(-1) & labels.acceptable.unsqueeze(-2)
    related = torch.zeros_like(pairs)
    for factor in ALIAS_FACTORS:
        related |= torch.abs(cents - 1200.0 * math.log2(factor)) <= ALIAS_TOLERANCE_CENTS
    alias = (related & pairs).any(dim=-1)
    return {"all-margin": negative, "alias-margin": alias}


def subharmonic_loss(
    logits: Tensor,
    candidate_mask: Tensor,
    targets: PitchTargets,
    negative_mask: Tensor | None,
    *,
    arm: str,
) -> Tensor:
    """Set CE plus 0.5 times a fixed one-logit margin on eligible covered rows.

    The auxiliary mean includes rows having at least one eligible negative,
    including rows already satisfying the margin (whose hinge is zero). `amax`
    shares gradients across tied extrema. Empty eligibility returns connected
    zero and never creates a false target. `set` is exactly the frozen set loss.
    """
    if arm not in ARMS:
        raise ValueError("Unknown subharmonic objective arm")
    _validate_types(candidate_mask, targets)
    base = selector_loss(logits, candidate_mask, targets, objective="acceptable-set")
    if arm == "set":
        return base
    if negative_mask is None or negative_mask.dtype != torch.bool or negative_mask.shape != logits.shape:
        raise ValueError("Margin arms require a boolean negative mask matching logits")
    if negative_mask.device != logits.device:
        raise ValueError("Negative mask and logits must be on the same device")
    allowed = candidate_mask & ~targets.acceptable & targets.covered.unsqueeze(-1)
    if torch.any(negative_mask & ~allowed):
        raise ValueError("Negative targets must be available, nonacceptable and covered")
    if arm == "all-margin" and not torch.equal(negative_mask, allowed):
        raise ValueError("All-margin must include every available nonacceptable candidate")
    eligible = targets.covered & negative_mask.any(dim=-1)
    if not bool(eligible.any()):
        return base + logits.sum() * 0.0
    selected = logits[eligible]
    good = torch.where(targets.acceptable[eligible], selected, torch.full_like(selected, -torch.inf))
    bad = torch.where(negative_mask[eligible], selected, torch.full_like(selected, -torch.inf))
    auxiliary = torch.relu(MARGIN + torch.amax(bad, dim=-1) - torch.amax(good, dim=-1)).mean()
    return base + MARGIN_WEIGHT * auxiliary
