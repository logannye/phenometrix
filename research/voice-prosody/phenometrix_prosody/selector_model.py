"""Pitch-only framewise 2x2 ablation; no fitting or voicing policy lives here.

The shared representation bundles near-duplicate aggregation, shared scaling and
weight sharing. It is not an isolated test of permutation equivariance. Inputs
are unchanged v2 acquisition features. References enter only build_pitch_targets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

REPRESENTATIONS = ("fixed-slots", "shared-candidates")
OBJECTIVES = ("nearest", "acceptable-set")
FEATURE_COUNT = 57
CANDIDATE_COUNT = 6
CANDIDATE_WIDTH = 8
GLOBAL_COUNT = 9
GROUP_TOLERANCE_CENTS = 5.0
ACCEPTABLE_CENTS = 50.0
MASKED_LOGIT = -1e9
SCHEMA = "phenometrix.pitch-selector-ablation.v1"


@dataclass(frozen=True)
class SelectorConfig:
    representation: Literal["fixed-slots", "shared-candidates"] = "fixed-slots"
    loss: Literal["nearest", "acceptable-set"] = "nearest"
    hidden_size: int = 32

    def __post_init__(self):
        if self.representation not in REPRESENTATIONS or self.loss not in OBJECTIVES:
            raise ValueError("Unknown selector representation or objective")
        if self.hidden_size != 32:
            raise ValueError("The bounded ablation fixes both hidden widths at 32")


@dataclass(frozen=True)
class SelectorBatch:
    features: Tensor
    candidate_hz: Tensor
    candidate_mask: Tensor
    member_mask: Tensor
    original_candidate_hz: Tensor
    original_candidate_mask: Tensor
    original_features: Tensor
    representation: str

    def to(self, device) -> "SelectorBatch":
        """Move neural inputs only; float64 Hz/provenance remain on CPU for MPS."""
        return SelectorBatch(self.features.to(device), self.candidate_hz,
            self.candidate_mask.to(device), self.member_mask,
            self.original_candidate_hz, self.original_candidate_mask,
            self.original_features, self.representation)

    def index_select(self, indices: Tensor | np.ndarray) -> "SelectorBatch":
        indices = torch.as_tensor(indices, dtype=torch.int64)
        values = {f.name: (getattr(self, f.name).index_select(0, indices.to(getattr(self, f.name).device))
                          if isinstance(getattr(self, f.name), Tensor) else getattr(self, f.name))
                  for f in fields(self)}
        return SelectorBatch(**values)


@dataclass(frozen=True)
class PitchTargets:
    acceptable: Tensor
    nearest: Tensor
    covered: Tensor

    def to(self, device) -> "PitchTargets":
        return PitchTargets(self.acceptable.to(device), self.nearest.to(device), self.covered.to(device))

    def index_select(self, indices: Tensor | np.ndarray) -> "PitchTargets":
        indices = torch.as_tensor(indices, dtype=torch.int64)
        return PitchTargets(*(value.index_select(0, indices.to(value.device))
                              for value in (self.acceptable, self.nearest, self.covered)))


def _numpy(value, dtype):
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def prepare_selector_batch(features, candidate_hz, candidate_mask, *, representation: str) -> SelectorBatch:
    """Deterministic input-only preprocessing, accepting arbitrary leading axes.

    Fixed slots retain raw v2 order and all 57 values. Shared candidates sort by
    Hz and greedily start a new group when its total min-to-max span exceeds 5
    cents (not adjacent-link chaining). Each source score is merged by maximum.
    The representative is an ORIGINAL member with greatest maximum evidence;
    lower Hz breaks ties. No averaged or reference-selected F0 is introduced.

    Identical duplicates cannot affect shared inputs, including groupcount/6.
    Distinct near-duplicates may change merged evidence and the representative;
    invariance to arbitrary near-duplicate perturbations is not claimed. Every
    raw feature/member is retained separately for provenance and coverage audit.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError("Unknown selector representation")
    raw = _numpy(features, np.float32).copy()
    hz = _numpy(candidate_hz, np.float64).copy()
    mask = _numpy(candidate_mask, np.bool_).copy()
    if raw.ndim < 2 or raw.shape[-1] != FEATURE_COUNT or hz.shape != raw.shape[:-1] + (CANDIDATE_COUNT,) or mask.shape != hz.shape:
        raise ValueError("Expected features [...,57] and candidates/mask [...,6]")
    if not np.isfinite(raw).all() or not np.isfinite(hz[mask]).all() or np.any(hz[mask] <= 0):
        raise ValueError("Nonfinite acquisition evidence or invalid supported frequency")
    blocks = raw[..., :48].reshape(*raw.shape[:-1], CANDIDATE_COUNT, CANDIDATE_WIDTH)
    if not np.array_equal(blocks[..., 0], mask.astype(np.float32)):
        raise ValueError("Feature validity and candidate mask disagree")
    if not np.allclose(blocks[..., 1][mask], np.log2(hz[mask] / 50), rtol=0, atol=1e-6):
        raise ValueError("Frequency evidence and candidate Hz disagree")
    if np.any(blocks[~mask] != 0):
        raise ValueError("Unavailable candidate feature blocks must be zero")
    if not np.allclose(blocks[..., 7][mask], blocks[..., 2:7].max(axis=-1)[mask], rtol=0, atol=1e-6):
        raise ValueError("Maximum evidence is inconsistent with source scores")
    # Missing metadata is explicitly unavailable, never a real 0-Hz candidate.
    hz[~mask] = np.nan
    output = raw.copy()
    output_hz = hz.copy()
    output_mask = mask.copy()
    members = np.zeros((*hz.shape, CANDIDATE_COUNT), bool)
    flat_blocks = blocks.reshape(-1, CANDIDATE_COUNT, CANDIDATE_WIDTH)
    flat_hz, flat_mask = hz.reshape(-1, CANDIDATE_COUNT), mask.reshape(-1, CANDIDATE_COUNT)
    flat_output = output.reshape(-1, FEATURE_COUNT)
    flat_output_hz = output_hz.reshape(-1, CANDIDATE_COUNT)
    flat_output_mask = output_mask.reshape(-1, CANDIDATE_COUNT)
    flat_members = members.reshape(-1, CANDIDATE_COUNT, CANDIDATE_COUNT)
    if representation == "fixed-slots":
        diagonal = np.arange(CANDIDATE_COUNT)
        flat_members[:, diagonal, diagonal] = flat_mask
    for row in range(len(flat_blocks) if representation == "shared-candidates" else 0):
        available = np.flatnonzero(flat_mask[row])
        ordered = sorted(available, key=lambda i: flat_hz[row, i])
        groups: list[list[int]] = []
        for index in ordered:
            if not groups or 1200 * np.log2(flat_hz[row, index] / flat_hz[row, groups[-1][0]]) > GROUP_TOLERANCE_CENTS:
                groups.append([int(index)])
            else:
                groups[-1].append(int(index))
        flat_output[row, :48] = 0
        flat_output_hz[row] = np.nan
        flat_output_mask[row] = False
        for group_index, group in enumerate(groups):
            representative = min(group, key=lambda i: (-float(flat_blocks[row, i, 7]), flat_hz[row, i]))
            scores = flat_blocks[row, group, 2:7].max(axis=0)
            out_block = flat_output[row, group_index * 8:(group_index + 1) * 8]
            out_block[:] = [1, np.log2(flat_hz[row, representative] / 50), *scores, float(scores.max())]
            flat_output_hz[row, group_index] = flat_hz[row, representative]
            flat_output_mask[row, group_index] = True
            flat_members[row, group_index, group] = True
        # This original global field otherwise leaks duplicate multiplicity.
        flat_output[row, 55] = len(groups) / CANDIDATE_COUNT
    return SelectorBatch(torch.from_numpy(output), torch.from_numpy(output_hz), torch.from_numpy(output_mask),
        torch.from_numpy(members), torch.from_numpy(hz), torch.from_numpy(mask), torch.from_numpy(raw), representation)


def build_pitch_targets(batch: SelectorBatch, reference_hz, voiced, valid) -> PitchTargets:
    """CPU float64 labels for actual emitted Hz; no target on uncovered frames.

    Reference validity/common eligibility are supplied by the existing scorer.
    Group membership is never permission to use a reference-nearest member as
    the emitted frequency. A near-duplicate group can therefore lose coverage at
    the 50-cent boundary; callers must audit raw and grouped coverage separately.
    """
    hz = batch.candidate_hz.detach().cpu().double()
    mask = batch.candidate_mask.detach().cpu().bool()
    reference = torch.as_tensor(_numpy(reference_hz, np.float64))
    voiced = torch.as_tensor(_numpy(voiced, np.bool_))
    valid = torch.as_tensor(_numpy(valid, np.bool_))
    if reference.shape != hz.shape[:-1] or voiced.shape != reference.shape or valid.shape != reference.shape:
        raise ValueError("Reference/voiced/valid shapes must equal candidate leading axes")
    include = voiced & valid
    if torch.any(include & (~torch.isfinite(reference) | (reference <= 0))):
        raise ValueError("A valid voiced reference requires finite positive Hz")
    safe_hz = torch.where(mask, hz, torch.ones_like(hz))
    safe_ref = torch.where(include, reference, torch.ones_like(reference))
    errors = torch.abs(1200 * torch.log2(safe_hz / safe_ref.unsqueeze(-1)))
    errors = torch.where(mask & include.unsqueeze(-1), errors, torch.full_like(errors, torch.inf))
    acceptable = (errors <= ACCEPTABLE_CENTS) & mask & include.unsqueeze(-1)
    covered = acceptable.any(dim=-1)
    # Argmin agrees with the historical nearest-index rule on sorted raw slots.
    nearest = errors.argmin(dim=-1)
    nearest = torch.where(covered, nearest, torch.full_like(nearest, -1))
    return PitchTargets(acceptable, nearest, covered)


class PitchSelector(nn.Module):
    """Framewise pitch selection only. Original voicing remains an external head."""
    def __init__(self, config: SelectorConfig = SelectorConfig()):
        super().__init__()
        self.config = config
        self.register_buffer("scaling_fitted", torch.tensor(False))
        if config.representation == "fixed-slots":
            self.register_buffer("feature_mean", torch.zeros(FEATURE_COUNT))
            self.register_buffer("feature_scale", torch.ones(FEATURE_COUNT))
            self.projection = nn.Linear(FEATURE_COUNT, 32)
            self.frame_hidden = nn.Linear(32, 32)
            self.candidate_head = nn.Linear(32, CANDIDATE_COUNT)
        else:
            self.register_buffer("candidate_mean", torch.zeros(CANDIDATE_WIDTH))
            self.register_buffer("candidate_scale", torch.ones(CANDIDATE_WIDTH))
            self.register_buffer("global_mean", torch.zeros(GLOBAL_COUNT))
            self.register_buffer("global_scale", torch.ones(GLOBAL_COUNT))
            self.candidate_encoder = nn.Linear(CANDIDATE_WIDTH, 32)
            self.shared_hidden = nn.Linear(32 + 32 + GLOBAL_COUNT, 32)
            self.shared_head = nn.Linear(32, 1)

    def fit_scaling(self, batch: SelectorBatch, *, split: str) -> None:
        if split != "train":
            raise ValueError("Scaling may only be fitted on the training split")
        self._check_batch(batch)
        x = batch.features.detach().reshape(-1, FEATURE_COUNT)
        mask = batch.candidate_mask.detach().reshape(-1, CANDIDATE_COUNT)
        if len(x) == 0 or not torch.isfinite(x).all():
            raise ValueError("Scaling requires nonempty finite training acquisition features")
        def fit(mean, scale, values):
            mean.copy_(values.mean(dim=0).to(mean))
            scale.copy_(values.std(dim=0, unbiased=False).clamp_min(1e-6).to(scale))
        with torch.no_grad():
            if self.config.representation == "fixed-slots":
                fit(self.feature_mean, self.feature_scale, x)
            else:
                candidates = x[:, :48].reshape(-1, CANDIDATE_COUNT, CANDIDATE_WIDTH)[mask]
                if len(candidates):
                    fit(self.candidate_mean, self.candidate_scale, candidates)
                else:
                    self.candidate_mean.zero_(); self.candidate_scale.fill_(1)
                fit(self.global_mean, self.global_scale, x[:, 48:])
            self.scaling_fitted.fill_(True)

    def _check_batch(self, batch: SelectorBatch):
        if batch.representation != self.config.representation:
            raise ValueError("Batch representation differs from model contract")
        if batch.features.shape[-1] != FEATURE_COUNT or batch.candidate_mask.shape != batch.features.shape[:-1] + (CANDIDATE_COUNT,):
            raise ValueError("Malformed selector batch")

    def forward(self, batch: SelectorBatch) -> Tensor:
        self._check_batch(batch)
        x, mask = batch.features, batch.candidate_mask
        if self.config.representation == "fixed-slots":
            scaled = (x - self.feature_mean) / self.feature_scale
            hidden = torch.tanh(self.frame_hidden(torch.tanh(self.projection(scaled))))
            logits = self.candidate_head(hidden)
        else:
            candidates = x[..., :48].reshape(*x.shape[:-1], CANDIDATE_COUNT, CANDIDATE_WIDTH)
            embedding = torch.tanh(self.candidate_encoder((candidates - self.candidate_mean) / self.candidate_scale))
            masked = torch.where(mask.unsqueeze(-1), embedding, torch.full_like(embedding, -torch.inf))
            context = masked.max(dim=-2).values
            context = torch.where(mask.any(dim=-1, keepdim=True), context, torch.zeros_like(context))
            context = context.unsqueeze(-2).expand(*embedding.shape)
            global_values = ((x[..., 48:] - self.global_mean) / self.global_scale).unsqueeze(-2)
            global_values = global_values.expand(*x.shape[:-1], CANDIDATE_COUNT, GLOBAL_COUNT)
            hidden = torch.tanh(self.shared_hidden(torch.cat((embedding, context, global_values), dim=-1)))
            logits = self.shared_head(hidden).squeeze(-1)
        return torch.where(mask, logits, torch.full_like(logits, MASKED_LOGIT))


class SlotSelector(PitchSelector):
    def __init__(self, loss: str = "nearest"):
        super().__init__(SelectorConfig(representation="fixed-slots", loss=loss))


class SetSelector(PitchSelector):
    def __init__(self, loss: str = "nearest"):
        super().__init__(SelectorConfig(representation="shared-candidates", loss=loss))


def selector_loss(logits: Tensor, candidate_mask: Tensor, targets: PitchTargets, *, objective: str = "nearest") -> Tensor:
    """Mean over covered valid voiced frames only; finite connected zero if none.

    Set loss is -log(sum_{acceptable} p(candidate)); it does not force one of
    several acceptable pitches to suppress the others. This is an uncalibrated
    conditional selector objective, not a voiced or measurement-validity loss.
    """
    if objective not in OBJECTIVES:
        raise ValueError("Unknown selector objective")
    if logits.shape != candidate_mask.shape or logits.shape != targets.acceptable.shape or logits.shape[:-1] != targets.covered.shape or targets.nearest.shape != targets.covered.shape:
        raise ValueError("Logits, masks and target shapes disagree")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite selector logits")
    if torch.any(targets.acceptable & ~candidate_mask) or not torch.equal(targets.covered, targets.acceptable.any(dim=-1)):
        raise ValueError("Acceptable targets must be available and match covered rows")
    if torch.any((targets.nearest >= 0) != targets.covered):
        raise ValueError("No false nearest target is allowed on uncovered rows")
    if not bool(targets.covered.any()):
        return logits.sum() * 0.0
    selected_logits = logits[targets.covered]
    selected_mask = candidate_mask[targets.covered]
    acceptable = targets.acceptable[targets.covered]
    nearest = targets.nearest[targets.covered]
    if torch.any(nearest >= logits.shape[-1]) or not torch.all(acceptable.gather(-1, nearest.unsqueeze(-1))):
        raise ValueError("Nearest target must name an acceptable candidate")
    safe = torch.where(selected_mask, selected_logits, torch.full_like(selected_logits, -torch.inf))
    if objective == "nearest":
        return F.cross_entropy(safe, nearest)
    numerator = torch.logsumexp(torch.where(acceptable, safe, torch.full_like(safe, -torch.inf)), dim=-1)
    return (torch.logsumexp(safe, dim=-1) - numerator).mean()


def selector_probabilities(logits: Tensor, candidate_mask: Tensor) -> Tensor:
    """Exactly zero mass on masked and all-unavailable rows."""
    safe = torch.where(candidate_mask, logits, torch.full_like(logits, -torch.inf))
    any_valid = candidate_mask.any(dim=-1, keepdim=True)
    maximum = torch.where(any_valid, safe.max(dim=-1, keepdim=True).values, torch.zeros_like(safe[..., :1]))
    weights = torch.exp(safe - maximum)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def predict_pitch_hz(logits: Tensor, batch: SelectorBatch) -> Tensor:
    """CPU float64 original-candidate Hz; NaN unavailable, never unvoiced.

    Probability ties select lowest actual output Hz, independent of permutation.
    The external frozen voicing policy must still decide voiced/unvoiced status.
    """
    logits = logits.detach().cpu()
    mask = batch.candidate_mask.detach().cpu()
    hz = batch.candidate_hz.detach().cpu()
    if logits.shape != mask.shape:
        raise ValueError("Prediction shape differs from selector batch")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite selector logits")
    safe = torch.where(mask, logits, torch.full_like(logits, -torch.inf))
    tied = mask & (safe == safe.max(dim=-1, keepdim=True).values)
    result = torch.where(tied, hz, torch.full_like(hz, torch.inf)).min(dim=-1).values
    return torch.where(mask.any(dim=-1), result, torch.full_like(result, torch.nan))


def selector_metadata(model: PitchSelector) -> dict:
    return {"schema": SCHEMA, "config": asdict(model.config), "feature_schema": "phenometrix.voice-pitch-features.v2",
        "parameter_count": sum(p.numel() for p in model.parameters()), "scaling_fitted": bool(model.scaling_fitted.item()),
        "framewise": True, "voicing": "external unchanged frozen original MLP; no selector voicing parameters",
        "grouping_applied": model.config.representation == "shared-candidates",
        "group_tolerance_cents": GROUP_TOLERANCE_CENTS, "group_linkage": "sorted greedy total-span complete-link",
        "group_representative": "highest maximum source evidence, lowest Hz tie, original candidate only",
        "group_source_evidence": "per-source maxima; original members/features retained separately",
        "group_global_count": "groupcount/6 replaces raw valid_candidate_fraction",
        "group_coverage": "both targets use actual emitted representative, not any-member oracle",
        "scaling": "train-only per-column for slots; valid-group pooled candidate fields plus globals for shared",
        "context_pooling": "masked elementwise maximum; empty context zero",
        "near_duplicate_invariance": False, "exact_duplicate_invariance_shared": True,
        "probability_semantics": "uncalibrated conditional candidate probabilities; not measurement confidence"}
