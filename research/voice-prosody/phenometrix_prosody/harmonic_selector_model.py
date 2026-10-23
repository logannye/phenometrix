"""Matched-capacity absolute/relative context for the pitch-only selector.

This module leaves the frozen step-1 model, grouping, scaling, labels and loss
unchanged. Both arms contain exactly the same parameter tensors. The only arm
switch is whether the ninth ordered-pair message input is zero or a normalized
log frequency ratio; it adds no waveform, candidate, reference or future input.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .selector_model import (
    CANDIDATE_COUNT, CANDIDATE_WIDTH, GLOBAL_COUNT, MASKED_LOGIT,
    PitchSelector, SelectorBatch, SelectorConfig, selector_metadata,
)

CONTEXT_MODES = ("absolute", "relative")
LOG2_FREQUENCY_RANGE = math.log2(700.0 / 50.0)
SCHEMA = "phenometrix.harmonic-selector.v1"


class HarmonicSelector(PitchSelector):
    """Framewise shared candidate scorer with query-relative neighbor messages.

    Inputs and output are exactly the step-1 SelectorBatch and [...,6] logits.
    The inherited config always has representation='shared-candidates'; objective
    selects the unchanged nearest or acceptable-set loss outside this forward.
    Original voicing stays external. Initialization must follow the caller's
    declared seed. context_mode and objective do not alter parameter allocation
    or random-number consumption, so same-seed arms start with identical tensors.

    Per candidate e_i=tanh Linear(8,32)(scaled x_i). Each ordered valid pair,
    INCLUDING i=j, receives tanh Linear(9,32)([scaled x_j, r_ij]); context_i is
    its elementwise maximum over valid j. r_ij is zero in the absolute arm and
    (raw log2Hz_j - raw log2Hz_i)/log2(14) in the relative arm. This ratio is a
    deterministic function of existing features, without fitted scaling/clamping.
    The unchanged shared head maps [e_i(32),context_i(32),globals(9)] through
    tanh Linear(73,32), then Linear(32,1). Total trainable parameters: 3,009.
    """

    def __init__(self, context_mode: str, objective: str = "acceptable-set"):
        if context_mode not in CONTEXT_MODES:
            raise ValueError("Unknown harmonic context mode")
        super().__init__(SelectorConfig(representation="shared-candidates", loss=objective))
        self.context_mode = context_mode
        self.message_encoder = nn.Linear(CANDIDATE_WIDTH + 1, 32)

    def _standardized_candidates(self, batch: SelectorBatch) -> Tensor:
        self._check_batch(batch)
        values = batch.features[..., :48].reshape(*batch.features.shape[:-1], CANDIDATE_COUNT, CANDIDATE_WIDTH)
        return (values - self.candidate_mean) / self.candidate_scale

    def relative_log_ratios(self, batch: SelectorBatch) -> Tensor:
        """Existing raw-frequency feature differences, masked on BOTH axes.

        Returns [..., query_i, neighbor_j]. This helper returns true ratios for
        either arm; message_inputs applies the predeclared absolute-arm zero.
        Ratio invariance to a common frequency multiplier does not imply that
        the entire predictor is frequency-shift invariant: x_j keeps its Hz.
        """
        self._check_batch(batch)
        raw = batch.features[..., :48].reshape(*batch.features.shape[:-1], CANDIDATE_COUNT, CANDIDATE_WIDTH)
        log_hz = raw[..., 1]
        ratio = (log_hz.unsqueeze(-2) - log_hz.unsqueeze(-1)) / LOG2_FREQUENCY_RANGE
        edges = batch.candidate_mask.unsqueeze(-1) & batch.candidate_mask.unsqueeze(-2)
        return torch.where(edges, ratio, torch.zeros_like(ratio))

    def message_inputs(self, batch: SelectorBatch) -> Tensor:
        """Auditable [...,query_i,neighbor_j,9] inputs before shared encoding."""
        candidates = self._standardized_candidates(batch)
        neighbors = candidates.unsqueeze(-3).expand(*candidates.shape[:-2], CANDIDATE_COUNT,
                                                   CANDIDATE_COUNT, CANDIDATE_WIDTH)
        ratio = self.relative_log_ratios(batch)
        if self.context_mode == "absolute":
            ratio = torch.zeros_like(ratio)
        values = torch.cat((neighbors, ratio.unsqueeze(-1)), dim=-1)
        edges = batch.candidate_mask.unsqueeze(-1) & batch.candidate_mask.unsqueeze(-2)
        return torch.where(edges.unsqueeze(-1), values, torch.zeros_like(values))

    def relation_context(self, batch: SelectorBatch) -> Tensor:
        """Symmetric neighbor max; invalid queries/all-empty sets return zero."""
        messages = torch.tanh(self.message_encoder(self.message_inputs(batch)))
        edges = batch.candidate_mask.unsqueeze(-1) & batch.candidate_mask.unsqueeze(-2)
        masked = torch.where(edges.unsqueeze(-1), messages, torch.full_like(messages, -torch.inf))
        context = masked.max(dim=-2).values
        return torch.where(edges.any(dim=-1, keepdim=True), context, torch.zeros_like(context))

    def forward(self, batch: SelectorBatch) -> Tensor:
        candidates = self._standardized_candidates(batch)
        embedding = torch.tanh(self.candidate_encoder(candidates))
        context = self.relation_context(batch)
        global_values = ((batch.features[..., 48:] - self.global_mean) / self.global_scale).unsqueeze(-2)
        global_values = global_values.expand(*batch.features.shape[:-1], CANDIDATE_COUNT, GLOBAL_COUNT)
        hidden = torch.tanh(self.shared_hidden(torch.cat((embedding, context, global_values), dim=-1)))
        logits = self.shared_head(hidden).squeeze(-1)
        return torch.where(batch.candidate_mask, logits, torch.full_like(logits, MASKED_LOGIT))


def harmonic_selector_metadata(model: HarmonicSelector) -> dict:
    metadata = selector_metadata(model)
    metadata.update(schema=SCHEMA, context_mode=model.context_mode,
        context_pooling="masked elementwise maximum of independent shared neighbor messages; self included; empty zero",
        message_input="8 shared-standardized neighbor candidate fields plus one query-relative ratio or zero",
        message_encoder={"input_width": 9, "output_width": 32, "activation": "tanh"},
        ratio="(unscaled log2Hz_neighbor - unscaled log2Hz_query)/log2(700/50); no fitted scale or clamp",
        absolute_control="ninth message input is identically zero; all other inputs/parameters/shapes identical",
        parameter_initialization="PyTorch defaults after caller seed; same random draws for both modes and objectives",
        acquisition_information_added=False, temporal_state=False,
        original_voicing_unchanged=True,
        frequency_shift_invariance="relative ratio only; absolute candidate evidence remains in both arms")
    return metadata
