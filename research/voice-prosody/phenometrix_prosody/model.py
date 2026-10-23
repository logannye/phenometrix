"""Small acquisition-feature models. No weights here have been trained or validated."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import Tensor, nn

FEATURE_SCHEMA = "phenometrix.voice-pitch-features.v2"
FEATURE_NAMES_V1 = tuple(
    name
    for i in range(3)
    for name in (
        f"candidate_{i}_valid", f"candidate_{i}_log2_hz_over_50",
        f"candidate_{i}_autocorrelation_score", f"candidate_{i}_amdf_score",
        f"candidate_{i}_zero_crossing_score", f"candidate_{i}_maximum_score",
    )
) + (
    "log_rms_floor_1e-8", "delta_log_rms", "dc_offset", "clipped_sample_fraction",
    "spectral_flux", "legacy_confidence_score", "legacy_estimator_agreement",
    "valid_candidate_fraction", "history_present",
)
FEATURE_NAMES_V2 = tuple(
    name for i in range(6) for name in (
        f"candidate_{i}_valid", f"candidate_{i}_log2_hz_over_50",
        f"candidate_{i}_autocorrelation_score", f"candidate_{i}_amdf_score",
        f"candidate_{i}_zero_crossing_score", f"candidate_{i}_acf40_refined_score",
        f"candidate_{i}_acf20_refined_score", f"candidate_{i}_maximum_score",
    )
) + FEATURE_NAMES_V1[-9:]
FEATURE_NAMES = FEATURE_NAMES_V2
MASKED_LOGIT = -1e9


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "causal-gru"
    feature_version: str = "v2"
    feature_count: int | None = None
    projection_size: int = 32
    hidden_size: int = 32
    candidate_count: int | None = None

    def __post_init__(self):
        if self.kind not in {"causal-gru", "framewise-mlp"}:
            raise ValueError("Unsupported model family member")
        if self.feature_version not in {"v1", "v2"}:
            raise ValueError("Unknown feature version")
        expected_features, expected_candidates = (27, 3) if self.feature_version == "v1" else (57, 6)
        if self.feature_count is None:
            object.__setattr__(self, "feature_count", expected_features)
        if self.candidate_count is None:
            object.__setattr__(self, "candidate_count", expected_candidates)
        if self.feature_count != expected_features or self.candidate_count != expected_candidates:
            raise ValueError("Feature and candidate dimensions must match the versioned contract")
        if self.projection_size <= 0 or self.hidden_size <= 0:
            raise ValueError("Model dimensions must be positive")


class VoiceCandidateModel(nn.Module):
    """One step with explicit state; decomposed GRU ops export without ONNX GRU.

    Inputs: features [B,F], candidate_mask [B,K] bool, state [B,32].
    Outputs: voicing_logit [B,1], masked candidate_logits [B,K], state_out [B,32].
    v1 has F=27/K=3; v2 has F=57/K=6.
    A voiced all-masked frame remains unavailable, not unvoiced. Callers must
    inspect candidate_mask; finite -1e9 logits are not evidence for any candidate.
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.register_buffer("feature_mean", torch.zeros(config.feature_count))
        self.register_buffer("feature_scale", torch.ones(config.feature_count))
        self.register_buffer("scaling_fitted", torch.tensor(False))
        self.projection = nn.Linear(config.feature_count, config.projection_size)
        if config.kind == "causal-gru":
            self.input_gates = nn.Linear(config.projection_size, 3 * config.hidden_size)
            self.state_gates = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        else:
            self.frame_hidden = nn.Linear(config.projection_size, config.hidden_size)
        self.voicing_head = nn.Linear(config.hidden_size, 1)
        self.candidate_head = nn.Linear(config.hidden_size, config.candidate_count)

    def initial_state(self, batch_size: int = 1, *, device=None) -> Tensor:
        return torch.zeros(batch_size, self.config.hidden_size, dtype=torch.float32, device=device)

    def fit_scaling(self, features: Tensor, *, split: str) -> None:
        """Only training acquisition features may set preprocessing buffers."""
        if split != "train":
            raise ValueError("Scaling may only be fitted on the training split")
        if features.ndim != 2 or features.shape[1] != self.config.feature_count or features.shape[0] == 0:
            raise ValueError("Expected nonempty [frames, feature_count] training features")
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite training features")
        with torch.no_grad():
            self.feature_mean.copy_(features.mean(dim=0))
            self.feature_scale.copy_(features.std(dim=0, unbiased=False).clamp_min(1e-6))
            self.scaling_fitted.fill_(True)

    def forward(self, features: Tensor, candidate_mask: Tensor, state: Tensor):
        scaled = (features - self.feature_mean) / self.feature_scale
        projected = torch.tanh(self.projection(scaled))
        if self.config.kind == "causal-gru":
            ir, iz, inn = self.input_gates(projected).chunk(3, dim=-1)
            hr, hz, hn = self.state_gates(state).chunk(3, dim=-1)
            reset = torch.sigmoid(ir + hr)
            update = torch.sigmoid(iz + hz)
            candidate = torch.tanh(inn + reset * hn)
            hidden = (1 - update) * candidate + update * state
            state_out = hidden
        else:
            hidden = torch.tanh(self.frame_hidden(projected))
            # Keep an explicit, identically shaped state interface for the ablation.
            state_out = state * 0.0
        voicing_logit = self.voicing_head(hidden)
        raw_logits = self.candidate_head(hidden)
        candidate_logits = torch.where(candidate_mask, raw_logits, torch.full_like(raw_logits, MASKED_LOGIT))
        return voicing_logit, candidate_logits, state_out


def candidate_probabilities(logits: Tensor, candidate_mask: Tensor) -> Tensor:
    """Zero distribution for all-masked frames, never a fictitious uniform F0."""
    safe = torch.where(candidate_mask, logits, torch.full_like(logits, -torch.inf))
    maximum = safe.max(dim=-1, keepdim=True).values
    maximum = torch.where(candidate_mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
    weights = torch.exp(safe - maximum)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def select_candidate_indices(logits: Tensor, candidate_mask: Tensor) -> Tensor:
    """Lowest sorted-Hz index wins ties; -1 means unavailable, never unvoiced."""
    safe = torch.where(candidate_mask, logits, torch.full_like(logits, -torch.inf))
    chosen = safe.argmax(dim=-1)
    return torch.where(candidate_mask.any(dim=-1), chosen, torch.full_like(chosen, -1))


def numpy_step(config: ModelConfig, weights: dict[str, np.ndarray], features: np.ndarray,
               candidate_mask: np.ndarray, state: np.ndarray):
    """Independent NumPy equations with no call into Torch or ONNX inference."""
    def linear(name, values):
        return values @ weights[f"{name}.weight"].T + weights[f"{name}.bias"]

    def sigmoid(values):
        # Stable for finite large magnitudes without overflow warnings.
        return np.exp(-np.logaddexp(np.float32(0), -values))

    scaled = (features - weights["feature_mean"]) / weights["feature_scale"]
    projected = np.tanh(linear("projection", scaled))
    if config.kind == "causal-gru":
        ir, iz, inn = np.split(linear("input_gates", projected), 3, axis=-1)
        hr, hz, hn = np.split(linear("state_gates", state), 3, axis=-1)
        reset = sigmoid(ir + hr)
        update = sigmoid(iz + hz)
        candidate = np.tanh(inn + reset * hn)
        hidden = (1 - update) * candidate + update * state
        next_state = hidden
    else:
        hidden = np.tanh(linear("frame_hidden", projected))
        next_state = state * np.float32(0)
    voicing = linear("voicing_head", hidden)
    logits = np.where(candidate_mask, linear("candidate_head", hidden), np.float32(MASKED_LOGIT))
    return voicing.astype(np.float32), logits.astype(np.float32), next_state.astype(np.float32)


def model_metadata(model: VoiceCandidateModel) -> dict:
    return {
        "schema_version": "phenometrix.voice-candidate-model.v1",
        "config": asdict(model.config), "feature_schema": f"phenometrix.voice-pitch-features.{model.config.feature_version}",
        "feature_names": list(FEATURE_NAMES_V1 if model.config.feature_version == "v1" else FEATURE_NAMES_V2), "scaling_fitted": bool(model.scaling_fitted.item()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_initialization": "zeros", "state_dtype": "float32",
        "candidate_order": "strictly-increasing-hz-exact-duplicates-merge",
        "no_candidate": "unavailable-retain-voicing-evidence",
        "masked_logit": MASKED_LOGIT, "tie_breaking": "lowest-valid-candidate-index",
        "probability_semantics": "voicing-and-candidate-logits-uncalibrated",
    }
