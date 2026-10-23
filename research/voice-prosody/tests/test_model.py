import numpy as np
import pytest
import torch

from phenometrix_prosody.model import (
    FEATURE_NAMES, ModelConfig, VoiceCandidateModel, candidate_probabilities,
    numpy_step, select_candidate_indices,
)


@pytest.mark.parametrize("kind", ["causal-gru", "framewise-mlp"])
def test_independent_numpy_sequence_parity(kind):
    torch.manual_seed(812)
    model = VoiceCandidateModel(ModelConfig(kind=kind)).eval()
    weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    generator = np.random.default_rng(812)
    state = model.initial_state()
    numpy_state = state.numpy().copy()
    with torch.no_grad():
        for _ in range(31):
            features = generator.normal(size=(1, len(FEATURE_NAMES))).astype(np.float32)
            mask = generator.integers(0, 2, size=(1, model.config.candidate_count)).astype(bool)
            actual = model(torch.from_numpy(features), torch.from_numpy(mask), state)
            expected = numpy_step(model.config, weights, features, mask, numpy_state)
            for target, reference in zip(actual, expected):
                np.testing.assert_allclose(target.numpy(), reference, atol=1e-6, rtol=1e-5)
            state = actual[2]
            numpy_state = expected[2]


def test_future_suffix_reset_and_chunk_equivalence():
    torch.manual_seed(1)
    model = VoiceCandidateModel().eval()
    sequence = torch.randn(14, 1, len(FEATURE_NAMES))
    mask = torch.ones(1, model.config.candidate_count, dtype=torch.bool)

    def run(values, chunks):
        state = model.initial_state()
        outputs = []
        offset = 0
        with torch.no_grad():
            for length in chunks:
                for frame in values[offset:offset + length]:
                    voiced, logits, state = model(frame, mask, state)
                    outputs.append(torch.cat([voiced, logits, state], dim=1))
                offset += length
        return outputs

    expected = run(sequence, [14])
    chunked = run(sequence, [2, 5, 1, 6])
    for left, right in zip(expected, chunked):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    changed = sequence.clone()
    changed[8:] *= -100
    altered = run(changed, [14])
    for left, right in zip(expected[:8], altered[:8]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(run(sequence, [14])[0], expected[0], rtol=0, atol=0)


def test_all_masked_is_unavailable_with_finite_logits_and_zero_distribution():
    model = VoiceCandidateModel()
    mask = torch.zeros(1, model.config.candidate_count, dtype=torch.bool)
    voiced, logits, state = model(torch.zeros(1, len(FEATURE_NAMES)), mask, model.initial_state())
    assert torch.isfinite(voiced).all() and torch.isfinite(logits).all() and torch.isfinite(state).all()
    assert candidate_probabilities(logits, mask).sum().item() == 0
    assert select_candidate_indices(logits, mask).item() == -1
    extreme = torch.tensor([[-1e10, -1e11, 0.0]])
    extreme_mask = torch.tensor([[True, True, False]])
    assert select_candidate_indices(extreme, extreme_mask).item() == 0
    assert candidate_probabilities(extreme, extreme_mask)[0, 0].item() == 1
    tie = torch.zeros(1, 3)
    assert select_candidate_indices(tie, torch.tensor([[False, True, True]])).item() == 1


def test_framewise_ablation_has_no_history_dependency():
    model = VoiceCandidateModel(ModelConfig(kind="framewise-mlp"))
    features = torch.randn(1, len(FEATURE_NAMES))
    mask = torch.ones(1, model.config.candidate_count, dtype=torch.bool)
    first = model(features, mask, model.initial_state())
    second = model(features, mask, torch.randn(1, 32))
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_scaling_cannot_fit_development_or_test_data():
    model = VoiceCandidateModel()
    features = torch.randn(10, len(FEATURE_NAMES))
    for split in ["development", "dev", "sealed-test", "test"]:
        with pytest.raises(ValueError, match="training split"):
            model.fit_scaling(features, split=split)
    model.fit_scaling(features, split="train")
    assert model.scaling_fitted.item()
    torch.testing.assert_close(model.feature_mean, features.mean(dim=0))
    assert torch.all(model.feature_scale > 0)


def test_v1_model_contract_remains_reproducible():
    model = VoiceCandidateModel(ModelConfig(feature_version="v1"))
    assert model.config.feature_count == 27 and model.config.candidate_count == 3
    result = model(torch.zeros(1, 27), torch.ones(1, 3, dtype=torch.bool), model.initial_state())
    assert result[1].shape == (1, 3)
