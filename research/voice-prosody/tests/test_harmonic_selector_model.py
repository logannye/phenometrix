from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from phenometrix_prosody.harmonic_selector_model import HarmonicSelector, harmonic_selector_metadata
from phenometrix_prosody.selector_model import (
    prepare_selector_batch, build_pitch_targets, selector_loss, selector_probabilities, predict_pitch_hz,
)


def fixture(rows):
    x = np.zeros((len(rows), 57), np.float32)
    hz = np.full((len(rows), 6), np.nan)
    mask = np.zeros((len(rows), 6), bool)
    for row, frequencies in enumerate(rows):
        for i, f in enumerate(frequencies):
            score = [.2 + .05 * i, .1, 0., .15, .1]
            x[row, i * 8:(i + 1) * 8] = [1, np.log2(f / 50), *score, max(score)]
            hz[row, i], mask[row, i] = f, True
        x[row, 48:] = [-3 + row * .01, .1, 0, 0, .2, .7, .8, len(frequencies) / 6, 1]
    return prepare_selector_batch(x, hz, mask, representation="shared-candidates")


@pytest.mark.parametrize("objective", ["nearest", "acceptable-set"])
def test_modes_have_identical_capacity_and_same_seed_initial_tensors(objective):
    torch.manual_seed(20261022); absolute = HarmonicSelector("absolute", objective)
    torch.manual_seed(20261022); relative = HarmonicSelector("relative", objective)
    assert sum(p.numel() for p in absolute.parameters()) == 3009
    assert absolute.state_dict().keys() == relative.state_dict().keys()
    for name, value in absolute.state_dict().items():
        torch.testing.assert_close(value, relative.state_dict()[name], atol=0, rtol=0)
    assert absolute.config.representation == "shared-candidates" and absolute.config.loss == objective
    assert not any("voice" in n or "state" in n for n, _ in absolute.named_parameters())
    metadata = harmonic_selector_metadata(relative)
    assert metadata["parameter_count"] == 3009 and metadata["context_mode"] == "relative"
    assert metadata["acquisition_information_added"] is False


def test_objective_does_not_change_initialization_and_unknown_modes_fail():
    torch.manual_seed(3); a = HarmonicSelector("relative", "nearest")
    torch.manual_seed(3); b = HarmonicSelector("relative", "acceptable-set")
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="context mode"):
        HarmonicSelector("attention")
    with pytest.raises(ValueError, match="objective"):
        HarmonicSelector("relative", "huber")


def test_only_ninth_message_input_changes_between_modes():
    batch = fixture([[100, 200, 300], [150]])
    absolute, relative = HarmonicSelector("absolute"), HarmonicSelector("relative")
    absolute.fit_scaling(batch, split="train"); relative.fit_scaling(batch, split="train")
    a, r = absolute.message_inputs(batch), relative.message_inputs(batch)
    torch.testing.assert_close(a[..., :8], r[..., :8], atol=0, rtol=0)
    assert torch.equal(a[..., 8], torch.zeros_like(a[..., 8]))
    assert r[0, 0, 1, 8] == pytest.approx(1 / math.log2(14))
    assert r[0, 1, 0, 8] == pytest.approx(-1 / math.log2(14))
    assert torch.equal(r[..., 8].diagonal(dim1=-2, dim2=-1), torch.zeros((2, 6)))
    assert torch.equal(a[0, 0, :3, :8], a[0, 1, :3, :8])


def test_relation_context_exposes_query_relationships_with_known_weights():
    batch = fixture([[100, 200, 400]])
    model = HarmonicSelector("relative")
    with torch.no_grad():
        model.message_encoder.weight.zero_(); model.message_encoder.bias.zero_()
        model.message_encoder.weight[0, 8] = 1
        model.message_encoder.weight[1, 8] = -1
    context = model.relation_context(batch)
    scale = math.log2(14)
    expected = torch.tensor([[math.tanh(2 / scale), 0.],
                             [math.tanh(1 / scale), math.tanh(1 / scale)],
                             [0., math.tanh(2 / scale)]])
    torch.testing.assert_close(context[0, :3, :2], expected, atol=1e-7, rtol=1e-6)
    control = HarmonicSelector("absolute"); control.load_state_dict(model.state_dict())
    assert torch.equal(control.relation_context(batch), torch.zeros((1, 6, 32)))


def permute_batch(batch, perm):
    x = batch.features.clone()
    x[..., :48] = x[..., :48].reshape(*x.shape[:-1], 6, 8)[..., perm, :].reshape(*x.shape[:-1], 48)
    return replace(batch, features=x, candidate_mask=batch.candidate_mask[..., perm],
                   candidate_hz=batch.candidate_hz[..., perm], member_mask=batch.member_mask[..., perm, :])


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_permutation_equivariance_and_frequency_decisions(mode):
    torch.manual_seed(29)
    batch = fixture([[100, 150, 299, 300], [110, 220], []])
    model = HarmonicSelector(mode); model.fit_scaling(batch, split="train")
    for perm in [torch.tensor([5, 2, 0, 4, 1, 3]), torch.tensor([3, 2, 1, 0, 5, 4])]:
        other = permute_batch(batch, perm)
        torch.testing.assert_close(model(other), model(batch)[..., perm], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(model.relation_context(other), model.relation_context(batch)[..., perm, :], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(predict_pitch_hz(model(other), other), predict_pitch_hz(model(batch), batch), equal_nan=True)


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_exact_duplicate_invariance_keeps_preprocessing_scaling_and_loss(mode):
    base = fixture([[100, 200, 300], [105, 210, 315]])
    x, hz, mask = base.original_features.clone(), base.original_candidate_hz.clone(), base.original_candidate_mask.clone()
    x[:, 24:32] = x[:, 8:16]; hz[:, 3] = hz[:, 1]; mask[:, 3] = True; x[:, 55] = 4 / 6
    copied = prepare_selector_batch(x, hz, mask, representation="shared-candidates")
    torch.testing.assert_close(base.features, copied.features, rtol=0, atol=0)
    torch.manual_seed(31); a = HarmonicSelector(mode); a.fit_scaling(base, split="train")
    torch.manual_seed(31); b = HarmonicSelector(mode); b.fit_scaling(copied, split="train")
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(a(base), b(copied), rtol=0, atol=0)
    ta = build_pitch_targets(base, [200., 210.], [True, True], [True, True])
    tb = build_pitch_targets(copied, [200., 210.], [True, True], [True, True])
    for objective in ["nearest", "acceptable-set"]:
        torch.testing.assert_close(selector_loss(a(base), base.candidate_mask, ta, objective=objective),
                                   selector_loss(b(copied), copied.candidate_mask, tb, objective=objective), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_empty_singleton_padding_and_self_edges(mode):
    batch = fixture([[], [200], [100, 200]])
    model = HarmonicSelector(mode)
    inputs = model.message_inputs(batch)
    assert torch.equal(inputs[0], torch.zeros_like(inputs[0]))
    assert torch.equal(inputs[1, 0, 0, :8], batch.features[1, :8])
    assert inputs[1, 0, 0, 8] == 0  # self is valid, with ratio zero
    context = model.relation_context(batch)
    torch.testing.assert_close(context[1, 0], torch.tanh(model.message_encoder(inputs[1, 0, 0])))
    assert torch.equal(context[0], torch.zeros_like(context[0]))
    assert torch.equal(context[1, 1:], torch.zeros_like(context[1, 1:]))
    logits = model(batch)
    assert torch.isfinite(logits).all()
    assert selector_probabilities(logits, batch.candidate_mask)[0].sum() == 0
    assert torch.isnan(predict_pitch_hz(logits, batch)[0])
    # Invalid-slot garbage is ignored at the message/pool boundary even if a
    # caller tampers with a prepared batch; valid preparation rejects it first.
    changed = batch.features.clone()
    blocks = changed[:, :48].reshape(3, 6, 8)
    blocks[~batch.candidate_mask] = 10000.
    altered = replace(batch, features=changed)
    torch.testing.assert_close(model(altered), logits, rtol=0, atol=0)
    torch.testing.assert_close(model.relation_context(altered), context, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["absolute", "relative"])
@pytest.mark.parametrize("objective", ["nearest", "acceptable-set"])
def test_gradients_finite_and_ratio_weight_behavior(mode, objective):
    batch = fixture([[100, 199, 201, 300], [110, 220, 330], []])
    model = HarmonicSelector(mode, objective); model.fit_scaling(batch, split="train")
    target = build_pitch_targets(batch, [200., 110., np.nan], [True, True, False], [True, True, False])
    loss = selector_loss(model(batch), batch.candidate_mask, target, objective=objective)
    assert torch.isfinite(loss); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if mode == "absolute":
        assert torch.equal(model.message_encoder.weight.grad[:, 8], torch.zeros(32))
    else:
        assert model.message_encoder.weight.grad[:, 8].abs().sum() > 0
    model.zero_grad()
    none = build_pitch_targets(batch, [np.nan] * 3, [False] * 3, [False] * 3)
    zero = selector_loss(model(batch), batch.candidate_mask, none, objective=objective)
    assert zero.item() == 0; zero.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() == 0 for p in model.parameters())


def test_relative_ratio_common_frequency_multiplier_invariance_only():
    a = fixture([[80, 160, 240]])
    b = fixture([[160, 320, 480]])
    model = HarmonicSelector("relative")
    torch.testing.assert_close(model.relative_log_ratios(a), model.relative_log_ratios(b), atol=1e-7, rtol=1e-6)
    assert not torch.equal(model.message_inputs(a)[..., :8], model.message_inputs(b)[..., :8])


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_framewise_no_future_or_batch_order_dependency(mode):
    batch = fixture([[100, 200, 300], [130, 260], [140, 280], [160, 320]])
    model = HarmonicSelector(mode); model.fit_scaling(batch, split="train")
    expected = model(batch)
    order = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(model(batch.index_select(order)), expected[order], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.cat([model(batch.index_select(torch.tensor([i]))) for i in range(4)]), expected, atol=1e-6, rtol=1e-6)
    changed = batch.features.clone(); changed[2:, 48:] += 100
    actual = model(replace(batch, features=changed))
    torch.testing.assert_close(actual[:2], expected[:2], rtol=0, atol=0)
    timed = prepare_selector_batch(batch.original_features.reshape(1, 4, 57),
        batch.original_candidate_hz.reshape(1, 4, 6), batch.original_candidate_mask.reshape(1, 4, 6),
        representation="shared-candidates")
    torch.testing.assert_close(model(timed).reshape(4, 6), expected, atol=1e-6, rtol=1e-6)


def test_scaling_train_only_and_predictions_use_existing_frequencies():
    batch = fixture([[100, 200, 300], [110, 220], []])
    model = HarmonicSelector("relative")
    for split in ["development", "sealed_test", "test"]:
        with pytest.raises(ValueError, match="training split"):
            model.fit_scaling(batch, split=split)
    model.fit_scaling(batch, split="train")
    result = predict_pitch_hz(model(batch), batch)
    assert result[0].item() in [100, 200, 300] and result[1].item() in [110, 220]
    assert torch.isnan(result[2])
