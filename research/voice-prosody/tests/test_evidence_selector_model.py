"""Fixture-only residual, exact-control, mask, invariance and gradient checks."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from phenometrix_prosody.evidence_selector_model import EvidenceHarmonicSelector, evidence_selector_metadata
from phenometrix_prosody.expanded_selector_model import ExpandedHarmonicSelector, prepare_expanded_selector_batch
from phenometrix_prosody.selector_model import build_pitch_targets, selector_loss


def batch(k=7):
    x = np.zeros((3, 8 * k + 9), np.float32)
    hz = np.full((3, k), np.nan)
    mask = np.zeros((3, k), bool)
    for row, frequencies in enumerate(([100., 200., 300.], [125., 250.], [])):
        for slot, f in enumerate(frequencies):
            scores = [.3, .2, 0., .5, .4]
            x[row, slot * 8:(slot + 1) * 8] = [1, np.log2(f / 50), *scores, .5]
            hz[row, slot], mask[row, slot] = f, True
        x[row, 8 * k:] = [-3., .1, 0., 0., .2, .7, .8, len(frequencies) / 6, 1.]
    return prepare_expanded_selector_batch(x, hz, mask)


def evidence(b):
    out = torch.zeros((*b.candidate_mask.shape, 3), dtype=torch.float32)
    out[..., 0] = torch.where(b.candidate_mask, .8, 0.)
    out[..., 1] = torch.where(b.candidate_mask, .2, 0.)
    out[..., 2] = b.candidate_mask.to(torch.float32)
    return out


@pytest.mark.parametrize('mode', ['off', 'periodicity'])
@pytest.mark.parametrize('k', [6, 7])
def test_initialization_baseline_logits_rng_and_scaling_are_exact(mode, k):
    torch.manual_seed(20261022); old = ExpandedHarmonicSelector(); expected_rng = torch.get_rng_state()
    torch.manual_seed(20261022); new = EvidenceHarmonicSelector(mode); actual_rng = torch.get_rng_state()
    torch.testing.assert_close(expected_rng, actual_rng, rtol=0, atol=0)
    assert sum(p.numel() for p in new.parameters()) == 3012
    assert torch.count_nonzero(new.evidence_head.weight) == 0
    old.fit_scaling(batch(6), split='train'); new.fit_scaling(batch(6), split='train')
    for name, value in old.state_dict().items(): torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    b = batch(k)
    torch.testing.assert_close(old(b), new(b, evidence(b)), rtol=0, atol=0)
    metadata = evidence_selector_metadata(new)
    assert metadata['parameter_count'] == 3012 and metadata['original_voicing_unchanged']
    assert not metadata['temporal_state']


def test_off_arm_optimizer_path_matches_old_control_and_cannot_use_supplied_evidence():
    torch.manual_seed(31); old = ExpandedHarmonicSelector()
    torch.manual_seed(31); new = EvidenceHarmonicSelector('off')
    old.fit_scaling(batch(6), split='train'); new.fit_scaling(batch(6), split='train')
    b, e = batch(), evidence(batch())
    target = build_pitch_targets(b, [200., 125., np.nan], [True, True, False], [True] * 3)
    optimizers = [torch.optim.Adam(m.parameters(), lr=.001) for m in (old, new)]
    for _ in range(3):
        for model, optimizer in zip((old, new), optimizers):
            optimizer.zero_grad()
            logits = model(b) if model is old else model(b, e)
            selector_loss(logits, b.candidate_mask, target, objective='acceptable-set').backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        for name, value in old.state_dict().items(): torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    assert torch.count_nonzero(new.evidence_head.weight) == 0
    with torch.no_grad(): new.evidence_head.weight.fill_(7.)
    torch.testing.assert_close(old(b), new(b, e), rtol=0, atol=0)


def test_residual_is_exact_shared_linear_formula_with_padding_masked():
    b = batch(); e = evidence(b)
    model = EvidenceHarmonicSelector('periodicity'); model.fit_scaling(batch(6), split='train')
    baseline = model(b, e).detach()
    with torch.no_grad(): model.evidence_head.weight.copy_(torch.tensor([[2., -1., .25]]))
    actual = model(b, e)
    expected = baseline + (e @ model.evidence_head.weight.T).squeeze(-1)
    expected = torch.where(b.candidate_mask, expected, torch.full_like(expected, -1e9))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.all(actual[~b.candidate_mask] == -1e9)


def test_validity_weight_cancels_only_when_all_available_candidates_have_valid_evidence():
    # Validity is an explicit input. It may distinguish low-energy candidates;
    # its shared offset cancels from softmax when all have valid evidence.
    b = batch(); e = evidence(b); e[0, 1] = 0
    model = EvidenceHarmonicSelector('periodicity')
    with torch.no_grad(): model.evidence_head.weight.copy_(torch.tensor([[0., 0., 1.]]))
    baseline = ExpandedHarmonicSelector(); baseline.load_state_dict({k: v for k, v in model.state_dict().items() if k != 'evidence_head.weight'})
    difference = model(b, e) - baseline(b)
    torch.testing.assert_close(difference[b.candidate_mask], e[..., 2][b.candidate_mask], rtol=0, atol=1e-7)


def test_permutation_and_shuffled_batch_preserve_outputs():
    b = batch(); e = evidence(b)
    model = EvidenceHarmonicSelector('periodicity')
    with torch.no_grad(): model.evidence_head.weight.copy_(torch.tensor([[.3, -.2, .1]]))
    p = torch.tensor([6, 2, 0, 5, 1, 4, 3])
    x = b.features.clone(); x[:, :56] = x[:, :56].reshape(3, 7, 8)[:, p].reshape(3, 56)
    permuted = replace(b, features=x, candidate_hz=b.candidate_hz[:, p], candidate_mask=b.candidate_mask[:, p],
                       member_mask=b.member_mask[:, p])
    torch.testing.assert_close(model(permuted, e[:, p]), model(b, e)[:, p], rtol=1e-6, atol=1e-6)
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(model(b.index_select(order), e[order]), model(b, e)[order], rtol=0, atol=0)


def test_all_empty_and_mixed_validity_produce_finite_connected_gradients():
    b, e = batch(), evidence(batch()); e[0, 1] = 0
    model = EvidenceHarmonicSelector('periodicity')
    target = build_pitch_targets(b, [200., 125., np.nan], [True, True, False], [True] * 3)
    logits = model(b, e)
    loss = selector_loss(logits, b.candidate_mask, target, objective='acceptable-set')
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.count_nonzero(model.evidence_head.weight.grad) > 0
    empty = b.index_select([2]); model.zero_grad()
    model(empty, e[[2]]).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('change', [
    lambda e: e.__setitem__((0, 0, 0), float('nan')),
    lambda e: e.__setitem__((0, 0, 0), 1.01),
    lambda e: e.__setitem__((0, 0, 2), .5),
    lambda e: e.__setitem__((0, 0, 2), 0),
    lambda e: e.__setitem__((2, 0, 2), 1),
])
def test_bad_evidence_refused_even_in_off_control(change):
    b = batch(); e = evidence(b); change(e)
    with pytest.raises(ValueError): EvidenceHarmonicSelector('off')(b, e)


def test_wrong_shape_dtype_mode_or_scaling_split_refused():
    b, e = batch(), evidence(batch())
    with pytest.raises(ValueError): EvidenceHarmonicSelector('unknown')
    with pytest.raises(ValueError): EvidenceHarmonicSelector()(b, e[..., :2])
    with pytest.raises(ValueError): EvidenceHarmonicSelector()(b, e.double())
    with pytest.raises(ValueError, match='six-candidate'): EvidenceHarmonicSelector().fit_scaling(b, split='train')
    with pytest.raises(ValueError, match='training'): EvidenceHarmonicSelector().fit_scaling(batch(6), split='development')


def test_seventh_padding_does_not_change_evidence_or_base_output():
    model = EvidenceHarmonicSelector('periodicity')
    with torch.no_grad(): model.evidence_head.weight.copy_(torch.tensor([[.3, -.2, .1]]))
    six, seven = batch(6), batch(7)
    torch.testing.assert_close(model(six, evidence(six)), model(seven, evidence(seven))[:, :6], rtol=1e-6, atol=1e-6)
