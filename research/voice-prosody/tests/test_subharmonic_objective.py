from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from phenometrix_prosody.selector_model import PitchTargets, build_pitch_targets, prepare_selector_batch, selector_loss
from phenometrix_prosody.subharmonic_objective import (
    ALIAS_FACTORS, ARMS, MARGIN, MARGIN_WEIGHT, build_negative_masks, subharmonic_loss,
)


def batch(rows):
    x = np.zeros((len(rows), 57), np.float32)
    hz = np.full((len(rows), 6), np.nan)
    mask = np.zeros_like(hz, bool)
    for row, frequencies in enumerate(rows):
        for i, frequency in enumerate(frequencies):
            hz[row, i], mask[row, i] = frequency, True
            x[row, 8*i:8*i+8] = [1, np.log2(frequency/50), .8, 0, 0, 0, 0, .8]
        x[row, 55] = len(frequencies) / 6
    return prepare_selector_batch(x, hz, mask, representation="shared-candidates")


def targets(b, hz, voiced=None, valid=None):
    return build_pitch_targets(b, hz, np.ones(len(hz), bool) if voiced is None else voiced,
                              np.ones(len(hz), bool) if valid is None else valid)


def test_frozen_constants_and_set_arm_is_exact_original_loss():
    assert ARMS == ("set", "all-margin", "alias-margin")
    assert (MARGIN, MARGIN_WEIGHT) == (1., .5)
    b = batch([[100, 150, 200], []]); t = targets(b, [200, 0], voiced=[True, False])
    z = torch.tensor([[.2, .4, .8, 0, 0, 0], [0.]*6], requires_grad=True)
    old = selector_loss(z, b.candidate_mask, t, objective="acceptable-set")
    assert torch.equal(subharmonic_loss(z, b.candidate_mask, t, None, arm="set"), old)


@pytest.mark.parametrize("factor", ALIAS_FACTORS)
def test_symmetric_alias_factors_in_both_directions(factor):
    good = 200. if factor <= 1 else 100.
    b = batch([[good, good*factor, good*1.31]])
    t = targets(b, [good]); masks = build_negative_masks(b, t)
    hz = b.candidate_hz[0]
    ix = torch.nonzero(torch.isclose(hz, torch.tensor(good*factor, dtype=torch.float64))).item()
    assert masks['alias-margin'][0, ix]
    assert masks['alias-margin'].sum() == 1
    assert masks['all-margin'].sum() == 2
    assert not torch.any(masks['alias-margin'] & t.acceptable)
    assert masks['alias-margin'].device.type == 'cpu'


@pytest.mark.parametrize("sign", [-1, 1])
def test_inclusive_float64_boundary_without_tolerance_widening(sign):
    inside = 200 * 2**(sign*(50-1e-7)/1200)
    outside = 200 * 2**(sign*(50+1e-7)/1200)
    # Separate rows prevent the existing five-cent grouping from merging them.
    b = batch([[100, inside], [100, outside]])
    t = targets(b, [100, 100]); masks = build_negative_masks(b, t)
    assert masks['alias-margin'][:, 1].tolist() == [True, False]


def test_any_acceptable_anchor_and_actual_emitted_hz_define_alias():
    # Both 200 and 205 are acceptable; 420 is within 50 cents of 2*205,
    # but farther than 50 cents from 2*200 or 2*the 200-Hz reference.
    b = batch([[200, 205, 420]])
    t = targets(b, [200]); assert t.acceptable.sum() == 2
    masks = build_negative_masks(b, t)
    assert masks['alias-margin'].tolist() == [[False, False, True, False, False, False]]
    # Near-duplicate representative is selected by evidence, not reference.
    raw = batch([[100, 200]])
    false_member_hz = raw.candidate_hz.clone(); false_member_hz[0, 1] = 260
    changed = replace(raw, candidate_hz=false_member_hz)
    assert not build_negative_masks(changed, targets(changed, [100]))['alias-margin'].any()


def test_exact_formula_averages_only_eligible_rows_including_zero_hinges():
    b = batch([[100, 200], [100, 200], [200], [100, 131]])
    t = targets(b, [200, 200, 200, 100]); masks = build_negative_masks(b, t)
    z = torch.tensor([[2., 0, 0, 0, 0, 0], [0., 3, 0, 0, 0, 0],
                      [4., 0, 0, 0, 0, 0], [0., 2, 0, 0, 0, 0]], requires_grad=True)
    base = selector_loss(z, b.candidate_mask, t, objective='acceptable-set')
    alias = subharmonic_loss(z, b.candidate_mask, t, masks['alias-margin'], arm='alias-margin')
    all_loss = subharmonic_loss(z, b.candidate_mask, t, masks['all-margin'], arm='all-margin')
    torch.testing.assert_close(alias-base, torch.tensor(.5*(3+0)/2))
    torch.testing.assert_close(all_loss-base, torch.tensor(.5*(3+0+3)/3))


def test_uncovered_unvoiced_invalid_and_all_padding_have_no_targets():
    b = batch([[100, 200], [100, 200], [100, 200], []])
    t = targets(b, [150, 200, 200, 0], voiced=[True, False, True, False], valid=[True, True, False, False])
    masks = build_negative_masks(b, t)
    assert not t.covered.any() and all(not value.any() for value in masks.values())
    for arm in ARMS:
        z = torch.randn(4, 6, requires_grad=True)
        loss = subharmonic_loss(z, b.candidate_mask, t, masks.get(arm), arm=arm)
        assert loss.item() == 0 and loss.requires_grad
        loss.backward(); assert torch.isfinite(z.grad).all() and torch.count_nonzero(z.grad) == 0


def test_no_alias_retains_base_and_padding_cannot_affect_loss_or_gradients():
    b = batch([[100, 131], []]); t = targets(b, [100, 0], voiced=[True, False])
    masks = build_negative_masks(b, t)
    z = torch.tensor([[.1, .2, 99, -99, 100, 200], [100.]*6], requires_grad=True)
    base = selector_loss(z, b.candidate_mask, t, objective='acceptable-set')
    loss = subharmonic_loss(z, b.candidate_mask, t, masks['alias-margin'], arm='alias-margin')
    assert torch.equal(loss, base)
    loss.backward(); assert torch.isfinite(z.grad).all() and not z.grad[~b.candidate_mask].any()


def test_amax_splits_tied_positive_and_negative_gradients_symmetrically():
    b = batch([[100, 103, 200, 205]])
    t = targets(b, [201]); masks = build_negative_masks(b, t)
    assert t.acceptable.sum() == 2 and masks['alias-margin'].sum() == 2
    z = torch.zeros(1, 6, requires_grad=True)
    base = selector_loss(z, b.candidate_mask, t, objective='acceptable-set')
    loss = subharmonic_loss(z, b.candidate_mask, t, masks['alias-margin'], arm='alias-margin')
    (loss-base).backward()
    torch.testing.assert_close(z.grad, torch.tensor([[.25, .25, -.25, -.25, 0, 0]]))


def test_candidate_permutation_preserves_masks_loss_and_gradient():
    b = batch([[100, 103, 200, 205, 270]])
    t = targets(b, [201]); masks = build_negative_masks(b, t)
    perm = torch.tensor([4, 2, 5, 0, 3, 1]); inverse = torch.argsort(perm)
    bp = replace(b, candidate_hz=b.candidate_hz[:, perm], candidate_mask=b.candidate_mask[:, perm])
    tp = PitchTargets(t.acceptable[:, perm], inverse[t.nearest], t.covered)
    mp = build_negative_masks(bp, tp)
    for arm in ('all-margin', 'alias-margin'):
        assert torch.equal(mp[arm], masks[arm][:, perm])
        z = torch.tensor([[.2, .3, .1, -.2, .4, 9]], requires_grad=True)
        zp = z.detach()[:, perm].requires_grad_()
        a = subharmonic_loss(z, b.candidate_mask, t, masks[arm], arm=arm)
        c = subharmonic_loss(zp, bp.candidate_mask, tp, mp[arm], arm=arm)
        torch.testing.assert_close(a, c); a.backward(); c.backward()
        torch.testing.assert_close(z.grad[:, perm], zp.grad)


def test_rejects_false_targets_invalid_masks_nonfinite_logits_and_unknown_arm():
    b = batch([[100, 200], []]); t = targets(b, [200, 0], voiced=[True, False])
    masks = build_negative_masks(b, t); z = torch.zeros(2, 6)
    bad = masks['alias-margin'].clone(); bad[0, 1] = True
    with pytest.raises(ValueError, match='nonacceptable'):
        subharmonic_loss(z, b.candidate_mask, t, bad, arm='alias-margin')
    bad = masks['alias-margin'].clone(); bad[1, 0] = True
    with pytest.raises(ValueError, match='covered'):
        subharmonic_loss(z, b.candidate_mask, t, bad, arm='alias-margin')
    with pytest.raises(ValueError, match='every'):
        subharmonic_loss(z, b.candidate_mask, t, torch.zeros_like(b.candidate_mask), arm='all-margin')
    for arm in ('bogus',):
        with pytest.raises(ValueError): subharmonic_loss(z, b.candidate_mask, t, None, arm=arm)
    with pytest.raises(ValueError, match='boolean'):
        subharmonic_loss(z, b.candidate_mask, t, masks['alias-margin'].float(), arm='alias-margin')
    with pytest.raises(ValueError, match='Nonfinite'):
        subharmonic_loss(z+torch.nan, b.candidate_mask, t, masks['alias-margin'], arm='alias-margin')
    bad_targets = replace(t, nearest=torch.tensor([1, 0]))
    with pytest.raises(ValueError, match='sentinel'):
        build_negative_masks(b, bad_targets)
    bad_targets = replace(t, acceptable=~b.candidate_mask, covered=torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match='available'):
        build_negative_masks(b, bad_targets)
    bad_hz = b.candidate_hz.clone(); bad_hz[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        build_negative_masks(replace(b, candidate_hz=bad_hz), t)
