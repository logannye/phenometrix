"""Known-answer width, provenance, invariance and gradient checks; no fitting run."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from phenometrix_prosody.expanded_selector_model import (
    ExpandedHarmonicSelector, prepare_expanded_selector_batch as prepare,
    expanded_selector_metadata,
)
from phenometrix_prosody.harmonic_selector_model import HarmonicSelector
from phenometrix_prosody.selector_model import (
    prepare_selector_batch, build_pitch_targets, selector_loss,
    selector_probabilities, predict_pitch_hz,
)


def raw(rows, k=6):
    x = np.zeros((len(rows), 8 * k + 9), np.float32)
    hz = np.full((len(rows), k), np.nan)
    mask = np.zeros((len(rows), k), bool)
    for row, frequencies in enumerate(rows):
        for i, f in enumerate(frequencies):
            scores = [.2 + .05 * i, .1, 0, .15, .1]
            x[row, i * 8:(i + 1) * 8] = [1, np.log2(f / 50), *scores, max(scores)]
            hz[row, i], mask[row, i] = f, True
        x[row, 8 * k:] = [-3 + row * .01, .1, 0, 0, .2, .7, .8, len(frequencies) / 6, 1]
    return x, hz, mask


def pad_arrays(arrays):
    x, hz, mask = arrays
    new = np.zeros((len(x), 65), np.float32)
    new[:, :48], new[:, 56:] = x[:, :48], x[:, 48:]
    return new, np.pad(hz, ((0, 0), (0, 1)), constant_values=np.nan), np.pad(mask, ((0, 0), (0, 1)))


@pytest.mark.parametrize('mode', ['absolute', 'relative'])
@pytest.mark.parametrize('objective', ['nearest', 'acceptable-set'])
def test_six_slot_preparation_scaling_initialization_and_logits_exact(mode, objective):
    arrays = raw([[100, 100.15, 200, 300], [130, 260], [], [70, 140, 420, 690]])
    old = prepare_selector_batch(*arrays, representation='shared-candidates')
    new = prepare(*arrays)
    for name in old.__dataclass_fields__:
        if isinstance(getattr(old, name), torch.Tensor):
            torch.testing.assert_close(getattr(old, name), getattr(new, name), rtol=0, atol=0, equal_nan=True)
    torch.manual_seed(20261022); a = HarmonicSelector(mode, objective)
    torch.manual_seed(20261022); b = ExpandedHarmonicSelector(mode, objective)
    assert sum(p.numel() for p in b.parameters()) == 3009
    assert a.state_dict().keys() == b.state_dict().keys()
    for key in a.state_dict(): torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
    a.fit_scaling(old, split='train'); b.fit_scaling(new, split='train')
    for key in a.state_dict(): torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(a(old), b(new), rtol=0, atol=0)
    torch.testing.assert_close(a.message_inputs(old), b.message_inputs(new), rtol=0, atol=0)
    b.load_state_dict(a.state_dict(), strict=True)
    assert expanded_selector_metadata(b)['expanded_features_permitted_for_voicing'] is False


@pytest.mark.parametrize('mode', ['absolute', 'relative'])
def test_zero_seventh_padding_invariance_with_shared_old_scaling(mode):
    arrays = raw([[100, 200, 300], [], [110, 220, 330, 440, 550, 660]])
    six, seven = prepare(*arrays), prepare(*pad_arrays(arrays))
    model = ExpandedHarmonicSelector(mode); model.fit_scaling(six, split='train')
    torch.testing.assert_close(six.features[..., :48], seven.features[..., :48], rtol=0, atol=0)
    torch.testing.assert_close(six.features[..., 48:], seven.features[..., 56:], rtol=0, atol=0)
    torch.testing.assert_close(model(six), model(seven)[..., :6], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(predict_pitch_hz(model(six), six), predict_pitch_hz(model(seven), seven), equal_nan=True)
    assert torch.all(model(seven)[..., 6] == -1e9)


def test_scaling_refuses_seven_and_nontraining_even_with_same_padding():
    six = prepare(*raw([[100, 200], [110]]))
    seven = prepare(*pad_arrays(raw([[100, 200], [110]])))
    model = ExpandedHarmonicSelector()
    with pytest.raises(ValueError, match='six-candidate'): model.fit_scaling(seven, split='train')
    with pytest.raises(ValueError, match='training'): model.fit_scaling(six, split='development')


def test_exact_duplicate_insertion_into_seventh_slot_does_not_change_grouped_prediction():
    arrays = raw([[100, 200, 300], [110, 220, 330]])
    x, hz, mask = pad_arrays(arrays)
    x[:, 48:56] = x[:, 8:16]; hz[:, 6] = hz[:, 1]; mask[:, 6] = True; x[:, 63] = 4 / 6
    six, seven = prepare(*arrays), prepare(x, hz, mask)
    torch.testing.assert_close(six.features[:, :48], seven.features[:, :48], rtol=0, atol=0)
    torch.testing.assert_close(six.features[:, 48:], seven.features[:, 56:], rtol=0, atol=0)
    assert seven.member_mask[:, 1, [1, 6]].all()
    torch.manual_seed(9); model = ExpandedHarmonicSelector(); model.fit_scaling(six, split='train')
    torch.testing.assert_close(model(six), model(seven)[:, :6], rtol=1e-6, atol=1e-6)


def test_seven_distinct_candidates_count_and_original_member_provenance():
    arrays = raw([[70, 100, 150, 200, 300, 450, 690]], k=7)
    batch = prepare(*arrays)
    assert batch.features[0, 63] == np.float32(7 / 6)
    assert torch.equal(batch.member_mask[0], torch.eye(7, dtype=torch.bool))
    assert batch.candidate_hz.dtype == torch.float64 and batch.candidate_hz.device.type == 'cpu'
    for name in ('original_features', 'original_candidate_hz', 'original_candidate_mask'):
        original = getattr(batch, name).clone()
        batch.features.zero_()
        torch.testing.assert_close(getattr(batch, name), original, equal_nan=True)


def permuted(batch, perm):
    k = len(perm)
    x = batch.features.clone()
    x[..., :8 * k] = x[..., :8 * k].reshape(*x.shape[:-1], k, 8)[..., perm, :].reshape(*x.shape[:-1], 8 * k)
    return replace(batch, features=x, candidate_hz=batch.candidate_hz[..., perm],
                   candidate_mask=batch.candidate_mask[..., perm], member_mask=batch.member_mask[..., perm, :])


@pytest.mark.parametrize('mode', ['absolute', 'relative'])
def test_seven_permutation_equivariance_and_lowest_hz_ties(mode):
    batch = prepare(*raw([[70, 100, 150, 200, 300, 450, 690], [110, 220], []], k=7))
    model = ExpandedHarmonicSelector(mode)
    model.fit_scaling(prepare(*raw([[100, 200], [110, 220]])), split='train')
    perm = torch.tensor([6, 2, 0, 5, 1, 4, 3]); other = permuted(batch, perm)
    torch.testing.assert_close(model(other), model(batch)[..., perm], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.relation_context(other), model.relation_context(batch)[..., perm, :], rtol=1e-6, atol=1e-6)
    tied = torch.zeros_like(batch.candidate_mask, dtype=torch.float32)
    torch.testing.assert_close(predict_pitch_hz(tied, other), predict_pitch_hz(tied, batch), equal_nan=True)


def test_added_raw_candidate_can_replace_group_representative_and_lose_coverage():
    # Old emitted member is49c above truth; added stronger member is53c above,
    # only4c away. Raw coverage is retained but actual group output is not.
    original_hz, added_hz = 200 * 2 ** (49 / 1200), 200 * 2 ** (53 / 1200)
    old = prepare(*raw([[original_hz]]))
    arrays = raw([[original_hz, added_hz]], k=7)
    new = prepare(*arrays)
    old_target = build_pitch_targets(old, [200.], [True], [True])
    raw_new = prepare(*arrays, representation='fixed-slots')
    assert old_target.covered.item() and build_pitch_targets(raw_new, [200.], [True], [True]).covered.item()
    assert not build_pitch_targets(new, [200.], [True], [True]).covered.item()
    assert new.candidate_hz[0, 0] == added_hz and new.member_mask[0, 0, :2].all()


def test_complete_span_grouping_is_not_single_link_chaining_and_merges_sources():
    base = 200.
    frequencies = [base, base * 2 ** (4 / 1200), base * 2 ** (8 / 1200)]
    x, hz, mask = raw([frequencies], k=7)
    x[0, 3] = .9; x[0, 7] = .9  # first candidate owns AMDF maximum
    x[0, 8 + 4] = .95; x[0, 8 + 7] = .95  # next owns ZC and representative
    batch = prepare(x, hz, mask)
    assert batch.candidate_mask.sum() == 2 and batch.member_mask[0, 0, :2].all()
    assert not batch.member_mask[0, 0, 2] and batch.member_mask[0, 1, 2]
    assert batch.features[0, 3] == np.float32(.9) and batch.features[0, 4] == np.float32(.95)
    assert batch.candidate_hz[0, 0] == frequencies[1]


@pytest.mark.parametrize('change', [
    lambda x, h, m: x.__setitem__((0, 0), 0),
    lambda x, h, m: x.__setitem__((0, 7), .99),
    lambda x, h, m: x.__setitem__((0, 16), 1),
    lambda x, h, m: x.__setitem__((0, 63), 2 / 7),
    lambda x, h, m: h.__setitem__((0, 0), 701),
    lambda x, h, m: x.__setitem__((0, 2), -1),
    lambda x, h, m: x.__setitem__((0, 60), np.nan),
])
def test_malformed_features_count_range_padding_or_nonfinite_refuse(change):
    arrays = raw([[100, 200]], k=7); change(*arrays)
    with pytest.raises(ValueError): prepare(*arrays)


def test_eight_candidates_refused_and_fixed_slots_only_preprocessing():
    with pytest.raises(ValueError, match='six or seven'): prepare(*raw([[100]], k=8))
    batch = prepare(*raw([[100]], k=7), representation='fixed-slots')
    with pytest.raises(ValueError, match='representation'): ExpandedHarmonicSelector()(batch)


@pytest.mark.parametrize('objective', ['nearest', 'acceptable-set'])
def test_seven_loss_targets_masks_empty_rows_and_gradients(objective):
    batch = prepare(*raw([[70, 100, 150, 200, 300, 450, 690], [], [110, 220]], k=7))
    model = ExpandedHarmonicSelector(objective=objective)
    model.fit_scaling(prepare(*raw([[100, 200], [110, 220]])), split='train')
    target = build_pitch_targets(batch, [200., np.nan, 440.], [True, False, True], [True, True, True])
    assert target.covered.tolist() == [True, False, False] and target.nearest.tolist() == [3, -1, -1]
    logits = model(batch)
    assert torch.isfinite(logits).all() and selector_probabilities(logits, batch.candidate_mask)[1].sum() == 0
    loss = selector_loss(logits, batch.candidate_mask, target, objective=objective)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad()
    none = build_pitch_targets(batch, [np.nan] * 3, [False] * 3, [False] * 3)
    zero = selector_loss(model(batch), batch.candidate_mask, none, objective=objective)
    assert zero.item() == 0; zero.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() == 0 for p in model.parameters())


def test_frame_order_chunking_metadata_and_inputs_are_immutable():
    arrays = raw([[70, 100, 150, 200, 300, 450, 690], [110, 220], [120, 240], []], k=7)
    originals = tuple(a.copy() for a in arrays)
    batch = prepare(*arrays)
    model = ExpandedHarmonicSelector(); expected = model(batch)
    order = np.array([3, 0, 2, 1]); shuffled = batch.to('cpu').index_select(order)
    torch.testing.assert_close(model(shuffled), expected[order], atol=1e-6, rtol=1e-6)
    for i in range(4):
        torch.testing.assert_close(model(batch.index_select(np.array([i])))[0], expected[i], atol=1e-6, rtol=1e-6)
    assert shuffled.candidate_hz.dtype == torch.float64 and shuffled.candidate_hz.device.type == 'cpu'
    for a, b in zip(arrays, originals): np.testing.assert_array_equal(a, b)
    if torch.backends.mps.is_available():
        moved = batch.to('mps').index_select(torch.tensor([0, 2]))
        assert moved.features.device.type == 'mps' and moved.candidate_hz.device.type == 'cpu'
        assert moved.member_mask.device.type == 'cpu' and moved.original_features.device.type == 'cpu'
