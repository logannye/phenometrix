from dataclasses import replace
import numpy as np
import pytest
import torch

from phenometrix_prosody.selector_model import (
    SelectorConfig, PitchSelector, SlotSelector, SetSelector, PitchTargets,
    build_pitch_targets, prepare_selector_batch, predict_pitch_hz,
    selector_loss, selector_metadata, selector_probabilities,
)


def fixture(rows, *, scores=None, representation="fixed-slots"):
    n = len(rows)
    hz = np.full((n, 6), np.nan)
    mask = np.zeros((n, 6), bool)
    x = np.zeros((n, 57), np.float32)
    for row, frequencies in enumerate(rows):
        for slot, frequency in enumerate(frequencies):
            hz[row, slot], mask[row, slot] = frequency, True
            evidence = np.array([.2 + .03 * slot, .1, 0, .15, .1], np.float32)
            if scores is not None:
                evidence = np.asarray(scores[row][slot], np.float32)
            x[row, slot * 8:(slot + 1) * 8] = [1, np.log2(frequency / 50), *evidence, evidence.max()]
        x[row, 48:] = [-3 + row * .01, .1, 0, 0, .2, .7, .8, len(frequencies) / 6, 1]
    return prepare_selector_batch(x, hz, mask, representation=representation)


def reprepare(batch, representation):
    return prepare_selector_batch(batch.original_features, batch.original_candidate_hz,
                                  batch.original_candidate_mask, representation=representation)


@pytest.mark.parametrize("kind,count", [("fixed-slots", 3110), ("shared-candidates", 2689)])
@pytest.mark.parametrize("objective", ["nearest", "acceptable-set"])
def test_exact_bounded_architecture_has_no_voice_or_state(kind, count, objective):
    model = PitchSelector(SelectorConfig(kind, objective))
    batch = fixture([[100, 200, 300], []], representation=kind)
    assert sum(p.numel() for p in model.parameters()) == count
    assert not any("voic" in name or "state" in name for name, _ in model.named_parameters())
    logits = model(batch)
    assert logits.shape == (2, 6) and torch.isfinite(logits).all()
    assert selector_metadata(model)["parameter_count"] == count
    assert torch.isnan(predict_pitch_hz(logits, batch)[1])
    assert selector_probabilities(logits, batch.candidate_mask)[1].sum() == 0


def test_config_rejects_unbounded_changes():
    for kwargs in [dict(hidden_size=64), dict(representation="gru"), dict(loss="huber")]:
        with pytest.raises(ValueError):
            SelectorConfig(**kwargs)


def test_complete_link_is_not_adjacent_chaining_and_uses_input_representative():
    f = [200 * 2 ** (c / 1200) for c in [0, 4, 8]]
    batch = fixture([f], scores=[[[.2, 0, 0, 0, 0], [0, .8, 0, 0, 0], [0, 0, .6, 0, 0]]],
                    representation="shared-candidates")
    assert batch.candidate_mask.tolist() == [[True, True, False, False, False, False]]
    assert batch.member_mask[0, 0].tolist() == [True, True, False, False, False, False]
    assert batch.member_mask[0, 1].tolist() == [False, False, True, False, False, False]
    assert batch.candidate_hz[0, 0] == f[1]  # greatest evidence, not centroid or minimum
    torch.testing.assert_close(batch.features[0, 2:7], torch.tensor([.2, .8, 0, 0, 0]))
    assert batch.features[0, 7] == pytest.approx(.8)
    assert batch.features[0, 55] == pytest.approx(2 / 6)
    assert batch.original_candidate_mask.sum() == 3
    assert torch.equal(batch.original_features[0, :24].reshape(3, 8)[:, 0], torch.ones(3))


def test_group_representative_evidence_tie_uses_lowest_hz():
    f = [201., 201 * 2 ** (3 / 1200)]
    batch = fixture([f[::-1]], scores=[[[.5, 0, 0, 0, 0]] * 2], representation="shared-candidates")
    assert batch.candidate_hz[0, 0] == f[0]


def test_input_permutations_canonicalize_identically_and_preserve_provenance():
    raw = fixture([[100, 100.1, 150, 299, 300]])
    expected = reprepare(raw, "shared-candidates")
    generator = np.random.default_rng(20261022)
    for _ in range(20):
        perm = generator.permutation(6)
        x = raw.features.numpy().copy()
        x[:, :48] = x[:, :48].reshape(1, 6, 8)[:, perm].reshape(1, 48)
        got = prepare_selector_batch(x, raw.candidate_hz[:, perm], raw.candidate_mask[:, perm], representation="shared-candidates")
        torch.testing.assert_close(got.features, expected.features, atol=0, rtol=0)
        torch.testing.assert_close(got.candidate_hz, expected.candidate_hz, atol=0, rtol=0, equal_nan=True)
        assert torch.equal(got.member_mask, expected.member_mask[:, :, perm])


def test_shared_network_is_equivariant_even_without_canonical_sort():
    torch.manual_seed(17)
    batch = fixture([[100, 200, 300], [110, 220]], representation="shared-candidates")
    model = SetSelector(); model.fit_scaling(batch, split="train")
    perm = torch.tensor([2, 4, 0, 5, 1, 3])
    x = batch.features.clone()
    x[:, :48] = x[:, :48].reshape(2, 6, 8)[:, perm].reshape(2, 48)
    altered = replace(batch, features=x, candidate_hz=batch.candidate_hz[:, perm],
                      candidate_mask=batch.candidate_mask[:, perm], member_mask=batch.member_mask[:, perm])
    torch.testing.assert_close(model(altered), model(batch)[:, perm], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(predict_pitch_hz(model(altered), altered), predict_pitch_hz(model(batch), batch))


def test_exact_duplicate_insertion_invariance_includes_scaling_loss_and_global_count():
    raw = fixture([[100, 200, 300], [105, 210, 315]])
    x, hz, mask = raw.features.numpy().copy(), raw.candidate_hz.numpy().copy(), raw.candidate_mask.numpy().copy()
    x[:, 24:32] = x[:, 8:16]; hz[:, 3] = hz[:, 1]; mask[:, 3] = True; x[:, 55] = 4 / 6
    original = reprepare(raw, "shared-candidates")
    copied = prepare_selector_batch(x, hz, mask, representation="shared-candidates")
    torch.testing.assert_close(copied.features, original.features, rtol=0, atol=0)
    assert copied.member_mask[:, 1].sum() == 4
    torch.manual_seed(19); a = SetSelector(); a.fit_scaling(original, split="train")
    torch.manual_seed(19); b = SetSelector(); b.fit_scaling(copied, split="train")
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    ta = build_pitch_targets(original, [200., 210.], [True, True], [True, True])
    tb = build_pitch_targets(copied, [200., 210.], [True, True], [True, True])
    for objective in ["nearest", "acceptable-set"]:
        torch.testing.assert_close(selector_loss(a(original), original.candidate_mask, ta, objective=objective),
                                   selector_loss(b(copied), copied.candidate_mask, tb, objective=objective), atol=0, rtol=0)


def test_group_targets_use_emitted_hz_not_covered_members_and_reportable_loss():
    raw = fixture([[200 * 2 ** (49 / 1200), 200 * 2 ** (52 / 1200)]],
                  scores=[[[.2, 0, 0, 0, 0], [.8, 0, 0, 0, 0]]])
    grouped = reprepare(raw, "shared-candidates")
    before = build_pitch_targets(raw, [200.], [True], [True])
    after = build_pitch_targets(grouped, [200.], [True], [True])
    assert before.covered.item() and not after.covered.item()
    assert before.nearest.item() == 0 and after.nearest.item() == -1
    assert not after.acceptable.any()
    assert grouped.candidate_hz[0, 0] == raw.candidate_hz[0, 1]


def test_grouping_cannot_create_a_frequency_or_reference_coverage_gain():
    generator = np.random.default_rng(13)
    rows = [sorted(generator.uniform(80, 500, size=6)) for _ in range(25)]
    raw = fixture(rows); grouped = reprepare(raw, "shared-candidates")
    ref = generator.uniform(80, 500, size=25)
    before = build_pitch_targets(raw, ref, np.ones(25, bool), np.ones(25, bool))
    after = build_pitch_targets(grouped, ref, np.ones(25, bool), np.ones(25, bool))
    assert not torch.any(after.covered & ~before.covered)
    for row in range(25):
        assert set(grouped.candidate_hz[row, grouped.candidate_mask[row]].tolist()) <= set(rows[row])


def test_invalid_unvoiced_uncovered_and_all_masked_have_no_false_targets():
    batch = fixture([[200], [200], [200], [], [100, 200]])
    targets = build_pitch_targets(batch, [200., np.nan, 350., 200., np.nan],
                                  [False, True, True, True, False], [True, False, True, True, False])
    assert not targets.covered.any() and not targets.acceptable.any()
    assert targets.nearest.tolist() == [-1] * 5
    for objective in ["nearest", "acceptable-set"]:
        logits = torch.randn(5, 6, requires_grad=True)
        loss = selector_loss(logits, batch.candidate_mask, targets, objective=objective)
        assert loss.item() == 0 and torch.isfinite(loss)
        loss.backward()
        assert torch.equal(logits.grad, torch.zeros_like(logits))
    with pytest.raises(ValueError, match="finite positive"):
        build_pitch_targets(batch, [np.nan] * 5, [True] * 5, [True] * 5)


def test_acceptable_boundary_and_nearest_are_same_for_both_representations():
    f = [200 * 2 ** (49.999 / 1200), 200 * 2 ** (50.001 / 1200)]
    for kind in ["fixed-slots", "shared-candidates"]:
        # Separate rows avoid grouping changing the representative.
        batch = fixture([[f[0]], [f[1]]], representation=kind)
        targets = build_pitch_targets(batch, [200., 200.], [True, True], [True, True])
        assert targets.covered.tolist() == [True, False]
        assert targets.nearest.tolist() == [0, -1]


def test_set_loss_matches_probability_mass_and_accepts_either_good_candidate():
    batch = fixture([[199, 201, 300]])
    targets = build_pitch_targets(batch, [200.], [True], [True])
    logits = torch.tensor([[.3, 1.4, -.2, 999., 999., 999.]], requires_grad=True)
    p = torch.softmax(logits[0, :3], dim=0)
    loss = selector_loss(logits, batch.candidate_mask, targets, objective="acceptable-set")
    torch.testing.assert_close(loss, -torch.log(p[:2].sum()))
    assert loss < selector_loss(logits, batch.candidate_mask, targets, objective="nearest")
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.equal(logits.grad[0, 3:], torch.zeros(3))
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] < 0 and logits.grad[0, 2] > 0


def test_set_loss_zero_when_every_available_candidate_is_acceptable():
    batch = fixture([[199, 201], []])
    targets = build_pitch_targets(batch, [200., 200.], [True, True], [True, True])
    logits = torch.tensor([[1000., -1000., 0., 0., 0., 0.], [0.] * 6], requires_grad=True)
    loss = selector_loss(logits, batch.candidate_mask, targets, objective="acceptable-set")
    assert loss.item() == 0
    loss.backward(); assert torch.isfinite(logits.grad).all()


def test_loss_rejects_false_targets_instead_of_silently_learning_them():
    batch = fixture([[200]])
    targets = build_pitch_targets(batch, [200.], [True], [True])
    bad = PitchTargets(targets.acceptable.clone(), torch.tensor([1]), targets.covered)
    with pytest.raises(ValueError, match="acceptable"):
        selector_loss(torch.zeros(1, 6), batch.candidate_mask, bad)
    bad = PitchTargets(torch.ones(1, 6, dtype=torch.bool), targets.nearest, targets.covered)
    with pytest.raises(ValueError, match="available"):
        selector_loss(torch.zeros(1, 6), batch.candidate_mask, bad)


@pytest.mark.parametrize("kind", ["fixed-slots", "shared-candidates"])
def test_scaling_train_only_and_empty_candidates_finite(kind):
    batch = fixture([[100, 200], [], [300]], representation=kind)
    model = PitchSelector(SelectorConfig(kind))
    for split in ["dev", "development", "test", "sealed_test", "sealed-test"]:
        with pytest.raises(ValueError, match="training split"):
            model.fit_scaling(batch, split=split)
    model.fit_scaling(batch, split="train")
    assert model.scaling_fitted.item() and torch.isfinite(model(batch)).all()
    empty = fixture([[], []], representation=kind)
    model.fit_scaling(empty, split="train")
    assert torch.isfinite(model(empty)).all()
    with pytest.raises(ValueError, match="differs"):
        model(reprepare(batch, "shared-candidates" if kind == "fixed-slots" else "fixed-slots"))


def test_shared_scaling_pools_valid_groups_only_and_preserves_global_fields():
    batch = fixture([[100, 200], [300]], representation="shared-candidates")
    model = SetSelector(); model.fit_scaling(batch, split="train")
    candidates = batch.features[:, :48].reshape(2, 6, 8)[batch.candidate_mask]
    torch.testing.assert_close(model.candidate_mean, candidates.mean(dim=0))
    torch.testing.assert_close(model.global_mean, batch.features[:, 48:].mean(dim=0))
    assert model.candidate_mean[0] == 1  # unavailable padding did not dilute valid values


def test_slots_match_historical_pitch_topology_with_copied_weights():
    from phenometrix_prosody.model import VoiceCandidateModel, ModelConfig
    torch.manual_seed(33)
    old = VoiceCandidateModel(ModelConfig(kind="framewise-mlp"))
    new = SlotSelector()
    for layer in ["projection", "frame_hidden", "candidate_head"]:
        getattr(new, layer).load_state_dict(getattr(old, layer).state_dict())
    batch = fixture([[100, 200, 300], [150]])
    old.fit_scaling(batch.features, split="train"); new.fit_scaling(batch, split="train")
    expected = old(batch.features, batch.candidate_mask, old.initial_state(2))[1]
    torch.testing.assert_close(new(batch), expected, rtol=0, atol=0)


def test_batch_index_device_contract_and_extra_time_axis():
    batch = fixture([[100], [200], [300]], representation="shared-candidates")
    selected = batch.index_select(torch.tensor([2, 0]))
    assert selected.candidate_hz[:, 0].tolist() == [300., 100.]
    assert selected.original_features.shape == (2, 57)
    targets = build_pitch_targets(batch, [100., 200., 300.], [True] * 3, [True] * 3)
    assert targets.index_select(np.array([2, 0])).nearest.tolist() == [0, 0]
    device_view = batch.to("meta")  # proves no float64 metadata move without requiring a GPU
    assert device_view.features.device.type == "meta" and device_view.candidate_mask.device.type == "meta"
    for field in ["candidate_hz", "member_mask", "original_features", "original_candidate_hz", "original_candidate_mask"]:
        assert getattr(device_view, field).device.type == "cpu"
    assert device_view.candidate_hz.dtype == torch.float64
    timed = prepare_selector_batch(batch.original_features.reshape(1, 3, 57),
        batch.original_candidate_hz.reshape(1, 3, 6), batch.original_candidate_mask.reshape(1, 3, 6),
        representation="shared-candidates")
    model = SetSelector(); model.fit_scaling(timed, split="train")
    assert model(timed).shape == (1, 3, 6)
    torch.testing.assert_close(model(timed).reshape(3, 6), model(batch))


def test_prediction_ties_choose_hz_and_unavailable_has_no_distribution():
    batch = fixture([[300, 100, 200], []])
    logits = torch.tensor([[0., 0., 0., 1e6, 1e6, 1e6], [0.] * 6])
    result = predict_pitch_hz(logits, batch)
    assert result[0] == 100 and torch.isnan(result[1])
    probs = selector_probabilities(logits, batch.candidate_mask)
    torch.testing.assert_close(probs[0, :3], torch.ones(3) / 3)
    assert probs[0, 3:].sum() == 0 and probs[1].sum() == 0


def test_preparation_fails_closed_on_malformed_feature_contract():
    batch = fixture([[100]])
    for field, value in [(0, 0), (1, 99), (7, .99), (8, 1), (48, np.nan)]:
        x = batch.features.clone(); x[0, field] = value
        with pytest.raises(ValueError):
            prepare_selector_batch(x, batch.candidate_hz, batch.candidate_mask, representation="fixed-slots")
    hz = batch.candidate_hz.clone(); hz[0, 0] = -100
    with pytest.raises(ValueError):
        prepare_selector_batch(batch.features, hz, batch.candidate_mask, representation="fixed-slots")
