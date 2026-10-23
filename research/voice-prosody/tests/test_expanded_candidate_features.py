"""Known-answer additive algebra only; no new scientific signal generation."""
import copy
import numpy as np
import pytest

from phenometrix_prosody import expanded_candidate_features as ec
from phenometrix_prosody.model import FEATURE_NAMES_V2


def feature(candidates=()):
    x = np.zeros((1, 57), np.float32)
    hz = np.full((1, 6), np.nan)
    mask = np.zeros((1, 6), bool)
    for j, (f, scores) in enumerate(sorted(candidates)):
        x[0, j * 8:(j + 1) * 8] = [1, np.log2(f / 50), *scores, max(scores)]
        hz[0, j], mask[0, j] = f, True
    x[0, 48:] = [-3, .2, .003, .01, .4, .7, .9, len(candidates) / 6, 1]
    return dict(features=x, candidate_hz=hz, candidate_mask=mask, reference_sample=np.array([2208]),
        feature_names=np.array(FEATURE_NAMES_V2), sample_rate_hz=np.array(48000), pcm_sample_count=np.array(2400))


def evidence(old=None, new=None):
    out = dict(pcm_sha256='a' * 64, pcm_sample_count=2400, ends=np.array([2208]), rss_bytes=1234)
    for label, pair in [('old', old), ('new', new)]:
        f, c = pair if pair is not None else (np.nan, 0.)
        out[label + '_hz'] = np.array([f], np.float64)
        out[label + '_confidence'] = np.array([c], np.float64)
        out[label + '_log2_hz_over_50'] = np.array([np.log2(f / 50) if np.isfinite(f) else 0.])
    return out


def old_provenance(e):
    p = {f'{label}_full40_{field}': e[f'{label}_{field}'].copy()
         for label in ('old', 'new') for field in ('hz', 'confidence')}
    p.update(candidate_policy=np.array(ec.previous.POLICY), pcm_sha256=np.array(e['pcm_sha256']),
             worker_rss_bytes=np.array(e['rss_bytes']))
    return p


def assert_retained(original, expanded, p):
    old = original['features'][:, :48].reshape(-1, 6, 8)
    new = expanded['features'][:, :56].reshape(-1, 7, 8)
    for i, j in zip(*np.where(original['candidate_mask'])):
        k = p['original_to_expanded_slot'][i, j]
        assert expanded['candidate_hz'][i, k] == original['candidate_hz'][i, j]
        np.testing.assert_array_equal(new[i, k, [0, 1, 2, 3, 4, 6]], old[i, j, [0, 1, 2, 3, 4, 6]])
        assert new[i, k, 5] >= old[i, j, 5]
        assert new[i, k, 7] >= old[i, j, 7]
    assert not p['old_frequency_evicted'].any()
    np.testing.assert_array_equal(expanded['features'][:, [56, 57, 58, 59, 60, 61, 62, 64]],
                                  original['features'][:, [48, 49, 50, 51, 52, 53, 54, 56]])


def test_empty_has_explicit_unavailable_seventh_slot_and_versioned_names():
    f = feature()
    out, p = ec.additive_from_evidence(f, evidence())
    assert out['features'].shape == (1, 65) and out['candidate_hz'].shape == (1, 7)
    assert not out['candidate_mask'].any() and np.isnan(out['candidate_hz']).all()
    assert np.all(p['original_to_expanded_slot'] == -1)
    assert len(ec.FEATURE_NAMES_EXPANDED) == len(set(ec.FEATURE_NAMES_EXPANDED)) == 65
    assert ec.FEATURE_NAMES_EXPANDED[63] == 'valid_candidate_count_over_six'
    assert str(p['feature_schema']) == ec.SCHEMA and str(p['candidate_policy']) == ec.POLICY
    assert_retained(f, out, p)


def test_seven_bound_preserves_all_six_sources_and_count_over_six():
    f = feature([(100, [0, 0, 0, .95, 0])] +
                [(150 + 50 * i, [.7, .6, 0, 0, .5]) for i in range(5)])
    before = copy.deepcopy(f)
    out, p = ec.additive_from_evidence(f, evidence((100, .95), (650, .75)))
    np.testing.assert_array_equal(out['candidate_hz'][0], [100, 150, 200, 250, 300, 350, 650])
    assert out['features'][0, 63] == np.float32(7 / 6)
    assert p['new_frequency_added'][0] and p['new_candidate_count'][0] == 7
    assert not p['existing_full40_evidence_enhanced'][0]
    assert_retained(f, out, p)
    for key in f:
        np.testing.assert_array_equal(f[key], before[key])
        assert not np.shares_memory(f[key], out[key])


def test_collision_adds_real_source_without_erasing_old_full40():
    f = feature([(100, [.8, .7, 0, .9, 0]), (300, [.6, 0, .8, 0, .7])])
    out, p = ec.additive_from_evidence(f, evidence((100, .9), (300, .75)))
    b = out['features'][0, :56].reshape(7, 8)
    np.testing.assert_array_equal(b[0], f['features'][0, :8])
    np.testing.assert_array_equal(b[1, 2:7], np.array([.6, 0, .8, .75, .7], np.float32))
    assert p['existing_full40_evidence_enhanced'][0] and not p['new_frequency_added'][0]
    assert out['candidate_mask'].sum() == 2
    assert_retained(f, out, p)


def test_identical_shortest_strongest_only_pads_and_keeps_original_log_bits():
    f = feature([(123.456789, [.8, 0, 0, .9, 0]), (300, [0, 0, 0, 0, .7])])
    out, p = ec.additive_from_evidence(f, evidence((123.456789, .9), (123.456789, .9)))
    np.testing.assert_array_equal(out['features'][:, :48], f['features'][:, :48])
    np.testing.assert_array_equal(out['features'][:, 56:], f['features'][:, 48:])
    assert not p['new_frequency_added'].any() and not p['existing_full40_evidence_enhanced'].any()


def test_cached_and_live_routes_agree_and_do_not_mutate_evidence():
    f = feature([(100, [0, 0, 0, .9, 0])])
    e = evidence((100, .9), (299.123456789, .71)); before = copy.deepcopy(e)
    class Worker:
        def inspect(self, path, ends):
            assert path == 'input.f32le'
            np.testing.assert_array_equal(ends, f['reference_sample'])
            return e
    a, pa = ec.expanded_feature(f, 'input.f32le', Worker())
    b, pb = ec.expanded_from_provenance(f, old_provenance(e))
    for key in a: np.testing.assert_array_equal(a[key], b[key])
    for key in pa: np.testing.assert_array_equal(pa[key], pb[key])
    for key in e: np.testing.assert_array_equal(e[key], before[key])


@pytest.mark.parametrize('change', [
    lambda e: e.update(pcm_sample_count=2399),
    lambda e: e.update(ends=np.array([2209])),
    lambda e: e.update(pcm_sha256='not-a-sha'),
    lambda e: e['old_hz'].__setitem__(0, 101.),
    lambda e: e['old_confidence'].__setitem__(0, .8),
    lambda e: e['new_hz'].__setitem__(0, np.inf),
    lambda e: e['new_hz'].__setitem__(0, 701.),
    lambda e: e['new_confidence'].__setitem__(0, np.nan),
    lambda e: e['new_confidence'].__setitem__(0, .34),
    lambda e: e.update(new_hz=np.array([300.], np.float32)),
    lambda e: e['new_hz'].__setitem__(0, 90.),
    lambda e: e['new_confidence'].__setitem__(0, .95),
])
def test_unbound_mismatched_or_impossible_peak_evidence_refuses(change):
    f = feature([(100, [0, 0, 0, .9, 0])]); e = evidence((100, .9), (300, .7))
    change(e)
    with pytest.raises(ValueError): ec.additive_from_evidence(f, e)


def test_same_frequency_cannot_claim_different_peak_score():
    f = feature([(100, [0, 0, 0, .9, 0])])
    with pytest.raises(ValueError, match='ordering'):
        ec.additive_from_evidence(f, evidence((100, .9), (100, .8)))


def test_missing_new_peak_cannot_change_availability():
    f = feature([(100, [0, 0, 0, .9, 0])])
    with pytest.raises(ValueError, match='availability'):
        ec.additive_from_evidence(f, evidence((100, .9), None))


@pytest.mark.parametrize('change', [
    lambda f: f['features'].__setitem__((0, 1), .5),
    lambda f: f['features'].__setitem__((0, 55), .5),
    lambda f: f['features'].__setitem__((0, 8), 1),
    lambda f: f.update(reference_hz=np.array([100.])),
])
def test_invalid_original_or_reference_contamination_refuses(change):
    f = feature([(100, [0, 0, 0, .9, 0])]); change(f)
    with pytest.raises(ValueError): ec.additive_from_evidence(f, evidence((100, .9), (300, .7)))


def test_expanded_input_cannot_recursively_expand_or_use_wrong_cached_policy():
    f = feature([(100, [0, 0, 0, .9, 0])]); e = evidence((100, .9), (300, .7))
    out, p = ec.additive_from_evidence(f, e)
    with pytest.raises(ValueError): ec.additive_from_evidence(out, e)
    with pytest.raises(ValueError, match='historical'):
        ec.expanded_from_provenance(f, p)
