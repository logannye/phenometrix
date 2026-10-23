"""Metadata and refusal paths only: never generate prospective challenge PCM."""
import hashlib
import json
from pathlib import Path
import pytest
from phenometrix_prosody import subharmonic_challenge as challenge


def fake_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(challenge, 'REPO', tmp_path)
    monkeypatch.setattr(challenge, 'RUNS', tmp_path)
    monkeypatch.setattr(challenge, 'challenge_source_hashes', lambda: {'toy.py': 'a'*64})
    run = tmp_path/'run'; (run/'checkpoints').mkdir(parents=True)
    fits = {}
    for name in challenge._expected_fits():
        p = run/'checkpoints'/(name+'.pt'); p.write_bytes(b'test-only checkpoint')
        fits[name] = {'weights_sha256': challenge.previous._sha(p)}
    value = {'status': 'checkpoints_frozen', 'challenge_spec_sha256': challenge.challenge_spec_sha256(),
             'source_sha256': {'toy.py': 'a'*64}, 'fits': fits}
    path = run/'checkpoint-freeze.json'; path.write_text(json.dumps(value))
    return path, value


def test_fixed_spec_is_new_metadata_only():
    spec = challenge.challenge_spec(); records = spec['records']
    assert len(records) == 12 and len({r['id'] for r in records}) == 12
    assert {r['frequency_hz'] for r in records} == {79.7,151.3,319.7,354.5,503.9,659.3}
    assert {tuple(r['phase_radians']) for r in records} == {(.19,-.83,.67),(1.43,.59,-1.07)}
    assert not set(challenge.FREQUENCIES) & set(challenge.previous.FREQUENCIES)
    assert not set(challenge.PHASES) & set(challenge.previous.PHASES)
    assert all(r['id'] == f'subharmonic-balance-f{i//2}-p{i%2}' for i,r in enumerate(records))
    assert all((r['sample_rate_hz'],r['sample_count'],r['change_sample'],r['gain']) == (48000,96000,39840,.06) for r in records)
    assert all(r['before'] == [1,.5,.25] and r['after'] == [0,1,.5] for r in records)
    assert spec['predesignated_candidate'] == 'alias-margin'
    assert challenge.challenge_spec_sha256() == hashlib.sha256(challenge.previous._canonical(spec)).hexdigest()


def test_source_closure_includes_both_helpers_and_replay():
    paths = challenge.challenge_source_hashes()
    assert paths.items() >= challenge.previous.challenge_source_hashes().items()
    assert str(Path(challenge.__file__).relative_to(challenge.REPO)) in paths
    assert 'research/voice-prosody/phenometrix_prosody/harmonic_challenge.py' in paths
    assert 'research/voice-prosody/replay.ts' in paths
    assert 'apps/capture-web/src/voice-pitch-features-v2.ts' in paths


def test_exact_nine_fit_inventory_accepts_frozen_metadata(tmp_path, monkeypatch):
    path, value = fake_freeze(tmp_path, monkeypatch)
    assert len(value['fits']) == 9
    assert set(value['fits']) == {f'{arm}__seed{seed}' for arm in ('set','all-margin','alias-margin') for seed in (20261022,20261023,20261024)}
    challenge._validate_freeze(path, challenge.previous._sha(path))


def test_missing_freeze_prevents_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(challenge, 'RUNS', tmp_path)
    with pytest.raises(ValueError, match='freeze is required'):
        challenge.generate_challenge(tmp_path/'challenge', tmp_path/'missing.json', '0'*64)
    assert not (tmp_path/'challenge').exists()


@pytest.mark.parametrize('field,value,match', [
    ('status','fitting','All checkpoints'),
    ('challenge_spec_sha256','b'*64,'specification'),
    ('source_sha256',{},'source closure'),
    ('fits',{},'declared nine'),
])
def test_incomplete_freeze_blocks_generation_before_any_output(tmp_path, monkeypatch, field, value, match):
    path, frozen = fake_freeze(tmp_path, monkeypatch)
    frozen[field] = value; path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match=match):
        challenge.generate_challenge(tmp_path/'challenge', path, challenge.previous._sha(path))
    assert not (tmp_path/'challenge').exists()


def test_previous_twelve_fit_inventory_is_not_accepted(tmp_path, monkeypatch):
    path, frozen = fake_freeze(tmp_path, monkeypatch)
    frozen['fits'] = {k:{'weights_sha256':'a'*64} for k in challenge.previous._expected_fits()}
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='declared nine'):
        challenge._validate_freeze(path, challenge.previous._sha(path))


def test_changed_freeze_identity_rejected(tmp_path, monkeypatch):
    path, _ = fake_freeze(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='identity changed'):
        challenge._validate_freeze(path, '0'*64)


@pytest.mark.parametrize('remove',[False, True])
def test_changed_or_missing_checkpoint_rejected(tmp_path, monkeypatch, remove):
    path, frozen = fake_freeze(tmp_path, monkeypatch)
    weight = path.parent/'checkpoints'/(next(iter(frozen['fits']))+'.pt')
    if remove: weight.unlink()
    else: weight.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='checkpoint changed or missing'):
        challenge._validate_freeze(path, challenge.previous._sha(path))


def test_integer_support_boundary_is_unchanged():
    scenario = challenge.previous._scenario(challenge.challenge_spec()['records'][0])
    assert scenario['initialHz'] == scenario['finalHz'] == 79.7
    assert challenge.challenge_region_at(scenario,39840) == 'before-observation-change'
    assert challenge.challenge_region_at(scenario,39841) == 'observation-boundary'
    assert challenge.challenge_region_at(scenario,41759) == 'observation-boundary'
    assert challenge.challenge_region_at(scenario,41760) == 'after-observation-change'
