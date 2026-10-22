"""Prospective challenge access tests; no actual challenge waveform is generated."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from phenometrix_prosody import harmonic_challenge as hc


def test_spec_is_twelve_unique_unevaluated_frequency_phase_configurations(monkeypatch):
    monkeypatch.setattr(hc, '_synthesize_pcm', lambda _: pytest.fail('Spec must not generate PCM'))
    spec = hc.challenge_spec()
    assert len(spec['records']) == 12
    assert len({r['id'] for r in spec['records']}) == 12
    assert len({(r['frequency_hz'], tuple(r['phase_radians'])) for r in spec['records']}) == 12
    assert not {100, 200, 300} & {r['frequency_hz'] for r in spec['records']}
    assert all(r['after'] == [0., 1., .5] and r['change_sample'] == 39840 for r in spec['records'])
    changed = copy.deepcopy(spec); changed['records'][0]['gain'] = .075
    assert hashlib.sha256(hc._canonical(changed)).hexdigest() != hc.challenge_spec_sha256()


def test_missing_freeze_prevents_any_output_or_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, 'RUNS', tmp_path)
    monkeypatch.setattr(hc, '_synthesize_pcm', lambda _: pytest.fail('No PCM before freeze'))
    with pytest.raises(ValueError, match='freeze is required'):
        hc.generate_challenge(tmp_path/'challenge', tmp_path/'missing.json', '0'*64)
    assert not (tmp_path/'challenge').exists()


def fake_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, 'REPO', tmp_path); monkeypatch.setattr(hc, 'RUNS', tmp_path)
    monkeypatch.setattr(hc, 'challenge_source_hashes', lambda: {'toy.py': 'a'*64})
    run = tmp_path/'run'; (run/'checkpoints').mkdir(parents=True)
    fits = {}
    for name in hc._expected_fits():
        p=run/'checkpoints'/(name+'.pt'); p.write_bytes(b'toy test-only checkpoint')
        fits[name]={'weights_sha256':hc._sha(p)}
    value={'status':'checkpoints_frozen','challenge_spec_sha256':hc.challenge_spec_sha256(),
           'source_sha256':{'toy.py':'a'*64},'fits':fits}
    path=run/'checkpoint-freeze.json'; path.write_text(json.dumps(value))
    return path,value


def test_all_twelve_source_bound_checkpoint_files_are_required(tmp_path, monkeypatch):
    path,value=fake_freeze(tmp_path,monkeypatch)
    hc._validate_freeze(path,hc._sha(path))
    removed=next(iter(value['fits']));del value['fits'][removed]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='declared12'):
        hc._validate_freeze(path,hc._sha(path))


@pytest.mark.parametrize('field,value,match',[
    ('status','fitting','All checkpoints'),('challenge_spec_sha256','b'*64,'specification'),
    ('source_sha256',{},'source closure')])
def test_freeze_contract_rejections(tmp_path,monkeypatch,field,value,match):
    path,freeze=fake_freeze(tmp_path,monkeypatch);freeze[field]=value;path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError,match=match):hc._validate_freeze(path,hc._sha(path))


def test_mutated_weight_rejected_without_challenge_generation(tmp_path,monkeypatch):
    path,value=fake_freeze(tmp_path,monkeypatch)
    (path.parent/'checkpoints'/(next(iter(value['fits']))+'.pt')).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):hc._validate_freeze(path,hc._sha(path))


def test_toy_waveform_only_preserves_carrier_phase_and_float32():
    # Separate toy parameters, not any of the 12 prospectively frozen cases.
    r={'sample_count':48,'sample_rate_hz':48000,'frequency_hz':125.,'change_sample':24,
       'phase_radians':[0.,0.,0.],'before':[1.,0.,0.],'after':[0.,1.,.5],'gain':.01}
    x=hc._synthesize_pcm(r);t=np.arange(48)/48000
    expected=.01*np.where(np.arange(48)<24,np.sin(2*np.pi*125*t),np.sin(2*np.pi*250*t)+.5*np.sin(2*np.pi*375*t))
    assert x.dtype==np.float32
    assert np.array_equal(x,expected.astype(np.float32))


def test_scenario_fields_retain_exact_change_sample():
    s=hc._scenario(hc.challenge_spec()['records'][0])
    assert s['kind']=='observation-change' and s['changeStartSeconds']==s['changeEndSeconds']==.83
    assert s['initialHz']==s['finalHz']==87.3 and s['slopeHzPerSecond']==0


def test_challenge_regions_use_sample_boundary_not_historical_point6():
    scenario=hc._scenario(hc.challenge_spec()['records'][0])
    assert hc.challenge_region_at(scenario,39840)=='before-observation-change'
    assert hc.challenge_region_at(scenario,39841)=='observation-boundary'
    assert hc.challenge_region_at(scenario,41759)=='observation-boundary'
    assert hc.challenge_region_at(scenario,41760)=='after-observation-change'
    assert hc.challenge_region_at(scenario,35000)=='before-observation-change'
