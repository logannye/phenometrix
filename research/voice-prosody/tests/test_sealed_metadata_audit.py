import json
from pathlib import Path
import pytest
from phenometrix_prosody import sealed_metadata_audit as audit


def fixture(tmp_path):
    data, runs = tmp_path / 'data', tmp_path / 'runs'
    meta = data / 'metadata'; meta.mkdir(parents=True); runs.mkdir()
    prompts = {'si1':'Training text.', 'si2':'Development text.', 'si5':'Pair five A.',
               'si6':'Pair five B.', 'si7':'Pair six A.', 'si8':'Pair six B.'}
    prompt_path = meta / 'TIMIT-PROMPTS.txt'
    prompt_path.write_text(''.join(f'{text} ({p})\n' for p,text in prompts.items()))
    rows = []
    members = [('F04','train',['si1']),('F01','development',['si2']),
               ('F05','sealed_test',['si5','si6']),('M05','sealed_test',['si5','si6']),
               ('F06','sealed_test',['si7','si8']),('M06','sealed_test',['si7','si8'])]
    for speaker, split, ids in members:
        for p in ids:
            for role in ('mic','ref','lar'):
                rows.append({'recording_id':speaker+'_'+p,'speaker':speaker,'pair':int(speaker[1:]),
                    'prompt':p,'role':role,'split':split,'primary_eligible':True})
    inventory = {'seed':audit.SEED,'split_assignment':{'train':[4,8,7,9,10,2],'development':[1,3],'sealed_test':[5,6]},
        'records':rows,'sources':[{'path':'metadata/TIMIT-PROMPTS.txt','sha256':audit.sha(prompt_path.read_bytes())}]}
    inventory_path = meta / 'inventory.json';inventory_path.write_text(json.dumps(inventory))
    expected = audit.sha(inventory_path.read_bytes())
    acquired = {'inventory_sha256':expected,'files':[rows[0]]}
    (data/'manifest.json').write_text(json.dumps(acquired))
    (runs/'prior').mkdir();(runs/'prior/manifest.json').write_text(json.dumps({'split':'train','recordings':[{'recording_id':'F04_si1'}]}))
    return data,runs,expected


def run(data,runs,expected):
    return audit.audit_metadata(data,runs,expected_inventory_sha256=expected,per_speaker=2)


def test_metadata_only_paired_selection_and_prior_scope(tmp_path,monkeypatch):
    data,runs,expected=fixture(tmp_path)
    real=Path.read_bytes
    def guarded(path):
        assert path.suffix not in ('.f0','.wav','.npz','.pt'), 'opened data content'
        return real(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    result=run(data,runs,expected)
    assert result['selected_recording_count']==8
    assert result['selected_unique_prompt_count']==4
    assert result['prior_metadata_files_checked']==1
    assert result['audio_or_reference_content_opened'] is False
    assert all(x['normalized_text_count']==0 for x in result['cross_split_intersections'].values())
    assert result['speaker_selection']['F05']['ordered_prompt_ids_sha256']==result['speaker_selection']['M05']['ordered_prompt_ids_sha256']


def test_detects_acquired_sealed_metadata_without_opening_reference(tmp_path):
    data,runs,expected=fixture(tmp_path)
    p=data/'manifest.json';value=json.loads(p.read_text())
    value['files'].append({'recording_id':'F05_si5','speaker':'F05','split':'sealed_test','role':'ref'})
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='acquired sealed'):run(data,runs,expected)


def test_detects_filename_and_history_but_never_reads_corpus(tmp_path,monkeypatch):
    data,runs,expected=fixture(tmp_path)
    p=data/'ref_F05_si5.f0';p.write_text('not read')
    with pytest.raises(ValueError,match='filenames'):run(data,runs,expected)
    p.unlink()
    (runs/'prior/manifest.json').write_text(json.dumps({'reference_hashes':{'F06_si7':'hash'}}))
    with pytest.raises(ValueError,match='Prior run metadata'):run(data,runs,expected)


def test_changed_metadata_and_inventory_are_rejected(tmp_path):
    data,runs,expected=fixture(tmp_path)
    with pytest.raises(ValueError,match='Inventory identity'):run(data,runs,'wrong')
    (data/'metadata/TIMIT-PROMPTS.txt').write_text('changed')
    with pytest.raises(ValueError,match='metadata source changed'):run(data,runs,expected)


def test_cross_split_normalized_text_leakage_rejected(tmp_path):
    data,runs,expected=fixture(tmp_path)
    p=data/'metadata/TIMIT-PROMPTS.txt';p.write_text(p.read_text().replace('Pair five A.',' TRAINING  TEXT.'))
    ip=data/'metadata/inventory.json';i=json.loads(ip.read_text());i['sources'][0]['sha256']=audit.sha(p.read_bytes());ip.write_text(json.dumps(i))
    expected=audit.sha(ip.read_bytes())
    with pytest.raises(ValueError,match='Cross-split'):run(data,runs,expected)


def test_content_reader_refuses_audio_and_reference(tmp_path):
    for name in ('mic_F05_si5.wav','ref_F05_si5.f0','ref_F05_si5.f0.txt'):
        with pytest.raises(ValueError,match='non-metadata'):audit.metadata_bytes(tmp_path/name)
