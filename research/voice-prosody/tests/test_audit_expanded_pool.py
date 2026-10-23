"""Synthetic feature-only checks; no waveform generation or scientific artifacts."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from phenometrix_prosody import audit_expanded_pool as a


def feature(rows, k=6):
    n = len(rows); x = np.zeros((n, 8*k+9), np.float32)
    hz = np.full((n, k), np.nan); mask = np.zeros((n, k), bool)
    for i, entries in enumerate(rows):
        for j, (frequency, source, score) in enumerate(sorted(entries)):
            b = x[i, j*8:(j+1)*8]; b[:2] = [1, np.log2(frequency/50)]
            b[source] = b[7] = score; hz[i, j] = frequency; mask[i, j] = True
        x[i, 8*k+7] = len(entries)/6
    return dict(features=x, candidate_hz=hz, candidate_mask=mask,
        feature_names=np.array(a.FEATURE_NAMES_V2 if k == 6 else a.FEATURE_NAMES_EXPANDED),
        reference_sample=np.arange(n, dtype=np.int64)*480+2208,
        sample_rate_hz=np.array(48000), pcm_sample_count=np.array(2208+480*n))


def expand(original, old_h, new_h, old_c=.9, new_c=.8):
    """Explicit toy union independent of the production candidate adapter."""
    n = len(original['features']); k = 7
    x = np.zeros((n, 65), np.float32); x[:, 56:] = original['features'][:, 48:]
    hz = np.full((n, k), np.nan); mask = np.zeros((n, k), bool)
    mapping = np.full((n, 6), -1, np.int64); added = np.zeros(n, bool); enhanced = np.zeros(n, bool)
    for i in range(n):
        blocks = original['features'][i, :48].reshape(6, 8)
        entries = {float(f): blocks[j].copy() for j, f in enumerate(original['candidate_hz'][i]) if original['candidate_mask'][i, j]}
        if np.isfinite(new_h):
            added[i] = new_h not in entries
            enhanced[i] = new_h in entries and np.float32(new_c) > entries[new_h][5]
            if new_h not in entries:entries[new_h] = np.array([1, np.log2(new_h/50), 0, 0, 0, 0, 0, 0], np.float32)
            entries[new_h][5] = max(entries[new_h][5], np.float32(new_c))
        for j, (f, b) in enumerate(sorted(entries.items())):
            b[7] = b[2:7].max(); x[i, j*8:(j+1)*8] = b; hz[i, j] = f; mask[i, j] = True
            mapping[i, np.flatnonzero(original['candidate_mask'][i] & (original['candidate_hz'][i] == f))] = j
        x[i, 63] = len(entries)/6
    result = {k:np.array(v, copy=True) for k,v in original.items()}
    result.update(features=x, candidate_hz=hz, candidate_mask=mask, feature_names=np.array(a.FEATURE_NAMES_EXPANDED))
    p = dict(old_full40_hz=np.full(n, old_h), new_full40_hz=np.full(n, new_h),
        old_full40_confidence=np.full(n, old_c), new_full40_confidence=np.full(n, new_c),
        candidate_policy=np.array('add-shortest-admitted-full40-v1'),
        feature_schema=np.array('phenometrix.voice-pitch-features.expanded7.v1'),pcm_sha256=np.array('a'*64),
        original_to_expanded_slot=mapping, old_candidate_count=original['candidate_mask'].sum(1),
        new_candidate_count=mask.sum(1), new_frequency_added=added, existing_full40_evidence_enhanced=enhanced,
        old_frequency_evicted=np.zeros(n, bool), worker_rss_bytes=np.array(4096))
    return result, p


def test_adds_seventh_without_removing_any_original():
    old = feature([[(70+i*30, 5 if i==0 else 2, .9 if i==0 else .6) for i in range(6)]])
    new, p = expand(old, 70., 517.3)
    got = a.validate_expansion(old,new,p,'a'*64)
    assert got['added_distinct_candidate_frames'] == 1
    assert got['raw_candidate_evictions'] == 0
    assert new['features'][0,63] == np.float32(7/6)


def test_collision_enhances_source_without_duplicating_candidate():
    old = feature([[(100.,5,.9),(200.,2,.7)]])
    new,p = expand(old,100.,200.)
    assert a.validate_expansion(old,new,p,'a'*64)['already_present_peak_frames'] == 1
    assert new['candidate_mask'].sum() == 2 and new['features'][0,13] == np.float32(.8)


def test_absent_peak_preserves_empty_pool():
    old=feature([[]]);new,p=expand(old,np.nan,np.nan,0.,0.)
    assert a.validate_expansion(old,new,p,'a'*64)['absent_full40_frames']==1
    assert not a.independent_groups(new)['candidate_mask'].any()


@pytest.mark.parametrize('bad', ['eviction','score_loss','mapping','count','global','voice_schema'])
def test_rejects_pool_contract_changes(bad):
    old=feature([[(100.,5,.9),(200.,2,.7)]]);new,p=expand(old,100.,300.)
    if bad=='eviction':p['old_frequency_evicted'][0]=True
    elif bad=='score_loss':new['features'][0,5]=.8;new['features'][0,7]=.8
    elif bad=='mapping':p['original_to_expanded_slot'][0,0]=2
    elif bad=='count':new['features'][0,63]=3/7
    elif bad=='global':new['features'][0,56]=1
    else:new['feature_names'][0]='wrong'
    with pytest.raises(ValueError):a.validate_expansion(old,new,p,'a'*64)


def test_complete_span_grouping_does_not_chain_neighbors():
    hz=[100*2**(c/1200) for c in [0,4,8]]
    f=feature([[(h,2,.7) for h in hz]])
    g=a.independent_groups(f)
    assert g['candidate_mask'].sum()==2
    assert g['member_mask'][0,0,:3].tolist()==[True,True,False]
    assert g['candidate_hz'][0,0]==hz[0]


def test_raw_retention_does_not_guarantee_grouped_retention():
    old=feature([[(50.,5,.95),(100.,2,.6)]])
    new,p=expand(old,50.,100*2**(3/1200),.95,.8)
    a.validate_expansion(old,new,p,'a'*64)
    reference=np.array([100*2**(-49/1200)])
    flags=[]
    for f in (old,new):
        g=a.independent_groups(f)
        raw=a.acceptable_labels(f['candidate_hz'],f['candidate_mask'],reference,[True],[True])[1]
        grouped=a.acceptable_labels(g['candidate_hz'],g['candidate_mask'],reference,[True],[True])[1]
        flags.append((raw[0],grouped[0]))
    assert flags==[(True,True),(True,False)]


def test_union_schedule_keeps_control_uncovered_rows():
    old=feature([[(100.,5,.9)],[(100.,5,.9)]])
    new,p=expand(old,100.,300.)
    fs={'original6':old,'expanded7':new};groups={k:a.independent_groups(v) for k,v in fs.items()}
    t=a.union_training_targets(fs,groups,[100.,300.],[True,True],[True,True],expected_rows=2)
    assert t['schedule'].tolist()==[0,1]
    assert t['original6__covered'].tolist()==[True,False]
    assert t['expanded7__covered'].tolist()==[True,True]


def test_schedules_keep_exact_updates_and_without_replacement_exposure():
    for seed in a.SEEDS:
        natural=a.natural_order(seed,0);synthetic=a.synthetic_order(seed,0)
        assert len(natural)==132466 and len(np.unique(natural))==132466
        assert len(range(0,len(natural),4096))*12==396
        assert synthetic.shape==(33,512) and len(np.unique(synthetic))==16896
        assert np.array_equal(synthetic,a.augmentation.paired_schedule(seed,0))
    with pytest.raises(ValueError):a.natural_order(a.SEEDS[0],0,135169)


def test_acceptable_set_keeps_all_positives_and_valid_denominators():
    h=np.array([[100.,100.1,np.nan],[100.,300.,np.nan],[100.,300.,np.nan]])
    accepted,covered=a.acceptable_labels(h,np.isfinite(h),[100.,300.,300.],[True,False,True],[True,True,False])
    assert accepted.tolist()==[[True,True,False],[False,False,False],[False,False,False]]
    assert covered.tolist()==[True,False,False]


def test_single_case_seed_failure_is_not_averaged_away():
    cases={g:{m:{f'case{i}':1. for i in range(n)} for m in a.CANDIDATE} for g,n in a.GATE_COUNTS.items()}
    assert sum(a.GATE_COUNTS.values())==66
    cases['consumed_augmentation'][a.CANDIDATE[1]]['case0']=.949
    result=a.finish_case_gates(cases)
    assert result['challenge_pass'] and not result['consumed_augmentation_pass']
    del cases['challenge'][a.CANDIDATE[0]]['case0']
    with pytest.raises(ValueError):a.finish_case_gates(cases)


def test_independent_dynamic_equations_preserve_six_pool_output():
    from phenometrix_prosody.harmonic_selector_model import HarmonicSelector
    from phenometrix_prosody.selector_model import prepare_selector_batch
    f=feature([[(100.,5,.9),(200.,2,.7)],[]]);g=a.independent_groups(f)
    batch=prepare_selector_batch(f['features'],f['candidate_hz'],f['candidate_mask'],representation='shared-candidates')
    torch.manual_seed(20261022);model=HarmonicSelector('relative').eval()
    with torch.inference_mode():
        got=a.independent_logits(model.state_dict(),g);expected=model(batch)
    assert torch.equal(got,expected)
    new,p=expand(feature([[(100.,5,.9)]]),100.,300.)
    with torch.inference_mode():z=a.independent_logits(model.state_dict(),a.independent_groups(new))
    assert z.shape==(1,7) and torch.isfinite(z).all()


def test_phase_order_allows_only_declared_prerequisite_reference_access():
    t=lambda n:f'2026-10-23T00:00:{n:02d}+00:00'
    names=['sources_frozen_at','prerequisite_references_read_at','training_frozen_at',
        'checkpoints_frozen_at','predictions_frozen_at','final_scoring_reference_read_at','completed_at']
    m=dict(zip(names,map(t,[0,2,4,5,7,8,9])))
    m.update(development_references_read_for_prerequisites=True,development_references_read_for_final_scoring=True)
    initial=dict(status='sources_frozen',fits={},development_references_read_for_prerequisites=False)
    inputs=dict(status='input_caches_frozen',frozen_at=t(1),reference_content_read=False)
    prereq=dict(status='complete',frozen_at=t(3),pass_=True);prereq['pass']=prereq.pop('pass_')
    training=dict(status='training_frozen',fits={});checkpoint=dict(status='checkpoints_frozen')
    prediction=dict(status='predictions_frozen',fits={f'{arm}__seed{s}':{} for arm in a.ARMS for s in a.SEEDS},
        development_references_read_for_prerequisites=True,development_references_read_for_final_scoring=False)
    assert a.phase_order(m,initial,inputs,prereq,training,checkpoint,prediction)
    bad=deepcopy(m);bad['final_scoring_reference_read_at']=t(6)
    with pytest.raises(ValueError):a.phase_order(bad,initial,inputs,prereq,training,checkpoint,prediction)
    inputs['reference_content_read']=True
    with pytest.raises(ValueError):a.phase_order(m,initial,inputs,prereq,training,checkpoint,prediction)


def test_natural_gate_uses_equal_cells_and_fixed_both_baselines():
    cells={method:{speaker+'/'+condition:dict(joint_error=.10,reference_voiced_success_rate=.8)
        for speaker in a.base.SPEAKERS['development'] for condition in a.base.CONDITIONS} for method in a.METHODS}
    for method in a.CANDIDATE:
        for value in cells[method].values():value['joint_error']=.094
    assert a.natural_gate(cells)['pass']
    for value in cells['original_hybrid'].values():value['joint_error']=.098
    result=a.natural_gate(cells)
    assert not result['pass'] and not result['checks']['mean_improves_both_by_half_point']
    assert result['improvement_over_paired_original']==pytest.approx(.006)


def test_synthetic_exposure_counts_zero_rows_without_replacement():
    order=a.synthetic_order(a.SEEDS[0],0)
    covered=np.ones(18336,bool);covered[order[0,:7]]=False
    indices=np.repeat(np.arange(96),191);region=np.array(['before']*18336)
    report=a.synthetic_exposure(order,covered,indices,region)
    assert report['selected_rows']==report['unique_rows']==16896
    assert report['covered_rows']==16889 and report['uncovered_zero_rows']==7
    assert sum(r['selected_rows'] for r in report['per_case'])==16896
    assert report['per_region']['before']['covered_rows']==16889
    bad=order.copy();bad[0,0]=bad[0,1]
    with pytest.raises(ValueError):a.synthetic_exposure(bad,covered,indices,region)


def test_nearest_labels_uses_actual_emission_and_uncovered_minus_one():
    hz=np.array([[100.,100.5,np.nan],[100.,200.,np.nan]])
    acceptable,covered,nearest=a.nearest_labels(hz,np.isfinite(hz),np.array([100.4,300.]),np.array([True,True]))
    assert acceptable[0].tolist()==[True,True,False]
    assert covered.tolist()==[True,False] and nearest.tolist()==[1,-1]
