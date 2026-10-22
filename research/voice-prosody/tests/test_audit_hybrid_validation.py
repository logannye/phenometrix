import copy
import math
from pathlib import Path
import pytest
from phenometrix_prosody import audit_hybrid_validation as a


def p(index, state, hz=None):
    return dict(sample=768+480*index, state=state, f0_hz=hz, correctness_probability=None)


def test_independent_counts_preserve_failures_boundaries_and_out_of_range_reference():
    rows = [(0.,0.,0.,0.)]*16
    for index,hz in ((5,100.),(7,200.),(8,200.),(10,900.),(12,200.),(15,100.)):
        rows[index] = (hz,1.,.1,.9)
    rows[11] = (100.,.5,.1,.9); rows[13] = (float('nan'),1.,.1,.9)
    predictions = [p(5,'voiced',100.),p(6,'voiced',200.),p(7,'unvoiced'),p(10,'voiced',450.),
        p(12,'voiced',205.),p(14,'unvoiced'),p(15,'voiced',110.),p(30,'unvoiced')]
    metric, errors, outside = a.independent_counts(rows,10000,predictions)
    expected = dict(eligible_valid_frames=9, reference_voiced_frames=6, reference_unvoiced_frames=3,
        excluded_invalid_reference=2, excluded_boundary=5, reference_outside_supported_bounds=1,
        correct_frames=3, voicing_false_positive_count=1, voicing_false_negative_count=1,
        unavailable_reference_voiced_count=1, voicing_false_negative_or_unavailable_count=2,
        reference_voiced_success_count=2, unavailable_frames=2, jointly_voiced_frames=4, octave_error_count=1)
    assert {k:metric[k] for k in a.COUNTS} == expected
    assert metric['joint_error'] == pytest.approx(2/3) and metric['measurement_coverage'] == pytest.approx(7/9)
    assert metric['reference_voiced_success_rate'] == pytest.approx(1/3)
    assert outside == 1 and len(errors) == 4 and metric['calibration_n'] == 0


def test_invalid_predictions_fail_even_outside_reference_extent():
    with pytest.raises(ValueError,match='state/frequency'):
        a.independent_counts([(0.,0.,0.,0.)]*12,10000,[p(30,'unvoiced',0.)])
    with pytest.raises(ValueError,match='Duplicate'):
        a.independent_counts([],10000,[p(6,'unvoiced'),p(6,'unvoiced')])
    row=p(6,'voiced',200.);row['correctness_probability']=.9
    with pytest.raises(ValueError,match='correctness'):
        a.independent_counts([],10000,[row])


def test_aggregate_quantiles_pool_errors_and_macro_weights_cells_equally():
    c = dict.fromkeys(a.COUNTS,0); c.update(eligible_valid_frames=100,correct_frames=90,reference_voiced_frames=100)
    d = dict(c,eligible_valid_frames=1,correct_frames=0,reference_voiced_frames=1)
    pooled=a.aggregate([(c,[1.,2.,100.]),(d,[3.])])
    assert pooled['median_jointly_voiced_cents'] == 2.
    macro=a.macro([a.summarize(c,[]),a.summarize(d,[])])
    assert macro['joint_error'] == pytest.approx(.55)
    assert pooled['joint_error'] == pytest.approx(11/101)


def bootstrap_records():
    rows=[]
    for speaker in a.SPEAKERS:
        for condition in a.CONDITIONS:
            for prompt in range(40):
                rows.append(dict(speaker=speaker,pair=int(speaker[1:]),condition=condition,prompt=f'p{prompt:02}',
                    metrics={m:dict(eligible_valid_frames=100,correct_frames=90 if m=='hybrid_original' else 80) for m in a.METHODS}))
    return rows


def test_prompt_bootstrap_fixed_paired_advantage_and_repeatability():
    rows=bootstrap_records(); result=a.independent_bootstrap(rows)
    assert result == a.independent_bootstrap(rows) and result['draws']==2000 and result['seed']==20261022
    for interval in result['joint_error_advantage_intervals'].values():
        assert interval['lower']==pytest.approx(.1) and interval['upper']==pytest.approx(.1)
    rows[0]['metrics']['dsp']['eligible_valid_frames']=0
    assert a.independent_bootstrap(rows)['status']=='unavailable'


def test_bootstrap_pairing_cancels_sex_specific_prompt_effects():
    rows=bootstrap_records()
    for row in rows:
        effect=20 if int(row['prompt'][1:])%2 else -20
        if row['speaker'].startswith('M'): effect=-effect
        for method in a.METHODS[:-1]:row['metrics'][method]['correct_frames']+=effect
    result=a.independent_bootstrap(rows)
    # Shared prompt draws across sexes must cancel the prompt effect in every draw.
    for interval in result['joint_error_advantage_intervals'].values():
        assert interval['lower']==pytest.approx(.1,abs=1e-12) and interval['upper']==pytest.approx(.1,abs=1e-12)


def test_gate_requires_each_comparator_and_complete_cell_denominators():
    overall={m:dict(joint_error=.12,reference_voiced_success_rate=.8,defined_cells=dict(joint_error=16,reference_voiced_success_rate=16)) for m in a.METHODS}
    overall['hybrid_original']['joint_error']=.10
    clean={m:dict(joint_error=.1) for m in a.METHODS}
    assert a.gate(overall,clean)['pass'] is True
    overall['yin']['joint_error']=.105
    assert a.gate(overall,clean)['pass'] is False
    overall['yin']['defined_cells']['reference_voiced_success_rate']=15
    assert a.gate(overall,clean)['status']=='incomplete_required_cell_denominators'


def test_reference_audit_requires_explicit_opt_in_before_any_file_read(tmp_path):
    with pytest.raises(ValueError,match='opt-in'):
        a.audit(tmp_path,tmp_path/'absent-freeze',tmp_path,'0'*64)


def test_chronology_requires_explicit_timezone_and_recursive_comparison():
    assert a.timestamp('2026-10-22T20:25:00.422426Z') == a.timestamp('2026-10-22T20:25:00.422426+00:00')
    with pytest.raises(ValueError,match='timezone'):a.timestamp('2026-10-22T20:25:00')
    a.compare({'count':2,'value':.1},{'count':2,'value':.10000000000001},'test')
    with pytest.raises(ValueError,match='numeric mismatch'):a.compare(.1,.2,'test')


def test_serialized_command_start_must_follow_complete_prediction_freeze(tmp_path):
    import json
    runner=tmp_path/'.local/run-hybrid-natural-stage.py';runner.parent.mkdir();runner.write_text('# fixture\n')
    research=tmp_path/'research/voice-prosody';research.mkdir(parents=True)
    execution=research/'runs/execution';execution.mkdir(parents=True)
    freeze=research/'runs/freeze/freeze.json';run=research/'runs/validation';digest='a'*64
    def time(n):return f'2026-10-22T20:25:{n:02}+00:00'
    for phase,start,end in [('predict',0,10),('acquire-ref',11,20),('score',21,30)]:
        log=execution/(phase+'.log');log.write_text('completed\n')
        command=[str(research/'.venv/bin/python'),'-m','phenometrix_prosody.hybrid_validation',phase,
                 '--freeze','runs/freeze/freeze.json','--output','runs/validation']
        row=dict(schema='phenometrix-local-serialized-natural-phase.v1',status='complete',exit_code=0,phase=phase,
                 compute='local',freeze_sha256=digest,runner_sha256=a.sha(runner),log_sha256=a.sha(log),
                 command=command,cwd=str(research),started_at=time(start),finished_at=time(end))
        (execution/(phase+'.json')).write_text(json.dumps(row))
    args=(execution,freeze,run,digest,runner,tuple(map(a.timestamp,[time(1),time(9),time(29)])),list(map(a.timestamp,[time(12),time(19)])))
    assert set(a.phase_chronology(*args))=={'predict','acquire-ref','score'}
    path=execution/'acquire-ref.json';row=json.loads(path.read_text());row['started_at']=time(8);path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='REF command launch'):a.phase_chronology(*args)
