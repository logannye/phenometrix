import copy
import json
import math
from pathlib import Path
import pytest
from phenometrix_prosody import audit_hybrid_browser as a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


def fixture(tmp_path):
    public = tmp_path / 'public'; public.mkdir()
    for name in ('model.onnx', 'runtime.wasm', 'runtime.mjs'):
        (public / name).write_bytes(('fixture ' + name).encode())
    manifest = dict(status='complete', training_performed=True, scaling_fitted=True,
        config=dict(kind='framewise-mlp', feature_version='v2', feature_count=57, candidate_count=6, hidden_size=32, projection_size=32),
        training_metadata=dict(weights_sha256=a.CHECKPOINT, measurement_delay_ms=20),
        artifacts=dict(onnx=dict(sha256=a.sha(public / 'model.onnx'))), feature_names=[f'f{i}' for i in range(57)])
    write(public / 'model-manifest.json', manifest)
    config = dict(schemaVersion='phenometrix.research-voice-model-config.v1', qualification='unqualified-functional-fixture', estimator=a.CONTRACT)
    for role, filename in dict(manifest='model-manifest.json', model='model.onnx', wasm='runtime.wasm', module='runtime.mjs').items():
        config[role + 'Url'] = '/' + filename; config[role + 'Sha256'] = a.sha(public / filename)
    config_path = public / 'config.json'; write(config_path, config)
    identity, _ = a.bound_assets(config_path, a.sha(config_path), public)
    run = tmp_path / 'run'; run.mkdir()
    (tmp_path / 'source.ts').write_text('// fixed source\n')
    write(run / 'manifest.json', dict(status='complete', source_changed=False, source_sha256={'source.ts': a.sha(tmp_path / 'source.ts')}))
    frames = []
    for index, end in enumerate((2400, 2880)):
        ready = 119 + .5 * index; availability = 100 + 10 * index
        research = dict(**identity, schemaVersion='phenometrix.research-voice-prediction.v1', qualification='unqualified-functional-fixture',
            featureSchema='phenometrix.voice-pitch-features.v2', measurementDelayMs=20, referenceSample=end-960,
            supportStartSample=end-1920, supportEndSampleExclusive=end, lastRequiredInputSample=end-1,
            candidateHeadRole='diagnostic-only-unused-by-hybrid', candidateIndex=None, correctnessProbability=None,
            probabilitySemantics='uncalibrated-model-probabilities-not-correctness', voicingLogit=math.log(4), voicingProbability=.8,
            state='voiced', f0Hz=200., featureValues=[0.]*57, candidateHz=[200., None, None, None, None, None],
            candidateMask=[True, False, False, False, False, False], candidateLogits=[2., -1e9, -1e9, -1e9, -1e9, -1e9],
            candidateProbabilities=[1., 0., 0., 0., 0., 0.], rawYin=dict(implementationVersion=a.YIN, referenceLibrary='librosa@0.11.0',
                rawF0Hz=200., selectedPeriod=240, parabolicShift=0., thresholdTrough=True, serviceMs=.1),
            inferenceServiceMs=1., elapsedFromFrameStartMs=2.5+index, modelReadyAtMs=ready,
            rawSampleAvailabilityToModelReadyMs=ready-availability, sampleAvailabilityToModelReadyMs=ready-availability,
            targetReferenceToModelReadyMs=ready-availability+20)
        timing = dict(sampleRateHz=48000, supportStartSample=end-1920, supportEndSampleExclusive=end,
            referenceSample=end, lastRequiredInputSample=end-1, sampleAvailabilityMs=availability,
            renderFreshness=dict(status='certified', renderAgeUpperBoundMs=30.))
        frames.append(dict(captureEpoch=1, sequence=index+1, trackSegmentId='audio-1-1', sampleRateHz=48000,
            tMs=(end-1920)/48, absoluteSampleIndex=end-1920, timing=timing, research=research))
    frames[1].update(deliveryDisposition='discarded', discardReason='expired-lease')
    frames[1]['timing']['renderFreshness'] = dict(status='withheld', reason='expired-lease')
    block = dict(captureEpoch=1, blockSequence=3, trackSegmentId='audio-1-1', supportStartSample=1920,
        supportEndSampleExclusive=2880, analyzedFrameCount=2, disposition='processed', workerServiceMs=4.,
        workerReceivedAtMs=116., workerCompletedAtMs=120.)
    rows = [dict(kind='frame', value=f) for f in frames] + [dict(kind='block-timing', value=block)]
    summary = dict(frameCount=1, researchFrameCount=1, discardedResultFrameCount=1, discardedResearchFrameCount=1,
        researchVoicedFrameCount=1, researchCorrectKnownPitchFrameCount=1, blockCount=1, knownPitchHz=200,
        modelInferenceServiceMs=a.distribution([1.]), modelElapsedFromFrameStartMs=a.distribution([2.5]),
        sampleAvailabilityToModelReadyMs=a.distribution([19.]), targetReferenceToModelReadyMs=a.distribution([39.]),
        rawYinServiceMs=a.distribution([.1]))
    result = dict(status='complete', validationStatus='passed', errors=[], sampleRateHz=48000,
        workerStopAcknowledged=True, transportStop=dict(status='complete'), researchModel=dict(**identity, qualification='unqualified-functional-fixture', loadMs=4),
        diagnostics=dict(processedFrameCount=3, researchCancelledFrameCount=1), summary=summary,
        rawArtifact=dict(file='raw.jsonl', sha256=None, recordCount=3, batchCount=1))
    def save():
        write(run / 'raw.jsonl', dict(schemaVersion='phenometric.voice-replay-telemetry.v1', sequence=1, records=rows))
        result['rawArtifact']['sha256'] = a.sha(run / 'raw.jsonl'); write(run / 'browser-replay.json', result)
    save()
    return dict(run=run, config_path=config_path, expected_config_sha256=a.sha(config_path), public_root=public, repository=tmp_path), rows, result, identity, save


def test_complete_discarded_and_cancelled_accounting(tmp_path):
    args, rows, result, identity, save = fixture(tmp_path)
    report = a.audit(**args)
    assert report['status'] == 'passed' and report['counts']['preparedResearchFrames'] == 3
    assert report['counts']['completedResearchFrames'] == 2 and report['counts']['cancelledResearchFrames'] == 1
    assert report['state_counts'] == {'delivered:voiced': 1, 'discarded:voiced': 1}


@pytest.mark.parametrize('field,value', [('checkpointSha256', '0'*64), ('configurationSha256', None),
    ('estimatorContract', 'candidate-mlp'), ('candidateIndex', 0), ('correctnessProbability', .9),
    ('referenceSample', 2400), ('f0Hz', 100.)])
def test_identity_support_or_candidate_pitch_substitution_rejected(tmp_path, field, value):
    args, rows, result, identity, save = fixture(tmp_path)
    rows[0]['value']['research'][field] = value; save()
    with pytest.raises(ValueError): a.audit(**args)


def test_discarded_research_cannot_disappear(tmp_path):
    args, rows, result, identity, save = fixture(tmp_path)
    del rows[1]['value']['research']; save()
    with pytest.raises(ValueError, match='including on discarded'): a.audit(**args)


def test_exact_half_and_out_of_range_rule_ignores_candidates(tmp_path):
    args, rows, result, identity, save = fixture(tmp_path)
    frame = copy.deepcopy(rows[0]['value']); r = frame['research']
    r.update(voicingLogit=0., voicingProbability=.5)
    a.validate_research(frame, identity)
    r['rawYin'].update(selectedPeriod=68, parabolicShift=0., rawF0Hz=48000/68)
    r.update(state='unavailable', f0Hz=None)
    a.validate_research(frame, identity)
    r.update(voicingLogit=-1., voicingProbability=1/(1+math.exp(1)), state='unvoiced')
    a.validate_research(frame, identity)
    r['state'] = 'voiced'
    with pytest.raises(ValueError, match='frozen voicing'): a.validate_research(frame, identity)


def test_negative_clock_projection_is_preserved_unqualified(tmp_path):
    args, rows, result, identity, save = fixture(tmp_path)
    frame = copy.deepcopy(rows[0]['value']); r = frame['research']; frame['timing']['sampleAvailabilityMs'] = 121.
    r.update(rawSampleAvailabilityToModelReadyMs=-2., sampleAvailabilityToModelReadyMs=None, targetReferenceToModelReadyMs=None)
    a.validate_research(frame, identity)
    r['inferenceServiceMs'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'): a.validate_research(frame, identity)


@pytest.mark.parametrize('change', ['hidden-frame', 'block-count', 'short-full-service', 'source', 'config', 'asset', 'raw'])
def test_hashes_and_full_work_accounting_fail_closed(tmp_path, change):
    args, rows, result, identity, save = fixture(tmp_path)
    if change == 'hidden-frame': result['diagnostics']['processedFrameCount'] += 1
    if change == 'block-count': rows[2]['value']['analyzedFrameCount'] -= 1
    if change == 'short-full-service': rows[2]['value']['workerServiceMs'] = 1.
    save()
    if change == 'source': (tmp_path / 'source.ts').write_text('// changed\n')
    if change == 'config': args['config_path'].write_text(args['config_path'].read_text()+' ')
    if change == 'asset': (args['public_root'] / 'model.onnx').write_bytes(b'changed')
    if change == 'raw': (args['run'] / 'raw.jsonl').write_text('{}\n')
    with pytest.raises(ValueError): a.audit(**args)


def test_unbound_functional_source_requires_explicit_limited_scope(tmp_path):
    args, rows, result, identity, save = fixture(tmp_path)
    (args['run'] / 'manifest.json').unlink()
    with pytest.raises(ValueError, match='functional-only opt-in'): a.audit(**args)
    report = a.audit(**args, allow_unbound_functional_source=True)
    assert report['source_bindings_checked'] == 0 and report['source_manifest_sha256'] is None
    assert report['measured_source_binding'].startswith('not established')


def test_old_yin_identity_cannot_qualify_repaired_series(tmp_path):
    args, rows, result, identity, save = fixture(tmp_path)
    rows[0]['value']['research']['rawYin']['implementationVersion'] = 'phenometrix.raw-yin-librosa011-semantics.ts.v1'
    save()
    with pytest.raises(ValueError, match='implementation evidence'): a.audit(**args)
