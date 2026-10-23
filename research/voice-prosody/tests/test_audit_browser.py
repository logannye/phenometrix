"""Offline proof checks reject altered or missing delivered-frame evidence."""
import json
import pytest
from phenometrix_prosody.audit_browser import audit, digest

VERSION='phenometric.voice-render-freshness.v1'


def write_run(tmp_path, mutate=lambda frame: None):
    lease=dict(schemaVersion=VERSION,leaseId=1,captureEpoch=2,clockMappingId=3,
        mainTimeOriginMs=10000.,issuedAtMainMs=100.,expiresAtMainMs=200.)
    proof=dict(schemaVersion=VERSION,status='certified',basis='main-lease-before-block-first-input',
        scope='frame-last-required-rendered-input',leaseId=1,issuedAtMainMs=100.,
        resultReceivedAtMainMs=150.,maximumAgeMs=100,renderAgeUpperBoundMs=50.,timestampPrecisionMs=None)
    frame=dict(captureEpoch=2,trackSegmentId='a',periodic=True,f0Hz=200.,renderLease=lease,
        timing=dict(clockMappingId=3,status='valid',workerServiceMs=1.,mainResultArrivalMs=150.,renderFreshness=proof))
    mutate(frame)
    raw=tmp_path/'raw.jsonl';raw.write_text(json.dumps({'sequence':1,'records':[{'kind':'frame','value':frame}]})+'\n')
    result=dict(status='complete',validationStatus='passed',sampleRateHz=48000,
        rawArtifact=dict(file=raw.name,sha256=digest(raw),recordCount=1,batchCount=1),
        summary=dict(certifiedFrameCount=1,frameCount=1,renderAgeUpperBoundMs=dict(n=1,p50=50.,p95=50.,p99=50.,maximum=50.)))
    (tmp_path/'browser-replay.json').write_text(json.dumps(result))
    return tmp_path


def test_independently_reconstructs_same_main_age(tmp_path):
    result=audit(write_run(tmp_path))
    assert result['status']=='passed'
    assert result['recomputed_summary']['certifiedFrameCount']==1


@pytest.mark.parametrize('mutation',[
    lambda f:f['timing'].pop('renderFreshness'),
    lambda f:f['renderLease'].update(leaseId=2),
    lambda f:f['renderLease'].update(captureEpoch=1),
    lambda f:f['renderLease'].update(clockMappingId=2),
    lambda f:f['renderLease'].update(expiresAtMainMs=250.),
    lambda f:f['timing']['renderFreshness'].update(renderAgeUpperBoundMs=25.),
    lambda f:f['timing']['renderFreshness'].update(resultReceivedAtMainMs=130.),
    lambda f:f['timing']['renderFreshness'].update(timestampPrecisionMs=0.),
    lambda f:f['timing']['renderFreshness'].update(scope='microphone-acquisition'),
    lambda f:f['timing'].update(mainResultArrivalMs=201.),
    lambda f:f['timing'].update(mainResultArrivalMs=99.),
])
def test_rejects_forged_expired_or_missing_certificate(tmp_path, mutation):
    with pytest.raises(ValueError):audit(write_run(tmp_path,mutation))


def test_raw_hash_is_checked_before_evidence(tmp_path):
    run=write_run(tmp_path);(run/'raw.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='hash'):audit(run)


def test_unreported_host_drop_fails_summary_audit(tmp_path):
    def discard(frame):
        frame.update(deliveryDisposition='discarded',discardReason='expired-lease')
        frame['timing']['renderFreshness']=dict(schemaVersion=VERSION,status='withheld',reason='expired-lease')
    run=write_run(tmp_path,discard)
    path=run/'browser-replay.json';result=json.loads(path.read_text())
    result['summary']=dict(frameCount=0,certifiedFrameCount=0,hostWithheldReasons={})
    path.write_text(json.dumps(result))
    evidence=audit(run)
    assert evidence['status']=='failed'
    assert evidence['mismatches']['hostWithheldReasons']['recomputed']=={'expired-lease':1}
    result['summary']['hostWithheldReasons']={'expired-lease':1}
    path.write_text(json.dumps(result))
    assert audit(run)['status']=='passed'
