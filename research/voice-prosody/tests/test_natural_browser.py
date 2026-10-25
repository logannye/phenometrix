import copy
import json
from pathlib import Path
import numpy as np
import pytest
from phenometrix_prosody import audit_natural_browser as a
from phenometrix_prosody.prepare_natural_browser import ptdb_prefix, first_sources, SPEAKERS
from test_audit_hybrid_browser import fixture as hybrid_fixture


def fixture(tmp_path):
    _, rows, result, identity, _ = hybrid_fixture(tmp_path)
    pcm=np.zeros(2977,np.float32)
    record=dict(id="synthetic",sampleCount=len(pcm),pcmSha256="a"*64,adapter="synthetic-index-v1")
    origin=dict(scheduledStartAudioFrame=1920,observedStartAudioFrame=1920,sourceOffset=0,
        captureEndSampleExclusive=len(pcm),sourceSampleCount=len(pcm),firstSamples=pcm[:16].tolist(),
        discardedPartialSampleCount=97,droppedSampleCount=2017,droppedBlockCount=2,sequence=3,
        timeline=dict(schemaVersion="phenometric.voice-natural-timeline.v2",status="complete",error=None,events=[],capacity=64,overflow=0))
    result["naturalSource"]=dict(**record,scheduledStartAudioFrame=1920,sourceOffset=0,origin=origin,sourceEnded=True,
        completion=dict(status="complete",missing=[]),
        credits=dict(type="research-natural-credits-drained",captureEpoch=1,scheduledStartAudioFrame=1920,
                     sourceSampleCount=len(pcm),sequence=3,inFlightBlockCount=0,pendingSampleCount=0),
        drainTimeoutMs=5000,mainDrainTimeoutMs=6000,finalPartialPolicy="discard-retain-count-no-padding",referenceOffsetSamples=0,
        drain=dict(status="drained",pumpIdle=True,events=[],eventCapacity=20000,eventOverflow=0,diagnostics=dict(inFlightResultBatchCount=0)))
    result["transportStop"]=dict(status="complete",diagnostics=dict(sampleClockEndExclusive=len(pcm),
        inFlightBlockCountAtDispose=0,discardedPartialSampleCount=97,droppedSampleCount=2017,droppedBlockCount=2,sequence=3))
    result["diagnostics"].update(hostDiscardedFrameCount=1,maximumInFlightBlocks=4,maximumInFlightResultBatches=4,researchMaximumQueuedBlocks=3)
    result["summary"].update(rawTelemetry=dict(streamedRecordCount=3,status="complete"),completeProcessedBlockServiceMs=a.distribution([4.]))
    batches=[dict(schemaVersion="phenometric.voice-replay-telemetry.v1",sequence=1,records=rows)]
    return result,record,pcm,batches,identity


def test_exact_prefix_changes_coordinates_not_pcm_or_provider_values():
    pcm=np.arange(4800,dtype=np.float32)/4800
    prefixed=ptdb_prefix(pcm)
    assert len(prefixed)==len(pcm)+192
    assert np.array_equal(prefixed[:192],np.zeros(192)) and np.array_equal(prefixed[192:],pcm)
    native_targets=np.arange(960,4800,480)
    assert np.array_equal(native_targets-192,768+480*np.arange(len(native_targets)))
    assert next(t for t in native_targets-192 if t>=3072)==3168
    with pytest.raises(ValueError):ptdb_prefix(np.array([np.nan]))


def test_first_source_selection_uses_declared_speakers_and_metadata_order():
    rows=[dict(role="mic",speaker=s,split="development",primary_eligible=True,recording_id=s+"_"+i)
          for s in SPEAKERS for i in ("si9","si1")]
    selected=first_sources(dict(files=rows))
    assert [r["recording_id"] for r in selected]==[s+"_si1" for s in SPEAKERS]
    with pytest.raises(ValueError):first_sources(dict(files=rows[2:]))


def test_missing_and_withheld_stay_in_full_source_denominator(tmp_path):
    args=fixture(tmp_path)
    frames,blocks,ledger,categories=a.audit_records(*args)
    assert len(frames)==2 and len(blocks)==1 and len(ledger)==3
    assert categories=={"transport-loss":1,"delivered":1,"withheld":1}
    assert [row["originalTargetSample"] for row in ledger]==[960,1440,1920]


@pytest.mark.parametrize("corruption",["model-hash","sample-origin","result-sequence","result-sequence-gap","nonfinite-prediction","status-receipt","reference-phase","hidden-withheld","terminal-credit","missing-worklet-credit","timeline-rejected","timeline-overflow","timeline-loss-accounting"])
def test_corrupted_evidence_is_rejected(tmp_path,corruption):
    result,record,pcm,batches,identity=fixture(tmp_path)
    frame=batches[0]["records"][0]["value"]
    if corruption=="timeline-rejected":result["naturalSource"]["origin"]["timeline"]["status"]="failed"
    if corruption=="timeline-overflow":result["naturalSource"]["origin"]["timeline"]["overflow"]=1
    if corruption=="timeline-loss-accounting":result["naturalSource"]["origin"]["droppedSampleCount"]+=128
    if corruption=="model-hash":frame["research"]["modelSha256"]="0"*64
    if corruption=="sample-origin":result["naturalSource"]["origin"]["observedStartAudioFrame"]+=128
    if corruption=="result-sequence":batches[0]["records"][1]["value"]["sequence"]=frame["sequence"]
    if corruption=="result-sequence-gap":batches[0]["records"][1]["value"]["sequence"]=3
    if corruption=="nonfinite-prediction":frame["research"]["voicingLogit"]=float("nan")
    if corruption=="status-receipt":result["naturalSource"]["drain"]["status"]="timeout"
    if corruption=="reference-phase":batches[0]["records"][2]["value"]["supportStartSample"]+=128
    if corruption=="hidden-withheld":result["summary"]["discardedResultFrameCount"]=0
    if corruption=="missing-worklet-credit":result["naturalSource"]["credits"]=None
    if corruption=="terminal-credit":result["transportStop"]["diagnostics"]["inFlightBlockCountAtDispose"]=1
    with pytest.raises(ValueError):a.audit_records(result,record,pcm,batches,identity)


def test_reference_code_corruption_is_rejected_by_authenticated_bytes(tmp_path):
    path=tmp_path/"ref.f0";path.write_text("100 1 .1 .9\n")
    record=dict(reference=dict(path="ref.f0",sha256=a.sha(path)),originalSampleCount=4800)
    path.write_text("100 2 .1 .9\n")
    with pytest.raises(ValueError,match="Reference content identity"):
        a.score_reference(record,[],[],tmp_path)


def test_render_age_certificate_has_independent_same_clock_arithmetic():
    mapping=dict(id=1,mainTimeOriginMs=100000)
    lease=dict(schemaVersion="phenometric.voice-render-freshness.v1",leaseId=3,issuedAtMainMs=20,expiresAtMainMs=120,
        captureEpoch=1,clockMappingId=1,mainTimeOriginMs=100000)
    certificate=dict(schemaVersion="phenometric.voice-render-freshness.v1",status="certified",leaseId=3,issuedAtMainMs=20,
        resultReceivedAtMainMs=70,maximumAgeMs=100,timestampPrecisionMs=None,renderAgeUpperBoundMs=50,
        basis="main-lease-before-block-first-input",scope="frame-last-required-rendered-input")
    frame=dict(captureEpoch=1,renderLease=lease,timing=dict(mainResultArrivalMs=70,clockMappingId=1,renderFreshness=certificate))
    assert a.validate_certificate(frame,mapping)==50
    certificate["renderAgeUpperBoundMs"]=49
    with pytest.raises(ValueError,match="same-clock"):a.validate_certificate(frame,mapping)


def test_case_inventory_and_full_source_fields_are_frozen_before_content_access(tmp_path):
    manifest=dict(playlist=["H","H"],indexFixture=dict(id="index",pcmPath="index.f32le",pcmSha256="a"*64),
        records=[dict(id="F01_si1__clean",pcmPath="source.f32le",pcmSha256="b"*64,
                      reference=dict(path="source.f0",sha256="c"*64))])
    expected=a.expected_cases(manifest)
    assert [r["caseId"] for r in expected]==["index-origin","0-H-F01_si1__clean","1-H-F01_si1__clean","injected-stall"]
    source=expected[1]
    case=dict(id=source["caseId"],role=source["role"],resultPath=source["caseId"]+"/result.json")
    result=dict(case=copy.deepcopy(source),inputManifestSha256="d"*64)
    a.validate_case_binding(case,result,source,"d"*64)
    for field,value in [("pcmPath","replacement.f32le"),("pcmSha256","e"*64),("id","M03_si9__clean"),
                        ("reference",dict(path="replacement.f0",sha256="f"*64)),("model","R"),("role","injected-stall")]:
        changed=copy.deepcopy(result);changed["case"][field]=value
        path=tmp_path/"result.json";path.write_text(json.dumps(changed))
        # A self-consistent replacement result hash cannot authorize a new workload.
        changed_case=dict(case,resultSha256=a.sha(path))
        assert changed_case["resultSha256"]==a.sha(path)
        with pytest.raises(ValueError,match="Full case record"):
            a.validate_case_binding(changed_case,changed,source,"d"*64)
    with pytest.raises(ValueError,match="input manifest"):
        a.validate_case_binding(case,dict(result,inputManifestSha256="0"*64),source,"d"*64)
    with pytest.raises(ValueError,match="case order"):
        a.validate_case_binding(case,result,expected[2],"d"*64)


def test_missing_provider_tail_retains_delivery_inventory_and_eligible_reference_denominator(tmp_path):
    path=tmp_path/"ref.f0"
    np.savetxt(path,np.tile([0.,0.,0.,1.],(8,1)))
    record=dict(reference=dict(path="ref.f0",sha256=a.sha(path)),originalSampleCount=7200)
    targets=[768+480*i for i in range(12)]
    frames=[dict(research=dict(referenceSample=t+192,state="unvoiced",f0Hz=None)) for t in targets]
    frames[-1]["deliveryDisposition"]="discarded"
    ledger=[dict(originalTargetSample=t,delivery="withheld" if i==11 else "delivered") for i,t in enumerate(targets)]
    quality=a.score_reference(record,frames,ledger,tmp_path)
    assert quality["outputsWithoutProviderReference"]==4
    assert quality["deliveredOutputsWithoutProviderReference"]==3
    assert quality["referenceStatuses"]["missing"]==4
    assert len(ledger)==12 and sum(r["delivery"]=="delivered" for r in ledger)==11
    assert quality["deliveredUnconditional"]["eligible_valid_frames"]==3
    assert quality["deliveredUnconditional"]["joint_error"]==0
    assert quality["deliveredUsableCorrectCoverage"]==1
