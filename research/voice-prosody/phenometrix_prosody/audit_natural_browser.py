"""Independent finite-source alignment, acoustic/model parity and delivery audit.

Acoustic support/history reconstruction runs outside transport using the shared
pure TypeScript feature equations. NumPy reruns the MLP and librosa independently
reruns YIN from authenticated PCM. This does not independently rederive the DSP
feature equations or establish physical input/display latency.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np
from .audit_hybrid_browser import bound_assets, validate_research, sha, require, distribution, within, REPO
from .model import ModelConfig, numpy_step
from .metrics import Prediction, score
from .pilot import references_from_array
from .prepare_natural_browser import first_sources
from .audit_clock import analyze_probes


def origin_and_targets(result, record, pcm):
    source = result.get("naturalSource") or {}
    receipt = source.get("origin") or {}
    count = record["sampleCount"]
    source_start=source.get("scheduledStartAudioFrame")
    require(type(source_start) is int and source_start % 1920 == 0 and source_start > 0, "Unbound scheduled source origin")
    source_end=source_start+count
    timeline=receipt.get("timeline") or {}
    require(timeline.get("schemaVersion")=="phenometric.voice-natural-timeline.v2" and timeline.get("status")=="complete"
            and timeline.get("error") is None and timeline.get("capacity")==64 and timeline.get("overflow")==0
            and isinstance(timeline.get("events"),list) and len(timeline["events"])<=64, "Source timeline receipt incomplete or rejected")
    gap_samples=0
    for event in timeline["events"]:
        kind=event.get("kind")
        require(kind in ("forward-gap","absent-input","grid-restart"), "Rejected source timeline event")
        if kind=="grid-restart":
            require(source_start<=event["startAudioFrame"]<event["endAudioFrameExclusive"]<=source_end
                    and event["targetAudioFrame"]==min(source_end,source_start+math.ceil((event["startAudioFrame"]-source_start)/960)*960),
                    "Grid restart changed the original source grid")
            skipped=event["endAudioFrameExclusive"]-event["startAudioFrame"]
            require(0<skipped<960 and skipped==event["skippedSampleCount"]
                    and event["endAudioFrameExclusive"]<=event["targetAudioFrame"], "Invalid retained grid restart")
            gap_samples+=skipped
        else:
            require(source_start<=event["expectedAudioFrame"]<=event["observedStartAudioFrame"]<=event["observedEndAudioFrameExclusive"]<=source_end,
                    "Source gap outside exact finite coordinates")
            last=event["observedEndAudioFrameExclusive"] if kind=="absent-input" else event["observedStartAudioFrame"]
            missing=last-event["expectedAudioFrame"]
            require(missing>=0 and missing==event["missingSampleCount"] and 0<=event["discardedPendingSampleCount"]<960,
                    "Invalid retained source gap")
            gap_samples+=missing+event["discardedPendingSampleCount"]
    require(gap_samples+receipt.get("discardedPartialSampleCount",0)+receipt.get("droppedBlockCount",0)*960==receipt.get("droppedSampleCount"),
            "Timeline and terminal sample losses do not reconcile")
    require(source.get("pcmSha256") == record["pcmSha256"] and source.get("sampleCount") == count
            and source.get("id") == record["id"] and source.get("adapter") == record["adapter"], "Source identity changed")
    start = source.get("scheduledStartAudioFrame")
    require(type(start) is int and start % 1920 == 0 and start > 0, "Unbound scheduled source origin")
    require(receipt.get("scheduledStartAudioFrame") == receipt.get("observedStartAudioFrame") == start
            and receipt.get("sourceOffset") == source.get("sourceOffset") == 0
            and receipt.get("captureEndSampleExclusive") == receipt.get("sourceSampleCount") == count,
            "Source-zero receipt mismatch")
    require(receipt.get("firstSamples") == pcm[:16].tolist(), "Observed source samples differ")
    require(source.get("sourceEnded") is True and source.get("drainTimeoutMs") == 5000
            and source.get("mainDrainTimeoutMs") == 6000 and source.get("finalPartialPolicy") == "discard-retain-count-no-padding",
            "Finite source or drain contract changed")
    drain = source.get("drain") or {}
    require(drain.get("status") == "drained" and drain.get("pumpIdle") is True
            and drain.get("diagnostics", {}).get("inFlightResultBatchCount") == 0,
            "Missing successful inference/result acknowledgement completion fence")
    credits=source.get("credits") or {}
    require(source.get("completion")==dict(status="complete",missing=[])
            and credits.get("type")=="research-natural-credits-drained" and credits.get("captureEpoch")==1
            and credits.get("scheduledStartAudioFrame")==start and credits.get("sourceSampleCount")==count
            and credits.get("sequence")==receipt.get("sequence") and credits.get("inFlightBlockCount")==0
            and credits.get("pendingSampleCount")==0, "Missing observed worklet credit/origin completion fence")
    require(result.get("workerStopAcknowledged") is True and result.get("transportStop", {}).get("status") == "complete",
            "Missing terminal transport acknowledgement")
    terminal = result["transportStop"]["diagnostics"]
    require(terminal["sampleClockEndExclusive"] == count and terminal["inFlightBlockCountAtDispose"] == 0
            and terminal["discardedPartialSampleCount"] == receipt["discardedPartialSampleCount"]
            and terminal["droppedSampleCount"] == receipt["droppedSampleCount"]
            and terminal["droppedBlockCount"] == receipt["droppedBlockCount"], "Terminal sample/credit accounting mismatch")
    require(receipt["sequence"]*960+gap_samples+receipt["discardedPartialSampleCount"]==count,
            "Finite source support inventory does not reconcile")
    # The prefix is fixed provider timing, never a fitted delay or output rounding.
    offset = 192 if record["adapter"] == "ptdb-prefix-192-v1" else 0
    require(source.get("referenceOffsetSamples") == offset, "Reference-origin adapter changed")
    if offset:
        require(np.array_equal(pcm[:192], np.zeros(192, np.float32))
                and record["originalSampleCount"] + 192 == count, "Wrong PTDB prefix")
    return list(range(1920, count + 1, 480)), offset


def audit_records(result, record, pcm, batches, identity):
    require(result.get("status") in ("complete", "failed") and isinstance(result.get("errors"), list)
            and (result["status"] != "complete" or not result["errors"]), "Invalid execution status receipt")
    intended, offset = origin_and_targets(result, record, pcm)
    expected = set(intended)
    frames, blocks, sequences, targets = [], [], set(), set()
    raw_count = 0
    events=result["naturalSource"]["drain"].get("events", [])
    require(result["naturalSource"]["drain"].get("eventCapacity") == 20000
            and result["naturalSource"]["drain"].get("eventOverflow") == 0 and len(events) <= 20000, "Finite event ledger incomplete")
    require(all(row.get("kind") in ("block-loss", "reset") for row in events), "Unknown source event")
    losses=[row for row in events if row["kind"] == "block-loss"]
    resets=[row for row in events if row["kind"] == "reset"]
    for index, batch in enumerate(batches, 1):
        require(batch.get("schemaVersion") == "phenometric.voice-replay-telemetry.v1" and batch.get("sequence") == index
                and 0 < len(batch["records"]) <= 100, "Raw telemetry order/capacity changed")
        for item in batch["records"]:
            raw_count += 1
            row = item["value"]
            if item["kind"] == "clock-probe":
                continue
            if item["kind"] == "block-timing":
                require(row["blockSequence"] not in {b["blockSequence"] for b in blocks}, "Duplicate block sequence")
                require(row["supportEndSampleExclusive"] - row["supportStartSample"] == 960
                        and row["workerServiceMs"] >= 0 and math.isfinite(row["workerServiceMs"]), "Invalid block support/service")
                if row["disposition"] == "processed":
                    require(row["supportStartSample"] % 480 == 0, "Restarted segment has invalid reference phase")
                blocks.append(row)
                continue
            require(item["kind"] == "frame", "Unknown raw record status")
            research = validate_research(row, identity)
            require(row["sequence"] not in sequences, "Duplicate result sequence")
            sequences.add(row["sequence"])
            end = research["supportEndSampleExclusive"]
            require(end in expected and end not in targets and research["referenceSample"] - offset == end - 960 - offset,
                    "Prediction is off-grid, repeated or outside finite source")
            targets.add(end)
            disposition = row.get("deliveryDisposition", "delivered")
            require(disposition in ("delivered", "discarded"), "Unknown result delivery status")
            if disposition == "discarded":
                require(isinstance(row.get("discardReason"), str) and bool(row["discardReason"]), "Missing withheld reason")
            frames.append(row)
    frames.sort(key=lambda row:row["research"]["supportEndSampleExclusive"])
    summary, diagnostic = result["summary"], result["diagnostics"]
    delivered = sum(f.get("deliveryDisposition") != "discarded" for f in frames)
    withheld = len(frames) - delivered
    require(delivered == summary["frameCount"] == summary["researchFrameCount"]
            and withheld == summary["discardedResultFrameCount"] == summary["discardedResearchFrameCount"]
            and withheld == diagnostic["hostDiscardedFrameCount"], "Delivered/withheld accounting mismatch")
    require(all(type(f["sequence"]) is int and 1<=f["sequence"]<=diagnostic["processedFrameCount"] for f in frames)
            and all(a["sequence"]<b["sequence"] for a,b in zip(frames,frames[1:])), "Result sequence outside prepared frame inventory")
    for block in blocks:
        completed=[f for f in frames if block["supportStartSample"]<f["research"]["supportEndSampleExclusive"]<=block["supportEndSampleExclusive"]]
        require(len(completed)==block["analyzedFrameCount"], "Per-block completed prediction inventory changed")
        require(all(b["sequence"]==a["sequence"]+1 for a,b in zip(completed,completed[1:])), "Result sequence gap inside completed block")
    require(len(frames) + diagnostic.get("researchCancelledFrameCount", 0) == diagnostic["processedFrameCount"],
            "Completed/canceled prediction accounting mismatch")
    require(len(blocks) == summary["blockCount"] and sum(b["analyzedFrameCount"] for b in blocks) == len(frames),
            "Processed block/prediction accounting mismatch")
    require(raw_count == summary["rawTelemetry"]["streamedRecordCount"] and summary["rawTelemetry"]["status"] == "complete",
            "Raw telemetry incomplete")
    require(max(diagnostic["maximumInFlightBlocks"], diagnostic["maximumInFlightResultBatches"]) <= 4
            and diagnostic.get("researchMaximumQueuedBlocks", 0) <= 3, "Bounded transport capacity exceeded")
    by_end = {f["research"]["supportEndSampleExclusive"]: f for f in frames}
    categories = Counter()
    ledger = []
    terminal = result["transportStop"]["diagnostics"]
    for end in intended:
        row = by_end.get(end)
        if row:
            category = "withheld" if row.get("deliveryDisposition") == "discarded" else "delivered"
        elif end > record["sampleCount"] - terminal["discardedPartialSampleCount"]:
            category = "terminal-partial-support"
        elif any(b["supportStartSample"] < end <= b["supportEndSampleExclusive"] and b["disposition"] != "processed" for b in blocks):
            category = "worker-withheld-support"
        elif any(b["supportStartSample"] < end <= b["supportEndSampleExclusive"] for b in losses):
            loss=next(b for b in losses if b["supportStartSample"] < end <= b["supportEndSampleExclusive"])
            category = loss["reason"]
        elif any(b["supportStartSample"] < end <= b["supportEndSampleExclusive"] for b in blocks):
            category = "reset-warmup"
        else:
            category = "transport-loss"
        categories[category] += 1
        ledger.append(dict(supportEndSampleExclusive=end, originalTargetSample=end-960-offset, delivery=category))
    require(sum(categories.values()) == len(intended), "Unconditional source-grid denominator mismatch")
    known_sequences=[b["blockSequence"] for b in blocks+losses]
    require(len(set(known_sequences)) == len(known_sequences), "Overlapping terminal block dispositions")
    require(len(known_sequences)+terminal["droppedBlockCount"] == terminal["sequence"],
            "Emitted, dropped and canceled block inventory does not reconcile")
    require(sum(b["reason"] == "inference-canceled" for b in losses) == diagnostic.get("researchCancelledBlockCount", 0),
            "Terminal inference cancellation inventory mismatch")
    processed_service = [b["workerServiceMs"] for b in blocks if b["disposition"] == "processed"]
    require(distribution(processed_service) == summary["completeProcessedBlockServiceMs"], "Complete-block service summary changed")
    return frames, blocks, ledger, dict(categories)


def validate_certificate(frame, mapping):
    timing=frame["timing"];lease=frame.get("renderLease") or {};certificate=timing.get("renderFreshness") or {}
    elapsed=timing["mainResultArrivalMs"]-lease.get("issuedAtMainMs",float("nan"))
    require(certificate.get("status")=="certified" and lease.get("schemaVersion")==certificate.get("schemaVersion")=="phenometric.voice-render-freshness.v1"
        and lease.get("leaseId")==certificate.get("leaseId")
        and lease.get("issuedAtMainMs")==certificate.get("issuedAtMainMs")
        and lease.get("mainTimeOriginMs")==mapping["mainTimeOriginMs"]
        and lease.get("captureEpoch")==frame["captureEpoch"] and lease.get("clockMappingId")==timing["clockMappingId"]==mapping["id"]
        and lease.get("expiresAtMainMs")==lease.get("issuedAtMainMs",0)+100
        and certificate.get("resultReceivedAtMainMs")==timing["mainResultArrivalMs"]
        and certificate.get("maximumAgeMs")==100 and certificate.get("timestampPrecisionMs") is None
        and certificate.get("basis")=="main-lease-before-block-first-input" and certificate.get("scope")=="frame-last-required-rendered-input"
        and math.isfinite(elapsed) and 0<=elapsed<=100
        and math.isclose(elapsed,certificate.get("renderAgeUpperBoundMs",float("nan")),abs_tol=1e-6), "Invalid same-clock render-age certificate")
    return elapsed


def audit_timing(result, frames, blocks, batches):
    require(result["clockDiagnostics"]["enabled"] is True, "Natural clock probes missing")
    mapping=result["clockDiagnostics"]["fixedMapping"]
    start=result["naturalSource"]["scheduledStartAudioFrame"]
    project=lambda end: mapping["mainPerformanceMs"]+((start+end)/48000-mapping["audioContextTimeSeconds"])*1000
    certified=[];invalid=0
    for frame in frames:
        timing=frame["timing"];end=frame["research"]["supportEndSampleExclusive"]
        block=next((b for b in blocks if b["disposition"]=="processed" and b["supportStartSample"]<end<=b["supportEndSampleExclusive"]),None)
        require(block is not None,"Completed prediction lacks processed block")
        expected=dict(sampleAvailabilityMs=project(end),rawSampleAvailabilityToWorkerReadyMs=timing["workerReadyAtMs"]-project(end),
            rawBlockEndToWorkerReceiptMs=timing["workerReceivedAtMs"]-project(block["supportEndSampleExclusive"]),
            rawSampleAvailabilityToMainResultMs=timing["mainResultArrivalMs"]-project(end))
        for key,value in expected.items():
            require(math.isfinite(timing[key]) and math.isclose(timing[key],value,abs_tol=1e-6,rel_tol=1e-10),"Fixed audio projection arithmetic changed: "+key)
        if frame.get("deliveryDisposition")=="discarded":continue
        certified.append(validate_certificate(frame,mapping))
        invalid+=int(timing["status"]!="valid")
        for qualified,raw in (("sampleAvailabilityToWorkerReadyMs","rawSampleAvailabilityToWorkerReadyMs"),
                              ("blockEndToWorkerReceiptMs","rawBlockEndToWorkerReceiptMs"),
                              ("sampleAvailabilityToMainResultMs","rawSampleAvailabilityToMainResultMs")):
            value=timing.get(qualified)
            if value is not None:
                require(value>=0 and math.isclose(value,timing[raw],abs_tol=1e-6),"Unavailable timing was clamped or altered")
    summary=result["summary"]
    require(summary["certifiedFrameCount"]==len(certified) and summary["renderAgeUpperBoundMs"]==distribution(certified)
            and summary["invalidFrameTimingCount"]==invalid,"Timing/certificate summary changed")
    service=[b["workerServiceMs"] for b in blocks]
    require(summary["completeBlockServiceMs"]==distribution(service)
            and summary["completeBlockDeadlineMisses"]==sum(v>20 for v in service)
            and summary["completeBlockServiceBudgetExceedances"]==sum(b["workerServiceMs"]>10 for b in blocks if b["disposition"]=="processed"),
            "Whole block service/deadline inventory changed")
    probes=[row["value"] for batch in batches for row in batch["records"] if row["kind"]=="clock-probe"]
    clock=analyze_probes(probes,mapping)
    require(clock["counts"].get("probes",0)==summary["clockProbeCount"] and clock["counts"].get("render_observed",0)==summary["clockProbeObservedCount"],
            "Clock probe inventory changed")
    return dict(status="passed",certifiedFrames=len(certified),invalidFrameTimingCount=invalid,clock=clock)


def acoustic_parity(pcm_path, frames, exported, output):
    requests = [dict(segment=f["trackSegmentId"], end=f["research"]["supportEndSampleExclusive"]) for f in frames]
    request_path, feature_path = output / "actual-supports.json", output / "reconstructed-features.json"
    request_path.write_text(json.dumps(requests) + "\n")
    subprocess.run(["node", "--import", "tsx", "../../research/voice-prosody/natural_browser_features.ts",
                    str(pcm_path), str(request_path), str(feature_path)], cwd=REPO / "apps/capture-web", check=True, capture_output=True)
    features = json.loads(feature_path.read_text())
    config = ModelConfig(**exported["manifest"]["config"])
    weights = exported["weights"]
    pcm = np.fromfile(pcm_path, dtype="<f4")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO / "research/voice-prosody/runs/.numba-cache"))
    import librosa
    maximum = dict(feature=0., voicing_logit=0., candidate_logit=0., raw_yin_hz=0.)
    for actual, independent in zip(frames, features):
        prediction = actual["research"]
        values = np.asarray(independent["values"], np.float32)
        require(independent["mask"] == prediction["candidateMask"], "Acoustic candidate mask parity failed")
        feature_error = float(np.max(np.abs(values - np.asarray(prediction["featureValues"]))))
        require(np.allclose(values, prediction["featureValues"], rtol=1e-5, atol=1e-6), "Actual-reset acoustic feature parity failed")
        voicing, logits, _ = numpy_step(config, weights, values[None], np.asarray(independent["mask"])[None], np.zeros((1,32), np.float32))
        logit = float(voicing[0,0]); logit_error = abs(logit-prediction["voicingLogit"])
        require(np.isclose(logit, prediction["voicingLogit"], rtol=1e-5, atol=1e-6)
                and np.allclose(logits[0], prediction["candidateLogits"], rtol=1e-5, atol=1e-6), "Independent NumPy model parity failed")
        end = independent["end"]
        raw = float(librosa.yin(pcm[end-1920:end].astype(np.float64), fmin=50, fmax=700, sr=48000,
                                 frame_length=1920, hop_length=480, center=False, trough_threshold=.1)[0])
        error = abs(raw-prediction["rawYin"]["rawF0Hz"])
        require(error <= max(.01, 1e-4 * abs(raw)), "Independent librosa raw YIN parity failed")
        probability = 1/(1+math.exp(-logit))
        state = "unvoiced" if probability < .5 else "voiced" if 50 <= raw <= 700 else "unavailable"
        require(state == prediction["state"], "Independent hybrid decision parity failed; boundary is not an unconditional pass")
        for name,value in (("feature",feature_error),("voicing_logit",logit_error),
                           ("candidate_logit",float(np.max(np.abs(logits[0]-prediction["candidateLogits"])))),("raw_yin_hz",error)):
            maximum[name]=max(maximum[name],value)
    require(len(features)==len(frames), "Reconstructed acoustic frame count changed")
    return dict(status="passed", frames=len(frames), maximum_absolute_errors=maximum,
                featureEquations="shared pure TypeScript; actual support/history independently reconstructed", modelEquations="NumPy", yin="librosa@0.11.0-float64")


def score_reference(record, frames, ledger, data):
    reference_path = within(data, record["reference"]["path"])
    require(sha(reference_path) == record["reference"]["sha256"], "Reference content identity changed")
    references = [replace(ref, eligible=ref.sample >= 3072 and ref.sample+1536 <= record["originalSampleCount"])
                  for ref in references_from_array(np.loadtxt(reference_path), record["originalSampleCount"])]
    by_target = {ref.sample:ref for ref in references}
    for row in ledger:
        ref=by_target.get(row["originalTargetSample"])
        row["referenceStatus"] = "missing" if ref is None else "invalid" if not ref.valid else "boundary" if not ref.eligible else "valid"
    def prediction(frame):
        p=frame["research"]
        return Prediction(p["referenceSample"]-192,p["state"],p["f0Hz"])
    all_outputs=[prediction(f) for f in frames]
    delivered_outputs=[prediction(f) for f in frames if f.get("deliveryDisposition") != "discarded"]
    # Missing provider rows are explicit reference-status evidence, not labels.
    # Keep every source target in the delivery ledger and every eligible provider
    # row in the quality denominator; only unmatched outputs cannot be scored.
    all_predictions=[p for p in all_outputs if p.sample in by_target]
    delivered=[p for p in delivered_outputs if p.sample in by_target]
    delivered_targets={p.sample for p in delivered}
    all_score=score(references,all_predictions)
    delivered_score=score(references,delivered)
    return dict(estimatorCompletedIncludingWithheld=all_score, deliveredUnconditional=delivered_score,
        deliveredUsableCorrectCoverage=1-delivered_score["joint_error"] if delivered_score["joint_error"] is not None else None,
        conditionalAmongDelivered=score([r for r in references if r.sample in delivered_targets],delivered),
        referenceStatuses=dict(Counter(row["referenceStatus"] for row in ledger)),
        outputsWithoutProviderReference=len(all_outputs)-len(all_predictions),
        deliveredOutputsWithoutProviderReference=len(delivered_outputs)-len(delivered))


def expected_cases(manifest):
    require(manifest.get("playlist") == ["H", "H"], "Frozen H/H fallback playlist changed")
    records=manifest["records"]
    require(records and len({row["id"] for row in records})==len(records), "Frozen source list is empty or duplicated")
    return [dict(**manifest["indexFixture"],caseId="index-origin",role="origin-fixture")] + [
        dict(**record,caseId=f"{index}-{model}-{record['id']}",role="ordinary",model=model)
        for index,model in enumerate(manifest["playlist"]) for record in records] + [
        dict(**records[0],caseId="injected-stall",role="injected-stall",model="H")]


def validate_case_binding(case, result, expected, input_digest):
    # Exact fields bind PCM/reference/source/model/role before opening source content.
    require(case.get("id")==expected["caseId"] and case.get("role")==expected["role"]
            and case.get("resultPath")==expected["caseId"]+"/result.json", "Run case order, role or path differs from frozen playlist")
    require(result.get("case")==expected, "Full case record differs from frozen source/condition/playlist")
    require(result.get("inputManifestSha256")==input_digest, "Per-case input manifest identity changed")


def audit(run, inputs, data, config, export, output):
    output.mkdir(parents=True, exist_ok=False)
    run_doc=json.loads((run/"run.json").read_text())
    manifest=json.loads(inputs.read_text())
    require(run_doc["inputManifestSha256"]==sha(inputs) and manifest["status"]=="frozen", "Input manifest changed")
    require(manifest["acquisitionManifestSha256"] == sha(inputs.parent/"acquisition-manifest.json"), "Acquisition identity changed")
    selected=first_sources(json.loads((inputs.parent/"acquisition-manifest.json").read_text()))
    require([r["id"] for r in manifest["records"]] == [r["recording_id"]+"__"+c for r in selected for c in ("clean","first_difference")], "Predetermined source list changed")
    identity,_=bound_assets(config,run_doc["configurationSha256"],REPO/"apps/capture-web/public")
    model_manifest=json.loads((export/"manifest.json").read_text())
    require(sha(export/"manifest.json")==identity["modelManifestSha256"], "Independent model manifest changed")
    require(sha(export/"weights.npz")==model_manifest["artifacts"]["weights"]["sha256"], "Independent model weights changed")
    with np.load(export/"weights.npz",allow_pickle=False) as stored:
        weights={k:stored[k].copy() for k in stored.files}
    for path,digest in run_doc["sourceSha256"].items():
        require(sha(within(run/"source-snapshot",path))==digest and sha(within(REPO,path))==digest,"Measured/audited source changed: "+path)
    require(not run_doc["changedSources"], "Measured source drift")
    expected=expected_cases(manifest)
    require(len(run_doc["cases"])==len(expected), "Missing or additional cases in frozen execution inventory")
    cases=[]
    for case,expected_case in zip(run_doc["cases"],expected):
        path=within(run,case["resultPath"]); require(sha(path)==case["resultSha256"],"Result status/bytes changed")
        result=json.loads(path.read_text())
        validate_case_binding(case,result,expected_case,sha(inputs))
        require(all(result.get("researchModel",{}).get(k)==v for k,v in identity.items()), "Per-case model header identity changed")
        record=expected_case
        require(result["status"]==case["status"] and result["sourceSha256"]==run_doc["sourceSha256"],"Execution/source receipt changed")
        pcm_path=within(inputs.parent,record["pcmPath"]);require(sha(pcm_path)==record["pcmSha256"],"PCM identity changed")
        raw_path=within(path.parent,result["rawArtifact"]["file"]);require(sha(raw_path)==result["rawArtifact"]["sha256"],"Raw telemetry changed")
        batches=[json.loads(line) for line in raw_path.read_text().splitlines()]
        pcm=np.fromfile(pcm_path,dtype="<f4")
        frames,blocks,ledger,categories=audit_records(result,record,pcm,batches,identity)
        timing_audit=audit_timing(result,frames,blocks,batches)
        case_output=output/case["id"];case_output.mkdir()
        parity=acoustic_parity(pcm_path,frames,dict(manifest=model_manifest,weights=weights),case_output)
        quality=None if case["role"]=="origin-fixture" else score_reference(record,frames,ledger,data)
        (case_output/"target-ledger.json").write_text(json.dumps(ledger)+"\n")
        cases.append(dict(id=case["id"],role=case["role"],executionStatus=result["status"],parity=parity,timingAudit=timing_audit,quality=quality,
            intendedTargets=len(ledger),deliveryCounts=categories,deliveryAvailability=categories.get("delivered",0)/len(ledger) if ledger else None,
            completeProcessedBlockServiceMs=result["summary"]["completeProcessedBlockServiceMs"],
            renderAgeUpperBoundMs=result["summary"]["renderAgeUpperBoundMs"],diagnostics=result["diagnostics"],
            terminal=result["transportStop"],sampledRendererJsHeapBytes=result.get("sampledJsHeapBytes"), runtimeGoalMet=result["summary"]["completeProcessedBlockServiceMs"]["p99"] is not None and result["summary"]["completeProcessedBlockServiceMs"]["p99"]<10))
    expected_case_count=2*len(manifest["records"])+2
    complete=len(cases)==expected_case_count and run_doc["executionStatus"]=="complete"
    receipt=dict(schema="phenometrix.natural-browser-audit.v2",executionStatus=run_doc["executionStatus"],
        artifactCompleteness="complete" if len(cases)==expected_case_count else "incomplete", numericalVerification="passed",
        browserQuality="assessed" if complete else "incomplete", scientificIndependence="development implementation replay only",
        auditSourceSha256=sha(__file__),runSha256=sha(run/"run.json"),inputManifestSha256=sha(inputs),cases=cases,
        auditedAt=datetime.now(timezone.utc).isoformat())
    (output/"audit.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    return dict(artifactCompleteness=receipt["artifactCompleteness"],cases=len(cases),numericalVerification="passed")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","inputs","data","config","export","output"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    try: print(json.dumps(audit(**{k:v.resolve() for k,v in vars(a).items()})))
    except BaseException as error:
        if a.output.exists(): (a.output/"audit-failure.json").write_text(json.dumps(dict(status="failed",error=repr(error),at=datetime.now(timezone.utc).isoformat()))+"\n")
        raise

if __name__=="__main__": main()
