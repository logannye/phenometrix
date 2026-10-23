"""Independently recompute bounded browser telemetry summaries from saved raw rows.

This is an offline audit. It neither runs inference nor treats mapped Web Audio
coordinates as measured microphone latency. Only standard-library dependencies.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def distribution(values):
    ordered = sorted(values)
    def q(fraction):
        return ordered[math.ceil(len(ordered) * fraction) - 1] if ordered else None
    return dict(n=len(ordered), p50=q(.5), p95=q(.95), p99=q(.99), maximum=q(1))


def audit(run: Path) -> dict:
    result_path = run / 'browser-replay.json'
    result = json.loads(result_path.read_text())
    raw_path = run / result['rawArtifact']['file']
    if raw_path.parent.resolve() != run.resolve():
        raise ValueError('Raw artifact must be inside the run directory')
    if digest(raw_path) != result['rawArtifact']['sha256']:
        raise ValueError('Raw telemetry hash differs from the recorded artifact')
    if result['status'] != 'complete' or result['validationStatus'] != 'passed':
        raise ValueError('Requires a completed and validated replay')
    samples, counts, frame_counts, segments = defaultdict(list), Counter(), Counter(), []
    five_minute_blocks = defaultdict(list)
    first_support = None
    maximum_pitch_error = 0.
    freshness_failures = Counter()
    host_discard_reasons = Counter()
    freshness_values = []
    with raw_path.open() as stream:
        for batch_index, line in enumerate(stream, 1):
            batch = json.loads(line)
            if batch['sequence'] != batch_index or len(batch['records']) > 100:
                raise ValueError('Raw batch sequence/capacity violation')
            counts['batches'] += 1
            for row in batch['records']:
                counts['records'] += 1
                value = row['value']
                if row['kind'] == 'block-timing':
                    counts['blockCount'] += 1
                    service = value['workerServiceMs']
                    if not math.isfinite(service) or service < 0:
                        raise ValueError('Invalid complete-block service time')
                    samples['completeBlockServiceMs'].append(service)
                    dispositions = {'stale-dropped': 'staleBlockServiceMs',
                                    'processed': 'completeProcessedBlockServiceMs',
                                    'freshness-withheld': 'freshnessWithheldBlockServiceMs'}
                    if value['disposition'] not in dispositions:
                        raise ValueError('Unknown block disposition')
                    field = dispositions[value['disposition']]
                    if value['disposition'] == 'freshness-withheld':
                        counts['freshnessWithheldBlockCount'] += 1
                    samples[field].append(service)
                    frame_counts[f"{value['disposition']}:{value['analyzedFrameCount']}"] += 1
                    first_support = value['supportStartSample'] if first_support is None else first_support
                    window = int((value['supportStartSample'] - first_support) / result['sampleRateHz'] // 300)
                    five_minute_blocks[window].append(service)
                    counts['completeBlockDeadlineMisses'] += int(service > 20)
                    counts['completeBlockServiceBudgetExceedances'] += int(field == 'completeProcessedBlockServiceMs' and service > 10)
                elif row['kind'] == 'clock-probe':
                    if value['schemaVersion'] != 'phenometric.voice-clock-observation.v1' or value['audioClockUncertaintyMs'] is not None:
                        raise ValueError('Invalid clock probe contract')
                    counts['clockProbeCount'] += 1
                    counts['clockProbeObservedCount'] += int(value['renderObservation']['status'] == 'observed')
                elif row['kind'] == 'frame':
                    if value.get('deliveryDisposition') == 'discarded':
                        host_discard_reasons[value['discardReason']] += 1
                        freshness = value.get('timing', {}).get('renderFreshness', {})
                        if freshness.get('status') == 'withheld':
                            freshness_failures[freshness['reason']] += 1
                        counts['discardedResultFrameCount'] += 1
                        counts['staleResultFrameCount'] += int(value['discardReason'] in ('result-age-exceeded','expired-lease'))
                        continue
                    counts['frameCount'] += 1
                    timing = value['timing']
                    freshness = timing.get('renderFreshness')
                    if 'certifiedFrameCount' in result['summary']:
                        if not freshness or freshness.get('status') != 'certified':
                            raise ValueError('Required freshness evidence missing on delivered frame')
                    if freshness and freshness.get('status') == 'certified':
                        lease = value.get('renderLease')
                        if not lease or lease['leaseId'] != freshness['leaseId'] or lease['issuedAtMainMs'] != freshness['issuedAtMainMs']:
                            raise ValueError('Freshness certificate differs from original block lease')
                        elapsed = timing['mainResultArrivalMs'] - lease['issuedAtMainMs']
                        if (lease.get('schemaVersion') != 'phenometric.voice-render-freshness.v1'
                                or freshness.get('schemaVersion') != 'phenometric.voice-render-freshness.v1'
                                or lease['captureEpoch'] != value['captureEpoch']
                                or lease['clockMappingId'] != timing['clockMappingId']
                                or lease['expiresAtMainMs'] != lease['issuedAtMainMs'] + 100
                                or freshness['resultReceivedAtMainMs'] != timing['mainResultArrivalMs']
                                or freshness['maximumAgeMs'] != 100 or freshness['timestampPrecisionMs'] is not None
                                or freshness['basis'] != 'main-lease-before-block-first-input'
                                or freshness['scope'] != 'frame-last-required-rendered-input'
                                or not math.isfinite(elapsed) or not 0 <= elapsed <= 100
                                or not math.isclose(elapsed, freshness['renderAgeUpperBoundMs'], abs_tol=1e-6)):
                            raise ValueError('Invalid same-main-clock freshness certificate')
                        counts['certifiedFrameCount'] += 1
                        freshness_values.append(elapsed)
                    counts['invalidFrameTimingCount'] += int(timing['status'] != 'valid')
                    samples['frameServiceMs'].append(timing['workerServiceMs'])
                    counts['frameServiceBudgetExceedances'] += int(timing['workerServiceMs'] > 5)
                    counts['frameCadenceDeadlineMisses'] += int(timing['workerServiceMs'] > 10)
                    for field in ('sampleAvailabilityToMainResultMs', 'blockEndToWorkerReceiptMs'):
                        if timing.get(field) is not None:
                            if not math.isfinite(timing[field]) or timing[field] < 0:
                                raise ValueError('Unavailable timing must remain null')
                            samples[field].append(timing[field])
                    if not segments or segments[-1] != value['trackSegmentId']:
                        segments.append(value['trackSegmentId'])
                    if value['periodic'] and value['f0Hz'] is not None:
                        counts['periodicFrameCount'] += 1
                        error = abs(value['f0Hz'] - 200)
                        maximum_pitch_error = max(maximum_pitch_error, error)
                        counts['correctKnownPitchFrameCount'] += int(error <= 2)
                    research = value.get('research')
                    if research:
                        counts['researchFrameCount'] += 1
                        if research['qualification'] != 'unqualified-functional-fixture' or research['correctnessProbability'] is not None:
                            raise ValueError('Research model lost its uncertainty/qualification boundary')
                        for source, target in (
                            ('inferenceServiceMs', 'modelInferenceServiceMs'),
                            ('elapsedFromFrameStartMs', 'modelElapsedFromFrameStartMs'),
                            ('sampleAvailabilityToModelReadyMs', 'sampleAvailabilityToModelReadyMs'),
                            ('targetReferenceToModelReadyMs', 'targetReferenceToModelReadyMs')):
                            if research.get(source) is not None:
                                samples[target].append(research[source])
                else:
                    raise ValueError('Unknown raw record kind')
    if counts['records'] != result['rawArtifact']['recordCount'] or counts['batches'] != result['rawArtifact']['batchCount']:
        raise ValueError('Raw record/batch count mismatch')
    recomputed = {k: distribution(v) for k, v in samples.items()}
    integer_fields = ('frameCount', 'blockCount', 'researchFrameCount', 'discardedResultFrameCount',
                      'staleResultFrameCount', 'invalidFrameTimingCount', 'periodicFrameCount',
                      'correctKnownPitchFrameCount', 'frameServiceBudgetExceedances',
                      'frameCadenceDeadlineMisses', 'completeBlockServiceBudgetExceedances',
                      'completeBlockDeadlineMisses', 'clockProbeCount', 'clockProbeObservedCount', 'certifiedFrameCount')
    recomputed.update({key: counts[key] for key in integer_fields})
    if freshness_values:
        recomputed['renderAgeUpperBoundMs'] = distribution(freshness_values)
    if 'hostWithheldReasons' in result['summary']:
        recomputed['hostWithheldReasons'] = dict(host_discard_reasons)
    recomputed.update(blockFrameCounts=dict(frame_counts), segmentCount=len(segments),
                      maximumKnownPitchErrorHz=maximum_pitch_error)
    mismatches = {k: {'recomputed': v, 'browser': result['summary'].get(k)}
                  for k, v in recomputed.items() if k in result['summary'] and v != result['summary'][k]}
    return {
        'schema': 'phenometrix-browser-independent-audit-v1',
        'status': 'passed' if not mismatches else 'failed', 'mismatches': mismatches,
        'result_sha256': digest(result_path), 'raw_sha256': digest(raw_path),
        'audit_source_sha256': digest(Path(__file__)), 'raw_records': counts['records'],
        'recomputed_summary': recomputed,
        'freshness_withheld_host_reasons': dict(freshness_failures),
        'freshness_withheld_block_records': counts['freshnessWithheldBlockCount'],
        'complete_block_service_by_five_minute_source_window': {
            str(k): distribution(v) for k, v in five_minute_blocks.items()},
        'limitation': 'Synthetic paced replay; no physical microphone latency or clinical accuracy claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = audit(args.run)
    with args.output.open('x') as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': evidence['status'], 'raw_records': evidence['raw_records'], 'output': str(args.output)}))
    if evidence['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
