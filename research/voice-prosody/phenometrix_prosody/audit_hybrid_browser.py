"""Offline identity, decision, support and accounting audit of hybrid browser replay.

Reads saved research telemetry and local pinned assets only. This does not rerun
inference or replace Python/TypeScript PCM parity or generic transport auditing.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit, unquote

CONTRACT = 'phenometrix.mlp-voicing-yin-pitch.v1'
YIN = 'phenometrix.raw-yin-librosa011-semantics.float64.ts.v2'
CHECKPOINT = '06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d'
HASH = re.compile('[0-9a-f]{64}')
REPO = Path(__file__).resolve().parents[3]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, field, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite or nonnumeric ' + field)
    require(not nonnegative or value >= 0, 'Negative ' + field)
    return value


def integer(value, field, minimum=0):
    require(type(value) is int and value >= minimum, 'Invalid integer ' + field)
    return value


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-6)


def distribution(values):
    values = sorted(values)
    def q(p):
        return values[math.ceil(p * len(values)) - 1] if values else None
    return dict(n=len(values), p50=q(.5), p95=q(.95), p99=q(.99), maximum=q(1))


def within(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Path escaped its declared root')
    return path


def bound_assets(config_path, expected_config_sha256, public_root):
    require(HASH.fullmatch(expected_config_sha256) is not None and sha(config_path) == expected_config_sha256,
            'Configuration bytes differ from the predeclared hash')
    config = json.loads(config_path.read_text())
    require(config.get('schemaVersion') == 'phenometrix.research-voice-model-config.v1'
            and config.get('qualification') == 'unqualified-functional-fixture'
            and config.get('estimator') == CONTRACT, 'Explicit hybrid configuration required')
    files = {}
    for role in ('manifest', 'model', 'wasm', 'module'):
        url = urlsplit(config[role + 'Url'])
        require(not url.scheme and not url.netloc and url.path.startswith('/') and not url.query and not url.fragment,
                'Only pinned root-relative local asset URLs are admitted')
        path = within(public_root, unquote(url.path).lstrip('/'))
        digest = config[role + 'Sha256']
        require(HASH.fullmatch(digest) is not None and sha(path) == digest, 'Changed local ' + role + ' asset')
        files[role] = path
    manifest = json.loads(files['manifest'].read_text())
    require(manifest.get('status') == 'complete' and manifest.get('training_performed') is True
            and manifest.get('scaling_fitted') is True, 'Incomplete trained model export')
    required = dict(kind='framewise-mlp', feature_version='v2', feature_count=57,
                    candidate_count=6, hidden_size=32, projection_size=32)
    require(all(manifest.get('config', {}).get(k) == v for k, v in required.items()), 'Wrong exported model contract')
    require(manifest.get('training_metadata', {}).get('weights_sha256') == CHECKPOINT
            and manifest['training_metadata'].get('measurement_delay_ms') == 20
            and manifest.get('artifacts', {}).get('onnx', {}).get('sha256') == config['modelSha256'],
            'Wrong checkpoint, delay or ONNX identity')
    names = manifest.get('feature_names', [])
    require(len(names) == 57 and len(set(names)) == 57, 'Invalid bound feature ordering')
    identity = dict(configurationSha256=expected_config_sha256, modelManifestSha256=config['manifestSha256'],
                    checkpointSha256=CHECKPOINT, modelSha256=config['modelSha256'], estimatorContract=CONTRACT)
    return identity, {role: dict(path=str(path.relative_to(public_root)), sha256=sha(path)) for role, path in files.items()}


def validate_research(frame, identity):
    """Independent recombination of logged raw evidence; not an acoustic estimator."""
    research = frame.get('research')
    require(isinstance(research, dict), 'Research prediction missing, including on discarded frame')
    require(all(research.get(k) == v for k, v in identity.items()), 'Per-frame research identity differs')
    require(research.get('schemaVersion') == 'phenometrix.research-voice-prediction.v1'
            and research.get('qualification') == 'unqualified-functional-fixture'
            and research.get('featureSchema') == 'phenometrix.voice-pitch-features.v2'
            and research.get('measurementDelayMs') == 20
            and 'correctnessProbability' in research and research['correctnessProbability'] is None
            and research.get('probabilitySemantics') == 'uncalibrated-model-probabilities-not-correctness'
            and research.get('candidateHeadRole') == 'diagnostic-only-unused-by-hybrid'
            and 'candidateIndex' in research and research['candidateIndex'] is None,
            'Hybrid qualification, uncertainty or candidate-head contract changed')
    timing = frame['timing']
    require(frame['sampleRateHz'] == timing['sampleRateHz'] == 48000, 'Wrong frame sample rate')
    end = integer(research['supportEndSampleExclusive'], 'support endpoint', 1920)
    start = integer(research['supportStartSample'], 'support start')
    require(start == end - 1920 and research['referenceSample'] == end - 960
            and research['lastRequiredInputSample'] == end - 1
            and timing['supportStartSample'] == start and timing['supportEndSampleExclusive'] == end
            and timing['lastRequiredInputSample'] == end - 1 and timing['referenceSample'] == end
            and frame['absoluteSampleIndex'] == start, 'Wrong causal support, target or availability mapping')
    # Browser replay starts at its first complete 40 ms window; this is not the PTDB 16+10k target grid.
    require(close(frame['tMs'], start / 48), 'Legacy frame start timestamp disagrees with support')
    logit = number(research['voicingLogit'], 'voicing logit')
    probability = number(research['voicingProbability'], 'voicing probability')
    expected_probability = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
    require(0 <= probability <= 1 and math.isclose(probability, expected_probability, abs_tol=1e-12, rel_tol=1e-12),
            'Voicing probability does not match its logit')
    for field, count in (('featureValues', 57), ('candidateLogits', 6), ('candidateProbabilities', 6)):
        values = research[field]
        require(isinstance(values, list) and len(values) == count, 'Wrong ' + field + ' shape')
        for value in values:
            number(value, field)
    mask, hz = research['candidateMask'], research['candidateHz']
    require(len(mask) == len(hz) == 6 and all(type(v) is bool for v in mask), 'Wrong candidate diagnostic shape')
    for value in hz:
        if value is not None:
            number(value, 'candidate Hz', True)
    logits, probabilities = research['candidateLogits'], research['candidateProbabilities']
    maximum = max((value for value, valid in zip(logits, mask) if valid), default=0)
    weights = [math.exp(value - maximum) if valid else 0 for value, valid in zip(logits, mask)]
    total = sum(weights)
    require(all(math.isclose(p, w / total if total else 0, abs_tol=1e-12, rel_tol=1e-12)
                for p, w in zip(probabilities, weights)), 'Candidate diagnostic probabilities changed')
    yin = research.get('rawYin', {})
    require(yin.get('implementationVersion') == YIN and yin.get('referenceLibrary') == 'librosa@0.11.0'
            and type(yin.get('thresholdTrough')) is bool, 'Raw YIN implementation evidence missing')
    period = integer(yin['selectedPeriod'], 'YIN selected period', 68)
    shift = number(yin['parabolicShift'], 'YIN parabolic shift')
    raw = number(yin['rawF0Hz'], 'raw YIN F0')
    require(period <= 960 and abs(shift) < 1 and (period not in (68, 960) or shift == 0)
            and close(raw, 48000 / (period + shift)), 'Raw YIN period-to-Hz mapping changed')
    state = 'unvoiced' if probability < .5 else 'voiced' if 50 <= raw <= 700 else 'unavailable'
    require(research['state'] == state and research['f0Hz'] == (raw if state == 'voiced' else None),
            'Final hybrid result differs from frozen voicing/raw-YIN rule')
    for field in ('inferenceServiceMs', 'elapsedFromFrameStartMs', 'modelReadyAtMs'):
        number(research[field], field, True)
    number(yin['serviceMs'], 'YIN service', True)
    raw_latency = number(research['rawSampleAvailabilityToModelReadyMs'], 'raw model projection')
    require(close(raw_latency, research['modelReadyAtMs'] - number(timing['sampleAvailabilityMs'], 'sample projection')),
            'Raw model availability projection changed')
    qualified, target = research['sampleAvailabilityToModelReadyMs'], research['targetReferenceToModelReadyMs']
    if qualified is None:
        require(target is None, 'Unavailable model projection must stay null')
    else:
        number(qualified, 'model availability projection', True)
        number(target, 'model target projection', True)
        require(close(qualified, raw_latency) and close(target, qualified + 20), 'Wrong delayed model projection')
    # A signed fixed-clock projection may be negative; it is not a freshness certificate.
    require(research['elapsedFromFrameStartMs'] + 1e-3 >= research['inferenceServiceMs'] + yin['serviceMs'],
            'Full frame elapsed time omits preparation or inference')
    return research


def audit(run, config_path, expected_config_sha256, public_root, repository=REPO, *, allow_unbound_functional_source=False):
    identity, assets = bound_assets(config_path, expected_config_sha256, public_root)
    result_path = run / 'browser-replay.json'
    result = json.loads(result_path.read_text())
    require(result['status'] == 'complete' and result['validationStatus'] == 'passed' and not result['errors'],
            'Replay is not complete and validated')
    require(result['sampleRateHz'] == 48000 and result['workerStopAcknowledged'] is True
            and result['transportStop']['status'] == 'complete', 'Wrong sample rate or incomplete stop')
    header = result.get('researchModel') or {}
    require(all(header.get(k) == v for k, v in identity.items())
            and header.get('qualification') == 'unqualified-functional-fixture', 'Ready/header hybrid identity mismatch')
    number(header['loadMs'], 'model loading duration', True)
    raw_path = within(run, result['rawArtifact']['file'])
    require(sha(raw_path) == result['rawArtifact']['sha256'], 'Raw telemetry hash changed')
    source_manifest = run / 'manifest.json'
    if source_manifest.exists():
        manifest = json.loads(source_manifest.read_text())
        sources = manifest.get('source_sha256', {})
        require(sources and manifest.get('source_changed') in (None, False, []) and manifest.get('status') == 'complete',
                'Complete unchanged-source run manifest required')
        for name, digest in sources.items():
            require(HASH.fullmatch(digest) is not None and sha(within(repository, name)) == digest,
                    'Measured source changed: ' + name)
    else:
        require(allow_unbound_functional_source, 'Missing measured source manifest; functional-only opt-in required')
        sources = {}
    counts, states, discard_reasons = Counter(), Counter(), Counter()
    values, pending, last_end, sequences = defaultdict(list), {}, {}, set()
    blocks = set()
    expected_hz = number(result['summary']['knownPitchHz'], 'declared synthetic pitch', True)
    require(expected_hz in (200, 300), 'Unknown declared replay waveform')
    with raw_path.open() as stream:
        for sequence, line in enumerate(stream, 1):
            batch = json.loads(line)
            require(batch.get('schemaVersion') == 'phenometric.voice-replay-telemetry.v1'
                    and batch['sequence'] == sequence and 0 < len(batch['records']) <= 100, 'Raw batch order/capacity changed')
            counts['batches'] += 1
            for row in batch['records']:
                counts['records'] += 1
                frame = row['value']
                if row['kind'] == 'clock-probe':
                    continue
                if row['kind'] == 'frame':
                    research = validate_research(frame, identity)
                    epoch = integer(frame['captureEpoch'], 'capture epoch', 1)
                    sequence = integer(frame['sequence'], 'frame sequence', 1)
                    key = epoch, sequence
                    require(key not in sequences, 'Duplicate research frame')
                    sequences.add(key)
                    group = epoch, frame['trackSegmentId']
                    end = research['supportEndSampleExclusive']
                    require(group not in last_end or end - last_end[group] == 480, 'Hidden frame gap or duplicate within a segment')
                    last_end[group] = end
                    pending[group + (end,)] = research
                    require(len(pending) <= 16, 'Research frames lack timely block completions')
                    counts['completedResearchFrames'] += 1
                    discarded = frame.get('deliveryDisposition') == 'discarded'
                    role = 'discarded' if discarded else 'delivered'
                    counts[role + 'ResearchFrames'] += 1
                    states[role + ':' + research['state']] += 1
                    if discarded:
                        require(isinstance(frame.get('discardReason'), str), 'Discarded frame lacks disposition reason')
                        discard_reasons[frame['discardReason']] += 1
                    else:
                        freshness = frame['timing'].get('renderFreshness', {})
                        require(freshness.get('status') == 'certified', 'Delivered research lacks freshness certificate')
                        age = number(freshness['renderAgeUpperBoundMs'], 'certified render age', True)
                        require(age <= 100, 'Delivered hybrid exceeds render-age ceiling')
                        counts['voicedFrames'] += research['state'] == 'voiced'
                        counts['correctKnownPitchFrames'] += research['state'] == 'voiced' and abs(research['f0Hz'] - expected_hz) <= 2
                        for field in ('inferenceServiceMs', 'elapsedFromFrameStartMs', 'sampleAvailabilityToModelReadyMs', 'targetReferenceToModelReadyMs'):
                            if research[field] is not None:
                                values[field].append(research[field])
                        values['rawYinServiceMs'].append(research['rawYin']['serviceMs'])
                    values['allCompletedInferenceServiceMs'].append(research['inferenceServiceMs'])
                    values['allCompletedYinServiceMs'].append(research['rawYin']['serviceMs'])
                elif row['kind'] == 'block-timing':
                    counts['blocks'] += 1
                    block = frame
                    key = (block['captureEpoch'], integer(block['blockSequence'], 'block sequence', 1))
                    require(key not in blocks, 'Duplicate completed block')
                    blocks.add(key)
                    service = number(block['workerServiceMs'], 'complete-block service', True)
                    completed = number(block['workerCompletedAtMs'], 'block completion', True)
                    received = number(block['workerReceivedAtMs'], 'block receipt', True)
                    require(completed >= received, 'Block completes before receipt')
                    start, end = block['supportStartSample'], block['supportEndSampleExclusive']
                    require(type(start) is int and type(end) is int and 0 <= start < end, 'Invalid block support')
                    group = block['captureEpoch'], block['trackSegmentId']
                    matching = [key for key in pending if key[:2] == group and start < key[2] <= end]
                    n = integer(block['analyzedFrameCount'], 'analyzed frame count')
                    require(n == len(matching), 'Completed block research-frame count mismatch')
                    require(block['disposition'] in ('processed', 'stale-dropped', 'freshness-withheld')
                            and (block['disposition'] == 'processed' or n == 0), 'Dropped block contains predictions')
                    combined_service = 0.
                    for key in matching:
                        research = pending.pop(key)
                        require(research['modelReadyAtMs'] <= completed + 1e-3
                                and research['elapsedFromFrameStartMs'] <= service + 1e-3,
                                'Complete block excludes model completion or elapsed frame work')
                        combined_service += research['inferenceServiceMs'] + research['rawYin']['serviceMs']
                    require(combined_service <= service + 1e-3, 'Complete block excludes serial inference/YIN work')
                    counts['blockAnalyzedFrames'] += n
                    values['completeBlockServiceMs'].append(service)
                else:
                    raise ValueError('Unknown telemetry record kind')
    require(not pending, 'Research frames have no block completion')
    require(counts['records'] == result['rawArtifact']['recordCount'] and counts['batches'] == result['rawArtifact']['batchCount'],
            'Raw artifact record/batch count mismatch')
    summary, diagnostics = result['summary'], result['diagnostics']
    expected = dict(frameCount=counts['deliveredResearchFrames'], researchFrameCount=counts['deliveredResearchFrames'],
                    discardedResultFrameCount=counts['discardedResearchFrames'], discardedResearchFrameCount=counts['discardedResearchFrames'],
                    researchVoicedFrameCount=counts['voicedFrames'], researchCorrectKnownPitchFrameCount=counts['correctKnownPitchFrames'],
                    blockCount=counts['blocks'])
    for source, target in (('inferenceServiceMs', 'modelInferenceServiceMs'), ('elapsedFromFrameStartMs', 'modelElapsedFromFrameStartMs'),
                           ('sampleAvailabilityToModelReadyMs', 'sampleAvailabilityToModelReadyMs'),
                           ('targetReferenceToModelReadyMs', 'targetReferenceToModelReadyMs'), ('rawYinServiceMs', 'rawYinServiceMs')):
        expected[target] = distribution(values[source])
    require(all(summary.get(k) == value for k, value in expected.items()), 'Browser summary differs from raw hybrid records')
    cancelled = integer(diagnostics['researchCancelledFrameCount'], 'cancelled research frame count')
    prepared = integer(diagnostics['processedFrameCount'], 'prepared research frame count')
    require(prepared == counts['completedResearchFrames'] + cancelled
            and counts['blockAnalyzedFrames'] == counts['completedResearchFrames'], 'Prepared/completed/cancelled frame accounting mismatch')
    require(not sequences or max(v for _, v in sequences) <= prepared, 'Frame sequence exceeds prepared count')
    return dict(schema='phenometrix-hybrid-browser-independent-audit.v1', status='passed',
                audit_source_sha256=sha(Path(__file__)), result_sha256=sha(result_path), raw_sha256=sha(raw_path),
                source_manifest_sha256=sha(source_manifest) if source_manifest.exists() else None,
                source_bindings_checked=len(sources), measured_source_binding='verified' if sources else 'not established; functional logged-consistency audit only',
                identity=identity, assets=assets,
                counts={**dict(counts), 'preparedResearchFrames': prepared, 'cancelledResearchFrames': cancelled},
                state_counts=dict(states), discard_reasons=dict(discard_reasons), recomputed_summary=expected,
                all_completed_service={k: distribution(values[k]) for k in ('allCompletedInferenceServiceMs', 'allCompletedYinServiceMs', 'completeBlockServiceMs')},
                limitation='Offline logged-evidence recombination and accounting; cancelled frames have no predictions. Does not recompute PCM YIN/MLP inference, replace separate cross-language parity or transport audit, establish physical latency, calibrate probabilities, or validate natural accuracy.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--public-root', type=Path, default=REPO / 'apps/capture-web/public')
    p.add_argument('--repository', type=Path, default=REPO)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-unbound-functional-source', action='store_true')
    a = p.parse_args()
    result = audit(a.run.resolve(), a.config.resolve(), a.config_sha256, a.public_root.resolve(), a.repository.resolve(),
                   allow_unbound_functional_source=a.allow_unbound_functional_source)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'output': str(a.output)}))


if __name__ == '__main__':
    main()
