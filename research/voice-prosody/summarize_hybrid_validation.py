"""Publish compact reproducible evidence; exclude raw PCM, labels and predictions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from run_hybrid_browser import PLAN


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def compact_empty_calibration(value):
    """Omit only redundant all-empty bins; preserve counts and every measured metric."""
    if isinstance(value, list):
        return [compact_empty_calibration(item) for item in value]
    if isinstance(value, dict):
        empty = (value.get('calibration_n') == 0 and 'reliability_bins' in value
                 and all(row.get('n') == 0 and row.get('mean_probability') is None
                         and row.get('observed_correct_fraction') is None for row in value['reliability_bins']))
        return {key: compact_empty_calibration(item) for key, item in value.items()
                if not (empty and key == 'reliability_bins')}
    return value


def runtime_components(result, method):
    s, d = result['summary'], result['diagnostics']
    p99 = s['completeProcessedBlockServiceMs']['p99']
    age = s['renderAgeUpperBoundMs']['maximum']
    lease = s['freshnessLeaseControl']
    components = {
        'completed': result['status'] == 'complete' and result['validationStatus'] == 'passed',
        'complete_block_p99_under_10ms': isinstance(p99, (int, float)) and math.isfinite(p99) and p99 < 10,
        'all_delivered_certified': s['frameCount'] > 0 and s['frameCount'] == s['certifiedFrameCount'],
        'recorded_age_bound_at_most_100ms': isinstance(age, (int, float)) and math.isfinite(age) and 0 <= age <= 100,
        'pcm_credits_bounded': d['maximumInFlightBlocks'] <= 4,
        'result_credits_bounded': d['maximumInFlightResultBatches'] <= 4,
        'lease_control_bounded': lease['maximumPendingRequests'] <= 1 and lease['maximumLedgerEntries'] <= 8,
        'segment_recovery': s['segmentCount'] >= 3,
        'stop_acknowledgments': result['workerStopAcknowledged'] and result['transportStop']['status'] == 'complete',
        'raw_telemetry_complete_bounded': s['rawTelemetry']['status'] == 'complete' and s['rawTelemetry']['maximumPendingRecords'] <= 200,
    }
    if method == 'hybrid':
        components['hybrid_output_for_every_delivered_frame'] = s['researchFrameCount'] == s['frameCount']
        components['known_pitch_correct_on_every_delivered_frame'] = s['researchCorrectKnownPitchFrameCount'] == s['frameCount']
        components['async_queue_bounded'] = d['researchMaximumQueuedBlocks'] <= 3
    return {'pass': all(components.values()), 'components': components}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--natural', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--preaccess-audit', type=Path, required=True)
    parser.add_argument('--natural-audit', type=Path, required=True)
    parser.add_argument('--parity', type=Path, action='append', default=[])
    parser.add_argument('--functional', type=Path, action='append', default=[])
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frozen = read(args.freeze)
    natural = read(args.natural / 'manifest.json')
    metrics_path = args.natural / 'metrics.json'
    predictions = args.natural / 'prediction-freeze.json'
    metrics = read(metrics_path)
    if (natural['status'] != 'complete' or natural['freeze_sha256'] != sha(args.freeze)
            or natural['metrics_sha256'] != sha(metrics_path)
            or metrics['prediction_freeze_sha256'] != sha(predictions)
            or frozen['preaccess_audit_sha256'] != sha(args.preaccess_audit)):
        raise ValueError('Natural evidence identity or completion mismatch')
    audit = read(args.natural_audit)
    if (audit.get('status') != 'passed' or audit.get('freeze_sha256') != sha(args.freeze)
            or audit.get('prediction_freeze_sha256') != sha(predictions)
            or audit.get('metrics_sha256') != sha(metrics_path)):
        raise ValueError('Independent natural audit identity mismatch')
    summary = {'schema': 'phenometrix-hybrid-browser-natural-validation-summary.v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Within-corpus unseen speakers and named local browser research path; no production promotion',
        'natural': {'status': natural['status'], 'manifest_sha256': sha(args.natural / 'manifest.json'),
            'metrics_sha256': sha(metrics_path), 'prediction_freeze_sha256': sha(predictions),
            'freeze_sha256': sha(args.freeze), 'freeze': frozen, 'metrics': compact_empty_calibration(metrics),
            'metrics_public_compaction': 'Only redundant all-empty reliability_bins with calibration_n=0 are omitted. All measured values, counts and nonempty bins are preserved. metrics_sha256 binds the complete local source.',
            'preaccess_audit_sha256': sha(args.preaccess_audit), 'preaccess_audit': read(args.preaccess_audit),
            'independent_audit_sha256': sha(args.natural_audit), 'independent_audit': audit},
        'runtime': None, 'production_estimator': 'browser-voice-dsp@1.1',
        'correctness_probability': None, 'all_computation_local': True}
    summary['browser_parity'] = []
    for folder in args.parity:
        path = folder / 'browser-hybrid-pcm-parity.json'
        raw = read(path)
        compact = {k: v for k, v in raw.items() if k != 'records'}
        compact['records'] = [{'id': r['id'], 'status': r['status'], 'pcm_sha256': r['pcmSha256'],
            'provenance': r['provenance'], 'frame_count': len(r['frames'])} for r in raw['records']]
        compact['all_declared_parity_checks_pass'] = raw['status'] == 'complete' and not raw['failures'] and not raw['boundaryExceptions']
        manifest_path = folder / 'manifest.json'
        summary['browser_parity'].append({'run': folder.name, 'artifact_sha256': sha(path),
            'result': compact, 'manifest_sha256': sha(manifest_path) if manifest_path.exists() else None,
            'manifest': read(manifest_path) if manifest_path.exists() else None})
    summary['browser_functional'] = []
    for path in args.functional:
        raw = read(path)
        compact = {k: v for k, v in raw.items() if k not in ('frames', 'blockTimings', 'clockProbes')}
        summary['browser_functional'].append({'run': path.parent.name, 'artifact': path.name,
            'artifact_sha256': sha(path), 'result': compact})
    if args.runtime:
        runtime = read(args.runtime / 'manifest.json')
        if runtime['status'] != 'complete' or len(runtime['runs']) != 5:
            raise ValueError('Requires the complete frozen five-run runtime series')
        expected = [(name, method, duration, duration//3, 2*duration//3, 300, 350) for name, method, duration in PLAN]
        actual = [(r['name'], r['method'], r['duration_ms'], r['worker_stall_at_ms'], r['main_stall_at_ms'], r['worker_stall_ms'], r['main_stall_ms']) for r in runtime['runs']]
        if actual != expected:
            raise ValueError('Runtime sequence differs from frozen plan')
        runs = []
        for item in runtime['runs']:
            if (item['status'] != 'complete' or item['exit_code'] != 0 or not item['source_stable']
                    or item['source_changed'] or item['source_sha256'] != runtime['source_sha256']):
                raise ValueError('Runtime source or completion binding failed')
            folder = args.runtime / item['name']
            if read(folder / 'manifest.json') != item:
                raise ValueError('Per-run manifest differs from the series record')
            result_path = folder / 'browser-replay.json'
            if item['artifacts']['browser-replay.json'] != sha(result_path):
                raise ValueError('Browser result bytes changed')
            result = read(result_path)
            if result['status'] != 'complete' or result['validationStatus'] != 'passed':
                raise ValueError('Browser run is not complete/passed')
            # Compact summaries and explicit terminal state; no frame-level signal or feature payload.
            selected = {k: result[k] for k in ('status', 'validationStatus', 'evidence', 'durationMs',
                'requestedSourceDurationMs', 'pacedSourceDurationMs', 'setupDurationMs', 'teardownDurationMs',
                'sampleRateHz', 'researchModel', 'browser', 'errors', 'diagnostics', 'transportStop',
                'workerStopAcknowledged', 'summary', 'rawArtifact', 'replayConfiguration',
                'timingBenchmark', 'timingLimitation', 'runPurpose')}
            audits = {}
            for name in ('independent-audit.json', 'clock-audit.json', 'hybrid-audit.json'):
                path = folder / name
                if path.exists():
                    audit = read(path)
                    if (audit.get('result_sha256') != sha(result_path)
                            or audit.get('raw_sha256') != result['rawArtifact']['sha256']):
                        raise ValueError('Audit belongs to different evidence: ' + name)
                    if audit.get('status') not in ('passed', 'pass'):
                        raise ValueError('Audit did not pass: ' + name)
                    audits[name] = {'sha256': sha(path), 'result': audit}
            if 'clock-audit.json' not in audits:
                raise ValueError('Independent clock audit required')
            if 'independent-audit.json' not in audits:
                raise ValueError('Independent raw telemetry audit required')
            if item['method'] == 'hybrid' and 'hybrid-audit.json' not in audits:
                raise ValueError('Hybrid raw telemetry audit required')
            runs.append({'name': item['name'], 'method': item['method'], 'result_sha256': sha(result_path),
                         'result': selected, 'audits': audits, 'declared_components': runtime_components(result, item['method'])})
        summary['runtime'] = {'manifest_sha256': sha(args.runtime / 'manifest.json'), 'manifest': runtime, 'runs': runs,
            'all_declared_runtime_components_pass': all(r['declared_components']['pass'] for r in runs)}
        series_audit_path = args.runtime / 'independent-series-audit.json'
        if series_audit_path.exists():
            series_audit = read(series_audit_path)
            if (series_audit['status'] != 'passed'
                    or series_audit['series_manifest_sha256'] != sha(args.runtime / 'manifest.json')
                    or series_audit['series_plan_sha256'] != sha(args.runtime / 'plan.json')):
                raise ValueError('Independent series audit identity mismatch')
            summary['runtime']['independent_series_audit'] = {'sha256': sha(series_audit_path), 'result': series_audit}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output), 'bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
