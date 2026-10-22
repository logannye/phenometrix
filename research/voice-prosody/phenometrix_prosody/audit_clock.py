"""Offline clock-coordinate diagnosis from independent render probes, never latency correction.

A probe bounds the callback in main performance time by request and response. It
neither timestamps microphone samples nor assumes output-device time is capture
time. Regression slopes are descriptive; scheduling can make a single affine
clock model infeasible. Arrival measurements never enter the clock fit.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from .audit_browser import digest, distribution


def close(actual, expected):
    if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-6, rel_tol=1e-10):
        raise ValueError('Stored clock arithmetic differs from independent reconstruction')


def line(points):
    if len(points) < 2:
        return {'n': len(points), 'slope_ms_per_minute': None, 'rate_difference_ppm': None}
    xmean = sum(x for x, _ in points) / len(points)
    ymean = sum(y for _, y in points) / len(points)
    denominator = sum((x-xmean)**2 for x, _ in points)
    slope = sum((x-xmean)*(y-ymean) for x,y in points) / denominator if denominator else 0.
    residual = [y-ymean-slope*(x-xmean) for x,y in points]
    return {'n': len(points), 'slope_ms_per_minute': slope*60_000,
            'rate_difference_ppm': slope*1_000_000,
            'residual_ms': distribution(residual),
            'interpretation': 'Descriptive least-squares residual trend; no independence or physical-clock guarantee'}


def affine_interval(bounds):
    """Find slopes compatible with every causal callback bracket, allowing free intercept."""
    low, high = -math.inf, math.inf
    ordered = sorted(bounds)
    for j, (xj, lj, uj) in enumerate(ordered):
        for xi, li, ui in ordered[:j]:
            dx = xj-xi
            if dx <= 0:
                if max(li,lj)>min(ui,uj):
                    return {'feasible':False,'reason':'disjoint-brackets-at-same-render-position'}
                continue
            low=max(low,(lj-ui)/dx)
            high=min(high,(uj-li)/dx)
    return {'feasible':low<=high, 'lower_residual_rate_ppm':low*1e6 if math.isfinite(low) else None,
            'upper_residual_rate_ppm':high*1e6 if math.isfinite(high) else None,
            'interpretation':'All observed callback brackets, no dropped outliers; infeasibility permits scheduling/phase variation'}


def analyze_probes(probes, mapping):
    project = lambda seconds: mapping['mainPerformanceMs']+(seconds-mapping['audioContextTimeSeconds'])*1000
    values=defaultdict(list); windows=defaultdict(lambda:defaultdict(list)); counts=Counter(); bounds=[]
    previous=0; epoch=None; rate=None; last_render=-1
    for p in probes:
        if p['schemaVersion']!='phenometric.voice-clock-observation.v1' or p['clockMappingId']!=mapping['id']:
            raise ValueError('Probe mapping identity mismatch')
        if p['audioClockUncertaintyMs'] is not None or p['probeId']<=previous:
            raise ValueError('Probe uncertainty or ordering violation')
        if epoch is None: epoch=p['captureEpoch']
        if p['captureEpoch']!=epoch or p['contextState']!='running':
            raise ValueError('Probe capture epoch or running state changed')
        previous=p['probeId'];counts['probes']+=1
        window=int(p['sourceElapsedMs']//60_000)
        c=p['contextRead']; mid=(c['beforeMs']+c['afterMs'])/2
        if c['afterMs']<c['beforeMs']: raise ValueError('Reversed clock read bracket')
        close(c['midpointMs'],mid); close(c['bracketWidthMs'],c['afterMs']-c['beforeMs'])
        residual=mid-project(c['audioContextTimeSeconds']);close(c['fixedAnchorResidualMs'],residual)
        x=(c['audioContextTimeSeconds']-mapping['audioContextTimeSeconds'])*1000
        values['context_read_residual'].append((x,residual));windows[window]['context_read_residual_ms'].append(residual)
        o=p['outputTimestamp']
        if o['status']=='observed':
            if o['coordinate']!='output-device-only-not-capture': raise ValueError('Output coordinate changed')
            if not math.isfinite(o['beforeMs']) or o['afterMs']<o['beforeMs']:
                raise ValueError('Reversed output read bracket')
            residual=o['performanceTimeMs']-project(o['contextTimeSeconds']);close(o['fixedAnchorResidualMs'],residual)
            x=(o['contextTimeSeconds']-mapping['audioContextTimeSeconds'])*1000
            values['output_device_residual'].append((x,residual));windows[window]['output_device_residual_ms'].append(residual)
        else: counts['output_unavailable']+=1
        r=p['renderObservation']
        if r['status']=='observed':
            if r['coordinate']!='render-callback-main-performance-bracket': raise ValueError('Render coordinate changed')
            if r['responseReceivedAtMs']<r['requestSentAtMs']: raise ValueError('Reversed render bracket')
            if rate is None: rate=r['sampleRateHz']
            if not math.isfinite(rate) or rate<=0 or r['sampleRateHz']!=rate or not isinstance(r['renderStartFrame'],int) or r['renderStartFrame']<=last_render:
                raise ValueError('Inconsistent render sample position/rate')
            last_render=r['renderStartFrame']
            projected=project(r['renderStartFrame']/r['sampleRateHz'])
            lo=r['requestSentAtMs']-projected; hi=r['responseReceivedAtMs']-projected
            close(r['fixedAnchorResidualLowerMs'],lo);close(r['fixedAnchorResidualUpperMs'],hi);close(r['roundTripMs'],hi-lo)
            x=(r['renderStartFrame']/r['sampleRateHz']-mapping['audioContextTimeSeconds'])*1000
            bounds.append((x,lo,hi));values['render_midpoint_residual'].append((x,(lo+hi)/2))
            for key,v in [('render_residual_lower_ms',lo),('render_residual_upper_ms',hi),('render_round_trip_ms',hi-lo)]:
                windows[window][key].append(v)
            counts['render_observed']+=1
        else: counts['render_unavailable_'+r['reason']]+=1
    return {'counts':dict(counts),'descriptive_trends':{k:line(v) for k,v in values.items()},
            'all_probe_affine_consistency':affine_interval(bounds),
            'one_minute_clock_windows':{str(k):{f:distribution(v) for f,v in fields.items()} for k,fields in sorted(windows.items())}}


def audit(run):
    result_path=run/'browser-replay.json';result=json.loads(result_path.read_text())
    if not result['clockDiagnostics']['enabled']:raise ValueError('Run did not collect clock probes')
    raw=run/result['rawArtifact']['file']
    if raw.parent.resolve()!=run.resolve() or digest(raw)!=result['rawArtifact']['sha256']:
        raise ValueError('Raw identity mismatch')
    probes=[]; windows=defaultdict(lambda:defaultdict(list)); five_windows=defaultdict(lambda:defaultdict(list)); total=0; batches=0
    invalid_components=Counter(); invalid_component_values=defaultdict(list)
    for batches,content in enumerate(raw.open(),1):
        batch=json.loads(content)
        if batch['sequence']!=batches:raise ValueError('Batch order changed')
        for record in batch['records']:
            total+=1;v=record['value']
            if record['kind']=='clock-probe':probes.append(v)
            elif record['kind']=='frame' and v.get('deliveryDisposition')!='discarded':
                t=v['timing'];window=int(t['supportEndSampleExclusive']/t['sampleRateHz']//60)
                if t['status']!='valid':
                    components={'worker_service':t['workerServiceMs'],
                        'availability_to_ready':t['rawSampleAvailabilityToWorkerReadyMs'],
                        'blockend_to_receipt':t['rawBlockEndToWorkerReceiptMs'],
                        'availability_to_main':t['rawSampleAvailabilityToMainResultMs'],
                        'ready_to_main':t['mainResultArrivalMs']-t['workerReadyAtMs']}
                    for component,value in components.items():
                        if not math.isfinite(value) or value<0:
                            invalid_components[component]+=1
                            if math.isfinite(value):invalid_component_values[component].append(value)
                for key in ('sampleAvailabilityToMainResultMs','blockEndToWorkerReceiptMs','workerServiceMs'):
                    if t.get(key) is not None:
                        windows[window][key].append(t[key]);five_windows[window//5][key].append(t[key])
                windows[window]['receipt_to_main_ms'].append(t['mainResultArrivalMs']-t['workerReceivedAtMs'])
                five_windows[window//5]['receipt_to_main_ms'].append(t['mainResultArrivalMs']-t['workerReceivedAtMs'])
                if t.get('renderFreshness',{}).get('status')=='certified':
                    windows[window]['render_age_upper_bound_ms'].append(t['renderFreshness']['renderAgeUpperBoundMs'])
                    five_windows[window//5]['render_age_upper_bound_ms'].append(t['renderFreshness']['renderAgeUpperBoundMs'])
    if total!=result['rawArtifact']['recordCount'] or batches!=result['rawArtifact']['batchCount']:
        raise ValueError('Raw completeness mismatch')
    summary=analyze_probes(probes,result['clockDiagnostics']['fixedMapping'])
    if summary['counts'].get('probes',0)!=result['summary']['clockProbeCount'] or summary['counts'].get('render_observed',0)!=result['summary']['clockProbeObservedCount']:
        raise ValueError('Probe summary mismatch')
    return {'schema':'phenometrix-clock-independent-audit-v1','status':'passed',
            'result_sha256':digest(result_path),'raw_sha256':digest(raw),'source_sha256':digest(Path(__file__)),
            **summary,'unavailable_timing_components':{'counts':dict(invalid_components),
                'raw_negative_values_ms':{k:distribution(v) for k,v in invalid_component_values.items()}},
            'one_minute_frame_windows':{str(k):{f:distribution(v) for f,v in fields.items()} for k,fields in sorted(windows.items())},
            'five_minute_frame_windows':{str(k):{f:distribution(v) for f,v in fields.items()} for k,fields in sorted(five_windows.items())},
            'limits':['No PCM/result arrival is used to fit clock trends.',
                      'Output timestamps describe the output device, not capture availability.',
                      'Probe round trips bound a render callback; no hardware input delay or sub-bracket precision is established.',
                      'Reported software timestamps have unknown timer quantization; interval consistency is not a physical confidence interval.',
                      'No trend correction is applied to any historical result.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.run)
    with args.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({'status':result['status'],'counts':result['counts'],'trends':result['descriptive_trends'],'output':str(args.output)}))
if __name__=='__main__':main()
