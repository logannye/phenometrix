"""Training-only reference timing sensitivity; never changes primary timestamps.

Offsets and sensor-cycle criteria are predeclared constants. LAR cycle detection
and all query estimates are completed before loading reference outcomes. Both
peak polarities are retained. There is no per-file offset selection, calibration,
model training, comparator substitution, or held-out acquisition in this module.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import wave

import numpy as np
from scipy import signal

from .data import DEFAULT_ROOT, read_reference, json_write, utc_now

OFFSETS_MS = (-30, -20, -10, 0, 10, 20, 30)
SUPPORTS_MS = {"cycle_interval": (0, 0), "centered32": (-16, 16),
               "trailing32": (-32, 0), "centered40": (-20, 20), "trailing40": (-40, 0)}
POLARITIES = (1, -1)
POLICY = {"derivative": "centered Savitzky-Golay,9 samples,degree2,first derivative",
    "noise_estimate": "MAD of derivative minus its centered9sample degree2 smoothing /0.67448975",
    "peak_prominence_noise_multiple": 6.0, "f0_bounds_hz": [50, 700],
    "neighbor_period_ratio_max": 1.15, "neighbor_cycle_shape_correlation_min": 0.85,
    "phase_shape_samples": 64, "window_strong_cycle_coverage_min": 0.9,
    "transition_threshold_cents_over_40ms": 80,
    "both_peak_polarities_retained": True,
    "offset_sign": "raw query time = nominal reference time + offset; positive queries later raw sensor",
    "common_mask": "same reference-voiced rows require independent-cycle support for all offsets and all five support modes, separately for each polarity",
    "offset_selection": "none; no primary score or timestamp changed"}
RESEARCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RESEARCH_ROOT.parents[1]
DEFAULT_COHORT = REPO_ROOT / "benchmarks/voice-prosody/ptdb-inventory-audit.json"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/voice-prosody/lar-alignment-sensitivity.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detect_strong_cycles(raw_lar, sample_rate=48000, polarity=1):
    """Find repeatable like-event intervals from raw LAR only.

    A cycle is a period average between consecutive same-sign derivative peaks,
    not an instantaneous physiological truth. Its two neighboring cycles must
    have similar duration and >=0.85 detrended phase-shape correlation. No
    reference F0/voicing or microphone signal enters this independent probe.
    """
    values = np.asarray(raw_lar, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or sample_rate != 48000 or polarity not in POLARITIES:
        raise ValueError("Finite mono48kHz training LAR and polarity +/-1 required")
    empty = {"start": np.array([],float), "end": np.array([],float), "hz": np.array([],float),
             "strength": np.array([],float), "candidate_peaks": 0, "polarity": polarity}
    if len(values) < 100:
        return empty
    derivative = signal.savgol_filter(values, 9, 2, deriv=1, delta=1/sample_rate, mode="interp")
    residual = derivative - signal.savgol_filter(derivative, 9, 2, mode="interp")
    sigma = np.median(np.abs(residual - np.median(residual))) / 0.67448975
    peaks, _ = signal.find_peaks(polarity * derivative,
        distance=int(np.floor(sample_rate / 700)), prominence=max(6*sigma,1e-12))
    peaks = peaks[(peaks >= 4) & (peaks < len(values)-4)]
    if len(peaks) < 4:
        return {**empty, "candidate_peaks": len(peaks)}
    periods = np.diff(peaks).astype(float)
    shapes = []
    for start, end in zip(peaks[:-1], peaks[1:]):
        positions = np.linspace(start,end,64,endpoint=False)
        shape = np.interp(positions,np.arange(start,end+1),values[start:end+1])
        shape = signal.detrend(shape,type="linear")
        norm = np.linalg.norm(shape)
        shapes.append(shape / norm if norm > 1e-12 else np.zeros(64))
    shapes = np.asarray(shapes)
    adjacent = np.einsum("ij,ij->i",shapes[:-1],shapes[1:])
    accepted, strengths = [], []
    for index in range(1,len(periods)-1):
        local = periods[index-1:index+2]
        ratio = np.max(local) / np.min(local)
        strength = min(adjacent[index-1],adjacent[index])
        if (np.all(local >= sample_rate/700) and np.all(local <= sample_rate/50)
            and ratio <= 1.15 and strength >= .85):
            accepted.append(index); strengths.append(strength)
    accepted = np.asarray(accepted,dtype=int)
    return {"start":peaks[accepted].astype(float), "end":peaks[accepted+1].astype(float),
        "hz":sample_rate/periods[accepted], "strength":np.asarray(strengths),
        "candidate_peaks":len(peaks), "polarity":polarity}


def cycle_values(cycles, query_samples, support_ms=(0,0), sample_rate=48000):
    """Piecewise cycle frequency, or coverage-weighted window average.

    Frequency is1/period on a strong cycle's event interval. Window estimates
    integrate that piecewise frequency and require90% strong-cycle coverage.
    Unlike an F0 tracker, this diagnostic does not fill gaps or unvoiced spans.
    """
    queries = np.asarray(query_samples,dtype=float)
    starts, ends, hz = (cycles[key] for key in ("start","end","hz"))
    if len(starts)==0:
        return np.full(queries.shape,np.nan)
    if support_ms==(0,0):
        indices=np.searchsorted(starts,queries,side="right")-1
        safe=np.clip(indices,0,len(starts)-1)
        valid=(indices>=0)&(queries<ends[safe])
        return np.where(valid,hz[safe],np.nan)
    widths=ends-starts
    cover_prefix=np.r_[0,np.cumsum(widths)]
    frequency_prefix=np.r_[0,np.cumsum(widths*hz)]

    def integral(points):
        count=np.searchsorted(ends,points,side="right")
        safe=np.minimum(count,len(starts)-1)
        partial=np.where(count<len(starts),np.clip(points-starts[safe],0,widths[safe]),0)
        return cover_prefix[count]+partial, frequency_prefix[count]+partial*hz[safe]

    left=queries+support_ms[0]*sample_rate/1000
    right=queries+support_ms[1]*sample_rate/1000
    if np.any(right<=left):
        raise ValueError("Window width must be positive")
    covered_left,freq_left=integral(left);covered_right,freq_right=integral(right)
    coverage=covered_right-covered_left
    valid=coverage>=(right-left)*.9
    return np.divide(freq_right-freq_left,coverage,out=np.full(queries.shape,np.nan),where=valid)


def estimate_grid(cycles, sample_count, sample_rate=48000):
    """Compute every fixed-offset support from LAR and PCM length only."""
    grid=np.arange(768,sample_count,480,dtype=np.int64)
    estimates={}
    for name,bounds in SUPPORTS_MS.items():
        for offset in OFFSETS_MS:
            estimates[(name,offset)]=cycle_values(cycles,grid+offset*48,bounds,sample_rate)
    before=cycle_values(cycles,grid-20*48)
    after=cycle_values(cycles,grid+20*48)
    with np.errstate(invalid="ignore",divide="ignore"):
        transition=np.isfinite(before)&np.isfinite(after)&(np.abs(1200*np.log2(after/before))>=80)
    return grid,estimates,transition


def error_summary(errors):
    errors=np.asarray(errors,dtype=float)
    if not len(errors):
        return {"n":0,"median_abs_cents":None,"p95_abs_cents":None,"mean_abs_cents":None,
                "within50_fraction":None,"median_signed_cents":None}
    absolute=np.abs(errors)
    return {"n":len(errors),"median_abs_cents":float(np.median(absolute)),
        "p95_abs_cents":float(np.quantile(absolute,.95)),"mean_abs_cents":float(np.mean(absolute)),
        "within50_fraction":float(np.mean(absolute<=50)),"median_signed_cents":float(np.median(errors))}


def compare_reference(reference_rows, estimates, transition):
    """Join reference outcomes only after sensor inference; one common frame set."""
    reference=np.asarray(reference_rows,dtype=float)
    count=len(reference)
    if any(len(value)<count for value in estimates.values()):
        raise ValueError("Reference exceeds independently generated sensor grid")
    valid=np.isfinite(reference).all(axis=1)
    voiced=valid&(reference[:,0]>0)&(reference[:,1]>0)
    common=np.logical_and.reduce([np.isfinite(value[:count]) for value in estimates.values()])
    masks={"all_common_voiced":common&voiced,
           "raw_transition_common_voiced":common&voiced&transition[:count]}
    rows,errors={},{}
    for stratum,mask in masks.items():
        for (mode,offset),values in estimates.items():
            selected=values[:count][mask]
            error=1200*np.log2(selected/reference[:,0][mask])
            key=(stratum,mode,offset)
            errors[key]=error
            rows[key]=error_summary(error)
    return {"reference_rows":count,"reference_voiced_rows":int(voiced.sum()),
        "independent_common_rows":int(common.sum()),
        "independent_common_reference_unvoiced_rows":int((common&valid&~voiced).sum()),
        "common_reference_voiced_rows":int((common&voiced).sum()),
        "raw_transition_common_voiced_rows":int(masks['raw_transition_common_voiced'].sum()),
        "rows":rows},errors


def run(root=DEFAULT_ROOT,cohort_path=DEFAULT_COHORT,output=DEFAULT_OUTPUT):
    root,cohort_path,output=map(Path,(root,cohort_path,output))
    cohort=json.loads(cohort_path.read_text())
    if cohort["pilot"]["split"]!="train":
        raise ValueError("Only the frozen training pilot cohort is permitted")
    ids=cohort["pilot"]["recording_ids"]
    manifest=json.loads((root/"manifest.json").read_text())
    files={(r["recording_id"],r["role"]):r for r in manifest["files"]}
    per_record=[];all_errors=defaultdict(list);speaker_errors=defaultdict(list);source_files=[]
    for recording in ids:
        entries={role:files[(recording,role)] for role in ("lar","ref")}
        if any(entry["split"]!="train" for entry in entries.values()):
            raise ValueError("Non-training outcome access refused")
        for role,entry in entries.items():
            if _sha(root/entry['path'])!=entry['sha256']:
                raise ValueError(f"Source integrity failed: {entry['path']}")
            source_files.append({key:entry[key] for key in ('recording_id','role','split','path','sha256')})
        with wave.open(str(root/entries['lar']['path']),'rb') as handle:
            if (handle.getframerate(),handle.getnchannels(),handle.getsampwidth())!=(48000,1,2):
                raise ValueError("Expected48kHz monoPCM16")
            raw=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2').astype(np.float64)/32768
        # Entire independent signal branch is computed before outcomes are read.
        independent=[]
        for polarity in POLARITIES:
            cycles=detect_strong_cycles(raw,polarity=polarity)
            grid,estimated,transition=estimate_grid(cycles,len(raw))
            independent.append((polarity,cycles,estimated,transition))
        references=read_reference(root/entries['ref']['path'])
        speaker=entries['lar']['speaker']
        for polarity,cycles,estimated,transition in independent:
            result,errors=compare_reference(references,estimated,transition)
            result.update(recording_id=recording,speaker=speaker,polarity=polarity,
                candidate_peaks=int(cycles['candidate_peaks']),strong_cycles=len(cycles['hz']),
                strong_cycle_seconds=float(np.sum(cycles['end']-cycles['start'])/48000))
            result['rows']=[{"stratum":key[0],"mode":key[1],"offset_ms":key[2],**value} for key,value in result['rows'].items()]
            per_record.append(result)
            for key,error in errors.items():
                all_errors[(polarity,*key)].append(error)
                speaker_errors[(speaker,polarity,*key)].append(error)
    per_speaker=[]
    for (speaker,polarity,stratum,mode,offset),arrays in speaker_errors.items():
        per_speaker.append({"speaker":speaker,"polarity":polarity,"stratum":stratum,"mode":mode,
            "offset_ms":offset,**error_summary(np.concatenate(arrays)),
            "records_with_common_frames":sum(len(a)>0 for a in arrays)})
    aggregate=[]
    for (polarity,stratum,mode,offset),arrays in all_errors.items():
        equal=[r for r in per_speaker if (r['polarity'],r['stratum'],r['mode'],r['offset_ms'])==(polarity,stratum,mode,offset) and r['n']]
        aggregate.append({"polarity":polarity,"stratum":stratum,"mode":mode,"offset_ms":offset,
            **error_summary(np.concatenate(arrays)),"speakers_with_common_frames":len(equal),
            "equal_speaker_mean_median_abs_cents":float(np.mean([r['median_abs_cents'] for r in equal])) if equal else None,
            "equal_speaker_mean_within50_fraction":float(np.mean([r['within50_fraction'] for r in equal])) if equal else None})
    result={"schema":"phenometrix-training-lar-alignment-sensitivity-v1","created_at":utc_now(),
        "status":"diagnostic_only_no_primary_timing_or_score_change",
        "offsets_ms":OFFSETS_MS,"supports_ms":SUPPORTS_MS,"policy":POLICY,
        "source_sha256":_sha(Path(__file__)),"cohort_sha256":_sha(cohort_path),
        "cohort_recording_ids":ids,"source_files":source_files,
        "per_record":per_record,"per_speaker":per_speaker,"aggregate":aggregate,
        "limitations":["Strong periodic cycles are a selected subset, not an infallible reference standard.",
          "A cycle interval estimates period-average frequency; it is not pointwise instantaneous F0.",
          "Window-support averaging can mimic a clock shift; offset agreement alone does not establish provider filter-delay compensation.",
          "No offsets selected per recording, no primary scores shifted, and no development/sealed outcomes accessed."]}
    json_write(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--cohort',type=Path,default=DEFAULT_COHORT)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();result=run(args.root,args.cohort,args.output)
    print(json.dumps({'output':str(args.output),'source_sha256':result['source_sha256'],
        'cohort_records':len(result['cohort_recording_ids']),
        'centered32':[r for r in result['aggregate'] if r['mode']=='centered32']},indent=2))


if __name__=='__main__':
    main()
