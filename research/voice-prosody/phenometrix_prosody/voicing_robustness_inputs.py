"""Versioned, label-independent acoustic bank; no model or reference inputs."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.signal import lfilter

NAMESPACE = 'phenometrix-voicing-robustness-v1-20261025'
SEEDS = (20261025, 20261026, 20261027)
ARMS = ('C', 'R')
ANCHORS = ('clean', 'noise10db', 'first_difference', 'clip01')
DIAGNOSTICS = ('gain_m12', 'gain_m3', 'emphasis_05', 'emphasis_095', 'white_snr8', 'ar_snr18')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seed_for(source_id, slot):
    """UTF-8 namespace:source_id:slot; first eight SHA256 bytes, little endian."""
    return int.from_bytes(hashlib.sha256(f'{NAMESPACE}:{source_id}:{slot}'.encode()).digest()[:8], 'little')


def draw(source_id, slot):
    if slot not in range(4):
        raise ValueError('Four fixed continuous slots required')
    seed = seed_for(source_id, slot)
    rng = np.random.default_rng(seed)
    p = dict(slot=slot, seed=seed, gain_db=float(rng.uniform(-18, 0)), filter='none', noise='none')
    family = None if slot == 0 else 'emphasis' if slot == 1 else 'lowpass' if slot == 2 else ('emphasis' if rng.random() < .5 else 'lowpass')
    if family == 'emphasis':
        p.update(filter=family, a=float(rng.uniform(.2, 1)))
    if family == 'lowpass':
        fc = float(np.exp(rng.uniform(np.log(1500), np.log(10000))))
        p.update(filter=family, cutoff_hz=fc, b=float(np.exp(-2*np.pi*fc/48000)))
    if slot == 3:
        p.update(noise='white' if rng.random() < .5 else 'ar1', snr_db=float(rng.uniform(5, 25)), ar_coefficient=.8)
    # A separate domain prevents parameter draw count from changing noise samples.
    p['noise_seed'] = seed_for(source_id, f'noise-{slot}')
    return p


def diagnostic_draw(source_id, condition):
    if condition not in DIAGNOSTICS:
        raise ValueError('Unknown fixed diagnostic')
    p = dict(condition=condition, gain_db=0., filter='none', noise='none', noise_seed=seed_for(source_id, 'diagnostic-'+condition))
    if condition.startswith('gain_'):
        p['gain_db'] = -12. if condition == 'gain_m12' else -3.
    elif condition.startswith('emphasis_'):
        p.update(filter='emphasis', a=.5 if condition == 'emphasis_05' else .95)
    else:
        p.update(noise='white' if condition == 'white_snr8' else 'ar1', snr_db=8. if condition == 'white_snr8' else 18., ar_coefficient=.8)
    return p


def channel(pcm, p):
    x = np.asarray(pcm, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Nonempty finite mono PCM required')
    if p['filter'] == 'emphasis':
        a = p['a']; y = lfilter([1/(1+a), -a/(1+a)], [1], x)
    elif p['filter'] == 'lowpass':
        b = p['b']; y = lfilter([1-b], [1, -b], x)
    elif p['filter'] == 'none':
        y = x.copy()
    else:
        raise ValueError('Unknown filter')
    return y * 10**(p['gain_db']/20)


def construct(pcm, p):
    """Whole source filter then gain then noise, Float64 arithmetic/Float32 PCM."""
    y = channel(pcm, p)
    power = float(np.mean(y*y))
    noise = np.zeros(len(y), np.float64)
    scale = 0.
    if p['noise'] != 'none':
        noise = np.random.default_rng(p['noise_seed']).standard_normal(len(y))
        if p['noise'] == 'ar1':
            noise = lfilter([math.sqrt(1-.8**2)], [1, -.8], noise)
        elif p['noise'] != 'white':
            raise ValueError('Unknown noise')
        scale = math.sqrt(power/(10**(p['snr_db']/10)*float(np.mean(noise*noise)))) if power else 0.
        noise *= scale
    final = (y+noise).astype('<f4')
    if not np.isfinite(final).all():
        raise ValueError('Nonfinite transformed PCM')
    actual = final.astype(np.float64)-y
    achieved = float(10*np.log10(power/np.mean(actual*actual))) if power and p['noise'] != 'none' else None
    receipt = dict(parameters=p, sample_count=len(final), pcm_sha256=hashlib.sha256(final.tobytes()).hexdigest(),
        minimum=float(final.min()), maximum=float(final.max()), outside_unit_count=int((np.abs(final)>1).sum()),
        signal_power_before_noise=power, noise_scale=scale, requested_snr_db=p.get('snr_db'), achieved_snr_db=achieved,
        construction_dtype='float64', stored_dtype='little-endian float32', saturation=False,
        scaling='whole realized recording, no reference mask; offline construction only')
    return final, receipt


def bank_recipe():
    return dict(namespace=NAMESPACE, seed_conversion='SHA256 UTF-8 namespace:source_id:slot; first 8 bytes little endian; NumPy PCG64',
        slots=['gain U[-18,0] dB', 'gain + normalized emphasis a U[0.2,1]',
               'gain + causal lowpass fc log-U[1500,10000] Hz', 'gain + equiprobable filter + equiprobable white/AR(.8) noise SNR U[5,25]'],
        order='filter, gain, noise; all state zero at source start', arithmetic='float64 -> little-endian float32, no clipping',
        anchors=list(ANCHORS), diagnostics=list(DIAGNOSTICS), seeds=list(SEEDS), arms=list(ARMS), candidate='R-20261025')
