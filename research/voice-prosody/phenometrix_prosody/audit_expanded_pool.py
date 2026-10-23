"""Independent arithmetic for the bounded six-versus-seven candidate study.

Never fits a model or generates a waveform. Historical sources are immutable.
Scores, grouping, targets and gates are reconstructed separately from the runner.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import audit_selector_experiment as base
from . import audit_harmonic_experiment as historical
from . import audit_augmentation_experiment as augmentation
from . import audit_candidate_coverage as retention
from .model import FEATURE_NAMES_V2

REPO, ROOT, RUNS = base.REPO, base.ROOT, base.RUNS
SEEDS = (20261022, 20261023, 20261024)
ARMS = ('original6', 'expanded7')
PIPELINES = ARMS + ('frozen-original6', 'frozen-expanded7')
SELECTORS = [f'{arm}__seed{seed}' for arm in PIPELINES for seed in SEEDS]
METHODS = ['original_mlp', 'original_hybrid', *SELECTORS]
CANDIDATE = [f'expanded7__seed{seed}' for seed in SEEDS]
GROUP_COUNTS = {'regression': 14, 'consumed_harmonic': 12,
    'consumed_subharmonic': 12, 'consumed_augmentation': 12,
    'consumed_coverage': 12, 'challenge': 12}
GATE_COUNTS = {'stationary': 3, 'harmonic_change': 3,
    **{g: 12 for g in GROUP_COUNTS if g != 'regression'}}
NATURAL_ROWS, SYNTHETIC_ROWS = 132466, 18336
EPOCHS, NATURAL_BATCH, SYNTHETIC_BATCH, STEPS = 12, 4096, 512, 33
PRIOR = RUNS / 'harmonic-augmentation-v1'
COVERAGE_PRIOR = RUNS / 'candidate-coverage-v1'
PRIOR_SHA = 'd5513b83672f14a711f23a36e5258d348ac9b01a062c5945faaeaf5730d69426'
COVERAGE_PRIOR_SHA = '1e2d7fbec0cb2dd71b177a0a5b646eff04d1d6dcf72a1c86eaac8f42f267aeff'
require = base.require
compare_nested = retention.compare_nested
load_npz = historical.load_npz
FEATURE_NAMES_EXPANDED = tuple(n.replace('candidate_0_', 'candidate_'+str(i)+'_') for i in range(7) for n in FEATURE_NAMES_V2[:8]) + FEATURE_NAMES_V2[-9:-2] + ('valid_candidate_count_over_six', FEATURE_NAMES_V2[-1])
FEATURE_KEYS = {'features', 'candidate_hz', 'candidate_mask', 'reference_sample',
    'feature_names', 'sample_rate_hz', 'pcm_sample_count'}


def validate_pool(feature, k):
    """Validate distinct raw frequencies and source evidence, including /6 units."""
    require(k in (6, 7) and set(feature) == FEATURE_KEYS, 'Pool schema differs')
    x, hz, mask = (np.asarray(feature[n]) for n in ('features', 'candidate_hz', 'candidate_mask'))
    n = len(x)
    require(x.shape == (n, 8*k+9) and x.dtype == np.float32, 'Pool feature shape/dtype differs')
    require(hz.shape == mask.shape == (n, k) and hz.dtype == np.float64 and mask.dtype == np.bool_,
        'Pool frequency/mask shape or dtype differs')
    require(np.isfinite(x).all() and np.isfinite(hz[mask]).all() and np.isnan(hz[~mask]).all(),
        'Pool contains invalid values')
    require(np.all((hz[mask] >= 50) & (hz[mask] <= 700)), 'Candidate outside inherited range')
    block = x[:, :8*k].reshape(n, k, 8)
    require(np.array_equal(block[..., 0], mask.astype(np.float32)) and not np.any(block[~mask]),
        'Invalid candidate validity/zero padding')
    require(np.allclose(block[..., 1][mask], np.log2(hz[mask]/50), rtol=0, atol=1e-6),
        'Candidate log frequency differs')
    require(np.all((block[..., 2:7] >= 0) & (block[..., 2:7] <= 1)) and
        np.all(np.any(block[..., 2:7] > 0, axis=-1)[mask]), 'Invalid source evidence')
    require(np.array_equal(block[..., 7], block[..., 2:7].max(-1)), 'Maximum source evidence differs')
    require(np.array_equal(x[:, 8*k+7], (mask.sum(-1)/6).astype(np.float32)), 'Count units changed from /6')
    for frequencies, support in zip(hz, mask):
        require(np.array_equal(support, np.arange(k) < support.sum()) and np.all(np.diff(frequencies[support]) > 0),
            'Raw candidates must be packed, sorted, and distinct')
    require(np.array_equal(feature['feature_names'], FEATURE_NAMES_V2 if k == 6 else FEATURE_NAMES_EXPANDED), 'Feature name inventory differs')
    ends = np.asarray(feature['reference_sample'])
    count, rate = np.asarray(feature['pcm_sample_count']), np.asarray(feature['sample_rate_hz'])
    require(ends.shape == (n,) and ends.dtype.kind in 'iu' and
        np.array_equal(ends, 2208+480*np.arange(n)), 'Acquisition endpoints differ')
    require(count.shape == rate.shape == () and count.dtype.kind in 'iu' and rate.dtype.kind in 'iu'
        and int(rate) == 48000 and int(count) > 0 and (not n or ends[-1] <= int(count)),
        'Acquisition extent/rate differs')
    return block


def validate_expansion(original, expanded, provenance, pcm_sha):
    """Rebuild a union, never trust a claimed subset or silently replace source3."""
    old = validate_pool(original, 6); new = validate_pool(expanded, 7)
    for field in ('reference_sample', 'sample_rate_hz', 'pcm_sample_count'):
        require(np.array_equal(original[field], expanded[field]), 'Paired acquisition identity differs: '+field)
    n = len(old)
    require(len(new) == n, 'Expanded frame count differs')
    keep = [0, 1, 2, 3, 4, 5, 6, 8]
    require(np.array_equal(original['features'][:, 48:][:, keep], expanded['features'][:, 56:][:, keep]),
        'Expansion changed a non-count global field')
    fields = ('old_full40_hz', 'old_full40_confidence', 'new_full40_hz', 'new_full40_confidence')
    require(all(field in provenance and np.asarray(provenance[field]).shape == (n,) for field in fields),
        'Peak provenance fields missing')
    expected = set(fields) | {'candidate_policy', 'feature_schema', 'pcm_sha256',
        'original_to_expanded_slot', 'old_candidate_count', 'new_candidate_count', 'new_frequency_added',
        'existing_full40_evidence_enhanced', 'old_frequency_evicted', 'worker_rss_bytes'}
    require(set(provenance) == expected, 'Expanded provenance inventory differs')
    require(str(provenance['pcm_sha256']) == pcm_sha and
        str(provenance['candidate_policy']) == 'add-shortest-admitted-full40-v1' and
        str(provenance['feature_schema']) == 'phenometrix.voice-pitch-features.expanded7.v1', 'Peak policy/schema/PCM differs')
    require(provenance['original_to_expanded_slot'].shape == (n, 6) and
        provenance['original_to_expanded_slot'].dtype == np.int64, 'Original slot map differs')
    for field in ('old_candidate_count', 'new_candidate_count', 'new_frequency_added',
                  'existing_full40_evidence_enhanced', 'old_frequency_evicted'):
        require(provenance[field].shape == (n,), 'Expansion provenance extent differs')
    require(0 < int(provenance['worker_rss_bytes']) <= 2*1024**3, 'Peak worker resource evidence invalid')
    added = duplicate = absent = 0
    for i in range(n):
        entries = {float(h): b.copy() for h, b, valid in zip(original['candidate_hz'][i], old[i], original['candidate_mask'][i]) if valid}
        old_h, new_h, old_c, new_c = (provenance[f][i] for f in ('old_full40_hz', 'new_full40_hz', 'old_full40_confidence', 'new_full40_confidence'))
        old_peak = [h for h, block in entries.items() if block[5] > 0]
        require(np.isfinite(old_h) == np.isfinite(new_h), 'Extra policy changed peak availability')
        is_added = bool(np.isfinite(new_h) and new_h not in entries)
        is_enhanced = bool(np.isfinite(new_h) and new_h in entries and np.float32(new_c) > entries[new_h][5])
        if np.isfinite(old_h):
            require(old_peak == [old_h] and entries[old_h][5] == np.float32(old_c), 'Original full40 evidence differs')
            require(50 <= old_h <= new_h <= 700 and .35 <= new_c <= old_c <= 1 and (new_h != old_h or new_c == old_c),
                'Extra peak is not a valid shortest admitted full40 peak')
            if new_h not in entries:
                entries[new_h] = np.array([1, np.log2(new_h/50), 0, 0, 0, 0, 0, 0], np.float32)
                added += 1
            else:
                duplicate += 1
            entries[new_h][5] = max(entries[new_h][5], np.float32(new_c))
        else:
            require(not old_peak and old_c == new_c == 0, 'Absent full40 peak carries evidence')
            absent += 1
        require(len(entries) <= 7, 'Expanded raw pool exceeds seven')
        hz, mask, blocks = np.full(7, np.nan), np.zeros(7, bool), np.zeros((7, 8), np.float32)
        for j, (frequency, block) in enumerate(sorted(entries.items())):
            hz[j], mask[j] = frequency, True
            block[7] = block[2:7].max(); blocks[j] = block
        mapping = np.full(6, -1, np.int64)
        for slot in np.flatnonzero(original['candidate_mask'][i]):
            mapping[slot] = np.flatnonzero(mask & (hz == original['candidate_hz'][i, slot]))[0]
        require(np.array_equal(mapping, provenance['original_to_expanded_slot'][i]), 'Original slot map is not exact')
        require(provenance['old_candidate_count'][i] == original['candidate_mask'][i].sum() and
            provenance['new_candidate_count'][i] == mask.sum(), 'Expanded provenance count differs')
        require(bool(provenance['new_frequency_added'][i]) == is_added and
            bool(provenance['existing_full40_evidence_enhanced'][i]) == is_enhanced and
            not bool(provenance['old_frequency_evicted'][i]), 'Expanded provenance flag differs')
        require(np.array_equal(expanded['candidate_hz'][i], hz, equal_nan=True) and
            np.array_equal(expanded['candidate_mask'][i], mask), 'Expanded pool is not the declared raw union')
        require(np.array_equal(new[i][:, [0, 2, 3, 4, 5, 6, 7]], blocks[:, [0, 2, 3, 4, 5, 6, 7]]) and
            np.allclose(new[i, :, 1], blocks[:, 1], rtol=0, atol=1e-6), 'Expanded candidate evidence differs')
    return {'frames': n, 'added_distinct_candidate_frames': added, 'already_present_peak_frames': duplicate,
        'absent_full40_frames': absent, 'raw_candidate_evictions': 0}


def independent_groups(feature):
    """Greedy complete-span groups and ORIGINAL-member representatives, in NumPy."""
    k = np.asarray(feature['candidate_hz']).shape[-1]
    raw = validate_pool(feature, k)
    x, hz, mask = feature['features'], feature['candidate_hz'], feature['candidate_mask']
    out = x.copy(); out[:, :8*k] = 0
    frequency = np.full_like(hz, np.nan); support = np.zeros_like(mask)
    members = np.zeros((len(x), k, k), bool)
    for row in range(len(x)):
        groups = []
        for index in np.flatnonzero(mask[row]):
            if not groups or 1200*np.log2(hz[row, index]/hz[row, groups[-1][0]]) > 5:
                groups.append([int(index)])
            else:
                groups[-1].append(int(index))
        for slot, indices in enumerate(groups):
            rep = min(indices, key=lambda i: (-float(raw[row, i, 7]), hz[row, i]))
            scores = raw[row, indices, 2:7].max(0)
            frequency[row, slot] = hz[row, rep]; support[row, slot] = True
            out[row, slot*8:(slot+1)*8] = [1, np.log2(hz[row, rep]/50), *scores, scores.max()]
            members[row, slot, indices] = True
        out[row, 8*k+7] = len(groups)/6
    return {'features': out, 'candidate_hz': frequency, 'candidate_mask': support, 'member_mask': members}


def acceptable_labels(hz, mask, reference, valid, voiced):
    hz, mask = np.asarray(hz, np.float64), np.asarray(mask, bool)
    reference, valid, voiced = np.asarray(reference, np.float64), np.asarray(valid, bool), np.asarray(voiced, bool)
    require(hz.ndim == 2 and mask.shape == hz.shape and reference.shape == valid.shape == voiced.shape == hz.shape[:1],
        'Target shape differs')
    include = valid & voiced
    require(np.isfinite(reference[include]).all() and np.all(reference[include] > 0), 'Included target is invalid')
    require(np.isfinite(hz[mask]).all() and np.all(hz[mask] > 0), 'Supported emission is invalid')
    errors = np.abs(1200*np.log2(np.where(mask, hz, 1)/np.where(include, reference, 1)[:, None]))
    acceptable = mask & include[:, None] & (errors <= 50)
    return acceptable, acceptable.any(-1)


def union_training_targets(features, grouped, reference, valid, voiced, *, expected_rows=NATURAL_ROWS):
    """Union is formed on RAW coverage; grouped-uncovered rows are kept as zero loss."""
    require(set(features) == set(grouped) == set(ARMS), 'Training pool arms differ')
    raw = {arm: acceptable_labels(f['candidate_hz'], f['candidate_mask'], reference, valid, voiced)[1]
        for arm, f in features.items()}
    require(not np.any(raw['original6'] & ~raw['expanded7']), 'Raw union evicted original coverage')
    schedule = np.flatnonzero(raw['original6'] | raw['expanded7'])
    require(len(schedule) == expected_rows, 'Union training schedule differs from predeclared count')
    target = {'schedule': schedule}
    for arm, group in grouped.items():
        acceptable, covered = acceptable_labels(group['candidate_hz'], group['candidate_mask'], reference, valid, voiced)
        require(not np.any(covered & ~raw[arm]), 'Grouping emitted a non-member covered target')
        target[arm+'__acceptable'] = acceptable[schedule]
        target[arm+'__covered'] = covered[schedule]
    return target


def natural_order(seed, epoch, rows=NATURAL_ROWS):
    require(seed in SEEDS and 0 <= epoch < EPOCHS, 'Undeclared natural schedule')
    require(rows == NATURAL_ROWS and math.ceil(rows/NATURAL_BATCH) == STEPS, 'Natural step count changed')
    return np.random.default_rng(seed+epoch).permutation(rows).astype(np.int64)


def synthetic_order(seed, epoch):
    require(seed in SEEDS and 0 <= epoch < EPOCHS, 'Undeclared synthetic schedule')
    return np.random.default_rng(seed+1_000_000+epoch).permutation(SYNTHETIC_ROWS)[:STEPS*SYNTHETIC_BATCH].reshape(STEPS, SYNTHETIC_BATCH).astype(np.int64)


def synthetic_exposure(order, covered, scenario_index, region):
    require(order.shape == (STEPS, SYNTHETIC_BATCH) and order.dtype == np.int64, 'Synthetic schedule shape/dtype differs')
    require(len(np.unique(order)) == order.size and order.min() >= 0 and order.max() < SYNTHETIC_ROWS,
        'Synthetic schedule repeats or escapes rows')
    covered, scenario_index, region = np.asarray(covered), np.asarray(scenario_index), np.asarray(region)
    require(covered.shape == scenario_index.shape == region.shape == (SYNTHETIC_ROWS,), 'Exposure metadata extent differs')
    require(covered.dtype == np.bool_ and scenario_index.dtype.kind in 'iu' and np.all((scenario_index >= 0) & (scenario_index < 96)),
        'Exposure metadata values differ')
    selected = order.reshape(-1)
    return {'selected_rows':len(selected), 'unique_rows':len(np.unique(selected)),
        'covered_rows':int(covered[selected].sum()), 'uncovered_zero_rows':int((~covered[selected]).sum()),
        'per_case':[{'case_index':i, 'selected_rows':int((scenario_index[selected]==i).sum()),
            'covered_rows':int(((scenario_index[selected]==i)&covered[selected]).sum())} for i in range(96)],
        'per_region':{str(r):{'selected_rows':int((region[selected]==r).sum()),
            'covered_rows':int(((region[selected]==r)&covered[selected]).sum())} for r in sorted(set(region))}}


def independent_logits(weights, grouped):
    """Explicit CPU equations: dynamic candidate axes, unchanged learned tensors."""
    x = torch.from_numpy(grouped['features'])
    mask = torch.from_numpy(grouped['candidate_mask'])
    k = mask.shape[-1]
    require(k in (6, 7) and x.shape[-1] == k*8+9, 'Inference pool shape differs')
    block = x[:, :8*k].reshape(-1, k, 8)
    standardized = (block-weights['candidate_mean'])/weights['candidate_scale']
    embed = torch.tanh(F.linear(standardized, weights['candidate_encoder.weight'], weights['candidate_encoder.bias']))
    ratio = (block[..., 1].unsqueeze(-2)-block[..., 1].unsqueeze(-1))/math.log2(14)
    edges = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    neighbors = standardized.unsqueeze(-3).expand(-1, k, k, 8)
    message_input = torch.cat((neighbors, torch.where(edges, ratio, torch.zeros_like(ratio)).unsqueeze(-1)), -1)
    message_input = torch.where(edges.unsqueeze(-1), message_input, torch.zeros_like(message_input))
    messages = torch.tanh(F.linear(message_input, weights['message_encoder.weight'], weights['message_encoder.bias']))
    context = torch.where(edges.unsqueeze(-1), messages, torch.full_like(messages, -torch.inf)).max(-2).values
    context = torch.where(edges.any(-1, keepdim=True), context, torch.zeros_like(context))
    globals_ = ((x[:, 8*k:]-weights['global_mean'])/weights['global_scale']).unsqueeze(-2).expand(-1, k, 9)
    hidden = torch.tanh(F.linear(torch.cat((embed, context, globals_), -1), weights['shared_hidden.weight'], weights['shared_hidden.bias']))
    logits = F.linear(hidden, weights['shared_head.weight'], weights['shared_head.bias']).squeeze(-1)
    return torch.where(mask, logits, torch.full_like(logits, -1e9))


def natural_gate(cells):
    require(set(cells) == set(METHODS), 'Natural method inventory differs')
    expected = {s+'/'+c for s in base.SPEAKERS['development'] for c in base.CONDITIONS}
    require(all(set(v) == expected for v in cells.values()), 'Natural cell inventory differs')
    mean = lambda m, key: float(np.mean([v[key] for v in cells[m].values()]))
    error = float(np.mean([mean(m, 'joint_error') for m in CANDIDATE]))
    success = float(np.mean([mean(m, 'reference_voiced_success_rate') for m in CANDIDATE]))
    gains = {m: mean(m, 'joint_error')-error for m in ('original_mlp', 'original_hybrid')}
    excess = {c: float(np.mean([v['joint_error'] for m in CANDIDATE for k, v in cells[m].items() if k.endswith('/'+c)]))-
        float(np.mean([v['joint_error'] for k, v in cells['original_mlp'].items() if k.endswith('/'+c)])) for c in base.CONDITIONS}
    checks = {'mean_improves_both_by_half_point': all(v >= .005 for v in gains.values()),
        'each_seed_improves_original_mlp': all(mean(m, 'joint_error') < mean('original_mlp', 'joint_error') for m in CANDIDATE),
        'no_condition_worsens_over_one_point': all(v <= .01 for v in excess.values()),
        'voiced_success_not_lower': success >= mean('original_mlp', 'reference_voiced_success_rate')}
    return {'predesignated_candidate': 'expanded7', 'pass': all(checks.values()), 'checks': checks,
        'mean_joint_error': error, 'mean_voiced_success': success, 'joint_error_improvement_over_frozen': gains,
        'condition_error_excess_over_mlp': excess,
        'improvement_over_paired_original': float(np.mean([mean(f'original6__seed{s}', 'joint_error') for s in SEEDS]))-error,
        'improvement_over_frozen_expanded': float(np.mean([mean(f'frozen-expanded7__seed{s}', 'joint_error') for s in SEEDS]))-error,
        'original_control_drift_from_historical': float(np.mean([mean(f'original6__seed{s}', 'joint_error')-
            mean(f'frozen-original6__seed{s}', 'joint_error') for s in SEEDS]))}


def finish_case_gates(cases):
    """Each of 66 mandatory cases must independently pass for all three seeds."""
    require(set(cases) == set(GATE_COUNTS), 'Analytic gate group inventory differs')
    result = {}
    for group, count in GATE_COUNTS.items():
        require(set(cases[group]) == set(CANDIDATE), 'Analytic gate selector inventory differs')
        require(all(len(v) == count for v in cases[group].values()), 'Analytic per-seed gate count differs')
        require(len({tuple(sorted(v)) for v in cases[group].values()}) == 1, 'Analytic seeds use different cases')
        values = {method: {case: bool(success >= .95) for case, success in rows.items()}
            for method, rows in cases[group].items()}
        require(all(np.isfinite(v) and 0 <= v <= 1 for rows in cases[group].values() for v in rows.values()),
            'Analytic gate rate invalid')
        result[group+'_by_seed_case'] = values
        result[group+'_pass'] = all(all(v.values()) for v in values.values())
    return result


def verify_checkpoints(audit, run, manifest):
    """Read six fits plus three fixed controls, check copied scaling and initialization."""
    from .harmonic_selector_model import HarmonicSelector
    prior = audit.json(PRIOR/'manifest.json', PRIOR_SHA)
    require(prior['status'] == 'complete', 'Historical augmentation is incomplete')
    require(set(manifest['fits']) == {f'{arm}__seed{s}' for arm in ARMS for s in SEEDS}, 'Fit inventory differs')
    frozen = {}; models = {}; scaling = None; initial = {}
    shapes = {'scaling_fitted': (), 'candidate_mean': (8,), 'candidate_scale': (8,),
        'global_mean': (9,), 'global_scale': (9,)}
    for name, shape in {'candidate_encoder': (32,8), 'shared_hidden': (32,73),
                       'shared_head': (1,32), 'message_encoder': (32,9)}.items():
        shapes[name+'.weight'] = shape; shapes[name+'.bias'] = (shape[0],)
    buffer_names = ('scaling_fitted', 'candidate_mean', 'candidate_scale', 'global_mean', 'global_scale')
    def validate(saved):
        require(saved['context_mode'] == 'relative' and saved['config'] == {
            'representation':'shared-candidates', 'loss':'acceptable-set', 'hidden_size':32}, 'Checkpoint configuration differs')
        state = saved['state_dict']; require(set(state) == set(shapes), 'Checkpoint tensor inventory differs')
        for name, shape in shapes.items():
            tensor = state[name]
            require(tuple(tensor.shape) == shape and bool(torch.isfinite(tensor).all()), 'Checkpoint tensor shape/finite check failed')
            require(tensor.dtype == (torch.bool if name == 'scaling_fitted' else torch.float32), 'Checkpoint dtype differs')
        require(bool(state['scaling_fitted']) and bool((state['candidate_scale'] > 0).all()) and bool((state['global_scale'] > 0).all()),
            'Checkpoint scaling invalid')
        require(sum(t.numel() for n,t in state.items() if n.endswith(('.weight','.bias'))) == 3009, 'Model parameter budget differs')
        return state
    for seed in SEEDS:
        name = f'balance-aug__seed{seed}'; digest = retention.CHECKPOINT_SHA[name]
        require(prior['fits'][name]['weights_sha256'] == digest, 'Historical weights identity differs')
        saved = torch.load(audit.bind(PRIOR/'checkpoints'/(name+'.pt'), digest), map_location='cpu', weights_only=True)
        require(saved['arm'] == 'balance-aug', 'Historical checkpoint arm differs')
        state = validate(saved); frozen[seed] = state
        if scaling is None:scaling = {n:state[n].clone() for n in buffer_names}
        require(all(torch.equal(scaling[n],state[n]) for n in buffer_names), 'Historical six-pool scaling differs across seeds')
        for arm in ARMS:models[f'frozen-{arm}__seed{seed}'] = state
        torch.manual_seed(seed); initialized = HarmonicSelector('relative', 'acceptable-set')
        with torch.no_grad():
            for name in buffer_names:getattr(initialized,name).copy_(scaling[name])
        h = hashlib.sha256()
        for name,tensor in sorted(initialized.state_dict().items()):h.update(name.encode());h.update(tensor.numpy().tobytes())
        initial[seed] = h.hexdigest()
        for arm in ARMS:
            name = f'{arm}__seed{seed}'; fit = manifest['fits'][name]
            require(fit['seed'] == seed and fit['arm'] == arm, 'Fit seed/arm differs')
            require(fit['optimizer_updates'] == 396 and fit['skipped_batches'] == 0 and fit['parameter_count'] == 3009,
                'Fit update/skip/parameter budget differs')
            for key in ('epoch_losses','epoch_base_losses','epoch_weighted_auxiliary_losses'):
                require(len(fit[key]) == 12 and all(math.isfinite(v) and v >= -1e-7 for v in fit[key]), 'Fit loss record invalid')
            require(np.allclose(fit['epoch_losses'],np.array(fit['epoch_base_losses'])+np.array(fit['epoch_weighted_auxiliary_losses']),
                rtol=0,atol=1e-7), 'Fit loss decomposition differs')
            require(fit['initial_tensors_sha256'] == initial[seed], 'Independently reconstructed initialization differs')
            saved = torch.load(audit.bind(run/'checkpoints'/(name+'.pt'),fit['weights_sha256']),map_location='cpu',weights_only=True)
            require(saved['arm'] == arm and saved['candidate_capacity'] == (6 if arm == 'original6' else 7),
                'Saved fitted arm/capacity differs')
            state = validate(saved)
            require(all(torch.equal(state[n],scaling[n]) for n in buffer_names), 'Fit changed original-six TRAIN scaling')
            models[name] = state
    return models, prior, {'new_checkpoints':6,'frozen_checkpoints':3,'parameters_each':3009,
        'independently_reconstructed_initial_tensors_sha256':initial,'copied_original_six_scaling_exact':True}


def check_inference(models, feature, probability, arrays, grouped):
    """Run each selected checkpoint through independent equations in bounded CPU chunks."""
    require(set(models) == set(arrays), 'Inference model/output inventory differs')
    require(np.isfinite(probability).all() and np.all((probability>=0)&(probability<=1)), 'Original voicing probability invalid')
    expected_state = np.where(probability < .5, 0, np.where(grouped['candidate_mask'].any(-1),1,2))
    total = 0
    for method, state_dict in models.items():
        state,hz = arrays[method]
        require(np.array_equal(state,expected_state), 'Pool-specific availability or fixed voicing differs: '+method)
        for start in range(0,len(probability),4096):
            sl = slice(start,start+4096)
            group = {k:v[sl] for k,v in grouped.items()}
            with torch.inference_mode():logits = independent_logits(state_dict,group)
            inferred_state,inferred_hz = retention.equations.emit(logits,group['candidate_hz'],group['candidate_mask'],probability[sl])
            require(np.array_equal(inferred_state,state[sl]) and np.array_equal(inferred_hz,hz[sl],equal_nan=True),
                'Independent checkpoint output differs: '+method)
        total += len(state)
    return total


def pool_of(method):
    pool = method.split('__seed')[0].removeprefix('frozen-')
    require(pool in ARMS,'Unknown selector pipeline')
    return pool


def analytic_coverage(features,groups,truth,eligible,region,probability,arrays):
    flags = {arm:{'fixed-slots':retention.covered_by_candidates(f['candidate_hz'],f['candidate_mask'],truth,eligible),
        'shared-candidates':retention.covered_by_candidates(groups[arm]['candidate_hz'],groups[arm]['candidate_mask'],truth,eligible)}
        for arm,f in features.items()}
    require(not np.any(flags['original6']['fixed-slots'] & ~flags['expanded7']['fixed-slots']), 'Analytic raw coverage was evicted')
    report = {}
    for name,include in {'all':eligible,**{r:eligible&(region==r) for r in sorted(set(region[eligible]))}}.items():
        item = {'frames':int(include.sum()),'policies':{},'policy_change':{},
            'original_voiced_gate_positive':int((include&(probability>=.5)).sum()),'methods':{}}
        for arm in ARMS:
            raw,grouped = flags[arm]['fixed-slots'],flags[arm]['shared-candidates']
            item['policies'][arm] = {'raw_covered':int((include&raw).sum()),'grouped_covered':int((include&grouped).sum()),
                'grouping_lost':int((include&raw&~grouped).sum()),
                'candidate_unavailable':int((include&~groups[arm]['candidate_mask'].any(-1)).sum())}
        for rep in ('fixed-slots','shared-candidates'):
            old,new = flags['original6'][rep],flags['expanded7'][rep]
            item['policy_change'][rep] = {'lost':int((include&old&~new).sum()),'gained':int((include&~old&new).sum()),
                'both':int((include&old&new).sum()),'neither':int((include&~old&~new).sum())}
        for method,(state,_) in arrays.items():
            item['methods'][method] = {label:int((include&(state==value)).sum()) for value,label in enumerate(('unvoiced','voiced','unavailable'))}
        report[name] = item
    return report


def fresh_records():
    return [{'id':f'expanded-balance-f{i}-p{j}','frequency_hz':frequency,'phase_radians':list(phases),
        'gain':.052,'sample_rate_hz':48000,'sample_count':96000,'change_sample':39840,
        'before':[1.,.5,.25],'after':[0.,1.,.5]}
        for i,frequency in enumerate((103.7,229.1,361.7,541.9,633.1,683.7))
        for j,phases in enumerate(((.53,-.97,1.47),(2.03,.71,-1.29)))]


def natural_and_coverage(audit,run,manifest,metrics,models,prior):
    predictions={name:load_npz(audit,run/'predictions'/(name+'.npz'),manifest['prediction_sha256']['predictions/'+name+'.npz']) for name in METHODS}
    for values in predictions.values():require(set(values)=={'state','hz'},'Natural prediction fields differ')
    for seed in SEEDS:
        name=f'original__seed{seed}';p='predictions/'+name+'.npz'
        old=load_npz(audit,COVERAGE_PRIOR/p,prior['prediction_sha256'][p])
        require(all(np.array_equal(old[k],predictions[f'frozen-original6__seed{seed}'][k],equal_nan=True) for k in ('state','hz')),'Original-policy natural control changed')
    pp=run/'probabilities-development.npz';p=load_npz(audit,pp,manifest['input_sha256'][str(pp.relative_to(REPO))])
    require(set(p)=={'original_probability'},'DEV probability fields differ');probability=p['original_probability']
    require(probability.ndim==1 and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all(),'DEV original probability invalid')
    oldfreeze=RUNS/'hybrid-natural-v1/prediction-freeze.json'
    baseline=audit.json(oldfreeze,manifest['input_sha256'][str(oldfreeze.relative_to(REPO))]);baserows={r['recording_id']:r for r in baseline['recordings']}
    totals={m:defaultdict(lambda:{k:0 for k in base.COUNTS}) for m in METHODS}
    all_cov={};headcounts={arm:{rep:defaultdict(lambda:{'eligible':0,'oracle_correct':0}) for rep in ('fixed-slots','shared-candidates')} for arm in ARMS}
    replacement_totals=defaultdict(int);inference_rows=0;exclusions={};training_parts={a:{'raw':[],'acceptable':[],'nearest':[]} for a in ARMS}
    for split in ('train','development'):
        folder,oldcache,metadata=base.load_cache(audit,split)
        cachepath=run/split/'manifest.json';cache=audit.json(cachepath,manifest['input_sha256'][str(cachepath.relative_to(REPO))])
        require(cache['schema']=='phenometrix-expanded-pool-natural-cache.v1' and cache['status']=='complete' and cache['split']==split and cache['reference_content_read'] is False,'Revised natural cache state differs')
        require(cache['source_sha256']==manifest['source_sha256'] and cache['candidate_policy']=='add-shortest-admitted-full40-v1','Revised cache policy/source differs')
        require(cache['original_manifest_path']==str((folder/'manifest.json').relative_to(REPO)) and cache['original_manifest_sha256']==base.sha(folder/'manifest.json'),'Original cache binding differs')
        require(len(cache['records'])==len(oldcache['recordings'])==({'train':960,'development':320}[split]),'Natural cache inventory differs')
        historical_cache=audit.json(COVERAGE_PRIOR/split/'manifest.json',manifest['input_sha256'][str((COVERAGE_PRIOR/split/'manifest.json').relative_to(REPO))])
        historical_rows={r['recording_id']:r for r in historical_cache['records']}
        refs_cache={};covcells={arm:defaultdict(lambda:{k:0 for k in ('voiced','raw','grouped','lost','gained')}) for arm in ARMS}
        changes=defaultdict(lambda:{'voiced':0,**{rep:{k:0 for k in ('lost','gained','both','neither')} for rep in ('fixed-slots','shared-candidates')}})
        offset=0;exclusions[split]={}
        for record,row in zip(oldcache['recordings'],cache['records']):
            for key in ('recording_id','source_recording_id','speaker','condition','sample_count','transformed_pcm_sha256'):
                require(record[key]==row[key],'Paired source/condition identity differs: '+key)
            original=base.load_feature(audit,folder,record)
            require(row['original_feature_path']==str((folder/'features'/(record['recording_id']+'.npz')).relative_to(REPO)) and row['original_feature_sha256']==record['feature_sha256'],'Original feature binding differs')
            revised=load_npz(audit,REPO/row['expanded_feature_path'],row['expanded_feature_sha256']);provenance=load_npz(audit,REPO/row['provenance_path'],row['provenance_sha256'])
            old_row=historical_rows[record['recording_id']]
            require(row['source_provenance_path']==old_row['provenance_path'] and row['source_provenance_sha256']==old_row['provenance_sha256'], 'Historical peak binding differs')
            old_provenance=load_npz(audit,REPO/row['source_provenance_path'],row['source_provenance_sha256'])
            for peak_field in ('old_full40_hz','new_full40_hz','old_full40_confidence','new_full40_confidence'):
                require(np.array_equal(old_provenance[peak_field],provenance[peak_field],equal_nan=True),'Historical peak evidence changed')
            check=validate_expansion(original,revised,provenance,record['transformed_pcm_sha256'])
            for k,v in check.items():replacement_totals[k]+=v
            features={'original6':original,'expanded7':revised};grouped={arm:independent_groups(f) for arm,f in features.items()}
            targets=original['reference_sample']-960;stop=offset+len(targets);sl=slice(offset,stop)
            require(row['start']==offset and row['stop']==stop,'Paired cache row offsets differ')
            source=record['source_recording_id']
            if source not in refs_cache:refs_cache[source]=base.references(audit,metadata[source],record,split)
            refs=refs_cache[source];truth,valid,voiced=retention.ref_arrays(refs,targets);include=valid&voiced
            excludes={'invalid_reference':sum(not r.valid for r in refs),'boundary_reference':sum(r.valid and not r.eligible for r in refs),
                      'predictions_outside_reference_extent':len(set(targets.tolist())-{r.sample for r in refs})}
            exclusions[split][record['recording_id']]=excludes
            cell=record['speaker']+'/'+record['condition'];flags={}
            for arm,f in features.items():
                raw=retention.covered_by_candidates(f['candidate_hz'],f['candidate_mask'],truth,include)
                g=retention.covered_by_candidates(grouped[arm]['candidate_hz'],grouped[arm]['candidate_mask'],truth,include)
                flags[arm]={'fixed-slots':raw,'shared-candidates':g}
                for k,v in retention.coverage_counts(include,raw,g).items():covcells[arm][cell][k]+=v
                if split=='train':
                    acceptable,covered=acceptable_labels(grouped[arm]['candidate_hz'],grouped[arm]['candidate_mask'],truth,valid,voiced)
                    errors=np.where(grouped[arm]['candidate_mask']&include[:,None],
                        np.abs(1200*np.log2(np.where(grouped[arm]['candidate_mask'],grouped[arm]['candidate_hz'],1)/np.where(include,truth,1)[:,None])),np.inf)
                    nearest=np.where(covered,errors.argmin(-1),-1).astype(np.int64)
                    training_parts[arm]['raw'].append(raw);training_parts[arm]['acceptable'].append(acceptable);training_parts[arm]['nearest'].append(nearest)
                if split=='development':
                    for rep,covered in flags[arm].items():
                        oracle=retention.fixed_voicing_oracle(valid,voiced,probability[sl],covered)
                        headcounts[arm][rep][cell]['eligible']+=oracle['eligible_valid_frames']
                        headcounts[arm][rep][cell]['oracle_correct']+=oracle['correct_frames']
            changes[cell]['voiced']+=int(include.sum())
            for rep in ('fixed-slots','shared-candidates'):
                c=retention.policy_change_counts(include,flags['original6'][rep],flags['expanded7'][rep])
                for key in ('lost','gained','neither'):changes[cell][rep][key]+=c[key]
                changes[cell][rep]['both']+=c['retained']
            if split=='development':
                binding=baserows[record['recording_id']]
                cached=json.loads(gzip.decompress(audit.bind(RUNS/'hybrid-natural-v1'/binding['hybrid_prediction_path'],binding['hybrid_prediction_sha256']).read_bytes()))['predictions']
                for m,old_name in [('original_mlp','mlp_original'),('original_hybrid','hybrid_original')]:
                    rows=cached[old_name];require([r['sample'] for r in rows]==targets.tolist(),'Baseline target grid differs')
                    states=np.array([{'unvoiced':0,'voiced':1,'unavailable':2}[r['state']] for r in rows])
                    frequencies=np.array([np.nan if r['f0_hz'] is None else r['f0_hz'] for r in rows])
                    require(np.array_equal(predictions[m]['state'][sl],states) and np.array_equal(predictions[m]['hz'][sl],frequencies,equal_nan=True),'Natural baseline changed')
                require(np.array_equal(probability[sl],[r['voicing_probability'] for r in cached['mlp_original']]),'Original DEV probability changed')
                for arm in ARMS:
                    arrays={name:(predictions[name]['state'][sl],predictions[name]['hz'][sl]) for name in SELECTORS if pool_of(name)==arm}
                    inference_rows+=check_inference({name:models[name] for name in arrays},features[arm],probability[sl],arrays,grouped[arm])
                valid_targets={r.sample for r in refs}
                for name,z in predictions.items():
                    objects=[o for o in base.prediction_objects(targets,z['state'][sl],z['hz'][sl]) if o.sample in valid_targets]
                    got=base.object_score(refs,objects)
                    expected=next(r['metrics'] for r in metrics['natural'][name]['per_record'] if r['recording_id']==record['recording_id'])
                    base.compare_metrics(audit,got,expected,name+'/'+record['recording_id'])
                    for k in base.COUNTS:totals[name][cell][k]+=got[k]
            offset=stop
        expected_cells={s+'/'+c for s in base.SPEAKERS[split] for c in base.CONDITIONS}
        require(set(changes)==expected_cells,'Coverage cell inventory differs')
        all_cov[split]={arm:retention.finish_coverage(covcells[arm]) for arm in ARMS}
        all_cov[split]['policy_change']=retention.finish_transition(changes)
        compare_nested(audit,all_cov[split],metrics['coverage'][split],split+'/coverage')
        if split=='development':require(offset==len(probability) and all(len(v['state'])==offset for v in predictions.values()),'Natural prediction extent differs')
    cells={m:{key:base.count_rates(value) for key,value in rows.items()} for m,rows in totals.items()}
    for name,values in cells.items():
        for cell,value in values.items():base.compare_metrics(audit,value,metrics['natural'][name]['cells'][cell],name+'/'+cell)
        for rate in base.RATES:
            audit.equal(float(np.mean([v[rate] for v in values.values()])),metrics['natural'][name]['macro'][rate],name+'/macro/'+rate)
            for field,keys in [('per_speaker',base.SPEAKERS['development']),('per_condition',base.CONDITIONS)]:
                for key in keys:
                    chosen=[v[rate] for cell,v in values.items() if cell.startswith(key+'/') or cell.endswith('/'+key)]
                    audit.equal(float(np.mean(chosen)),metrics['natural'][name][field][key][rate],name+'/'+field+'/'+key+'/'+rate)
        pooled=base.count_rates({k:sum(v[k] for v in values.values()) for k in base.COUNTS})
        base.compare_metrics(audit,pooled,metrics['natural'][name]['pooled'],name+'/pooled')
    headroom={}
    for arm in ARMS:
        headroom[arm]={}
        for rep,counts in headcounts[arm].items():
            for cell,c in counts.items():
                require(c['eligible']==cells['original_mlp'][cell]['eligible_valid_frames'],'Oracle denominator differs')
                c['mlp_correct']=cells['original_mlp'][cell]['correct_frames'];c['gain']=(c['oracle_correct']-c['mlp_correct'])/c['eligible']
            gain=float(np.mean([v['gain'] for v in counts.values()]));headroom[arm][rep]={'cells':dict(counts),'mean_gain':gain,'pass':gain>=.01}
    compare_nested(audit,headroom,metrics['headroom'],'headroom')
    compare_nested(audit,exclusions,metrics['exclusions'],'natural exclusions')
    raw_flags={arm:np.concatenate(parts['raw']) for arm,parts in training_parts.items()}
    require(not np.any(raw_flags['original6']&~raw_flags['expanded7']),'Expanded raw TRAIN coverage evicted a candidate')
    schedule=np.flatnonzero(raw_flags['original6']|raw_flags['expanded7'])
    require(len(schedule)==NATURAL_ROWS and raw_flags['original6'].sum()==132181,'Training raw-union inventory differs')
    training={'schedule':schedule}
    for arm,parts in training_parts.items():
        acceptable=np.concatenate(parts['acceptable'])[schedule]
        training[arm+'__acceptable']=acceptable;training[arm+'__covered']=acceptable.any(-1)
        training[arm+'__nearest']=np.concatenate(parts['nearest'])[schedule]
    return cells,all_cov,headroom,training,{'methods':14,'development_condition_records':320,'train_condition_records':960,
        'development_rows':len(probability),'checkpoint_frame_pairs':inference_rows,'replacement_reconstruction':dict(replacement_totals)}



def nearest_labels(hz, mask, reference, include):
    errors = np.abs(1200*np.log2(np.where(mask,hz,1)/np.where(include,reference,1)[:,None]))
    errors = np.where(mask&include[:,None],errors,np.inf)
    accepted = (errors <= 50)&mask&include[:,None]
    return accepted,accepted.any(-1),np.where(accepted.any(-1),errors.argmin(-1),-1).astype(np.int64)


def audit_training(audit,run,manifest,natural_targets):
    saved = load_npz(audit,run/'training-targets.npz',manifest['training_target_sha256'])
    require(set(saved)==set(natural_targets),'Natural training target inventory differs')
    for key,value in natural_targets.items():
        require(saved[key].dtype==value.dtype and np.array_equal(saved[key],value),'Natural training target differs: '+key)
    require(manifest['scheduled_rows']==NATURAL_ROWS and
        manifest['grouped_uncovered_scheduled_rows']=={a:int((~saved[a+'__covered']).sum()) for a in ARMS},
        'Natural scheduled/zero-loss count differs')
    old_cache=audit.json(PRIOR/'training/manifest.json');old_spec=audit.json(PRIOR/'training/spec.json')
    expected=augmentation.expected_training_records()['balance-aug']
    require(old_cache['status']=='complete' and old_spec['records_by_arm']['balance-aug']==expected,'Historical synthetic TRAIN configurations differ')
    old_rows={r['id']:r for r in old_cache['records'] if r['arm']=='balance-aug'}
    cache=audit.json(run/'training/manifest.json',manifest['input_sha256'][str((run/'training/manifest.json').relative_to(REPO))])
    require(cache['schema']=='phenometrix-expanded-pool-training-cache.v1' and cache['status']=='complete' and
        cache['source_sha256']==manifest['source_sha256'] and cache['original_spec_sha256']==base.sha(PRIOR/'training/spec.json'),
        'Expanded TRAIN cache provenance differs')
    require([r['id'] for r in cache['records']]==[r['id'] for r in expected] and set(old_rows)=={r['id'] for r in expected},
        'Synthetic training case inventory/order differs')
    parts={a:defaultdict(list) for a in ARMS};addition=defaultdict(int)
    for index,(params,row) in enumerate(zip(expected,cache['records'])):
        old=old_rows[params['id']]
        for key in ('original_feature_path','original_feature_sha256','pcm_path','pcm_sha256'):
            expected_value=str((PRIOR/'training'/old['feature_path' if key.startswith('original_feature') else 'pcm_path']).relative_to(REPO)) if key.endswith('_path') else old['feature_sha256' if key=='original_feature_sha256' else 'pcm_sha256']
            require(row[key]==expected_value,'Synthetic historical binding differs: '+key)
        audit.bind(REPO/row['pcm_path'],row['pcm_sha256'])
        f=load_npz(audit,REPO/row['original_feature_path'],row['original_feature_sha256'])
        expanded=load_npz(audit,REPO/row['expanded_feature_path'],row['expanded_feature_sha256'])
        provenance=load_npz(audit,REPO/row['provenance_path'],row['provenance_sha256'])
        for k,v in validate_expansion(f,expanded,provenance,row['pcm_sha256']).items():addition[k]+=v
        end,target,include=historical.validate_feature(f,96000)
        require(include.sum()==191,'Synthetic TRAIN common denominator differs')
        scenario=row['scenario']
        require(scenario['kind']=='observation-change' and scenario['initialHz']==scenario['finalHz']==params['frequency_hz'] and
            scenario['changeStartSeconds']==scenario['changeEndSeconds']==.83 and scenario['durationSeconds']==2.,
            'Synthetic training scenario differs')
        region=historical.regions(scenario,end)
        for arm,feature in (('original6',f),('expanded7',expanded)):
            group=independent_groups(feature);ix=np.flatnonzero(include)
            reference=np.full(len(ix),params['frequency_hz']);hz=group['candidate_hz'][ix];mask=group['candidate_mask'][ix]
            acceptable,covered,nearest=nearest_labels(hz,mask,reference,np.ones(len(ix),bool))
            values={'acceptable':acceptable,'covered':covered,'nearest':nearest,'reference_hz':reference,
                'scenario_index':np.full(len(ix),index,np.int64),'reference_sample':target[ix],
                'availability_sample':end[ix],'region':region[ix]}
            for key,value in values.items():parts[arm][key].append(value)
    expected_targets={a+'__'+k:np.concatenate(v) for a,fields in parts.items() for k,v in fields.items()}
    saved_synthetic=load_npz(audit,run/'synthetic-targets.npz',manifest['synthetic_target_sha256'])
    require(set(saved_synthetic)==set(expected_targets),'Synthetic target inventory differs')
    for key,value in expected_targets.items():
        require(saved_synthetic[key].dtype==value.dtype and np.array_equal(saved_synthetic[key],value),'Synthetic target differs: '+key)
    for arm in ARMS:
        covered=saved_synthetic[arm+'__covered'];require(len(covered)==SYNTHETIC_ROWS,'Synthetic row count differs')
        compare_nested(audit,{'rows':len(covered),'covered_rows':int(covered.sum()),'uncovered_zero_rows':int((~covered).sum())},
            manifest['synthetic_pool'][arm],arm+'/synthetic pool')
    orders={name:load_npz(audit,run/(name+'-schedules.npz'),manifest['input_sha256'][str((run/(name+'-schedules.npz')).relative_to(REPO))])
        for name in ('natural','synthetic')}
    names={f'seed{s}__epoch{e}' for s in SEEDS for e in range(EPOCHS)}
    require(all(set(v)==names for v in orders.values()),'Paired schedule inventory differs')
    for seed in SEEDS:
        for epoch in range(EPOCHS):
            key=f'seed{seed}__epoch{epoch}';natural=natural_order(seed,epoch);synthetic=synthetic_order(seed,epoch)
            for name,value in (('natural',natural),('synthetic',synthetic)):
                require(orders[name][key].dtype==np.int64 and np.array_equal(orders[name][key],value),'Paired sampling order differs: '+name+'/'+key)
            for arm in ARMS:
                fit=manifest['fits'][f'{arm}__seed{seed}'];covered=saved[arm+'__covered']
                require(all(covered[natural[i:i+NATURAL_BATCH]].any() for i in range(0,len(natural),NATURAL_BATCH)),
                    'A zero-covered natural batch violates the fixed update schedule')
                audit.equal(int(covered[natural].sum()),fit['epoch_natural_covered_rows'][epoch],arm+'/natural epoch denominator')
                exposure=synthetic_exposure(synthetic,saved_synthetic[arm+'__covered'],saved_synthetic[arm+'__scenario_index'],saved_synthetic[arm+'__region'])
                compare_nested(audit,exposure,fit['synthetic_exposure_by_epoch'][epoch],arm+'/synthetic exposure/'+key)
    return {'natural_scheduled_rows':NATURAL_ROWS,'natural_grouped_uncovered_rows':manifest['grouped_uncovered_scheduled_rows'],
        'synthetic_cases':96,'synthetic_rows_per_pool':SYNTHETIC_ROWS,'epochs':EPOCHS,'updates_per_fit':396,
        'synthetic_rows_per_epoch':16896,'natural_schedule_arrays_checked':36,'synthetic_schedule_arrays_checked':36,
        'targets_and_exposure_exact':True,'training_addition':dict(addition)}


def expected_analytic(audit,run,manifest,last):
    previous=retention.expected_analytic(audit,COVERAGE_PRIOR,last)
    records={(('consumed_coverage' if group=='challenge' else group),name):row for (group,name),row in previous.items()}
    folder=run/'challenge';cm=audit.json(folder/'manifest.json');spec=audit.json(folder/'spec.json')
    require(spec['records']==fresh_records() and cm['status']=='complete' and cm['reference_content_read'] is False,
        'Prospective physical configurations/cache state differ')
    require(cm['schema']=='phenometrix-expanded-pool-challenge-cache-v1' and
        hashlib.sha256(historical.canonical(spec)).hexdigest()==cm['spec_sha256']==manifest['challenge_spec_sha256'],
        'Fresh specification binding differs')
    require(cm['freeze_path']==str((run/'checkpoint-freeze.json').relative_to(REPO)) and
        cm['freeze_sha256']==base.sha(run/'checkpoint-freeze.json'),'Fresh generation checkpoint binding differs')
    require(historical.dt(manifest['checkpoints_frozen_at'])<=historical.dt(cm['started_at'])<=historical.dt(cm['completed_at'])<=
        historical.dt(manifest['predictions_frozen_at']),'Fresh challenge chronology differs')
    require([r['id'] for r in cm['records']]==[r['id'] for r in fresh_records()],'Fresh record inventory/order differs')
    for path,digest in cm['source_sha256'].items():
        audit.bind(REPO/path,digest);audit.bind(folder/'source-snapshot'/path,digest)
        require(manifest['source_sha256'].get(path)==digest,'Fresh helper source escaped frozen closure')
    params={r['id']:r for r in fresh_records()}
    for row in cm['records']:
        p=params[row['id']]
        require(row['parameters_sha256']==hashlib.sha256(historical.canonical(p)).hexdigest(),'Fresh parameters digest differs')
        audit.bind(folder/row['pcm_path'],row['pcm_sha256'])
        records[('challenge',row['id'])]={'scenario':{'id':row['id'],'kind':'observation-change','durationSeconds':2.,
            'initialHz':p['frequency_hz'],'finalHz':p['frequency_hz'],'changeStartSeconds':.83,'changeEndSeconds':.83,'slopeHzPerSecond':0},
            'original_feature_path':str((folder/row['feature_path']).relative_to(REPO)),
            'original_feature_sha256':row['feature_sha256'],'pcm_sha256':row['pcm_sha256']}
    require(len(records)==74,'Analytic inventory differs')
    return records


def audit_analytic(audit,run,manifest,metrics,models,last):
    path=run/'analytic-inventory.json';inventory=audit.json(path,manifest['input_sha256'][str(path.relative_to(REPO))])
    require({g:len(v) for g,v in inventory.items()}==GROUP_COUNTS,'Analytic group inventory differs')
    expected=expected_analytic(audit,run,manifest,last)
    historical_inventory=audit.json(COVERAGE_PRIOR/'analytic-inventory.json',last['input_sha256'][str((COVERAGE_PRIOR/'analytic-inventory.json').relative_to(REPO))])
    old_rows={(('consumed_coverage' if g=='challenge' else g),r['id']):r for g,rows in historical_inventory.items() for r in rows}
    case_gates={g:{m:{} for m in CANDIDATE} for g in GATE_COUNTS};summaries={};addition=defaultdict(int)
    original=torch.load(audit.bind(RUNS/'delayed-mlp-seed20261022/weights.pt',base.WEIGHTS),map_location='cpu',weights_only=True)['state_dict']
    max_p_difference=0.;pairs=controls=0
    for group,rows in inventory.items():
        require(set(metrics[group])=={r['id'] for r in rows},'Reported analytic inventory differs')
        for row in rows:
            name=row['id'];want=expected[(group,name)];scenario=row['scenario']
            for field in ('kind','durationSeconds','initialHz','finalHz','changeStartSeconds','changeEndSeconds','slopeHzPerSecond'):
                require(scenario[field]==want['scenario'][field],'Analytic physical scenario differs: '+field)
            for field in ('original_feature_path','original_feature_sha256','pcm_sha256'):
                require(row[field]==want[field],'Analytic original artifact identity differs: '+field)
            features={'original6':load_npz(audit,REPO/row['original_feature_path'],row['original_feature_sha256']),
                'expanded7':load_npz(audit,REPO/row['expanded_feature_path'],row['expanded_feature_sha256'])}
            provenance=load_npz(audit,REPO/row['provenance_path'],row['provenance_sha256'])
            if group!='challenge':
                prior_row=old_rows[(group,name)]
                require(row['source_provenance_path']==prior_row['provenance_path'] and row['source_provenance_sha256']==prior_row['provenance_sha256'],
                    'Consumed analytic source provenance binding differs')
                old_provenance=load_npz(audit,REPO/row['source_provenance_path'],row['source_provenance_sha256'])
                for peak in ('old_full40_hz','new_full40_hz','old_full40_confidence','new_full40_confidence'):
                    require(np.array_equal(old_provenance[peak],provenance[peak],equal_nan=True),'Consumed observed peak evidence changed')
            for k,v in validate_expansion(features['original6'],features['expanded7'],provenance,row['pcm_sha256']).items():addition[k]+=v
            n=76800 if group=='regression' else 96000
            end,target,eligible=historical.validate_feature(features['original6'],n)
            require(row['sample_count']==n,'Analytic signal extent differs')
            z=load_npz(audit,run/row['prediction_path'],row['prediction_sha256'])
            require(row['prediction_sha256']==manifest['prediction_sha256'][row['prediction_path']],'Analytic prediction binding differs')
            require(set(z)=={'original_probability'}|{m+s for m in SELECTORS for s in ('__state','__hz')},'Analytic prediction field inventory differs')
            probability=z['original_probability']
            with torch.inference_mode():actual_p=retention.equations.original_probability(original,features['original6']['features'])
            require(probability.shape==actual_p.shape and np.isfinite(probability).all() and np.all((probability>=0)&(probability<=1)),
                'Analytic original probability invalid')
            delta=float(np.max(np.abs(probability-actual_p)));max_p_difference=max(max_p_difference,delta)
            require(delta<=1e-6 and np.array_equal(probability<.5,actual_p<.5),'Original57 analytic voicing differs')
            arrays={m:(z[m+'__state'],z[m+'__hz']) for m in SELECTORS}
            groups={arm:independent_groups(f) for arm,f in features.items()}
            for arm in ARMS:
                subset={m:arrays[m] for m in SELECTORS if pool_of(m)==arm}
                pairs+=check_inference({m:models[m] for m in subset},features[arm],probability,subset,groups[arm])
            if group!='challenge':
                prior_row=old_rows[(group,name)];pp=COVERAGE_PRIOR/prior_row['prediction_path']
                require(row['prior_prediction_path']==str(pp.relative_to(REPO)) and row['prior_prediction_sha256']==prior_row['prediction_sha256'],
                    'Consumed control prediction identity differs')
                old=load_npz(audit,pp,prior_row['prediction_sha256'])
                require(np.array_equal(old['original_probability'],probability),'Consumed original voicing bytes changed')
                for seed in SEEDS:
                    state,hz=arrays[f'frozen-original6__seed{seed}']
                    require(np.array_equal(state,old[f'original__seed{seed}__state']) and np.array_equal(hz,old[f'original__seed{seed}__hz'],equal_nan=True),
                        'Consumed frozen-original6 output changed')
                    controls+=len(state)
            truth=historical.curve(scenario,target/48000);region=historical.regions(scenario,end);report=metrics[group][name]
            coverage=analytic_coverage(features,groups,truth,eligible,region,probability,arrays)
            compare_nested(audit,coverage,report['candidate_coverage'],group+'/'+name+'/coverage')
            require(set(report['methods'])==set(SELECTORS),'Analytic reported pipeline inventory differs')
            case={}
            for method,(state,hz) in arrays.items():
                base.prediction_objects(target,state,hz)
                for label,include in {'all':eligible,**{r:eligible&(region==r) for r in sorted(set(region[eligible]))}}.items():
                    got=historical.analytic_summary(truth[include],state[include],hz[include],region[include],scenario['slopeHzPerSecond'])
                    want_summary=report['methods'][method]['all'] if label=='all' else report['methods'][method]['regions'][label]
                    compare_nested(audit,got,want_summary,group+'/'+name+'/'+method+'/'+label)
                if scenario['kind']=='step':
                    compare_nested(audit,retention.step_summary(scenario,target[eligible],end[eligible],state[eligible],hz[eligible]),
                        report['methods'][method]['step_response'],group+'/'+name+'/'+method+'/step')
                else:require(report['methods'][method]['step_response'] is None,'Unexpected analytic step outcome')
                if method not in CANDIDATE:continue
                if group!='regression':gate=group;include=eligible&(region=='after-observation-change')
                elif scenario['kind']=='constant':gate='stationary';include=eligible
                elif scenario['kind']=='observation-change':gate='harmonic_change';include=eligible&(region=='after-observation-change')
                else:continue
                summary=historical.analytic_summary(truth[include],state[include],hz[include],region[include],0)
                success=summary['success_within_50c_all_frames'];case_gates[gate][method][name]=success
                case[method]={'frames':summary['frames'],'success':success,'pass':success>=.95}
            summaries[group+'/'+name]=case
    gates=finish_case_gates(case_gates);compare_nested(audit,gates,metrics['decision']['analytic'],'Analytic gates')
    return {'cases':74,'selector_methods_per_case':12,'mandatory_case_gates_per_seed':66,
        'checkpoint_frame_pairs':pairs,'original_control_frame_pairs':controls,'maximum_original_probability_difference':max_p_difference,
        'raw_pool_addition':dict(addition),'case_gates':summaries,'gates':gates}


def phase_order(manifest,initial,input_freeze,prerequisite,training,checkpoint,prediction):
    """Explicit prerequisite label access is allowed; final scoring follows predictions."""
    require(initial['status']=='sources_frozen' and input_freeze['status']=='input_caches_frozen' and
        prerequisite['status']=='complete' and prerequisite['pass'] is True and training['status']=='training_frozen' and
        checkpoint['status']=='checkpoints_frozen' and prediction['status']=='predictions_frozen','Required phase status differs')
    times=[manifest['sources_frozen_at'],input_freeze['frozen_at'],manifest['prerequisite_references_read_at'],
        prerequisite['frozen_at'],manifest['training_frozen_at'],manifest['checkpoints_frozen_at'],
        manifest['predictions_frozen_at'],manifest['final_scoring_reference_read_at'],manifest['completed_at']]
    require(all(historical.dt(a)<=historical.dt(b) for a,b in zip(times,times[1:])), 'Required phase ordering differs')
    require(input_freeze['reference_content_read'] is False and initial['development_references_read_for_prerequisites'] is False and
        prediction['development_references_read_for_prerequisites'] is True and prediction['development_references_read_for_final_scoring'] is False and
        manifest['development_references_read_for_prerequisites'] is True and manifest['development_references_read_for_final_scoring'] is True,
        'Declared prerequisite/final reference access differs')
    require(initial['fits']=={} and training['fits']=={} and set(prediction['fits'])=={f'{a}__seed{s}' for a in ARMS for s in SEEDS},
        'Fit inventory crosses the wrong freeze phase')
    return True


def audit_run(run,output):
    run=base.inside(RUNS,run);output=base.inside(run,output)
    require(not output.exists(),'Refuses to overwrite independent audit evidence')
    torch.set_num_threads(1);audit=base.Audit();m=audit.json(run/'manifest.json')
    require(m['schema']=='phenometrix-expanded-pool-experiment.v1' and m['status']=='complete','Completed expanded-pool experiment required')
    require(m['arms']==list(ARMS) and m['seeds']==list(SEEDS) and m['maximum_fits']==6 and m['predesignated_candidate']=='expanded7' and
        m['candidate_policy']=='add-shortest-admitted-full40-v1' and m['reserved_content_read'] is False,'Declared comparison/data boundary differs')
    require((m['epochs'],m['batch_size'],m['learning_rate'])==(12,4096,.001),'Training configuration differs')
    require(m['elapsed_seconds']<=4500 and m['fit_elapsed_seconds']<=1800 and m['peak_observed_python_rss_bytes']<=16*1024**3 and
        m['peak_observed_node_rss_bytes']<=2*1024**3 and m['peak_observed_mps_allocated_bytes']<=8*1024**3 and m['output_bytes']<=2*1024**3,
        'Declared resource ceiling exceeded')
    last=audit.json(COVERAGE_PRIOR/'manifest.json',COVERAGE_PRIOR_SHA)
    last_prediction=audit.json(COVERAGE_PRIOR/'prediction-freeze.json','fd0bbe50c3e4be845484c2e77ef47cb538fdee1d2b978e2543a5269079f4b89b')
    prior=audit.json(PRIOR/'manifest.json',PRIOR_SHA)
    require(last['status']=='complete' and last_prediction['prediction_sha256']==last['prediction_sha256'],'Historical coverage freeze differs')
    for path,digest in m['source_sha256'].items():audit.bind(REPO/path,digest);audit.bind(run/'source-snapshot'/path,digest)
    require(all(m['source_sha256'].get(p)==h for p,h in last['source_sha256'].items()),'Historical scientific source closure changed')
    historical_inputs={str(Path(p).relative_to(REPO)) if Path(p).is_absolute() else p:h for p,h in prior['input_sha256'].items()}
    for p,h in last['input_sha256'].items():
        require(p not in historical_inputs or historical_inputs[p]==h,'Historical input maps disagree')
        historical_inputs[p]=h
    require(m['historical_input_sha256']==historical_inputs,'Historical input inventory changed')
    for path,digest in historical_inputs.items():audit.bind(REPO/path,digest)
    for path,digest in m['input_sha256'].items():audit.bind(REPO/path,digest)
    expected_weights={name:{'path':str((PRIOR/'checkpoints'/(name+'.pt')).relative_to(REPO)),'sha256':digest}
        for name,digest in retention.CHECKPOINT_SHA.items()}
    require(m['historical_checkpoints']==last['checkpoints']==expected_weights,'Historical checkpoint identities changed')
    initial=audit.json(run/'source-freeze.json');input_freeze=audit.json(run/'input-cache-freeze.json')
    prerequisite=audit.json(run/'prerequisites.json');training=audit.json(run/'training-freeze.json')
    checkpoint=audit.json(run/'checkpoint-freeze.json');prediction=audit.json(run/'prediction-freeze.json')
    phases={'source-freeze':(initial,'source-freeze'),'input-cache-freeze':(input_freeze,'input-cache-freeze'),
        'prerequisites':(prerequisite,'prerequisites'),'training-freeze':(training,'experiment'),
        'checkpoint-freeze':(checkpoint,'checkpoint-freeze'),'prediction-freeze':(prediction,'experiment')}
    for name,(value,schema) in phases.items():
        require(value['schema']=='phenometrix-expanded-pool-'+schema+'.v1','Freeze schema differs: '+name)
        require(value['source_sha256']==m['source_sha256'],'Source changed across freezes: '+name)
    phase_order(m,initial,input_freeze,prerequisite,training,checkpoint,prediction)
    for value in (initial,training,prediction):
        for key in ('arms','seeds','maximum_fits','historical_checkpoints','candidate_policy','challenge_spec_sha256','sources_frozen_at'):
            require(value[key]==m[key],'Identity changed across phase: '+key)
    require(checkpoint['source_freeze_path']==str((run/'source-freeze.json').relative_to(REPO)) and
        checkpoint['source_freeze_sha256']==base.sha(run/'source-freeze.json') and
        checkpoint['challenge_spec_sha256']==m['challenge_spec_sha256'] and checkpoint['arms']==list(ARMS) and
        checkpoint['seeds']==list(SEEDS) and checkpoint['maximum_fits']==6,'Final checkpoint source/spec binding differs')
    expected_checkpoints={name:{'path':str((run/'checkpoints'/(name+'.pt')).relative_to(REPO)),'sha256':fit['weights_sha256']}
        for name,fit in m['fits'].items()}
    require(checkpoint['checkpoints']==expected_checkpoints and prediction['fits']==m['fits'],'Final checkpoints changed after freeze')
    require(prerequisite['input_cache_freeze_sha256']==base.sha(run/'input-cache-freeze.json'),'Prerequisite input-cache binding differs')
    for phase in (input_freeze,training,prediction):
        require(all(m['input_sha256'].get(p)==h for p,h in phase['input_sha256'].items()),'Frozen input changed')
    reference_paths={};reference_hashes={}
    for split in ('train','development'):
        _,_,metadata=base.load_cache(audit,split)
        reference_paths[split]={str((ROOT/'data/ptdb-tug'/v['path']).relative_to(REPO)) for v in metadata.values()}
        reference_hashes[split]={str(Path(p).relative_to(REPO)) if Path(p).is_absolute() else p:h
            for p,h in prerequisite['reference_sha256'][split].items()}
        require(set(reference_hashes[split])==reference_paths[split],'Prerequisite REF-role inventory differs')
        for p,h in reference_hashes[split].items():require(m['input_sha256'].get(p)==h and prediction['input_sha256'].get(p)==h,'Reference identity changed after prerequisite phase')
    retention.no_reference_inputs(input_freeze['input_sha256'],set.union(*reference_paths.values()))
    source=(run/'source-snapshot/research/voice-prosody/phenometrix_prosody/expanded_pool_experiment.py').read_text()
    tokens=["save(output/'source-freeze.json',source_freeze)","original[split]=old.load_inputs(split)",
        "save(output/'input-cache-freeze.json'",'labels={s:old.load_labels(data)',"save(output/'prerequisites.json'",
        "save(output/'training-freeze.json'",'for seed in SEEDS:',"save(output/'checkpoint-freeze.json'",
        'fresh,h=cases.generate_challenge(',"save(output/'prediction-freeze.json'",'dl=old.load_labels(dev)']
    start=source.index('def run(output):');positions=[source.index(token,start) for token in tokens]
    require(positions==sorted(positions),'Source phase ordering differs')
    require('del labels,baseline,raw_targets' in source and 'if not prerequisite[\'pass\']:' in source,
        'Prerequisite stop/reference disposal source contract missing')
    require(prediction['prediction_sha256']==m['prediction_sha256'],'Prediction freeze changed')
    for path,digest in m['prediction_sha256'].items():audit.bind(run/path,digest)
    inventory=audit.json(run/'analytic-inventory.json');expected_predictions={'predictions/'+n+'.npz' for n in METHODS}|{r['prediction_path'] for rows in inventory.values() for r in rows}
    require(set(m['prediction_sha256'])==expected_predictions and len(expected_predictions)==88,'Prediction inventory differs')
    metrics=audit.json(run/'metrics.json',m['metrics_sha256']);require(metrics['schema']=='phenometrix-expanded-pool-metrics.v1','Metrics schema differs')
    models,prior,checkpoint_receipt=verify_checkpoints(audit,run,m)
    cells,coverage,headroom,targets,natural=natural_and_coverage(audit,run,m,metrics,models,last)
    for key,value in (('coverage',coverage),('headroom',headroom),('exclusions',metrics['exclusions'])):
        compare_nested(audit,value,prerequisite[key],'Prerequisite/'+key)
    checks={'raw_fixed_voicing_oracle_headroom':headroom['expanded7']['fixed-slots']['pass'],
        'train_grouping_loss':coverage['train']['expanded7']['pass'],'development_grouping_loss':coverage['development']['expanded7']['pass']}
    require(all(checks.values()),'Prerequisites should have stopped fitting')
    compare_nested(audit,checks,prerequisite['checks'],'Prerequisite checks');compare_nested(audit,checks,metrics['decision']['prerequisites'],'Final prerequisite checks')
    training_receipt=audit_training(audit,run,m,targets)
    analytic_receipt=audit_analytic(audit,run,m,metrics,models,last)
    decision=natural_gate(cells);compare_nested(audit,decision,{k:metrics['decision'][k] for k in decision},'Natural gate')
    promising=decision['pass'] and all(checks.values()) and all(v for k,v in analytic_receipt['gates'].items() if k.endswith('_pass'))
    audit.equal(promising,metrics['decision']['promising_for_next_study'],'Combined gate')
    control_pairs=3*natural['development_rows']+analytic_receipt['original_control_frame_pairs']
    compare_nested(audit,{'state_hz_frame_pairs':control_pairs,'state_mismatches':0,'hz_mismatches':0,'all_exact':True},m['control_reproduction'],'Historical six-pool controls')
    tests=ROOT/'tests/test_audit_expanded_pool.py'
    modules=(Path(__file__),tests,Path(base.__file__),Path(historical.__file__),Path(augmentation.__file__),Path(retention.__file__),
        Path(retention.subharmonic.__file__),Path(retention.equations.__file__),Path(__file__).with_name('selector_model.py'),
        Path(__file__).with_name('harmonic_selector_model.py'),Path(__file__).with_name('metrics.py'),Path(__file__).with_name('pilot.py'))
    for path in modules:audit.bind(path)
    audit.close()
    receipt={'schema':'phenometrix-expanded-pool-independent-audit.v1','status':'complete','pass_result':True,'audited_at':datetime.now(timezone.utc).isoformat(),
        'run_path':str(run.relative_to(REPO)),'run_manifest_sha256':base.sha(run/'manifest.json'),'metrics_sha256':base.sha(run/'metrics.json'),
        'source_freeze_sha256':base.sha(run/'source-freeze.json'),'prerequisites_sha256':base.sha(run/'prerequisites.json'),
        'training_freeze_sha256':base.sha(run/'training-freeze.json'),'checkpoint_freeze_sha256':base.sha(run/'checkpoint-freeze.json'),
        'prediction_freeze_sha256':base.sha(run/'prediction-freeze.json'),'audit_source_sha256':base.sha(Path(__file__)),
        'audit_tests_sha256':base.sha(tests),'bound_artifacts':len(audit.files),'numeric_comparisons':audit.comparisons,
        'maximum_numeric_difference':audit.max_rate_delta,'training':training_receipt,'checkpoints':checkpoint_receipt,
        'natural':natural,'analytic':analytic_receipt,'coverage':coverage,'headroom':headroom,'natural_decision':decision,
        'prerequisites':checks,'promising_for_next_study':promising,'original_control_frame_pairs':control_pairs,
        'total_checkpoint_frame_pairs':natural['checkpoint_frame_pairs']+analytic_receipt['checkpoint_frame_pairs'],
        'opened_input_source_sha256':audit.files,'limitations':[
          'No fitting, waveform generation, policy adjustment, reserved/test content, or production promotion.',
          'Independent object-based natural scoring, separate analytic count/gate arithmetic, and explicit CPU checkpoint equations.',
          'Raw union and source scores are rebuilt from recorded peak evidence. Raw autocorrelation curves are not recomputed.',
          'Natural DEV references were explicitly accessed for fixed prerequisites before fitting; predictions freeze before final scoring. This is not global label blinding.',
          'Phase evidence binds source order, actual files and manifest timestamps, not operating-system access logs.',
          'Training schedules, targets, loss records, scaling and initialization are checked; optimizer execution is not independently rerun.',
          'Repeated development and consumed regressions remain exploratory; fresh noiseless cases do not establish external or clinical generalization.']}
    output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'complete','pass_result':True,'numeric_comparisons':audit.comparisons,'maximum_numeric_difference':audit.max_rate_delta,
        'checkpoint_frame_pairs':receipt['total_checkpoint_frame_pairs'],'promising_for_next_study':promising,'output':str(output)}))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit_run(args.run.resolve(),args.output.resolve())
