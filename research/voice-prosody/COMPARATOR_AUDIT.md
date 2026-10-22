# Pinned tiny comparator audit

This is an input-only diagnostic comparator using `torchcrepe==0.0.24`, pinned upstream commit `19e2ec3d494c0797a5ff2a11408ec5838fba6681`. Its defaults are centered frames, weighted argmax without Viterbi, periodicity threshold 0.5, and raw-frame RMS threshold 0.003. These thresholds are diagnostic settings, not frozen calibration. Periodicity is never presented as a probability of pitch correctness.

## Time and support contracts

Both adapters use the input-length-only grid `16 + 10*k` ms, with no reference rows, labels or eligibility mask accepted by inference. At 48 kHz, these are sample indices `768 + 480*k`.

| Adapter | Source interval for target t | Model window center | Earliest input availability | Declared lookahead |
|---|---|---|---|---|
| Default centered | [t−32 ms, t+32 ms) | t | t+32 ms | 32 ms |
| Explicit end anchored modification | [t−64 ms, t) | t−32 ms | t | 0 ms |

The default method is `torchcrepe-tiny-0.0.24-centered-frame-local-v1`. The explicitly modified method is `torchcrepe-tiny-end-anchored64-modified-v1`, enabled with `TinyConfig(anchor="end_anchored")`. The latter evaluates the network on a past-centered window while retaining target t; it is a separate diagnostic time contract, not a replacement baseline or an implicitly shifted score. Unsupported boundary frames remain unavailable. This documents sample support, not measured compute or wall-clock availability.

Each source interval is cropped before resampling. At 48 kHz its 3,072 samples are converted to 1,024 samples at 16 kHz using SciPy `resample_poly(up=1, down=3, window=("kaiser", 5), padtype="constant", cval=0)`. The first output sample maps to the first raw-frame timestamp. Zero extension is local to that frame, so no other source samples are read. Native 16 kHz frames are copied. Other rates are rejected. This deliberately bounded adapter differs from upstream whole-file `resampy` preprocessing and must be disclosed with results. No F0 interpolation or post-hoc time shift occurs.

Each frame is mean centered and divided by its sample standard deviation, with floor 1e−10. The pinned model emits sigmoid activations. The pinned weighted-argmax decoder applies another sigmoid after retaining nine bins around the winning allowed bin; the adapter preserves this unusual behavior exactly. Its 360 triangular cents-dither values use local `RandomState(20261022)`, preventing global-RNG and batch-order dependence. Frequency bounds are 50–700 Hz; periodicity is the winning allowed bin's activation before the second sigmoid. There is no temporal smoothing, Viterbi decoding, reference-derived tuning or correctness calibration.

## Source and artifact identity

The installed model, decoder, conversion, core and tiny-weight Git blob identities matched the pinned upstream tree. Loader checks require version 0.0.24 and the exact model-source and tiny-weight SHA-256 before inference.

| Artifact | Bytes where relevant | SHA-256 |
|---|---:|---|
| tiny.pth | 1,962,363 | d4993eea36ed1a0ad9ac549c740dae5265b049ce72004f00c2f59e01c0be8432 |
| model.py | | 028e240a191d5ddb28d0af6febe999b9f692f22459e49bf9c5681c9dfea4bd22 |
| core.py | | a94d90e76670322889efedaf86c5812a599f96140535d49b1b4e9e6dc1b05840 |
| decode.py | | cf3380cf1ece5bc9c73936a7927ad42e224b71b4936366264b24f725c1550522 |
| convert.py | | f5b10b93e7d5b9b16ec5d590c292c3d31716ba3ed5d3ebca1a3741d19ce37485 |
| PyPI torchcrepe-0.0.24-py3-none-any.whl | 72,326,298 | ec054c23c9d45328f213f93a0131570a3f0e5903e9382792bed95f17a8c36d5a |

The tiny checkpoint has 486,552 parameters and 44 state tensors. The package also ships an 88,991,291-byte full checkpoint; this adapter never loads or uses it. Only prebuilt wheels were installed in the existing local environment: torchcrepe 0.0.24, torchaudio 2.9.1, resampy 0.4.3, tqdm 4.70.1. Relevant existing versions are torch 2.9.1, numpy 2.2.6 and scipy 1.17.1. No remote compute or native build was performed.

Primary sources: pinned [README](https://github.com/maxrmorrison/torchcrepe/blob/19e2ec3d494c0797a5ff2a11408ec5838fba6681/README.md), [core](https://github.com/maxrmorrison/torchcrepe/blob/19e2ec3d494c0797a5ff2a11408ec5838fba6681/torchcrepe/core.py), [decoder](https://github.com/maxrmorrison/torchcrepe/blob/19e2ec3d494c0797a5ff2a11408ec5838fba6681/torchcrepe/decode.py), [conversion](https://github.com/maxrmorrison/torchcrepe/blob/19e2ec3d494c0797a5ff2a11408ec5838fba6681/torchcrepe/convert.py), [model](https://github.com/maxrmorrison/torchcrepe/blob/19e2ec3d494c0797a5ff2a11408ec5838fba6681/torchcrepe/model.py), and [official PyPI release metadata](https://pypi.org/pypi/torchcrepe/0.0.24/json).

## Training overlap and rights

The torchcrepe README attributes the converted tiny weights to `marl/crepe` via MMdnn. The [original CREPE README at c9b71ce61491454125a0693f584f7244f29d9884](https://github.com/marl/crepe/blob/c9b71ce61491454125a0693f584f7244f29d9884/README.md) lists MIR-1K, Bach10, RWC-Synth, MedleyDB, MDB-STEM-Synth and NSynth training data. It does not report PTDB-TUG training. Exact per-record training manifests and the source TensorFlow-checkpoint-to-converted-weight mapping were not established, so zero overlap is not proven.

The inspected [torchcrepe license](https://github.com/maxrmorrison/torchcrepe/blob/19e2ec3d494c0797a5ff2a11408ec5838fba6681/LICENSE) is MIT, copyright 2020 Max Morrison; [original CREPE](https://github.com/marl/crepe/blob/c9b71ce61491454125a0693f584f7244f29d9884/LICENSE) is MIT, copyright 2018 Jong Wook Kim. No separate checkpoint restriction appeared in those inspected sources. Preserve their notices and attribution. Corpus rights remain separate from code/model notices.

## Verification

Seven synthetic adapter tests cover exact support and targets, future-suffix isolation for both anchors, per-frame normalization, equivalence to the pinned weighted decoder, input-only inference, and batch independence. Test backends avoid pretrained neural forward passes. Actual model-quality and runtime measurements belong in separately identified run artifacts; this source audit does not imply an accuracy or latency gate passed.

Adapter SHA-256: `e113f1118029e47c851cc0629b07aa800b294cf1c6fff784629c3ceaf0731564`. Test SHA-256: `1451fe0971bfd7d28d2dccdc1f5de0a449e9231d606bad691ef78aa4f3e2d485`.

## Frozen feature-cache runner

`phenometrix_prosody.comparator_cache` accepts only completed input-only training or development feature caches and refuses sealed-test content. It validates the frozen acquisition metadata, speaker split, complete source/condition order and source snapshots. Each condition is reconstructed from its pinned MIC bytes with the same deterministic corruption function, and its exact float32 PCM SHA-256 must match the cache before prediction. The runner never opens reference or laryngograph values and contains no scoring step.

Centered predictions retain nominal target t and 32 ms lookahead; they do not inherit the feature method’s 20 ms delay. All prediction files have per-record hashes. `prediction-freeze.json` is written only after all predictions complete and the input cache/source identities are rechecked. Scoring belongs to a subsequent process. Seven synthetic CPU fixture tests cover exact corruption reconstruction, changed-contract and sealed guards, order/coverage checks, and a complete input-only prediction freeze. No pretrained forward pass or MPS workload was launched by these tests.

After the relevant cache completes, run locally when the GPU is free:

```sh
PYTHONPATH=research/voice-prosody research/voice-prosody/.venv/bin/python -m phenometrix_prosody.comparator_cache --features research/voice-prosody/runs/delayed-train-features --data research/voice-prosody/data/ptdb-tug --output research/voice-prosody/runs/delayed-train-comparator --device mps
PYTHONPATH=research/voice-prosody research/voice-prosody/.venv/bin/python -m phenometrix_prosody.comparator_cache --features research/voice-prosody/runs/delayed-development-features --data research/voice-prosody/data/ptdb-tug --output research/voice-prosody/runs/delayed-development-comparator --device mps
```

Output directories must not already exist. Synthetic backend tests label their artifacts explicitly and are not valid comparator measurements.

`phenometrix_prosody/comparator_cache.py` SHA-256: `7bc12c2e181c969131f0dd7a1fcee7085da3534c96af9ba18fe12a9fd63a9374`.

`tests/test_comparator_cache.py` SHA-256: `9860bf13b053e8f1115b35b8632bdbf75c4a2b248c8a6f24da645d2993678fec`.

The original centered pilot predates the optional end-anchored adapter. Its exact source was recovered by reversing that addition and accepting it only after matching the existing run-manifest SHA-256 `3252701555c198371231feb57039349678dc2cfa25cf10c8f0bb3919931d3b46`. The byte-identical source is retained in `benchmarks/voice-prosody/source-snapshots/comparator-centered-3252701555.py`. Both original comparator prediction sets were independently hash-verified by `benchmarks/voice-prosody/delay-diagnostic-complete-input-integrity-audit.json`; no historical prediction or score was changed.
