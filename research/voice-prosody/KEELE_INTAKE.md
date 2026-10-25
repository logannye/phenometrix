# KEELE intake: external assessment blocked

Intake date: October 24, 2026 (Pacific). No KEELE waveform, laryngograph,
reference array, feature summary, prediction, or score was opened. The numerical
adapter was tested with synthetic inputs only. This is a research reference
contract decision; it is not a measured model result.

## Decision and evidence

The retained release README says:

> If you use different parameters in your own pitch tracker, the values should not be used as a ground truth for comparisons

It recommends recomputing the laryngograph reference with matching parameters.
The fixed study uses 40 ms YIN support, while the package reference metadata
specifies 25.6 ms autocorrelation. The README also describes imperfect correction
of acoustic delay between laryngograph and microphone. Its earlier description
says 26.5 ms, while its final caution and all ten reference metadata records say
25.6 ms. None establishes exact physiological alignment for our 40/10/20 ms
estimator. Constructing a new reference or fitting a delay would change the
prescribed external contract and is outside this intervention.

**External assessment is not admitted.** Keep KEELE unopened beyond the metadata
access recorded below, retain the synthetic adapter, and complete the predetermined
PTDB natural-browser fallback. Do not replace this block with an external accuracy
claim or silently reinterpret the reference caution. A later study would need a
prospective reference decision before outcomes, not retrospective lag fitting.
The public, source-bound decision is
[voicing-robustness-external-intake.json](../../benchmarks/voice-prosody/voicing-robustness-external-intake.json).

## Provenance and rights

- [Zenodo record 3921794](https://zenodo.org/records/3921794), release 1.1.0,
  June 29, 2020, by Bastian Bechtold, describes KEELE as available for
  noncommercial use. It lists `KEELE.zip`, 22,782,212 bytes, publisher MD5
  `f5a87014bad14744660b90187de7d43f`. `KEELE_mod.zip` is a separate package and
  was not accessed. The record's historical rights link was unavailable.
  The retained README contains no separate license grant. Do not infer commercial
  use or permission to publish recordings from this intake.
- [The ZIP preview](https://zenodo.org/records/3921794/preview/KEELE.zip?include_deleted=0)
  and retrieved central directory contain ten record IDs: `f1nw0000` through
  `f5nw0000`, and `m1nw0000` through `m5nw0000`. Metadata bindings agree with these
  microphone/source identities. The declared source duration exceeds the reference
  duration by 100 native samples for m1/m2 and 300 for m3; these remain explicit
  trailing unlabeled extent, with no extrapolation. This is ten declared package records, not a
  completed waveform or participant validation.
- [Plante, Meyer and Ainsworth (1995)](https://www.isca-archive.org/eurospeech_1995/plante95b_eurospeech.html)
  describes a 15-speaker core. That literature count does not describe the
  available ten-record package.
- [Wang (2001), §2.5.1 and Table 2-1](https://sls.csail.mit.edu/archives/root/publications/2001/Wang_phd_thesis.pdf)
  distinguishes positive voiced references, zero unvoiced references, and
  uncertain negative values, including the original −1 sentinel. It also notes
  that the five child recordings described in the earlier paper were unavailable.
- [The conversion source](https://github.com/bastibe/Replication-Dataset-Scripts/blob/de3e2cbcc8ab0ccb7df60af68c6e6f72c59afdb0/keele.py)
  is pinned to `de3e2cbcc8ab0ccb7df60af68c6e6f72c59afdb0`, committed June 29, 2020;
  file SHA-256 `aa3d2285a7397fb086391c97db8344d672a43755d5661ce0e6491b6c2cf0edcd`.
  It converts 20 kHz PCM and reciprocal signed periods into stored Hz values,
  retaining zero. Consequently original −1 becomes −20000, and negative periods
  remain uncertain negative frequencies. The time expression is integer
  `arange(N)` divided by 100; multiplying by float `.01` is not byte-equivalent
  for every frame. This code was inspected, not executed or installed.

The JSON release documentation SHA-256 is
`9a2575643d5d19448fe11f6faf71c13dd696bc589811ac8a7580f7cbcbfc50ba`.
This pins the retained documentation bytes. The complete archive was not
transferred, so its publisher MD5 was **not locally verified** and its local
SHA-256 is **unavailable**. Central-directory CRC32 checks authenticate internal
consistency of the retrieved metadata entries; they are not a cryptographic
verification of the whole archive or original recording bytes. The conversion
repository has no embedded archive-to-commit assertion; the pinned code and
release date are provenance evidence, not proof of the historical build process.

## Access chronology and boundaries

Initial read-only filename/acquisition-record searches in this research checkout
found existing PTDB records and no KEELE-named acquisition. This bounded search
cannot prove that the user has never encountered KEELE elsewhere.

The intake read primary web documentation, the archive listing, and the pinned
conversion source. HTTPS Range requests then retrieved the ZIP end record,
central directory, and only 31 JSON members (dataset README, ten item metadata,
ten signal metadata, ten pitch metadata). There were 85 ZIP members and 18,893
transferred bytes. No `.wav`, `.npy`, Python module, or bytecode member was
requested or decompressed. CRC32 and SHA-256 were recorded for metadata members.
Item metadata unavoidably included the common North Wind transcript and coarse
transcription boundaries. These were not used for selection, fitting or scoring.

The create-new ignored receipt is
`runs/keele-metadata-intake-v1/metadata-receipt.json`; the exact metadata retrieval
script is retained beside it. It records UTC access time, range coordinates,
member inventory and hashes. No raw corpus payload is in public outputs.

## Synthetic adapter contract

`phenometrix_prosody.keele_adapter` is separate from the historical PTDB helper.
It validates role IDs, declared metadata counts/rates/origins, and exact converted
reference time representation. Real array access remains scorer-only and is not
allowed by this blocked intake. Metadata validation never admits an assessment.

The target grid is `480*k` at 48 kHz, with support
`[target−960, target+960)` and earliest availability `target+960`. The first
complete support targets 20 ms. The unchanged 64 ms start and 32 ms end common
mask first admits 70 ms. An exact browser join uses
`supportEnd−960−sourceOffset` and rejects off-grid source/reset phase, duplicates,
backwards order, or incomplete support. It never uses the endpoint reference
index, nearest labels, interpolation, fitted delay or predicted pitch.

Positive finite references remain voiced, including those outside 50–700 Hz;
out-of-range values retain unconditional denominators and a separate count.
Zero is unvoiced. Finite negatives remain uncertain; nonfinite values are
invalid. Uncertain/invalid references and boundary targets retain their own
counts; zero eligible references stays zero rather than becoming a perfect score.

Offline synthetic resampling uses installed SciPy 1.17.1, float64 arithmetic,
12/5 rational `resample_poly`, explicit 481-tap Kaiser β=5 FIR with cutoff 1/12,
zero extension and removal of the centered FIR delay. Output has
`ceil(N*12/5)` samples, cast once to float32 without normalization or clipping.
Receipts bind settings, coefficient hash, PCM hash, versions and endpoint rules.
This preparation has future dependence and is not a causal device pipeline.
The tested origin is sample zero; impulses at native indices divisible by five
map to exact canonical indices. A separate direct FIR sum verifies index and
endpoint arithmetic without using `resample_poly` as its own oracle.

Validation: **35 synthetic tests passed**, including malformed/hash-mismatched
files, mismatched source IDs/counts, uncertain/nonfinite/out-of-range values,
zero eligible denominators, endpoint masks, source/reset phase corruption,
converter floating-point representation, impulse anchoring and direct-sum
resampling. No native compilation or corpus inference was performed.

```sh
.venv/bin/python -m phenometrix_prosody.keele_adapter contract
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_keele_adapter.py
```
