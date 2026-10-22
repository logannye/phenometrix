# PTDB-TUG acquisition and reference audit

This document preserves the initial 24-record training audit. The later bounded
expansion acquired 240 training and 80 development microphone/reference pairs,
plus the original 24 training laryngograph files: 664 files and 260,682,333 data
bytes. The initial-pilot statements below describe that earlier stage. Sealed-test
acoustic content remains unacquired. See [RESULTS.md](RESULTS.md) for final cohort
and evaluation status and [ALIGNMENT_AUDIT.md](ALIGNMENT_AUDIT.md) for the later
source-level and raw-cycle timing checks.

The complete public directory inventory passes the speaker/prompt and file-role checks. The first training pilot is present locally. Physiological reference timing remains unresolved, so this audit does not close G1 or authorize a final accuracy claim. The machine-readable evidence is [`ptdb-inventory-audit.json`](../../benchmarks/voice-prosody/ptdb-inventory-audit.json).

## Source and grouping

Live official metadata retrieved on October 22, 2026 contains **4,718 recordings**, with 4,718 files in each of MIC, LAR and REF (14,154 files). All speakers have 236 recordings except F08, which has 234. The missing recordings are exactly `F08_si1912` and `F08_si1913`, as documented by the provider. Every inventoried recording has all three roles. The complete archive is unnecessary for this workflow.

Using Python `random.Random(20261022).shuffle(list(range(1,11)))`, training pairs are `[4,8,7,9,10,2]`, development pairs `[1,3]`, and sealed-test pairs `[5,6]`. Each pair contains matching female/male speaker numbers. Excluding universal `sa1`/`sa2` removes 40 recordings, leaving 4,678 primary recordings. All cross-split prompt-ID intersections and case/whitespace-normalized prompt-text intersections are empty. Pair 08 differs only at the two documented missing prompts. Full prompt text and raw listings remain in ignored local data, with source URLs, retrieval times and hashes recorded in the evidence JSON.

Source evidence: [official dataset page](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html), [file index](https://www2.spsc.tugraz.at/databases/PTDB-TUG/), [speaker profiles](https://www2.spsc.tugraz.at/databases/PTDB-TUG/DOCUMENTATION/SPEAKER-PROFILES.txt), [prompt metadata](https://www2.spsc.tugraz.at/databases/PTDB-TUG/DOCUMENTATION/TIMIT-PROMPTS.txt), and [technical report](https://www2.spsc.tugraz.at/databases/PTDB-TUG/DOCUMENTATION/PTDB-TUG_REPORT.pdf), especially sections 2–5 and 7.

## Actual pilot

Two prompts per training speaker were selected by ascending SHA-256 of `20261022:integer_pair:prompt`, using metadata only. All 24 microphone/reference/laryngograph triplets were acquired: 72 files, **33,858,276 file bytes**, and **172.97 seconds** of microphone audio. The complete data directory, including metadata and manifests, is approximately 41.6 MiB. No development or sealed-test acoustic content or reference outcome was acquired. No full archive or duplicated extracted corpus exists.

All 24 MIC/LAR pairs have equal sample counts and are 48 kHz, mono, uncompressed signed PCM16. This confirms the file/sample coordinate contract, not acoustic-versus-glottal physiological synchrony. The provider reports simultaneous stereo recording followed by channel extraction, with no other processing of these raw signals.

The reference files have four finite columns and 17,153 rows in total. In this pilot, 13,139 rows have nonpositive F0; voicing-column values are exactly 0 or 1, with no positive-F0/zero-voicing or nonpositive-F0/positive-voicing contradictions. Positive reference F0 spans **53.5937652588–326.452911377 Hz**. Retain both F0 and voicing fields; these observations do not establish universal validity rules for the whole corpus.

## Timing and physiological uncertainty

The report specifies RAPT/ESPS with a 32 ms window and 10 ms hop. The [Snack ESPS manual](https://www.speech.kth.se/snack/man/snack2.2/tcl-man.html#spitch) places the first estimate at half the window length, yielding the nominal grid `16 + 10*k` ms. On that grid every pilot recording ends **54 ms after its final reference target**. Missing end rows are missing labels, not evidence of unvoiced speech. Common support and existing-reference masks must be applied explicitly.

Before extraction, the provider high-pass filtered LAR in Matlab with a linear-phase Kaiser filter (`n=2400`, beta 0.5, cutoff 25 Hz for female and 15 Hz for male). The report does not specify the filter invocation, whether `n` denotes order or taps, delay compensation, or cropping. Approximately 25 ms of causal group delay is possible; it is not established and no subtraction is applied. The documentation directory contains no standalone processing script beyond the extraction-script listing in the report. The exact producer version is unverified. Although the manual lists a 60 Hz minimum default, the subsequently inspected official Snack 2.2.10 source defaults to 50 Hz and predicts all 24 pilot reference row counts exactly. The full source and independent raw-cycle timing sensitivity are documented in [ALIGNMENT_AUDIT.md](ALIGNMENT_AUDIT.md); they support the nominal grid on selected strong cycles while leaving filter provenance unresolved.

A deterministic raw-LAR check on training recording `F04_si1146` probes the midpoint of its five longest reference-positive runs, using 32 ms centered first-difference windows and normalized autocorrelation over 50–700 Hz. At nominal target 7476 ms, a raw-sensor cycle of 288 samples gives 166.6667 Hz versus reference 166.6594 Hz (0.075 cents). Four probes differ by 0.075–25.59 cents; another differs by 102.38 cents with a weak 0.108 normalized peak. This supports partial cycle correspondence while retaining the counterexample. It does not establish transition timing or filter-delay compensation. The probe makes no per-recording time-shift search, and the nominal timing convention is unchanged.

The remaining G1 reference task is to establish whether a single justified timing convention adequately represents physiological transitions, or explicitly bound the unresolved timing error and its effect before freezing the benchmark. A stronger microphone model cannot resolve ambiguous reference provenance.

## Reproduction and manifest contract

From the repository root, use the existing environment; these commands do not build native packages:

```sh
cd research/voice-prosody
.venv/bin/python -m phenometrix_prosody.data inventory
.venv/bin/python -m phenometrix_prosody.data acquire --split train --per-speaker 2 --roles mic,ref,lar
.venv/bin/python -m phenometrix_prosody.data probe
.venv/bin/python -m unittest discover -s tests -p test_data.py -v
```

The acquisition root is `research/voice-prosody/data/ptdb-tug/`, ignored by Git. `metadata/inventory.json` contains the full filename inventory and source identities; `manifest.json` contains acquired files; `audit-train.json` contains header/column/timing summaries; `alignment-probe-train.json` contains the numerical cycle inspection. Rebuilding inventory creates a fresh retrieval snapshot: the existing acquisition manifest intentionally rejects an inventory hash change. Retain the current pinned snapshot for resume, or use a new `--root` for a fresh acquisition.

Each `manifest.files` entry contains `recording_id`, `speaker`, `pair`, `prompt`, `role`, `split`, `filename`, `url`, relative `path`, exact `bytes`, local `sha256`, `downloaded_at`, HTTP metadata, and the explicit lack of a provider checksum for that individual file. Resolve paths against the acquisition root, for example `records/train/F04/mic_F04_si1146.wav`. `read_reference(path)` returns four-float rows ordered as F0 Hz, voicing probability, local RMS, and peak-normalized cross-correlation. Reference and LAR roles are targets/audit inputs only, never inference features.

Acquisition defaults to training and a 1 GiB data-directory ceiling, with a 1 MiB manifest/audit reserve. Development requires the explicit split. Sealed-test acquisition additionally requires a freeze record that binds the inventory and declares source, model, configuration and protocol identities. It is closed in this run. Existing files are accepted only after SHA-256 verification; partial transfers are removed on failure. Corpus checksums advertised in the source index apply to `SPEECH_DATA_ZIPPED.zip` only and cannot verify individually downloaded recordings.

## License and attribution

The provider licenses the database under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) and individual contents under [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/). Local copies of both notices are retained. The repository's MIT license does not relabel the dataset. Preserve attribution and notices; public derivative-database use or distribution brings the ODbL's applicable notice, share-alike and access requirements. No corpus redistribution or public model upload was performed.

Citation: G. Pirker, M. Wohlmayr, S. Petrik, and F. Pernkopf, “A Pitch Tracking Corpus with Evaluation on Multipitch Tracking Scenario,” Interspeech, pp. 1509–1512, 2011.
