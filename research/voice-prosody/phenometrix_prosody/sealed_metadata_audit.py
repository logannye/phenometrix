"""Pre-access sealed-speaker audit. Reads metadata only; never PCM or REF content.

This is scoped filesystem/manifest evidence, not a claim about all historical
machine access. The complete filename inventory is permitted metadata.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

SEED = 20261022
SPEAKERS = ("F05", "F06", "M05", "M06")
PER_SPEAKER = 40
INVENTORY_SHA256 = "414c9b8e1ec8b462ae01c92d0b434519631e66bf67428120acb9b881085c798b"
HISTORY_NAMES = {"manifest.json", "prediction-freeze.json", "acquisition-manifest.json", "microphone-selection.json"}
SEALED_RECORD = re.compile(r"(?<![A-Za-z0-9])([FM]0[56]_s[axi]\d+)(?!\d)")
SEALED_FILE = re.compile(r"(?:mic|lar|ref)_[FM]0[56]_s[axi]\d+\.(?:wav|f0)(?:\.|$)")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def metadata_bytes(path):
    """The only content-reading entry point, restricted to explicit metadata."""
    path = Path(path)
    if path.suffix.lower() not in {".json", ".txt", ".html", ".md5", ".sha1"} or SEALED_FILE.search(path.name):
        raise ValueError("Refusing non-metadata content")
    return path.read_bytes()


def parse_prompt_metadata(raw):
    prompts = {}
    for line in raw.decode().splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        match = re.fullmatch(r"(.+?)\s+\((s[axi]\d+)\)\s*", line)
        if not match or match[2] in prompts:
            raise ValueError("Invalid or duplicate prompt metadata")
        prompts[match[2]] = " ".join(match[1].casefold().split())
    return prompts


def mentioned_sealed_records(value):
    if isinstance(value, dict):
        return set().union(*(mentioned_sealed_records(k) | mentioned_sealed_records(v) for k, v in value.items())) if value else set()
    if isinstance(value, list):
        return set().union(*(mentioned_sealed_records(v) for v in value)) if value else set()
    return set(SEALED_RECORD.findall(value)) if isinstance(value, str) else set()


def audit_metadata(data_root, runs_root, *, expected_inventory_sha256=INVENTORY_SHA256, per_speaker=PER_SPEAKER):
    data_root, runs_root = Path(data_root).resolve(), Path(runs_root).resolve()
    inventory_path = data_root / "metadata/inventory.json"
    inventory_raw = metadata_bytes(inventory_path)
    if sha(inventory_raw) != expected_inventory_sha256:
        raise ValueError("Inventory identity differs from the pre-access declaration")
    inventory = json.loads(inventory_raw)
    pairs = list(range(1, 11)); random.Random(SEED).shuffle(pairs)
    assignment = {"train": pairs[:6], "development": pairs[6:8], "sealed_test": pairs[8:]}
    if inventory["seed"] != SEED or inventory["split_assignment"] != assignment:
        raise ValueError("Speaker split differs from the frozen permutation")
    verified_sources = []
    for source in inventory["sources"]:
        path = (data_root / source["path"]).resolve()
        if not path.is_relative_to(data_root / "metadata"):
            raise ValueError("Inventory source escaped metadata directory")
        if sha(metadata_bytes(path)) != source["sha256"]:
            raise ValueError("Pinned metadata source changed")
        verified_sources.append({"path": source["path"], "sha256": source["sha256"]})
    prompts = parse_prompt_metadata(metadata_bytes(data_root / "metadata/TIMIT-PROMPTS.txt"))
    groups = defaultdict(dict); split_ids = defaultdict(set); split_texts = defaultdict(set)
    for record in inventory["records"]:
        speaker, prompt, role = record["speaker"], record["prompt"], record["role"]
        expected_split = next(split for split, pair_ids in assignment.items() if int(speaker[1:]) in pair_ids)
        if record["split"] != expected_split or record["primary_eligible"] != (prompt not in {"sa1", "sa2"}):
            raise ValueError("Inventory split or universal-prompt exclusion differs")
        key = speaker + "_" + prompt
        if record["recording_id"] != key or role in groups[key] or role not in {"mic", "ref", "lar"}:
            raise ValueError("Duplicate or inconsistent role identity")
        groups[key][role] = record
        if record["primary_eligible"]:
            split_ids[expected_split].add(prompt); split_texts[expected_split].add(prompts[prompt])
    if any(set(roles) != {"mic", "ref", "lar"} for roles in groups.values()):
        raise ValueError("Incomplete metadata role triplet")
    intersections = {}
    for left, right in itertools.combinations(assignment, 2):
        ids = split_ids[left] & split_ids[right]; texts = split_texts[left] & split_texts[right]
        intersections[left + "__" + right] = {"prompt_id_count": len(ids), "normalized_text_count": len(texts)}
        if ids or texts:
            raise ValueError("Cross-split prompt leakage")
    selected = []; speaker_counts = {}; selection_order = {}
    for speaker in SPEAKERS:
        choices = {p: roles["mic"] for key, roles in groups.items() if (p := roles["mic"]["prompt"])
                   and roles["mic"]["speaker"] == speaker and roles["mic"]["primary_eligible"]}
        if len(choices) < per_speaker:
            raise ValueError("Insufficient eligible sealed inventory")
        ordered = sorted(choices, key=lambda p: (sha(f"{SEED}:{int(speaker[1:])}:{p}".encode()), p))[:per_speaker]
        selection_order[speaker] = ordered
        speaker_counts[speaker] = {"eligible": len(choices), "selected": len(ordered),
            "prompt_families": dict(Counter(p[:2] for p in ordered)), "ordered_prompt_ids_sha256": sha("\n".join(ordered).encode())}
        selected.extend({"recording_id": speaker + "_" + p, "speaker": speaker,
            "pair": int(speaker[1:]), "prompt": p, "split": "sealed_test"} for p in ordered)
    for pair in (5, 6):
        if selection_order[f"F{pair:02}"] != selection_order[f"M{pair:02}"]:
            raise ValueError("Matched-sex prompt selection differs")
    acquisition_path = data_root / "manifest.json"
    acquisition_raw = metadata_bytes(acquisition_path); acquisition = json.loads(acquisition_raw)
    if acquisition["inventory_sha256"] != sha(inventory_raw):
        raise ValueError("Acquisition ledger has a different inventory")
    sealed_acquired = [r["recording_id"] for r in acquisition["files"] if r["split"] == "sealed_test" or r["speaker"] in SPEAKERS]
    if sealed_acquired:
        raise ValueError("Pre-access audit found acquired sealed files")
    filesystem_hits = []
    for base in (data_root, runs_root):
        for path in base.rglob("*"):
            if path.is_file() and SEALED_FILE.search(path.name):
                filesystem_hits.append(str(path.relative_to(base)))
    if filesystem_hits:
        raise ValueError("Pre-access audit found sealed corpus filenames on disk")
    history = []
    for path in sorted(runs_root.rglob("*.json")):
        if path.name not in HISTORY_NAMES:
            continue
        raw = metadata_bytes(path); value = json.loads(raw)
        if mentioned_sealed_records(value) or isinstance(value, dict) and value.get("split") in {"sealed_test", "sealed-test"}:
            raise ValueError(f"Prior run metadata names sealed content: {path.relative_to(runs_root)}")
        history.append({"path": str(path.relative_to(runs_root)), "sha256": sha(raw)})
    counts = Counter((r["split"], r["role"]) for r in acquisition["files"])
    return {"schema": "phenometrix-sealed-pre-access-metadata-audit-v1", "status": "pass",
        "audited_at": datetime.now(timezone.utc).isoformat(), "audit_source_sha256": sha(Path(__file__).read_bytes()),
        "inventory_sha256": sha(inventory_raw), "acquisition_manifest_sha256": sha(acquisition_raw),
        "split_assignment": assignment, "selection_algorithm": "ascending SHA256 UTF8(20261022:integer_pair:prompt), then prompt ID; fixed count per speaker",
        "per_speaker": per_speaker, "selected_recording_count": len(selected),
        "selected_unique_prompt_count": len({r["prompt"] for r in selected}),
        "selected_identity_sha256": sha(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()),
        "selected_recordings": selected, "speaker_selection": speaker_counts,
        "cross_split_intersections": intersections, "verified_metadata_sources": verified_sources,
        "acquired_counts": [{"split": split, "role": role, "files": count} for (split, role), count in sorted(counts.items())],
        "sealed_files_in_acquisition_ledger": 0, "sealed_corpus_filename_hits": [],
        "prior_run_metadata_without_sealed_record_ids": history, "prior_metadata_files_checked": len(history),
        "audio_or_reference_content_opened": False,
        "independence": "Four unseen speaker IDs in two matched prompt pairs. Prompt bootstrap is conditional on these four speakers; frames, conditions and matched-sex prompt repetitions are not independent people.",
        "limits": ["No sealed content in the checked acquisition ledger, corpus paths or historical manifest scope; this is not an operating-system access log or proof of all external activity.",
            "Public filename, speaker and prompt metadata were inspected before access; sealed acoustic/reference outcomes were not.",
            "Corpus recording protocol, microphone setup and extraction procedure are shared with training. This is within-corpus unseen-speaker evidence, not an external cohort.",
            "Reference extraction timing uncertainty remains; exact nominal scoring does not establish physiological alignment."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    research = Path(__file__).resolve().parents[1]
    p.add_argument("--data", type=Path, default=research / "data/ptdb-tug")
    p.add_argument("--runs", type=Path, default=research / "runs")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite pre-access evidence")
    result = audit_metadata(args.data, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "selected_recordings": result["selected_recording_count"],
        "prior_manifests": result["prior_metadata_files_checked"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
