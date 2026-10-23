"""PTDB-TUG metadata audit and bounded, split-aware acquisition (stdlib only).

The inventory reads directory listings and prompt metadata, never audio/reference
contents. References and laryngographs are separate target/audit roles; they must
never become inference features. Provider archive checksums do not authenticate
individual files. SHA-256 recorded here pins downloaded bytes, not label truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import wave

BASE = "https://www2.spsc.tugraz.at/databases/PTDB-TUG/"
SEED = 20261022
MAX_BYTES = 1 << 30
UNIVERSAL = {"sa1", "sa2"}
ROLES = ("mic", "ref", "lar")
FILE_PATTERN = re.compile(r"(mic|lar|ref)_([FM]\d{2})_(s[axi]\d+)\.(wav|f0)$")
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "ptdb-tug"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def split_assignment(seed=SEED):
    pairs = list(range(1, 11))
    random.Random(seed).shuffle(pairs)
    return {"train": pairs[:6], "development": pairs[6:8], "sealed_test": pairs[8:]}


def speaker_split(speaker, seed=SEED):
    if not re.fullmatch(r"[FM](0[1-9]|10)", speaker):
        raise ValueError(f"Invalid speaker: {speaker}")
    return next(name for name, pairs in split_assignment(seed).items() if int(speaker[1:]) in pairs)


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_url(url, max_bytes=8 << 20):
    """Read bounded public metadata bytes; failures are never replaced by guesses."""
    with urlopen(Request(url, headers={"User-Agent": "PhenoMetrix-research/0.1"}), timeout=60) as response:
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"Metadata exceeds {max_bytes} byte limit: {url}")
        return data, {"url": url, "resolved_url": response.url, "downloaded_at": utc_now(),
                      "bytes": len(data), "sha256": digest(data),
                      "last_modified": response.headers.get("Last-Modified"),
                      "etag": response.headers.get("ETag")}


class ListingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.entries = []
        self.href = None
        self.in_row = False
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.in_row, self.href, self.text = True, None, []
        if tag == "a" and self.in_row:
            href = dict(attrs).get("href", "")
            if FILE_PATTERN.fullmatch(href):
                self.href = href

    def handle_data(self, data):
        if self.in_row:
            self.text.append(data.strip())

    def handle_endtag(self, tag):
        if tag == "tr":
            if self.href:
                self.entries.append((self.href, " ".join(x for x in self.text if x)))
            self.in_row = False


def parse_listing(text, url, speaker, role):
    parser = ListingParser()
    parser.feed(text)
    records = []
    for filename, display in parser.entries:
        found_role, found_speaker, prompt, extension = FILE_PATTERN.fullmatch(filename).groups()
        if (found_role, found_speaker) != (role, speaker):
            raise ValueError(f"Directory identity mismatch: {filename}")
        if extension != ("f0" if role == "ref" else "wav"):
            raise ValueError(f"Unexpected role extension: {filename}")
        records.append({"recording_id": f"{speaker}_{prompt}", "speaker": speaker,
                        "pair": int(speaker[1:]), "prompt": prompt, "role": role,
                        "filename": filename, "url": urljoin(url, filename),
                        "split": speaker_split(speaker), "primary_eligible": prompt not in UNIVERSAL,
                        "listing_display": display, "source_file_checksum": None})
    if not records or len({r["filename"] for r in records}) != len(records):
        raise ValueError(f"Empty or duplicate directory inventory: {url}")
    return records


def parse_prompts(text):
    prompts = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        match = re.fullmatch(r"(.+?)\s+\((s[axi]\d+)\)\s*", line)
        if not match:
            raise ValueError(f"Unrecognized prompt metadata line: {line[:80]}")
        sentence, prompt = match.groups()
        if prompt in prompts:
            raise ValueError(f"Duplicate prompt: {prompt}")
        prompts[prompt] = " ".join(sentence.casefold().split())
    return prompts


def inventory_audit(records, prompts):
    roles = defaultdict(dict)
    split_prompts = defaultdict(set)
    split_texts = defaultdict(set)
    speakers = defaultdict(set)
    for record in records:
        role_map = roles[record["recording_id"]]
        if record["role"] in role_map:
            raise ValueError(f"Duplicate role in inventory: {record['recording_id']}")
        role_map[record["role"]] = record
        if record["prompt"] not in prompts:
            raise ValueError(f"Unknown prompt: {record['prompt']}")
        if record["primary_eligible"]:
            split_prompts[record["split"]].add(record["prompt"])
            split_texts[record["split"]].add(prompts[record["prompt"]])
            speakers[record["speaker"]].add(record["prompt"])
    incomplete = {key: sorted(set(ROLES) - set(value)) for key, value in roles.items() if set(value) != set(ROLES)}
    intersections = {}
    for left, right in (("train", "development"), ("train", "sealed_test"), ("development", "sealed_test")):
        intersections[f"{left}__{right}"] = {"prompt_ids": sorted(split_prompts[left] & split_prompts[right]),
            "normalized_text_count": len(split_texts[left] & split_texts[right])}
    pair_differences = {f"{pair:02}": {"female_only": sorted(speakers[f"F{pair:02}"] - speakers[f"M{pair:02}"]),
        "male_only": sorted(speakers[f"M{pair:02}"] - speakers[f"F{pair:02}"])} for pair in range(1, 11)}
    expected_missing = []
    for sex in "FM":
        for pair in range(1, 11):
            speaker = f"{sex}{pair:02}"
            expected = {f"sx{i}" for i in range(3 + 45*(pair-1), 3 + 45*pair)} | {f"si{i}" for i in range(453 + 189*(pair-1), 453 + 189*pair)}
            expected_missing.extend(f"{speaker}_{prompt}" for prompt in sorted(expected - speakers[speaker]))
            unexpected = speakers[speaker] - expected
            if unexpected:
                raise ValueError(f"Unexpected prompt membership for {speaker}: {unexpected}")
    overlap = any(value["prompt_ids"] or value["normalized_text_count"] for value in intersections.values())
    return {"recordings": len(roles), "files": len(records),
            "files_per_role": dict(Counter(r["role"] for r in records)),
            "recordings_per_speaker": dict(sorted(Counter(r["speaker"] for r in records if r["role"] == "mic").items())),
            "primary_recordings": sum(r["role"] == "mic" and r["primary_eligible"] for r in records),
            "excluded_universal_recordings": sum(r["role"] == "mic" and not r["primary_eligible"] for r in records),
            "primary_prompt_intersections": intersections, "matched_pair_differences": pair_differences,
            "missing_from_documented_membership": expected_missing, "incomplete_recordings": incomplete,
            "no_shared_prompt_gate": not overlap and not incomplete}


def build_inventory(root=DEFAULT_ROOT):
    root = Path(root)
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    sources = []
    documents = {
        "provider.html": "https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html",
        "SPEAKER-PROFILES.txt": BASE + "DOCUMENTATION/SPEAKER-PROFILES.txt",
        "RECORDING-PROTOCOL.txt": BASE + "DOCUMENTATION/RECORDING-PROTOCOL.txt",
        "TIMIT-PROMPTS.txt": BASE + "DOCUMENTATION/TIMIT-PROMPTS.txt",
        "archive.md5": BASE + "SPEECH_DATA_ZIPPED.zip.md5",
        "archive.sha1": BASE + "SPEECH_DATA_ZIPPED.zip.sha1",
        "ODbL-1.0.txt": "https://opendatacommons.org/licenses/odbl/odbl-10.txt",
        "DbCL-1.0.txt": "https://opendatacommons.org/licenses/dbcl/dbcl-10.txt",
    }
    for name, url in documents.items():
        data, source = read_url(url)
        (metadata / name).write_bytes(data)
        source["path"] = str((metadata / name).relative_to(root))
        sources.append(source)
    prompts = parse_prompts((metadata / "TIMIT-PROMPTS.txt").read_text())
    jobs = [(sex, pair, role) for sex in "FM" for pair in range(1, 11) for role in ROLES]

    def listing(job):
        sex, pair, role = job
        speaker = f"{sex}{pair:02}"
        gender = "FEMALE" if sex == "F" else "MALE"
        url = BASE + f"SPEECH%20DATA/{gender}/{role.upper()}/{speaker}/"
        data, source = read_url(url)
        destination = metadata / "listings" / f"{speaker}-{role}.html"
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(data)
        source["path"] = str(destination.relative_to(root))
        return parse_listing(data.decode("utf-8"), url, speaker, role), source

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for entries, source in pool.map(listing, jobs):
            records.extend(entries)
            sources.append(source)
    audit = inventory_audit(records, prompts)
    result = {"schema": "phenometrix-ptdb-inventory-v1", "created_at": utc_now(), "seed": SEED,
              "split_assignment": split_assignment(), "selection_inputs": "public filename and prompt metadata only",
              "sources": sources, "records": records, "audit": audit}
    json_write(metadata / "inventory.json", result)
    if not audit["no_shared_prompt_gate"]:
        raise ValueError("Inventory is saved but prompt/role audit failed; acquisition is blocked")
    return result


def select_records(inventory, split="train", per_speaker=2, roles=ROLES):
    if split not in split_assignment() or per_speaker < 1 or not set(roles) <= set(ROLES):
        raise ValueError("Invalid split, count, or role")
    if not inventory["audit"]["no_shared_prompt_gate"]:
        raise ValueError("Inventory prompt/role audit has not passed")
    grouped = defaultdict(dict)
    for record in inventory["records"]:
        if record["split"] == split and record["primary_eligible"]:
            grouped[record["speaker"]].setdefault(record["prompt"], []).append(record)
    selected = []
    for speaker in sorted(grouped):
        ordered = sorted(grouped[speaker], key=lambda prompt: (digest(f"{SEED}:{int(speaker[1:])}:{prompt}".encode()), prompt))
        for prompt in ordered[:per_speaker]:
            selected.extend(r for r in grouped[speaker][prompt] if r["role"] in roles)
    return selected


def authorize_split(split, inventory_path, freeze_path=None):
    if split in ("train", "development"):
        return
    if split != "sealed_test" or freeze_path is None:
        raise ValueError("Sealed-test acquisition requires explicit --split sealed_test and --freeze")
    freeze = json.loads(Path(freeze_path).read_text())
    required = ("source_revision", "protocol_sha256", "model_sha256", "configuration_sha256")
    if (freeze.get("schema") != "phenometrix-prosody-freeze-v1"
        or freeze.get("sealed_test_authorized") is not True
        or any(not isinstance(freeze.get(key), str) or not freeze[key] for key in required)
        or freeze.get("inventory_sha256") != digest(Path(inventory_path).read_bytes())):
        raise ValueError("Invalid or mismatched final freeze; sealed test remains closed")


def tree_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def download_record(record, root, max_bytes=MAX_BYTES):
    root = Path(root)
    url = record["url"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "www2.spsc.tugraz.at" or not url.startswith(BASE + "SPEECH%20DATA/"):
        raise ValueError("Recording must come from the official HTTPS dataset directory")
    expected = f"{record['role']}_{record['speaker']}_{record['prompt']}." + ("f0" if record["role"] == "ref" else "wav")
    if record["filename"] != expected or parsed.path.rsplit("/", 1)[-1] != expected:
        raise ValueError("Recording identity mismatch")
    if speaker_split(record["speaker"]) != record["split"] or record["prompt"] in UNIVERSAL:
        raise ValueError("Split or universal-prompt exclusion mismatch")
    destination = root / "records" / record["split"] / record["speaker"] / record["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Existing file requires manifest verification: {destination}")
    remaining = max_bytes - tree_bytes(root) - (1 << 20)  # reserve manifest/audit growth
    if remaining <= 0 or shutil.disk_usage(root).free < min(remaining, 64 << 20):
        raise ValueError("Data budget or free-space reserve exhausted")
    temporary = None
    try:
        with urlopen(Request(url, headers={"User-Agent": "PhenoMetrix-research/0.1"}), timeout=60) as response:
            if response.url != url:
                raise ValueError("Unexpected recording redirect")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > remaining:
                raise ValueError("File exceeds remaining data budget")
            sha, total = hashlib.sha256(), 0
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                while block := response.read(min(65536, remaining - total + 1)):
                    total += len(block)
                    if total > remaining:
                        raise ValueError("Transfer exceeded remaining data budget")
                    sha.update(block)
                    handle.write(block)
            if length is not None and total != int(length):
                raise ValueError("Truncated transfer")
            temporary.replace(destination)
            temporary = None
            return {**record, "path": str(destination.relative_to(root)), "bytes": total,
                    "sha256": sha.hexdigest(), "downloaded_at": utc_now(),
                    "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag"),
                    "source_checksum_status": "not published per file; archive checksums do not apply"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_reference(path):
    rows = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"Reference row {index + 1} must have four columns")
        row = tuple(float(value) for value in fields)
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"Nonfinite reference row {index + 1}")
        rows.append(row)
    if not rows:
        raise ValueError("Empty reference")
    return rows


def audit_acquired(root, entries):
    root = Path(root)
    grouped = defaultdict(dict)
    for entry in entries:
        grouped[entry["recording_id"]][entry["role"]] = entry
    reports = []
    for recording, roles in sorted(grouped.items()):
        details = {"recording_id": recording, "split": next(iter(roles.values()))["split"]}
        for role in ("mic", "lar"):
            if role in roles:
                with wave.open(str(root / roles[role]["path"]), "rb") as handle:
                    details[role] = {"sample_rate": handle.getframerate(), "channels": handle.getnchannels(),
                                     "sample_width": handle.getsampwidth(), "sample_count": handle.getnframes(),
                                     "compression": handle.getcomptype()}
                if any(details[role][key] != expected for key, expected in (("sample_rate", 48000), ("channels", 1), ("sample_width", 2), ("compression", "NONE"))):
                    raise ValueError(f"Unexpected audio format: {recording}/{role}")
        if "mic" in details and "lar" in details:
            details["paired_sample_count_equal"] = details["mic"]["sample_count"] == details["lar"]["sample_count"]
            if not details["paired_sample_count_equal"]:
                raise ValueError(f"Paired sample lengths disagree: {recording}")
        if "ref" in roles:
            rows = read_reference(root / roles["ref"]["path"])
            f0 = [r[0] for r in rows]
            details["reference"] = {"rows": len(rows), "first_target_ms": 16,
                "last_target_ms": 16 + 10 * (len(rows) - 1), "hop_ms": 10,
                "positive_f0_min_hz": min((v for v in f0 if v > 0), default=None),
                "positive_f0_max_hz": max((v for v in f0 if v > 0), default=None),
                "nonpositive_f0_rows": sum(v <= 0 for v in f0),
                "voicing_values": sorted({r[1] for r in rows}),
                "positive_f0_zero_voicing_rows": sum(r[0] > 0 and r[1] == 0 for r in rows),
                "nonpositive_f0_positive_voicing_rows": sum(r[0] <= 0 and r[1] > 0 for r in rows)}
            if "mic" in details:
                details["reference"]["trailing_after_last_target_ms"] = details["mic"]["sample_count"] / 48 - details["reference"]["last_target_ms"]
        reports.append(details)
    return {"schema": "phenometrix-ptdb-file-audit-v1", "audited_at": utc_now(), "recordings": reports,
        "timing_status": "documented nominal 16 + 10*k ms; high-pass delay compensation unresolved",
        "alignment_status": "sample-format and paired-length audit only; physiological transition audit pending",
        "reference_columns": ["f0_hz", "voicing_probability", "local_rms", "peak_normalized_cross_correlation"]}


def acquire(root=DEFAULT_ROOT, split="train", per_speaker=2, roles=ROLES, max_bytes=MAX_BYTES, freeze_path=None):
    root = Path(root)
    inventory_path = root / "metadata" / "inventory.json"
    authorize_split(split, inventory_path, freeze_path)
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("seed") != SEED or inventory.get("split_assignment") != split_assignment():
        raise ValueError("Inventory split differs from frozen assignment")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema": "phenometrix-ptdb-acquisition-v1", "inventory_sha256": digest(inventory_path.read_bytes()),
        "licenses": {"database": "ODbL-1.0", "contents": "DbCL-1.0"}, "files": []}
    if manifest["inventory_sha256"] != digest(inventory_path.read_bytes()):
        raise ValueError("Manifest belongs to a different inventory snapshot")
    existing = {entry["url"]: entry for entry in manifest["files"]}
    for record in select_records(inventory, split, per_speaker, roles):
        if record["url"] in existing:
            entry = existing[record["url"]]
            path = root / entry["path"]
            if not path.is_file() or digest(path.read_bytes()) != entry["sha256"]:
                raise ValueError(f"Acquired file integrity failed: {entry['path']}")
            continue
        entry = download_record(record, root, max_bytes)
        manifest["files"].append(entry)
        manifest["updated_at"] = utc_now()
        manifest["max_data_bytes"] = max_bytes
        json_write(manifest_path, manifest)
    relevant = [entry for entry in manifest["files"] if entry["split"] == split]
    audit = audit_acquired(root, relevant)
    json_write(root / f"audit-{split}.json", audit)
    return {"manifest": str(manifest_path), "files": len(manifest["files"]),
            "data_tree_bytes": tree_bytes(root), "audit": str(root / f"audit-{split}.json")}



def training_cycle_probe(root=DEFAULT_ROOT, recording_id="F04_si1146"):
    """Inspect five deterministic training-only raw LAR intervals; not a scorer.

    Probe midpoints of the five longest reference-positive runs, using 32 ms
    centered raw first-difference windows. Select the shortest local ACF peak
    within 90% of the largest normalized peak in 50..700 Hz. This exploratory
    cycle check cannot establish boundary timing or filter-delay compensation.
    """
    import numpy as np
    from scipy.signal import correlate, find_peaks

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = {entry["role"]: entry for entry in manifest["files"] if entry["recording_id"] == recording_id}
    if not {"lar", "ref"} <= set(entries) or any(entry["split"] != "train" for entry in entries.values()):
        raise ValueError("Cycle probes require acquired training-only LAR and reference files")
    reference = np.asarray(read_reference(root / entries["ref"]["path"]))
    with wave.open(str(root / entries["lar"]["path"]), "rb") as handle:
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(float) / 32768
    voiced = (reference[:, 0] > 0) & (reference[:, 1] > 0)
    edges = np.diff(np.r_[False, voiced, False].astype(int))
    runs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
    runs.sort(key=lambda run: (-(run[1] - run[0]), run[0]))
    result = []
    for begin, end in runs[:5]:
        index = int((begin + end) // 2)
        target_sample = 768 + 480 * index
        values = np.diff(samples[target_sample - 768:target_sample + 768])
        values -= values.mean()
        correlation = correlate(values, values, mode="full", method="direct")[len(values) - 1:]
        norm = np.array([np.sqrt(np.dot(values[:-lag], values[:-lag]) * np.dot(values[lag:], values[lag:])) for lag in range(1, 961)])
        normalized = correlation[1:961] / np.maximum(norm, 1e-18)
        peaks, _ = find_peaks(normalized)
        peaks = peaks[(peaks >= round(48000 / 700) - 1) & (peaks <= round(48000 / 50) - 1)]
        if len(peaks) == 0:
            result.append({"reference_index": index, "status": "no_cycle_peak"})
            continue
        best = max(normalized[peaks])
        lag = int(min(peak + 1 for peak in peaks if normalized[peak] >= 0.9 * best))
        frequency = 48000 / lag
        result.append({"reference_index": index, "nominal_target_ms": target_sample / 48,
            "voiced_run_indices": [int(begin), int(end)], "reference_hz": float(reference[index, 0]),
            "raw_derivative_acf_hz": frequency, "selected_cycle_samples": lag,
            "normalized_correlation": float(normalized[lag - 1]),
            "absolute_cents": float(abs(1200 * np.log2(frequency / reference[index, 0])))})
    output = {"schema": "phenometrix-training-cycle-probe-v1", "recording_id": recording_id,
        "split": "train", "lar_sha256": entries["lar"]["sha256"], "reference_sha256": entries["ref"]["sha256"],
        "method": training_cycle_probe.__doc__, "probes": result,
        "interpretation": "Partial cycle correspondence, not a timing calibration. No reference shifts applied; high-pass delay compensation unresolved."}
    json_write(root / "alignment-probe-train.json", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "acquire", "probe"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=tuple(split_assignment()), default="train")
    parser.add_argument("--per-speaker", type=int, default=2)
    parser.add_argument("--roles", default=",".join(ROLES))
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.command == "inventory":
        result = build_inventory(args.root)
        print(json.dumps({"inventory": str(args.root / "metadata" / "inventory.json"), "audit": result["audit"]}, indent=2))
    elif args.command == "probe":
        print(json.dumps(training_cycle_probe(args.root), indent=2))
    else:
        print(json.dumps(acquire(args.root, args.split, args.per_speaker, tuple(args.roles.split(",")), args.max_bytes, args.freeze), indent=2))


if __name__ == "__main__":
    main()
