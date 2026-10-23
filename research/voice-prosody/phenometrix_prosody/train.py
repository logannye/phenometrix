"""Bounded local MPS training for the explicitly declared measurement-delay task."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model import VoiceCandidateModel, ModelConfig, model_metadata
from .pilot import references_from_array, digest, REPO
from .measurement_alignment import align_feature_timing


def training_labels(feature, reference_values, *, measurement_delay_ms):
    timing = align_feature_timing(feature, measurement_delay_ms=measurement_delay_ms)
    references = references_from_array(reference_values, int(feature["pcm_sample_count"]))
    by_sample = {r.sample: r for r in references}
    n = len(feature["features"])
    voiced = np.zeros(n, np.float32)
    valid = np.zeros(n, bool)
    target = np.full(n, -1, np.int64)
    for i, sample in enumerate(timing.reference_sample):
        r = by_sample.get(int(sample))
        if r is None or not r.valid or not timing.common_eligible[i]:
            continue
        valid[i] = True
        voiced[i] = float(r.voiced)
        if not r.voiced:
            continue
        frequencies = feature["candidate_hz"][i]
        errors = np.full(len(frequencies), np.inf)
        supported = feature["candidate_mask"][i]
        errors[supported] = np.abs(1200 * np.log2(frequencies[supported] / r.f0_hz))
        candidate = int(np.argmin(errors))
        if errors[candidate] <= 50:
            target[i] = candidate
    return voiced, valid, target


def supervised_loss(logits, candidates, voiced, valid, targets):
    """Uncovered voiced frames retain voicing loss; no false candidate target."""
    vloss = (F.binary_cross_entropy_with_logits(logits, voiced, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    covered = valid & (targets >= 0)
    raw = F.cross_entropy(candidates.reshape(-1, candidates.shape[-1]), targets.clamp_min(0).reshape(-1), reduction="none").reshape_as(targets)
    closs = (raw * covered).sum() / covered.sum().clamp_min(1)
    return vloss + closs


def train(feature_run, data_root, output, kind, seed, epochs=12, batch_size=64, chunk_length=128, synthetic_curriculum=None):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Local Metal GPU is required for this training run")
    if epochs < 1 or epochs > 30 or batch_size < 1 or chunk_length < 1:
        raise ValueError("Invalid bounded training configuration")
    source = json.loads((feature_run / "manifest.json").read_text())
    if source.get("status") != "complete" or source.get("split") != "train" or source.get("measurement_delay_ms") != 20:
        raise ValueError("Training requires a completed train-only declared-20ms cache")
    data_path = feature_run / "acquisition-manifest.json"
    if digest(data_path) != source["data_manifest_sha256"]:
        raise ValueError("Frozen acquisition manifest changed")
    data = json.loads(data_path.read_text())
    refs = {r["recording_id"]: r for r in data["files"] if r["role"] == "ref" and r["split"] == "train"}
    config = ModelConfig(kind=kind, feature_version=source["feature_version"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.mps.manual_seed(seed)
    model = VoiceCandidateModel(config)
    sequences = []
    source_hashes = {}
    reference_hashes = {}
    for record in source["recordings"]:
        identity = record["recording_id"]
        source_id = record["source_recording_id"]
        if source_id not in refs:
            raise ValueError("Missing training reference or split mismatch")
        file = feature_run / "features" / (identity + ".npz")
        if digest(file) != record["feature_sha256"]:
            raise ValueError("Feature identity changed")
        with np.load(file, allow_pickle=False) as feature:
            if list(feature["feature_names"]) != model_metadata(model)["feature_names"]:
                raise ValueError("Feature ordering mismatch")
            reference = data_root / refs[source_id]["path"]
            if digest(reference) != refs[source_id]["sha256"]:
                raise ValueError("Reference identity changed")
            voiced, valid, target = training_labels(feature, np.loadtxt(reference), measurement_delay_ms=20)
            sequences.append({"id": identity, "x": torch.from_numpy(feature["features"].copy()),
                "mask": torch.from_numpy(feature["candidate_mask"].copy()), "voiced": torch.from_numpy(voiced),
                "valid": torch.from_numpy(valid), "target": torch.from_numpy(target)})
        source_hashes[identity] = digest(file)
        reference_hashes[source_id] = digest(reference)
    supplement = None
    supplement_hashes = {}
    if synthetic_curriculum is not None:
        if kind != "framewise-mlp":
            raise ValueError("The one augmented follow-up is restricted to the fixed MLP")
        synthetic_curriculum = Path(synthetic_curriculum).resolve()
        supplement_path = synthetic_curriculum / "manifest.json"
        supplement = json.loads(supplement_path.read_text())
        if supplement.get("status") != "complete" or supplement.get("split") != "train":
            raise ValueError("Synthetic curriculum must be a completed training-only artifact")
        for record in supplement["records"]:
            identity = record["id"]
            file = (synthetic_curriculum / record["feature_path"]).resolve()
            reference = (synthetic_curriculum / record["reference_path"]).resolve()
            if not file.is_relative_to(synthetic_curriculum) or not reference.is_relative_to(synthetic_curriculum):
                raise ValueError("Synthetic artifact escaped its root")
            if digest(file) != record["feature_sha256"] or digest(reference) != record["reference_sha256"]:
                raise ValueError("Frozen synthetic feature/analytic-label identity changed")
            with np.load(file, allow_pickle=False) as feature:
                if list(feature["feature_names"]) != model_metadata(model)["feature_names"]:
                    raise ValueError("Synthetic feature order differs")
                if int(feature["pcm_sample_count"]) != record["pcm_sample_count"]:
                    raise ValueError("Synthetic duration differs")
                voiced, valid, target = training_labels(feature, np.loadtxt(reference), measurement_delay_ms=20)
                sequence = {"x": torch.from_numpy(feature["features"].copy()),
                    "mask": torch.from_numpy(feature["candidate_mask"].copy()), "voiced": torch.from_numpy(voiced),
                    "valid": torch.from_numpy(valid), "target": torch.from_numpy(target)}
            # Explicit replay weight, not additional independent generated data.
            for repeat in range(4):
                sequences.append({**sequence, "id": identity + "__replay" + str(repeat)})
            supplement_hashes[identity] = {"feature_sha256": digest(file), "reference_sha256": digest(reference)}
    model.fit_scaling(torch.cat([s["x"] for s in sequences]), split="train")
    model = model.to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("measurement_alignment.py"),
               Path(__file__).with_name("pilot.py"), REPO / "research/voice-prosody/PROTOCOL.md"]
    if synthetic_curriculum is not None:
        generator = Path(__file__).with_name("synthetic_curriculum.py")
        if generator.exists():
            sources.append(generator)
        (output / "synthetic-curriculum-manifest.json").write_bytes((synthetic_curriculum / "manifest.json").read_bytes())
    source_code = {str(p.relative_to(REPO)): digest(p) for p in sources}
    for p in sources:
        dest = output / "source-snapshot" / p.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
    manifest = {"schema": "phenometrix-local-voice-training-v2", "status": "running", "split": "train",
        "started_at": datetime.now(timezone.utc).isoformat(), "device": "mps", "torch": torch.__version__,
        "kind": kind, "seed": seed, "epochs": epochs, "batch_size": batch_size, "tbptt_frames": chunk_length,
        "measurement_delay_ms": 20, "learning_rate": .001, "gradient_clip_norm": 1.,
        "voicing_loss": "unweighted binary cross entropy",
        "candidate_loss": "conditional cross entropy only on voiced frames covered within 50 cents",
        "padding": "loss masked; no loss on invalid/boundary labels; skip unsupervised optimizer steps",
        "state_reset": "zero at recording boundaries; detached across chunks; never true prior F0",
        "feature_run_manifest_sha256": digest(feature_run / "manifest.json"), "feature_hashes": source_hashes,
        "reference_hashes": reference_hashes, "source_sha256": source_code, "epoch_losses": [],
        "test_or_development_labels_used": False, "conditions": source["conditions"],
        "sequence_count": len(sequences), "feature_frames": sum(len(s["x"]) for s in sequences),
        "valid_supervision_frames": sum(int(s["valid"].sum()) for s in sequences),
        "covered_candidate_frames": sum(int((s["valid"] & (s["target"] >= 0)).sum()) for s in sequences),
        "peak_observed_mps_allocated_bytes": 0,
        "synthetic_curriculum": None if supplement is None else {
            "manifest_sha256": digest(synthetic_curriculum / "manifest.json"),
            "source_kind": "analytic known-truth synthesis; not PTDB or independent participants",
            "unique_sequence_count": len(supplement["records"]), "repetitions_for_training_and_scaling": 4,
            "feature_and_reference_sha256": supplement_hashes,
            "challenge_content_or_outcomes_used": False}}
    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save_manifest()
    began = time.monotonic()
    try:
        for epoch in range(epochs):
            order = list(range(len(sequences))); random.Random(seed + epoch).shuffle(order)
            totals = []
            for offset in range(0, len(order), batch_size):
                group = [sequences[i] for i in order[offset:offset + batch_size]]
                lengths = [len(s["x"]) for s in group]
                length, batch = max(lengths), len(group)
                def padded(name, value, dtype, extra=()):
                    t = torch.full((batch, length, *extra), value, dtype=dtype)
                    for i, s in enumerate(group):
                        t[i, :lengths[i]] = s[name]
                    return t.to("mps")
                x = padded("x", 0., torch.float32, (config.feature_count,))
                masks = padded("mask", False, torch.bool, (config.candidate_count,))
                voiced = padded("voiced", 0., torch.float32)
                valid = padded("valid", False, torch.bool)
                target = padded("target", -1, torch.int64)
                state = model.initial_state(batch, device="mps")
                for start in range(0, length, chunk_length):
                    optimizer.zero_grad(set_to_none=True)
                    end = min(length, start + chunk_length)
                    if kind == "framewise-mlp":
                        # The ablation has no history; batching its time axis changes no equations.
                        v, candidates, _ = model(x[:, start:end], masks[:, start:end],
                            torch.zeros((batch, end-start, config.hidden_size), device="mps"))
                        logits = v.squeeze(-1)
                    else:
                        vl, cl = [], []
                        for t in range(start, end):
                            v, c, state = model(x[:, t], masks[:, t], state)
                            vl.append(v.squeeze(-1)); cl.append(c)
                        logits, candidates = torch.stack(vl, dim=1), torch.stack(cl, dim=1)
                    include = valid[:, start:end]
                    if bool(include.any()):
                        loss = supervised_loss(logits, candidates, voiced[:, start:end], include, target[:, start:end])
                        if not bool(torch.isfinite(loss)):
                            raise ValueError("Nonfinite training loss")
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                        totals.append(float(loss.detach().cpu()))
                    state = state.detach()
                manifest["peak_observed_mps_allocated_bytes"] = max(manifest["peak_observed_mps_allocated_bytes"], torch.mps.current_allocated_memory())
                if epoch == 0 and offset == 0:
                    print(json.dumps({"first_batch_seconds": time.monotonic() - began, "batch_size": batch, "max_frames": length}), flush=True)
            manifest["epoch_losses"].append({"epoch": epoch + 1, "mean_chunk_loss": float(np.mean(totals)),
                                             "elapsed_seconds": time.monotonic() - began})
            save_manifest(); print(json.dumps(manifest["epoch_losses"][-1]), flush=True)
        torch.mps.synchronize()
        model = model.to("cpu").eval()
        checkpoint = output / "weights.pt"
        torch.save({"config": asdict(config), "state_dict": model.state_dict()}, checkpoint)
        for name, expected in source_code.items():
            if digest(REPO / name) != expected:
                raise ValueError("Training source changed during fitting")
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - began,
            model=model_metadata(model), weights_sha256=digest(checkpoint),
            finished_at=datetime.now(timezone.utc).isoformat(),
            evidence_status="trained_on_public_training_split_not_yet_development_qualified")
        save_manifest(); return manifest
    except BaseException as error:
        manifest.update(status="failed", failure=repr(error), elapsed_seconds=time.monotonic() - began)
        save_manifest(); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True); p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=["causal-gru", "framewise-mlp"], default="causal-gru")
    p.add_argument("--seed", type=int, default=20261022); p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--synthetic-curriculum", type=Path)
    a = p.parse_args()
    result = train(a.features.resolve(), a.data.resolve(), a.output.resolve(), a.kind, a.seed, a.epochs, synthetic_curriculum=a.synthetic_curriculum)
    print(json.dumps({k: result[k] for k in ["status", "kind", "elapsed_seconds", "weights_sha256"]}, indent=2))

if __name__ == "__main__":
    main()
