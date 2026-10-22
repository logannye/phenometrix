"""Explicit-state model export and arithmetic parity; no automatic quality qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .model import ModelConfig, VoiceCandidateModel, model_metadata, numpy_step


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_spike(output: Path, *, seed: int = 20261022, kind: str = "causal-gru",
                 check_mps: bool = False, feature_version: str = "v2") -> dict:
    torch.manual_seed(seed)
    model = VoiceCandidateModel(ModelConfig(kind=kind, feature_version=feature_version)).eval()
    return export_model(model, output, seed=seed, check_mps=check_mps,
                        evidence_status="untrained-random-weight-synthetic-export-spike")


def export_model(model: VoiceCandidateModel, output: Path, *, seed: int = 20261022,
                 check_mps: bool = False, evidence_status: str = "provided-weights-export-unqualified",
                 training_metadata: dict | None = None) -> dict:
    """Export provided weights; caller supplies training provenance if it exists.

    This verifies arithmetic only. It neither trains weights nor establishes
    held-out quality, calibration, browser timing or a deployable operating point.
    """
    if training_metadata is not None and not bool(model.scaling_fitted.item()):
        raise ValueError("A trained export must retain its training-fitted scaling")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = model.cpu().eval()
    generator = np.random.default_rng(seed)
    features = generator.normal(0, 1, (24, 1, model.config.feature_count)).astype(np.float32)
    masks = generator.integers(0, 2, (24, 1, model.config.candidate_count), dtype=np.int32).astype(bool)
    masks[0] = False
    masks[1] = True
    graph = output / "model.onnx"
    state = model.initial_state()
    with torch.no_grad():
        torch.onnx.export(model, (torch.from_numpy(features[0]), torch.from_numpy(masks[0]), state),
                          str(graph), input_names=["features", "candidate_mask", "state_in"],
                          output_names=["voicing_logit", "candidate_logits", "state_out"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(graph)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
    weights = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    weight_path = output / "weights.npz"
    np.savez(weight_path, **weights)
    numpy_state = np.zeros((1, model.config.hidden_size), np.float32)
    onnx_state = numpy_state.copy()
    fixtures = []
    maximum_error = {"numpy": 0.0, "onnxruntime": 0.0}
    with torch.no_grad():
        for i, (feature, mask) in enumerate(zip(features, masks)):
            state_before = state.detach().numpy().copy()
            expected = model(torch.from_numpy(feature), torch.from_numpy(mask), state)
            independent = numpy_step(model.config, weights, feature, mask, numpy_state)
            actual = session.run(None, {"features": feature, "candidate_mask": mask, "state_in": onnx_state})
            for name, result in [("numpy", independent), ("onnxruntime", actual)]:
                for target, observed in zip(expected, result):
                    reference = target.numpy()
                    error = float(np.max(np.abs(reference - observed)))
                    maximum_error[name] = max(maximum_error[name], error)
                    np.testing.assert_allclose(observed, reference, rtol=1e-5, atol=1e-6)
            fixtures.append({"index": i, "features": feature.reshape(-1).tolist(),
                             "candidateMask": mask.reshape(-1).tolist(),
                             "stateIn": state_before.reshape(-1).tolist(),
                             "voicingLogit": expected[0].numpy().reshape(-1).tolist(),
                             "candidateLogits": expected[1].numpy().reshape(-1).tolist(),
                             "stateOut": expected[2].numpy().reshape(-1).tolist()})
            state = expected[2]
            numpy_state = independent[2]
            onnx_state = actual[2]
    fixture_path = output / "parity-fixtures.json"
    fixture_path.write_text(json.dumps({"schemaVersion": "phenometrix.voice-model-parity.v1",
        "seed": seed, "atol": 1e-6, "rtol": 1e-5, "fixtures": fixtures}, indent=2) + "\n")
    mps = {"available": torch.backends.mps.is_available(), "checked": False}
    if check_mps and mps["available"]:
        model.to("mps")
        with torch.no_grad():
            observed = model(torch.from_numpy(features[1]).to("mps"), torch.from_numpy(masks[1]).to("mps"),
                             model.initial_state(device="mps"))
        model.cpu()
        with torch.no_grad():
            expected = model(torch.from_numpy(features[1]), torch.from_numpy(masks[1]), model.initial_state())
        mps_error = max(float(np.max(np.abs(a.detach().cpu().numpy() - b.numpy())))
                        for a, b in zip(observed, expected))
        if mps_error > 1e-5:
            raise AssertionError(f"MPS step parity error {mps_error}")
        mps.update({"checked": True, "maximum_absolute_error": mps_error,
                    "claim": "single-step execution feasibility only; no acceleration or latency claim"})
    manifest = {**model_metadata(model), "evidence_status": evidence_status,
        "seed": seed, "source_labels_used": training_metadata is not None,
        "training_performed": training_metadata is not None, "training_metadata": training_metadata,
        "export_source_sha256": {name: sha256(Path(__file__).parent / name) for name in ["model.py", "export.py"]},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "artifacts": {"onnx": {"path": graph.name, "sha256": sha256(graph)},
                      "weights": {"path": weight_path.name, "sha256": sha256(weight_path)},
                      "parity_fixtures": {"path": fixture_path.name, "sha256": sha256(fixture_path)}},
        "parity": {"steps": len(fixtures), "absolute_tolerance": 1e-6,
                   "relative_tolerance": 1e-5, "maximum_absolute_error": maximum_error,
                   "providers": session.get_providers()}, "mps": mps,
        "browser_status": "not-tested-by-this-Python-export", "status": "complete"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_checkpoint(checkpoint_dir: Path, output: Path, *, check_mps: bool = False) -> dict:
    manifest_path = checkpoint_dir / "manifest.json"
    training = json.loads(manifest_path.read_text())
    weights_path = checkpoint_dir / "weights.pt"
    if (training.get("status") != "complete" or training.get("split") != "train"
            or training.get("test_or_development_labels_used") is not False
            or training.get("measurement_delay_ms") != 20
            or sha256(weights_path) != training.get("weights_sha256")):
        raise ValueError("Export requires a verified completed train-only 20 ms checkpoint")
    saved = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = VoiceCandidateModel(ModelConfig(**saved["config"]))
    model.load_state_dict(saved["state_dict"], strict=True)
    if model_metadata(model) != training.get("model"):
        raise ValueError("Checkpoint model contract differs from training manifest")
    provenance = {"training_manifest_sha256": sha256(manifest_path),
                  "weights_sha256": training["weights_sha256"],
                  "feature_run_manifest_sha256": training["feature_run_manifest_sha256"],
                  "measurement_delay_ms": 20, "kind": training["kind"], "seed": training["seed"]}
    return export_model(model, output, seed=training["seed"], check_mps=check_mps,
                        evidence_status="trained-export-arithmetic-only-not-deployment-qualified",
                        training_metadata=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--spike", action="store_true")
    mode.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--kind", choices=["causal-gru", "framewise-mlp"], default="causal-gru")
    parser.add_argument("--check-mps", action="store_true")
    parser.add_argument("--feature-version", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()
    if args.checkpoint is not None:
        result = export_checkpoint(args.checkpoint, args.output, check_mps=args.check_mps)
    else:
        result = export_spike(args.output, seed=args.seed, kind=args.kind, check_mps=args.check_mps, feature_version=args.feature_version)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
