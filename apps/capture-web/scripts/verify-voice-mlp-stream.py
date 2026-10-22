"""Independent NumPy check of raw inputs and predictions from the real browser stream.
No training, Torch inference, native build, or microphone access is performed.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/voice-prosody"))
from phenometrix_prosody.model import ModelConfig, numpy_step

run = Path(sys.argv[1]).resolve()
evidence = json.loads(Path(sys.argv[2]).read_text())
manifest_bytes = (run / "manifest.json").read_bytes()
manifest = json.loads(manifest_bytes)
weights_bytes = (run / "weights.npz").read_bytes()
assert hashlib.sha256(weights_bytes).hexdigest() == manifest["artifacts"]["weights"]["sha256"]
assert evidence["researchModel"]["modelSha256"] == manifest["artifacts"]["onnx"]["sha256"]
assert manifest["config"]["kind"] == "framewise-mlp"
config = ModelConfig(**manifest["config"])
with np.load(run / "weights.npz") as archive:
    weights = {name: archive[name] for name in archive.files}
maximum_error = 0.0
steps = 0
for frame in evidence["frames"]:
    actual = frame["research"]
    assert actual["qualification"] == "unqualified-functional-fixture"
    assert len(actual["featureValues"]) == 57 and len(actual["candidateMask"]) == 6
    features = np.asarray([actual["featureValues"]], dtype=np.float32)
    mask = np.asarray([actual["candidateMask"]], dtype=bool)
    assert np.isfinite(features).all()
    voicing, logits, state = numpy_step(config, weights, features, mask, np.zeros((1, 32), np.float32))
    expected = np.concatenate((voicing.ravel(), logits.ravel()))
    observed = np.asarray([actual["voicingLogit"], *actual["candidateLogits"]], dtype=np.float64)
    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=1e-5)
    maximum_error = max(maximum_error, float(np.abs(observed - expected).max()))
    probability = float(np.exp(-np.logaddexp(0.0, -float(voicing[0, 0]))))
    assert abs(actual["voicingProbability"] - probability) <= 1e-6
    candidates = np.flatnonzero(mask[0])
    selected = int(candidates[np.argmax(logits[0, candidates])]) if candidates.size else None
    state_name = "unvoiced" if probability < 0.5 else "unavailable" if selected is None else "voiced"
    assert actual["state"] == state_name
    assert actual["candidateIndex"] == (selected if state_name == "voiced" else None)
    expected_probabilities = np.zeros(6, np.float64)
    if selected is not None:
        valid_logits = logits[0, candidates].astype(np.float64)
        values = np.exp(valid_logits - np.max(valid_logits))
        expected_probabilities[candidates] = values / values.sum()
    np.testing.assert_allclose(actual["candidateProbabilities"], expected_probabilities, atol=1e-6, rtol=1e-5)
    assert actual["f0Hz"] == (actual["candidateHz"][selected] if state_name == "voiced" else None)
    assert np.array_equal(state, np.zeros((1, 32), np.float32))
    timing = frame["timing"]
    assert actual["referenceSample"] == timing["supportEndSampleExclusive"] - 960
    assert actual["lastRequiredInputSample"] == timing["supportEndSampleExclusive"] - 1
    assert actual["correctnessProbability"] is None
    steps += 1
assert steps >= 50
print(json.dumps({"status": "complete", "steps": steps, "maximumAbsoluteError": maximum_error,
    "absoluteTolerance": 1e-6, "relativeTolerance": 1e-5, "decisionMismatches": 0,
    "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "modelSha256": manifest["artifacts"]["onnx"]["sha256"],
    "qualification": "unqualified-functional-fixture", "featureCount": 57,
    "evidenceBoundary": "Independent NumPy equations on actual browser feature inputs; separate worker regression checks shared features and unchanged DSP frames."}))
