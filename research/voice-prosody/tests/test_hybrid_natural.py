import copy
import json
from pathlib import Path
import numpy as np
import pytest
from phenometrix_prosody import hybrid_natural as hn
from phenometrix_prosody import evaluate as ev
from phenometrix_prosody.model import FEATURE_NAMES_V2


def fixture(tmp_path, monkeypatch):
    features, data, output, plan_folder = [tmp_path / s for s in ("features", "data", "output", "plan")]
    evaluations = {v: tmp_path / v for v in hn.VARIANTS}
    for p in (features / "features", features / "predictions", data, plan_folder,
              *(p / "predictions" for p in evaluations.values())):
        p.mkdir(parents=True)
    x = np.zeros((8, 57), np.float32)
    hz = np.full((8, 6), np.nan, np.float64); hz[:, :2] = [100., 200.]
    feature = {"features": x, "candidate_hz": hz, "candidate_mask": np.isfinite(hz),
        "reference_sample": 2208 + np.arange(8, dtype=np.int64) * 480,
        "feature_names": np.array(FEATURE_NAMES_V2), "sample_rate_hz": np.array(48000),
        "pcm_sample_count": np.array(9600)}
    records, acquisition = [], []
    for speaker in hn.SPEAKERS:
        source = speaker + "_si10"
        ref = data / (source + ".txt")
        np.savetxt(ref, np.tile([200., 1., 0., 0.], (20, 1)))
        for role in ("mic", "ref"):
            acquisition.append({"recording_id": source, "speaker": speaker, "split": "development",
                "role": role, "path": ref.name, "sha256": ev.sha(ref), "primary_eligible": True})
        for condition in hn.CONDITIONS:
            identity = source + "__" + condition
            f = features / "features" / (identity + ".npz")
            np.savez_compressed(f, **feature)
            baseline = [{"sample": int(t), "state": "unvoiced", "f0_hz": None, "correctness_probability": None}
                        for t in feature["reference_sample"]]
            raw = [{"sample": int(t), "raw_f0_hz": 200., "window_rms": 0., "normalized_lag_correlation": 0.}
                   for t in feature["reference_sample"]]
            path = features / "predictions" / (identity + ".json")
            ev.write_json(path, {"recording_id": identity, "dsp": baseline, "yin": baseline, "yin_evidence": raw})
            records.append({"recording_id": identity, "source_recording_id": source, "speaker": speaker,
                "pair": int(speaker[-2:]), "prompt": "si10", "condition": condition,
                "sample_count": 9600, "microphone_sha256": ev.sha(ref), "feature_sha256": ev.sha(f),
                "prediction_sha256": ev.sha(path)})
    ev.write_json(features / "acquisition-manifest.json", {"files": acquisition})
    cache = {"schema": "phenometrix-input-only-feature-cache-v1", "status": "complete", "split": "development",
        "feature_version": "v2", "measurement_delay_ms": 20, "reference_or_laryngograph_content_read": False,
        "data_manifest_sha256": ev.sha(features / "acquisition-manifest.json"),
        "conditions": list(hn.CONDITIONS), "recordings": records}
    ev.write_json(features / "manifest.json", cache)
    plan = {"source_sha256": {}, "models": {}, "natural_features": {"directory": str(features),
        "manifest_sha256": ev.sha(features / "manifest.json")}, "natural_evaluations": {}}
    for variant, folder in evaluations.items():
        model = {"manifest_sha256": "model-manifest-" + variant, "weights_sha256": "weights-" + variant,
                 "model": {"config": {"kind": "framewise-mlp"}}}
        plan["models"][variant] = {k: v for k, v in model.items() if k != "model"}
        rows = []
        for record in records:
            timing = ev.validate_feature(feature, record)
            baseline = [{"sample": int(t), "state": "unvoiced", "f0_hz": None, "correctness_probability": None}
                        for t in timing.reference_sample]
            learned = [{**r, "voicing_probability": .9 if variant == "original" else .8, "candidate_index": None}
                       for r in baseline]
            relative = "predictions/" + record["recording_id"] + ".json.gz"
            ev.write_predictions(folder / relative, {"recording_id": record["recording_id"],
                "availability_sample": timing.availability_sample.tolist(), "time_contract": timing.metadata(),
                "predictions": {"dsp": baseline, "yin": baseline, "continuity": baseline, "gru": baseline, "mlp": learned}})
            rows.append({**record, "evaluated_prediction_path": relative, "evaluated_prediction_sha256": ev.sha(folder / relative)})
        common = {"schema": "phenometrix-delayed-evaluation-v1", "split": "development", "measurement_delay_ms": 20,
            "reference_content_read_before_prediction_freeze": False,
            "feature_manifest_sha256": ev.sha(features / "manifest.json"), "conditions": list(hn.CONDITIONS),
            "models": {"mlp": model}, "recordings": rows}
        ev.write_json(folder / "prediction-freeze.json", {**common, "status": "predictions_frozen"})
        ev.write_json(folder / "manifest.json", {**common, "status": "complete"})
        plan["natural_evaluations"][variant] = {"directory": str(folder),
            "manifest_sha256": ev.sha(folder / "manifest.json"), "prediction_freeze_sha256": ev.sha(folder / "prediction-freeze.json")}
    ev.write_json(plan_folder / "plan.json", plan)
    ev.write_json(plan_folder / "declaration.json", {"test_fixture": True})
    monkeypatch.setattr(hn, "read_frozen_plan", lambda _: copy.deepcopy(plan))
    return plan_folder, features, evaluations, data, output, plan


def test_complete_both_variant_freeze_precedes_first_reference_load(tmp_path, monkeypatch):
    pf, features, evaluations, data, output, _ = fixture(tmp_path, monkeypatch)
    real = np.loadtxt
    def reference_read(path, *args, **kwargs):
        frozen = json.loads((output / "prediction-freeze.json").read_text())
        assert frozen["status"] == "predictions_frozen"
        assert len(frozen["recordings"]) == 16
        for record in frozen["recordings"]:
            p = ev.read_predictions(output / record["hybrid_prediction_path"])
            assert set(p["predictions"]) == set(hn.METHODS)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(np, "loadtxt", reference_read)
    hn.freeze_predictions(pf, features, evaluations, output)
    first = ev.read_predictions(output / "predictions/F01_si10__clean.json.gz")
    assert first["predictions"]["yin"][0]["state"] == "unvoiced"
    for variant in hn.VARIANTS:
        row = first["predictions"]["hybrid_" + variant][0]
        assert row["f0_hz"] == 200 and row["state"] == "voiced"
        assert row["sample"] == 1248 and row["correctness_probability"] is None
    result = hn.score_frozen(pf, features, evaluations, data, output)
    assert result["keep"] is False
    assert result["equal_speaker_equal_condition"]["hybrid_original"]["cell_count"] == 16
    with pytest.raises(FileExistsError):
        hn.score_frozen(pf, features, evaluations, data, output)


def test_changed_cached_probability_bytes_rejected_before_reference(tmp_path, monkeypatch):
    pf, features, evaluations, data, output, _ = fixture(tmp_path, monkeypatch)
    hn.freeze_predictions(pf, features, evaluations, output)
    p = evaluations["original"] / "predictions/F01_si10__clean.json.gz"
    p.write_bytes(p.read_bytes() + b"changed")
    monkeypatch.setattr(np, "loadtxt", lambda *_: pytest.fail("reference opened before input integrity"))
    with pytest.raises(ValueError, match="prediction bytes"):
        hn.score_frozen(pf, features, evaluations, data, output)


def test_changed_frozen_hybrid_rejected_before_reference(tmp_path, monkeypatch):
    pf, features, evaluations, data, output, _ = fixture(tmp_path, monkeypatch)
    hn.freeze_predictions(pf, features, evaluations, output)
    p = output / "predictions/F01_si10__clean.json.gz"; p.write_bytes(p.read_bytes() + b"changed")
    monkeypatch.setattr(np, "loadtxt", lambda *_: pytest.fail("reference opened before prediction integrity"))
    with pytest.raises(ValueError, match="Hybrid prediction changed"):
        hn.score_frozen(pf, features, evaluations, data, output)


def test_sealed_or_train_cache_refused_even_with_updated_plan_hash(tmp_path, monkeypatch):
    pf, features, evaluations, _, output, plan = fixture(tmp_path, monkeypatch)
    p = features / "manifest.json";d = json.loads(p.read_text());d["split"] = "sealed-test";ev.write_json(p,d)
    plan["natural_features"]["manifest_sha256"] = ev.sha(p)
    with pytest.raises(ValueError):hn.freeze_predictions(pf, features, evaluations, output)
    assert not output.exists()


def test_missing_variant_or_changed_feature_manifest_refused(tmp_path, monkeypatch):
    pf, features, evaluations, _, output, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Both"):
        hn.freeze_predictions(pf, features, {"original": evaluations["original"]}, output)
    p = features / "manifest.json";p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="Feature cache"):
        hn.freeze_predictions(pf, features, evaluations, output)
    assert not output.exists()


def test_plan_basenames_resolve_under_research_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(hn, "REPO", tmp_path)
    assert hn._declared_path({"directory": "delayed-development-features"}) == (
        tmp_path / "research/voice-prosody/runs/delayed-development-features").resolve()
    assert hn._declared_path({"directory": str(tmp_path / "absolute")}) == (tmp_path / "absolute").resolve()


def test_artifact_cannot_escape_frozen_directory(tmp_path):
    with pytest.raises(ValueError, match="escaped"):
        hn._inside(tmp_path / "run", "../reference.txt")
