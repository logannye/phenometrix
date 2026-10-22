import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
import torch
from phenometrix_prosody import evaluate as ev
from phenometrix_prosody.model import VoiceCandidateModel, ModelConfig, FEATURE_NAMES_V2, model_metadata


def feature_fixture(n=8):
    hz = np.full((n, 6), np.nan, np.float64)
    hz[:, :2] = [100., 200.]
    x = np.zeros((n, 57), np.float32)
    names = list(FEATURE_NAMES_V2)
    x[:, names.index("log_rms_floor_1e-8")] = np.log(.01)
    for slot in range(2):
        x[:, names.index(f"candidate_{slot}_valid")] = 1
        x[:, names.index(f"candidate_{slot}_maximum_score")] = .8
    return {"features": x, "candidate_hz": hz, "candidate_mask": np.isfinite(hz),
            "reference_sample": 2208 + np.arange(n, dtype=np.int64) * 480,
            "feature_names": np.array(FEATURE_NAMES_V2), "sample_rate_hz": np.array(48000),
            "pcm_sample_count": np.array(9600)}


def weights(kind="causal-gru", voice=1.):
    torch.manual_seed(15)
    model = VoiceCandidateModel(ModelConfig(kind=kind))
    with torch.no_grad():
        model.voicing_head.weight.zero_()
        model.voicing_head.bias.fill_(voice)
        model.candidate_head.weight.zero_()
        model.candidate_head.bias.zero_()
    return model, {k: v.numpy().copy() for k, v in model.state_dict().items()}


def test_neural_tie_and_all_masked_semantics_at_exact_half_voicing():
    feature = feature_fixture()
    feature["candidate_mask"][1] = False
    model, state = weights(voice=0.)
    rows = ev.model_predictions(feature, model.config, state)
    assert rows[0]["sample"] == 1248
    assert rows[0]["voicing_probability"] == .5
    assert rows[0]["f0_hz"] == 100 and rows[0]["candidate_index"] == 0
    assert rows[1]["state"] == "unavailable" and rows[1]["f0_hz"] is None
    assert rows[0]["correctness_probability"] is None
    model, state = weights(voice=-1.)
    assert ev.model_predictions(feature, model.config, state)[1]["state"] == "unvoiced"


def test_recurrent_state_is_carried_and_resets_between_calls(monkeypatch):
    feature = feature_fixture(3)
    seen = []
    def step(config, weight, x, mask, state):
        seen.append(state.copy())
        return np.ones((1, 1), np.float32), np.zeros((1, 6), np.float32), state + 1
    monkeypatch.setattr(ev, "numpy_step", step)
    ev.model_predictions(feature, ModelConfig(), {})
    ev.model_predictions(feature, ModelConfig(), {})
    assert [float(x[0, 0]) for x in seen] == [0, 1, 2, 0, 1, 2]


def test_future_feature_intervention_does_not_change_prefix_predictions():
    feature = feature_fixture()
    torch.manual_seed(18)
    model = VoiceCandidateModel()
    state = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    first = ev.model_predictions(feature, model.config, state)
    feature["features"][5:] = 500
    later = ev.model_predictions(feature, model.config, state)
    assert first[:5] == later[:5]


def test_continuity_uses_evidence_penalty_and_resets_after_unvoiced():
    feature = feature_fixture(5)
    names = list(FEATURE_NAMES_V2)
    a, b = names.index("candidate_0_maximum_score"), names.index("candidate_1_maximum_score")
    feature["features"][:, [a, b]] = [[.9, .8], [.7, .85], [.7, .85], [.7, .85], [.8, .8]]
    feature["features"][2, names.index("log_rms_floor_1e-8")] = np.log(.001)
    rows = ev.continuity_predictions(feature)
    assert [r["f0_hz"] for r in rows] == [100., 100., None, 200., 200.]
    assert ev.continuity_predictions(feature)[0]["f0_hz"] == 100.
    feature["candidate_mask"][0] = False
    assert ev.continuity_predictions(feature)[0]["state"] == "unvoiced"


def test_baseline_shift_requires_exact_acquisition_endpoints():
    feature = feature_fixture(2)
    baseline = [{"sample": int(e), "state": "voiced", "f0_hz": 100.} for e in feature["reference_sample"]]
    rows = ev.delayed_baseline(baseline, feature)
    assert [r["sample"] for r in rows] == [1248, 1728]
    assert baseline[0]["sample"] == 2208
    baseline[0]["sample"] += 1
    with pytest.raises(ValueError, match="timestamps"):
        ev.delayed_baseline(baseline, feature)


def test_aggregation_weights_cells_equally_not_frame_counts():
    one = {name: .1 for name in ev.METRICS}
    two = {name: .9 for name in ev.METRICS}
    one["eligible_valid_frames"], two["eligible_valid_frames"] = 1000, 1
    result = ev.macro([one, two])
    assert result["joint_error"] == .5 and result["cell_count"] == 2
    two["p95_jointly_voiced_cents"] = None
    assert ev.macro([one, two])["defined_cells"]["p95_jointly_voiced_cents"] == 1


def test_quality_success_cannot_be_mislabeled_final_keep():
    overall = {name: {"joint_error": .1, "reference_voiced_success_rate": .8} for name in ["gru", *ev.SIMPLE_METHODS]}
    overall["gru"]["joint_error"] = .08
    gate = ev.quality_gate(overall, overall)
    assert gate["gru"]["status"] == "quality_components_pass_pending_other_gates"
    assert gate["keep"] is False
    overall["mlp"]["joint_error"] = .085
    assert ev.quality_gate(overall, overall)["gru"]["status"] == "quality_components_fail"


def make_run(tmp_path, split="train", speaker="F01"):
    folder = tmp_path / split
    (folder / "features").mkdir(parents=True)
    (folder / "predictions").mkdir()
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    source = speaker + "_001"
    reference = data / (source + ".f0")
    # Four-column PTDB rows: F0 and binary periodicity, extra columns ignored.
    reference.write_text("100 1 0 0\n" * 20)
    acquisition = {"files": [
        {"recording_id": source, "role": "mic", "split": split, "speaker": speaker, "sha256": "mic-bytes", "path": source + ".wav"},
        {"recording_id": source, "role": "ref", "split": split, "speaker": speaker, "sha256": ev.sha(reference), "path": reference.name},
    ]}
    ev.write_json(folder / "acquisition-manifest.json", acquisition)
    feature = feature_fixture(12)
    identity = source + "__clean"
    np.savez_compressed(folder / "features" / (identity + ".npz"), **feature)
    baseline = [{"sample": int(e), "state": "voiced", "f0_hz": 100., "correctness_probability": None} for e in feature["reference_sample"]]
    ev.write_json(folder / "predictions" / (identity + ".json"), {"recording_id": identity, "dsp": baseline, "yin": baseline})
    record = {"recording_id": identity, "source_recording_id": source, "speaker": speaker, "condition": "clean",
              "sample_count": 9600, "microphone_sha256": "mic-bytes",
              "feature_sha256": ev.sha(folder / "features" / (identity + ".npz")),
              "prediction_sha256": ev.sha(folder / "predictions" / (identity + ".json"))}
    manifest = {"schema": "phenometrix-input-only-feature-cache-v1", "status": "complete", "split": split,
                "feature_version": "v2", "measurement_delay_ms": 20, "reference_or_laryngograph_content_read": False,
                "conditions": ["clean"], "data_manifest_sha256": ev.sha(folder / "acquisition-manifest.json"), "recordings": [record]}
    ev.write_json(folder / "manifest.json", manifest)
    return folder, data, manifest, reference


def make_checkpoint(tmp_path, kind, train_folder, train_manifest):
    folder = tmp_path / kind
    folder.mkdir()
    model, _ = weights(kind)
    model.fit_scaling(torch.from_numpy(feature_fixture()["features"]), split="train")
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, folder / "weights.pt")
    manifest = {"status": "complete", "split": "train", "measurement_delay_ms": 20,
                "test_or_development_labels_used": False, "kind": kind,
                "feature_run_manifest_sha256": ev.sha(train_folder / "manifest.json"),
                "feature_hashes": {r["recording_id"]: r["feature_sha256"] for r in train_manifest["recordings"]},
                "model": model_metadata(model), "weights_sha256": ev.sha(folder / "weights.pt"),
                "reference_hashes": {r["recording_id"]: r["sha256"] for r in json.loads((train_folder / "acquisition-manifest.json").read_text())["files"] if r["role"] == "ref"}}
    ev.write_json(folder / "manifest.json", manifest)
    return folder


def test_checkpoint_provenance_and_config_are_enforced(tmp_path):
    train, _, manifest, _ = make_run(tmp_path)
    folder = make_checkpoint(tmp_path, "causal-gru", train, manifest)
    config, _, _ = ev.load_frozen_model(folder, train, manifest)
    assert config.feature_count == 57
    value = json.loads((folder / "manifest.json").read_text())
    value["measurement_delay_ms"] = 0
    ev.write_json(folder / "manifest.json", value)
    with pytest.raises(ValueError, match="provenance"):
        ev.load_frozen_model(folder, train, manifest)


def test_end_to_end_predictions_are_frozen_before_any_label_read(tmp_path, monkeypatch):
    train, data, manifest, reference = make_run(tmp_path)
    models = {"gru": make_checkpoint(tmp_path, "causal-gru", train, manifest),
              "mlp": make_checkpoint(tmp_path, "framewise-mlp", train, manifest)}
    output = tmp_path / "evaluation"
    original_read = Path.read_bytes
    reads = []
    def audited_read(path):
        if path == reference:
            assert (output / "prediction-freeze.json").exists()
            freeze = json.loads((output / "prediction-freeze.json").read_text())
            assert freeze["status"] == "predictions_frozen"
            for r in freeze["recordings"]:
                assert ev.sha(output / "predictions" / (r["recording_id"] + ".json.gz")) == r["evaluated_prediction_sha256"]
            reads.append(path)
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", audited_read)
    result = ev.evaluate(train, train, data, models, output)
    assert reads
    assert result["equal_speaker_equal_condition"]["dsp"]["joint_error"] > 0  # trailing missing predictions count as failures
    assert result["per_record"][0]["metrics"]["dsp"]["calibration_n"] == 0
    raw = ev.read_predictions(next((output / "predictions").glob("*.json.gz")))
    assert raw["availability_sample"][0] - raw["predictions"]["gru"][0]["sample"] == 960
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"


def test_development_source_speaker_overlap_is_rejected_before_predictions(tmp_path):
    train, data, manifest, _ = make_run(tmp_path)
    development, _, _, _ = make_run(tmp_path, "development", "F01")
    with pytest.raises(ValueError, match="overlaps"):
        ev.evaluate(development, train, data, {}, tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def test_continuity_rms_threshold_is_in_feature_log_coordinates():
    feature = feature_fixture(1)
    feature["features"][0, list(FEATURE_NAMES_V2).index("log_rms_floor_1e-8")] = np.log(.003)
    assert ev.continuity_predictions(feature)[0]["state"] == "voiced"


def test_corrupt_candidate_masks_and_missing_recording_prefix_are_rejected():
    feature = feature_fixture(2)
    record = {"sample_count": 9600}
    ev.validate_feature(feature, record)
    feature["candidate_mask"][0, 0] = False
    with pytest.raises(ValueError, match="absence"):
        ev.validate_feature(feature, record)
    feature = feature_fixture(2)
    feature["reference_sample"] += 480
    with pytest.raises(ValueError, match="warmup"):
        ev.validate_feature(feature, record)


def test_prediction_compression_is_lossless_and_deterministic(tmp_path):
    value = {"sample": [1, 2], "f0_hz": [None, 123.456], "state": ["unavailable", "voiced"]}
    ev.write_predictions(tmp_path / "a.json.gz", value)
    ev.write_predictions(tmp_path / "b.json.gz", value)
    assert (tmp_path / "a.json.gz").read_bytes() == (tmp_path / "b.json.gz").read_bytes()
    assert ev.read_predictions(tmp_path / "a.json.gz") == value


def test_augmented_checkpoint_requires_exact_train_only_supplement_snapshot(tmp_path):
    train, _, manifest, _ = make_run(tmp_path)
    folder = make_checkpoint(tmp_path, "framewise-mlp", train, manifest)
    curriculum = {"status": "complete", "split": "train", "feature_version": "v2", "measurement_delay_ms": 20,
                  "records": [{"id": "synth1", "feature_sha256": "features", "reference_sha256": "analytic-reference"}]}
    supplement_file = folder / "synthetic-curriculum-manifest.json"
    ev.write_json(supplement_file, curriculum)
    checkpoint = json.loads((folder / "manifest.json").read_text())
    checkpoint["synthetic_curriculum"] = {"manifest_sha256": ev.sha(supplement_file), "unique_sequence_count": 1,
        "repetitions_for_training_and_scaling": 4, "challenge_content_or_outcomes_used": False,
        "feature_and_reference_sha256": {"synth1": {"feature_sha256": "features", "reference_sha256": "analytic-reference"}}}
    ev.write_json(folder / "manifest.json", checkpoint)
    ev.load_frozen_model(folder, train, manifest)
    checkpoint["synthetic_curriculum"]["feature_and_reference_sha256"]["synth1"]["reference_sha256"] = "changed"
    ev.write_json(folder / "manifest.json", checkpoint)
    with pytest.raises(ValueError, match="record identities"):
        ev.load_frozen_model(folder, train, manifest)
