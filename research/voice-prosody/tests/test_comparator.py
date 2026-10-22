import inspect
import numpy as np
import pytest
import torch
import torchcrepe

from phenometrix_prosody.comparator import (TinyComparator, TinyConfig, decoder_weights,
    frame_pcm, normalize_frames, support, target_grid, weighted_argmax)


def fixture_backend(frames):
    # A frame-independent, deterministic test double; no pretrained inference.
    activations = torch.full((len(frames), 360), 0.01, device=frames.device)
    activations[:, 100] = 0.9
    return activations


def test_exact_16ms_phase_half_open_support_and_boundary():
    assert list(target_grid(4800))[:4] == [768, 1248, 1728, 2208]
    assert support(2208) == (672, 3744)
    assert frame_pcm(np.zeros(4800, np.float32), 768) is None
    assert frame_pcm(np.zeros(4800, np.float32), 1728).shape == (1024,)
    rows = TinyComparator(backend=fixture_backend).predict(np.ones(4800, np.float32) * .01)
    assert [r["sample"] for r in rows[:4]] == [768, 1248, 1728, 2208]
    assert [r["state"] for r in rows[:2]] == ["unavailable", "unavailable"]
    assert rows[2]["support_end_sample_exclusive"] == rows[2]["available_at_sample"] == 3264
    assert rows[2]["declared_lookahead_ms"] == 32
    assert rows[-1]["state"] == "unavailable"


def test_no_audio_outside_declared_support_changes_frame_or_prediction():
    rng = np.random.default_rng(7)
    audio = rng.normal(size=9600).astype(np.float32) * .03
    target = 4800  # 100 ms; direct frame support test is not reference-phase dependent.
    start, end = support(target)
    changed = audio.copy(); changed[:start] = 100; changed[end:] = -100
    np.testing.assert_array_equal(frame_pcm(audio, target), frame_pcm(changed, target))
    comparator = TinyComparator(backend=fixture_backend)
    rows = comparator.predict(audio)
    target = rows[9]["sample"]
    start, end = support(target)
    changed = audio.copy(); changed[:start] = 20; changed[end:] = -20
    assert rows[9] == comparator.predict(changed)[9]
    assert set(inspect.signature(TinyComparator.predict).parameters) == {"self", "audio", "sample_rate"}


def test_frame_local_resample_matches_pinned_explicit_filter_and_16k_identity():
    from scipy.signal import resample_poly
    audio = np.random.default_rng(17).normal(size=9600).astype(np.float32)
    begin, end = support(2208)
    expected = resample_poly(audio[begin:end], 1, 3, window=("kaiser",5.0), padtype="constant", cval=0)
    np.testing.assert_array_equal(frame_pcm(audio,2208), expected)
    mono16 = np.arange(2048, dtype=np.float32)
    np.testing.assert_array_equal(frame_pcm(mono16, 512, 16000), mono16[:1024])
    with pytest.raises(ValueError, match="16 or 48"):
        TinyComparator(backend=fixture_backend).predict(audio, 44100)


def test_frame_normalization_matches_upstream_without_whole_record_scaling():
    audio = torch.arange(2048,dtype=torch.float32).reshape(1,-1) / 3000
    upstream = next(torchcrepe.preprocess(audio,16000,hop_length=1024,batch_size=2,pad=False))
    expected = normalize_frames(audio.reshape(2,1024))
    torch.testing.assert_close(upstream,expected,rtol=1e-6,atol=1e-6)
    assert torch.isfinite(normalize_frames(torch.zeros(1,1024))).all()


def test_weighted_decoder_matches_pinned_semantics_with_fixed_local_dither(monkeypatch):
    config = TinyConfig()
    activations = torch.tensor(np.random.default_rng(29).uniform(size=(4,360)),dtype=torch.float32)
    monkeypatch.setattr(torchcrepe.decode.weighted_argmax,"weights",decoder_weights(config.dither_seed)[None,:,None],raising=False)
    expected_pitch, expected_periodicity = torchcrepe.postprocess(activations.T[None,:,:].clone(),
        fmin=config.fmin, fmax=config.fmax, decoder=torchcrepe.decode.weighted_argmax,return_periodicity=True)
    actual_pitch, actual_periodicity = weighted_argmax(activations,config)
    torch.testing.assert_close(actual_pitch,expected_pitch[0],rtol=2e-6,atol=1e-4)
    torch.testing.assert_close(actual_periodicity,expected_periodicity[0],rtol=0,atol=0)
    state=np.random.get_state(); decoder_weights(); state_after=np.random.get_state()
    assert state[0]==state_after[0] and np.array_equal(state[1],state_after[1]) and state[2:]==state_after[2:]


def test_batches_and_repeated_calls_do_not_create_temporal_dependence():
    audio=np.random.default_rng(9).normal(size=9600).astype(np.float32)*.1
    one=TinyComparator(TinyConfig(batch_size=1),backend=fixture_backend)
    many=TinyComparator(TinyConfig(batch_size=7),backend=fixture_backend)
    assert one.predict(audio)==many.predict(audio)==one.predict(audio)
    silent=one.predict(np.zeros(9600,np.float32))
    assert all(r["state"]=="unvoiced" for r in silent if r["reason"]!="insufficient_centered_support")
    assert all(r["correctness_probability"] is None for r in silent)
    broken=audio.copy(); broken[8000]=np.nan
    broken_rows=one.predict(broken)
    assert broken_rows[2]==one.predict(audio)[2]
    assert any(r["reason"]=="nonfinite_centered_support" for r in broken_rows)


def test_end_anchored_contract_keeps_targets_and_never_reads_future_suffix():
    audio=np.random.default_rng(43).normal(size=9600).astype(np.float32)*.03
    comparator=TinyComparator(TinyConfig(anchor="end_anchored"),backend=fixture_backend)
    rows=comparator.predict(audio)
    assert [r["sample"] for r in rows]==list(target_grid(len(audio)))
    assert all(r["state"]=="unavailable" for r in rows[:5])
    row=rows[5]
    assert row["sample"]==3168  #66ms, first point with64ms of source support
    assert row["support_start_sample"]==96 and row["support_end_sample_exclusive"]==3168
    assert row["available_at_sample"]==row["sample"]
    assert row["model_window_center_sample"]==row["sample"]-1536
    assert row["declared_lookahead_ms"]==0
    assert row["method"]=="torchcrepe-tiny-end-anchored64-modified-v1"
    changed=audio.copy();changed[row["sample"]:]=np.nan
    assert row==comparator.predict(changed)[5]
    assert TinyConfig().anchor=="centered"
