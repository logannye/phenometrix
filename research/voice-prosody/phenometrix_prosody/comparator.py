"""Input-only centered torchcrepe-tiny diagnostic comparator.

Raw source support is exactly [target-32 ms, target+32 ms). Frame-local
resampling uses zero extension *of that frame*, so it reads no additional audio.
This is an explicitly disclosed adapter, not whole-file resampy equivalence.
No reference, voicing label, eligibility mask or calibration target is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import signal, stats
import torch

TORCHCREPE_COMMIT = "19e2ec3d494c0797a5ff2a11408ec5838fba6681"
TINY_SHA256 = "d4993eea36ed1a0ad9ac549c740dae5265b049ce72004f00c2f59e01c0be8432"
MODEL_SOURCE_SHA256 = "028e240a191d5ddb28d0af6febe999b9f692f22459e49bf9c5681c9dfea4bd22"
MODEL_SAMPLE_RATE = 16000
WINDOW_SAMPLES = 1024
CENTS_OFFSET = 1997.3794084376191


@dataclass(frozen=True)
class TinyConfig:
    periodicity_threshold: float = 0.5
    silence_rms: float = 0.003
    fmin: float = 50.0
    fmax: float = 700.0
    dither_seed: int = 20261022
    batch_size: int = 128
    device: str = "cpu"
    anchor: str = "centered"

    def __post_init__(self):
        if not math.isfinite(self.periodicity_threshold) or not 0 <= self.periodicity_threshold <= 1:
            raise ValueError("Periodicity threshold must lie in [0,1]")
        if not math.isfinite(self.silence_rms) or self.silence_rms < 0:
            raise ValueError("Silence RMS threshold must be finite and nonnegative")
        if not (math.isfinite(self.fmin) and math.isfinite(self.fmax) and 32 <= self.fmin < self.fmax <= 2006):
            raise ValueError("Frequency range must fit CREPE bins")
        if not isinstance(self.dither_seed, int) or not 0 <= self.dither_seed < 2**32:
            raise ValueError("A deterministic uint32 dither seed is required")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("Batch size must be positive")
        if self.anchor not in ("centered", "end_anchored"):
            raise ValueError("Anchor must be centered or end_anchored")
        if self.device not in ("cpu", "mps"):
            raise ValueError("Only local CPU or Apple MPS is supported")


def validate_audio(audio, sample_rate):
    if sample_rate not in (16000, 48000):
        raise ValueError("Exact-grid adapter currently supports only 16 or 48 kHz")
    values = np.asarray(audio)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Audio must be one-dimensional floating-point PCM")
    return values.astype(np.float32, copy=False)


def target_grid(sample_count, sample_rate=48000):
    """Derive 16+10*k ms targets from PCM length only, without reference files."""
    if sample_rate not in (16000, 48000) or sample_count < 0:
        raise ValueError("Unsupported rate or negative sample count")
    first, hop = sample_rate * 16 // 1000, sample_rate // 100
    return range(first, sample_count, hop)


def support(target_sample, sample_rate=48000, anchor="centered"):
    if sample_rate not in (16000, 48000) or not isinstance(target_sample, (int, np.integer)):
        raise ValueError("Integral target and supported sample rate required")
    half = sample_rate * 32 // 1000
    if anchor == "centered":
        return int(target_sample - half), int(target_sample + half)
    if anchor == "end_anchored":
        return int(target_sample - 2 * half), int(target_sample)
    raise ValueError("Unknown support anchor")


def frame_pcm(audio, target_sample, sample_rate=48000, anchor="centered"):
    """Return 1024 samples anchored at target-32ms, or None at a boundary.

    At 48 kHz, crop the 3072-sample raw window before resample_poly(up=1,
    down=3, window=('kaiser',5), padtype='constant', cval=0). Its first
    output sample shares the raw window's first timestamp; no interpolation
    of F0 or future model outputs occurs. 16 kHz frames are copied directly.
    """
    start, end = support(target_sample, sample_rate, anchor)
    if start < 0 or end > len(audio):
        return None
    segment = np.asarray(audio[start:end], dtype=np.float32).copy()
    if not np.isfinite(segment).all():
        raise ValueError("Nonfinite centered support is not measurable")
    if sample_rate == 48000:
        segment = signal.resample_poly(segment, 1, 3, window=("kaiser", 5.0),
                                       padtype="constant", cval=0).astype(np.float32)
    if segment.shape != (WINDOW_SAMPLES,):
        raise ValueError("Resampler did not preserve exact 64 ms frame length")
    return segment


def normalize_frames(frames):
    """Match pinned torchcrepe centering/sample-standard-deviation semantics."""
    if frames.ndim != 2 or frames.shape[1] != WINDOW_SAMPLES:
        raise ValueError("CREPE requires rows of 1024 samples")
    values = frames.clone()
    values -= values.mean(dim=1, keepdim=True)
    return values / values.std(dim=1, keepdim=True, correction=1).clamp_min(1e-10)


def decoder_weights(seed=20261022, device="cpu"):
    """Pin upstream triangular cents dither without changing global RNG state."""
    cents = 20 * torch.arange(360, dtype=torch.float32) + CENTS_OFFSET
    noise = stats.triang.rvs(c=0.5, loc=-20, scale=40, size=360,
                             random_state=np.random.RandomState(seed))
    return (cents + torch.as_tensor(noise, dtype=torch.float32)).to(device)


def weighted_argmax(activations, config=TinyConfig()):
    """Pinned decoder semantics, including its second sigmoid and fixed dither.

    Upstream model.py returns sigmoid activations; decode.weighted_argmax applies
    another sigmoid after its argmax-local mask. Preserve that behavior rather
    than silently substituting original TensorFlow CREPE's averaging formula.
    Periodicity is the pre-decoder activation at the winning allowed bin, never
    a calibrated probability that F0 is correct within a stated tolerance.
    """
    if activations.ndim != 2 or activations.shape[1] != 360:
        raise ValueError("Expected one row of 360 activations per frame")
    if not torch.isfinite(activations).all() or torch.any((activations < 0) | (activations > 1)):
        raise ValueError("Model must return finite sigmoid activations")
    values = activations.detach().clone()
    # Use float32 conversion as pinned postprocess does, avoiding near-bin drift.
    low = int(torch.floor((1200 * torch.log2(torch.tensor(config.fmin) / 10) - CENTS_OFFSET) / 20))
    high = int(torch.ceil((1200 * torch.log2(torch.tensor(config.fmax) / 10) - CENTS_OFFSET) / 20))
    values[:, :max(0, low)] = -float("inf")
    values[:, min(360, high):] = -float("inf")
    bins = values.argmax(dim=1)
    periodicity = values.gather(1, bins[:, None]).squeeze(1)
    indices = torch.arange(360, device=values.device)[None, :]
    values[(indices < bins[:, None] - 4) | (indices >= bins[:, None] + 5)] = -float("inf")
    probabilities = torch.sigmoid(values)
    weights = decoder_weights(config.dither_seed, values.device)
    cents = (weights[None, :] * probabilities).sum(dim=1) / probabilities.sum(dim=1)
    pitch = 10 * 2 ** (cents / 1200)
    return pitch, periodicity


class TinyComparator:
    """Lazily load only pinned tiny weights; batching never adds temporal state.

    Tests may supply a backend that maps normalized frame tensors to activations.
    Real inference is explicitly requested by predict(); importing this module
    neither loads weights nor predicts. No data loader or reference reader exists.
    """
    def __init__(self, config=TinyConfig(), backend: Callable | None = None):
        self.config = config
        self._backend = backend
        self._model = None

    def _load(self):
        import torchcrepe
        if metadata.version("torchcrepe") != "0.0.24":
            raise ValueError("Expected torchcrepe 0.0.24")
        package = Path(torchcrepe.__file__).parent
        model_source = package / "model.py"
        weights = package / "assets" / "tiny.pth"
        if hashlib.sha256(model_source.read_bytes()).hexdigest() != MODEL_SOURCE_SHA256:
            raise ValueError("CREPE model source differs from audited pinned implementation")
        if hashlib.sha256(weights.read_bytes()).hexdigest() != TINY_SHA256:
            raise ValueError("CREPE tiny weight digest differs from pinned artifact")
        self._model = torchcrepe.Crepe("tiny")
        self._model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self._model.to(self.config.device).eval()

    def _activations(self, frames):
        if self._backend is not None:
            return self._backend(frames)
        if self._model is None:
            self._load()
        with torch.no_grad():
            return self._model(frames)

    def predict(self, audio, sample_rate=48000):
        values = validate_audio(audio, sample_rate)
        output, pending, pending_indices = [], [], []

        def flush():
            if not pending:
                return
            frames = torch.from_numpy(np.stack(pending)).to(self.config.device)
            normalized = normalize_frames(frames)
            with torch.no_grad():
                pitch, periodicity = weighted_argmax(self._activations(normalized), self.config)
            pitch_values = pitch.cpu().numpy()
            periodicity_values = periodicity.cpu().numpy()
            for index, frequency, strength in zip(pending_indices, pitch_values, periodicity_values):
                record = output[index]
                voiced = strength >= self.config.periodicity_threshold and record["rms"] >= self.config.silence_rms
                record.update(state="voiced" if voiced else "unvoiced", f0_hz=float(frequency) if voiced else None,
                    raw_f0_hz=float(frequency), periodicity=float(strength), correctness_probability=None,
                    reason="periodicity_and_rms_gate" if not voiced else None)
            pending.clear()
            pending_indices.clear()

        for target in target_grid(len(values), sample_rate):
            begin, end = support(target, sample_rate, self.config.anchor)
            record = {"sample": target, "sample_rate": sample_rate,
                "support_start_sample": begin, "support_end_sample_exclusive": end,
                "available_at_sample": end, "declared_lookahead_ms": 32 if self.config.anchor == "centered" else 0,
                "model_window_center_sample": (begin + end) // 2, "anchor": self.config.anchor,
                "state": "unavailable", "f0_hz": None, "raw_f0_hz": None,
                "periodicity": None, "correctness_probability": None,
                "rms": None, "reason": "insufficient_centered_support",
                "method": ("torchcrepe-tiny-0.0.24-centered-frame-local-v1" if self.config.anchor == "centered"
                           else "torchcrepe-tiny-end-anchored64-modified-v1")}
            output.append(record)
            try:
                frame = frame_pcm(values, target, sample_rate, self.config.anchor)
            except ValueError as error:
                if str(error) != "Nonfinite centered support is not measurable":
                    raise
                record["reason"] = "nonfinite_centered_support"
                continue
            if frame is None:
                continue
            record["rms"] = float(np.sqrt(np.mean(values[begin:end].astype(np.float64)**2)))
            pending.append(frame)
            pending_indices.append(len(output) - 1)
            if len(pending) == self.config.batch_size:
                flush()
        flush()
        return output
