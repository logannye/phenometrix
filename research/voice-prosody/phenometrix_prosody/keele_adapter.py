"""KEELE v1 metadata contract and scorer-only reference adapter.

This module does not acquire data or admit an external evaluation.  Reference
reading belongs after a successful development/freeze/prediction receipt.  Its
pure functions are exercised only with synthetic inputs during initial intake.
The historical PTDB measurement_alignment helper remains unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "phenometrix.keele-adapter.v1"
SOURCE_RATE = 20_000
RATE = 48_000
HOP = 480
WINDOW = 1920
DELAY = 960
SCIPY_VERSION = "1.17.1"
CONVERTER_COMMIT = "de3e2cbcc8ab0ccb7df60af68c6e6f72c59afdb0"
CONVERTER_SHA256 = "aa3d2285a7397fb086391c97db8344d672a43755d5661ce0e6491b6c2cf0edcd"


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _samples(values, name):
    result = np.asarray(values)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(result < 0) or np.any(result > np.iinfo(np.int64).max - WINDOW):
        raise ValueError(f"{name} outside supported sample range")
    return result.astype(np.int64, copy=True)


def rational_time_to_sample(numerator: int, denominator: int) -> int:
    """Seconds as an exact fraction; never round to a provider label."""
    numerator = _integer(numerator, "time numerator")
    denominator = _integer(denominator, "time denominator", 1)
    sample, remainder = divmod(numerator * RATE, denominator)
    if remainder or sample % HOP:
        raise ValueError("Time is not on the exact KEELE zero-origin 10 ms grid")
    return sample


@dataclass(frozen=True)
class KeeleTiming:
    target_sample: np.ndarray
    support_start_sample: np.ndarray
    support_end_sample: np.ndarray
    earliest_availability_sample: np.ndarray
    complete_support: np.ndarray
    common_eligible: np.ndarray


def timing_for_targets(target_samples, pcm_sample_count: int) -> KeeleTiming:
    count = _integer(pcm_sample_count, "PCM sample count")
    target = _samples(target_samples, "target samples")
    if np.any(target % HOP) or (len(target) > 1 and np.any(np.diff(target) <= 0)):
        raise ValueError("Targets must be unique increasing points on the exact KEELE grid")
    if np.any(target >= count):
        raise ValueError("Reference target outside microphone recording")
    start, end = target - DELAY, target + DELAY
    complete = (start >= 0) & (end <= count)
    # Integer 64 ms / 32 ms mask, independent of reference values and predictions.
    common = complete & (target >= 3072) & (target + 1536 <= count)
    return KeeleTiming(*[_readonly(x) for x in (target, start, end, end, complete, common)])


def intended_timing(pcm_sample_count: int) -> KeeleTiming:
    """Every source-grid target strictly within the finite microphone duration."""
    count = _integer(pcm_sample_count, "PCM sample count")
    return timing_for_targets(np.arange(0, count, HOP, dtype=np.int64), count)


def join_support_ends(support_end_samples, *, source_offset: int, first_admitted_block_start: int,
                      pcm_sample_count: int) -> KeeleTiming:
    """Map actual browser supports; replay endpoint referenceIndex is not used."""
    offset = _integer(source_offset, "source offset")
    first = _integer(first_admitted_block_start, "first admitted block start")
    if first < offset or (first - offset) % HOP:
        raise ValueError("Restart/source phase mismatch: exact reference scoring is invalid")
    end = _samples(support_end_samples, "support ends")
    if np.any(end < first + WINDOW):
        raise ValueError("Result has incomplete support after the actual segment reset")
    timing = timing_for_targets(end - DELAY - offset, pcm_sample_count)
    if not np.all(timing.complete_support):
        raise ValueError("Result support extends outside the finite source")
    return timing


@dataclass(frozen=True)
class KeeleReference:
    source_id: str
    timing: KeeleTiming
    raw_pitch_hz: np.ndarray
    status: np.ndarray
    out_of_range: np.ndarray
    scoring_eligible: np.ndarray

    def counts(self) -> dict:
        result = {key: int(np.sum(self.status == key)) for key in
                  ("voiced", "unvoiced", "uncertain", "invalid")}
        result.update(total=len(self.status), boundary=int(np.sum(~self.timing.common_eligible)),
                      out_of_range=int(np.sum(self.out_of_range)),
                      eligible_reference=int(np.sum(self.scoring_eligible)))
        return result


def reference_from_arrays(*, source_id: str, reference_source_id: str, time_seconds,
                          pitch_hz, pcm_sample_count: int, expected_reference_rows: int) -> KeeleReference:
    """Scorer-only package representation, validated without fitted timing repair.

    Finite negative values remain uncertain whatever their magnitude. Positive
    references outside estimator bounds remain voiced in unconditional counts.
    Nonfinite values are invalid, retained and counted, never made unvoiced.
    """
    if not source_id or source_id != reference_source_id:
        raise ValueError("Microphone/reference source identity mismatch")
    count = _integer(expected_reference_rows, "expected reference rows", 1)
    time = np.asarray(time_seconds)
    raw = np.asarray(pitch_hz)
    if (time.ndim != 1 or raw.ndim != 1 or len(time) != count or len(raw) != count
            or time.dtype.kind not in "fiu" or raw.dtype.kind not in "fiu"):
        raise ValueError("Malformed reference columns or frozen row-count mismatch")
    # This reproduces the pinned converter representation, rather than estimating
    # time from values, accepting an epsilon, or rounding nearest labels.
    expected_time = np.arange(count, dtype=np.int64) / 100
    if not np.array_equal(time, expected_time):
        raise ValueError("Reference timestamps differ from the pinned zero-origin converter grid")
    timing = timing_for_targets(np.arange(count, dtype=np.int64) * HOP, pcm_sample_count)
    raw = np.asarray(raw, dtype=np.float64)
    finite = np.isfinite(raw)
    status = np.full(count, "invalid", dtype="U9")
    status[finite & (raw < 0)] = "uncertain"
    status[finite & (raw == 0)] = "unvoiced"
    status[finite & (raw > 0)] = "voiced"
    out_of_range = finite & (raw > 0) & ((raw < 50) | (raw > 700))
    valid = (status == "voiced") | (status == "unvoiced")
    return KeeleReference(source_id, timing, _readonly(raw), _readonly(status),
                          _readonly(out_of_range), _readonly(timing.common_eligible & valid))


def validate_metadata_binding(*, source_id: str, item_metadata: dict,
                              signal_metadata: dict, pitch_metadata: dict) -> dict:
    """Validate declared JSON roles only; does not open microphone/reference data.

    Declared END counts still require verification against actual file headers
    after admission. A metadata pass cannot authorize scientific evaluation.
    """
    def normalized(value):
        if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
            raise ValueError("Malformed metadata object")
        result = {k.strip(): v for k, v in value.items()}
        if len(result) != len(value):
            raise ValueError("Ambiguous whitespace-normalized metadata keys")
        return result
    item, signal, pitch = map(normalized, (item_metadata, signal_metadata, pitch_metadata))
    try:
        if (not source_id or item["SRC"] != source_id + ".pes" or pitch["SRC"] != item["SRC"]
            or int(item["BEG"]) != 0 or int(pitch["BEG"]) != 0
            or int(item["SAM"]) != SOURCE_RATE or int(signal["samplerate"]) != SOURCE_RATE
            or int(pitch["SAM"]) != 100 or int(pitch["samplerate"]) != 100
            or signal["_filename"] != f"KEELE/{source_id}/signal.wav"
            or pitch["_filename"] != f"KEELE/{source_id}/pitch.npy"):
            raise ValueError("Microphone/reference metadata role, rate or origin mismatch")
        samples, rows = int(item["END"]), int(pitch["END"])
        if samples <= 0 or rows <= 0 or (rows - 1) * 200 >= samples:
            raise ValueError("Declared microphone/reference duration mismatch")
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Malformed or incomplete metadata") from error
    return {"source_id": source_id, "declared_source_samples": samples,
            "declared_reference_rows": rows,
            "declared_trailing_source_samples": samples - rows * 200,
            "declared_missing_reference_grid_targets": max(0, (samples + 199) // 200 - rows),
            "data_headers_verified": False,
            "external_assessment_admitted": False}


def read_reference_npy(path: Path, *, source_id: str, reference_source_id: str,
                       expected_sha256: str, pcm_sample_count: int,
                       expected_reference_rows: int) -> KeeleReference:
    """Explicit scorer-only I/O; caller must freeze input-only predictions first."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Reference digest mismatch")
    import io
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
        if not isinstance(array, np.ndarray) or array.ndim != 1 or array.dtype.names != ("time", "pitch"):
            raise ValueError("Expected the pinned structured time/pitch reference array")
        return reference_from_arrays(source_id=source_id, reference_source_id=reference_source_id,
            time_seconds=array["time"], pitch_hz=array["pitch"], pcm_sample_count=pcm_sample_count,
            expected_reference_rows=expected_reference_rows)
    except (OSError, EOFError, TypeError, AttributeError) as error:
        raise ValueError("Malformed KEELE reference file") from error


def resample_microphone(pcm, *, source_rate_hz=SOURCE_RATE) -> tuple[np.ndarray, dict]:
    """Whole-recording OFFLINE preparation, never a causal device pipeline.

    Symmetric 481-tap FIR, 12/5 rational resampling, zero extension at both ends,
    centered output with sample-zero anchoring. Output length ceil(N*12/5).
    No clipping, normalization or reference-dependent processing occurs.
    """
    if _integer(source_rate_hz, "source sample rate", 1) != SOURCE_RATE:
        raise ValueError("The authenticated KEELE source contract requires 20000 Hz")
    source = np.asarray(pcm)
    if source.ndim != 1 or len(source) == 0 or source.dtype.kind not in "fiu" or not np.all(np.isfinite(source)):
        raise ValueError("Microphone must be nonempty finite mono numeric PCM")
    import scipy
    from scipy.signal import firwin, resample_poly
    if scipy.__version__ != SCIPY_VERSION:
        raise ValueError(f"Resampling requires the frozen scipy {SCIPY_VERSION}, found {scipy.__version__}")
    source = np.asarray(source, dtype="<f8")
    # Explicit settings and coefficients: no dependence on scipy's default filter.
    half_length = 20 * 12
    coefficients = firwin(2 * half_length + 1, 1 / 12, window=("kaiser", 5.0)).astype("<f8")
    output = resample_poly(source, 12, 5, window=coefficients, padtype="constant", cval=0).astype("<f4")
    expected_count = (len(source) * 12 + 4) // 5
    if len(output) != expected_count or not np.all(np.isfinite(output)):
        raise ValueError("Resampler output violates finite PCM/sample count contract")
    receipt = {"schema": SCHEMA, "implementation": "scipy.signal.resample_poly",
        "scipy_version": scipy.__version__, "numpy_version": np.__version__,
        "input_rate_hz": SOURCE_RATE, "output_rate_hz": RATE, "up": 12, "down": 5,
        "input_sample_count": len(source), "output_sample_count": len(output),
        "output_length_rule": "ceil(input_sample_count*12/5)",
        "filter": {"design": "scipy.signal.firwin", "taps": len(coefficients),
                   "cutoff_nyquist": 1 / 12, "window": ["kaiser", 5.0],
                   "coefficients_f64le_sha256": hashlib.sha256(coefficients.tobytes()).hexdigest()},
        "padding": "zero extension both ends; centered FIR delay removed by resample_poly",
        "source_zero_output_sample": 0, "arithmetic": "float64 then float32 little endian",
        "input_f64le_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "output_f32le_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "offline_preparation": True, "clipping_or_normalization": False}
    return _readonly(output), receipt


def contract() -> dict:
    return {"schema": SCHEMA, "admission": "synthetic adapter verified; real corpus intake/freeze required",
        "converter_commit": CONVERTER_COMMIT, "converter_sha256": CONVERTER_SHA256,
        "sample_rate_hz": RATE, "source_rate_hz": SOURCE_RATE, "reference_origin_sample": 0,
        "hop_samples": HOP, "support_samples": WINDOW, "measurement_delay_samples": DELAY,
        "first_complete_target_sample": 960, "first_common_eligible_target_sample": 3360,
        "reference_join": "support_end - 960 - source_offset; exact, never nearest",
        "reference_status": {"finite_positive": "voiced", "zero": "unvoiced",
                             "finite_negative": "uncertain", "nonfinite": "invalid"},
        "out_of_range_positive": "retain voiced denominator; separately count outside inclusive 50–700 Hz",
        "resampling": {"scipy_version": SCIPY_VERSION, "up": 12, "down": 5,
                       "fir_taps": 481, "window": ["kaiser", 5.0], "padding": "constant zero"}}


def main():
    parser = argparse.ArgumentParser(description="Print the KEELE v1 timing contract without accessing corpus data")
    parser.add_argument("command", choices=["contract"])
    parser.parse_args()
    print(json.dumps(contract(), indent=2))


if __name__ == "__main__":
    main()
