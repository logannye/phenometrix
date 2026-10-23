"""Exact sample coordinates for the separate declared-delay measurement experiment.

The original endpoint experiment remains delay=0 and retains its failed gate.
For delay=20 ms, a feature whose support ends at e estimates t=e-20 ms and
cannot become available before e. No signal, reference label or shift fitting is
performed here. Cache timestamps remain acquisition endpoints and are immutable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class AlignedFeatureTiming:
    support_start_sample: np.ndarray
    support_end_sample_exclusive: np.ndarray
    reference_sample: np.ndarray
    availability_sample: np.ndarray
    last_required_input_sample: np.ndarray
    common_eligible: np.ndarray
    reset_before_frame: np.ndarray
    sample_rate_hz: int
    measurement_delay_ms: int

    def metadata(self):
        return {"schema": "phenometrix.declared-measurement-delay.v1",
                "sample_rate_hz": self.sample_rate_hz, "support_ms": 40,
                "hop_ms": 10, "reference_grid_origin_ms": 16,
                "measurement_delay_ms": self.measurement_delay_ms,
                "lookahead_relative_to_reference_ms": self.measurement_delay_ms,
                "reference_relation": "exclusive_support_end_minus_declared_delay",
                "availability_relation": "exclusive_support_end_before_transport_and_service",
                "shared_boundary": "reference>=64ms and reference+32ms<=recording sample count",
                "gap_policy": "reject discontinuous cache; regenerate with explicit segment resets"}


def align_feature_timing(feature: Mapping, *, measurement_delay_ms: int) -> AlignedFeatureTiming:
    """Map input-only feature-cache endpoints; never overwrite legacy sample keys.

    Only the original 0 ms and newly declared 20 ms contracts are admitted. This
    helper intentionally has no offset-search or per-recording alignment option.
    """
    if measurement_delay_ms not in (0, 20):
        raise ValueError("Only the frozen 0 ms and 20 ms measurement contracts are supported")
    rate_value = np.asarray(feature["sample_rate_hz"]).item()
    count_value = np.asarray(feature["pcm_sample_count"]).item()
    if not isinstance(rate_value, (int, np.integer)) or rate_value < 8000 or rate_value % 100:
        raise ValueError("Sample rate must have exact 10 ms sample counts")
    if not isinstance(count_value, (int, np.integer)) or count_value < 0:
        raise ValueError("Invalid recording sample count")
    rate, count = int(rate_value), int(count_value)
    raw = np.asarray(feature["reference_sample"])
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Feature availability endpoints must be a one-dimensional integer array")
    end = raw.astype(np.int64, copy=True)
    values = np.asarray(feature["features"])
    if values.ndim != 2 or len(values) != len(end):
        raise ValueError("Feature and timestamp row counts differ")
    window, hop, delay = rate * 40 // 1000, rate // 100, rate * measurement_delay_ms // 1000
    origin = round(rate * .016)
    if len(end) and (np.any(end < window) or np.any(end > count) or np.any((end - origin) % hop != 0)):
        raise ValueError("Feature endpoint violates complete support or the frozen reference grid")
    if len(end) > 1 and np.any(np.diff(end) != hop):
        raise ValueError("Feature cache is discontinuous; regenerate with explicit segment state resets")
    target = end - delay
    common = (target >= math.ceil(rate * .064)) & (target + math.ceil(rate * .032) <= count)
    reset = np.zeros(len(end), dtype=bool)
    if len(reset):
        reset[0] = True
    arrays = [end - window, end, target, end.copy(), end - 1, common, reset]
    for array in arrays:
        array.setflags(write=False)
    return AlignedFeatureTiming(*arrays, sample_rate_hz=rate, measurement_delay_ms=measurement_delay_ms)
