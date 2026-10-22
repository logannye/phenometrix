"""Scorer-only, exact-grid metrics. No reference-dependent prediction path."""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
from typing import Literal, Sequence


@dataclass(frozen=True)
class Reference:
    sample: int
    f0_hz: float | None
    voiced: bool
    valid: bool = True
    eligible: bool = True


@dataclass(frozen=True)
class Prediction:
    sample: int
    state: Literal["voiced", "unvoiced", "unavailable"]
    f0_hz: float | None
    correctness_probability: float | None = None


def cents(prediction: float, reference: float) -> float:
    if not all(math.isfinite(x) and x > 0 for x in (prediction, reference)):
        raise ValueError("Cents requires finite positive F0.")
    return 1200 * math.log2(prediction / reference)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def score(references: Sequence[Reference], predictions: Sequence[Prediction],
          tolerance_cents: float = 50) -> dict:
    """Count abstention as failure. Missing predictions are explicit unavailability.

    No interpolation, time-shift search, alignment fitting or label repair occurs here.
    Invalid reference rows are excluded and counted. Bounds do not exclude voiced rows.
    """
    if not math.isfinite(tolerance_cents) or tolerance_cents <= 0:
        raise ValueError("Positive finite tolerance required.")
    if len({r.sample for r in references}) != len(references):
        raise ValueError("Duplicate reference target.")
    by_sample = {p.sample: p for p in predictions}
    if len(by_sample) != len(predictions):
        raise ValueError("Duplicate prediction target.")
    if set(by_sample) - {r.sample for r in references}:
        raise ValueError("Prediction does not match the exact reference grid.")
    for p in predictions:
        if p.state not in ("voiced", "unvoiced", "unavailable"):
            raise ValueError("Unknown prediction state.")
        if p.state == "voiced":
            if p.f0_hz is None or not math.isfinite(p.f0_hz) or p.f0_hz <= 0:
                raise ValueError("Voiced predictions need a finite positive F0.")
        elif p.f0_hz is not None:
            raise ValueError("Unvoiced/unavailable F0 must be null, never zero.")
        if p.correctness_probability is not None and not (
                math.isfinite(p.correctness_probability) and 0 <= p.correctness_probability <= 1):
            raise ValueError("Invalid correctness probability.")
    n = nv = nu = correct = voiced_success = fp = fn = abstain = jointly_voiced = octaves = outside = strict_fn = voiced_unavailable = 0
    errors: list[float] = []
    calibration: list[tuple[float, int]] = []
    for r in references:
        if not r.valid or not r.eligible:
            continue
        if r.voiced and (r.f0_hz is None or not math.isfinite(r.f0_hz) or r.f0_hz <= 0):
            raise ValueError("Valid voiced reference needs finite positive F0.")
        if not r.voiced and r.f0_hz is not None:
            raise ValueError("Unvoiced reference F0 must be null.")
        p = by_sample.get(r.sample, Prediction(r.sample, "unavailable", None))
        n += 1
        nv += int(r.voiced)
        nu += int(not r.voiced)
        abstain += int(p.state == "unavailable")
        fp += int(not r.voiced and p.state == "voiced")
        fn += int(r.voiced and p.state != "voiced")
        strict_fn += int(r.voiced and p.state == "unvoiced")
        voiced_unavailable += int(r.voiced and p.state == "unavailable")
        outside += int(r.voiced and not 50 <= r.f0_hz <= 700)
        ok = not r.voiced and p.state == "unvoiced"
        if r.voiced and p.state == "voiced":
            signed = cents(p.f0_hz, r.f0_hz)
            error = abs(signed)
            errors.append(error)
            jointly_voiced += 1
            ok = error <= tolerance_cents
            voiced_success += int(ok)
            octave = round(signed / 1200)
            octaves += int(octave != 0 and abs(signed - octave * 1200) <= tolerance_cents)
        correct += int(ok)
        if p.state == "voiced" and p.correctness_probability is not None:
            calibration.append((p.correctness_probability, int(ok)))
    bins = []
    for index in range(10):
        rows = [(p, y) for p, y in calibration if min(9, int(p * 10)) == index]
        bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                     "n": len(rows), "mean_probability": mean(p for p, _ in rows) if rows else None,
                     "observed_correct_fraction": mean(y for _, y in rows) if rows else None})
    return {
        "eligible_valid_frames": n, "reference_voiced_frames": nv, "reference_unvoiced_frames": nu,
        "excluded_invalid_reference": sum(not r.valid for r in references),
        "excluded_boundary": sum(r.valid and not r.eligible for r in references),
        "reference_outside_supported_bounds": outside,
        "correct_frames": correct, "joint_error": _ratio(n - correct, n),
        "voicing_false_positive_count": fp, "voicing_false_positive_rate": _ratio(fp, nu),
        "voicing_false_negative_count": strict_fn, "voicing_false_negative_rate": _ratio(strict_fn, nv),
        "unavailable_reference_voiced_count": voiced_unavailable,
        "voicing_false_negative_or_unavailable_count": fn,
        "voicing_false_negative_or_unavailable_rate": _ratio(fn, nv),
        "reference_voiced_success_count": voiced_success,
        "reference_voiced_success_rate": _ratio(voiced_success, nv),
        "unavailable_frames": abstain, "measurement_coverage": _ratio(n - abstain, n),
        "jointly_voiced_frames": jointly_voiced, "jointly_voiced_coverage": _ratio(jointly_voiced, nv),
        "median_jointly_voiced_cents": _quantile(errors, .5),
        "p95_jointly_voiced_cents": _quantile(errors, .95),
        "octave_error_count": octaves, "octave_error_rate_reference_voiced": _ratio(octaves, nv),
        "calibration_n": len(calibration),
        "brier_score": mean((p - y) ** 2 for p, y in calibration) if calibration else None,
        "reliability_bins": bins,
    }


def oracle_coverage(references: Sequence[Reference], candidate_hz: dict[int, list[float]],
                    tolerance_cents: float = 50) -> dict:
    rows = [r for r in references if r.valid and r.eligible and r.voiced]
    covered = sum(any(abs(cents(hz, r.f0_hz)) <= tolerance_cents
                      for hz in candidate_hz.get(r.sample, [])) for r in rows)
    return {"reference_voiced_frames": len(rows), "covered_frames": covered,
            "coverage": _ratio(covered, len(rows))}


def error_coverage_curve(references: Sequence[Reference], predictions: Sequence[Prediction],
                         thresholds: Sequence[float]) -> list[dict]:
    result = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be in [0,1].")
        kept = [Prediction(p.sample, "unavailable", None)
                if p.state == "voiced" and (p.correctness_probability is None or p.correctness_probability < threshold)
                else p for p in predictions]
        result.append({"threshold": threshold, **score(references, kept)})
    return result
