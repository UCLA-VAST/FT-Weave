"""Helpers for Rz instruction parameters."""

from __future__ import annotations

import math
from numbers import Real


def rz_target_angles(instruction: dict) -> dict[int, float]:
    """Return per-qubit rotation angles for an Rz layer instruction."""
    params = instruction.get("params", {})
    targets = instruction.get("targets", [])
    if not isinstance(targets, list):
        raise ValueError("Rz instruction targets must be a list")

    if "angles" in params:
        raw = params["angles"]
        if not isinstance(raw, dict):
            raise ValueError(
                "Rz params['angles'] must be a dict mapping qubit -> theta"
            )
        return {int(q): float(raw[q]) for q in targets}

    theta = params.get("theta")
    if not isinstance(theta, Real):
        raise ValueError("Rz instruction requires params['theta'] or params['angles']")
    angle = float(theta)
    return {int(q): angle for q in targets}


def normalize_rz_angle(theta: float) -> float:
    return math.remainder(theta, 2 * math.pi)


def combine_rz_angles(
    existing: dict[int, float], new: dict[int, float]
) -> dict[int, float]:
    """Combine consecutive Rz layers on the same qubits by summing angles."""
    merged = dict(existing)
    for qubit, theta in new.items():
        q = int(qubit)
        merged[q] = normalize_rz_angle(merged.get(q, 0.0) + float(theta))
    return merged


def rz_params_from_angles(angles: dict[int, float]) -> dict:
    """Build params dict for an Rz layer from per-qubit angles."""
    params: dict = {"angles": dict(angles)}
    unique = set(angles.values())
    if len(unique) == 1:
        params["theta"] = next(iter(unique))
    return params
