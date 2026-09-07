"""Transparent geometric spray-cone calculations."""

from __future__ import annotations

import numpy as np


def cone_sample(nozzle, spray_axis, point, half_angle_deg: float) -> tuple[bool, float, float]:
    """Return cone membership, nozzle distance, and axis angle in degrees."""
    origin = np.asarray(nozzle, dtype=float)
    axis = np.asarray(spray_axis, dtype=float)
    target = np.asarray(point, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        raise ValueError("spray_axis must be non-zero")
    axis /= axis_norm
    vector = target - origin
    distance = float(np.linalg.norm(vector))
    if distance < 1e-12:
        return True, 0.0, 0.0
    direction = vector / distance
    angle = float(np.degrees(np.arccos(np.clip(np.dot(axis, direction), -1.0, 1.0))))
    return angle <= half_angle_deg, distance, angle


def geometric_weight(
    distance: float,
    incidence_angle_deg: float,
    nominal_stand_off: float,
    cone_half_angle_deg: float,
) -> float:
    """Compute a bounded distance-and-angle process score, not paint thickness."""
    distance_weight = max(
        0.0,
        1.0 - abs(distance - nominal_stand_off) / max(nominal_stand_off, 1e-9) / 1.8,
    )
    angle_weight = max(
        0.0,
        1.0 - incidence_angle_deg / max(cone_half_angle_deg, 1e-9),
    )
    return float(distance_weight * angle_weight)
