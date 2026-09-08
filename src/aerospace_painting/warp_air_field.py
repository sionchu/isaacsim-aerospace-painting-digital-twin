"""Portable external-air model used by the W1 Warp validation.

The model is deliberately small: an axisymmetric Gaussian jet whose speed
decays along the nozzle-local spray axis.  Its parameters are fitted to the
solved OpenFOAM carrier velocity samples, never to a deposition map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1.0e-12


def frame_axes(incidence_angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nozzle-local ``(+X, +Y, +Z)`` axes in world coordinates."""

    angle = np.deg2rad(float(incidence_angle_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    # The teacher rotates the inlet and injection about world +Y.
    return (
        np.asarray((c, 0.0, -s), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
        np.asarray((s, 0.0, c), dtype=float),
    )


@dataclass(frozen=True)
class AirFieldParameters:
    """Axisymmetric external carrier-air jet parameters."""

    u0_m_s: float
    sigma0_m: float
    sigma_slope: float
    decay_z0_m: float
    decay_exponent: float

    def __post_init__(self) -> None:
        values = (
            self.u0_m_s,
            self.sigma0_m,
            self.sigma_slope,
            self.decay_z0_m,
            self.decay_exponent,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("air-field parameters must be finite")
        if self.u0_m_s <= 0.0:
            raise ValueError("u0_m_s must be positive")
        if self.sigma0_m <= 0.0:
            raise ValueError("sigma0_m must be positive")
        if self.sigma_slope < 0.0:
            raise ValueError("sigma_slope must be non-negative")
        if self.decay_z0_m <= 0.0:
            raise ValueError("decay_z0_m must be positive")
        if self.decay_exponent <= 0.0:
            raise ValueError("decay_exponent must be positive")

    def to_dict(self) -> dict[str, float]:
        return {
            "u0_m_s": float(self.u0_m_s),
            "sigma0_m": float(self.sigma0_m),
            "sigma_slope": float(self.sigma_slope),
            "decay_z0_m": float(self.decay_z0_m),
            "decay_exponent": float(self.decay_exponent),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "AirFieldParameters":
        return cls(
            u0_m_s=float(payload["u0_m_s"]),
            sigma0_m=float(payload["sigma0_m"]),
            sigma_slope=float(payload["sigma_slope"]),
            decay_z0_m=float(payload["decay_z0_m"]),
            decay_exponent=float(payload["decay_exponent"]),
        )


def local_coordinates(
    points_world: np.ndarray,
    incidence_angle_deg: float,
    *,
    nozzle_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.002),
) -> np.ndarray:
    """Transform points to the teacher-compatible nozzle-local frame."""

    points = np.asarray(points_world, dtype=float)
    if points.shape[-1] != 3:
        raise ValueError("points_world must end in a length-3 dimension")
    x_axis, y_axis, z_axis = frame_axes(incidence_angle_deg)
    relative = points - np.asarray(nozzle_origin_m, dtype=float)
    return np.stack(
        (
            np.einsum("...i,i->...", relative, x_axis),
            np.einsum("...i,i->...", relative, y_axis),
            np.einsum("...i,i->...", relative, z_axis),
        ),
        axis=-1,
    )


def axial_speed_local(local_points: np.ndarray, parameters: AirFieldParameters) -> np.ndarray:
    """Evaluate the scalar Gaussian-jet speed in nozzle-local coordinates."""

    points = np.asarray(local_points, dtype=float)
    if points.shape[-1] != 3:
        raise ValueError("local_points must end in a length-3 dimension")
    z = np.maximum(points[..., 2], 0.0)
    sigma = parameters.sigma0_m + parameters.sigma_slope * z
    decay = (parameters.decay_z0_m / (z + parameters.decay_z0_m)) ** parameters.decay_exponent
    radial = (points[..., 0] / sigma) ** 2 + (points[..., 1] / sigma) ** 2
    return parameters.u0_m_s * decay * np.exp(-0.5 * radial)


def velocity_world(
    points_world: np.ndarray,
    incidence_angle_deg: float,
    parameters: AirFieldParameters,
    *,
    nozzle_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.002),
) -> np.ndarray:
    """Evaluate the carrier velocity vector in world coordinates."""

    local = local_coordinates(points_world, incidence_angle_deg, nozzle_origin_m=nozzle_origin_m)
    speed = axial_speed_local(local, parameters)
    return speed[..., None] * frame_axes(incidence_angle_deg)[2]


def schiller_naumann_cd(reynolds_number: np.ndarray | float) -> np.ndarray:
    """Sphere drag coefficient with the standard high-Re continuation."""

    re = np.asarray(reynolds_number, dtype=float)
    if np.any(~np.isfinite(re)) or np.any(re < 0.0):
        raise ValueError("Reynolds number must be finite and non-negative")
    safe = np.maximum(re, _EPS)
    low_re = 24.0 / safe * (1.0 + 0.15 * safe**0.687)
    return np.where(re <= 1000.0, low_re, 0.44)
