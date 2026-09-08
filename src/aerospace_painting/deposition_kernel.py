"""Small, interpretable deposition kernels for the portable runtime.

This module intentionally provides an analytic S1 baseline only.  It is not a
CFD result and is not calibrated to a spray gun.  The same function can later
be fitted to a CFD/reference dataset without changing the surface-local data
contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnisotropicGaussian:
    """Mass-normalised elliptical footprint in local ``(u, v)`` coordinates."""

    mass_kg: float
    centroid_uv_m: tuple[float, float]
    sigma_major_m: float
    sigma_minor_m: float
    rotation_deg: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.mass_kg,
            *self.centroid_uv_m,
            self.sigma_major_m,
            self.sigma_minor_m,
            self.rotation_deg,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("kernel parameters must be finite")
        if self.mass_kg < 0.0:
            raise ValueError("mass_kg must be non-negative")
        if self.sigma_major_m <= 0.0 or self.sigma_minor_m <= 0.0:
            raise ValueError("kernel sigmas must be positive")

    def density(self, uv_m: np.ndarray) -> np.ndarray:
        """Return deposited-mass density in kg/m² at ``[..., 2]`` points."""

        points = np.asarray(uv_m, dtype=float)
        if points.shape[-1] != 2:
            raise ValueError("uv_m must end in a length-2 dimension")
        centered = points - np.asarray(self.centroid_uv_m, dtype=float)
        angle = np.deg2rad(self.rotation_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        major = centered[..., 0] * cos_a + centered[..., 1] * sin_a
        minor = -centered[..., 0] * sin_a + centered[..., 1] * cos_a
        normaliser = 2.0 * np.pi * self.sigma_major_m * self.sigma_minor_m
        return (self.mass_kg / normaliser) * np.exp(
            -0.5
            * (
                (major / self.sigma_major_m) ** 2
                + (minor / self.sigma_minor_m) ** 2
            )
        )

    def integrate_regular_grid(self, uv_m: np.ndarray, cell_area_m2: float) -> float:
        """Approximate integrated mass on a regular grid."""

        if cell_area_m2 <= 0.0 or not np.isfinite(cell_area_m2):
            raise ValueError("cell_area_m2 must be finite and positive")
        return float(np.sum(self.density(uv_m)) * cell_area_m2)
