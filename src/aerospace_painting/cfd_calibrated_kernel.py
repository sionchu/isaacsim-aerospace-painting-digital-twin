"""Interpretable S2 surrogate calibrated from OpenFOAM footprint moments.

The canonical representation is mass, centroid, and covariance.  Rendering
delegates to the existing :class:`AnisotropicGaussian` implementation so the
project has one Gaussian density implementation rather than separate S1/S2
versions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deposition_kernel import AnisotropicGaussian


_COVARIANCE_TOL = 1.0e-12


def _normalise_orientation_deg(angle_deg: float) -> float:
    angle = float(angle_deg)
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


@dataclass(frozen=True)
class GaussianMoments:
    """Mass-normalised footprint moments in the surface-local ``(u, v)`` frame."""

    deposited_mass_kg: float
    centroid_uv_m: tuple[float, float]
    covariance_uv_m2: np.ndarray

    def __post_init__(self) -> None:
        mass = float(self.deposited_mass_kg)
        centroid = tuple(float(value) for value in self.centroid_uv_m)
        covariance = np.asarray(self.covariance_uv_m2, dtype=float)
        if len(centroid) != 2 or not np.all(np.isfinite(centroid)):
            raise ValueError("centroid_uv_m must be two finite values")
        if not np.isfinite(mass) or mass < 0.0:
            raise ValueError("deposited_mass_kg must be finite and non-negative")
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            raise ValueError("covariance_uv_m2 must be a finite 2x2 matrix")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=_COVARIANCE_TOL):
            raise ValueError("covariance_uv_m2 must be symmetric")
        eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) * 0.5)
        if float(eigenvalues.min()) < -_COVARIANCE_TOL:
            raise ValueError("covariance_uv_m2 must be positive semidefinite")
        covariance = (covariance + covariance.T) * 0.5
        covariance.setflags(write=False)
        object.__setattr__(self, "deposited_mass_kg", mass)
        object.__setattr__(self, "centroid_uv_m", centroid)
        object.__setattr__(self, "covariance_uv_m2", covariance)

    @classmethod
    def from_metrics(cls, metrics: dict) -> "GaussianMoments":
        """Construct moments from one canonical post-processing metrics object."""
        deposition = metrics["deposition"]
        return cls(
            deposited_mass_kg=float(metrics["mass_ledger"]["deposited_kg"]),
            centroid_uv_m=(
                float(deposition["centroid_u_m"]),
                float(deposition["centroid_v_m"]),
            ),
            covariance_uv_m2=np.asarray(deposition["covariance_uv_m2"], dtype=float),
        )

    def to_anisotropic_gaussian(self) -> AnisotropicGaussian:
        """Convert covariance moments to the existing S1 Gaussian renderer."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_uv_m2)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], _COVARIANCE_TOL)
        major_axis = eigenvectors[:, order[0]]
        orientation = _normalise_orientation_deg(
            np.degrees(np.arctan2(major_axis[1], major_axis[0]))
        )
        return AnisotropicGaussian(
            mass_kg=self.deposited_mass_kg,
            centroid_uv_m=self.centroid_uv_m,
            sigma_major_m=float(np.sqrt(eigenvalues[0])),
            sigma_minor_m=float(np.sqrt(eigenvalues[1])),
            rotation_deg=orientation,
        )

    def density(self, uv_m: np.ndarray) -> np.ndarray:
        """Evaluate the moment-matched Gaussian density in kg/m²."""
        return self.to_anisotropic_gaussian().density(uv_m)

    def integrate_regular_grid(self, uv_m: np.ndarray, cell_area_m2: float) -> float:
        """Integrate the rendered density over a regular grid."""
        return self.to_anisotropic_gaussian().integrate_regular_grid(uv_m, cell_area_m2)


def interpolate_moments(
    incidence_angle_deg: float,
    endpoint_zero: GaussianMoments,
    endpoint_fifteen: GaussianMoments,
    *,
    maximum_angle_deg: float = 15.0,
) -> GaussianMoments:
    """Linearly interpolate mass, centroid, and covariance within S2's domain."""
    theta = float(incidence_angle_deg)
    maximum = float(maximum_angle_deg)
    if not np.isfinite(theta) or not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("incidence angle and calibrated maximum must be finite")
    if theta < 0.0 or theta > maximum:
        raise ValueError(f"incidence_angle_deg must be within [0, {maximum:g}] degrees")
    t = theta / maximum
    mass = (1.0 - t) * endpoint_zero.deposited_mass_kg + t * endpoint_fifteen.deposited_mass_kg
    centroid = tuple(
        (1.0 - t) * a + t * b
        for a, b in zip(endpoint_zero.centroid_uv_m, endpoint_fifteen.centroid_uv_m)
    )
    covariance = (1.0 - t) * endpoint_zero.covariance_uv_m2 + t * endpoint_fifteen.covariance_uv_m2
    return GaussianMoments(mass, centroid, covariance)


def predict_deposition_kernel(
    incidence_angle_deg: float,
    endpoint_zero: GaussianMoments,
    endpoint_fifteen: GaussianMoments,
) -> GaussianMoments:
    """Portable S2 v1 API for the calibrated 0-15 degree interval."""
    return interpolate_moments(incidence_angle_deg, endpoint_zero, endpoint_fifteen)
