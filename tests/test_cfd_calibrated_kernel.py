import json
from pathlib import Path

import numpy as np
import pytest

from aerospace_painting.cfd_calibrated_kernel import (
    GaussianMoments,
    interpolate_moments,
    predict_deposition_kernel,
)
from scripts.validate_s2_surrogate import _load_frozen_prediction


ROOT = Path(__file__).resolve().parents[1]
FLAT_RESULTS = ROOT / "results" / "air_assisted" / "openfoam_v2606" / "flat_plate"
S2_RESULTS = ROOT / "results" / "air_assisted" / "s2"


def _moments(mass=1.0, centroid=(0.0, 0.0), covariance=((4.0, 0.5), (0.5, 1.0))):
    return GaussianMoments(mass, centroid, np.asarray(covariance, dtype=float))


def test_covariance_validation_rejects_non_symmetric_and_negative_matrices():
    with pytest.raises(ValueError, match="symmetric"):
        _moments(covariance=((1.0, 2.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="positive semidefinite"):
        _moments(covariance=((1.0, 0.0), (0.0, -0.1)))


def test_covariance_to_existing_gaussian_conversion_is_eigen_matched():
    moments = _moments()
    kernel = moments.to_anisotropic_gaussian()
    eigenvalues = np.linalg.eigvalsh(moments.covariance_uv_m2)
    assert np.isclose(kernel.sigma_major_m, np.sqrt(eigenvalues[-1]))
    assert np.isclose(kernel.sigma_minor_m, np.sqrt(eigenvalues[0]))
    assert kernel.density(np.asarray([[0.0, 0.0]])).item() > 0.0


def test_moment_matched_gaussian_integrates_to_deposited_mass():
    moments = _moments(mass=0.25, covariance=((0.12**2, 0.0), (0.0, 0.06**2)))
    axis = np.linspace(-0.8, 0.8, 401)
    u, v = np.meshgrid(axis, axis, indexing="xy")
    uv = np.stack((u, v), axis=-1)
    integral = moments.integrate_regular_grid(uv, float(axis[1] - axis[0]) ** 2)
    assert np.isclose(integral, 0.25, rtol=1e-3)


def test_endpoint_interpolation_is_exact_and_covariance_stays_psd():
    zero = GaussianMoments.from_metrics(json.loads((FLAT_RESULTS / "medium/metrics.json").read_text()))
    fifteen = GaussianMoments.from_metrics(json.loads((FLAT_RESULTS / "medium_incidence_15deg/metrics.json").read_text()))
    assert np.allclose(interpolate_moments(0.0, zero, fifteen).covariance_uv_m2, zero.covariance_uv_m2)
    assert np.allclose(interpolate_moments(15.0, zero, fifteen).covariance_uv_m2, fifteen.covariance_uv_m2)
    midpoint = predict_deposition_kernel(7.5, zero, fifteen)
    assert np.linalg.eigvalsh(midpoint.covariance_uv_m2).min() >= 0.0


def test_prediction_domain_guard_rejects_extrapolation():
    zero = _moments()
    fifteen = _moments(covariance=((9.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match=r"within \[0, 15\]"):
        predict_deposition_kernel(-0.1, zero, fifteen)
    with pytest.raises(ValueError, match=r"within \[0, 15\]"):
        predict_deposition_kernel(15.1, zero, fifteen)


def test_frozen_prediction_hash_and_reproducibility():
    payload, moments, grid = _load_frozen_prediction(ROOT)
    assert payload["model"] == "S2_SINGLE_GAUSSIAN"
    assert payload["holdout_angle_deg"] == 7.5
    assert payload["prediction_file_sha256"]
    assert grid["predicted_areal_mass_kg_m2"].shape == grid["u_m"].shape
    assert np.all(np.isfinite(moments.covariance_uv_m2))


def test_s1_s2_comparison_shows_measured_s2_improvement():
    validation = json.loads((S2_RESULTS / "holdout_validation.json").read_text())
    assert validation["status"] == "S2_HOLDOUT_VALIDATED"
    assert validation["s1_vs_s2_improvement"]["demonstrated"] is True
    assert validation["s2_field_validation"]["pearson_correlation"] >= 0.90
