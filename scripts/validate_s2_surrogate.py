"""Freeze and validate the S2 CFD-calibrated deposition surrogate.

The ``--freeze`` phase reads only the checked-in 0-degree and 15-degree
medium artifacts.  The ``--validate`` phase refuses to rewrite that prediction
and compares it with the separately solved 7.5-degree OpenFOAM case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from aerospace_painting.cfd_calibrated_kernel import (
    GaussianMoments,
    predict_deposition_kernel,
)
from aerospace_painting.deposition_kernel import AnisotropicGaussian

try:
    from scripts.postprocess_openfoam_flat_plate import process_case
    from scripts.validate_openfoam_incidence_response import compare_incidence_inputs
except ModuleNotFoundError:  # direct execution from the scripts directory
    from postprocess_openfoam_flat_plate import process_case
    from validate_openfoam_incidence_response import compare_incidence_inputs


HOLDOUT_ANGLE_DEG = 7.5
CALIBRATED_MAX_DEG = 15.0
S2_ROOT_REL = Path("results") / "air_assisted" / "s2"
FLAT_RESULT_REL = Path("results") / "air_assisted" / "openfoam_v2606" / "flat_plate"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_without_hash(payload: dict) -> bytes:
    canonical = dict(payload)
    canonical["prediction_file_sha256"] = ""
    return (json.dumps(canonical, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _load_metrics(results_root: Path, label: str) -> dict:
    return json.loads((results_root / label / "metrics.json").read_text(encoding="utf-8"))


def _load_map(results_root: Path, label: str) -> dict[str, np.ndarray]:
    with np.load(results_root / label / "deposition_map.npz") as data:
        return {key: np.asarray(data[key], dtype=float) for key in data.files}


def _cell_area(map_data: dict[str, np.ndarray]) -> float:
    area = np.asarray(map_data["area_m2"], dtype=float)
    if np.any(area <= 0.0) or not np.all(np.isfinite(area)):
        raise ValueError("deposition map area must be finite and positive")
    if not np.allclose(area, area[0], rtol=1.0e-10, atol=1.0e-15):
        raise ValueError("S2 v1 requires a uniform face-area grid")
    return float(area[0])


def _map_moments(
    uv_m: np.ndarray,
    density_kg_m2: np.ndarray,
    cell_area_m2: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    values = np.asarray(density_kg_m2, dtype=float)
    points = np.asarray(uv_m, dtype=float)
    weights = values * cell_area_m2
    mass = float(weights.sum())
    if mass <= 0.0:
        return 0.0, np.zeros(2), np.zeros((2, 2))
    centroid = np.average(points, axis=0, weights=weights)
    centered = points - centroid
    covariance = (centered * weights[:, None]).T @ centered / mass
    return mass, centroid, covariance


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std == 0.0 or right_std == 0.0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_field(
    predicted_density_kg_m2: np.ndarray,
    actual_density_kg_m2: np.ndarray,
    uv_m: np.ndarray,
    cell_area_m2: float,
    actual_metrics: dict,
) -> dict[str, float]:
    """Compare two fields on one identical face-center grid."""
    predicted = np.asarray(predicted_density_kg_m2, dtype=float)
    actual = np.asarray(actual_density_kg_m2, dtype=float)
    if predicted.shape != actual.shape:
        raise ValueError("predicted and actual maps must have identical shapes")
    predicted_mass, predicted_centroid, predicted_covariance = _map_moments(uv_m, predicted, cell_area_m2)
    actual_mass, actual_centroid, actual_covariance = _map_moments(uv_m, actual, cell_area_m2)
    actual_deposition = actual_metrics["deposition"]
    actual_covariance = np.asarray(actual_deposition["covariance_uv_m2"], dtype=float)
    actual_centroid = np.asarray(
        [actual_deposition["centroid_u_m"], actual_deposition["centroid_v_m"]],
        dtype=float,
    )
    actual_mass_from_ledger = float(actual_metrics["mass_ledger"]["deposited_kg"])
    covariance_error = float(
        np.linalg.norm(predicted_covariance - actual_covariance)
        / max(np.linalg.norm(actual_covariance), 1.0e-30)
    )
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    density_range = float(np.max(actual) - np.min(actual))
    nrmse = rmse / density_range if density_range > 0.0 else 0.0
    peak = float(np.max(actual))
    return {
        "predicted_integrated_mass_kg": predicted_mass,
        "actual_integrated_mass_kg": actual_mass,
        "ledger_deposited_mass_kg": actual_mass_from_ledger,
        "integrated_mass_error_kg": predicted_mass - actual_mass,
        "integrated_mass_relative_error": abs(predicted_mass - actual_mass) / max(actual_mass, 1.0e-30),
        "integrated_absolute_mass_density_error_kg": float(np.sum(np.abs(predicted - actual)) * cell_area_m2),
        "centroid_error_m": float(np.linalg.norm(predicted_centroid - actual_centroid)),
        "predicted_centroid_u_m": float(predicted_centroid[0]),
        "predicted_centroid_v_m": float(predicted_centroid[1]),
        "covariance_frobenius_relative_error": covariance_error,
        "normalized_rmse": float(nrmse),
        "pearson_correlation": _pearson_correlation(predicted, actual),
        "peak_density_relative_error": abs(float(np.max(predicted)) - peak) / max(peak, 1.0e-30),
        "predicted_peak_density_kg_m2": float(np.max(predicted)),
        "actual_peak_density_kg_m2": peak,
        "predicted_covariance_uv_m2": predicted_covariance.tolist(),
        "actual_covariance_uv_m2": actual_covariance.tolist(),
        "actual_map_mass_kg": actual_mass,
    }


def _shape_from_covariance(covariance: np.ndarray) -> tuple[float, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    major = eigenvectors[:, order[0]]
    orientation = float(np.degrees(np.arctan2(major[1], major[0])))
    while orientation >= 90.0:
        orientation -= 180.0
    while orientation < -90.0:
        orientation += 180.0
    return float(np.sqrt(eigenvalues[0])), float(np.sqrt(eigenvalues[1])), orientation


def _parameter_metrics(predicted: GaussianMoments, actual_metrics: dict, expected_mass_kg: float) -> dict[str, float]:
    actual_deposition = actual_metrics["deposition"]
    actual_mass = float(actual_metrics["mass_ledger"]["deposited_kg"])
    actual_centroid = np.asarray(
        [actual_deposition["centroid_u_m"], actual_deposition["centroid_v_m"]],
        dtype=float,
    )
    actual_covariance = np.asarray(actual_deposition["covariance_uv_m2"], dtype=float)
    predicted_major, predicted_minor, _ = _shape_from_covariance(predicted.covariance_uv_m2)
    actual_major = float(actual_deposition["sigma_major_m"])
    actual_minor = float(actual_deposition["sigma_minor_m"])
    return {
        "predicted_deposited_mass_kg": predicted.deposited_mass_kg,
        "actual_deposited_mass_kg": actual_mass,
        "deposited_mass_relative_error": abs(predicted.deposited_mass_kg - actual_mass) / max(actual_mass, 1.0e-30),
        "predicted_transfer_efficiency": predicted.deposited_mass_kg / expected_mass_kg,
        "actual_transfer_efficiency": float(actual_metrics["mass_ledger"]["transfer_efficiency"]),
        "transfer_efficiency_error": abs(predicted.deposited_mass_kg / expected_mass_kg - float(actual_metrics["mass_ledger"]["transfer_efficiency"])),
        "predicted_centroid_u_m": predicted.centroid_uv_m[0],
        "predicted_centroid_v_m": predicted.centroid_uv_m[1],
        "actual_centroid_u_m": float(actual_centroid[0]),
        "actual_centroid_v_m": float(actual_centroid[1]),
        "centroid_distance_m": float(np.linalg.norm(np.asarray(predicted.centroid_uv_m) - actual_centroid)),
        "covariance_frobenius_relative_error": float(
            np.linalg.norm(predicted.covariance_uv_m2 - actual_covariance)
            / max(np.linalg.norm(actual_covariance), 1.0e-30)
        ),
        "predicted_sigma_major_m": predicted_major,
        "actual_sigma_major_m": actual_major,
        "sigma_major_relative_error": abs(predicted_major - actual_major) / max(actual_major, 1.0e-30),
        "predicted_sigma_minor_m": predicted_minor,
        "actual_sigma_minor_m": actual_minor,
        "sigma_minor_relative_error": abs(predicted_minor - actual_minor) / max(actual_minor, 1.0e-30),
    }


def _config_scalar(config_path: Path, key: str) -> float:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$", text)
    if not match:
        raise ValueError(f"missing scalar {key!r} in {config_path}")
    return float(match.group(1))


def build_s1_baseline(config_path: Path, mass_kg: float) -> AnisotropicGaussian:
    """Build the existing S1 analytic baseline from canonical YAML values."""
    stand_off = _config_scalar(config_path, "stand_off_m")
    major_half_angle = _config_scalar(config_path, "major_half_angle_deg")
    minor_half_angle = _config_scalar(config_path, "minor_half_angle_deg")
    rotation = _config_scalar(config_path, "rotation_deg")
    return AnisotropicGaussian(
        mass_kg=mass_kg,
        centroid_uv_m=(0.0, 0.0),
        sigma_major_m=stand_off * math.tan(math.radians(major_half_angle)),
        sigma_minor_m=stand_off * math.tan(math.radians(minor_half_angle)),
        rotation_deg=rotation,
    )


def _teacher_context(repo: Path) -> tuple[Path, Path, dict, dict, dict, dict]:
    flat_results = repo / FLAT_RESULT_REL
    medium_metrics = _load_metrics(flat_results, "medium")
    incidence_metrics = _load_metrics(flat_results, "medium_incidence_15deg")
    medium_map = _load_map(flat_results, "medium")
    incidence_map = _load_map(flat_results, "medium_incidence_15deg")
    return flat_results, repo / "configs" / "air_assisted_spray.yaml", medium_metrics, incidence_metrics, medium_map, incidence_map


def freeze_prediction(repo: Path) -> dict:
    flat_results, config_path, medium_metrics, incidence_metrics, medium_map, _ = _teacher_context(repo)
    endpoint_zero = GaussianMoments.from_metrics(medium_metrics)
    endpoint_fifteen = GaussianMoments.from_metrics(incidence_metrics)
    teacher_reconstruction = {
        "medium": evaluate_field(
            endpoint_zero.density(np.stack([medium_map["u_m"], medium_map["v_m"]], axis=-1)),
            medium_map["areal_mass_kg_m2"],
            np.stack([medium_map["u_m"], medium_map["v_m"]], axis=-1),
            _cell_area(medium_map),
            medium_metrics,
        ),
        "medium_incidence_15deg": evaluate_field(
            endpoint_fifteen.density(np.stack([medium_map["u_m"], medium_map["v_m"]], axis=-1)),
            _load_map(flat_results, "medium_incidence_15deg")["areal_mass_kg_m2"],
            np.stack([medium_map["u_m"], medium_map["v_m"]], axis=-1),
            _cell_area(medium_map),
            incidence_metrics,
        ),
    }
    single_pass = all(
        item["normalized_rmse"] <= 0.25 and item["pearson_correlation"] >= 0.90
        for item in teacher_reconstruction.values()
    )
    if not single_pass:
        raise RuntimeError("S2 single-Gaussian teacher reconstruction failed; no unapproved mixture was fit")

    prediction = predict_deposition_kernel(HOLDOUT_ANGLE_DEG, endpoint_zero, endpoint_fifteen)
    uv = np.stack([medium_map["u_m"], medium_map["v_m"]], axis=-1)
    prediction_density = prediction.density(uv)
    output_root = repo / S2_ROOT_REL
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_json = output_root / "holdout_7p5_prediction.json"
    prediction_npz = output_root / "holdout_7p5_prediction.npz"
    if prediction_json.exists() or prediction_npz.exists():
        raise FileExistsError("frozen 7.5-degree prediction already exists; refusing to overwrite it")
    np.savez(
        prediction_npz,
        u_m=medium_map["u_m"],
        v_m=medium_map["v_m"],
        predicted_areal_mass_kg_m2=prediction_density,
        cell_area_m2=np.asarray([_cell_area(medium_map)]),
    )
    payload = {
        "schema_version": "s2_holdout_prediction_v1",
        "model": "S2_SINGLE_GAUSSIAN",
        "representation": "deposited_mass_kg + centroid_uv_m + covariance_uv_m2",
        "interpolation_variable": "incidence_angle_deg",
        "calibrated_domain_deg": [0.0, 15.0],
        "endpoint_cases": {"0_deg": "medium", "15_deg": "medium_incidence_15deg"},
        "holdout_angle_deg": HOLDOUT_ANGLE_DEG,
        "expected_injected_mass_kg": float(medium_metrics["mass_ledger"]["expected_injected_mass_kg"]),
        "prediction": {
            "deposited_mass_kg": prediction.deposited_mass_kg,
            "centroid_uv_m": list(prediction.centroid_uv_m),
            "covariance_uv_m2": prediction.covariance_uv_m2.tolist(),
            "transfer_efficiency": prediction.deposited_mass_kg / float(medium_metrics["mass_ledger"]["expected_injected_mass_kg"]),
        },
        "grid": {
            "source_case": "medium",
            "face_count": int(medium_map["u_m"].size),
            "u_extent_m": [float(np.min(medium_map["u_m"])), float(np.max(medium_map["u_m"]))],
            "v_extent_m": [float(np.min(medium_map["v_m"])), float(np.max(medium_map["v_m"]))],
            "cell_area_m2": _cell_area(medium_map),
        },
        "teacher_reconstruction_gate": {
            "single_gaussian_pass": True,
            "warning_bands": {"normalized_rmse_max": 0.25, "pearson_min": 0.90},
            "metrics": teacher_reconstruction,
        },
        "prediction_npz_sha256": _sha256_bytes(prediction_npz.read_bytes()),
        "prediction_hash_scope": "canonical JSON with prediction_file_sha256 blank",
        "prediction_file_sha256": "",
    }
    payload["prediction_file_sha256"] = _sha256_bytes(_canonical_json_without_hash(payload))
    prediction_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "teacher_reconstruction.json").write_text(
        json.dumps({"schema_version": "s2_teacher_reconstruction_v1", "model": "S2_SINGLE_GAUSSIAN", "metrics": teacher_reconstruction}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    render_teacher_reconstruction(repo, medium_map, teacher_reconstruction, endpoint_zero, endpoint_fifteen, incidence_metrics)
    return payload


def _load_frozen_prediction(repo: Path) -> tuple[dict, GaussianMoments, dict[str, np.ndarray]]:
    output_root = repo / S2_ROOT_REL
    prediction_json = output_root / "holdout_7p5_prediction.json"
    prediction_npz = output_root / "holdout_7p5_prediction.npz"
    payload = json.loads(prediction_json.read_text(encoding="utf-8"))
    expected_hash = payload["prediction_file_sha256"]
    actual_hash = _sha256_bytes(_canonical_json_without_hash(payload))
    if expected_hash != actual_hash:
        raise ValueError("frozen prediction JSON hash mismatch")
    if payload["prediction_npz_sha256"] != _sha256_bytes(prediction_npz.read_bytes()):
        raise ValueError("frozen prediction NPZ hash mismatch")
    prediction = payload["prediction"]
    moments = GaussianMoments(
        deposited_mass_kg=prediction["deposited_mass_kg"],
        centroid_uv_m=tuple(prediction["centroid_uv_m"]),
        covariance_uv_m2=np.asarray(prediction["covariance_uv_m2"], dtype=float),
    )
    with np.load(prediction_npz) as data:
        grid = {key: np.asarray(data[key], dtype=float) for key in data.files}
    return payload, moments, grid


def render_teacher_reconstruction(
    repo: Path,
    medium_map: dict[str, np.ndarray],
    teacher_reconstruction: dict,
    endpoint_zero: GaussianMoments,
    endpoint_fifteen: GaussianMoments,
    incidence_metrics: dict,
) -> None:
    media_root = repo / "media" / "air_assisted_spray"
    media_root.mkdir(parents=True, exist_ok=True)
    incidence_map = _load_map(repo / FLAT_RESULT_REL, "medium_incidence_15deg")
    uv = np.stack([medium_map["u_m"], medium_map["v_m"]], axis=-1)
    fields = [
        ("0° CFD", medium_map["areal_mass_kg_m2"], endpoint_zero.density(uv)),
        ("15° CFD", incidence_map["areal_mass_kg_m2"], endpoint_fifteen.density(uv)),
    ]
    vmax = max(float(np.max(actual)) for _, actual, _ in fields)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True, sharex=True, sharey=True)
    norm = Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    for row, (label, actual, reconstructed) in enumerate(fields):
        axes[row, 0].scatter(medium_map["u_m"], medium_map["v_m"], c=actual, s=22, marker="s", cmap="viridis", norm=norm)
        axes[row, 1].scatter(medium_map["u_m"], medium_map["v_m"], c=reconstructed, s=22, marker="s", cmap="viridis", norm=norm)
        axes[row, 0].set_title(f"{label} deposition")
        axes[row, 1].set_title(f"{label} S2 Gaussian")
    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(-0.15, 0.15)
        ax.set_xlabel("u [m]")
        ax.set_ylabel("v [m]")
    fig.colorbar(axes[0, 0].collections[0], ax=axes.ravel().tolist(), label="areal mass [kg m$^{-2}$]")
    fig.suptitle("S2 teacher-map reconstruction")
    fig.savefig(media_root / "s2_teacher_reconstruction.png", dpi=180)
    plt.close(fig)


def render_holdout_comparison(
    repo: Path,
    uv: np.ndarray,
    actual: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
) -> None:
    media_root = repo / "media" / "air_assisted_spray"
    vmax = max(float(np.max(actual)), float(np.max(s1)), float(np.max(s2)), float(np.max(np.abs(s2 - actual))))
    norm = Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    fields = [("OpenFOAM 7.5°", actual), ("S1", s1), ("S2", s2), ("absolute error", np.abs(s2 - actual))]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True, sharex=True, sharey=True)
    for ax, (label, field) in zip(axes, fields):
        scatter = ax.scatter(uv[:, 0], uv[:, 1], c=field, s=22, marker="s", cmap="viridis", norm=norm)
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(-0.15, 0.15)
        ax.set_xlabel("u [m]")
    axes[0].set_ylabel("v [m]")
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="areal mass / absolute error [kg m$^{-2}$]")
    fig.suptitle("7.5° hold-out: CFD, S1, S2, and S2 absolute error")
    fig.savefig(media_root / "s2_holdout_7p5_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True, sharex=True, sharey=True)
    for ax, label, field in zip(axes, ("S1", "S2"), (s1, s2)):
        ax.scatter(uv[:, 0], uv[:, 1], c=field, s=22, marker="s", cmap="viridis", norm=norm)
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(-0.15, 0.15)
        ax.set_xlabel("u [m]")
    axes[0].set_ylabel("v [m]")
    fig.colorbar(axes[1].collections[0], ax=axes.ravel().tolist(), label="areal mass [kg m$^{-2}$]")
    fig.suptitle("S1 vs S2 on the 7.5° CFD map")
    fig.savefig(media_root / "s1_vs_s2_holdout.png", dpi=180)
    plt.close(fig)


def validate_holdout(repo: Path) -> dict:
    payload, prediction, frozen_grid = _load_frozen_prediction(repo)
    flat_results = repo / FLAT_RESULT_REL
    case_dir = repo / "reference_cfd" / "openfoam_v2606" / "flat_plate" / "medium_incidence_7p5deg"
    holdout_metrics = process_case(case_dir, flat_results, "medium_incidence_7p5deg")
    holdout_map = _load_map(flat_results, "medium_incidence_7p5deg")
    uv = np.stack([holdout_map["u_m"], holdout_map["v_m"]], axis=-1)
    frozen_uv = np.stack([frozen_grid["u_m"], frozen_grid["v_m"]], axis=-1)
    if not np.allclose(uv, frozen_uv, rtol=0.0, atol=1.0e-12):
        raise ValueError("7.5-degree map grid differs from frozen medium prediction grid")
    s2_density = frozen_grid["predicted_areal_mass_kg_m2"]
    config_path = repo / "configs" / "air_assisted_spray.yaml"
    expected_mass = float(payload["expected_injected_mass_kg"])
    s1_kernel = build_s1_baseline(config_path, expected_mass)
    s1_density = s1_kernel.density(uv)
    area = _cell_area(holdout_map)
    s2_parameter = _parameter_metrics(prediction, holdout_metrics, expected_mass)
    s2_field = evaluate_field(s2_density, holdout_map["areal_mass_kg_m2"], uv, area, holdout_metrics)
    s1_field = evaluate_field(s1_density, holdout_map["areal_mass_kg_m2"], uv, area, holdout_metrics)
    parameter_pass = (
        s2_parameter["deposited_mass_relative_error"] <= 0.05
        and s2_parameter["centroid_distance_m"] <= 0.005
        and s2_parameter["sigma_major_relative_error"] <= 0.10
        and s2_parameter["sigma_minor_relative_error"] <= 0.10
    )
    field_pass = s2_field["pearson_correlation"] >= 0.90 and s2_field["normalized_rmse"] <= 0.25
    if parameter_pass and field_pass:
        status = "S2_HOLDOUT_VALIDATED"
    elif parameter_pass:
        status = "S2_MOMENTS_VALID_MAP_POOR"
    else:
        status = "S2_INTERPOLATION_NOT_VALIDATED"
    improvement = {
        "nrmse": s2_field["normalized_rmse"] < s1_field["normalized_rmse"],
        "centroid_error": s2_field["centroid_error_m"] < s1_field["centroid_error_m"],
        "mass_prediction": s2_field["integrated_mass_relative_error"] < s1_field["integrated_mass_relative_error"],
    }
    s2_improves = any(improvement.values())
    if not s2_improves:
        status = "S2_NO_VALUE_OVER_S1"
    result = {
        "schema_version": "s2_holdout_validation_v1",
        "status": status,
        "model": payload["model"],
        "frozen_prediction_file_sha256": payload["prediction_file_sha256"],
        "teacher_reconstruction": payload["teacher_reconstruction_gate"],
        "holdout_case": "medium_incidence_7p5deg",
        "holdout_metrics": holdout_metrics,
        "s2_parameter_validation": s2_parameter,
        "s2_field_validation": s2_field,
        "s1_field_validation": s1_field,
        "s1_vs_s2_improvement": {**improvement, "demonstrated": s2_improves},
        "acceptance": {
            "parameter_pass": parameter_pass,
            "field_pass": field_pass,
            "s2_improvement_pass": s2_improves,
        },
    }
    output_root = repo / S2_ROOT_REL
    (output_root / "holdout_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "s1_vs_s2_holdout.json").write_text(
        json.dumps({"schema_version": "s1_vs_s2_holdout_v1", "s1": s1_field, "s2": s2_field, "improvement": result["s1_vs_s2_improvement"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    render_holdout_comparison(repo, uv, holdout_map["areal_mass_kg_m2"], s1_density, s2_density)
    return result


def write_7p5_input_diff(repo: Path) -> dict:
    case_root = repo / "reference_cfd" / "openfoam_v2606" / "flat_plate"
    diff = compare_incidence_inputs(case_root / "medium", case_root / "medium_incidence_7p5deg", angle_deg=HOLDOUT_ANGLE_DEG)
    output_root = repo / S2_ROOT_REL
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "holdout7p5_input_diff.json").write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not diff["pass"]:
        raise RuntimeError("7.5-degree input-diff audit failed")
    return diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--input-diff-7p5", action="store_true")
    args = parser.parse_args()
    selected = sum(bool(value) for value in (args.freeze, args.validate, args.input_diff_7p5))
    if selected != 1:
        parser.error("choose exactly one of --freeze, --validate, or --input-diff-7p5")
    repo = args.repo.resolve()
    if args.freeze:
        print(json.dumps(freeze_prediction(repo), indent=2, sort_keys=True))
    elif args.validate:
        print(json.dumps(validate_holdout(repo), indent=2, sort_keys=True))
    else:
        print(json.dumps(write_7p5_input_diff(repo), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
