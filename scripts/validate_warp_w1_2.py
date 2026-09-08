"""Validate the compact W1.2 full-vector carrier against the blind 7.5° hold-out."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerospace_painting.s2_runtime import S2RuntimeModel
from aerospace_painting.warp_air_field import AirFieldParameters
from aerospace_painting.warp_spray import WarpCaseConfig, WarpCaseResult, run_warp_case
from aerospace_painting.warp_teacher_flow import TeacherFlowGrid, _sha256_file
from aerospace_painting.warp_vector_carrier import (
    COROTATING_VECTOR_INTERP,
    MODEL_ID,
    VectorCarrierError,
    VectorCarrierModel,
    load_model_artifacts,
    sample_vector_carrier_warp,
)
from scripts.build_warp_vector_carrier import MODEL_JSON, MODEL_NPZ, build_model
from scripts.validate_warp_spray import (
    CASE_DIR,
    CASE_LABELS,
    EXPECTED_MASS_KG,
    MEDIA_DIR,
    RESULT_DIR,
    WARP_RESULT_DIR,
    _field_comparison,
    _load_teacher_map,
    _map_moments,
    _pearson,
    _result_summary,
    _teacher_properties,
    _write_json,
)


W1_2_ROOT = ROOT / "results" / "air_assisted" / "warp_w1_2"
PREDICTION_JSON = W1_2_ROOT / "carrier_7p5_prediction.json"
PREDICTION_NPZ = W1_2_ROOT / "carrier_7p5_prediction.npz"
ANGLES = (0.0, 7.5, 15.0)
HIGH_PARTICLES_PER_BIN = 2000


def _load_air_parameters() -> AirFieldParameters:
    payload = json.loads((ROOT / "models" / "warp_air_field_v1.json").read_text(encoding="utf-8"))
    return AirFieldParameters.from_dict(payload["parameters"])


def _load_carrier_properties() -> tuple[float, float]:
    payload = json.loads((WARP_RESULT_DIR / "environment.json").read_text(encoding="utf-8"))
    carrier = payload["carrier_properties"]
    return float(carrier["ideal_gas_density_kg_m3"]), float(carrier["dynamic_viscosity_pa_s"])


def _grid_points(grid: TeacherFlowGrid) -> np.ndarray:
    first = np.asarray(grid.first_center_m, dtype=float)
    spacing = np.asarray(grid.spacing_m, dtype=float)
    x = first[0] + np.arange(grid.nx, dtype=float) * spacing[0]
    y = first[1] + np.arange(grid.ny, dtype=float) * spacing[1]
    z = first[2] + np.arange(grid.nz, dtype=float) * spacing[2]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.stack((xx, yy, zz), axis=-1)


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _freeze_prediction(model: VectorCarrierModel, model_payload: dict[str, Any]) -> dict[str, Any]:
    """Create once, then verify without rewriting, the blind 7.5° prediction."""

    W1_2_ROOT.mkdir(parents=True, exist_ok=True)
    if PREDICTION_JSON.exists() or PREDICTION_NPZ.exists():
        if not (PREDICTION_JSON.exists() and PREDICTION_NPZ.exists()):
            raise VectorCarrierError("frozen carrier prediction pair is incomplete")
        payload = json.loads(PREDICTION_JSON.read_text(encoding="utf-8"))
        if payload.get("model_sha256") != model_payload["model_sha256"]:
            raise VectorCarrierError("frozen carrier prediction was built from another model")
        if payload.get("prediction_sha256") != _sha256_file(PREDICTION_NPZ):
            raise VectorCarrierError("frozen carrier prediction file hash mismatch")
        with np.load(PREDICTION_NPZ, allow_pickle=False) as stored:
            stored_u = np.asarray(stored["U"], dtype=np.float32)
            stored_inside = np.asarray(stored["inside_mask"], dtype=bool)
        points = _grid_points(model.anchor_0)
        predicted_u, predicted_inside = model.sample(points, 7.5, interpolation=COROTATING_VECTOR_INTERP)
        predicted_u = predicted_u.reshape(stored_u.shape).astype(np.float32)
        predicted_inside = predicted_inside.reshape(stored_inside.shape)
        if not np.array_equal(stored_u, predicted_u) or not np.array_equal(stored_inside, predicted_inside):
            raise VectorCarrierError("frozen carrier prediction is not reproducible from the model")
        return payload

    points = _grid_points(model.anchor_0)
    predicted_u, predicted_inside = model.sample(points, 7.5, interpolation=COROTATING_VECTOR_INTERP)
    predicted_u = predicted_u.reshape(model.anchor_0.values_nz_ny_nx_3.shape).astype(np.float32)
    predicted_inside = predicted_inside.reshape(model.anchor_0.values_nz_ny_nx_3.shape[:3])
    np.savez_compressed(
        PREDICTION_NPZ,
        U=predicted_u,
        inside_mask=predicted_inside,
        first_center_m=np.asarray(model.first_center_m, dtype=np.float64),
        spacing_m=np.asarray(model.spacing_m, dtype=np.float64),
        dimensions_nx_ny_nz=np.asarray(model.dimensions_nx_ny_nz, dtype=np.int32),
        angle_deg=np.asarray([7.5], dtype=np.float64),
    )
    payload = {
        "schema_version": "warp_w1_2_frozen_carrier_prediction_v1",
        "model_id": MODEL_ID,
        "model_sha256": model_payload["model_sha256"],
        "angle_deg": 7.5,
        "interpolation_method": COROTATING_VECTOR_INTERP,
        "prediction_npz_path": PREDICTION_NPZ.name,
        "prediction_sha256": _sha256_file(PREDICTION_NPZ),
        "prediction_array_sha256": _array_hash(predicted_u),
        "inside_mask_array_sha256": _array_hash(predicted_inside.astype(np.uint8)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": {
            "dimensions_nx_ny_nz": list(model.dimensions_nx_ny_nz),
            "first_center_m": list(model.first_center_m),
            "spacing_m": list(model.spacing_m),
            "array_layout": "U[nz, ny, nx, 3]",
        },
        "holdout_policy": "frozen before reading the solved 7.5-degree OpenFOAM U field",
    }
    _write_json(PREDICTION_JSON, payload)
    return payload


def _direction_errors(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    pred = np.asarray(predicted, dtype=float)
    truth = np.asarray(actual, dtype=float)
    pred_norm = np.linalg.norm(pred, axis=-1)
    truth_norm = np.linalg.norm(truth, axis=-1)
    denominator = pred_norm * truth_norm
    angles = np.full(truth_norm.shape, 90.0, dtype=float)
    both_zero = (pred_norm <= 1.0e-12) & (truth_norm <= 1.0e-12)
    valid = denominator > 1.0e-12
    cosine = np.zeros_like(truth_norm)
    cosine[valid] = np.sum(pred[valid] * truth[valid], axis=-1) / denominator[valid]
    angles[valid] = np.degrees(np.arccos(np.clip(cosine[valid], -1.0, 1.0)))
    angles[both_zero] = 0.0
    return angles


def _region_metrics(predicted: np.ndarray, actual: np.ndarray, mask: np.ndarray, name: str, inside: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=float)[mask]
    truth = np.asarray(actual, dtype=float)[mask]
    if pred.size == 0:
        raise VectorCarrierError(f"carrier validation region is empty: {name}")
    error = pred - truth
    actual_norm = float(np.sqrt(np.mean(truth**2)))
    vector_rmse = float(np.sqrt(np.mean(error**2)))
    direction = _direction_errors(pred, truth)
    speed_pred = np.linalg.norm(pred, axis=-1)
    speed_truth = np.linalg.norm(truth, axis=-1)
    return {
        "region": name,
        "sample_count": int(len(truth)),
        "vector_rmse_m_s": vector_rmse,
        "normalized_vector_rmse": vector_rmse / max(actual_norm, 1.0e-12),
        "speed_rmse_m_s": float(np.sqrt(np.mean((speed_pred - speed_truth) ** 2))),
        "Ux_rmse_m_s": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "Uy_rmse_m_s": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "Uz_rmse_m_s": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "vector_component_correlation": _pearson(pred.reshape(-1), truth.reshape(-1)),
        "speed_correlation": _pearson(speed_pred, speed_truth),
        "mean_direction_error_deg": float(np.mean(direction)),
        "median_direction_error_deg": float(np.median(direction)),
        "p95_direction_error_deg": float(np.percentile(direction, 95.0)),
        "predicted_out_of_field_count": int(np.count_nonzero(~np.asarray(inside, dtype=bool)[mask])),
    }


def _carrier_holdout_metrics(
    model: VectorCarrierModel,
    prediction_payload: dict[str, Any],
    actual_grid: TeacherFlowGrid,
) -> dict[str, Any]:
    with np.load(PREDICTION_NPZ, allow_pickle=False) as prediction:
        predicted = np.asarray(prediction["U"], dtype=np.float32)
        inside = np.asarray(prediction["inside_mask"], dtype=bool)
    actual = np.asarray(actual_grid.values_nz_ny_nx_3, dtype=np.float32)
    if predicted.shape != actual.shape or inside.shape != actual.shape[:3]:
        raise VectorCarrierError("frozen prediction and 7.5-degree teacher grid shapes differ")
    actual_points = _grid_points(actual_grid)
    speed = np.linalg.norm(actual, axis=-1)
    bounds = actual_grid.bounds_max_m
    regions = {
        "whole_flow_domain": np.ones(speed.shape, dtype=bool),
        "high_speed_jet": speed >= 0.5 * float(speed.max()),
        "near_target_region": actual_points[..., 2] >= float(bounds[2] - 0.06),
    }
    report = {
        "schema_version": "warp_w1_2_carrier_holdout_v1",
        "model_id": MODEL_ID,
        "model_sha256": prediction_payload["model_sha256"],
        "prediction_sha256": prediction_payload["prediction_sha256"],
        "angle_deg": 7.5,
        "actual_openfoam_version": actual_grid.openfoam_version,
        "prediction_inside_fraction": float(np.mean(inside)),
        "prediction_out_of_field_count": int(np.count_nonzero(~inside)),
        "region_definitions": {
            "high_speed_jet": "actual OpenFOAM speed >= 0.5 * whole-domain maximum speed",
            "near_target_region": "cell-center z >= finite-volume upper z bound - 0.06 m",
        },
        "regions": {},
    }
    for name, region in regions.items():
        report["regions"][name] = _region_metrics(predicted, actual, region, name, inside)
    whole = report["regions"]["whole_flow_domain"]
    report["carrier_gate"] = {
        "normalized_vector_rmse_max": 0.20,
        "vector_correlation_min": 0.90,
        "median_direction_error_deg_max": 5.0,
        "normalized_vector_rmse": whole["normalized_vector_rmse"],
        "vector_correlation": whole["vector_component_correlation"],
        "median_direction_error_deg": whole["median_direction_error_deg"],
        "pass": bool(
            whole["normalized_vector_rmse"] <= 0.20
            and whole["vector_component_correlation"] >= 0.90
            and whole["median_direction_error_deg"] <= 5.0
        ),
    }
    return report


def _sampler_agreement(model: VectorCarrierModel, *, device: str) -> dict[str, Any]:
    origin = np.asarray(model.nozzle_origin_m, dtype=float)
    points = np.asarray(
        [
            origin + (0.0, 0.0, 0.03),
            origin + (0.02, 0.0, 0.06),
            origin + (-0.03, 0.01, 0.09),
            origin + (0.0, -0.02, 0.12),
            (0.0, 0.0, 0.30),
        ],
        dtype=float,
    )
    report: dict[str, Any] = {}
    for angle in ANGLES:
        cpu_values, cpu_inside = model.sample(points, angle, interpolation=COROTATING_VECTOR_INTERP)
        gpu_values, gpu_inside = sample_vector_carrier_warp(model, points, angle, device=device, interpolation=COROTATING_VECTOR_INTERP)
        error = float(np.max(np.abs(cpu_values[cpu_inside] - gpu_values[cpu_inside]))) if np.any(cpu_inside) else 0.0
        report[f"{angle:g}_deg"] = {
            "sample_count": int(len(points)),
            "inside_mask_match": bool(np.array_equal(cpu_inside, gpu_inside)),
            "max_abs_velocity_error_m_s": error,
            "comparison_tolerance_m_s": 2.0e-5,
            "pass": bool(np.array_equal(cpu_inside, gpu_inside) and error <= 2.0e-5),
        }
    return report


def _make_config(angle: float, air: AirFieldParameters, rho: float, mu: float) -> WarpCaseConfig:
    return WarpCaseConfig(
        incidence_angle_deg=angle,
        particle_count_per_bin=HIGH_PARTICLES_PER_BIN,
        air_field=air,
        air_density_kg_m3=rho,
        air_dynamic_viscosity_pa_s=mu,
    )


def _metric_row(summary: dict[str, Any]) -> dict[str, Any]:
    deposition = summary["deposition"]
    ledger = summary["mass_ledger"]
    field = summary["comparison"]["openfoam_teacher"]
    return {
        "transfer_efficiency": ledger["transfer_efficiency"],
        "deposited_mass_kg": deposition["mass_kg"],
        "escaped_mass_kg": ledger["escaped_mass_kg"],
        "centroid_u_m": deposition["centroid_uv_m"][0],
        "centroid_v_m": deposition["centroid_uv_m"][1],
        "sigma_major_m": deposition["sigma_major_m"],
        "sigma_minor_m": deposition["sigma_minor_m"],
        "peak_density_kg_m2": deposition["peak_density_kg_m2"],
        "nrmse": field["nrmse"],
        "pearson_correlation": field["pearson_correlation"],
    }


def _s2_row(s2_density: np.ndarray, teacher_map: dict[str, np.ndarray], teacher_metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    uv = np.column_stack((teacher_map["u_m"], teacher_map["v_m"]))
    area = teacher_map["area_m2"]
    moments = _map_moments(uv, s2_density, area)
    teacher_dep = teacher_metrics["deposition"]
    ledger = teacher_metrics["mass_ledger"]
    comparison = _field_comparison(s2_density, teacher_map["areal_mass_kg_m2"], uv, area)
    row = {
        "transfer_efficiency": moments["mass_kg"] / EXPECTED_MASS_KG,
        "deposited_mass_kg": moments["mass_kg"],
        "escaped_mass_kg": None,
        "centroid_u_m": moments["centroid_uv_m"][0],
        "centroid_v_m": moments["centroid_uv_m"][1],
        "sigma_major_m": moments["sigma_major_m"],
        "sigma_minor_m": moments["sigma_minor_m"],
        "peak_density_kg_m2": moments["peak_density_kg_m2"],
        "nrmse": comparison["nrmse"],
        "pearson_correlation": comparison["pearson_correlation"],
        "teacher_reference_deposited_mass_kg": float(ledger["deposited_kg"]),
        "teacher_reference_centroid_u_m": float(teacher_dep["centroid_u_m"]),
    }
    return row, comparison


def _gate(summary: dict[str, Any]) -> dict[str, bool]:
    deposition = summary["deposition"]
    field = summary["comparison"]["openfoam_teacher"]
    ledger = summary["mass_ledger"]
    return {
        "mass_balance": abs(float(ledger["relative_balance_error"])) <= 1.0e-5,
        "deposited_mass": float(deposition["deposited_mass_relative_error"]) <= 0.10,
        "centroid": float(deposition["centroid_error_m"]) <= 0.010,
        "sigma_major": float(deposition["sigma_major_relative_error"]) <= 0.15,
        "sigma_minor": float(deposition["sigma_minor_relative_error"]) <= 0.15,
        "correlation": float(field["pearson_correlation"]) >= 0.80,
        "nrmse": float(field["nrmse"]) <= 0.40,
    }


def _endpoint_regression(vector: dict[str, Any], teacher: dict[str, Any]) -> dict[str, Any]:
    return {
        "deposited_mass_delta_kg": vector["deposition"]["mass_kg"] - teacher["deposition"]["mass_kg"],
        "transfer_efficiency_delta": vector["mass_ledger"]["transfer_efficiency"] - teacher["mass_ledger"]["transfer_efficiency"],
        "centroid_u_delta_m": vector["deposition"]["centroid_uv_m"][0] - teacher["deposition"]["centroid_uv_m"][0],
        "centroid_v_delta_m": vector["deposition"]["centroid_uv_m"][1] - teacher["deposition"]["centroid_uv_m"][1],
        "sigma_major_delta_m": vector["deposition"]["sigma_major_m"] - teacher["deposition"]["sigma_major_m"],
        "sigma_minor_delta_m": vector["deposition"]["sigma_minor_m"] - teacher["deposition"]["sigma_minor_m"],
        "escaped_mass_delta_kg": vector["mass_ledger"]["escaped_mass_kg"] - teacher["mass_ledger"]["escaped_mass_kg"],
    }


def _render_carrier_holdout(actual: np.ndarray, predicted: np.ndarray, inside: np.ndarray, grid: TeacherFlowGrid) -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    speed_actual = np.linalg.norm(actual, axis=-1)
    speed_pred = np.linalg.norm(predicted, axis=-1)
    error = np.linalg.norm(predicted - actual, axis=-1)
    z_index = min(grid.nz - 1, max(0, int(round((0.12 - grid.first_center_m[2]) / grid.spacing_m[2]))))
    fields = (
        ("OpenFOAM |U|", speed_actual[z_index]),
        ("W1.2 predicted |U|", speed_pred[z_index]),
        ("vector error", error[z_index]),
        ("predicted inside", inside[z_index].astype(float)),
    )
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), constrained_layout=True)
    extent = [grid.bounds_min_m[0], grid.bounds_max_m[0], grid.bounds_min_m[1], grid.bounds_max_m[1]]
    for axis, (label, values) in zip(axes, fields):
        image = axis.imshow(values, origin="lower", extent=extent, aspect="equal", cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle("W1.2 7.5° carrier hold-out at z≈0.12 m")
    fig.savefig(MEDIA_DIR / "warp_w1_2_carrier_holdout.png", dpi=180)
    plt.close(fig)


def _render_deposition_comparison(rows: dict[str, dict[str, Any]], teacher_density: np.ndarray, s2_density: np.ndarray, uv: np.ndarray, vector_density: np.ndarray, teacher_u_density: np.ndarray, analytic_density: np.ndarray) -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    fields = (
        ("OpenFOAM teacher", teacher_density),
        ("S2 surrogate", s2_density),
        ("Warp analytic W1", analytic_density),
        ("Warp vector W1.2", vector_density),
        ("Warp teacher-U W1.1", teacher_u_density),
    )
    vmax = max(float(np.max(values)) for _, values in fields)
    fig, axes = plt.subplots(1, len(fields), figsize=(19, 4.0), constrained_layout=True, sharex=True, sharey=True)
    for axis, (label, values) in zip(axes, fields):
        image = axis.scatter(uv[:, 0], uv[:, 1], c=values, s=22, marker="s", cmap="viridis", vmin=0.0, vmax=vmax)
        axis.set_title(label)
        axis.set_aspect("equal")
        axis.set_xlim(-0.15, 0.15)
        axis.set_ylim(-0.15, 0.15)
        axis.set_xlabel("u [m]")
    axes[0].set_ylabel("v [m]")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="areal mass [kg m$^{-2}$]")
    fig.suptitle("W1.2 7.5° deposition fidelity hierarchy")
    fig.savefig(MEDIA_DIR / "warp_w1_2_openfoam_7p5_comparison.png", dpi=180)
    plt.close(fig)

    metric_names = ("transfer_efficiency", "centroid_u_m", "sigma_major_m")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    labels = tuple(rows.keys())
    colors = ("#2166ac", "#5e3c99", "#e66101", "#1b7837", "#762a83")
    for axis, metric in zip(axes, metric_names):
        values = [rows[label][metric] for label in labels]
        axis.bar(labels, values, color=colors)
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("W1.2 fidelity hierarchy metrics")
    fig.savefig(MEDIA_DIR / "warp_w1_2_fidelity_hierarchy.png", dpi=180)
    plt.close(fig)


def _render_angle_response(angle_rows: dict[float, dict[str, dict[str, Any]]]) -> None:
    angles = np.asarray(ANGLES, dtype=float)
    series = (
        ("OpenFOAM", "openfoam", "#2166ac"),
        ("Analytic W1", "analytic_w1", "#e66101"),
        ("Vector W1.2", "vector_w1_2", "#1b7837"),
        ("Teacher-U W1.1", "teacher_u_w1_1", "#5e3c99"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    for axis, metric, ylabel in zip(axes, ("centroid_u_m", "sigma_major_m", "transfer_efficiency"), ("centroid U [m]", "sigma major [m]", "transfer efficiency")):
        for label, key, color in series:
            axis.plot(angles, [angle_rows[float(angle)][key][metric] for angle in ANGLES], "o-", color=color, label=label)
        axis.set_xlabel("incidence [deg]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("W1.2 static angle response")
    fig.savefig(MEDIA_DIR / "warp_w1_2_angle_response.png", dpi=180)
    plt.close(fig)


def run_validation(*, device: str = "cuda:0") -> dict[str, Any]:
    started = time.perf_counter()
    W1_2_ROOT.mkdir(parents=True, exist_ok=True)
    # Anchor construction and prediction freeze intentionally happen before
    # the 7.5° carrier field is opened below.
    model, model_payload = build_model(force=False)
    prediction_payload = _freeze_prediction(model, model_payload)
    sampler_agreement = _sampler_agreement(model, device=device)

    holdout_grid = TeacherFlowGrid.from_case_dir(CASE_DIR / CASE_LABELS[7.5], 7.5, repo_root=ROOT)
    carrier_validation = _carrier_holdout_metrics(model, prediction_payload, holdout_grid)
    _write_json(W1_2_ROOT / "carrier_holdout_validation.json", carrier_validation)

    air_parameters = _load_air_parameters()
    rho, mu = _load_carrier_properties()
    grids = {
        angle: TeacherFlowGrid.from_case_dir(CASE_DIR / CASE_LABELS[angle], angle, repo_root=ROOT)
        for angle in ANGLES
    }
    teacher_maps: dict[float, dict[str, np.ndarray]] = {}
    teacher_metrics: dict[float, dict[str, Any]] = {}
    for angle in ANGLES:
        teacher_maps[angle], teacher_metrics[angle] = _load_teacher_map(angle)
    uv = np.column_stack((teacher_maps[7.5]["u_m"], teacher_maps[7.5]["v_m"]))
    s2_model = S2RuntimeModel.load(ROOT / "models" / "s2_air_assisted_v1.json")
    s2_model.verify_sources(ROOT)
    s2_density = s2_model.predict(7.5).density(uv)

    results: dict[str, dict[float, WarpCaseResult]] = {"analytic_w1": {}, "teacher_u_w1_1": {}, "vector_w1_2": {}}
    for angle in ANGLES:
        config = _make_config(angle, air_parameters, rho, mu)
        print(f"WARP_W1_2_RUN mode=ANALYTIC_AIR angle={angle:g} particles={config.particle_count}", flush=True)
        results["analytic_w1"][angle] = run_warp_case(config, device=device, carrier_mode="ANALYTIC_AIR")
        print(f"WARP_W1_2_RUN mode=TEACHER_FORCED_U angle={angle:g} particles={config.particle_count}", flush=True)
        results["teacher_u_w1_1"][angle] = run_warp_case(config, device=device, carrier_mode="TEACHER_FORCED_U", teacher_flow=grids[angle])
        print(f"WARP_W1_2_RUN mode=COROTATING_VECTOR_INTERP angle={angle:g} particles={config.particle_count}", flush=True)
        results["vector_w1_2"][angle] = run_warp_case(config, device=device, carrier_mode=COROTATING_VECTOR_INTERP, vector_carrier=model)

    summaries: dict[str, dict[float, dict[str, Any]]] = {key: {} for key in results}
    for key, cases in results.items():
        for angle in ANGLES:
            s2 = s2_density if angle == 7.5 else None
            summaries[key][angle] = _result_summary(angle, cases[angle], teacher_maps[angle], teacher_metrics[angle], s2)

    for angle in ANGLES:
        label = "0deg" if angle == 0.0 else "7p5deg" if angle == 7.5 else "15deg"
        payload = {
            "schema_version": "warp_w1_2_case_metrics_v1",
            "incidence_angle_deg": angle,
            "carrier_modes": {key: summaries[key][angle] for key in summaries},
            "endpoint_or_holdout": "7.5-degree blind deposition comparison" if angle == 7.5 else "anchor endpoint regression",
            "endpoint_regression_vs_teacher_u": _endpoint_regression(summaries["vector_w1_2"][angle], summaries["teacher_u_w1_1"][angle]) if angle in (0.0, 15.0) else None,
        }
        _write_json(W1_2_ROOT / f"case_{label}_metrics.json", payload)

    vector_holdout = summaries["vector_w1_2"][7.5]
    vector_gate = _gate(vector_holdout)
    carrier_gate = carrier_validation["carrier_gate"]
    deposition_pass = bool(all(vector_gate.values()))
    carrier_pass = bool(carrier_gate["pass"])
    if carrier_pass and deposition_pass:
        status = "WARP_W1_VECTOR_CARRIER_VALIDATED"
        result = "PASS"
        overall = "WARP_W1_VALIDATED"
    elif carrier_pass:
        status = "WARP_W1_VECTOR_FIELD_VALID_TRANSPORT_REGRESSION"
        result = "PARTIAL"
        overall = "WARP_W1_VECTOR_FIELD_VALID_TRANSPORT_REGRESSION"
    else:
        status = "WARP_VECTOR_CARRIER_NOT_VALIDATED"
        result = "PARTIAL"
        overall = "WARP_VECTOR_CARRIER_NOT_VALIDATED"

    s2_row, s2_comparison = _s2_row(s2_density, teacher_maps[7.5], teacher_metrics[7.5])
    hierarchy = {
        "OpenFOAM": {
            "transfer_efficiency": float(teacher_metrics[7.5]["mass_ledger"]["transfer_efficiency"]),
            "deposited_mass_kg": float(teacher_metrics[7.5]["mass_ledger"]["deposited_kg"]),
            "escaped_mass_kg": float(teacher_metrics[7.5]["mass_ledger"]["escaped_kg"]),
            "centroid_u_m": float(teacher_metrics[7.5]["deposition"]["centroid_u_m"]),
            "centroid_v_m": float(teacher_metrics[7.5]["deposition"]["centroid_v_m"]),
            "sigma_major_m": float(teacher_metrics[7.5]["deposition"]["sigma_major_m"]),
            "sigma_minor_m": float(teacher_metrics[7.5]["deposition"]["sigma_minor_m"]),
            "peak_density_kg_m2": float(teacher_metrics[7.5]["deposition"]["peak_areal_mass_kg_m2"]),
            "nrmse": 0.0,
            "pearson_correlation": 1.0,
        },
        "S2": s2_row,
        "Analytic W1": _metric_row(summaries["analytic_w1"][7.5]),
        "Vector W1.2": _metric_row(vector_holdout),
        "Teacher-U W1.1": _metric_row(summaries["teacher_u_w1_1"][7.5]),
    }

    openfoam_15 = teacher_metrics[15.0]["mass_ledger"]
    vector_15 = summaries["vector_w1_2"][15.0]["mass_ledger"]
    response_15 = {
        "openfoam_transfer_efficiency": float(openfoam_15["transfer_efficiency"]),
        "openfoam_deposited_mass_kg": float(openfoam_15["deposited_kg"]),
        "openfoam_escaped_mass_kg": float(openfoam_15["escaped_kg"]),
        "w1_2_transfer_efficiency": float(vector_15["transfer_efficiency"]),
        "w1_2_deposited_mass_kg": float(vector_15["deposited_mass_kg"]),
        "w1_2_escaped_mass_kg": float(vector_15["escaped_mass_kg"]),
        "transfer_efficiency_error": float(vector_15["transfer_efficiency"] - openfoam_15["transfer_efficiency"]),
        "centroid_u_error_m": float(summaries["vector_w1_2"][15.0]["deposition"]["centroid_uv_m"][0] - teacher_metrics[15.0]["deposition"]["centroid_u_m"]),
    }

    performance = {}
    for key in results:
        perf = summaries[key][7.5]["performance"]
        performance[key] = {
            "carrier_mode": perf["carrier_mode"],
            "device": perf["device"],
            "particle_count": perf["particle_count"],
            "substeps": perf["substeps"],
            "model_upload_seconds": perf.get("carrier_field_upload_seconds", perf.get("teacher_field_upload_seconds", 0.0)),
            "mean_step_compute_seconds": perf["mean_step_compute_seconds"],
            "p95_step_compute_seconds": perf["p95_step_compute_seconds"],
            "wall_seconds": perf["wall_seconds"],
        }
    performance["openfoam_reference"] = {"status": "NOT_MEASURED_IN_EXISTING_STATIC_ARTIFACTS"}

    angle_rows: dict[float, dict[str, dict[str, Any]]] = {}
    for angle in ANGLES:
        angle_rows[angle] = {
            "openfoam": hierarchy["OpenFOAM"] if angle == 7.5 else {
                "centroid_u_m": float(teacher_metrics[angle]["deposition"]["centroid_u_m"]),
                "sigma_major_m": float(teacher_metrics[angle]["deposition"]["sigma_major_m"]),
                "transfer_efficiency": float(teacher_metrics[angle]["mass_ledger"]["transfer_efficiency"]),
            },
            "analytic_w1": _metric_row(summaries["analytic_w1"][angle]),
            "vector_w1_2": _metric_row(summaries["vector_w1_2"][angle]),
            "teacher_u_w1_1": _metric_row(summaries["teacher_u_w1_1"][angle]),
        }

    validation_summary = {
        "schema_version": "warp_w1_2_validation_summary_v1",
        "result": result,
        "status": status,
        "overall_engineering_state": overall,
        "primary_interpretation": "two-anchor co-rotating full-vector CFD surrogate evaluated on a blind 7.5-degree carrier and deposition hold-out",
        "model": {
            "model_id": MODEL_ID,
            "interpolation": COROTATING_VECTOR_INTERP,
            "endpoint_fields": {"0_deg": "solved OpenFOAM U", "15_deg": "solved OpenFOAM U"},
            "calibrated_angle_domain_deg": [0.0, 15.0],
            "artifact_json": str(MODEL_JSON.relative_to(ROOT)),
            "artifact_npz": str(MODEL_NPZ.relative_to(ROOT)),
            "model_sha256": model_payload["model_sha256"],
            "model_json_sha256": model_payload["model_json_sha256"],
            "npz_sha256": model_payload["npz_sha256"],
            "source_commit": model_payload["source_commit"],
        },
        "frozen_prediction": prediction_payload,
        "carrier_holdout": carrier_validation,
        "deposition_gates_7p5": vector_gate,
        "deposition_gate_pass": deposition_pass,
        "carrier_gate_pass": carrier_pass,
        "seven5_fidelity_hierarchy": hierarchy,
        "fifteen_degree_response": response_15,
        "endpoint_regression_vs_teacher_u": {
            "0_deg": _endpoint_regression(summaries["vector_w1_2"][0.0], summaries["teacher_u_w1_1"][0.0]),
            "15_deg": _endpoint_regression(summaries["vector_w1_2"][15.0], summaries["teacher_u_w1_1"][15.0]),
        },
        "angle_response": angle_rows,
        "performance": performance,
        "verification": {
            "warp_device": performance["vector_w1_2"]["device"],
            "particle_count_per_bin": HIGH_PARTICLES_PER_BIN,
            "total_particles": HIGH_PARTICLES_PER_BIN * 5,
            "particle_dt_s": 5.0e-5,
            "substeps": 1200,
            "prediction_reproducibility": True,
            "source_hashes_verified": True,
            "cpu_vs_warp_sampler": sampler_agreement,
        },
        "fidelity_boundary": {
            "description": "W1.2 uses interpolated full-vector CFD anchor fields; it is not CFD solved online.",
            "not_modeled": ["primary atomization", "breakup", "evaporation", "splash", "wall film", "curing", "validated film thickness"],
        },
        "artifacts": {
            "model_json": str(MODEL_JSON.relative_to(ROOT)),
            "model_npz": str(MODEL_NPZ.relative_to(ROOT)),
            "prediction_json": str(PREDICTION_JSON.relative_to(ROOT)),
            "prediction_npz": str(PREDICTION_NPZ.relative_to(ROOT)),
            "case_metrics": [f"results/air_assisted/warp_w1_2/case_{label}_metrics.json" for label in ("0deg", "7p5deg", "15deg")],
            "plots": [
                "media/air_assisted_spray/warp_w1_2_carrier_holdout.png",
                "media/air_assisted_spray/warp_w1_2_openfoam_7p5_comparison.png",
                "media/air_assisted_spray/warp_w1_2_fidelity_hierarchy.png",
                "media/air_assisted_spray/warp_w1_2_angle_response.png",
            ],
        },
    }
    _write_json(W1_2_ROOT / "validation_summary.json", validation_summary)

    with np.load(PREDICTION_NPZ, allow_pickle=False) as prediction:
        predicted = np.asarray(prediction["U"], dtype=np.float32)
        inside = np.asarray(prediction["inside_mask"], dtype=bool)
    _render_carrier_holdout(actual=holdout_grid.values_nz_ny_nx_3, predicted=predicted, inside=inside, grid=holdout_grid)
    _render_deposition_comparison(
        hierarchy,
        teacher_maps[7.5]["areal_mass_kg_m2"],
        s2_density,
        uv,
        results["vector_w1_2"][7.5].map_density_kg_m2,
        results["teacher_u_w1_1"][7.5].map_density_kg_m2,
        results["analytic_w1"][7.5].map_density_kg_m2,
    )
    _render_angle_response(angle_rows)
    print(json.dumps({"result": result, "status": status, "carrier_gate": carrier_gate, "deposition_gate": vector_gate}, indent=2, sort_keys=True))
    print(f"WARP_W1_2_TOTAL_WALL_SECONDS {time.perf_counter() - started:.3f}")
    return validation_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = run_validation(device=args.device)
    if summary["result"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
