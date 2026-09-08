"""Run the W1.3 fixed-grid vector-carrier audit and 10-degree hold-out."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
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

from aerospace_painting.warp_air_field import AirFieldParameters, frame_axes
from aerospace_painting.warp_spray import WarpCaseConfig, WarpCaseResult, run_warp_case
from aerospace_painting.warp_teacher_flow import TeacherFlowGrid, _sha256_file
from aerospace_painting.warp_vector_carrier import (
    FIXED_GRID_LOCAL_VECTOR_INTERP,
    MODEL_ID,
    COROTATING_VECTOR_INTERP,
    VectorCarrierError,
    VectorCarrierModel,
    load_model_artifacts,
)
from scripts.build_warp_vector_carrier import build_model_variant
from scripts.validate_openfoam_incidence_response import (
    _air_inlet_vector,
    _spray_directions,
    compare_incidence_inputs,
)
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
from scripts.validate_warp_w1_2 import (
    _carrier_holdout_metrics,
    _grid_points,
    _load_air_parameters,
    _load_carrier_properties,
    _make_config,
    _metric_row,
    _region_metrics,
    _s2_row,
)


W1_3_ROOT = ROOT / "results" / "air_assisted" / "warp_w1_3"
MODEL_JSON = ROOT / "models" / "warp_vector_carrier_v3.json"
MODEL_NPZ = ROOT / "models" / "warp_vector_carrier_v3.npz"
CASE_10 = CASE_DIR / "medium_incidence_10deg"
RESULT_10 = RESULT_DIR / "medium_incidence_10deg"
PREDICTED_U_NPZ = W1_3_ROOT / "holdout_10deg_carrier_prediction.npz"
PREDICTED_U_JSON = W1_3_ROOT / "holdout_10deg_carrier_prediction.json"
PREDICTED_DEP_NPZ = W1_3_ROOT / "holdout_10deg_deposition_prediction.npz"
PREDICTED_DEP_JSON = W1_3_ROOT / "holdout_10deg_deposition_prediction.json"
ANGLES = (0.0, 7.5, 10.0, 15.0)
HIGH_PARTICLES_PER_BIN = 2000


def _json_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _load_grids() -> dict[float, TeacherFlowGrid]:
    return {
        angle: TeacherFlowGrid.from_case_dir(CASE_DIR / CASE_LABELS[angle], angle, repo_root=ROOT)
        for angle in (0.0, 7.5, 15.0)
    }


def _wall_signature(case_dir: Path) -> dict[str, Any]:
    try:
        from scripts.postprocess_openfoam_flat_plate import read_faces, read_patch_range, read_points
    except ModuleNotFoundError:
        from postprocess_openfoam_flat_plate import read_faces, read_patch_range, read_points
    poly = case_dir / "constant" / "polyMesh"
    points = read_points(poly / "points")
    faces = read_faces(poly / "faces")
    start, count = read_patch_range(poly / "boundary", "targetWall")
    wall_faces = faces[start : start + count]
    wall_points = points[np.unique(np.concatenate(wall_faces))]
    centres = np.asarray([points[face].mean(axis=0) for face in wall_faces], dtype=np.float64)
    areas = np.asarray(
        [0.5 * np.linalg.norm(np.sum(np.cross(points[face], np.roll(points[face], -1, axis=0)), axis=0)) for face in wall_faces],
        dtype=np.float64,
    )
    geometry_hash_input = np.concatenate(
        (
            np.asarray(wall_points, dtype=np.float64).ravel(),
            np.asarray(centres, dtype=np.float64).ravel(),
            np.asarray(areas, dtype=np.float64).ravel(),
        )
    )
    return {
        "boundary_sha256": _sha256_file(poly / "boundary"),
        "target_wall_face_count": int(count),
        "target_wall_vertex_count": int(len(wall_points)),
        "target_wall_geometry_sha256": _json_hash(np.round(geometry_hash_input, 12)),
        "target_wall_bounds_min_m": np.min(wall_points, axis=0).tolist(),
        "target_wall_bounds_max_m": np.max(wall_points, axis=0).tolist(),
    }


def _input_signature(case_dir: Path) -> dict[str, Any]:
    inlet = _air_inlet_vector((case_dir / "0" / "U").read_text(encoding="utf-8", errors="replace"))
    spray_text = (case_dir / "constant" / "sprayCloudProperties").read_text(encoding="utf-8", errors="replace")
    directions = _spray_directions(spray_text)
    positions = re.findall(r"\bposition\s*\(([^)]*)\)", spray_text)
    position_values = [tuple(float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", raw)) for raw in positions]
    return {
        "air_inlet_vector_m_s": list(inlet),
        "spray_direction_count": len(directions),
        "spray_directions": [list(direction) for direction in directions],
        "unique_spray_directions": sorted({tuple(round(value, 9) for value in direction) for direction in directions}),
        "injection_positions_m": [list(position) for position in position_values],
        "source_nozzle_positions_identical": len(set(position_values)) == 1,
    }


def build_domain_audit(grids: dict[float, TeacherFlowGrid]) -> dict[str, Any]:
    labels = {0.0: "medium", 7.5: "medium_incidence_7p5deg", 15.0: "medium_incidence_15deg"}
    block_hashes = {str(angle): _sha256_file(CASE_DIR / label / "system" / "blockMeshDict") for angle, label in labels.items()}
    c_hashes = {str(angle): grid.source_c_sha256 for angle, grid in grids.items()}
    coordinate_hashes = {str(angle): _json_hash(np.round(_grid_points(grid), 12)) for angle, grid in grids.items()}
    wall = {str(angle): _wall_signature(CASE_DIR / label) for angle, label in labels.items()}
    signatures = {str(angle): _input_signature(CASE_DIR / label) for angle, label in labels.items()}
    input_diffs = {
        str(angle): compare_incidence_inputs(CASE_DIR / "medium", CASE_DIR / label, angle_deg=angle)
        for angle, label in ((7.5, labels[7.5]), (15.0, labels[15.0]))
    }
    dimensions = {str(angle): list(grid.dimensions_nx_ny_nz) for angle, grid in grids.items()}
    bounds = {
        str(angle): {
            "min_m": grid.bounds_min_m.tolist(),
            "max_m": grid.bounds_max_m.tolist(),
        }
        for angle, grid in grids.items()
    }
    fixed_domain = (
        len(set(block_hashes.values())) == 1
        and len(set(c_hashes.values())) == 1
        and len(set(coordinate_hashes.values())) == 1
        and len({tuple(value) for value in dimensions.values()}) == 1
        and len({json.dumps(value, sort_keys=True) for value in bounds.values()}) == 1
        and len({wall[str(angle)]["target_wall_geometry_sha256"] for angle in labels}) == 1
        and all(input_diffs[str(angle)]["pass"] for angle in (7.5, 15.0))
        and all(signature["source_nozzle_positions_identical"] for signature in signatures.values())
    )
    return {
        "schema_version": "warp_w1_3_domain_audit_v1",
        "cases": {str(angle): labels[angle] for angle in labels},
        "blockMeshDict_sha256": block_hashes,
        "C_coordinate_sha256": c_hashes,
        "canonical_coordinate_array_sha256": coordinate_hashes,
        "grid_dimensions_nx_ny_nz": dimensions,
        "finite_volume_bounds_m": bounds,
        "target_wall": wall,
        "source_and_nozzle_inputs": signatures,
        "intended_input_diffs": input_diffs,
        "conclusion": "FIXED_EULERIAN_DOMAIN = true" if fixed_domain else "FIXED_EULERIAN_DOMAIN = false",
        "fixed_eulerian_domain": fixed_domain,
    }


def _w12_mapping(points: np.ndarray, model: VectorCarrierModel) -> dict[str, np.ndarray]:
    origin = np.asarray(model.nozzle_origin_m, dtype=float)
    xq, yq, zq = frame_axes(7.5)
    relative = points - origin
    q = np.stack((relative @ xq, relative @ yq, relative @ zq), axis=-1)
    x0, y0, z0 = frame_axes(0.0)
    x15, y15, z15 = frame_axes(15.0)
    p0 = origin + q[:, 0, None] * x0 + q[:, 1, None] * y0 + q[:, 2, None] * z0
    p15 = origin + q[:, 0, None] * x15 + q[:, 1, None] * y15 + q[:, 2, None] * z15
    lo, hi = model.bounds_min_m, model.bounds_max_m
    inside0 = np.all((p0 >= lo) & (p0 <= hi), axis=1)
    inside15 = np.all((p15 >= lo) & (p15 <= hi), axis=1)
    outside0 = ~inside0
    outside15 = ~inside15
    outside_x = (p0[:, 0] < lo[0]) | (p0[:, 0] > hi[0]) | (p15[:, 0] < lo[0]) | (p15[:, 0] > hi[0])
    outside_y = (p0[:, 1] < lo[1]) | (p0[:, 1] > hi[1]) | (p15[:, 1] < lo[1]) | (p15[:, 1] > hi[1])
    outside_z = (p0[:, 2] < lo[2]) | (p0[:, 2] > hi[2]) | (p15[:, 2] < lo[2]) | (p15[:, 2] > hi[2])
    return {
        "p0_world": p0,
        "p15_world": p15,
        "inside0": inside0,
        "inside15": inside15,
        "outside_x": outside_x,
        "outside_y": outside_y,
        "outside_z": outside_z,
    }


def build_w12_coverage_audit(model_v2: VectorCarrierModel, grid_7p5: TeacherFlowGrid) -> dict[str, Any]:
    points = _grid_points(grid_7p5).reshape(-1, 3)
    mapping = _w12_mapping(points, model_v2)
    inside0 = mapping["inside0"]
    inside15 = mapping["inside15"]
    categories = {
        "inside_both_anchors": inside0 & inside15,
        "outside_anchor_0_only": ~inside0 & inside15,
        "outside_anchor_15_only": inside0 & ~inside15,
        "outside_both_anchors": ~inside0 & ~inside15,
        "outside_due_to_x_bounds": mapping["outside_x"],
        "outside_due_to_y_bounds": mapping["outside_y"],
        "outside_due_to_z_bounds": mapping["outside_z"],
    }
    report = {
        "schema_version": "warp_w1_3_w1_2_coverage_audit_v1",
        "angle_deg": 7.5,
        "grid_sample_count": int(len(points)),
        "bounds_min_m": model_v2.bounds_min_m.tolist(),
        "bounds_max_m": model_v2.bounds_max_m.tolist(),
        "categories": {
            name: {"count": int(np.count_nonzero(mask)), "fraction": float(np.mean(mask))}
            for name, mask in categories.items()
        },
        "w1_2_inside_both_fraction": float(np.mean(inside0 & inside15)),
        "w1_2_out_of_field_count": int(np.count_nonzero(~(inside0 & inside15))),
        "mapping_rule": "W1.2 rotated query coordinates into endpoint fields before sampling",
    }
    y_values = points[:, 1]
    z_values = points[:, 2]
    slice_mask = np.isclose(y_values, y_values[np.argmin(np.abs(y_values))], atol=grid_7p5.spacing_m[1] * 0.01)
    colours = np.full(len(points), "#7f8c8d", dtype=object)
    colours[inside0 & inside15] = "#1b9e77"
    colours[~inside0 & inside15] = "#d95f02"
    colours[inside0 & ~inside15] = "#7570b3"
    colours[~inside0 & ~inside15] = "#e7298a"
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.scatter(points[slice_mask, 0], points[slice_mask, 2], c=colours[slice_mask], s=22, marker="s")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world z [m]")
    ax.set_title("W1.2 7.5° coverage loss on y≈0 slice")
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color="#1b9e77", label="inside both"),
        plt.Line2D([], [], marker="s", linestyle="", color="#d95f02", label="outside anchor 0 only"),
        plt.Line2D([], [], marker="s", linestyle="", color="#7570b3", label="outside anchor 15 only"),
        plt.Line2D([], [], marker="s", linestyle="", color="#e7298a", label="outside both"),
    ]
    ax.legend(handles=handles, frameon=False)
    ax.set_aspect("equal")
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA_DIR / "warp_w1_3_w1_2_coverage_loss.png", dpi=180)
    plt.close(fig)
    return report


def _fixed_coverage(model: VectorCarrierModel) -> dict[str, Any]:
    points = _grid_points(model.anchor_0).reshape(-1, 3)
    result: dict[str, Any] = {}
    for angle in (7.5, 10.0):
        _, inside = model.sample(points, angle, interpolation=FIXED_GRID_LOCAL_VECTOR_INTERP)
        result[f"{angle:g}_deg"] = {
            "inside_fraction": float(np.mean(inside)),
            "out_of_field_count": int(np.count_nonzero(~inside)),
            "pass": bool(np.all(inside)),
        }
    return result


def _freeze_carrier_prediction(model: VectorCarrierModel, payload: dict[str, Any]) -> dict[str, Any]:
    if PREDICTED_U_JSON.exists() or PREDICTED_U_NPZ.exists():
        if not (PREDICTED_U_JSON.exists() and PREDICTED_U_NPZ.exists()):
            raise VectorCarrierError("10-degree carrier prediction pair is incomplete")
        record = json.loads(PREDICTED_U_JSON.read_text(encoding="utf-8"))
        if record.get("model_id") != payload["model_id"] or record["model_sha256"] != payload["model_sha256"] or record["prediction_sha256"] != _sha256_file(PREDICTED_U_NPZ):
            raise VectorCarrierError("frozen 10-degree carrier prediction provenance mismatch")
        with np.load(PREDICTED_U_NPZ, allow_pickle=False) as stored:
            stored_u = np.asarray(stored["U"], dtype=np.float32)
            stored_inside = np.asarray(stored["inside_mask"], dtype=bool)
        expected_u, expected_inside = model.sample(_grid_points(model.anchor_0), 10.0, interpolation=FIXED_GRID_LOCAL_VECTOR_INTERP)
        if not np.array_equal(stored_u, expected_u.astype(np.float32)) or not np.array_equal(stored_inside, expected_inside):
            raise VectorCarrierError("frozen 10-degree carrier prediction is not reproducible")
        return record
    if CASE_10.exists():
        raise VectorCarrierError("10-degree case exists before carrier prediction freeze")
    predicted, inside = model.sample(_grid_points(model.anchor_0), 10.0, interpolation=FIXED_GRID_LOCAL_VECTOR_INTERP)
    predicted = predicted.reshape(model.anchor_0.values_nz_ny_nx_3.shape).astype(np.float32)
    inside = inside.reshape(model.anchor_0.values_nz_ny_nx_3.shape[:3])
    np.savez_compressed(
        PREDICTED_U_NPZ,
        U=predicted,
        inside_mask=inside,
        first_center_m=np.asarray(model.first_center_m, dtype=np.float64),
        spacing_m=np.asarray(model.spacing_m, dtype=np.float64),
        dimensions_nx_ny_nz=np.asarray(model.dimensions_nx_ny_nz, dtype=np.int32),
        angle_deg=np.asarray([10.0], dtype=np.float64),
    )
    record = {
        "schema_version": "warp_w1_3_frozen_10deg_carrier_v1",
        "model_id": payload["model_id"],
        "model_sha256": payload["model_sha256"],
        "angle_deg": 10.0,
        "interpolation_method": FIXED_GRID_LOCAL_VECTOR_INTERP,
        "prediction_sha256": _sha256_file(PREDICTED_U_NPZ),
        "prediction_array_sha256": _json_hash(predicted),
        "inside_mask_array_sha256": _json_hash(inside.astype(np.uint8)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "OPENFOAM_10DEG_NOT_READ_AT_PREDICTION_TIME": True,
        "case_created_at_prediction_time": False,
    }
    _write_json(PREDICTED_U_JSON, record)
    return record


def _freeze_deposition_prediction(model: VectorCarrierModel, payload: dict[str, Any], air: AirFieldParameters, rho: float, mu: float, *, device: str) -> dict[str, Any]:
    if PREDICTED_DEP_JSON.exists() or PREDICTED_DEP_NPZ.exists():
        if not (PREDICTED_DEP_JSON.exists() and PREDICTED_DEP_NPZ.exists()):
            raise VectorCarrierError("10-degree deposition prediction pair is incomplete")
        record = json.loads(PREDICTED_DEP_JSON.read_text(encoding="utf-8"))
        if record.get("model_id") != payload["model_id"] or record["model_sha256"] != payload["model_sha256"] or record["prediction_sha256"] != _sha256_file(PREDICTED_DEP_NPZ):
            raise VectorCarrierError("frozen 10-degree deposition prediction provenance mismatch")
        return record
    if CASE_10.exists():
        raise VectorCarrierError("10-degree case exists before deposition prediction freeze")
    config = _make_config(10.0, air, rho, mu)
    result = run_warp_case(config, device=device, carrier_mode=FIXED_GRID_LOCAL_VECTOR_INTERP, vector_carrier=model)
    np.savez_compressed(
        PREDICTED_DEP_NPZ,
        map_mass_kg=result.map_mass_kg,
        map_density_kg_m2=result.map_density_kg_m2,
        expected_injected_mass_kg=np.asarray([result.mass_ledger["expected_injected_mass_kg"]], dtype=np.float64),
        deposited_mass_kg=np.asarray([result.mass_ledger["deposited_mass_kg"]], dtype=np.float64),
        escaped_mass_kg=np.asarray([result.mass_ledger["escaped_mass_kg"]], dtype=np.float64),
        relative_balance_error=np.asarray([result.mass_ledger["relative_balance_error"]], dtype=np.float64),
    )
    record = {
        "schema_version": "warp_w1_3_frozen_10deg_deposition_v1",
        "model_id": payload["model_id"],
        "model_sha256": payload["model_sha256"],
        "angle_deg": 10.0,
        "carrier_mode": FIXED_GRID_LOCAL_VECTOR_INTERP,
        "prediction_sha256": _sha256_file(PREDICTED_DEP_NPZ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "OPENFOAM_10DEG_NOT_READ_AT_PREDICTION_TIME": True,
        "case_created_at_prediction_time": False,
        "mass_ledger": result.mass_ledger,
        "performance": result.performance,
    }
    _write_json(PREDICTED_DEP_JSON, record)
    return record


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_existing_rows() -> dict[str, dict[str, Any]]:
    w1 = _load_json(WARP_RESULT_DIR / "case_7p5deg_metrics.json")["high"]
    w12 = _load_json(ROOT / "results" / "air_assisted" / "warp_w1_2" / "case_7p5deg_metrics.json")["carrier_modes"]["vector_w1_2"]
    w11 = _load_json(ROOT / "results" / "air_assisted" / "warp_w1_1" / "case_7p5deg_teacher_u_metrics.json")["carrier_modes"]["teacher_forced_u"]
    return {"W1": w1, "W1.2": w12, "W1.1 Teacher-U": w11}


def _run_regression(model_v3: VectorCarrierModel, model_v2: VectorCarrierModel, grids: dict[float, TeacherFlowGrid], air: AirFieldParameters, rho: float, mu: float, *, device: str) -> dict[str, Any]:
    teacher_map, teacher_metrics = _load_teacher_map(7.5)
    s2_model = __import__("aerospace_painting.s2_runtime", fromlist=["S2RuntimeModel"]).S2RuntimeModel.load(ROOT / "models" / "s2_air_assisted_v1.json")
    s2_model.verify_sources(ROOT)
    uv = np.column_stack((teacher_map["u_m"], teacher_map["v_m"]))
    s2_density = s2_model.predict(7.5).density(uv)
    config = _make_config(7.5, air, rho, mu)
    runs: dict[str, WarpCaseResult] = {}
    runs["analytic_w1"] = run_warp_case(config, device=device, carrier_mode="ANALYTIC_AIR")
    runs["teacher_u_w1_1"] = run_warp_case(config, device=device, carrier_mode="TEACHER_FORCED_U", teacher_flow=grids[7.5])
    runs["vector_w1_2"] = run_warp_case(config, device=device, carrier_mode=COROTATING_VECTOR_INTERP, vector_carrier=model_v2)
    runs["vector_w1_3"] = run_warp_case(config, device=device, carrier_mode=FIXED_GRID_LOCAL_VECTOR_INTERP, vector_carrier=model_v3)
    summaries = {
        key: _result_summary(7.5, result, teacher_map, teacher_metrics, s2_density if key == "analytic_w1" else None)
        for key, result in runs.items()
    }
    rows: dict[str, dict[str, Any]] = {}
    rows["Analytic W1"] = _metric_row(summaries["analytic_w1"])
    rows["Teacher-U W1.1"] = _metric_row(summaries["teacher_u_w1_1"])
    rows["W1.2"] = _metric_row(summaries["vector_w1_2"])
    rows["W1.3"] = _metric_row(summaries["vector_w1_3"])
    s2_row, _ = _s2_row(s2_density, teacher_map, teacher_metrics)
    rows["S2"] = s2_row
    rows["OpenFOAM"] = {
        "transfer_efficiency": teacher_metrics["mass_ledger"]["transfer_efficiency"],
        "deposited_mass_kg": teacher_metrics["mass_ledger"]["deposited_kg"],
        "centroid_u_m": teacher_metrics["deposition"]["centroid_u_m"],
        "centroid_v_m": teacher_metrics["deposition"]["centroid_v_m"],
        "sigma_major_m": teacher_metrics["deposition"]["sigma_major_m"],
        "sigma_minor_m": teacher_metrics["deposition"]["sigma_minor_m"],
        "peak_density_kg_m2": teacher_metrics["deposition"]["peak_areal_mass_kg_m2"],
        "nrmse": 0.0,
        "pearson_correlation": 1.0,
    }
    figure_fields = (
        ("OpenFOAM", teacher_map["areal_mass_kg_m2"]),
        ("S2", s2_density),
        ("W1 analytic", runs["analytic_w1"].map_density_kg_m2),
        ("W1.2", runs["vector_w1_2"].map_density_kg_m2),
        ("W1.3", runs["vector_w1_3"].map_density_kg_m2),
        ("Teacher-U", runs["teacher_u_w1_1"].map_density_kg_m2),
    )
    vmax = max(float(np.max(field)) for _, field in figure_fields)
    fig, axes = plt.subplots(1, len(figure_fields), figsize=(22, 4.0), constrained_layout=True, sharex=True, sharey=True)
    for ax, (label, field) in zip(axes, figure_fields):
        image = ax.scatter(uv[:, 0], uv[:, 1], c=field, s=20, marker="s", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(-0.15, 0.15)
        ax.set_xlabel("u [m]")
    axes[0].set_ylabel("v [m]")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="areal mass [kg m$^{-2}$]")
    fig.suptitle("W1.3 7.5° fixed-grid regression")
    fig.savefig(MEDIA_DIR / "warp_w1_3_7p5_regression.png", dpi=180)
    plt.close(fig)
    coverage = _fixed_coverage(model_v3)
    regression = {
        "schema_version": "warp_w1_3_regression_7p5_v1",
        "angle_deg": 7.5,
        "carrier_status": "REGRESSION_ONLY_NOT_BLIND_HOLDOUT",
        "coverage": coverage["7.5_deg"],
        "comparison": rows,
        "w1_3_summary": summaries["vector_w1_3"],
        "w1_3_gate": {
            "mass_balance": abs(summaries["vector_w1_3"]["mass_ledger"]["relative_balance_error"]) <= 1e-5,
            "deposited_mass": summaries["vector_w1_3"]["deposition"]["deposited_mass_relative_error"] <= 0.10,
            "centroid": summaries["vector_w1_3"]["deposition"]["centroid_error_m"] <= 0.010,
            "sigma_major": summaries["vector_w1_3"]["deposition"]["sigma_major_relative_error"] <= 0.15,
            "sigma_minor": summaries["vector_w1_3"]["deposition"]["sigma_minor_relative_error"] <= 0.15,
            "correlation": summaries["vector_w1_3"]["comparison"]["openfoam_teacher"]["pearson_correlation"] >= 0.80,
            "nrmse": summaries["vector_w1_3"]["comparison"]["openfoam_teacher"]["nrmse"] <= 0.40,
        },
        "performance": {key: summaries[key]["performance"] for key in summaries},
    }
    _write_json(W1_3_ROOT / "regression_7p5.json", regression)
    return {"regression": regression, "runs": runs, "summaries": summaries, "rows": rows}


def _render_fidelity_hierarchy(regression: dict[str, Any]) -> None:
    rows = regression["comparison"]
    labels = ["OpenFOAM", "S2", "Analytic W1", "W1.2", "W1.3", "Teacher-U W1.1"]
    keys = ["transfer_efficiency", "centroid_u_m", "sigma_major_m"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), constrained_layout=True)
    colours = ("#2166ac", "#5e3c99", "#e66101", "#7570b3", "#1b7837", "#762a83")
    for ax, key in zip(axes, keys):
        values = [rows[label][key] for label in labels]
        ax.bar(labels, values, color=colours)
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("W1.3 measured 7.5° fidelity hierarchy")
    fig.savefig(MEDIA_DIR / "warp_w1_3_fidelity_hierarchy.png", dpi=180)
    plt.close(fig)


def _carrier_report_10(predicted: np.ndarray, inside: np.ndarray, actual_grid: TeacherFlowGrid) -> dict[str, Any]:
    actual = np.asarray(actual_grid.values_nz_ny_nx_3, dtype=np.float32)
    regions = {
        "whole_flow_domain": np.ones(actual.shape[:3], dtype=bool),
        "high_speed_jet": np.linalg.norm(actual, axis=-1) >= 0.5 * float(np.linalg.norm(actual, axis=-1).max()),
        "near_target_region": _grid_points(actual_grid)[..., 2] >= float(actual_grid.bounds_max_m[2] - 0.06),
    }
    result = {
        "schema_version": "warp_w1_3_carrier_holdout_10deg_v1",
        "angle_deg": 10.0,
        "openfoam_version": actual_grid.openfoam_version,
        "prediction_inside_fraction": float(np.mean(inside)),
        "prediction_out_of_field_count": int(np.count_nonzero(~inside)),
        "regions": {},
        "region_definitions": {
            "high_speed_jet": "actual OpenFOAM speed >= 0.5 * whole-domain maximum speed",
            "near_target_region": "cell-center z >= finite-volume upper z bound - 0.06 m",
        },
    }
    for name, mask in regions.items():
        result["regions"][name] = _region_metrics(predicted, actual, mask, name, inside)
    whole = result["regions"]["whole_flow_domain"]
    result["carrier_gate"] = {
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
    return result


def _render_carrier_holdout_10(predicted: np.ndarray, actual_grid: TeacherFlowGrid) -> None:
    actual = np.asarray(actual_grid.values_nz_ny_nx_3, dtype=np.float32)
    points = _grid_points(actual_grid)
    speed_actual = np.linalg.norm(actual, axis=-1)
    speed_predicted = np.linalg.norm(predicted, axis=-1)
    vector_error = np.linalg.norm(predicted - actual, axis=-1)
    slice_index = int(np.argmin(np.abs(points[0, :, 0, 1])))
    x = points[:, slice_index, :, 0].ravel()
    z = points[:, slice_index, :, 2].ravel()
    fields = (
        ("OpenFOAM speed", speed_actual[:, slice_index, :].ravel(), "m/s"),
        ("W1.3 frozen speed", speed_predicted[:, slice_index, :].ravel(), "m/s"),
        ("absolute vector error", vector_error[:, slice_index, :].ravel(), "m/s"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), constrained_layout=True, sharex=True, sharey=True)
    for ax, (title, field, label) in zip(axes, fields):
        image = ax.scatter(x, z, c=field, s=18, marker="s", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("world x [m]")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, label=label)
    axes[0].set_ylabel("world z [m]")
    fig.suptitle("W1.3 frozen 10° carrier hold-out")
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA_DIR / "warp_w1_3_carrier_holdout_10deg.png", dpi=180)
    plt.close(fig)


def _load_frozen_deposition() -> tuple[np.ndarray, dict[str, Any]]:
    record = _load_json(PREDICTED_DEP_JSON)
    with np.load(PREDICTED_DEP_NPZ, allow_pickle=False) as values:
        return np.asarray(values["map_density_kg_m2"], dtype=float), record


def prepare(*, device: str) -> dict[str, Any]:
    W1_3_ROOT.mkdir(parents=True, exist_ok=True)
    grids = _load_grids()
    domain_audit = build_domain_audit(grids)
    _write_json(W1_3_ROOT / "domain_audit.json", domain_audit)
    if not domain_audit["fixed_eulerian_domain"]:
        raise SystemExit("BLOCKED_DOMAIN_ASSUMPTION")
    model_v2, _ = load_model_artifacts(ROOT / "models" / "warp_vector_carrier_v2.json", repo_root=ROOT, verify_sources=True)
    coverage = build_w12_coverage_audit(model_v2, grids[7.5])
    _write_json(W1_3_ROOT / "w1_2_coverage_audit.json", coverage)
    model_v3, payload_v3 = build_model_variant(
        model_id="warp_vector_carrier_v3",
        interpolation_method=FIXED_GRID_LOCAL_VECTOR_INTERP,
        json_path=MODEL_JSON,
        npz_path=MODEL_NPZ,
        force=False,
    )
    fixed_coverage = _fixed_coverage(model_v3)
    if not all(item["pass"] for item in fixed_coverage.values()):
        raise SystemExit("BLOCKED_FIXED_GRID_COVERAGE")
    air = _load_air_parameters()
    rho, mu = _load_carrier_properties()
    regression = _run_regression(model_v3, model_v2, grids, air, rho, mu, device=device)
    _render_fidelity_hierarchy(regression["regression"])
    carrier_prediction = _freeze_carrier_prediction(model_v3, payload_v3)
    deposition_prediction = _freeze_deposition_prediction(model_v3, payload_v3, air, rho, mu, device=device)
    if CASE_10.exists():
        raise SystemExit("BLOCKED_OPENFOAM_10DEG")
    summary = {
        "schema_version": "warp_w1_3_prepare_summary_v1",
        "fixed_grid_domain": domain_audit,
        "w1_2_coverage": coverage,
        "w1_3_model": payload_v3,
        "fixed_grid_coverage": fixed_coverage,
        "regression_7p5": regression["regression"],
        "frozen_10deg_carrier": carrier_prediction,
        "frozen_10deg_deposition": deposition_prediction,
        "OPENFOAM_10DEG_NOT_READ_AT_PREDICTION_TIME": True,
        "next_step": "Create the medium_incidence_10deg case and run blockMesh/checkMesh/sprayFoam only after these hashes exist.",
    }
    _write_json(W1_3_ROOT / "prepare_summary.json", summary)
    print(json.dumps({"status": "PREPARED_W1_3", "fixed_grid_coverage": fixed_coverage, "carrier_hash": carrier_prediction["prediction_sha256"], "deposition_hash": deposition_prediction["prediction_sha256"]}, indent=2, sort_keys=True))
    return summary


def validate(*, device: str, openfoam_runtime_seconds: float | None = None) -> dict[str, Any]:
    if not (PREDICTED_U_JSON.exists() and PREDICTED_DEP_JSON.exists()):
        raise SystemExit("BLOCKED_OTHER: frozen 10-degree predictions are missing")
    grids = _load_grids()
    model_v3, payload_v3 = load_model_artifacts(MODEL_JSON, repo_root=ROOT, verify_sources=True)
    carrier_prediction_record = _load_json(PREDICTED_U_JSON)
    deposition_prediction_record = _load_json(PREDICTED_DEP_JSON)
    if not carrier_prediction_record.get("OPENFOAM_10DEG_NOT_READ_AT_PREDICTION_TIME") or not deposition_prediction_record.get("OPENFOAM_10DEG_NOT_READ_AT_PREDICTION_TIME"):
        raise SystemExit("BLOCKED_OTHER: prediction provenance flag is false")
    if not CASE_10.exists():
        raise SystemExit("BLOCKED_OPENFOAM_10DEG")
    input_diff = compare_incidence_inputs(CASE_DIR / "medium", CASE_10, angle_deg=10.0)
    _write_json(W1_3_ROOT / "holdout_10deg_input_diff.json", input_diff)
    if not input_diff["pass"]:
        raise SystemExit("BLOCKED_OPENFOAM_10DEG: input diff failed")
    try:
        from scripts.postprocess_openfoam_flat_plate import process_case
    except ModuleNotFoundError:
        from postprocess_openfoam_flat_plate import process_case
    metrics_10 = process_case(CASE_10, RESULT_DIR, "medium_incidence_10deg")
    actual_grid = TeacherFlowGrid.from_case_dir(CASE_10, 10.0, repo_root=ROOT)
    with np.load(PREDICTED_U_NPZ, allow_pickle=False) as values:
        predicted_u = np.asarray(values["U"], dtype=np.float32)
        predicted_inside = np.asarray(values["inside_mask"], dtype=bool)
    carrier_validation = _carrier_report_10(predicted_u, predicted_inside, actual_grid)
    _render_carrier_holdout_10(predicted_u, actual_grid)
    _write_json(W1_3_ROOT / "holdout_10deg_carrier_validation.json", carrier_validation)
    with np.load(RESULT_10 / "deposition_map.npz", allow_pickle=False) as actual_map:
        actual_map_data = {key: np.asarray(actual_map[key], dtype=float) for key in actual_map.files}
    predicted_density, _ = _load_frozen_deposition()
    uv = np.column_stack((actual_map_data["u_m"], actual_map_data["v_m"]))
    area = actual_map_data["area_m2"]
    deposition_comparison = _field_comparison(predicted_density, actual_map_data["areal_mass_kg_m2"], uv, area)
    predicted_moments = _map_moments(uv, predicted_density, area)
    actual_dep = metrics_10["deposition"]
    dep_validation = {
        "schema_version": "warp_w1_3_deposition_holdout_10deg_v1",
        "angle_deg": 10.0,
        "frozen_prediction_sha256": deposition_prediction_record["prediction_sha256"],
        "predicted": predicted_moments,
        "openfoam": {
            "deposited_mass_kg": metrics_10["mass_ledger"]["deposited_kg"],
            "escaped_mass_kg": metrics_10["mass_ledger"]["escaped_kg"],
            "transfer_efficiency": metrics_10["mass_ledger"]["transfer_efficiency"],
            "centroid_u_m": actual_dep["centroid_u_m"],
            "centroid_v_m": actual_dep["centroid_v_m"],
            "sigma_major_m": actual_dep["sigma_major_m"],
            "sigma_minor_m": actual_dep["sigma_minor_m"],
            "peak_density_kg_m2": actual_dep["peak_areal_mass_kg_m2"],
        },
        "comparison": deposition_comparison,
        "predicted_mass_balance": deposition_prediction_record["mass_ledger"],
        "gates": {
            "mass_balance": abs(float(deposition_prediction_record["mass_ledger"]["relative_balance_error"])) <= 1.0e-5,
            "deposited_mass": deposition_comparison["integrated_mass_relative_error"] <= 0.10,
            "centroid": deposition_comparison["centroid_error_m"] <= 0.010,
            "sigma_major": deposition_comparison["sigma_major_relative_error"] <= 0.15,
            "sigma_minor": deposition_comparison["sigma_minor_relative_error"] <= 0.15,
            "correlation": deposition_comparison["pearson_correlation"] >= 0.80,
            "nrmse": deposition_comparison["nrmse"] <= 0.40,
        },
    }
    dep_validation["gate_pass"] = bool(all(dep_validation["gates"].values()))
    dep_validation["integrated_absolute_density_error_kg"] = deposition_comparison["integrated_absolute_mass_density_error_kg"]
    _write_json(W1_3_ROOT / "holdout_10deg_deposition_validation.json", dep_validation)

    vector_gate = carrier_validation["carrier_gate"]["pass"]
    overall_pass = bool(vector_gate and dep_validation["gate_pass"])
    status = "WARP_W1_VECTOR_CARRIER_VALIDATED" if overall_pass else "WARP_VECTOR_CARRIER_V3_NOT_VALIDATED"
    result = "PASS" if overall_pass else "PARTIAL"
    perf = _load_json(W1_3_ROOT / "regression_7p5.json")["performance"]
    perf["vector_w1_3_10deg"] = deposition_prediction_record["performance"]
    perf["openfoam_10deg"] = {"runtime_seconds": openfoam_runtime_seconds, "status": "MEASURED" if openfoam_runtime_seconds is not None else "NOT_PROVIDED"}
    endpoint_15, endpoint_inside_15 = model_v3.sample(
        _grid_points(grids[15.0]),
        15.0,
        interpolation=FIXED_GRID_LOCAL_VECTOR_INTERP,
    )
    endpoint_error = np.asarray(endpoint_15, dtype=np.float64).reshape(-1, 3) - np.asarray(
        grids[15.0].values_nz_ny_nx_3.reshape(-1, 3), dtype=np.float64
    )
    openfoam_15_metrics = _load_json(RESULT_DIR / CASE_LABELS[15.0] / "metrics.json")
    air = _load_air_parameters()
    rho, mu = _load_carrier_properties()
    w1_3_15_result = run_warp_case(
        _make_config(15.0, air, rho, mu),
        device=device,
        carrier_mode=FIXED_GRID_LOCAL_VECTOR_INTERP,
        vector_carrier=model_v3,
    )
    w1_1_15_metrics = _load_json(ROOT / "results" / "air_assisted" / "warp_w1_1" / "case_15deg_teacher_u_metrics.json")["carrier_modes"]["teacher_forced_u"]
    sanity15 = {
        "endpoint_inside_fraction": float(np.mean(endpoint_inside_15)),
        "endpoint_max_abs_error_m_s": float(np.max(np.abs(endpoint_error))),
        "endpoint_exact": bool(np.all(endpoint_inside_15) and np.max(np.abs(endpoint_error)) <= 1.0e-6),
        "openfoam_transfer_efficiency": openfoam_15_metrics["mass_ledger"]["transfer_efficiency"],
        "openfoam_deposited_mass_kg": openfoam_15_metrics["mass_ledger"]["deposited_kg"],
        "openfoam_centroid_u_m": openfoam_15_metrics["deposition"]["centroid_u_m"],
        "openfoam_centroid_v_m": openfoam_15_metrics["deposition"]["centroid_v_m"],
        "openfoam_sigma_major_m": openfoam_15_metrics["deposition"]["sigma_major_m"],
        "openfoam_sigma_minor_m": openfoam_15_metrics["deposition"]["sigma_minor_m"],
        "w1_1_transfer_efficiency": w1_1_15_metrics["mass_ledger"]["transfer_efficiency"],
        "w1_1_centroid_u_m": w1_1_15_metrics["deposition"]["centroid_uv_m"][0],
        "w1_3_transfer_efficiency": w1_3_15_result.mass_ledger["transfer_efficiency"],
        "w1_3_centroid_u_m": _map_moments(
            np.column_stack((_load_teacher_map(15.0)[0]["u_m"], _load_teacher_map(15.0)[0]["v_m"])),
            w1_3_15_result.map_mass_kg,
            _load_teacher_map(15.0)[0]["area_m2"],
        )["centroid_uv_m"][0],
    }
    final = {
        "schema_version": "warp_w1_3_validation_summary_v1",
        "result": result,
        "status": status,
        "domain_audit": _load_json(W1_3_ROOT / "domain_audit.json"),
        "w1_2_coverage_audit": _load_json(W1_3_ROOT / "w1_2_coverage_audit.json"),
        "model": payload_v3,
        "regression_7p5": _load_json(W1_3_ROOT / "regression_7p5.json"),
        "frozen_10deg_prediction": {
            "carrier": carrier_prediction_record,
            "deposition": deposition_prediction_record,
            "created_before_cfd": True,
        },
        "holdout_10deg_carrier": carrier_validation,
        "holdout_10deg_deposition": dep_validation,
        "15deg_sanity": sanity15,
        "performance": perf,
        "fidelity_boundary": {
            "description": "W1.3 is fixed-grid two-anchor full-vector interpolation plus Lagrangian droplet transport; it is not online CFD.",
            "not_modeled": ["primary atomization", "breakup", "evaporation", "splash", "wall film", "curing", "validated physical film thickness"],
        },
        "artifacts": {
            "model_json": str(MODEL_JSON.relative_to(ROOT)),
            "model_npz": str(MODEL_NPZ.relative_to(ROOT)),
            "domain_audit": str((W1_3_ROOT / "domain_audit.json").relative_to(ROOT)),
            "coverage_audit": str((W1_3_ROOT / "w1_2_coverage_audit.json").relative_to(ROOT)),
            "input_diff": str((W1_3_ROOT / "holdout_10deg_input_diff.json").relative_to(ROOT)),
            "carrier_validation": str((W1_3_ROOT / "holdout_10deg_carrier_validation.json").relative_to(ROOT)),
            "deposition_validation": str((W1_3_ROOT / "holdout_10deg_deposition_validation.json").relative_to(ROOT)),
        },
    }
    _write_json(W1_3_ROOT / "validation_summary.json", final)
    print(json.dumps({"result": result, "status": status, "carrier_gate": carrier_validation["carrier_gate"], "deposition_gate": dep_validation["gates"]}, indent=2, sort_keys=True))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="audit, regress, and freeze 10-degree predictions before CFD")
    parser.add_argument("--validate", action="store_true", help="read the completed 10-degree CFD case and validate it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--openfoam-runtime-seconds", type=float, default=None)
    args = parser.parse_args()
    if args.prepare == args.validate:
        parser.error("choose exactly one of --prepare or --validate")
    if args.prepare:
        prepare(device=args.device)
    else:
        validate(device=args.device, openfoam_runtime_seconds=args.openfoam_runtime_seconds)


if __name__ == "__main__":
    main()
