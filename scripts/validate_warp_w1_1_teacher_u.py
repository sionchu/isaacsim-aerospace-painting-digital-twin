"""Run the W1.1 teacher-forced vector-field transport diagnostic.

This diagnostic intentionally keeps the original W1 artifacts untouched.  It
replays the canonical analytic-air ensemble for comparison, then runs the
same GPU particle transport with each angle's solved OpenFOAM vector field as
the carrier source.  No deposition data is used to fit or alter the field.
"""

from __future__ import annotations

import argparse
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

from aerospace_painting.warp_air_field import AirFieldParameters
from aerospace_painting.warp_spray import WarpCaseConfig, WarpCaseResult, run_warp_case
from aerospace_painting.warp_teacher_flow import TeacherFlowGrid, sample_teacher_flow_warp
from scripts.validate_warp_spray import (
    CASE_DIR,
    CASE_LABELS,
    MEDIA_DIR,
    RESULT_DIR,
    _field_comparison,
    _load_teacher_map,
    _map_moments,
    _result_summary,
    _sha256_file,
    _write_json,
)


W1_ROOT = ROOT / "results" / "air_assisted" / "warp_w1"
DIAGNOSTIC_ROOT = ROOT / "results" / "air_assisted" / "warp_w1_1"
NPZ_ROOT = DIAGNOSTIC_ROOT
HIGH_PARTICLES_PER_BIN = 2000
ANGLES = (0.0, 7.5, 15.0)


def _load_air_parameters() -> AirFieldParameters:
    payload = json.loads((ROOT / "models" / "warp_air_field_v1.json").read_text(encoding="utf-8"))
    return AirFieldParameters.from_dict(payload["parameters"])


def _load_carrier_properties() -> tuple[float, float]:
    payload = json.loads((W1_ROOT / "environment.json").read_text(encoding="utf-8"))
    carrier = payload["carrier_properties"]
    return float(carrier["ideal_gas_density_kg_m3"]), float(carrier["dynamic_viscosity_pa_s"])


def _load_teacher_grids() -> dict[float, TeacherFlowGrid]:
    grids: dict[float, TeacherFlowGrid] = {}
    for angle in ANGLES:
        grid = TeacherFlowGrid.from_case_dir(
            CASE_DIR / CASE_LABELS[angle],
            angle,
            repo_root=ROOT,
        )
        npz_path = NPZ_ROOT / f"teacher_u_{'0deg' if angle == 0.0 else '7p5deg' if angle == 7.5 else '15deg'}.npz"
        grid.to_npz(npz_path)
        grids[angle] = grid
    return grids


def _sampler_agreement(grids: dict[float, TeacherFlowGrid], *, device: str) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for angle, grid in grids.items():
        first = np.asarray(grid.first_center_m, dtype=float)
        spacing = np.asarray(grid.spacing_m, dtype=float)
        last = first + (np.asarray(grid.dimensions_nx_ny_nz, dtype=float) - 1.0) * spacing
        fractions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
                [0.25, 0.75, 0.4],
                [0.8, 0.2, 0.65],
            ],
            dtype=float,
        )
        points = first[None, :] + fractions * (last - first)[None, :]
        # Use centre/interior points for the CPU↔GPU numerical agreement;
        # boundary and outside policy are covered separately by the portable
        # tests and the explicit mask check below.
        points = np.vstack((points, grid.bounds_min_m + 0.25 * spacing, grid.bounds_max_m - 0.25 * spacing, grid.bounds_min_m + np.asarray((0.0, 0.0, -1.0e-4))))
        cpu_values, cpu_inside = grid.trilinear_velocity(points)
        gpu_values, gpu_inside = sample_teacher_flow_warp(grid, points, device=device)
        inside_match = bool(np.array_equal(cpu_inside, gpu_inside))
        value_error = float(np.max(np.abs(cpu_values[cpu_inside] - gpu_values[cpu_inside]))) if np.any(cpu_inside) else 0.0
        report[f"{angle:g}_deg"] = {
            "sample_count": int(len(points)),
            "inside_mask_match": inside_match,
            "max_abs_velocity_error_m_s": value_error,
            "comparison_tolerance_m_s": 2.0e-5,
            "pass": bool(inside_match and value_error <= 2.0e-5),
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
        "deposited_mass_kg": deposition["mass_kg"],
        "transfer_efficiency": ledger["transfer_efficiency"],
        "centroid_u_m": deposition["centroid_uv_m"][0],
        "centroid_v_m": deposition["centroid_uv_m"][1],
        "sigma_major_m": deposition["sigma_major_m"],
        "sigma_minor_m": deposition["sigma_minor_m"],
        "covariance_uv_m2": deposition["covariance_uv_m2"],
        "peak_density_kg_m2": deposition["peak_density_kg_m2"],
        "nrmse": field["nrmse"],
        "pearson_correlation": field["pearson_correlation"],
        "mass_balance_relative_error": ledger["relative_balance_error"],
        "escaped_mass_kg": ledger["escaped_mass_kg"],
        "teacher_out_of_field_mass_kg": ledger.get("teacher_out_of_field_mass_kg", 0.0),
    }


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "deposited_mass_kg": right["deposited_mass_kg"] - left["deposited_mass_kg"],
        "transfer_efficiency": right["transfer_efficiency"] - left["transfer_efficiency"],
        "centroid_u_m": right["centroid_u_m"] - left["centroid_u_m"],
        "centroid_v_m": right["centroid_v_m"] - left["centroid_v_m"],
        "sigma_major_m": right["sigma_major_m"] - left["sigma_major_m"],
        "sigma_minor_m": right["sigma_minor_m"] - left["sigma_minor_m"],
        "escaped_mass_kg": right["escaped_mass_kg"] - left["escaped_mass_kg"],
    }


def _teacher_gate(summary: dict[str, Any]) -> dict[str, bool]:
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


def _render_7p5(openfoam: np.ndarray, analytic: np.ndarray, teacher: np.ndarray, uv: np.ndarray) -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    analytic_error = np.abs(analytic - openfoam)
    teacher_error = np.abs(teacher - openfoam)
    values = (openfoam, analytic, teacher, analytic_error, teacher_error)
    vmax = max(float(np.max(value)) for value in values)
    norm = plt.Normalize(vmin=0.0, vmax=vmax if vmax > 0.0 else 1.0)
    fields = (
        ("OpenFOAM 7.5°", openfoam),
        ("Warp analytic-air", analytic),
        ("Warp teacher-forced-U", teacher),
        ("absolute error: analytic", analytic_error),
        ("absolute error: teacher", teacher_error),
    )
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2), constrained_layout=True, sharex=True, sharey=True)
    for axis, (label, field) in zip(axes, fields):
        scatter = axis.scatter(uv[:, 0], uv[:, 1], c=field, s=22, marker="s", cmap="viridis", norm=norm)
        axis.set_title(label)
        axis.set_aspect("equal")
        axis.set_xlim(-0.15, 0.15)
        axis.set_ylim(-0.15, 0.15)
        axis.set_xlabel("u [m]")
    axes[0].set_ylabel("v [m]")
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="areal mass / absolute error [kg m$^{-2}$]")
    fig.suptitle("W1.1 teacher-forced U diagnostic at 7.5°")
    fig.savefig(MEDIA_DIR / "warp_w1_1_teacher_u_7p5_comparison.png", dpi=180)
    plt.close(fig)


def _render_angle_response(rows: dict[float, dict[str, dict[str, Any]]]) -> None:
    angles = np.asarray(ANGLES, dtype=float)
    names = ("OpenFOAM", "Warp analytic-air", "Warp teacher-forced-U")
    colors = ("#2166ac", "#e66101", "#5e3c99")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for axis, key, ylabel in zip(
        axes,
        ("centroid_u_m", "sigma_major_m", "transfer_efficiency"),
        ("centroid U [m]", "sigma major [m]", "transfer efficiency"),
    ):
        for name, color in zip(names, colors):
            label_key = {"OpenFOAM": "openfoam", "Warp analytic-air": "analytic_air", "Warp teacher-forced-U": "teacher_forced_u"}[name]
            values = [rows[float(angle)][label_key][key] for angle in ANGLES]
            axis.plot(angles, values, "o-", color=color, label=name)
        axis.set_xlabel("incidence [deg]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("W1.1 angle response: teacher-forced vector field")
    fig.savefig(MEDIA_DIR / "warp_w1_1_angle_response.png", dpi=180)
    plt.close(fig)


def run_diagnostic(*, device: str = "cuda:0") -> dict[str, Any]:
    DIAGNOSTIC_ROOT.mkdir(parents=True, exist_ok=True)
    grids = _load_teacher_grids()
    manifest = {
        "schema_version": "warp_w1_1_teacher_flow_manifest_v1",
        "mode": "TEACHER_FORCED_U",
        "description": "Diagnostic teacher-forced carrier input; not a deployable runtime air model.",
        "openfoam_version_required": "v2606",
        "cases": {},
        "structured_grid_checks": {
            "cartesian_regular": True,
            "c_u_ordering_reconstructed_by_coordinate_lookup": True,
            "outside_domain_policy": "explicit teacher_out_of_field escape",
            "boundary_policy": "finite-volume half-cell bounds with linear first/last-centre extension",
        },
    }
    for angle, grid in grids.items():
        npz_name = f"teacher_u_{'0deg' if angle == 0.0 else '7p5deg' if angle == 7.5 else '15deg'}.npz"
        entry = grid.manifest_entry(DIAGNOSTIC_ROOT / npz_name)
        entry["npz_path"] = npz_name
        manifest["cases"][f"{angle:g}_deg"] = entry
    sampler_report = _sampler_agreement(grids, device=device)
    manifest["cpu_vs_warp_sampler"] = sampler_report
    _write_json(DIAGNOSTIC_ROOT / "teacher_flow_manifest.json", manifest)

    air_parameters = _load_air_parameters()
    rho, mu = _load_carrier_properties()
    analytic_results: dict[float, WarpCaseResult] = {}
    teacher_results: dict[float, WarpCaseResult] = {}
    for angle in ANGLES:
        config = _make_config(angle, air_parameters, rho, mu)
        print(f"WARP_W1_1_RUN mode=ANALYTIC_AIR angle={angle:g} particles={config.particle_count}", flush=True)
        analytic_results[angle] = run_warp_case(config, device=device, carrier_mode="ANALYTIC_AIR")
        print(f"WARP_W1_1_RUN mode=TEACHER_FORCED_U angle={angle:g} particles={config.particle_count}", flush=True)
        teacher_results[angle] = run_warp_case(config, device=device, carrier_mode="TEACHER_FORCED_U", teacher_flow=grids[angle])

    case_payloads: dict[float, dict[str, Any]] = {}
    angle_rows: dict[float, dict[str, dict[str, Any]]] = {}
    for angle in ANGLES:
        teacher_map, teacher_metrics = _load_teacher_map(angle)
        analytic_summary = _result_summary(angle, analytic_results[angle], teacher_map, teacher_metrics, None)
        teacher_summary = _result_summary(angle, teacher_results[angle], teacher_map, teacher_metrics, None)
        openfoam_row = {
            "deposited_mass_kg": float(teacher_metrics["mass_ledger"]["deposited_kg"]),
            "transfer_efficiency": float(teacher_metrics["mass_ledger"]["transfer_efficiency"]),
            "centroid_u_m": float(teacher_metrics["deposition"]["centroid_u_m"]),
            "centroid_v_m": float(teacher_metrics["deposition"]["centroid_v_m"]),
            "sigma_major_m": float(teacher_metrics["deposition"]["sigma_major_m"]),
            "sigma_minor_m": float(teacher_metrics["deposition"]["sigma_minor_m"]),
            "covariance_uv_m2": teacher_metrics["deposition"]["covariance_uv_m2"],
            "peak_density_kg_m2": float(teacher_metrics["deposition"]["peak_areal_mass_kg_m2"]),
            "escaped_mass_kg": float(teacher_metrics["mass_ledger"]["escaped_kg"]),
        }
        analytic_row = _metric_row(analytic_summary)
        teacher_row = _metric_row(teacher_summary)
        angle_rows[angle] = {"openfoam": openfoam_row, "analytic_air": analytic_row, "teacher_forced_u": teacher_row}
        case_payloads[angle] = {
            "schema_version": "warp_w1_1_case_metrics_v1",
            "incidence_angle_deg": angle,
            "carrier_modes": {
                "analytic_air": analytic_summary,
                "teacher_forced_u": teacher_summary,
            },
            "openfoam_reference": {
                "case": CASE_LABELS[angle],
                "metrics_path": str((RESULT_DIR / CASE_LABELS[angle] / "metrics.json").relative_to(ROOT)),
                "metrics": teacher_metrics,
            },
            "teacher_forced_minus_analytic": _delta(analytic_row, teacher_row),
            "teacher_forced_gate": _teacher_gate(teacher_summary),
            "analytic_w1_artifact_path": str((W1_ROOT / f"case_{'0deg' if angle == 0.0 else '7p5deg' if angle == 7.5 else '15deg'}_metrics.json").relative_to(ROOT)),
        }
        label = "0deg" if angle == 0.0 else "7p5deg" if angle == 7.5 else "15deg"
        _write_json(DIAGNOSTIC_ROOT / f"case_{label}_teacher_u_metrics.json", case_payloads[angle])

    holdout = case_payloads[7.5]["carrier_modes"]["teacher_forced_u"]
    teacher_gate = case_payloads[7.5]["teacher_forced_gate"]
    all_gate_pass = bool(all(teacher_gate.values()))
    if all_gate_pass:
        status = "W1_TRANSPORT_CORE_VALIDATED_WITH_TEACHER_U"
        primary_blocker = "PRIMARY_BLOCKER = ANALYTIC_CARRIER_FIELD"
    else:
        status = "W1_TRANSPORT_CORE_NOT_VALIDATED"
        primary_blocker = "PRIMARY_BLOCKER = PARTICLE_TRANSPORT_CORE"

    performance_rows = {}
    for angle in ANGLES:
        analytic_performance = analytic_results[angle].performance
        teacher_performance = teacher_results[angle].performance
        performance_rows[f"{angle:g}_deg"] = {
            "analytic_air": analytic_performance,
            "teacher_forced_u": teacher_performance,
            "mean_step_overhead_seconds": teacher_performance["mean_step_compute_seconds"] - analytic_performance["mean_step_compute_seconds"],
            "p95_step_overhead_seconds": teacher_performance["p95_step_compute_seconds"] - analytic_performance["p95_step_compute_seconds"],
            "wall_overhead_seconds": teacher_performance["wall_seconds"] - analytic_performance["wall_seconds"],
        }

    summary = {
        "schema_version": "warp_w1_1_diagnostic_summary_v1",
        "result": "PASS" if all_gate_pass else "PARTIAL",
        "status": status,
        "primary_blocker": primary_blocker,
        "diagnostic_interpretation": "teacher-forced U isolates the particle transport subsystem; 7.5-degree deposition remains a comparison target and was not used for fitting",
        "baseline": {
            "starting_commit": "519ee5425142c7c86056ece9dcc411e47474e3aa",
            "warp": "1.13.0",
            "gpu": "cuda:0 / NVIDIA GeForce RTX 4070 Ti",
            "particles_per_bin": HIGH_PARTICLES_PER_BIN,
            "total_particles": HIGH_PARTICLES_PER_BIN * 5,
            "dt_s": 5.0e-5,
            "steps": 1200,
        },
        "teacher_flow": manifest,
        "original_w1_7p5_gates_with_teacher_u": teacher_gate,
        "7p5_comparison": {
            "openfoam": angle_rows[7.5]["openfoam"],
            "warp_analytic_air": angle_rows[7.5]["analytic_air"],
            "warp_teacher_forced_u": angle_rows[7.5]["teacher_forced_u"],
            "teacher_forced_gate_metrics": holdout["comparison"]["openfoam_teacher"],
        },
        "15deg_diagnostic": {
            "openfoam_centroid_u_m": angle_rows[15.0]["openfoam"]["centroid_u_m"],
            "analytic_warp_centroid_u_m": angle_rows[15.0]["analytic_air"]["centroid_u_m"],
            "teacher_u_warp_centroid_u_m": angle_rows[15.0]["teacher_forced_u"]["centroid_u_m"],
            "openfoam_escaped_mass_kg": angle_rows[15.0]["openfoam"]["escaped_mass_kg"],
            "analytic_warp_escaped_mass_kg": angle_rows[15.0]["analytic_air"]["escaped_mass_kg"],
            "teacher_u_warp_escaped_mass_kg": angle_rows[15.0]["teacher_forced_u"]["escaped_mass_kg"],
        },
        "angle_response": angle_rows,
        "performance": performance_rows,
        "artifacts": {
            "teacher_flow_manifest": str((DIAGNOSTIC_ROOT / "teacher_flow_manifest.json").relative_to(ROOT)),
            "case_metrics": [str((DIAGNOSTIC_ROOT / f"case_{label}_teacher_u_metrics.json").relative_to(ROOT)) for label in ("0deg", "7p5deg", "15deg")],
            "plots": [
                "media/air_assisted_spray/warp_w1_1_teacher_u_7p5_comparison.png",
                "media/air_assisted_spray/warp_w1_1_angle_response.png",
            ],
        },
    }
    _write_json(DIAGNOSTIC_ROOT / "diagnostic_summary.json", summary)

    map_7p5, _ = _load_teacher_map(7.5)
    uv = np.column_stack((map_7p5["u_m"], map_7p5["v_m"]))
    _render_7p5(
        map_7p5["areal_mass_kg_m2"],
        analytic_results[7.5].map_density_kg_m2,
        teacher_results[7.5].map_density_kg_m2,
        uv,
    )
    _render_angle_response(angle_rows)
    print(json.dumps({"result": summary["result"], "status": status, "teacher_gate": teacher_gate, "diagnosis": primary_blocker}, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    summary = run_diagnostic(device=args.device)
    print(f"WARP_W1_1_TOTAL_WALL_SECONDS {time.perf_counter() - started:.3f}")
    if summary["status"] == "W1_TRANSPORT_CORE_NOT_VALIDATED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
