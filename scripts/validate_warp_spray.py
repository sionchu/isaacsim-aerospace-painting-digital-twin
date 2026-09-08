"""Validate the static NVIDIA Warp W1 spray transport against the v2606 teacher.

The script reads the already-solved canonical 0°, 7.5°, and 15° OpenFOAM
cases, fits only the carrier-air velocity field from the 0°/15° solutions,
then executes deterministic low/high CUDA particle ensembles.  The 7.5°
deposition case is held out from the air-field fit and is used only for the
transport comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerospace_painting.s2_runtime import S2RuntimeModel
from aerospace_painting.warp_air_field import AirFieldParameters, axial_speed_local, frame_axes, local_coordinates
from aerospace_painting.warp_spray import WarpCaseConfig, WarpCaseResult, run_warp_case


CASE_LABELS = {
    0.0: "medium",
    7.5: "medium_incidence_7p5deg",
    15.0: "medium_incidence_15deg",
}
CASE_DIR = ROOT / "reference_cfd" / "openfoam_v2606" / "flat_plate"
RESULT_DIR = ROOT / "results" / "air_assisted" / "openfoam_v2606" / "flat_plate"
WARP_RESULT_DIR = ROOT / "results" / "air_assisted" / "warp_w1"
MODEL_PATH = ROOT / "models" / "warp_air_field_v1.json"
MEDIA_DIR = ROOT / "media" / "air_assisted_spray"
EXPECTED_BINS_M = np.asarray((4.0e-5, 7.0e-5, 1.0e-4, 1.5e-4, 2.2e-4), dtype=float)
EXPECTED_FRACTIONS = np.asarray((0.10, 0.20, 0.30, 0.25, 0.15), dtype=float)
EXPECTED_MASS_KG = 1.0e-6
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _balanced(text: str, start: int, opening: str = "(", closing: str = ")") -> str:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise ValueError("unclosed OpenFOAM list")


def _read_internal_vectors(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = text.find("internalField")
    if marker < 0:
        raise ValueError(f"missing internalField in {path}")
    match = re.search(r"nonuniform\s+List<vector>\s+\d+\s*\(", text[marker:])
    if match is None:
        uniform = re.search(r"uniform\s*\(([^)]+)\)", text[marker:])
        if uniform is None:
            raise ValueError(f"unsupported vector field in {path}")
        values = re.findall(rf"({FLOAT})", uniform.group(1))
        return np.asarray([values], dtype=float)
    opening = marker + match.end() - 1
    values = re.findall(rf"\(\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*\)", _balanced(text, opening))
    return np.asarray(values, dtype=float)


def _read_uniform_scalar(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"internalField\s+uniform\s+({FLOAT})", text)
    if match is None:
        raise ValueError(f"missing uniform scalar in {path}")
    return float(match.group(1))


def _latest_time(case_dir: Path) -> Path:
    candidates = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            candidates.append((float(child.name), child))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"no solved time directory in {case_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _load_case_velocity(angle: float) -> tuple[np.ndarray, np.ndarray, Path]:
    case_dir = CASE_DIR / CASE_LABELS[angle]
    time_dir = _latest_time(case_dir)
    centres = _read_internal_vectors(time_dir / "C")
    velocity = _read_internal_vectors(time_dir / "U")
    if centres.shape != velocity.shape or centres.shape[1] != 3:
        raise ValueError(f"C/U shape mismatch in {case_dir}")
    return centres, velocity, time_dir


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if np.std(a) <= 1.0e-15 or np.std(b) <= 1.0e-15:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _air_model_values(local_points: np.ndarray, values: np.ndarray) -> np.ndarray:
    parameters = AirFieldParameters(*[float(x) for x in values])
    return axial_speed_local(local_points, parameters)


def _fit_air_field(samples: dict[float, tuple[np.ndarray, np.ndarray]]) -> tuple[AirFieldParameters, dict[str, Any]]:
    """Fit one shared Gaussian-jet model to solved carrier velocity samples."""

    local_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    per_case: dict[str, Any] = {}
    for angle in (0.0, 15.0):
        centres, velocity = samples[angle]
        local = local_coordinates(centres, angle)
        _, _, z_axis = frame_axes(angle)
        axial = velocity @ z_axis
        local_parts.append(local)
        target_parts.append(np.maximum(axial, 0.0))
    local_all = np.concatenate(local_parts, axis=0)
    target_all = np.concatenate(target_parts, axis=0)
    # A fixed stride keeps fitting deterministic and avoids making the fit
    # dependent on the number of OpenFOAM cells in a future mesh refresh.
    stride = max(1, int(math.ceil(len(target_all) / 12000)))
    local_fit = local_all[::stride]
    target_fit = target_all[::stride]

    bounds = np.asarray(((10.0, 40.0), (0.01, 0.8), (0.0, 12.0), (0.01, 3.0), (0.5, 3.0)), dtype=float)
    parameters = np.asarray((18.0, 0.12, 3.0, 0.5, 2.0), dtype=float)
    steps = np.asarray((6.0, 0.08, 2.0, 0.3, 0.5), dtype=float)

    def objective(candidate: np.ndarray) -> float:
        predicted = _air_model_values(local_fit, candidate)
        return float(np.mean((predicted - target_fit) ** 2))

    best = objective(parameters)
    for _ in range(12):
        for index in range(parameters.size):
            candidates = []
            for delta in (-steps[index], 0.0, steps[index]):
                candidate = parameters.copy()
                candidate[index] = np.clip(candidate[index] + delta, bounds[index, 0], bounds[index, 1])
                candidates.append((objective(candidate), candidate))
            candidate_best, value_best = min(candidates, key=lambda item: item[0])
            if candidate_best < best:
                best = candidate_best
                parameters = value_best
        steps *= 0.5

    fitted = AirFieldParameters(*[float(value) for value in parameters])
    for angle in (0.0, 15.0):
        centres, velocity = samples[angle]
        local = local_coordinates(centres, angle)
        _, _, z_axis = frame_axes(angle)
        actual = np.maximum(velocity @ z_axis, 0.0)
        predicted = axial_speed_local(local, fitted)
        per_case[f"{angle:g}_deg"] = {
            "sample_count": int(actual.size),
            "fit_rmse_m_s": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "fit_pearson_correlation": _pearson(predicted, actual),
            "actual_axial_speed_min_m_s": float(actual.min()),
            "actual_axial_speed_max_m_s": float(actual.max()),
            "actual_axial_speed_mean_m_s": float(actual.mean()),
        }
    return fitted, {
        "fit_method": "deterministic_coordinate_descent",
        "fit_targets": "solved OpenFOAM carrier axial velocity from C/U only",
        "downsample_stride": stride,
        "combined_sample_count": int(target_all.size),
        "fit_sample_count": int(target_fit.size),
        "per_case": per_case,
    }


def _teacher_properties() -> dict[str, Any]:
    medium = CASE_DIR / "medium"
    fractions = {
        "N2": _read_uniform_scalar(medium / "0" / "N2"),
        "O2": _read_uniform_scalar(medium / "0" / "O2"),
    }
    temperature = _read_uniform_scalar(medium / "0" / "T")
    pressure = _read_uniform_scalar(medium / "0" / "p")
    transport_text = (medium / "chemkin" / "transportProperties").read_text(encoding="utf-8")
    as_match = re.search(rf"\bAs\s+({FLOAT})", transport_text)
    ts_match = re.search(rf"\bTs\s+({FLOAT})", transport_text)
    if as_match is None or ts_match is None:
        raise ValueError("Sutherland coefficients missing from teacher transportProperties")
    sutherland = {"As": float(as_match.group(1)), "Ts": float(ts_match.group(1))}
    molecular_weights = {"N2": 28.0134e-3, "O2": 31.9988e-3}
    # The OpenFOAM N2/O2 fields are mass fractions (Y fields), so use the
    # reciprocal mixture rule rather than treating them as mole fractions.
    molar_mass = 1.0 / sum(fractions[key] / molecular_weights[key] for key in fractions)
    density = pressure * molar_mass / (8.31446261815324 * temperature)
    viscosity = sutherland["As"] * temperature ** 1.5 / (temperature + sutherland["Ts"])
    files = [medium / "0" / key for key in ("N2", "O2", "T", "p")]
    files += [medium / "chemkin" / "transportProperties", medium / "constant" / "thermophysicalProperties", medium / "constant" / "g"]
    return {
        "composition_mass_or_mole_fields": fractions,
        "temperature_K": temperature,
        "pressure_Pa": pressure,
        "molecular_weights_kg_per_mol": molecular_weights,
        "mixture_molar_mass_kg_per_mol": molar_mass,
        "ideal_gas_density_kg_m3": density,
        "sutherland": sutherland,
        "dynamic_viscosity_pa_s": viscosity,
        "liquid_density_kg_m3": 1000.0,
        "gravity_m_s2": [0.0, 0.0, -9.81],
        "source_files": {str(path.relative_to(ROOT)): _sha256_file(path) for path in files},
    }


def _load_teacher_map(angle: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    label = CASE_LABELS[angle]
    result_dir = RESULT_DIR / label
    with np.load(result_dir / "deposition_map.npz") as data:
        map_data = {key: np.asarray(data[key], dtype=float) for key in data.files}
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    return map_data, metrics


def _map_moments(uv: np.ndarray, density: np.ndarray, area: np.ndarray | float) -> dict[str, Any]:
    points = np.asarray(uv, dtype=float)
    values = np.asarray(density, dtype=float)
    cell_area = np.broadcast_to(np.asarray(area, dtype=float), values.shape)
    weights = values * cell_area
    mass = float(weights.sum())
    if mass <= 0.0:
        return {
            "mass_kg": 0.0,
            "centroid_uv_m": [0.0, 0.0],
            "covariance_uv_m2": [[0.0, 0.0], [0.0, 0.0]],
            "sigma_major_m": 0.0,
            "sigma_minor_m": 0.0,
            "peak_density_kg_m2": 0.0,
        }
    centroid = np.average(points, axis=0, weights=weights)
    centered = points - centroid
    covariance = (centered * weights[:, None]).T @ centered / mass
    eig = np.maximum(np.linalg.eigvalsh((covariance + covariance.T) * 0.5), 0.0)
    return {
        "mass_kg": mass,
        "centroid_uv_m": centroid.tolist(),
        "covariance_uv_m2": covariance.tolist(),
        "sigma_major_m": float(np.sqrt(eig[-1])),
        "sigma_minor_m": float(np.sqrt(eig[0])),
        "peak_density_kg_m2": float(values.max()),
    }


def _field_comparison(predicted: np.ndarray, actual: np.ndarray, uv: np.ndarray, area: np.ndarray | float) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    predicted_moments = _map_moments(uv, predicted, area)
    actual_moments = _map_moments(uv, actual, area)
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    actual_range = float(actual.max() - actual.min())
    return {
        "predicted": predicted_moments,
        "actual": actual_moments,
        "integrated_mass_error_kg": predicted_moments["mass_kg"] - actual_moments["mass_kg"],
        "integrated_mass_relative_error": abs(predicted_moments["mass_kg"] - actual_moments["mass_kg"]) / max(actual_moments["mass_kg"], 1.0e-30),
        "centroid_error_m": float(np.linalg.norm(np.asarray(predicted_moments["centroid_uv_m"]) - np.asarray(actual_moments["centroid_uv_m"]))),
        "sigma_major_relative_error": abs(predicted_moments["sigma_major_m"] - actual_moments["sigma_major_m"]) / max(actual_moments["sigma_major_m"], 1.0e-30),
        "sigma_minor_relative_error": abs(predicted_moments["sigma_minor_m"] - actual_moments["sigma_minor_m"]) / max(actual_moments["sigma_minor_m"], 1.0e-30),
        "rmse_kg_m2": rmse,
        "nrmse": rmse / actual_range if actual_range > 0.0 else 0.0,
        "pearson_correlation": _pearson(predicted, actual),
        "integrated_absolute_mass_density_error_kg": float(np.sum(np.abs(predicted - actual) * np.asarray(area))),
    }


def _result_summary(angle: float, result: WarpCaseResult, teacher_map: dict[str, np.ndarray], teacher_metrics: dict[str, Any], s2_density: np.ndarray | None) -> dict[str, Any]:
    uv = np.column_stack((teacher_map["u_m"], teacher_map["v_m"]))
    area = teacher_map["area_m2"]
    teacher_density = teacher_map["areal_mass_kg_m2"]
    warp_field = _field_comparison(result.map_density_kg_m2, teacher_density, uv, area)
    teacher_ledger = teacher_metrics["mass_ledger"]
    teacher_dep = teacher_metrics["deposition"]
    warp_moments = _map_moments(uv, result.map_density_kg_m2, area)
    deposited_reference = float(teacher_ledger["deposited_kg"])
    warp_moments["deposited_mass_relative_error"] = abs(warp_moments["mass_kg"] - deposited_reference) / max(deposited_reference, 1.0e-30)
    warp_moments["centroid_error_m"] = float(np.linalg.norm(np.asarray(warp_moments["centroid_uv_m"]) - np.asarray((teacher_dep["centroid_u_m"], teacher_dep["centroid_v_m"]))))
    warp_moments["sigma_major_relative_error"] = abs(warp_moments["sigma_major_m"] - float(teacher_dep["sigma_major_m"])) / float(teacher_dep["sigma_major_m"])
    warp_moments["sigma_minor_relative_error"] = abs(warp_moments["sigma_minor_m"] - float(teacher_dep["sigma_minor_m"])) / float(teacher_dep["sigma_minor_m"])
    comparison = {"openfoam_teacher": warp_field}
    if s2_density is not None:
        comparison["s2_holdout"] = _field_comparison(s2_density, teacher_density, uv, area)
        comparison["warp_vs_s2"] = _field_comparison(result.map_density_kg_m2, s2_density, uv, area)
    return {
        "schema_version": "warp_w1_case_metrics_v1",
        "incidence_angle_deg": angle,
        "particle_count": result.particle_count,
        "particle_count_per_bin": result.particle_count_per_bin,
        "mass_ledger": result.mass_ledger,
        "deposition": warp_moments,
        "comparison": comparison,
        "performance": result.performance,
        "state_counts": {
            "released": int(np.count_nonzero(result.released_flags)),
            "deposited": int(np.count_nonzero(result.deposited_flags)),
            "escaped": int(np.count_nonzero(result.escaped_flags)),
            "alive": int(np.count_nonzero(result.alive_flags)),
        },
        "teacher_reference": {
            "case": CASE_LABELS[angle],
            "deposited_mass_kg": deposited_reference,
            "centroid_uv_m": [teacher_dep["centroid_u_m"], teacher_dep["centroid_v_m"]],
            "sigma_major_m": teacher_dep["sigma_major_m"],
            "sigma_minor_m": teacher_dep["sigma_minor_m"],
        },
    }


def _sensitivity(low: dict[str, Any], high: dict[str, Any], physical_response: dict[str, float]) -> dict[str, Any]:
    low_dep = low["deposition"]
    high_dep = high["deposition"]
    centroid_delta = float(np.linalg.norm(np.asarray(high_dep["centroid_uv_m"]) - np.asarray(low_dep["centroid_uv_m"])))
    major_delta = abs(high_dep["sigma_major_m"] - low_dep["sigma_major_m"])
    minor_delta = abs(high_dep["sigma_minor_m"] - low_dep["sigma_minor_m"])
    mass_delta = abs(high_dep["mass_kg"] - low_dep["mass_kg"])
    limits = {
        "centroid_m": max(physical_response["centroid_shift_m"], 1.0e-12),
        "sigma_major_m": max(physical_response["sigma_major_shift_m"], 1.0e-12),
        "sigma_minor_m": max(physical_response["sigma_minor_shift_m"], 1.0e-12),
        "mass_kg": max(physical_response["mass_shift_kg"], EXPECTED_MASS_KG * 1.0e-6),
    }
    return {
        "low_particle_count": low["particle_count"],
        "high_particle_count": high["particle_count"],
        "absolute_deltas": {
            "centroid_m": centroid_delta,
            "sigma_major_m": major_delta,
            "sigma_minor_m": minor_delta,
            "deposited_mass_kg": mass_delta,
        },
        "physical_0_to_15_response": physical_response,
        "numerical_sensitivity_smaller_than_physical_response": {
            "centroid": centroid_delta < limits["centroid_m"],
            "sigma_major": major_delta < limits["sigma_major_m"],
            "sigma_minor": minor_delta < limits["sigma_minor_m"],
            "deposited_mass": mass_delta < limits["mass_kg"],
        },
    }


def _render_air_fit(samples: dict[float, tuple[np.ndarray, np.ndarray]], parameters: AirFieldParameters) -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, angle in zip(axes, (0.0, 15.0)):
        centres, velocity = samples[angle]
        local = local_coordinates(centres, angle)
        _, _, z_axis = frame_axes(angle)
        actual = np.maximum(velocity @ z_axis, 0.0)
        predicted = axial_speed_local(local, parameters)
        stride = max(1, len(actual) // 8000)
        axis.scatter(actual[::stride], predicted[::stride], s=2, alpha=0.25, color="#2166ac")
        max_value = max(float(actual.max()), float(predicted.max()))
        axis.plot((0.0, max_value), (0.0, max_value), color="#222222", linewidth=1.0)
        axis.set_title(f"{angle:g}° solved carrier")
        axis.set_xlabel("OpenFOAM axial speed [m/s]")
        axis.set_ylabel("Warp fit axial speed [m/s]")
        axis.set_aspect("equal", adjustable="box")
    fig.suptitle("W1 Gaussian external-air fit (carrier velocity only)")
    fig.savefig(MEDIA_DIR / "warp_w1_air_field_fit.png", dpi=180)
    plt.close(fig)


def _render_holdout(comparison: dict[str, Any], uv: np.ndarray, actual: np.ndarray, warp: np.ndarray, s2: np.ndarray) -> None:
    vmax = max(float(actual.max()), float(warp.max()), float(s2.max()))
    error = np.abs(warp - actual)
    norm = plt.Normalize(vmin=0.0, vmax=vmax if vmax > 0.0 else 1.0)
    fields = (("OpenFOAM 7.5°", actual, norm), ("Warp HIGH", warp, norm), ("S2 frozen", s2, norm), ("Warp absolute error", error, plt.Normalize(vmin=0.0, vmax=max(float(error.max()), 1.0e-12))))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True, sharex=True, sharey=True)
    for axis, (label, values, field_norm) in zip(axes, fields):
        scatter = axis.scatter(uv[:, 0], uv[:, 1], c=values, s=22, marker="s", cmap="viridis", norm=field_norm)
        axis.set_title(label)
        axis.set_aspect("equal")
        axis.set_xlim(-0.15, 0.15)
        axis.set_ylim(-0.15, 0.15)
        axis.set_xlabel("u [m]")
    axes[0].set_ylabel("v [m]")
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="areal mass / absolute error [kg m$^{-2}$]")
    fig.suptitle("W1 7.5° hold-out comparison")
    fig.savefig(MEDIA_DIR / "warp_w1_openfoam_7p5_comparison.png", dpi=180)
    plt.close(fig)


def _render_angle_response(case_metrics: dict[float, dict[str, Any]], teacher_metrics: dict[float, dict[str, Any]]) -> None:
    angles = np.asarray((0.0, 7.5, 15.0))
    warp_mass = np.asarray([case_metrics[float(angle)]["deposition"]["mass_kg"] for angle in angles]) / EXPECTED_MASS_KG
    teacher_mass = np.asarray([teacher_metrics[float(angle)]["mass_ledger"]["deposited_kg"] for angle in angles]) / EXPECTED_MASS_KG
    warp_centroid = np.asarray([case_metrics[float(angle)]["deposition"]["centroid_uv_m"][0] for angle in angles])
    teacher_centroid = np.asarray([teacher_metrics[float(angle)]["deposition"]["centroid_u_m"] for angle in angles])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].plot(angles, teacher_mass, "o-", label="OpenFOAM teacher")
    axes[0].plot(angles, warp_mass, "s--", label="Warp HIGH")
    axes[0].set_ylabel("deposited / injected mass")
    axes[0].set_xlabel("incidence [deg]")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(frameon=False)
    axes[1].plot(angles, teacher_centroid, "o-", label="OpenFOAM teacher")
    axes[1].plot(angles, warp_centroid, "s--", label="Warp HIGH")
    axes[1].set_ylabel("major-axis centroid u [m]")
    axes[1].set_xlabel("incidence [deg]")
    axes[1].legend(frameon=False)
    fig.suptitle("W1 static angle response")
    fig.savefig(MEDIA_DIR / "warp_w1_angle_response.png", dpi=180)
    plt.close(fig)


def _build_model(parameters: AirFieldParameters, fit_report: dict[str, Any], teacher: dict[str, Any], samples: dict[float, tuple[np.ndarray, np.ndarray]], time_dirs: dict[float, Path]) -> dict[str, Any]:
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for angle in (0.0, 15.0):
        label = CASE_LABELS[angle]
        time_dir = time_dirs[angle]
        for field in ("C", "U"):
            path = time_dir / field
            key = f"{label}_{field}"
            source_paths[key] = str(path.relative_to(ROOT))
            source_hashes[key] = _sha256_file(path)
    payload: dict[str, Any] = {
        "schema_version": "warp_air_field_v1",
        "model_id": "warp_air_field_v1",
        "model_type": "axisymmetric_gaussian_external_air_jet",
        "frame": {"world_axes": "+X major/u, +Y minor/v, +Z spray/stand-off", "rotation": "about world +Y"},
        "parameters": parameters.to_dict(),
        "fit_report": fit_report,
        "teacher_cases_used": ["medium", "medium_incidence_15deg"],
        "holdout_case": "medium_incidence_7p5deg",
        "teacher_properties": teacher,
        "source_paths": source_paths,
        "source_hashes": source_hashes,
        "source_case_time_dirs": {f"{angle:g}_deg": str(time_dirs[angle].relative_to(ROOT)) for angle in (0.0, 15.0)},
    }
    payload["model_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    _write_json(MODEL_PATH, payload)
    return payload


def _verify_model(payload: dict[str, Any]) -> None:
    recorded = payload.get("model_sha256", "")
    without = dict(payload)
    without.pop("model_sha256", None)
    if recorded != hashlib.sha256(_canonical_json(without)).hexdigest():
        raise ValueError("Warp air-field model hash mismatch")
    for key, expected in payload["source_hashes"].items():
        source = ROOT / payload["source_paths"][key]
        if _sha256_file(source) != expected:
            raise ValueError(f"Warp air-field source hash mismatch: {source}")


def run_validation(low_count: int, high_count: int, device: str) -> dict[str, Any]:
    samples: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    time_dirs: dict[float, Path] = {}
    for angle in (0.0, 15.0):
        centres, velocity, time_dir = _load_case_velocity(angle)
        samples[angle] = (centres, velocity)
        time_dirs[angle] = time_dir
    parameters, fit_report = _fit_air_field(samples)
    teacher = _teacher_properties()
    model_payload = _build_model(parameters, fit_report, teacher, samples, time_dirs)
    _verify_model(model_payload)
    _render_air_fit(samples, parameters)

    s2_model = S2RuntimeModel.load(ROOT / "models" / "s2_air_assisted_v1.json")
    s2_model.verify_sources(ROOT)
    s2_prediction = s2_model.predict(7.5)

    teacher_maps: dict[float, dict[str, np.ndarray]] = {}
    teacher_metrics: dict[float, dict[str, Any]] = {}
    for angle in (0.0, 7.5, 15.0):
        teacher_maps[angle], teacher_metrics[angle] = _load_teacher_map(angle)
    uv = np.column_stack((teacher_maps[7.5]["u_m"], teacher_maps[7.5]["v_m"]))
    s2_density = s2_prediction.density(uv)

    configs = {
        "low": {angle: WarpCaseConfig(angle, low_count, air_field=parameters, air_density_kg_m3=teacher["ideal_gas_density_kg_m3"], air_dynamic_viscosity_pa_s=teacher["dynamic_viscosity_pa_s"]) for angle in (0.0, 7.5, 15.0)},
        "high": {angle: WarpCaseConfig(angle, high_count, air_field=parameters, air_density_kg_m3=teacher["ideal_gas_density_kg_m3"], air_dynamic_viscosity_pa_s=teacher["dynamic_viscosity_pa_s"]) for angle in (0.0, 7.5, 15.0)},
    }
    results: dict[str, dict[float, WarpCaseResult]] = {"low": {}, "high": {}}
    for level in ("low", "high"):
        for angle in (0.0, 7.5, 15.0):
            print(f"WARP_RUN level={level} angle={angle:g} particles={configs[level][angle].particle_count}", flush=True)
            results[level][angle] = run_warp_case(configs[level][angle], device=device)

    case_payloads: dict[str, Any] = {}
    for angle in (0.0, 7.5, 15.0):
        s2 = s2_density if angle == 7.5 else None
        low_summary = _result_summary(angle, results["low"][angle], teacher_maps[angle], teacher_metrics[angle], s2)
        high_summary = _result_summary(angle, results["high"][angle], teacher_maps[angle], teacher_metrics[angle], s2)
        teacher_dep0 = teacher_metrics[0.0]["deposition"]
        teacher_dep15 = teacher_metrics[15.0]["deposition"]
        physical = {
            "centroid_shift_m": abs(float(teacher_dep15["centroid_u_m"]) - float(teacher_dep0["centroid_u_m"])),
            "sigma_major_shift_m": abs(float(teacher_dep15["sigma_major_m"]) - float(teacher_dep0["sigma_major_m"])),
            "sigma_minor_shift_m": abs(float(teacher_dep15["sigma_minor_m"]) - float(teacher_dep0["sigma_minor_m"])),
            "mass_shift_kg": abs(float(teacher_metrics[15.0]["mass_ledger"]["deposited_kg"]) - float(teacher_metrics[0.0]["mass_ledger"]["deposited_kg"])),
        }
        case_payloads[angle] = {
            "schema_version": "warp_w1_case_metrics_v1",
            "incidence_angle_deg": angle,
            "low": low_summary,
            "high": high_summary,
            "sensitivity": _sensitivity(low_summary, high_summary, physical),
        }
        label = "0deg" if angle == 0.0 else "7p5deg" if angle == 7.5 else "15deg"
        _write_json(WARP_RESULT_DIR / f"case_{label}_metrics.json", case_payloads[angle])

    high_summaries = {angle: case_payloads[angle]["high"] for angle in (0.0, 7.5, 15.0)}
    low_summaries = {angle: case_payloads[angle]["low"] for angle in (0.0, 7.5, 15.0)}
    balance_pass = all(abs(high_summaries[angle]["mass_ledger"]["relative_balance_error"]) <= 1.0e-5 for angle in (0.0, 7.5, 15.0))
    holdout = high_summaries[7.5]
    holdout_dep = holdout["deposition"]
    holdout_field = holdout["comparison"]["openfoam_teacher"]
    moments_pass = (
        holdout_dep["deposited_mass_relative_error"] <= 0.10
        and holdout_dep["centroid_error_m"] <= 0.010
        and holdout_dep["sigma_major_relative_error"] <= 0.15
        and holdout_dep["sigma_minor_relative_error"] <= 0.15
    )
    field_pass = holdout_field["pearson_correlation"] >= 0.80 and holdout_field["nrmse"] <= 0.40
    sensitivity_pass = all(
        all(item["sensitivity"]["numerical_sensitivity_smaller_than_physical_response"].values())
        for item in case_payloads.values()
    )
    if balance_pass and moments_pass and field_pass and sensitivity_pass:
        status = "WARP_W1_VALIDATED"
    elif balance_pass and moments_pass and sensitivity_pass:
        status = "WARP_MOMENTS_VALID_FIELD_POOR"
    else:
        status = "WARP_TRANSPORT_NOT_VALIDATED"

    environment = {
        "schema_version": "warp_w1_environment_v1",
        "warp_version": str(__import__("warp").__version__),
        "warp_devices": [str(device) for device in __import__("warp").get_devices()],
        "selected_device": device,
        "cuda_device_count": int(__import__("warp").get_cuda_device_count()),
        "teacher_openfoam_version": "v2606",
        "teacher_case_time_dirs": {f"{angle:g}_deg": str(_latest_time(CASE_DIR / CASE_LABELS[angle]).relative_to(ROOT)) for angle in (0.0, 7.5, 15.0)},
        "teacher_settings": {
            "stand_off_m": 0.24,
            "nozzle_position_m": [0.0, 0.0, 0.002],
            "droplet_speed_m_s": 12.0,
            "outer_diameter_m": 8.0e-4,
            "cone_inner_half_angle_deg": 0.0,
            "cone_outer_half_angle_deg": 22.0,
            "injection_duration_s": 0.01,
            "droplet_bins_m": EXPECTED_BINS_M.tolist(),
            "mass_fractions": EXPECTED_FRACTIONS.tolist(),
            "mass_total_kg": EXPECTED_MASS_KG,
            "gravity_m_s2": teacher["gravity_m_s2"],
            "wall_interaction": "targetWall stick; airInlet/farField escape",
        },
        "carrier_properties": teacher,
        "air_field_model": {
            "path": str(MODEL_PATH.relative_to(ROOT)),
            "model_sha256": model_payload["model_sha256"],
            "parameters": parameters.to_dict(),
            "fit_report": fit_report,
        },
        "solver_state_arrays": ["position", "previous_position", "velocity", "diameter", "represented_mass", "bin_id", "alive", "released", "deposited", "escaped"],
        "dynamics": {
            "drag": "sphere Schiller-Naumann",
            "gravity": True,
            "integrator": "explicit Euler",
            "collision": "Warp mesh_query_ray segment query",
            "map_accumulation": "CUDA atomic_add kg/m2",
            "stokes_response_time_smallest_bin_s": float(teacher["liquid_density_kg_m3"] * EXPECTED_BINS_M.min() ** 2 / (18.0 * teacher["dynamic_viscosity_pa_s"])),
            "dt_to_smallest_response_time_ratio": float(5.0e-5 / (teacher["liquid_density_kg_m3"] * EXPECTED_BINS_M.min() ** 2 / (18.0 * teacher["dynamic_viscosity_pa_s"]))),
        },
        "canonical_high_particle_count_per_bin": high_count,
        "low_particle_count_per_bin": low_count,
        "command": f"Isaac Sim Python launcher scripts/validate_warp_spray.py --low-particles-per-bin {low_count} --high-particles-per-bin {high_count}",
    }
    _write_json(WARP_RESULT_DIR / "environment.json", environment)
    sensitivity_payload = {
        "schema_version": "warp_w1_particle_sensitivity_v1",
        "canonical_high_particle_count_per_bin": high_count,
        "low_particle_count_per_bin": low_count,
        "cases": {str(angle): case_payloads[angle]["sensitivity"] for angle in (0.0, 7.5, 15.0)},
        "mass_balance_pass": balance_pass,
        "sensitivity_pass": sensitivity_pass,
    }
    _write_json(WARP_RESULT_DIR / "particle_sensitivity.json", sensitivity_payload)
    validation = {
        "schema_version": "warp_w1_validation_summary_v1",
        "status": status,
        "acceptance": {
            "relative_mass_balance_max": 1.0e-5,
            "holdout_deposited_mass_relative_error_max": 0.10,
            "holdout_centroid_error_m_max": 0.010,
            "holdout_sigma_relative_error_max": 0.15,
            "holdout_field_pearson_min": 0.80,
            "holdout_field_nrmse_max": 0.40,
            "sensitivity_smaller_than_physical_response": True,
        },
        "gates": {
            "mass_balance": balance_pass,
            "holdout_moments": moments_pass,
            "holdout_field": field_pass,
            "particle_sensitivity": sensitivity_pass,
        },
        "holdout_7p5": {
            "moments": holdout_dep,
            "field": holdout_field,
            "s2_frozen_comparison": holdout["comparison"].get("s2_holdout"),
            "warp_vs_s2": holdout["comparison"].get("warp_vs_s2"),
        },
        "angle_response_high": {
            str(angle): {
                "warp_deposited_mass_kg": high_summaries[angle]["deposition"]["mass_kg"],
                "teacher_deposited_mass_kg": teacher_metrics[angle]["mass_ledger"]["deposited_kg"],
                "warp_centroid_u_m": high_summaries[angle]["deposition"]["centroid_uv_m"][0],
                "teacher_centroid_u_m": teacher_metrics[angle]["deposition"]["centroid_u_m"],
            }
            for angle in (0.0, 7.5, 15.0)
        },
        "model": {"path": str(MODEL_PATH.relative_to(ROOT)), "model_sha256": model_payload["model_sha256"]},
    }
    _write_json(WARP_RESULT_DIR / "validation_summary.json", validation)

    holdout_warp = results["high"][7.5].map_density_kg_m2
    _render_holdout(holdout["comparison"], uv, teacher_maps[7.5]["areal_mass_kg_m2"], holdout_warp, s2_density)
    _render_angle_response({angle: high_summaries[angle] for angle in (0.0, 7.5, 15.0)}, teacher_metrics)
    print(json.dumps({"status": status, "gates": validation["gates"], "holdout_field": holdout_field}, indent=2, sort_keys=True))
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-particles-per-bin", type=int, default=500)
    parser.add_argument("--high-particles-per-bin", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.low_particles_per_bin < 1 or args.high_particles_per_bin < args.low_particles_per_bin:
        raise SystemExit("particle counts must be positive and high >= low")
    started = time.perf_counter()
    result = run_validation(args.low_particles_per_bin, args.high_particles_per_bin, args.device)
    print(f"WARP_W1_TOTAL_WALL_SECONDS {time.perf_counter() - started:.3f}")
    if result["status"] == "WARP_TRANSPORT_NOT_VALIDATED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
