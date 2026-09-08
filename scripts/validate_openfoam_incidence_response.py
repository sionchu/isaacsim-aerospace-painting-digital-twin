"""Validate the v2606 15-degree incidence response against nominal medium.

The input comparison is intentionally strict: after normalising the two
requested changes, every tracked OpenFOAM input file must be byte-equivalent.
The response gate then compares the real incidence deposition map with both
nominal mesh levels so a mesh artifact is not mistaken for an angle response.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

try:
    from scripts.postprocess_openfoam_flat_plate import process_case, write_sensitivity
except ModuleNotFoundError:  # direct execution from the scripts directory
    from postprocess_openfoam_flat_plate import process_case, write_sensitivity


AIR_VECTOR_NOMINAL = (0.0, 0.0, 18.0)
AIR_VECTOR_INCIDENCE = (4.658742, 0.0, 17.386666)
SPRAY_VECTOR_NOMINAL = (0.0, 0.0, 1.0)
SPRAY_VECTOR_INCIDENCE = (0.258819, 0.0, 0.965926)
INPUT_DIRS = ("0", "constant", "system", "chemkin")
ROOT_INPUTS = ("Allrun", "Allclean")


def _input_files(case_dir: Path) -> list[str]:
    paths: set[str] = set(ROOT_INPUTS)
    for directory in INPUT_DIRS:
        root = case_dir / directory
        if root.exists():
            paths.update(
                path.relative_to(case_dir).as_posix()
                for path in root.rglob("*")
                if path.is_file()
                and path.relative_to(case_dir).as_posix() != "constant/polyMesh"
                and not path.relative_to(case_dir).as_posix().startswith("constant/polyMesh/")
            )
    return sorted(paths)


def _vector_tuple(raw: str) -> tuple[float, float, float]:
    values = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", raw)]
    if len(values) != 3:
        raise ValueError(f"expected a 3-vector, got {raw!r}")
    return tuple(values)  # type: ignore[return-value]


def _air_inlet_vector(text: str) -> tuple[float, float, float]:
    match = re.search(
        r"airInlet\s*\{.*?value\s+uniform\s*\(([^)]*)\)",
        text,
        flags=re.S,
    )
    if not match:
        raise ValueError("airInlet fixedValue vector not found")
    return _vector_tuple(match.group(1))


def _normalise_air_vector(text: str, vector: tuple[float, float, float]) -> str:
    pattern = re.compile(
        r"(airInlet\s*\{.*?value\s+uniform\s*)\([^)]*\)(\s*;)",
        flags=re.S,
    )
    replacement = "(%g %g %g)" % vector
    normalised, count = pattern.subn(lambda match: match.group(1) + replacement + match.group(2), text, count=1)
    if count != 1:
        raise ValueError("airInlet fixedValue vector could not be normalised")
    return normalised


def _spray_directions(text: str) -> list[tuple[float, float, float]]:
    return [_vector_tuple(raw) for raw in re.findall(r"\bdirection\s*\(([^)]*)\)", text)]


def _normalise_spray_directions(text: str, vector: tuple[float, float, float]) -> str:
    pattern = re.compile(r"(\bdirection\s*)\([^)]*\)(\s*;)")
    replacement = "(%g %g %g)" % vector
    return pattern.sub(lambda match: match.group(1) + replacement + match.group(2), text)


def _diff_lines(left: str, right: str, rel: str) -> list[str]:
    return list(
        difflib.unified_diff(
            left.splitlines(),
            right.splitlines(),
            fromfile=f"nominal/{rel}",
            tofile=f"incidence/{rel}",
            lineterm="",
        )
    )


def compare_incidence_inputs(nominal_case: Path, incidence_case: Path) -> dict:
    """Compare input dictionaries and return a machine-readable audit."""
    nominal_files = set(_input_files(nominal_case))
    incidence_files = set(_input_files(incidence_case))
    unexpected: list[dict] = []
    files_checked: list[str] = []

    for rel in sorted(nominal_files | incidence_files):
        files_checked.append(rel)
        nominal_path = nominal_case / rel
        incidence_path = incidence_case / rel
        if not nominal_path.exists() or not incidence_path.exists():
            unexpected.append({"file": rel, "reason": "missing counterpart"})
            continue
        nominal_text = nominal_path.read_text(encoding="utf-8", errors="replace")
        incidence_text = incidence_path.read_text(encoding="utf-8", errors="replace")

        if rel == "0/U":
            nominal_vector = _air_inlet_vector(nominal_text)
            incidence_vector = _air_inlet_vector(incidence_text)
            if not np.allclose(nominal_vector, AIR_VECTOR_NOMINAL, rtol=0, atol=1e-9):
                unexpected.append({"file": rel, "reason": "unexpected nominal air vector", "actual": nominal_vector})
            if not np.allclose(incidence_vector, AIR_VECTOR_INCIDENCE, rtol=0, atol=1e-9):
                unexpected.append({"file": rel, "reason": "unexpected incidence air vector", "actual": incidence_vector})
            normalised_nominal = _normalise_air_vector(nominal_text, AIR_VECTOR_NOMINAL)
            normalised_incidence = _normalise_air_vector(incidence_text, AIR_VECTOR_NOMINAL)
            if normalised_nominal != normalised_incidence:
                unexpected.append({"file": rel, "reason": "unexpected non-airInlet difference", "diff": _diff_lines(normalised_nominal, normalised_incidence, rel)})
        elif rel == "constant/sprayCloudProperties":
            nominal_directions = _spray_directions(nominal_text)
            incidence_directions = _spray_directions(incidence_text)
            if len(nominal_directions) != 5:
                unexpected.append({"file": rel, "reason": "nominal injection direction count is not five", "count": len(nominal_directions)})
            if len(incidence_directions) != 5:
                unexpected.append({"file": rel, "reason": "incidence injection direction count is not five", "count": len(incidence_directions)})
            if nominal_directions and not np.allclose(nominal_directions, np.asarray(SPRAY_VECTOR_NOMINAL), rtol=0, atol=1e-9):
                unexpected.append({"file": rel, "reason": "unexpected nominal injection direction", "actual": nominal_directions})
            if incidence_directions and not np.allclose(incidence_directions, np.asarray(SPRAY_VECTOR_INCIDENCE), rtol=0, atol=1e-9):
                unexpected.append({"file": rel, "reason": "unexpected incidence injection direction", "actual": incidence_directions})
            normalised_nominal = _normalise_spray_directions(nominal_text, SPRAY_VECTOR_NOMINAL)
            normalised_incidence = _normalise_spray_directions(incidence_text, SPRAY_VECTOR_NOMINAL)
            if normalised_nominal != normalised_incidence:
                unexpected.append({"file": rel, "reason": "unexpected non-direction difference", "diff": _diff_lines(normalised_nominal, normalised_incidence, rel)})
        elif nominal_text != incidence_text:
            unexpected.append({"file": rel, "reason": "unexpected dictionary difference", "diff": _diff_lines(nominal_text, incidence_text, rel)})

    return {
        "schema_version": "incidence15_input_diff_v1",
        "nominal_case": nominal_case.name,
        "incidence_case": incidence_case.name,
        "incidence_angle_deg": 15.0,
        "rotation_axis": "+Y",
        "files_checked": files_checked,
        "expected_changes": [
            {
                "file": "0/U",
                "field": "airInlet.value",
                "nominal": list(AIR_VECTOR_NOMINAL),
                "incidence": list(AIR_VECTOR_INCIDENCE),
                "count": 1,
            },
            {
                "file": "constant/sprayCloudProperties",
                "field": "injection.direction",
                "nominal": list(SPRAY_VECTOR_NOMINAL),
                "incidence": list(SPRAY_VECTOR_INCIDENCE),
                "count": 5,
            },
        ],
        "unexpected_differences": unexpected,
        "pass": not unexpected,
    }


def _centroid(metrics: dict) -> np.ndarray:
    deposition = metrics["deposition"]
    return np.asarray([deposition["centroid_u_m"], deposition["centroid_v_m"]], dtype=float)


def compute_response(metrics_by_label: dict[str, dict]) -> dict:
    coarse = metrics_by_label["coarse"]
    medium = metrics_by_label["medium"]
    incidence = metrics_by_label["medium_incidence_15deg"]

    mesh_centroid_noise = float(np.linalg.norm(_centroid(medium) - _centroid(coarse)))
    incidence_centroid_shift = float(np.linalg.norm(_centroid(incidence) - _centroid(medium)))
    shape_fields = {
        "sigma_major_m": "sigma_major_delta_m",
        "sigma_minor_m": "sigma_minor_delta_m",
        "peak_areal_mass_kg_m2": "peak_areal_mass_delta_kg_m2",
    }
    mesh_deltas: dict[str, float] = {}
    incidence_deltas: dict[str, float] = {}
    shape_gate: dict[str, bool] = {}
    for field, output_name in shape_fields.items():
        coarse_value = float(coarse["deposition"][field])
        medium_value = float(medium["deposition"][field])
        incidence_value = float(incidence["deposition"][field])
        mesh_deltas[output_name] = medium_value - coarse_value
        incidence_deltas[output_name] = incidence_value - medium_value
        shape_gate[field] = abs(incidence_deltas[output_name]) > abs(mesh_deltas[output_name])

    u_delta = float(incidence["deposition"]["centroid_u_m"] - medium["deposition"]["centroid_u_m"])
    v_delta = float(incidence["deposition"]["centroid_v_m"] - medium["deposition"]["centroid_v_m"])
    direction_pass = u_delta > 0.0
    centroid_gate = incidence_centroid_shift > mesh_centroid_noise
    shape_or_intensity_gate = any(shape_gate.values())
    distinguishable = centroid_gate and shape_or_intensity_gate and direction_pass
    if not direction_pass:
        status = "FAIL_INCIDENCE_DIRECTION"
    elif distinguishable:
        status = "INCIDENCE_RESPONSE_VALIDATED"
    else:
        status = "INCIDENCE_RESPONSE_NOT_DISTINGUISHABLE"

    return {
        "schema_version": "incidence_response_v1",
        "baseline": "medium",
        "incidence_case": "medium_incidence_15deg",
        "incidence_angle_deg": 15.0,
        "rotation_axis": "+Y",
        "mesh_centroid_noise_m": mesh_centroid_noise,
        "incidence_centroid_shift_m": incidence_centroid_shift,
        "centroid_response_to_noise_ratio": float(incidence_centroid_shift / mesh_centroid_noise) if mesh_centroid_noise else float("inf"),
        "mesh_shape_deltas": mesh_deltas,
        "incidence_shape_deltas": incidence_deltas,
        "direction_check": {
            "centroid_u_nominal_m": float(medium["deposition"]["centroid_u_m"]),
            "centroid_u_incidence_m": float(incidence["deposition"]["centroid_u_m"]),
            "delta_u_m": u_delta,
            "delta_v_m": v_delta,
            "expected_positive_u_shift": True,
            "pass": direction_pass,
        },
        "distinguishability_gate": {
            "centroid_shift_gt_mesh_noise": centroid_gate,
            "shape_or_intensity_delta_gt_nominal_mesh_delta": shape_or_intensity_gate,
            "shape_checks": shape_gate,
            "pass": distinguishable,
        },
        "status": status,
    }


def _draw_principal_axis(ax, metrics: dict, color: str) -> None:
    covariance = np.asarray(metrics["deposition"]["covariance_uv_m2"], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, np.argmax(eigenvalues)]
    sigma = float(np.sqrt(max(float(np.max(eigenvalues)), 0.0)))
    centroid = _centroid(metrics)
    end = centroid + 3.0 * sigma * major
    start = centroid - 3.0 * sigma * major
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2.0, label="3σ principal axis")


def render_incidence_comparison(metrics_by_label: dict[str, dict], output_path: Path, results_root: Path) -> None:
    labels = ("medium", "medium_incidence_15deg")
    maps = []
    for label in labels:
        data = np.load(results_root / label / "deposition_map.npz")
        maps.append({key: data[key] for key in data.files})
        data.close()
    vmax = max(float(np.max(item["areal_mass_kg_m2"])) for item in maps)
    norm = Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True, sharex=True, sharey=True)
    for ax, label, map_data, color in zip(axes, labels, maps, ("#111827", "#dc2626")):
        scatter = ax.scatter(
            map_data["u_m"],
            map_data["v_m"],
            c=map_data["areal_mass_kg_m2"],
            s=26,
            marker="s",
            cmap="viridis",
            norm=norm,
        )
        ax.scatter(*_centroid(metrics_by_label[label]), marker="x", s=75, linewidths=2.0, color=color, label="mass centroid")
        _draw_principal_axis(ax, metrics_by_label[label], color)
        ax.set_title("Nominal medium" if label == "medium" else "15° incidence")
        ax.set_xlabel("u = fan-major x [m]")
        ax.set_aspect("equal")
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(-0.15, 0.15)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("v = fan-minor y [m]")
    fig.colorbar(scatter, ax=axes.tolist(), label="stuck mass / face area [kg m$^{-2}$]")
    fig.suptitle("v2606 air-assisted spray: nominal vs 15° incidence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_metrics(results_root: Path, label: str) -> dict:
    return json.loads((results_root / label / "metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inputs-only", action="store_true", help="write only the strict input dictionary audit")
    args = parser.parse_args()
    repo = args.repo.resolve()
    case_root = repo / "reference_cfd" / "openfoam_v2606" / "flat_plate"
    results_root = repo / "results" / "air_assisted" / "openfoam_v2606" / "flat_plate"
    media_root = repo / "media" / "air_assisted_spray"

    input_diff = compare_incidence_inputs(case_root / "medium", case_root / "medium_incidence_15deg")
    input_diff_path = results_root / "incidence15_input_diff.json"
    input_diff_path.write_text(json.dumps(input_diff, indent=2) + "\n", encoding="utf-8")
    if not input_diff["pass"]:
        print(json.dumps(input_diff, indent=2))
        raise SystemExit(2)
    if args.inputs_only:
        print(json.dumps(input_diff, indent=2))
        return

    metrics = {
        "coarse": _load_metrics(results_root, "coarse"),
        "medium": _load_metrics(results_root, "medium"),
    }
    metrics["medium_incidence_15deg"] = process_case(
        case_root / "medium_incidence_15deg",
        results_root,
        "medium_incidence_15deg",
    )
    write_sensitivity(metrics, results_root)
    response = compute_response(metrics)
    response_path = results_root / "incidence_response.json"
    response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
    render_incidence_comparison(metrics, media_root / "openfoam_incidence_response.png", results_root)
    velocity_source = media_root / "openfoam_air_velocity_medium_incidence_15deg.png"
    if velocity_source.exists():
        shutil.copy2(velocity_source, media_root / "openfoam_air_velocity_incidence15.png")

    print(json.dumps({"input_diff": input_diff, "response": response}, indent=2))


if __name__ == "__main__":
    main()
