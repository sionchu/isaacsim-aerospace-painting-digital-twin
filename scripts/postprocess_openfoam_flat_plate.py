"""Post-process the v2606 stationary flat-plate spray benchmark.

The parser deliberately reads the ASCII fields written by sprayFoam rather
than recreating deposition from an analytic kernel.  The wall map therefore
comes from the installed v2606 localInteraction massStick field.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _balanced(text: str, start: int, opening: str = "(", closing: str = ")") -> str:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise ValueError("unclosed OpenFOAM list")


def _list_after(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing marker {marker!r}")
    opening = text.find("(", start + len(marker))
    if opening < 0:
        raise ValueError(f"missing list after {marker!r}")
    return _balanced(text, opening)


def _float_list(text: str) -> np.ndarray:
    return np.asarray([float(value) for value in re.findall(FLOAT, text)], dtype=float)


def _vector_list(text: str) -> np.ndarray:
    values = re.findall(
        rf"\(\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*\)", text
    )
    return np.asarray(values, dtype=float)


def _field_file(time_dir: Path, stem: str) -> Path:
    exact = time_dir / stem
    if exact.exists():
        return exact
    candidates = sorted(path for path in time_dir.iterdir() if stem in path.name)
    if not candidates:
        raise FileNotFoundError(f"no field containing {stem!r} in {time_dir}")
    return candidates[0]


def read_internal_vectors(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "internalField"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing internalField in {path}")
    list_match = re.search(r"nonuniform\s+List<vector>\s+\d+\s*\(", text[start:])
    if list_match:
        opening = start + list_match.end() - 1
        return _vector_list(_balanced(text, opening))
    uniform = re.search(r"\buniform\s*(\([^;]+\))", text[start:])
    if uniform:
        return _vector_list(uniform.group(1))
    raise ValueError(f"unsupported vector field in {path}")


def read_boundary_scalars(path: Path, patch_name: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    patch_match = re.search(rf"(?m)^\s*{re.escape(patch_name)}\s*\{{", text)
    if not patch_match:
        raise ValueError(f"missing patch {patch_name!r} in {path}")
    opening = text.find("{", patch_match.start())
    block = _balanced(text, opening, "{", "}")
    list_match = re.search(r"nonuniform\s+List<scalar>\s+\d+\s*\(", block)
    if list_match:
        values = _float_list(_balanced(block, list_match.end() - 1))
        return values
    uniform = re.search(r"value\s+uniform\s+(%s)" % FLOAT, block)
    return np.asarray([float(uniform.group(1)) if uniform else 0.0], dtype=float)


def read_points(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    section = _list_after(text, "points")
    return _vector_list(section)


def read_faces(path: Path) -> list[list[int]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    section = _list_after(text, "faces")
    faces: list[list[int]] = []
    for line in section.splitlines():
        match = re.match(r"\s*\d+\s*\(([^()]*)\)", line)
        if match:
            faces.append([int(value) for value in match.group(1).split()])
    if not faces:
        raise ValueError(f"no faces parsed from {path}")
    return faces


def read_patch_range(path: Path, patch_name: str) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"(?m)^\s*{re.escape(patch_name)}\s*\{{", text)
    if not match:
        raise ValueError(f"missing mesh patch {patch_name!r}")
    opening = text.find("{", match.start())
    block = _balanced(text, opening, "{", "}")
    n_faces = int(re.search(r"\bnFaces\s+(\d+)", block).group(1))
    start_face = int(re.search(r"\bstartFace\s+(\d+)", block).group(1))
    return start_face, n_faces


def _last_injection_masses(log_text: str) -> dict[str, float]:
    current = None
    values: dict[str, float] = {}
    for line in log_text.splitlines():
        match = re.search(r"Injector\s+(\S+)", line)
        if match:
            current = match.group(1).rstrip(":")
        match = re.search(r"mass introduced\s*=\s*(%s)" % FLOAT, line)
        if match and current:
            values[current] = float(match.group(1))
    return values


def _last_fate_masses(log_text: str) -> dict[str, dict[str, float]]:
    current = None
    values: dict[str, dict[str, float]] = {}
    for line in log_text.splitlines():
        match = re.search(r"Parcel fate: (?:patch )?(\S+) \(number, mass\)", line)
        if match:
            current = match.group(1)
        match = re.search(r"-\s*(escape|stick)\s*=\s*\d+,\s*(%s)" % FLOAT, line)
        if match and current:
            values.setdefault(current, {})[match.group(1)] = float(match.group(2))
    return values


def _last_phase_change(log_text: str) -> float:
    values = re.findall(r"Mass transfer phase change\s*=\s*(%s)" % FLOAT, log_text)
    return float(values[-1]) if values else 0.0


def _last_time(case_dir: Path) -> Path:
    times = []
    for child in case_dir.iterdir():
        if child.is_dir():
            try:
                times.append((float(child.name), child))
            except ValueError:
                pass
    if not times:
        raise FileNotFoundError(f"no written time directories in {case_dir}")
    return max(times, key=lambda item: item[0])[1]


def process_case(case_dir: Path, output_root: Path, label: str) -> dict:
    time_dir = _last_time(case_dir)
    log_text = (case_dir / "log.sprayFoam").read_text(encoding="utf-8", errors="replace")
    injection = _last_injection_masses(log_text)

    poly_mesh = case_dir / "constant" / "polyMesh"
    points = read_points(poly_mesh / "points")
    faces = read_faces(poly_mesh / "faces")
    start_face, n_faces = read_patch_range(poly_mesh / "boundary", "targetWall")
    wall_faces = faces[start_face : start_face + n_faces]
    wall_centres = np.asarray([points[face].mean(axis=0) for face in wall_faces])
    wall_areas = np.asarray(
        [0.5 * np.linalg.norm(np.sum(np.cross(points[face], np.roll(points[face], -1, axis=0)), axis=0)) for face in wall_faces]
    )

    mass_stick = read_boundary_scalars(_field_file(time_dir, "massStick"), "targetWall")
    mass_escape_field = read_boundary_scalars(_field_file(time_dir, "massEscape"), "targetWall")
    if len(mass_stick) != n_faces:
        raise ValueError(f"targetWall massStick length {len(mass_stick)} != {n_faces}")
    deposited = float(mass_stick.sum())

    escape_values = []
    for patch in ("airInlet", "farField", "targetWall"):
        try:
            escape_values.append(float(read_boundary_scalars(_field_file(time_dir, "massEscape"), patch).sum()))
        except (FileNotFoundError, ValueError):
            pass
    escaped = float(sum(escape_values))
    evaporated = _last_phase_change(log_text)
    injected = float(sum(injection.values()))

    centres = read_internal_vectors(time_dir / "C")
    velocity = read_internal_vectors(time_dir / "U")
    speeds = np.linalg.norm(velocity, axis=1)
    dy = float(np.min(np.diff(np.unique(np.round(centres[:, 1], 12))))) if len(np.unique(centres[:, 1])) > 1 else 1.0
    mid_y = float(np.unique(centres[:, 1])[np.argmin(np.abs(np.unique(centres[:, 1])))])
    mid_plane = np.isclose(centres[:, 1], mid_y, atol=dy * 0.01)

    u = wall_centres[:, 0]
    v = wall_centres[:, 1]
    mass_density = np.divide(mass_stick, wall_areas, out=np.zeros_like(mass_stick), where=wall_areas > 0)
    if deposited > 0:
        centroid = np.average(np.column_stack([u, v]), axis=0, weights=mass_stick)
        offsets = np.column_stack([u, v]) - centroid
        covariance = (offsets * mass_stick[:, None]).T @ offsets / deposited
    else:
        centroid = np.zeros(2)
        covariance = np.zeros((2, 2))

    residual = injected - deposited - escaped - evaporated
    relative_error = residual / injected if injected else float("nan")
    transfer_efficiency = deposited / injected if injected else 0.0
    transfer_efficiency = min(1.0, max(0.0, transfer_efficiency))
    metrics = {
        "schema_version": "level1_flat_plate_v1",
        "solver": "sprayFoam",
        "openfoam_version": "v2606",
        "case": label,
        "latest_time_s": float(time_dir.name),
        "mesh": {
            "cell_count": int(len(centres)),
            "point_count": int(len(points)),
            "target_wall_face_count": int(n_faces),
        },
        "inputs": {
            "mass_flow_kg_s": 1.0e-4,
            "injection_duration_s": 0.01,
            "expected_injected_mass_kg": 1.0e-6,
            "air_velocity_m_s": 18.0,
            "droplet_initial_speed_m_s": 12.0,
            "diameter_bins_m": [4.0e-5, 7.0e-5, 1.0e-4, 1.5e-4, 2.2e-4],
            "mass_fraction": [0.10, 0.20, 0.30, 0.25, 0.15],
        },
        "mass_ledger": {
            "injected_kg": injected,
            "deposited_kg": deposited,
            "escaped_kg": escaped,
            "evaporated_kg": evaporated,
            "residual_kg": residual,
            "relative_balance_error": relative_error,
            "transfer_efficiency": transfer_efficiency,
            "bin_injected_kg": injection,
        },
        "deposition": {
            "centroid_u_m": float(centroid[0]),
            "centroid_v_m": float(centroid[1]),
            "covariance_uv_m2": covariance.tolist(),
            "nonzero_face_count": int(np.count_nonzero(mass_stick > 0)),
            "peak_areal_mass_kg_m2": float(mass_density.max()),
        },
        "gas_velocity": {
            "min_m_s": float(speeds.min()),
            "max_m_s": float(speeds.max()),
            "mean_m_s": float(speeds.mean()),
            "mid_plane_cell_count": int(mid_plane.sum()),
        },
        "quality": {
            "solver_end_marker": "End" in log_text,
            "mass_balance_pass": bool(abs(relative_error) <= 1e-6),
            "deposition_present": bool(deposited > 0),
        },
    }

    result_dir = output_root / label
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    np.savez(
        result_dir / "deposition_map.npz",
        u_m=u,
        v_m=v,
        mass_kg=mass_stick,
        area_m2=wall_areas,
        areal_mass_kg_m2=mass_density,
    )
    np.savetxt(
        result_dir / "deposition_map.csv",
        np.column_stack([u, v, mass_stick, wall_areas, mass_density]),
        delimiter=",",
        header="u_m,v_m,mass_kg,area_m2,areal_mass_kg_m2",
        comments="",
    )

    media_dir = output_root.parents[3] / "media" / "air_assisted_spray"
    media_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    scatter = ax.scatter(u, v, c=mass_density, s=28, marker="s", cmap="viridis")
    fig.colorbar(scatter, ax=ax, label="stuck mass / face area [kg m$^{-2}$]")
    ax.set_title(f"v2606 flat-plate deposition ({label})")
    ax.set_xlabel("u = fan-major x [m]")
    ax.set_ylabel("v = fan-minor y [m]")
    ax.set_aspect("equal")
    fig.savefig(media_dir / f"openfoam_flat_plate_deposition_{label}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plane = mid_plane
    scatter = ax.scatter(centres[plane, 0], centres[plane, 2], c=speeds[plane], s=13, cmap="magma")
    fig.colorbar(scatter, ax=ax, label="|U| [m s$^{-1}$]")
    ax.set_title(f"v2606 carrier-air speed, y≈0 ({label})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    fig.savefig(media_dir / f"openfoam_air_velocity_{label}.png", dpi=180)
    if label == "coarse":
        fig.savefig(media_dir / "openfoam_air_velocity.png", dpi=180)
    plt.close(fig)
    return metrics


def write_sensitivity(metrics_by_label: dict[str, dict], output_root: Path) -> None:
    labels = list(metrics_by_label)
    coarse = metrics_by_label[labels[0]]
    medium = metrics_by_label[labels[-1]]
    coarse_c = np.asarray(
        [coarse["deposition"]["centroid_u_m"], coarse["deposition"]["centroid_v_m"]]
    )
    medium_c = np.asarray(
        [medium["deposition"]["centroid_u_m"], medium["deposition"]["centroid_v_m"]]
    )
    sensitivity = {
        "schema_version": "level1_flat_plate_mesh_sensitivity_v1",
        "meshes": {
            label: {
                "cell_count": data["mesh"]["cell_count"],
                "transfer_efficiency": data["mass_ledger"]["transfer_efficiency"],
                "centroid_uv_m": [data["deposition"]["centroid_u_m"], data["deposition"]["centroid_v_m"]],
            }
            for label, data in metrics_by_label.items()
        },
        "difference_medium_minus_coarse": {
            "transfer_efficiency_abs": abs(
                medium["mass_ledger"]["transfer_efficiency"]
                - coarse["mass_ledger"]["transfer_efficiency"]
            ),
            "centroid_distance_m": float(np.linalg.norm(medium_c - coarse_c)),
        },
        "pass": bool(
            abs(medium["mass_ledger"]["transfer_efficiency"] - coarse["mass_ledger"]["transfer_efficiency"]) <= 0.02
            and np.linalg.norm(medium_c - coarse_c) <= 0.01
        ),
    }
    out = output_root / "mesh_sensitivity.json"
    out.write_text(json.dumps(sensitivity, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(labels, [metrics_by_label[label]["mass_ledger"]["transfer_efficiency"] for label in labels], color=["#1677ff", "#15a36d"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("wall transfer efficiency [-]")
    ax.set_title("v2606 flat-plate mesh sensitivity")
    fig.savefig(output_root.parents[3] / "media" / "air_assisted_spray" / "openfoam_mesh_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    case_root = repo / "reference_cfd" / "openfoam_v2606" / "flat_plate"
    output_root = repo / "results" / "air_assisted" / "openfoam_v2606" / "flat_plate"
    metrics = {}
    for label in ("coarse", "medium"):
        metrics[label] = process_case(case_root / label, output_root, label)
    write_sensitivity(metrics, output_root)
    print(json.dumps({label: data["mass_ledger"] for label, data in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
