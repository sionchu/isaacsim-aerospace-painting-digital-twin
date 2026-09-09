"""Build and validate the compact OpenFOAM 7.5-degree flow artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "reference_cfd" / "openfoam_v2606" / "flat_plate" / "medium_incidence_7p5deg"
NPZ_PATH = ROOT / "models" / "openfoam_flow_field_7p5deg_v1.npz"
JSON_PATH = ROOT / "models" / "openfoam_flow_field_7p5deg_v1.json"


def _load_source(case_dir: Path):
    sys.path.insert(0, str(ROOT / "src"))
    from aerospace_painting.openfoam_flow_field import OpenFOAMFlowField
    from aerospace_painting.warp_teacher_flow import TeacherFlowGrid

    teacher = TeacherFlowGrid.from_case_dir(case_dir, 7.5, repo_root=ROOT)
    field = OpenFOAMFlowField.from_teacher_grid(teacher)
    return teacher, field


def validate_artifact(npz_path: Path, json_path: Path, case_dir: Path) -> dict:
    from aerospace_painting.openfoam_flow_field import OpenFOAMFlowField

    source_teacher, source_field = _load_source(case_dir)
    artifact = OpenFOAMFlowField.load(npz_path, json_path)
    if artifact.dimensions_nx_ny_nz != (24, 24, 48) or artifact.cell_count != 27_648:
        raise RuntimeError(f"unexpected field dimensions: {artifact.dimensions_nx_ny_nz} / {artifact.cell_count}")
    expected_bounds_min = np.asarray((-0.15, -0.15, 0.0), dtype=float)
    expected_bounds_max = np.asarray((0.15, 0.15, 0.24), dtype=float)
    np.testing.assert_allclose(artifact.bounds_min_m, expected_bounds_min, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(artifact.bounds_max_m, expected_bounds_max, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(artifact.velocity_m_s, source_field.velocity_m_s, rtol=0.0, atol=2.0e-6)
    np.testing.assert_allclose(artifact.grid_centres_m, source_field.grid_centres_m, rtol=0.0, atol=2.0e-7)
    if artifact.source_u_sha256 != source_teacher.source_u_sha256 or artifact.source_c_sha256 != source_teacher.source_c_sha256:
        raise RuntimeError("compact field source hash does not match solved U/C")
    if not np.all(np.isfinite(artifact.velocity_m_s)) or not np.all(np.isfinite(artifact.speed_magnitude_m_s)):
        raise RuntimeError("compact field contains NaN/Inf")
    return {
        "dimensions_nx_ny_nz": list(artifact.dimensions_nx_ny_nz),
        "cells": artifact.cell_count,
        "bounds_min_m": list(artifact.bounds_min_m),
        "bounds_max_m": list(artifact.bounds_max_m),
        "spacing_m": list(artifact.spacing_m),
        "speed_magnitude_m_s": artifact.stats,
        "source_u_sha256": artifact.source_u_sha256,
        "source_c_sha256": artifact.source_c_sha256,
        "artifact_sha256": artifact.artifact_sha256,
        "latest_time_s": artifact.latest_time_s,
        "openfoam_version": artifact.openfoam_version,
    }


def build(case_dir: Path = CASE_DIR, npz_path: Path = NPZ_PATH, json_path: Path = JSON_PATH) -> dict:
    _, field = _load_source(case_dir)
    field.write(npz_path, json_path)
    report = validate_artifact(npz_path, json_path, case_dir)
    print(f"OPENFOAM_FLOW_FIELD_PASS cells={report['cells']} dims={report['dimensions_nx_ny_nz']} latest_time_s={report['latest_time_s']}")
    print(f"SPEED_M_S min={report['speed_magnitude_m_s']['min_m_s']:.9g} mean={report['speed_magnitude_m_s']['mean_m_s']:.9g} max={report['speed_magnitude_m_s']['max_m_s']:.9g}")
    print(f"SOURCE_U_SHA256 {report['source_u_sha256']}")
    print(f"SOURCE_C_SHA256 {report['source_c_sha256']}")
    print(f"ARTIFACT_SHA256 {report['artifact_sha256']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=CASE_DIR)
    parser.add_argument("--npz", type=Path, default=NPZ_PATH)
    parser.add_argument("--json", type=Path, default=JSON_PATH)
    args = parser.parse_args()
    build(args.case, args.npz, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
