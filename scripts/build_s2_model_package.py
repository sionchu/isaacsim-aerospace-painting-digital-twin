"""Build the compact deterministic S2 runtime model package."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerospace_painting.cfd_calibrated_kernel import GaussianMoments


VALIDATED_COMMIT = "ae724ac534c89a8d8ee9a3093608e6efcbea878c"
ZERO_METRICS = ROOT / "results/air_assisted/openfoam_v2606/flat_plate/medium/metrics.json"
FIFTEEN_METRICS = ROOT / "results/air_assisted/openfoam_v2606/flat_plate/medium_incidence_15deg/metrics.json"
HOLDOUT = ROOT / "results/air_assisted/s2/holdout_validation.json"
CONFIG = ROOT / "configs/air_assisted_spray.yaml"
OUTPUT = ROOT / "models/s2_air_assisted_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def number_from_config(name: str) -> float:
    text = CONFIG.read_text(encoding="utf-8")
    match = re.search(rf"^\s*{re.escape(name)}:\s*([-+0-9.eE]+)\s*$", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing {name} in {CONFIG}")
    return float(match.group(1))


def endpoint(metrics: dict[str, Any]) -> dict[str, Any]:
    moments = GaussianMoments.from_metrics(metrics)
    return {
        "deposited_mass_kg": moments.deposited_mass_kg,
        "centroid_uv_m": list(moments.centroid_uv_m),
        "covariance_uv_m2": moments.covariance_uv_m2.tolist(),
        "transfer_efficiency": float(metrics["mass_ledger"]["transfer_efficiency"]),
        "case": metrics["case"],
    }


def main() -> int:
    zero = json.loads(ZERO_METRICS.read_text(encoding="utf-8"))
    fifteen = json.loads(FIFTEEN_METRICS.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != VALIDATED_COMMIT:
        raise RuntimeError(f"model package must be built from {VALIDATED_COMMIT}, got {commit}")
    if zero["openfoam_version"] != "v2606" or fifteen["openfoam_version"] != "v2606":
        raise RuntimeError("teacher evidence is not OpenCFD/Keysight OpenFOAM v2606")
    inputs = zero["inputs"]
    if inputs != fifteen["inputs"]:
        raise RuntimeError("endpoint operating inputs differ")
    source_paths = {
        "zero_degree_metrics": ZERO_METRICS.relative_to(ROOT).as_posix(),
        "fifteen_degree_metrics": FIFTEEN_METRICS.relative_to(ROOT).as_posix(),
        "holdout_validation": HOLDOUT.relative_to(ROOT).as_posix(),
        "air_assisted_config": CONFIG.relative_to(ROOT).as_posix(),
    }
    payload: dict[str, Any] = {
        "schema_version": "s2_runtime_model_v1",
        "model_id": "s2_air_assisted_v1",
        "model_type": "S2_SINGLE_GAUSSIAN",
        "calibrated_incidence_deg": [0.0, 15.0],
        "teacher": {
            "openfoam_version": "v2606",
            "solver": "sprayFoam",
            "validated_commit": commit,
            "teacher_case_paths": [source_paths["zero_degree_metrics"], source_paths["fifteen_degree_metrics"]],
        },
        "operating_point": {
            "stand_off_m": number_from_config("stand_off_m"),
            "air_velocity_m_s": float(inputs["air_velocity_m_s"]),
            "mass_flow_kg_s": float(inputs["mass_flow_kg_s"]),
            "canonical_injected_mass_kg": float(inputs["expected_injected_mass_kg"]),
            "reference_duration_s": float(inputs["injection_duration_s"]),
            "droplet_bins_m": [float(value) for value in inputs["diameter_bins_m"]],
            "mass_fraction": [float(value) for value in inputs["mass_fraction"]],
            "droplet_initial_speed_m_s": float(inputs["droplet_initial_speed_m_s"]),
        },
        "endpoints": {
            "0_deg": endpoint(zero),
            "15_deg": endpoint(fifteen),
        },
        "holdout": {
            "status": holdout["status"],
            "case": holdout["holdout_case"],
            "frozen_prediction_file_sha256": holdout["frozen_prediction_file_sha256"],
            "parameter_pass": bool(holdout["acceptance"]["parameter_pass"]),
            "field_pass": bool(holdout["acceptance"]["field_pass"]),
        },
        "source_paths": source_paths,
        "source_hashes": {key: sha256(ROOT / path) for key, path in source_paths.items()},
    }
    payload["model_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "model_sha256": payload["model_sha256"], "source_hashes": payload["source_hashes"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
