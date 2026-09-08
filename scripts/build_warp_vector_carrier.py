"""Build the compact W1.2 full-vector carrier from 0°/15° OpenFOAM anchors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerospace_painting.warp_teacher_flow import TeacherFlowGrid
from aerospace_painting.warp_vector_carrier import (
    MODEL_ID,
    VectorCarrierModel,
    build_model_payload,
    load_model_artifacts,
)
from scripts.validate_warp_spray import CASE_DIR, CASE_LABELS


MODEL_DIR = ROOT / "models"
MODEL_JSON = MODEL_DIR / "warp_vector_carrier_v2.json"
MODEL_NPZ = MODEL_DIR / "warp_vector_carrier_v2.npz"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "c6a17feab4d8850bf879dfe6b8ca8cc2ad2def02"


def build_model(*, force: bool = False) -> tuple[VectorCarrierModel, dict[str, Any]]:
    if (MODEL_JSON.exists() or MODEL_NPZ.exists()) and not force:
        if not (MODEL_JSON.exists() and MODEL_NPZ.exists()):
            raise RuntimeError("vector carrier artifact pair is incomplete; use --force to regenerate")
        return load_model_artifacts(MODEL_JSON, repo_root=ROOT, verify_sources=True)

    grids = {
        angle: TeacherFlowGrid.from_case_dir(CASE_DIR / CASE_LABELS[angle], angle, repo_root=ROOT)
        for angle in (0.0, 15.0)
    }
    model = VectorCarrierModel.from_teacher_grids(grids[0.0], grids[15.0])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    npz_sha256 = model.save_runtime_npz(MODEL_NPZ)
    payload = build_model_payload(
        model,
        npz_path=MODEL_NPZ.name,
        npz_sha256=npz_sha256,
        source_commit=_source_commit(),
    )
    _write_json(MODEL_JSON, payload)
    loaded, loaded_payload = load_model_artifacts(MODEL_JSON, repo_root=ROOT, verify_sources=True)
    if loaded.model_id != MODEL_ID or loaded_payload["model_sha256"] != payload["model_sha256"]:
        raise RuntimeError("vector carrier artifact self-verification failed")
    return loaded, loaded_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate the model pair")
    args = parser.parse_args()
    model, payload = build_model(force=args.force)
    print(
        json.dumps(
            {
                "model_id": model.model_id,
                "model_sha256": payload["model_sha256"],
                "model_json_sha256": payload["model_json_sha256"],
                "npz_sha256": payload["npz_sha256"],
                "grid": payload["grid"],
                "source_commit": payload["source_commit"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
