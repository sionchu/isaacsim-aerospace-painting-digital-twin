import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "air_assisted" / "openfoam_v2606" / "flat_plate"
CASE = ROOT / "reference_cfd" / "openfoam_v2606" / "flat_plate"


def test_v2606_mesh_runs_close_the_mass_ledger():
    for label in ("coarse", "medium"):
        metrics = json.loads((RESULTS / label / "metrics.json").read_text())
        ledger = metrics["mass_ledger"]
        assert metrics["openfoam_version"] == "v2606"
        assert metrics["solver"] == "sprayFoam"
        assert metrics["quality"]["solver_end_marker"] is True
        assert metrics["quality"]["deposition_present"] is True
        assert abs(ledger["residual_kg"]) <= 1e-12
        assert abs(ledger["injected_kg"] - 1.0e-6) <= 1e-12
        assert abs(ledger["deposited_kg"] - ledger["injected_kg"]) <= 1e-12


def test_deposition_map_is_the_written_mass_stick_field():
    for label in ("coarse", "medium"):
        metrics = json.loads((RESULTS / label / "metrics.json").read_text())
        deposited = metrics["mass_ledger"]["deposited_kg"]
        map_data = np.load(RESULTS / label / "deposition_map.npz")
        assert map_data["mass_kg"].size == metrics["mesh"]["target_wall_face_count"]
        assert np.isclose(map_data["mass_kg"].sum(), deposited, rtol=0, atol=1e-12)


def test_case_declares_five_v2606_cone_injectors_and_local_interaction():
    cloud = (CASE / "coarse" / "constant" / "sprayCloudProperties").read_text()
    assert cloud.count("type coneNozzleInjection") == 5
    assert "patchInteractionModel localInteraction;" in cloud
    assert "targetWall { type stick; }" in cloud
    assert "airInlet { type escape; }" in cloud
    assert "farField { type escape; }" in cloud


def test_mesh_sensitivity_passes():
    sensitivity = json.loads((RESULTS / "mesh_sensitivity.json").read_text())
    assert sensitivity["pass"] is True
    assert sensitivity["difference_medium_minus_coarse"]["centroid_distance_m"] < 0.01
