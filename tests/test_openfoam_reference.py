import json
from pathlib import Path

import numpy as np

from scripts.postprocess_openfoam_flat_plate import canonical_footprint_metrics
from scripts.validate_openfoam_incidence_response import (
    compare_incidence_inputs,
    compute_response,
)

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


def test_canonical_footprint_metrics_uses_descending_covariance_eigenvalues():
    metrics = canonical_footprint_metrics(np.asarray([[4.0, 0.0], [0.0, 1.0]]), 3.5)
    assert np.isclose(metrics["sigma_major_m"], 2.0)
    assert np.isclose(metrics["sigma_minor_m"], 1.0)
    assert np.isclose(metrics["principal_orientation_deg"], 0.0)
    assert np.isclose(metrics["peak_areal_mass_kg_m2"], 3.5)


def test_incidence_input_diff_has_only_requested_changes():
    diff = compare_incidence_inputs(CASE / "medium", CASE / "medium_incidence_15deg")
    assert diff["pass"] is True
    assert diff["unexpected_differences"] == []
    assert [item["count"] for item in diff["expected_changes"]] == [1, 5]


def test_incidence_response_exceeds_nominal_mesh_noise_and_has_positive_u_shift():
    metrics = {
        label: json.loads((RESULTS / label / "metrics.json").read_text())
        for label in ("coarse", "medium", "medium_incidence_15deg")
    }
    response = compute_response(metrics)
    assert response["status"] == "INCIDENCE_RESPONSE_VALIDATED"
    assert response["direction_check"]["pass"] is True
    assert response["distinguishability_gate"]["centroid_shift_gt_mesh_noise"] is True
    assert response["distinguishability_gate"]["pass"] is True


def test_incidence_ledger_and_map_close_without_openfoam_runtime():
    metrics = json.loads((RESULTS / "medium_incidence_15deg" / "metrics.json").read_text())
    ledger = metrics["mass_ledger"]
    accounted = sum(
        ledger[key]
        for key in (
            "deposited_kg",
            "escaped_kg",
            "remaining_live_mass_kg",
            "evaporated_kg",
            "other_accounted_mass_kg",
        )
    )
    assert abs(ledger["injected_kg"] - accounted) <= 1e-12
    map_data = np.load(RESULTS / "medium_incidence_15deg" / "deposition_map.npz")
    assert np.isclose(map_data["mass_kg"].sum(), ledger["deposited_kg"], rtol=0, atol=1e-12)
