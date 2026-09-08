import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from aerospace_painting.cfd_calibrated_kernel import GaussianMoments
from aerospace_painting.s2_runtime import (
    S2Runtime,
    S2RuntimeModel,
    StructuredSurfaceGrid,
    surface_frame,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "s2_air_assisted_v1.json"


def _model() -> S2RuntimeModel:
    return S2RuntimeModel.load(MODEL_PATH)


def test_model_provenance_hash_and_teacher_contract_are_loadable():
    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    recorded = payload.pop("model_sha256")
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert recorded == actual
    model = _model()
    model.verify_sources(ROOT)
    assert model.openfoam_version == "v2606"
    assert model.model_type == "S2_SINGLE_GAUSSIAN"
    assert model.source_hashes
    assert model.payload["holdout"]["status"] == "S2_HOLDOUT_VALIDATED"


def test_runtime_efficiency_and_dt_mass_scaling():
    model = _model()
    runtime_a = S2Runtime(model)
    runtime_b = S2Runtime(model)
    kwargs = dict(
        incidence_angle_deg=7.5,
        stand_off_m=model.stand_off_m,
        air_velocity_m_s=model.air_velocity_m_s,
        mass_flow_kg_s=model.mass_flow_kg_s,
        spray_on=True,
    )
    first = runtime_a.step(dt_s=0.01, **kwargs)
    second = runtime_b.step(dt_s=0.02, **kwargs)
    assert np.isclose(second.injected_mass_kg, 2.0 * first.injected_mass_kg)
    assert np.isclose(second.deposited_mass_kg, 2.0 * first.deposited_mass_kg)
    assert np.isclose(runtime_a.as_dict()["balance_error_kg"], 0.0, atol=1e-18)
    assert np.isclose(runtime_b.as_dict()["balance_error_kg"], 0.0, atol=1e-18)


def test_runtime_rejects_angle_extrapolation_and_records_event():
    model = _model()
    runtime = S2Runtime(model)
    step = runtime.step(
        dt_s=0.01,
        incidence_angle_deg=15.01,
        stand_off_m=model.stand_off_m,
        air_velocity_m_s=model.air_velocity_m_s,
        mass_flow_kg_s=model.mass_flow_kg_s,
        spray_on=True,
    )
    assert step.deposited_mass_kg == 0.0
    assert step.overspray_mass_kg == step.injected_mass_kg
    assert runtime.as_dict()["out_of_domain_steps"] == 1
    assert runtime.as_dict()["balance_error_kg"] == 0.0


def test_surface_frame_sign_and_minor_component_guard():
    frame = surface_frame(
        surface_normal=(-1.0, 0.0, 0.0),
        fan_major_axis=(0.0, 1.0, 0.0),
        spray_axis=(np.cos(np.deg2rad(7.5)), np.sin(np.deg2rad(7.5)), 0.0),
    )
    assert np.allclose(frame.w_inward, (1.0, 0.0, 0.0))
    assert np.allclose(frame.u_fan_major, (0.0, 1.0, 0.0))
    assert np.allclose(frame.v_fan_minor, (0.0, 0.0, 1.0))
    assert np.isclose(frame.incidence_u_deg, 7.5)
    with pytest.raises(ValueError, match="fan-minor"):
        surface_frame(
            surface_normal=(-1.0, 0.0, 0.0),
            fan_major_axis=(0.0, 1.0, 0.0),
            spray_axis=(np.cos(np.deg2rad(7.5)), np.sin(np.deg2rad(7.5)), np.sin(np.deg2rad(3.0))),
        )


def test_structured_curved_surface_area_weights_and_conservation():
    y = np.linspace(-0.4, 0.4, 9)
    z = np.linspace(-0.3, 0.3, 7)
    positions = np.asarray([[0.08 * yy * yy, yy, zz] for yy in y for zz in z])
    normals = np.asarray([(-1.0, 0.16 * yy, 0.0) for yy in y for zz in z], dtype=float)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    grid = StructuredSurfaceGrid.from_structured(positions, normals, (len(y), len(z)))
    assert np.all(grid.area_weights_m2 > 0.0)
    assert grid.total_area_m2 > 0.0
    frame = surface_frame((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    moments = GaussianMoments(2.5e-6, (0.0, 0.0), np.diag((0.04**2, 0.03**2)))
    deposited = grid.deposit(target_position=(0.0, 0.0, 0.0), frame=frame, moments=moments)
    assert np.isclose(deposited, moments.deposited_mass_kg, rtol=0.0, atol=1e-18)
    assert np.isclose(grid.integrated_mass_kg, moments.deposited_mass_kg, rtol=0.0, atol=1e-18)
