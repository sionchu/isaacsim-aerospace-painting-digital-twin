import json

import numpy as np
import pytest

from aerospace_painting.warp_air_field import frame_axes
from aerospace_painting.warp_teacher_flow import TeacherFlowGrid
from aerospace_painting.warp_vector_carrier import (
    COROTATING_VECTOR_INTERP,
    FIXED_GRID_LOCAL_VECTOR_INTERP,
    WORLD_LINEAR_DIAGNOSTIC,
    VectorCarrierModel,
    build_model_payload,
    load_model_artifacts,
    sample_vector_carrier_warp,
    wp,
)


def _grid(angle: float, local_vector: tuple[float, float, float], *, value_scale: float = 1.0) -> TeacherFlowGrid:
    values = np.empty((3, 3, 3, 3), dtype=np.float32)
    x_axis, y_axis, z_axis = frame_axes(angle)
    world = value_scale * (
        local_vector[0] * x_axis + local_vector[1] * y_axis + local_vector[2] * z_axis
    )
    values[...] = np.asarray(world, dtype=np.float32)
    return TeacherFlowGrid(
        angle_deg=angle,
        values_nz_ny_nx_3=values,
        first_center_m=(-1.0, -1.0, -1.0),
        spacing_m=(1.0, 1.0, 1.0),
        source_case=f"synthetic/{angle:g}",
        source_c_path=f"synthetic/{angle:g}/C",
        source_u_path=f"synthetic/{angle:g}/U",
        source_c_sha256="c" * 64,
        source_u_sha256="u" * 64,
        openfoam_version="v2606",
        latest_time_s=0.06,
    )


def _model() -> VectorCarrierModel:
    return VectorCarrierModel.from_teacher_grids(
        _grid(0.0, (1.0, 2.0, 3.0)),
        _grid(15.0, (1.0, 2.0, 3.0)),
        nozzle_origin_m=(0.0, 0.0, 0.0),
    )


def test_endpoint_exactness_uses_corresponding_anchor_only():
    model = _model()
    points = np.asarray([[0.0, 0.0, 0.0], [0.25, -0.2, 0.3]], dtype=float)
    value0, inside0 = model.sample(points, 0.0)
    value15, inside15 = model.sample(points, 15.0)
    np.testing.assert_array_equal(inside0, True)
    np.testing.assert_array_equal(inside15, True)
    expected0 = _grid(0.0, (1.0, 2.0, 3.0)).values_nz_ny_nx_3[1, 1, 1]
    expected15 = _grid(15.0, (1.0, 2.0, 3.0)).values_nz_ny_nx_3[1, 1, 1]
    np.testing.assert_allclose(value0, np.broadcast_to(expected0, value0.shape), atol=1.0e-6)
    np.testing.assert_allclose(value15, np.broadcast_to(expected15, value15.shape), atol=1.0e-6)


def test_corotating_interpolation_preserves_local_transverse_vector():
    model = _model()
    points = np.asarray([[0.1, 0.2, 0.3]], dtype=float)
    sampled, inside = model.sample(points, 7.5, interpolation=COROTATING_VECTOR_INTERP)
    expected = np.asarray(frame_axes(7.5)[0]) + 2.0 * np.asarray(frame_axes(7.5)[1]) + 3.0 * np.asarray(frame_axes(7.5)[2])
    assert bool(inside[0])
    np.testing.assert_allclose(sampled[0], expected, rtol=0.0, atol=1.0e-6)


def test_world_linear_diagnostic_is_distinct_and_outside_is_explicit():
    model = _model()
    points = np.asarray([[0.1, 0.2, 0.3], [3.0, 0.0, 0.0]], dtype=float)
    diagnostic, inside = model.sample(points, 7.5, interpolation=WORLD_LINEAR_DIAGNOSTIC)
    assert bool(inside[0])
    assert not bool(inside[1])
    np.testing.assert_allclose(diagnostic[1], 0.0)


def test_fixed_grid_interpolation_keeps_same_world_query_and_preserves_local_vector():
    model = _model()
    points = np.asarray([[0.1, 0.2, 0.3]], dtype=float)
    sampled, inside = model.sample(points, 10.0, interpolation=FIXED_GRID_LOCAL_VECTOR_INTERP)
    expected = np.asarray(frame_axes(10.0)[0]) + 2.0 * np.asarray(frame_axes(10.0)[1]) + 3.0 * np.asarray(frame_axes(10.0)[2])
    assert bool(inside[0])
    np.testing.assert_allclose(sampled[0], expected, rtol=0.0, atol=1.0e-6)


def test_fixed_grid_intermediate_cell_centres_have_full_coverage():
    model = _model()
    x = np.linspace(-0.5, 0.5, 3)
    y = np.linspace(-0.5, 0.5, 3)
    z = np.linspace(-0.5, 0.5, 3)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    for angle in (7.5, 10.0):
        _, inside = model.sample(points, angle, interpolation=FIXED_GRID_LOCAL_VECTOR_INTERP)
        assert bool(np.all(inside))


def test_model_artifact_provenance_round_trip(tmp_path):
    model = _model()
    npz_path = tmp_path / "model.npz"
    json_path = tmp_path / "model.json"
    npz_hash = model.save_runtime_npz(npz_path)
    payload = build_model_payload(model, npz_path=npz_path.name, npz_sha256=npz_hash, source_commit="test-commit")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    loaded, loaded_payload = load_model_artifacts(json_path, repo_root=tmp_path, verify_sources=False)
    assert loaded_payload["model_sha256"] == payload["model_sha256"]
    np.testing.assert_array_equal(loaded.anchor_0.values_nz_ny_nx_3, model.anchor_0.values_nz_ny_nx_3)
    np.testing.assert_array_equal(loaded.anchor_15.values_nz_ny_nx_3, model.anchor_15.values_nz_ny_nx_3)


def test_cpu_and_warp_vector_sampling_agree_when_warp_is_available():
    if wp is None:
        pytest.skip("native Isaac Sim Warp is not installed in the portable test interpreter")
    model = _model()
    points = np.asarray([[0.1, 0.2, 0.3], [-0.25, 0.1, 0.2], [3.0, 0.0, 0.0]], dtype=float)
    cpu_values, cpu_inside = model.sample(points, 7.5)
    gpu_values, gpu_inside = sample_vector_carrier_warp(model, points, 7.5)
    np.testing.assert_array_equal(cpu_inside, gpu_inside)
    np.testing.assert_allclose(cpu_values, gpu_values, rtol=0.0, atol=1.0e-5)
    cpu_fixed, cpu_fixed_inside = model.sample(points, 10.0, interpolation=FIXED_GRID_LOCAL_VECTOR_INTERP)
    gpu_fixed, gpu_fixed_inside = sample_vector_carrier_warp(model, points, 10.0, interpolation=FIXED_GRID_LOCAL_VECTOR_INTERP)
    np.testing.assert_array_equal(cpu_fixed_inside, gpu_fixed_inside)
    np.testing.assert_allclose(cpu_fixed, gpu_fixed, rtol=0.0, atol=1.0e-5)
