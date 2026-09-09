import hashlib
from pathlib import Path

import numpy as np

from aerospace_painting.openfoam_flow_field import (
    OpenFOAMFlowField,
    VIEW_MODES,
    benchmark_points_to_world,
    benchmark_vectors_to_world,
    clip_flow_roi,
    deterministic_streamlines,
    evaluate_cfd_direction,
)


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "reference_cfd" / "openfoam_v2606" / "flat_plate" / "medium_incidence_7p5deg"
NPZ = ROOT / "models" / "openfoam_flow_field_7p5deg_v1.npz"
MANIFEST = ROOT / "models" / "openfoam_flow_field_7p5deg_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compact_artifact_provenance_and_grid_fidelity():
    field = OpenFOAMFlowField.load(NPZ, MANIFEST)
    assert field.angle_deg == 7.5
    assert field.openfoam_version == "v2606"
    assert field.latest_time_s == 0.06
    assert field.dimensions_nx_ny_nz == (24, 24, 48)
    assert field.cell_count == 27_648
    assert field.source_u_sha256 == _sha256(CASE / "0.06" / "U")
    assert field.source_c_sha256 == _sha256(CASE / "0.06" / "C")
    np.testing.assert_allclose(field.bounds_min_m, (-0.15, -0.15, 0.0), atol=1.0e-9)
    np.testing.assert_allclose(field.bounds_max_m, (0.15, 0.15, 0.24), atol=1.0e-9)
    assert field.velocity_m_s.dtype == np.float32
    assert np.all(np.isfinite(field.velocity_m_s))
    assert np.all(np.isfinite(field.speed_magnitude_m_s))
    assert field.stats["min_m_s"] < field.stats["mean_m_s"] < field.stats["max_m_s"]


def test_benchmark_to_world_mapping_uses_u_v_w_axes():
    points = np.asarray([[0.0, 0.0, 0.002], [0.1, -0.2, 0.052]], dtype=float)
    mapped = benchmark_points_to_world(points, (1.0, 2.0, 3.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    np.testing.assert_allclose(mapped[0], (1.0, 2.0, 3.0))
    np.testing.assert_allclose(mapped[1], (1.2, 2.1, 3.05))


def test_benchmark_velocity_rotation_preserves_components():
    vectors = np.asarray([[[1.0, 2.0, 3.0]]])
    mapped = benchmark_vectors_to_world(vectors, (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    np.testing.assert_allclose(mapped, [[[ -2.0, 1.0, 3.0]]])


def test_streamlines_are_deterministic_and_derived_from_field():
    field = OpenFOAMFlowField.load(NPZ, MANIFEST)
    seeds = np.asarray(
        [
            [-0.025, -0.025, 0.005],
            [0.0, 0.0, 0.005],
            [0.025, 0.025, 0.005],
        ],
        dtype=float,
    )
    first = deterministic_streamlines(field, seeds, step_m=0.004, max_steps=24)
    second = deterministic_streamlines(field, seeds, step_m=0.004, max_steps=24)
    assert len(first) == len(second) == 3
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
        assert left.shape[1] == 3
        assert np.all(np.isfinite(left))


def test_magnitude_slice_is_actual_field_data():
    field = OpenFOAMFlowField.load(NPZ, MANIFEST)
    centres, speeds, index = field.magnitude_slice(axis="y", coordinate_m=0.0)
    assert centres.shape == (48, 24, 3)
    assert speeds.shape == (48, 24)
    assert index == 11
    np.testing.assert_allclose(speeds, field.speed_magnitude_m_s[:, index, :])
    np.testing.assert_allclose(centres, field.grid_centres_m[:, index, :, :])
    assert np.all(np.isfinite(speeds))


def test_view_modes_have_explicit_public_contract():
    assert VIEW_MODES == ("process", "flow", "combined", "result")
    assert set(VIEW_MODES) == {"process", "flow", "combined", "result"}


def test_cfd_direction_gate_and_roi_clipping_use_actual_u_field():
    field = OpenFOAMFlowField.load(NPZ, MANIFEST)
    roi_min = (-0.06, -0.06, 0.0)
    roi_max = (0.06, 0.06, 0.24)
    points, velocity, speed, mask = clip_flow_roi(field, roi_min, roi_max)
    assert len(points) == len(velocity) == len(speed) > 0
    assert mask.shape == (field.cell_count,)
    assert np.all(points >= np.asarray(roi_min) - 1.0e-12)
    assert np.all(points <= np.asarray(roi_max) + 1.0e-12)
    angle = np.deg2rad(7.5)
    report = evaluate_cfd_direction(
        field,
        roi_min_m=roi_min,
        roi_max_m=roi_max,
        spray_axis_world=(np.sin(angle), 0.0, np.cos(angle)),
        u_world=(1.0, 0.0, 0.0),
        v_world=(0.0, 1.0, 0.0),
        w_world=(0.0, 0.0, 1.0),
    )
    assert report.status == "CFD_DIRECTION_PASS"
    assert report.sample_count == len(points)
    assert report.mean_axial_velocity_m_s > 0.0
    assert report.min_axial_velocity_m_s > 0.0
    assert report.max_axial_velocity_m_s >= report.min_axial_velocity_m_s
    assert report.positive_axial_fraction == 1.0
