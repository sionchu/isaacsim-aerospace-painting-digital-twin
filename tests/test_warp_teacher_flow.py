import numpy as np
import pytest

from aerospace_painting.warp_teacher_flow import (
    TeacherFlowGrid,
    reconstruct_structured_grid,
    sample_teacher_flow_warp,
    wp,
)


def _synthetic_grid() -> TeacherFlowGrid:
    values = np.empty((3, 3, 3, 3), dtype=np.float32)
    for iz in range(3):
        for iy in range(3):
            for ix in range(3):
                x, y, z = float(ix), float(iy), float(iz)
                values[iz, iy, ix] = (x + 2.0 * y + 3.0 * z, 2.0 * x - y + 0.5 * z, -x + y + z)
    return TeacherFlowGrid(
        angle_deg=0.0,
        values_nz_ny_nx_3=values,
        first_center_m=(0.0, 0.0, 0.0),
        spacing_m=(1.0, 1.0, 1.0),
        source_case="synthetic",
        source_c_path="synthetic/C",
        source_u_path="synthetic/U",
        source_c_sha256="c" * 64,
        source_u_sha256="u" * 64,
        openfoam_version="v2606",
        latest_time_s=0.06,
    )


def test_structured_reconstruction_uses_coordinates_not_list_order():
    rows = []
    for z in (0.0, 1.0):
        for y in (0.0, 1.0):
            for x in (0.0, 1.0, 2.0):
                rows.append(((x, y, z), (10.0 * x + y, z, x - y)))
    rows = list(reversed(rows))
    centres = np.asarray([row[0] for row in rows], dtype=float)
    velocity = np.asarray([row[1] for row in rows], dtype=float)
    grid, xs, ys, zs = reconstruct_structured_grid(centres, velocity)
    assert grid.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(xs, (0.0, 1.0, 2.0))
    np.testing.assert_allclose(ys, (0.0, 1.0))
    np.testing.assert_allclose(zs, (0.0, 1.0))
    np.testing.assert_allclose(grid[1, 0, 2], (20.0, 1.0, 2.0))


def test_cpu_trilinear_interpolation_is_exact_for_linear_vector_field():
    grid = _synthetic_grid()
    points = np.asarray([[0.25, 0.75, 1.25], [1.5, 1.5, 1.5]])
    values, inside = grid.trilinear_velocity(points)
    expected = np.asarray(
        [
            (0.25 + 2.0 * 0.75 + 3.0 * 1.25, 2.0 * 0.25 - 0.75 + 0.5 * 1.25, -0.25 + 0.75 + 1.25),
            (1.5 + 2.0 * 1.5 + 3.0 * 1.5, 2.0 * 1.5 - 1.5 + 0.5 * 1.5, -1.5 + 1.5 + 1.5),
        ]
    )
    assert np.all(inside)
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1.0e-6)


def test_outside_teacher_domain_is_explicit_and_not_edge_clamped():
    grid = _synthetic_grid()
    values, inside = grid.trilinear_velocity(np.asarray([[3.0, 1.0, 1.0], [0.0, 0.0, -0.6]]))
    np.testing.assert_array_equal(inside, (False, False))
    np.testing.assert_allclose(values, 0.0)


def test_cpu_and_warp_teacher_sampling_agree_when_warp_is_available():
    if wp is None:
        pytest.skip("native Isaac Sim Warp is not installed in the portable test interpreter")
    grid = _synthetic_grid()
    points = np.asarray([[0.25, 0.75, 1.25], [1.5, 1.5, 1.5], [4.0, 0.0, 0.0]])
    cpu_values, cpu_inside = grid.trilinear_velocity(points)
    gpu_values, gpu_inside = sample_teacher_flow_warp(grid, points)
    np.testing.assert_array_equal(cpu_inside, gpu_inside)
    np.testing.assert_allclose(cpu_values, gpu_values, rtol=0.0, atol=1.0e-5)
