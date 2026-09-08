import numpy as np

from aerospace_painting.warp_air_field import AirFieldParameters, frame_axes, schiller_naumann_cd
from aerospace_painting.warp_spray import WarpCaseConfig, make_injection_samples


def test_warp_injection_is_deterministic_and_closes_each_teacher_bin():
    config = WarpCaseConfig(incidence_angle_deg=7.5, particle_count_per_bin=32)
    first = make_injection_samples(config)
    second = make_injection_samples(config)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])

    masses = np.asarray(first["represented_mass"], dtype=float)
    bin_ids = first["bin_id"]
    expected = config.expected_injected_mass_kg * np.asarray(config.mass_fractions)
    actual = np.asarray([masses[bin_ids == index].sum() for index in range(5)])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-18)
    assert np.isclose(masses.sum(), config.expected_injected_mass_kg, rtol=0.0, atol=1.0e-18)


def test_warp_frame_matches_teacher_axis_convention():
    x_axis, y_axis, z_axis = frame_axes(15.0)
    np.testing.assert_allclose(z_axis, (np.sin(np.deg2rad(15.0)), 0.0, np.cos(np.deg2rad(15.0))))
    np.testing.assert_allclose(x_axis, (np.cos(np.deg2rad(15.0)), 0.0, -np.sin(np.deg2rad(15.0))))
    np.testing.assert_allclose(y_axis, (0.0, 1.0, 0.0))
    np.testing.assert_allclose(np.column_stack((x_axis, y_axis, z_axis)).T @ np.column_stack((x_axis, y_axis, z_axis)), np.eye(3), atol=1.0e-7)


def test_schiller_naumann_drag_limits_are_finite():
    values = schiller_naumann_cd(np.asarray((0.0, 1.0, 1000.0, 1001.0, 1.0e6)))
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)
    assert values[-1] == 0.44


def test_air_field_parameters_reject_nonphysical_values():
    AirFieldParameters(18.0, 0.05, 0.3, 0.05, 1.0)
    for values in ((0.0, 0.05, 0.3, 0.05, 1.0), (18.0, 0.0, 0.3, 0.05, 1.0), (18.0, 0.05, -0.1, 0.05, 1.0)):
        try:
            AirFieldParameters(*values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid air-field parameters accepted: {values}")
