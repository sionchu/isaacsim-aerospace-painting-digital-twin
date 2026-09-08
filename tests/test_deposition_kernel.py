import numpy as np

from aerospace_painting.deposition_kernel import AnisotropicGaussian


def test_anisotropic_kernel_peaks_at_centroid():
    kernel = AnisotropicGaussian(
        mass_kg=1.0,
        centroid_uv_m=(0.1, -0.2),
        sigma_major_m=0.2,
        sigma_minor_m=0.05,
        rotation_deg=30.0,
    )
    points = np.array([[0.1, -0.2], [0.4, -0.2]])
    density = kernel.density(points)
    assert density[0] > density[1]
    assert density[0] > 0.0


def test_kernel_regular_grid_integral_is_close_to_mass():
    kernel = AnisotropicGaussian(
        mass_kg=0.25,
        centroid_uv_m=(0.0, 0.0),
        sigma_major_m=0.12,
        sigma_minor_m=0.06,
        rotation_deg=18.0,
    )
    axis = np.linspace(-0.8, 0.8, 401)
    u, v = np.meshgrid(axis, axis, indexing="xy")
    uv = np.stack((u, v), axis=-1)
    cell_area = float(axis[1] - axis[0]) ** 2
    assert np.isclose(kernel.integrate_regular_grid(uv, cell_area), 0.25, rtol=1e-3)
