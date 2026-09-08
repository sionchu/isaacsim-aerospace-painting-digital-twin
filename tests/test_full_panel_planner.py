import numpy as np
import pytest

from aerospace_painting.cfd_calibrated_kernel import GaussianMoments
from aerospace_painting.full_panel_planner import PanelSurface, film_statistics
from aerospace_painting.s2_runtime import (
    FiniteSurfaceAccumulator,
    FiniteSurfaceQuadratureError,
    RuntimeStep,
    StructuredSurfaceGrid,
    surface_frame,
)


def _grid():
    surface = PanelSurface.synthetic()
    return StructuredSurfaceGrid.from_structured(
        surface.positions.reshape(-1, 3), surface.normals.reshape(-1, 3), surface.shape
    )


def test_finite_surface_quadrature_records_edge_loss_without_renormalizing():
    grid = _grid()
    accumulator = FiniteSurfaceAccumulator(grid, tolerance_rel=0.5)
    moments = GaussianMoments(1.0e-6, (0.0, 0.0), np.diag((0.08**2, 0.08**2)))
    frame = surface_frame((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    step = RuntimeStep(0, 0.01, 0.0, True, 1.0e-6, 1.0e-6, 0.0, moments, ())
    accumulator.apply(step, target_position=(0.0, 0.0, 0.0), frame=frame)
    ledger = accumulator.as_dict()
    assert 0.0 < ledger["surface_captured_kg"] < ledger["plane_deposited_kg"]
    assert ledger["geometric_edge_loss_kg"] > 0.0
    assert np.isclose(ledger["combined_closure_error_kg"], 0.0, atol=1.0e-18)


def test_finite_surface_quadrature_blocks_capture_beyond_tolerance():
    grid = _grid()
    accumulator = FiniteSurfaceAccumulator(grid, tolerance_rel=0.0)
    moments = GaussianMoments(1.0e-6, (0.0, 0.0), np.diag((0.001**2, 0.001**2)))
    frame = surface_frame((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    step = RuntimeStep(0, 0.01, 0.0, True, 1.0e-6, 1.0e-6, 0.0, moments, ())
    with pytest.raises(FiniteSurfaceQuadratureError) as error:
        accumulator.apply(step, target_position=grid.positions[4], frame=frame)
    assert error.value.code == "BLOCKED_FINITE_SURFACE_QUADRATURE"


def test_film_statistics_keep_wft_primary_and_dft_explicitly_illustrative():
    masses = np.asarray([[1.0e-6, 2.0e-6], [3.0e-6, 4.0e-6]])
    areas = np.ones_like(masses)
    result = film_statistics(masses, areas, liquid_density_kg_m3=1000.0, volume_solids_fraction=0.5)
    assert result["wft_um"]["mean_um"] == pytest.approx(2.5e-3)
    assert result["wft_um"]["min_um"] == pytest.approx(1.0e-3)
    assert result["wft_um"]["max_um"] == pytest.approx(4.0e-3)
    assert result["wft_um"]["std_um"] == pytest.approx(np.std(masses / 1000.0) * 1.0e6)
    assert result["wft_um"]["total_mass_kg"] == pytest.approx(1.0e-5)
    assert result["dft_um"]["assumed_volume_solids_fraction"] == 0.5
    assert "Illustrative DFT estimate" in result["dft_um"]["label"]
