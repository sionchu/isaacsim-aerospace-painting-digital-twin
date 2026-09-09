import numpy as np
import pytest

from aerospace_painting.spray_analysis import (
    DEPOSITION_VIEWS,
    ParticleTrailHistory,
    TimelineSync,
    validate_deposition_view,
    visual_layer_mass_contract,
)


def test_deposition_views_are_explicit_and_validated():
    assert DEPOSITION_VIEWS == ("s2", "warp", "compare")
    assert validate_deposition_view("COMPARE") == "compare"
    with pytest.raises(ValueError):
        validate_deposition_view("unknown")


def test_trails_preserve_actual_ids_positions_and_bounded_history():
    history = ParticleTrailHistory(max_particles=3, history_frames=3)
    history.update([11, 22, 33], np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), 0.0)
    history.update([11, 22], np.asarray([[1.1, 0.0, 0.0], [2.1, 0.0, 0.0]]), 1.0 / 30.0)
    lines = history.lines()
    assert len(history.samples) == 3
    assert len(lines) == 2
    assert all(line.shape == (2, 3) for line in lines)
    assert any(np.allclose(line[-1], (1.1, 0.0, 0.0)) for line in lines)


def test_timeline_offsets_and_zero_mass_visual_contract():
    sync = TimelineSync()
    sync.observe(simulated_time_s=1.0, s2_time_s=1.0, overlay_time_s=1.0, warp_emission_time_s=1.0, hit_time_s=[0.99])
    payload = sync.as_dict()
    assert payload["max_warp_to_s2_time_offset_s"] == 0.0
    assert np.isclose(payload["max_hit_to_overlay_time_offset_s"], 0.01)
    assert payload["same_simulated_timeline"] is True
    contract = visual_layer_mass_contract()
    assert contract
    assert all(not value["adds_mass"] and not value["adds_deposition"] for value in contract.values())
