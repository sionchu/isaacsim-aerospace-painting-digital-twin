import numpy as np
import pytest

from aerospace_painting.warp_plume_runtime import (
    WarpBatchFrame,
    WarpHitEvent,
    batch_mass_kg,
    benchmark_to_world,
    world_to_benchmark,
)


def _frame() -> WarpBatchFrame:
    return WarpBatchFrame(
        origin_world_m=np.asarray((1.0, 2.0, 3.0)),
        target_world_m=np.asarray((1.0, 2.0, 3.24)),
        u_world=np.asarray((1.0, 0.0, 0.0)),
        v_world=np.asarray((0.0, 1.0, 0.0)),
        w_world=np.asarray((0.0, 0.0, 1.0)),
        incidence_deg=7.5,
        stand_off_m=0.24,
    )


def test_w2_batch_mass_and_frozen_frame_transform_close():
    assert np.isclose(batch_mass_kg(1.0e-4, 15.0), 1.0e-4 / 15.0)
    frame = _frame()
    local = np.asarray([[0.0, 0.0, 0.002], [0.05, -0.02, 0.24]])
    world = benchmark_to_world(local, frame)
    np.testing.assert_allclose(world_to_benchmark(world, frame), local, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(world[0], frame.origin_world_m)


def test_w2_frame_rejects_non_orthogonal_or_out_of_domain_inputs():
    with pytest.raises(ValueError, match="right-handed|orthogonal"):
        WarpBatchFrame(
            origin_world_m=(0.0, 0.0, 0.0),
            target_world_m=(0.0, 0.0, 0.24),
            u_world=(1.0, 0.0, 0.0),
            v_world=(0.0, 0.0, 1.0),
            w_world=(0.0, 0.0, 1.0),
            incidence_deg=0.0,
            stand_off_m=0.24,
        )
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        WarpBatchFrame(
            origin_world_m=(0.0, 0.0, 0.0),
            target_world_m=(0.0, 0.0, 0.24),
            u_world=(1.0, 0.0, 0.0),
            v_world=(0.0, 1.0, 0.0),
            w_world=(0.0, 0.0, 1.0),
            incidence_deg=15.1,
            stand_off_m=0.24,
        )


def test_w2_mass_input_validation():
    with pytest.raises(ValueError):
        batch_mass_kg(0.0, 15.0)
    with pytest.raises(ValueError):
        batch_mass_kg(1.0e-4, 0.0)


def test_actual_warp_hit_event_is_visual_only_and_provenanced():
    event = WarpHitEvent(
        batch_id=7,
        particle_index=12,
        time_s=0.125,
        position_world_m=(1.0, 2.0, 3.0),
        represented_mass_kg=1.0e-9,
    )
    payload = event.as_dict()
    assert payload["source"] == "NVIDIA Warp actual mesh hit"
    assert payload["visual_only"] is True
    assert payload["adds_mass"] is False
    assert payload["adds_deposition"] is False
    assert payload["position_world_m"] == [1.0, 2.0, 3.0]
