import numpy as np

from aerospace_painting.frames import NozzleFrame


def test_nozzle_frame_is_right_handed_and_round_trips_points():
    frame = NozzleFrame.from_axes(
        origin=np.array([1.0, -2.0, 0.5]),
        spray_axis=np.array([0.0, 0.0, 1.0]),
        fan_major_axis=np.array([1.0, 0.2, 0.0]),
    )
    local = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.4]])
    world = frame.to_world(local)
    assert np.allclose(frame.to_local(world), local)
    assert np.allclose(np.cross(frame.fan_major_axis, frame.fan_minor_axis), frame.spray_axis)


def test_nozzle_frame_uses_canonical_axis_order():
    frame = NozzleFrame.from_axes(
        origin=np.zeros(3),
        spray_axis=np.array([0.0, 0.0, 1.0]),
        fan_major_axis=np.array([1.0, 0.0, 0.0]),
    )
    assert np.allclose(frame.fan_major_axis, [1.0, 0.0, 0.0])
    assert np.allclose(frame.fan_minor_axis, [0.0, 1.0, 0.0])
    assert np.allclose(frame.spray_axis, [0.0, 0.0, 1.0])
