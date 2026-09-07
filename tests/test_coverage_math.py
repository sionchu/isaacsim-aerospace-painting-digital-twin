import math

import numpy as np

from aerospace_painting import build_tcp_path, compute_coverage, sample_surface
from aerospace_painting.spray_geometry import cone_sample, geometric_weight


def curved_surface(y: float, z: float) -> float:
    return -0.35 + 0.06 * y**2 + 0.02 * (z - 2.0) ** 2


def test_surface_normals_are_unit_and_process_facing():
    samples = sample_surface(curved_surface, y_count=7, z_count=5)
    assert len(samples) == 35
    for sample in samples:
        normal = np.asarray(sample["normal"])
        assert math.isclose(float(np.linalg.norm(normal)), 1.0, rel_tol=1e-9)
        assert normal[0] < 0


def test_tcp_path_preserves_stand_off_and_alternates_rows():
    samples = sample_surface(curved_surface, y_count=5, z_count=3)
    path = build_tcp_path(samples, stand_off=0.32)
    assert len(path) == len(samples)
    assert path[0]["tcp"][1] < path[4]["tcp"][1]
    assert path[5]["tcp"][1] > path[9]["tcp"][1]
    for waypoint in path:
        delta = np.asarray(waypoint["tcp"]) - np.asarray(waypoint["position"])
        assert math.isclose(float(np.linalg.norm(delta)), 0.32, rel_tol=1e-9)


def test_cone_and_weight_are_geometric_and_bounded():
    inside, distance, angle = cone_sample([0, 0, 0], [1, 0, 0], [0.32, 0, 0], 18)
    assert inside
    assert math.isclose(distance, 0.32)
    assert math.isclose(angle, 0.0)
    assert math.isclose(geometric_weight(distance, angle, 0.32, 18), 1.0)


def test_coverage_reports_full_direct_waypoint_exposure():
    samples = sample_surface(curved_surface, y_count=5, z_count=3)
    path = build_tcp_path(samples, stand_off=0.32)
    report = compute_coverage(samples, path, nominal_stand_off=0.32)
    assert report["status"] == "GEOMETRIC_PROCESS_DEMO_EXECUTED"
    assert report["sampled_surface_points"] == 15
    assert report["coverage_ratio"] == 1.0
    assert report["uncovered_sample_count"] == 0
