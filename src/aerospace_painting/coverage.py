"""Geometric coverage accumulation for sampled surfaces."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .spray_geometry import cone_sample, geometric_weight


def compute_coverage(
    samples: Iterable[dict],
    path: Iterable[dict],
    nominal_stand_off: float = 0.32,
    cone_half_angle_deg: float = 18.0,
    threshold: float = 0.35,
) -> dict:
    """Evaluate geometric exposure from discrete TCP waypoints."""
    sample_list = list(samples)
    path_list = list(path)
    records: list[dict] = []

    for sample in sample_list:
        point = np.asarray(sample["position"], dtype=float)
        total_score = 0.0
        hit_count = 0
        nearest_distance = float("inf")
        nearest_angle = 180.0
        for waypoint in path_list:
            inside, distance, angle = cone_sample(
                waypoint["tcp"], waypoint["tool_axis"], point, cone_half_angle_deg
            )
            if inside:
                contribution = geometric_weight(
                    distance, angle, nominal_stand_off, cone_half_angle_deg
                )
                total_score += contribution
                if contribution >= threshold:
                    hit_count += 1
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_angle = angle

        label = "under-covered"
        if total_score >= threshold:
            label = "overlapped" if hit_count >= 2 else "nominal-covered"
        records.append(
            {
                "sample_index": int(sample["index"]),
                "score": float(total_score),
                "coverage_pass_count": int(hit_count),
                "nearest_stand_off": float(nearest_distance),
                "nearest_angle_deg": float(nearest_angle),
                "class": label,
            }
        )

    scores = np.asarray([record["score"] for record in records])
    distances = np.asarray([record["nearest_stand_off"] for record in records])
    angles = np.asarray([record["nearest_angle_deg"] for record in records])
    return {
        "status": "GEOMETRIC_PROCESS_DEMO_EXECUTED",
        "sampled_surface_points": len(sample_list),
        "tcp_waypoints": len(path_list),
        "nominal_stand_off_m": float(nominal_stand_off),
        "cone_half_angle_deg": float(cone_half_angle_deg),
        "coverage_ratio": float(np.mean(scores >= threshold)) if len(scores) else 0.0,
        "mean_stand_off_error_m": (
            float(np.mean(np.abs(distances - nominal_stand_off))) if len(distances) else 0.0
        ),
        "max_stand_off_error_m": (
            float(np.max(np.abs(distances - nominal_stand_off))) if len(distances) else 0.0
        ),
        "mean_surface_normal_angle_error_deg": float(np.mean(angles)) if len(angles) else 0.0,
        "uncovered_sample_count": int(np.sum(scores < threshold)),
        "records": records,
    }
