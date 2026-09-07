"""Run the portable geometry/coverage demonstration without Isaac Sim."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from aerospace_painting import build_tcp_path, compute_coverage, sample_surface


def generic_fuselage_surface(y: float, z: float) -> float:
    """A generic curved panel used only for the portable math demonstration."""
    radius = 2.45
    center_z = 2.10
    radial = max(radius**2 - y**2 - 0.30 * (z - center_z) ** 2, 0.01)
    return -math.sqrt(radial) + radius - 0.35


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/coverage_metrics.json"))
    args = parser.parse_args()
    samples = sample_surface(generic_fuselage_surface, y_count=31, z_count=13)
    path = build_tcp_path(samples, stand_off=0.32)
    report = compute_coverage(samples, path, nominal_stand_off=0.32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
