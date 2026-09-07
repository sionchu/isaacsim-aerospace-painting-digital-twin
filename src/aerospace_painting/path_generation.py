"""Surface-normal-aware TCP path generation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def build_tcp_path(samples: Iterable[dict], stand_off: float = 0.32) -> list[dict]:
    """Create a boustrophedon path at a fixed geometric stand-off."""
    if stand_off <= 0:
        raise ValueError("stand_off must be positive")
    rows: dict[int, list[dict]] = {}
    for sample in samples:
        rows.setdefault(int(sample["grid_z"]), []).append(sample)

    path: list[dict] = []
    for row_index, row_key in enumerate(sorted(rows)):
        row = sorted(
            rows[row_key],
            key=lambda sample: int(sample["grid_y"]),
            reverse=bool(row_index % 2),
        )
        for sample in row:
            point = np.asarray(sample["position"], dtype=float)
            normal = np.asarray(sample["normal"], dtype=float)
            tcp = point + normal * stand_off
            path.append(
                {
                    "sequence": len(path),
                    "sample_index": int(sample["index"]),
                    "position": point.tolist(),
                    "normal": normal.tolist(),
                    "tcp": tcp.tolist(),
                    "stand_off": float(stand_off),
                    "tool_axis": (-normal).tolist(),
                    "row_index": row_index,
                }
            )
    return path
