"""Surface sampling and finite-difference normal estimation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def sample_surface(
    surface_fn: Callable[[float, float], float],
    y_range: tuple[float, float] = (-1.50, 1.50),
    z_range: tuple[float, float] = (1.65, 2.55),
    y_count: int = 61,
    z_count: int = 19,
    epsilon: float = 0.004,
) -> list[dict]:
    """Sample positions and unit normals on an x=f(y,z) surface.

    The implementation mirrors the validated portfolio prototype: tangents are
    estimated with central finite differences and the process-side normal is
    consistently oriented toward negative X.
    """
    if y_count < 2 or z_count < 2:
        raise ValueError("y_count and z_count must both be at least 2")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    samples: list[dict] = []
    for grid_z, z in enumerate(np.linspace(*z_range, z_count)):
        for grid_y, y in enumerate(np.linspace(*y_range, y_count)):
            x = float(surface_fn(float(y), float(z)))
            xp = float(surface_fn(float(y + epsilon), float(z)))
            xm = float(surface_fn(float(y - epsilon), float(z)))
            zp = float(surface_fn(float(y), float(z + epsilon)))
            zm = float(surface_fn(float(y), float(z - epsilon)))
            tangent_y = np.array([xp - xm, 2.0 * epsilon, 0.0])
            tangent_z = np.array([zp - zm, 0.0, 2.0 * epsilon])
            normal = np.cross(tangent_y, tangent_z)
            length = float(np.linalg.norm(normal))
            if not np.isfinite(x) or length < 1e-12:
                continue
            normal /= length
            if normal[0] > 0:
                normal = -normal
            samples.append(
                {
                    "index": len(samples),
                    "grid_y": grid_y,
                    "grid_z": grid_z,
                    "position": [x, float(y), float(z)],
                    "normal": normal.tolist(),
                }
            )
    return samples
