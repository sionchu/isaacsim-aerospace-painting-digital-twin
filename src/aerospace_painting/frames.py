"""Portable coordinate contracts for the air-assisted spray extension.

The canonical nozzle frame is right-handed: ``+Z`` points in the spray/process
direction, ``+X`` is the fan-major direction, and ``+Y`` is the fan-minor
direction.  The same convention is intended for Isaac Sim, reference-case
metadata, deposition maps, and a future batched runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,):
        raise ValueError(f"{name} must be a length-3 vector")
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} must be finite and non-zero")
    return value / norm


@dataclass(frozen=True)
class NozzleFrame:
    """A rigid nozzle frame with an explicit world-to-local contract."""

    origin: np.ndarray
    rotation_world_from_local: np.ndarray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=float)
        rotation = np.asarray(self.rotation_world_from_local, dtype=float)
        if origin.shape != (3,):
            raise ValueError("origin must be a length-3 vector")
        if rotation.shape != (3, 3):
            raise ValueError("rotation_world_from_local must be 3x3")
        if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
            raise ValueError("frame values must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
            raise ValueError("rotation must be right-handed")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "rotation_world_from_local", rotation)

    @classmethod
    def from_axes(
        cls,
        origin: np.ndarray,
        spray_axis: np.ndarray,
        fan_major_axis: np.ndarray,
    ) -> "NozzleFrame":
        """Construct a right-handed frame from the process and fan axes.

        The supplied major axis is projected onto the plane normal to the
        spray axis, which makes small authoring errors explicit and keeps the
        resulting frame orthonormal.
        """

        z_axis = _unit(spray_axis, "spray_axis")
        major = np.asarray(fan_major_axis, dtype=float)
        if major.shape != (3,):
            raise ValueError("fan_major_axis must be a length-3 vector")
        x_projected = major - np.dot(major, z_axis) * z_axis
        x_axis = _unit(x_projected, "fan_major_axis")
        y_axis = _unit(np.cross(z_axis, x_axis), "fan_minor_axis")
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        return cls(np.asarray(origin, dtype=float), rotation)

    @property
    def spray_axis(self) -> np.ndarray:
        return self.rotation_world_from_local[:, 2].copy()

    @property
    def fan_major_axis(self) -> np.ndarray:
        return self.rotation_world_from_local[:, 0].copy()

    @property
    def fan_minor_axis(self) -> np.ndarray:
        return self.rotation_world_from_local[:, 1].copy()

    def to_world(self, local_point: np.ndarray) -> np.ndarray:
        """Transform one or more points from nozzle-local to world space."""

        points = np.asarray(local_point, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("local_point must end in a length-3 dimension")
        return points @ self.rotation_world_from_local.T + self.origin

    def to_local(self, world_point: np.ndarray) -> np.ndarray:
        """Transform one or more points from world space to nozzle-local."""

        points = np.asarray(world_point, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("world_point must end in a length-3 dimension")
        return (points - self.origin) @ self.rotation_world_from_local
