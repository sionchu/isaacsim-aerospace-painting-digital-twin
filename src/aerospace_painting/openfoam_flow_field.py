"""Offline OpenFOAM flow-field artifacts and deterministic local-frame helpers.

The compact artifact is a visualization input only.  It preserves the solved
cell-centre ``U`` field and its ``C`` provenance; it is never a live CFD solve.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .warp_teacher_flow import TeacherFlowGrid


SCHEMA_VERSION = "openfoam_flow_field_v1"
BENCHMARK_ORIGIN_M = np.asarray((0.0, 0.0, 0.002), dtype=np.float64)
VIEW_MODES = ("process", "flow", "combined", "result")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit(value: Iterable[float], name: str) -> np.ndarray:
    vector = np.asarray(tuple(value), dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


@dataclass(frozen=True)
class OpenFOAMFlowField:
    """Structured solved OpenFOAM ``U``/``C`` data for visualization."""

    angle_deg: float
    grid_centres_m: np.ndarray
    velocity_m_s: np.ndarray
    speed_magnitude_m_s: np.ndarray
    spacing_m: tuple[float, float, float]
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    source_case: str
    source_c_path: str
    source_u_path: str
    latest_time_s: float
    openfoam_version: str
    source_c_sha256: str
    source_u_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        centres = np.asarray(self.grid_centres_m, dtype=np.float32)
        velocity = np.asarray(self.velocity_m_s, dtype=np.float32)
        speed = np.asarray(self.speed_magnitude_m_s, dtype=np.float32)
        if centres.ndim != 4 or centres.shape[-1] != 3:
            raise ValueError("grid_centres_m must have shape [nz, ny, nx, 3]")
        if velocity.shape != centres.shape or speed.shape != centres.shape[:-1]:
            raise ValueError("flow arrays have inconsistent shapes")
        if not np.all(np.isfinite(centres)) or not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(speed)):
            raise ValueError("flow arrays contain NaN or Inf")
        expected_speed = np.linalg.norm(velocity.astype(np.float64), axis=-1).astype(np.float32)
        if not np.allclose(speed, expected_speed, rtol=0.0, atol=2.0e-6):
            raise ValueError("speed magnitude does not match U")
        spacing = tuple(float(value) for value in self.spacing_m)
        bounds_min = tuple(float(value) for value in self.bounds_min_m)
        bounds_max = tuple(float(value) for value in self.bounds_max_m)
        if len(spacing) != 3 or not all(value > 0.0 for value in spacing):
            raise ValueError("spacing_m must contain positive values")
        if len(bounds_min) != 3 or len(bounds_max) != 3 or not all(lo < hi for lo, hi in zip(bounds_min, bounds_max)):
            raise ValueError("invalid flow-field bounds")
        if not self.source_c_sha256 or not self.source_u_sha256:
            raise ValueError("source hashes are required")
        centres.setflags(write=False)
        velocity.setflags(write=False)
        speed.setflags(write=False)
        object.__setattr__(self, "grid_centres_m", centres)
        object.__setattr__(self, "velocity_m_s", velocity)
        object.__setattr__(self, "speed_magnitude_m_s", speed)
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "bounds_min_m", bounds_min)
        object.__setattr__(self, "bounds_max_m", bounds_max)

    @property
    def dimensions_nx_ny_nz(self) -> tuple[int, int, int]:
        nz, ny, nx, _ = self.velocity_m_s.shape
        return int(nx), int(ny), int(nz)

    @property
    def cell_count(self) -> int:
        return int(np.prod(self.velocity_m_s.shape[:3]))

    @property
    def first_center_m(self) -> np.ndarray:
        return np.asarray(self.grid_centres_m[0, 0, 0], dtype=np.float64)

    @property
    def stats(self) -> dict[str, float]:
        values = self.speed_magnitude_m_s.astype(np.float64)
        return {"min_m_s": float(values.min()), "mean_m_s": float(values.mean()), "max_m_s": float(values.max())}

    @classmethod
    def from_teacher_grid(cls, grid: TeacherFlowGrid) -> "OpenFOAMFlowField":
        nx, ny, nz = grid.dimensions_nx_ny_nz
        x = grid.first_center_m[0] + np.arange(nx, dtype=np.float64) * grid.spacing_m[0]
        y = grid.first_center_m[1] + np.arange(ny, dtype=np.float64) * grid.spacing_m[1]
        z = grid.first_center_m[2] + np.arange(nz, dtype=np.float64) * grid.spacing_m[2]
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        centres = np.stack((xx, yy, zz), axis=-1).transpose(2, 1, 0, 3)
        velocity = np.asarray(grid.values_nz_ny_nx_3, dtype=np.float32)
        speed = np.linalg.norm(velocity.astype(np.float64), axis=-1).astype(np.float32)
        return cls(
            angle_deg=float(grid.angle_deg),
            grid_centres_m=centres,
            velocity_m_s=velocity,
            speed_magnitude_m_s=speed,
            spacing_m=grid.spacing_m,
            bounds_min_m=tuple(float(value) for value in grid.bounds_min_m),
            bounds_max_m=tuple(float(value) for value in grid.bounds_max_m),
            source_case=grid.source_case.replace("\\", "/"),
            source_c_path=grid.source_c_path.replace("\\", "/"),
            source_u_path=grid.source_u_path.replace("\\", "/"),
            latest_time_s=float(grid.latest_time_s),
            openfoam_version=grid.openfoam_version,
            source_c_sha256=grid.source_c_sha256,
            source_u_sha256=grid.source_u_sha256,
        )

    def _npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "grid_centres_m": np.ascontiguousarray(self.grid_centres_m, dtype=np.float32),
            "U_m_s": np.ascontiguousarray(self.velocity_m_s, dtype=np.float32),
            "speed_magnitude_m_s": np.ascontiguousarray(self.speed_magnitude_m_s, dtype=np.float32),
            "dimensions_nx_ny_nz": np.asarray(self.dimensions_nx_ny_nz, dtype=np.int32),
            "spacing_m": np.asarray(self.spacing_m, dtype=np.float64),
            "bounds_min_m": np.asarray(self.bounds_min_m, dtype=np.float64),
            "bounds_max_m": np.asarray(self.bounds_max_m, dtype=np.float64),
            "angle_deg": np.asarray([self.angle_deg], dtype=np.float64),
            "latest_time_s": np.asarray([self.latest_time_s], dtype=np.float64),
        }

    def write(self, npz_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
        npz_target = Path(npz_path)
        manifest_target = Path(manifest_path)
        npz_target.parent.mkdir(parents=True, exist_ok=True)
        manifest_target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_target, **self._npz_payload())
        artifact_sha = _sha256_file(npz_target)
        payload = self.manifest(npz_target, artifact_sha)
        manifest_target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
        return payload

    def manifest(self, npz_path: Path, artifact_sha256: str | None = None) -> dict[str, Any]:
        artifact = artifact_sha256 or self.artifact_sha256
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "field": "OpenFOAM cell-centre velocity reference field",
            "angle_deg": float(self.angle_deg),
            "openfoam_version": self.openfoam_version,
            "latest_time_s": float(self.latest_time_s),
            "source_case": self.source_case,
            "source_paths": {"C": self.source_c_path, "U": self.source_u_path},
            "source_hashes": {"C": self.source_c_sha256, "U": self.source_u_sha256},
            "grid": {
                "array_layout": "U[nz, ny, nx, 3]",
                "dimensions_nx_ny_nz": list(self.dimensions_nx_ny_nz),
                "cells": self.cell_count,
                "bounds_min_m": list(self.bounds_min_m),
                "bounds_max_m": list(self.bounds_max_m),
                "spacing_m": list(self.spacing_m),
                "centres_layout": "grid_centres_m[nz, ny, nx, 3]",
            },
            "speed_magnitude_m_s": self.stats,
            # Keep the manifest portable when the builder receives an
            # absolute local path; the artifact and source paths are resolved
            # relative to the repository by convention.
            "npz_path": npz_path.name,
            "artifact_sha256": artifact,
            "float_dtype": "float32",
            "ordering_verified": True,
        }
        return payload

    @classmethod
    def load(cls, npz_path: str | Path, manifest_path: str | Path, *, verify_hash: bool = True) -> "OpenFOAMFlowField":
        npz_file = Path(npz_path)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported OpenFOAM flow-field artifact schema")
        actual_hash = _sha256_file(npz_file)
        if verify_hash and actual_hash != manifest.get("artifact_sha256"):
            raise ValueError("OpenFOAM flow-field artifact hash mismatch")
        with np.load(npz_file, allow_pickle=False) as data:
            field = cls(
                angle_deg=float(data["angle_deg"][0]),
                grid_centres_m=data["grid_centres_m"],
                velocity_m_s=data["U_m_s"],
                speed_magnitude_m_s=data["speed_magnitude_m_s"],
                spacing_m=tuple(float(value) for value in data["spacing_m"]),
                bounds_min_m=tuple(float(value) for value in data["bounds_min_m"]),
                bounds_max_m=tuple(float(value) for value in data["bounds_max_m"]),
                source_case=str(manifest["source_case"]),
                source_c_path=str(manifest["source_paths"]["C"]),
                source_u_path=str(manifest["source_paths"]["U"]),
                latest_time_s=float(data["latest_time_s"][0]),
                openfoam_version=str(manifest["openfoam_version"]),
                source_c_sha256=str(manifest["source_hashes"]["C"]),
                source_u_sha256=str(manifest["source_hashes"]["U"]),
                artifact_sha256=actual_hash,
            )
        if tuple(int(value) for value in manifest["grid"]["dimensions_nx_ny_nz"]) != field.dimensions_nx_ny_nz:
            raise ValueError("flow-field manifest dimensions do not match NPZ")
        if int(manifest["grid"]["cells"]) != field.cell_count:
            raise ValueError("flow-field manifest cell count does not match NPZ")
        return field

    def trilinear_velocity(self, points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample the compact field and return values plus explicit inside mask."""

        points = np.asarray(points_m, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("points_m must end in a length-3 dimension")
        flat = points.reshape(-1, 3)
        lower = np.asarray(self.bounds_min_m, dtype=np.float64)
        upper = np.asarray(self.bounds_max_m, dtype=np.float64)
        inside = np.all((flat >= lower - 1.0e-10) & (flat <= upper + 1.0e-10), axis=1)
        output = np.zeros((len(flat), 3), dtype=np.float32)
        first = self.first_center_m
        spacing = np.asarray(self.spacing_m, dtype=np.float64)
        nx, ny, nz = self.dimensions_nx_ny_nz

        def bracket(value: float, axis_first: float, step: float, count: int) -> tuple[int, float]:
            scaled = (value - axis_first) / step
            base = int(np.floor(scaled))
            if base < 0:
                return 0, float(scaled)
            if base >= count - 1:
                return count - 2, float(scaled - (count - 2))
            return base, float(scaled - base)

        values = self.velocity_m_s
        for row, point in enumerate(flat):
            if not inside[row]:
                continue
            ix, tx = bracket(point[0], first[0], spacing[0], nx)
            iy, ty = bracket(point[1], first[1], spacing[1], ny)
            iz, tz = bracket(point[2], first[2], spacing[2], nz)
            c000 = values[iz, iy, ix]
            c100 = values[iz, iy, ix + 1]
            c010 = values[iz, iy + 1, ix]
            c110 = values[iz, iy + 1, ix + 1]
            c001 = values[iz + 1, iy, ix]
            c101 = values[iz + 1, iy, ix + 1]
            c011 = values[iz + 1, iy + 1, ix]
            c111 = values[iz + 1, iy + 1, ix + 1]
            c00 = c000 * (1.0 - tx) + c100 * tx
            c10 = c010 * (1.0 - tx) + c110 * tx
            c01 = c001 * (1.0 - tx) + c101 * tx
            c11 = c011 * (1.0 - tx) + c111 * tx
            c0 = c00 * (1.0 - ty) + c10 * ty
            c1 = c01 * (1.0 - ty) + c11 * ty
            output[row] = c0 * (1.0 - tz) + c1 * tz
        return output.reshape(points.shape), inside.reshape(points.shape[:-1])

    def magnitude_slice(self, *, axis: str = "y", coordinate_m: float = 0.0) -> tuple[np.ndarray, np.ndarray, int]:
        """Return a deterministic cell-centre |U| slice for visualization."""

        axis_index = {"x": 2, "y": 1, "z": 0}.get(axis)
        if axis_index is None:
            raise ValueError("axis must be one of x, y, z")
        if axis == "x":
            coordinates = self.grid_centres_m[0, 0, :, 0]
            index = int(np.argmin(np.abs(coordinates - float(coordinate_m))))
            return self.grid_centres_m[:, :, index, :], self.speed_magnitude_m_s[:, :, index], index
        if axis == "y":
            coordinates = self.grid_centres_m[0, :, 0, 1]
            index = int(np.argmin(np.abs(coordinates - float(coordinate_m))))
            return self.grid_centres_m[:, index, :, :], self.speed_magnitude_m_s[:, index, :], index
        coordinates = self.grid_centres_m[:, 0, 0, 2]
        index = int(np.argmin(np.abs(coordinates - float(coordinate_m))))
        return self.grid_centres_m[index, :, :, :], self.speed_magnitude_m_s[index, :, :], index


def benchmark_points_to_world(points_m: np.ndarray, origin_world_m: Iterable[float], u_world: Iterable[float], v_world: Iterable[float], w_world: Iterable[float]) -> np.ndarray:
    """Map benchmark +X/+Y/+Z into the current local process frame."""

    points = np.asarray(points_m, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("points_m must end in a length-3 dimension")
    origin = np.asarray(tuple(origin_world_m), dtype=np.float64)
    u = _unit(u_world, "u_world")
    v = _unit(v_world, "v_world")
    w = _unit(w_world, "w_world")
    relative = points - BENCHMARK_ORIGIN_M
    return origin + relative[..., 0, None] * u + relative[..., 1, None] * v + relative[..., 2, None] * w


def benchmark_vectors_to_world(vectors_m_s: np.ndarray, u_world: Iterable[float], v_world: Iterable[float], w_world: Iterable[float]) -> np.ndarray:
    """Rotate benchmark velocity vectors into the current process frame."""

    vectors = np.asarray(vectors_m_s, dtype=np.float64)
    if vectors.shape[-1] != 3:
        raise ValueError("vectors_m_s must end in a length-3 dimension")
    u = _unit(u_world, "u_world")
    v = _unit(v_world, "v_world")
    w = _unit(w_world, "w_world")
    return vectors[..., 0, None] * u + vectors[..., 1, None] * v + vectors[..., 2, None] * w


def deterministic_streamlines(field: OpenFOAMFlowField, seeds_m: np.ndarray, *, step_m: float = 0.006, max_steps: int = 48) -> list[np.ndarray]:
    """Integrate deterministic arc-length streamlines from the actual U field."""

    seeds = np.asarray(seeds_m, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError("seeds_m must have shape [N, 3]")
    if step_m <= 0.0 or max_steps < 2:
        raise ValueError("invalid streamline integration settings")
    result: list[np.ndarray] = []
    for seed in seeds:
        point = seed.copy()
        line = [point.copy()]
        for _ in range(max_steps - 1):
            velocity, inside = field.trilinear_velocity(point[None, :])
            if not bool(inside[0]):
                break
            direction = velocity[0].astype(np.float64)
            speed = float(np.linalg.norm(direction))
            if speed <= 1.0e-9:
                break
            direction /= speed
            midpoint = point + 0.5 * step_m * direction
            midpoint_velocity, midpoint_inside = field.trilinear_velocity(midpoint[None, :])
            if not bool(midpoint_inside[0]):
                break
            midpoint_direction = midpoint_velocity[0].astype(np.float64)
            midpoint_speed = float(np.linalg.norm(midpoint_direction))
            if midpoint_speed <= 1.0e-9:
                break
            point = point + step_m * midpoint_direction / midpoint_speed
            if not bool(field.trilinear_velocity(point[None, :])[1][0]):
                break
            line.append(point.copy())
        if len(line) >= 2:
            result.append(np.asarray(line, dtype=np.float64))
    return result
