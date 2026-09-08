"""Structured OpenFOAM cell-centre fields and deterministic Warp sampling.

The teacher-forced field is a diagnostic input, not a deployable carrier
model.  OpenFOAM writes cell-centred values, so the interpolation domain is
the finite-volume domain (half a cell beyond the first/last centre).  Within
the half-cell boundary extension, the first/last centre pair is linearly
extrapolated; positions outside the finite-volume domain are explicitly
reported out of field rather than silently clamped.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ROUND_DIGITS = 12
_GRID_TOL = 1.0e-10


try:  # Optional on the normal portable test interpreter.
    import warp as wp
except ImportError:  # pragma: no cover - native Isaac Python supplies Warp.
    wp = None  # type: ignore[assignment]


class TeacherFlowGridError(ValueError):
    """Raised when a solved C/U pair cannot be safely reconstructed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced(text: str, start: int, opening: str = "(", closing: str = ")") -> str:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise TeacherFlowGridError("unclosed OpenFOAM list")


def read_internal_vectors(path: str | Path) -> np.ndarray:
    """Read an ASCII OpenFOAM internal vector field without guessing order."""

    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    marker = text.find("internalField")
    if marker < 0:
        raise TeacherFlowGridError(f"missing internalField in {source}")
    match = re.search(r"nonuniform\s+List<vector>\s+\d+\s*\(", text[marker:])
    if match is None:
        uniform = re.search(r"uniform\s*\(([^)]+)\)", text[marker:])
        if uniform is None:
            raise TeacherFlowGridError(f"unsupported vector field in {source}")
        values = re.findall(rf"({FLOAT})", uniform.group(1))
        if len(values) != 3:
            raise TeacherFlowGridError(f"invalid uniform vector in {source}")
        return np.asarray([values], dtype=float)
    opening = marker + match.end() - 1
    values = re.findall(
        rf"\(\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*\)",
        _balanced(text, opening),
    )
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or result.size == 0:
        raise TeacherFlowGridError(f"invalid vector list in {source}")
    return result


def _latest_time(case_dir: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            candidates.append((float(child.name), child))
        except ValueError:
            continue
    if not candidates:
        raise TeacherFlowGridError(f"no numeric solved time in {case_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _axis_values(values: np.ndarray, axis: int) -> np.ndarray:
    unique = np.unique(np.round(np.asarray(values[:, axis], dtype=float), _ROUND_DIGITS))
    if unique.size < 2:
        raise TeacherFlowGridError(f"axis {axis} has fewer than two centres")
    spacing = np.diff(unique)
    if not np.all(spacing > 0.0) or not np.allclose(spacing, spacing[0], rtol=0.0, atol=_GRID_TOL):
        raise TeacherFlowGridError(f"axis {axis} spacing is not uniform")
    return unique


def reconstruct_structured_grid(centres: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map arbitrary C/U ordering to ``U[nz, ny, nx, 3]`` safely.

    The coordinate lookup is explicit: every C row must map to one unique
    `(z,y,x)` slot and every slot must be filled exactly once.  No assumed
    OpenFOAM list ordering is used.
    """

    c = np.asarray(centres, dtype=float)
    u = np.asarray(velocity, dtype=float)
    if c.ndim != 2 or c.shape[1] != 3 or u.shape != c.shape:
        raise TeacherFlowGridError("C and U must have identical (N,3) shapes")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(u)):
        raise TeacherFlowGridError("C/U contain non-finite values")
    x_values = _axis_values(c, 0)
    y_values = _axis_values(c, 1)
    z_values = _axis_values(c, 2)
    nx, ny, nz = len(x_values), len(y_values), len(z_values)
    if nx * ny * nz != len(c):
        raise TeacherFlowGridError("C does not contain a complete Cartesian product")
    x_index = {float(value): index for index, value in enumerate(x_values)}
    y_index = {float(value): index for index, value in enumerate(y_values)}
    z_index = {float(value): index for index, value in enumerate(z_values)}
    grid = np.empty((nz, ny, nx, 3), dtype=np.float32)
    occupied = np.zeros((nz, ny, nx), dtype=bool)
    for row, (point, vector) in enumerate(zip(c, u)):
        key = tuple(float(value) for value in np.round(point, _ROUND_DIGITS))
        try:
            ix = x_index[key[0]]
            iy = y_index[key[1]]
            iz = z_index[key[2]]
        except KeyError as exc:
            raise TeacherFlowGridError(f"C coordinate cannot be mapped: {key}") from exc
        if occupied[iz, iy, ix]:
            raise TeacherFlowGridError(f"duplicate C coordinate at {key}")
        grid[iz, iy, ix] = vector
        occupied[iz, iy, ix] = True
    if not np.all(occupied):
        raise TeacherFlowGridError("C/U mapping left unfilled grid slots")
    return grid, x_values.astype(float), y_values.astype(float), z_values.astype(float)


def _axis_bracket(value: float, first: float, spacing: float, count: int) -> tuple[int, float]:
    scaled = (float(value) - float(first)) / float(spacing)
    base = int(np.floor(scaled))
    if base < 0:
        return 0, scaled
    if base >= count - 1:
        return count - 2, scaled - float(count - 2)
    return base, scaled - float(base)


@dataclass(frozen=True)
class TeacherFlowGrid:
    """One solved cell-centred OpenFOAM vector field and its grid metadata."""

    angle_deg: float
    values_nz_ny_nx_3: np.ndarray
    first_center_m: tuple[float, float, float]
    spacing_m: tuple[float, float, float]
    source_case: str
    source_c_path: str
    source_u_path: str
    source_c_sha256: str
    source_u_sha256: str
    openfoam_version: str
    latest_time_s: float
    ordering_verified: bool = True

    def __post_init__(self) -> None:
        values = np.asarray(self.values_nz_ny_nx_3, dtype=np.float32)
        if values.ndim != 4 or values.shape[-1] != 3 or min(values.shape[:3]) < 2:
            raise TeacherFlowGridError("teacher U grid must be [nz,ny,nx,3] with each axis >= 2")
        if not np.all(np.isfinite(values)):
            raise TeacherFlowGridError("teacher U grid contains non-finite values")
        first = tuple(float(value) for value in self.first_center_m)
        spacing = tuple(float(value) for value in self.spacing_m)
        if len(first) != 3 or len(spacing) != 3 or not all(value > 0.0 for value in spacing):
            raise TeacherFlowGridError("invalid grid origin or spacing")
        values.setflags(write=False)
        object.__setattr__(self, "values_nz_ny_nx_3", values)
        object.__setattr__(self, "first_center_m", first)
        object.__setattr__(self, "spacing_m", spacing)

    @property
    def nz(self) -> int:
        return int(self.values_nz_ny_nx_3.shape[0])

    @property
    def ny(self) -> int:
        return int(self.values_nz_ny_nx_3.shape[1])

    @property
    def nx(self) -> int:
        return int(self.values_nz_ny_nx_3.shape[2])

    @property
    def dimensions_nx_ny_nz(self) -> tuple[int, int, int]:
        return self.nx, self.ny, self.nz

    @property
    def bounds_min_m(self) -> np.ndarray:
        return np.asarray(self.first_center_m) - 0.5 * np.asarray(self.spacing_m)

    @property
    def bounds_max_m(self) -> np.ndarray:
        return np.asarray(self.first_center_m) + (np.asarray(self.dimensions_nx_ny_nz, dtype=float) - 0.5) * np.asarray(self.spacing_m)

    @property
    def flat_values_vec3f(self) -> np.ndarray:
        return np.ascontiguousarray(self.values_nz_ny_nx_3.reshape(-1, 3), dtype=np.float32)

    @classmethod
    def from_case_dir(cls, case_dir: str | Path, angle_deg: float, *, repo_root: str | Path | None = None) -> "TeacherFlowGrid":
        case = Path(case_dir)
        time_dir = _latest_time(case)
        c_path = time_dir / "C"
        u_path = time_dir / "U"
        centres = read_internal_vectors(c_path)
        velocity = read_internal_vectors(u_path)
        grid, x_values, y_values, z_values = reconstruct_structured_grid(centres, velocity)
        header = c_path.read_text(encoding="utf-8", errors="replace")[:1200]
        version_match = re.search(r"Version:\s*v?([0-9]+)", header, re.IGNORECASE)
        version = f"v{version_match.group(1)}" if version_match else "unknown"
        root = Path(repo_root).resolve() if repo_root is not None else None

        def display(path: Path) -> str:
            return str(path.resolve().relative_to(root)) if root is not None else str(path.resolve())

        return cls(
            angle_deg=float(angle_deg),
            values_nz_ny_nx_3=grid,
            first_center_m=(float(x_values[0]), float(y_values[0]), float(z_values[0])),
            spacing_m=(float(np.diff(x_values)[0]), float(np.diff(y_values)[0]), float(np.diff(z_values)[0])),
            source_case=display(case),
            source_c_path=display(c_path),
            source_u_path=display(u_path),
            source_c_sha256=_sha256_file(c_path),
            source_u_sha256=_sha256_file(u_path),
            openfoam_version=version,
            latest_time_s=float(time_dir.name),
        )

    def to_npz(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            U=self.values_nz_ny_nx_3,
            first_center_m=np.asarray(self.first_center_m, dtype=np.float64),
            spacing_m=np.asarray(self.spacing_m, dtype=np.float64),
            dimensions_nx_ny_nz=np.asarray(self.dimensions_nx_ny_nz, dtype=np.int32),
            angle_deg=np.asarray([self.angle_deg], dtype=np.float64),
            latest_time_s=np.asarray([self.latest_time_s], dtype=np.float64),
            source_case=np.asarray(self.source_case),
            source_c_path=np.asarray(self.source_c_path),
            source_u_path=np.asarray(self.source_u_path),
            source_c_sha256=np.asarray(self.source_c_sha256),
            source_u_sha256=np.asarray(self.source_u_sha256),
            openfoam_version=np.asarray(self.openfoam_version),
        )

    def manifest_entry(self, npz_path: str | Path | None = None) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "angle_deg": self.angle_deg,
            "dimensions_nx_ny_nz": list(self.dimensions_nx_ny_nz),
            "array_layout": "U[nz, ny, nx, 3]",
            "first_center_m": list(self.first_center_m),
            "spacing_m": list(self.spacing_m),
            "finite_volume_bounds_min_m": self.bounds_min_m.tolist(),
            "finite_volume_bounds_max_m": self.bounds_max_m.tolist(),
            "source_case": self.source_case,
            "source_c_path": self.source_c_path,
            "source_u_path": self.source_u_path,
            "source_c_sha256": self.source_c_sha256,
            "source_u_sha256": self.source_u_sha256,
            "openfoam_version": self.openfoam_version,
            "latest_time_s": self.latest_time_s,
            "ordering_verified": self.ordering_verified,
        }
        if npz_path is not None:
            entry["npz_path"] = str(npz_path)
            entry["npz_sha256"] = _sha256_file(Path(npz_path))
        return entry

    def trilinear_velocity(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """CPU reference interpolation and explicit in-field mask."""

        points = np.asarray(points_world, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("points_world must end in a length-3 dimension")
        flat_points = points.reshape(-1, 3)
        lower = self.bounds_min_m
        upper = self.bounds_max_m
        inside = np.all((flat_points >= lower - _GRID_TOL) & (flat_points <= upper + _GRID_TOL), axis=1)
        output = np.zeros((flat_points.shape[0], 3), dtype=np.float32)
        values = self.values_nz_ny_nx_3
        for row, point in enumerate(flat_points):
            if not inside[row]:
                continue
            ix, tx = _axis_bracket(point[0], self.first_center_m[0], self.spacing_m[0], self.nx)
            iy, ty = _axis_bracket(point[1], self.first_center_m[1], self.spacing_m[1], self.ny)
            iz, tz = _axis_bracket(point[2], self.first_center_m[2], self.spacing_m[2], self.nz)
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


if wp is not None:

    @wp.func
    def teacher_flow_inside(
        position: wp.vec3f,
        first_center: wp.vec3f,
        spacing: wp.vec3f,
        nx: int,
        ny: int,
        nz: int,
    ) -> bool:
        lower = wp.vec3f(
            first_center[0] - 0.5 * spacing[0],
            first_center[1] - 0.5 * spacing[1],
            first_center[2] - 0.5 * spacing[2],
        )
        upper = wp.vec3f(
            first_center[0] + (float(nx) - 0.5) * spacing[0],
            first_center[1] + (float(ny) - 0.5) * spacing[1],
            first_center[2] + (float(nz) - 0.5) * spacing[2],
        )
        return (
            position[0] >= lower[0]
            and position[0] <= upper[0]
            and position[1] >= lower[1]
            and position[1] <= upper[1]
            and position[2] >= lower[2]
            and position[2] <= upper[2]
        )


    @wp.func
    def teacher_flow_trilinear(
        values: wp.array(dtype=wp.vec3f),
        position: wp.vec3f,
        first_center: wp.vec3f,
        spacing: wp.vec3f,
        nx: int,
        ny: int,
        nz: int,
    ) -> wp.vec3f:
        sx = (position[0] - first_center[0]) / spacing[0]
        sy = (position[1] - first_center[1]) / spacing[1]
        sz = (position[2] - first_center[2]) / spacing[2]
        fx = wp.floor(sx)
        fy = wp.floor(sy)
        fz = wp.floor(sz)
        ix = int(fx)
        iy = int(fy)
        iz = int(fz)
        tx = sx - fx
        ty = sy - fy
        tz = sz - fz
        if ix < 0:
            ix = 0
            tx = sx
        elif ix >= nx - 1:
            ix = nx - 2
            tx = sx - float(nx - 2)
        if iy < 0:
            iy = 0
            ty = sy
        elif iy >= ny - 1:
            iy = ny - 2
            ty = sy - float(ny - 2)
        if iz < 0:
            iz = 0
            tz = sz
        elif iz >= nz - 1:
            iz = nz - 2
            tz = sz - float(nz - 2)
        i000 = (iz * ny + iy) * nx + ix
        i100 = i000 + 1
        i010 = i000 + nx
        i110 = i010 + 1
        i001 = i000 + nx * ny
        i101 = i001 + 1
        i011 = i001 + nx
        i111 = i011 + 1
        c000 = values[i000]
        c100 = values[i100]
        c010 = values[i010]
        c110 = values[i110]
        c001 = values[i001]
        c101 = values[i101]
        c011 = values[i011]
        c111 = values[i111]
        c00 = c000 * (1.0 - tx) + c100 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        return c0 * (1.0 - tz) + c1 * tz


    @wp.kernel
    def _sample_teacher_flow_kernel(
        values: wp.array(dtype=wp.vec3f),
        positions: wp.array(dtype=wp.vec3f),
        output: wp.array(dtype=wp.vec3f),
        inside: wp.array(dtype=wp.bool),
        first_center: wp.vec3f,
        spacing: wp.vec3f,
        nx: int,
        ny: int,
        nz: int,
    ):
        index = wp.tid()
        position = positions[index]
        inside[index] = teacher_flow_inside(position, first_center, spacing, nx, ny, nz)
        if inside[index]:
            output[index] = teacher_flow_trilinear(values, position, first_center, spacing, nx, ny, nz)
        else:
            output[index] = wp.vec3f(0.0, 0.0, 0.0)


def sample_teacher_flow_warp(grid: TeacherFlowGrid, points_world: np.ndarray, *, device: str = "cuda:0") -> tuple[np.ndarray, np.ndarray]:
    """Sample a grid on CUDA for CPU/Warp agreement checks."""

    if wp is None:
        raise RuntimeError("Warp is not available")
    wp.init()
    device_obj = wp.get_device(device)
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    values = wp.array(grid.flat_values_vec3f, dtype=wp.vec3f, device=device_obj)
    point_array = wp.array(points, dtype=wp.vec3f, device=device_obj)
    output = wp.zeros(len(points), dtype=wp.vec3f, device=device_obj)
    inside = wp.zeros(len(points), dtype=wp.bool, device=device_obj)
    wp.launch(
        kernel=_sample_teacher_flow_kernel,
        dim=len(points),
        inputs=[
            values,
            point_array,
            output,
            inside,
            wp.vec3f(*np.asarray(grid.first_center_m, dtype=np.float32)),
            wp.vec3f(*np.asarray(grid.spacing_m, dtype=np.float32)),
            grid.nx,
            grid.ny,
            grid.nz,
        ],
        device=device_obj,
    )
    wp.synchronize_device(device_obj)
    return output.numpy(), inside.numpy().astype(bool)
