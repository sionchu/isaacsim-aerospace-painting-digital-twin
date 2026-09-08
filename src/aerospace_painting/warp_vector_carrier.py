"""Compact two-anchor full-vector carrier surrogate for static Warp validation.

The canonical model interpolates the solved 0-degree and 15-degree OpenFOAM
velocity fields in a co-rotating nozzle-local frame.  It is deliberately
small, deterministic, and independent of the moving Isaac runtime.  The
existing :mod:`warp_teacher_flow` trilinear sampler is the single source of
truth for structured-grid interpolation and out-of-field handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .warp_air_field import frame_axes
from .warp_teacher_flow import TeacherFlowGrid, _sha256_file


COROTATING_VECTOR_INTERP = "COROTATING_VECTOR_INTERP"
WORLD_LINEAR_DIAGNOSTIC = "WORLD_LINEAR_DIAGNOSTIC"
MODEL_ID = "warp_vector_carrier_v2"
ANGLE_MIN_DEG = 0.0
ANGLE_MAX_DEG = 15.0


try:  # Native Isaac Sim supplies Warp; portable tests do not.
    import warp as wp
    from .warp_teacher_flow import teacher_flow_inside, teacher_flow_trilinear
except ImportError:  # pragma: no cover - exercised on portable Python only.
    wp = None  # type: ignore[assignment]
    teacher_flow_inside = None  # type: ignore[assignment]
    teacher_flow_trilinear = None  # type: ignore[assignment]


class VectorCarrierError(ValueError):
    """Raised when a two-anchor vector carrier is unsafe or unverifiable."""


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rotation_axes(angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis, y_axis, z_axis = frame_axes(angle_deg)
    return (
        np.asarray(x_axis, dtype=np.float64),
        np.asarray(y_axis, dtype=np.float64),
        np.asarray(z_axis, dtype=np.float64),
    )


def _world_to_local(points_world: np.ndarray, angle_deg: float, origin: np.ndarray) -> np.ndarray:
    x_axis, y_axis, z_axis = _rotation_axes(angle_deg)
    relative = np.asarray(points_world, dtype=np.float64) - origin
    return np.stack(
        (
            np.einsum("...i,i->...", relative, x_axis),
            np.einsum("...i,i->...", relative, y_axis),
            np.einsum("...i,i->...", relative, z_axis),
        ),
        axis=-1,
    )


def _local_to_world(local: np.ndarray, angle_deg: float) -> np.ndarray:
    x_axis, y_axis, z_axis = _rotation_axes(angle_deg)
    values = np.asarray(local, dtype=np.float64)
    return (
        values[..., 0, None] * x_axis
        + values[..., 1, None] * y_axis
        + values[..., 2, None] * z_axis
    )


@dataclass(frozen=True)
class VectorCarrierModel:
    """Two solved full-vector anchors on one shared structured grid."""

    anchor_0: TeacherFlowGrid
    anchor_15: TeacherFlowGrid
    nozzle_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.002)
    model_id: str = MODEL_ID

    def __post_init__(self) -> None:
        if not np.isclose(self.anchor_0.angle_deg, 0.0, rtol=0.0, atol=1.0e-10):
            raise VectorCarrierError("anchor_0 must be the 0-degree field")
        if not np.isclose(self.anchor_15.angle_deg, 15.0, rtol=0.0, atol=1.0e-10):
            raise VectorCarrierError("anchor_15 must be the 15-degree field")
        if self.anchor_0.values_nz_ny_nx_3.shape != self.anchor_15.values_nz_ny_nx_3.shape:
            raise VectorCarrierError("anchor fields must have identical grid shapes")
        if not np.allclose(self.anchor_0.first_center_m, self.anchor_15.first_center_m, rtol=0.0, atol=1.0e-12):
            raise VectorCarrierError("anchor fields must share first centers")
        if not np.allclose(self.anchor_0.spacing_m, self.anchor_15.spacing_m, rtol=0.0, atol=1.0e-12):
            raise VectorCarrierError("anchor fields must share spacing")
        origin = tuple(float(value) for value in self.nozzle_origin_m)
        if len(origin) != 3 or not all(np.isfinite(value) for value in origin):
            raise VectorCarrierError("nozzle origin must be finite and length three")
        object.__setattr__(self, "nozzle_origin_m", origin)

    @property
    def nx(self) -> int:
        return self.anchor_0.nx

    @property
    def ny(self) -> int:
        return self.anchor_0.ny

    @property
    def nz(self) -> int:
        return self.anchor_0.nz

    @property
    def dimensions_nx_ny_nz(self) -> tuple[int, int, int]:
        return self.anchor_0.dimensions_nx_ny_nz

    @property
    def first_center_m(self) -> tuple[float, float, float]:
        return self.anchor_0.first_center_m

    @property
    def spacing_m(self) -> tuple[float, float, float]:
        return self.anchor_0.spacing_m

    @property
    def bounds_min_m(self) -> np.ndarray:
        return self.anchor_0.bounds_min_m

    @property
    def bounds_max_m(self) -> np.ndarray:
        return self.anchor_0.bounds_max_m

    @property
    def flat_anchor_0_vec3f(self) -> np.ndarray:
        return self.anchor_0.flat_values_vec3f

    @property
    def flat_anchor_15_vec3f(self) -> np.ndarray:
        return self.anchor_15.flat_values_vec3f

    @classmethod
    def from_teacher_grids(
        cls,
        anchor_0: TeacherFlowGrid,
        anchor_15: TeacherFlowGrid,
        *,
        nozzle_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.002),
    ) -> "VectorCarrierModel":
        return cls(anchor_0=anchor_0, anchor_15=anchor_15, nozzle_origin_m=nozzle_origin_m)

    def sample(
        self,
        points_world: np.ndarray,
        angle_deg: float,
        *,
        interpolation: str = COROTATING_VECTOR_INTERP,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample the canonical or diagnostic world-linear vector field on CPU."""

        angle = float(angle_deg)
        if angle < ANGLE_MIN_DEG - 1.0e-10 or angle > ANGLE_MAX_DEG + 1.0e-10:
            raise VectorCarrierError("query angle must be within [0, 15] degrees")
        if interpolation not in {COROTATING_VECTOR_INTERP, WORLD_LINEAR_DIAGNOSTIC}:
            raise VectorCarrierError(f"unsupported interpolation method: {interpolation}")
        points = np.asarray(points_world, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("points_world must end in a length-3 dimension")
        flat = points.reshape(-1, 3)
        origin = np.asarray(self.nozzle_origin_m, dtype=np.float64)
        if abs(angle) <= 1.0e-10:
            values, inside = self.anchor_0.trilinear_velocity(flat)
            return values.reshape(points.shape), inside.reshape(points.shape[:-1])
        if abs(angle - ANGLE_MAX_DEG) <= 1.0e-10:
            values, inside = self.anchor_15.trilinear_velocity(flat)
            return values.reshape(points.shape), inside.reshape(points.shape[:-1])

        t = angle / ANGLE_MAX_DEG
        if interpolation == WORLD_LINEAR_DIAGNOSTIC:
            u0_same, inside0 = self.anchor_0.trilinear_velocity(flat)
            u15_same, inside15 = self.anchor_15.trilinear_velocity(flat)
            inside = np.asarray(inside0, dtype=bool) & np.asarray(inside15, dtype=bool)
            output = np.zeros_like(u0_same, dtype=np.float32)
            output[inside] = ((1.0 - t) * u0_same[inside] + t * u15_same[inside]).astype(np.float32)
            return output.reshape(points.shape), inside.reshape(points.shape[:-1])
        q_local = _world_to_local(flat, angle, origin)
        p0_world = origin + _local_to_world(q_local, 0.0)
        p15_world = origin + _local_to_world(q_local, ANGLE_MAX_DEG)
        u0_world, inside0 = self.anchor_0.trilinear_velocity(p0_world)
        u15_world, inside15 = self.anchor_15.trilinear_velocity(p15_world)
        inside = np.asarray(inside0, dtype=bool) & np.asarray(inside15, dtype=bool)
        output = np.zeros_like(u0_world, dtype=np.float32)
        if np.any(inside):
            x0, y0, z0 = _rotation_axes(0.0)
            x15, y15, z15 = _rotation_axes(ANGLE_MAX_DEG)
            u0_local = np.stack(
                (u0_world @ x0, u0_world @ y0, u0_world @ z0),
                axis=-1,
            )
            u15_local = np.stack(
                (u15_world @ x15, u15_world @ y15, u15_world @ z15),
                axis=-1,
            )
            local = (1.0 - t) * u0_local + t * u15_local
            output[inside] = _local_to_world(local[inside], angle).astype(np.float32)
        return output.reshape(points.shape), inside.reshape(points.shape[:-1])

    def runtime_npz_payload(self) -> dict[str, np.ndarray]:
        """Return only arrays required by the runtime sampler."""

        return {
            "U0": np.asarray(self.anchor_0.values_nz_ny_nx_3, dtype=np.float32),
            "U15": np.asarray(self.anchor_15.values_nz_ny_nx_3, dtype=np.float32),
            "first_center_m": np.asarray(self.first_center_m, dtype=np.float64),
            "spacing_m": np.asarray(self.spacing_m, dtype=np.float64),
            "dimensions_nx_ny_nz": np.asarray(self.dimensions_nx_ny_nz, dtype=np.int32),
            "nozzle_origin_m": np.asarray(self.nozzle_origin_m, dtype=np.float64),
            "anchor_angles_deg": np.asarray((0.0, 15.0), dtype=np.float64),
        }

    def save_runtime_npz(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **self.runtime_npz_payload())
        return _sha256_file(target)

    @classmethod
    def from_runtime_npz(
        cls,
        path: str | Path,
        *,
        source_0: dict[str, Any],
        source_15: dict[str, Any],
        openfoam_version: str,
    ) -> "VectorCarrierModel":
        source = Path(path)
        with np.load(source, allow_pickle=False) as payload:
            required = {"U0", "U15", "first_center_m", "spacing_m", "dimensions_nx_ny_nz", "nozzle_origin_m", "anchor_angles_deg"}
            missing = required.difference(payload.files)
            if missing:
                raise VectorCarrierError(f"vector carrier NPZ missing arrays: {sorted(missing)}")
            values0 = np.asarray(payload["U0"], dtype=np.float32)
            values15 = np.asarray(payload["U15"], dtype=np.float32)
            first = tuple(float(value) for value in payload["first_center_m"])
            spacing = tuple(float(value) for value in payload["spacing_m"])
            origin = tuple(float(value) for value in payload["nozzle_origin_m"])
            dims = tuple(int(value) for value in payload["dimensions_nx_ny_nz"])
            angles = np.asarray(payload["anchor_angles_deg"], dtype=float)
        if values0.shape != values15.shape or values0.shape[:3] != (dims[2], dims[1], dims[0]):
            raise VectorCarrierError("vector carrier NPZ dimensions do not match payload")
        if angles.shape != (2,) or not np.allclose(angles, (0.0, 15.0), rtol=0.0, atol=1.0e-10):
            raise VectorCarrierError("vector carrier NPZ anchor angles are invalid")
        common = {
            "first_center_m": first,
            "spacing_m": spacing,
            "source_case": str(source_0.get("source_case", "artifact")),
            "source_c_path": str(source_0["source_c_path"]),
            "source_u_path": str(source_0["source_u_path"]),
            "source_c_sha256": str(source_0["source_c_sha256"]),
            "source_u_sha256": str(source_0["source_u_sha256"]),
            "openfoam_version": openfoam_version,
            "latest_time_s": float(source_0.get("latest_time_s", 0.06)),
        }
        anchor0 = TeacherFlowGrid(angle_deg=0.0, values_nz_ny_nx_3=values0, **common)
        common15 = dict(common)
        common15.update(
            {
                "source_case": str(source_15.get("source_case", "artifact")),
                "source_c_path": str(source_15["source_c_path"]),
                "source_u_path": str(source_15["source_u_path"]),
                "source_c_sha256": str(source_15["source_c_sha256"]),
                "source_u_sha256": str(source_15["source_u_sha256"]),
                "latest_time_s": float(source_15.get("latest_time_s", 0.06)),
            }
        )
        anchor15 = TeacherFlowGrid(angle_deg=15.0, values_nz_ny_nx_3=values15, **common15)
        return cls(anchor_0=anchor0, anchor_15=anchor15, nozzle_origin_m=origin)


def _validate_anchor_metadata(model: VectorCarrierModel, payload: dict[str, Any]) -> None:
    grid = payload.get("grid", {})
    if tuple(int(value) for value in grid.get("dimensions_nx_ny_nz", ())) != model.dimensions_nx_ny_nz:
        raise VectorCarrierError("model JSON grid dimensions mismatch")
    if not np.allclose(grid.get("first_center_m"), model.first_center_m, rtol=0.0, atol=1.0e-12):
        raise VectorCarrierError("model JSON first center mismatch")
    if not np.allclose(grid.get("spacing_m"), model.spacing_m, rtol=0.0, atol=1.0e-12):
        raise VectorCarrierError("model JSON spacing mismatch")
    if not np.allclose(payload.get("nozzle_origin_m"), model.nozzle_origin_m, rtol=0.0, atol=1.0e-12):
        raise VectorCarrierError("model JSON nozzle origin mismatch")


def build_model_payload(
    model: VectorCarrierModel,
    *,
    npz_path: str,
    npz_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    def anchor_payload(grid: TeacherFlowGrid) -> dict[str, Any]:
        return {
            "angle_deg": float(grid.angle_deg),
            "source_case": grid.source_case,
            "source_paths": {"C": grid.source_c_path, "U": grid.source_u_path},
            "source_hashes": {"C": grid.source_c_sha256, "U": grid.source_u_sha256},
            "latest_time_s": float(grid.latest_time_s),
        }

    payload: dict[str, Any] = {
        "schema_version": "warp_vector_carrier_v2",
        "model_id": model.model_id,
        "model_type": "two_anchor_full_vector_structured_grid",
        "angle_range_deg": [ANGLE_MIN_DEG, ANGLE_MAX_DEG],
        "interpolation_method": COROTATING_VECTOR_INTERP,
        "diagnostic_interpolation_method": WORLD_LINEAR_DIAGNOSTIC,
        "grid": {
            "dimensions_nx_ny_nz": list(model.dimensions_nx_ny_nz),
            "array_layout": "U[nz, ny, nx, 3]",
            "first_center_m": list(model.first_center_m),
            "spacing_m": list(model.spacing_m),
            "finite_volume_bounds_min_m": model.bounds_min_m.tolist(),
            "finite_volume_bounds_max_m": model.bounds_max_m.tolist(),
        },
        "nozzle_origin_m": list(model.nozzle_origin_m),
        "frame_convention": {
            "local_axes": "+X fan major/u, +Y fan minor/v, +Z spray/stand-off",
            "rotation": "about world +Y",
            "query_rule": "co-rotating local-frame vector interpolation",
        },
        "anchors": {"0_deg": anchor_payload(model.anchor_0), "15_deg": anchor_payload(model.anchor_15)},
        "openfoam_version": model.anchor_0.openfoam_version,
        "source_commit": source_commit,
        "npz_path": npz_path,
        "npz_sha256": npz_sha256,
        "model_sha256": "",
        "model_json_sha256": "",
    }
    base = dict(payload)
    base.pop("model_sha256")
    base.pop("model_json_sha256")
    payload["model_sha256"] = sha256_bytes(canonical_json_bytes(base))
    without_json_hash = dict(payload)
    without_json_hash.pop("model_json_sha256")
    payload["model_json_sha256"] = sha256_bytes(canonical_json_bytes(without_json_hash))
    return payload


def verify_model_payload(
    model: VectorCarrierModel,
    payload: dict[str, Any],
    *,
    npz_path: str | Path,
    repo_root: str | Path | None = None,
    verify_sources: bool = True,
) -> None:
    _validate_anchor_metadata(model, payload)
    source = Path(npz_path)
    if payload.get("npz_sha256") != _sha256_file(source):
        raise VectorCarrierError("vector carrier NPZ hash mismatch")
    expected_model = dict(payload)
    recorded_model = expected_model.pop("model_sha256", None)
    recorded_json = expected_model.pop("model_json_sha256", None)
    if recorded_model != sha256_bytes(canonical_json_bytes(expected_model)):
        raise VectorCarrierError("vector carrier model hash mismatch")
    with_model = dict(expected_model)
    with_model["model_sha256"] = recorded_model
    if recorded_json != sha256_bytes(canonical_json_bytes(with_model)):
        raise VectorCarrierError("vector carrier JSON provenance hash mismatch")
    if not verify_sources:
        return
    root = Path(repo_root).resolve() if repo_root is not None else None
    for grid, key in ((model.anchor_0, "0_deg"), (model.anchor_15, "15_deg")):
        anchor = payload["anchors"][key]
        for field, expected in anchor["source_hashes"].items():
            path = Path(anchor["source_paths"][field])
            if root is not None and not path.is_absolute():
                path = root / path
            if _sha256_file(path) != expected:
                raise VectorCarrierError(f"vector carrier source hash mismatch: {path}")


def load_model_artifacts(
    json_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    verify_sources: bool = True,
) -> tuple[VectorCarrierModel, dict[str, Any]]:
    """Load and verify the compact model JSON/NPZ pair."""

    json_file = Path(json_path)
    payload = json.loads(json_file.read_text(encoding="utf-8"))
    if payload.get("model_id") != MODEL_ID:
        raise VectorCarrierError("unexpected vector carrier model id")
    npz_file = json_file.parent / str(payload["npz_path"])
    anchors = payload["anchors"]
    model = VectorCarrierModel.from_runtime_npz(
        npz_file,
        source_0={
            "source_case": anchors["0_deg"]["source_case"],
            "source_c_path": anchors["0_deg"]["source_paths"]["C"],
            "source_u_path": anchors["0_deg"]["source_paths"]["U"],
            "source_c_sha256": anchors["0_deg"]["source_hashes"]["C"],
            "source_u_sha256": anchors["0_deg"]["source_hashes"]["U"],
            "latest_time_s": anchors["0_deg"]["latest_time_s"],
        },
        source_15={
            "source_case": anchors["15_deg"]["source_case"],
            "source_c_path": anchors["15_deg"]["source_paths"]["C"],
            "source_u_path": anchors["15_deg"]["source_paths"]["U"],
            "source_c_sha256": anchors["15_deg"]["source_hashes"]["C"],
            "source_u_sha256": anchors["15_deg"]["source_hashes"]["U"],
            "latest_time_s": anchors["15_deg"]["latest_time_s"],
        },
        openfoam_version=str(payload["openfoam_version"]),
    )
    verify_model_payload(model, payload, npz_path=npz_file, repo_root=repo_root, verify_sources=verify_sources)
    return model, payload


if wp is not None:

    @wp.func
    def vector_carrier_sample(
        values0: wp.array(dtype=wp.vec3f),
        values15: wp.array(dtype=wp.vec3f),
        position: wp.vec3f,
        origin: wp.vec3f,
        query_x: wp.vec3f,
        query_y: wp.vec3f,
        query_z: wp.vec3f,
        first_center: wp.vec3f,
        spacing: wp.vec3f,
        nx: int,
        ny: int,
        nz: int,
        interpolation_t: float,
        interpolation_mode: int,
    ) -> wp.vec3f:
        if interpolation_t <= 0.0:
            return teacher_flow_trilinear(values0, position, first_center, spacing, nx, ny, nz)
        if interpolation_t >= 1.0:
            return teacher_flow_trilinear(values15, position, first_center, spacing, nx, ny, nz)
        relative = position - origin
        q_local = wp.vec3f(
            wp.dot(relative, query_x),
            wp.dot(relative, query_y),
            wp.dot(relative, query_z),
        )
        x0 = wp.vec3f(1.0, 0.0, 0.0)
        y0 = wp.vec3f(0.0, 1.0, 0.0)
        z0 = wp.vec3f(0.0, 0.0, 1.0)
        angle15 = 15.0 * 3.141592653589793 / 180.0
        x15 = wp.vec3f(wp.cos(angle15), 0.0, -wp.sin(angle15))
        y15 = wp.vec3f(0.0, 1.0, 0.0)
        z15 = wp.vec3f(wp.sin(angle15), 0.0, wp.cos(angle15))
        p0 = origin + x0 * q_local[0] + y0 * q_local[1] + z0 * q_local[2]
        p15 = origin + x15 * q_local[0] + y15 * q_local[1] + z15 * q_local[2]
        u0_world = teacher_flow_trilinear(values0, p0, first_center, spacing, nx, ny, nz)
        u15_world = teacher_flow_trilinear(values15, p15, first_center, spacing, nx, ny, nz)
        if interpolation_mode == 1:
            u0_same = teacher_flow_trilinear(values0, position, first_center, spacing, nx, ny, nz)
            u15_same = teacher_flow_trilinear(values15, position, first_center, spacing, nx, ny, nz)
            return (1.0 - interpolation_t) * u0_same + interpolation_t * u15_same
        u0_local = wp.vec3f(wp.dot(u0_world, x0), wp.dot(u0_world, y0), wp.dot(u0_world, z0))
        u15_local = wp.vec3f(wp.dot(u15_world, x15), wp.dot(u15_world, y15), wp.dot(u15_world, z15))
        local = (1.0 - interpolation_t) * u0_local + interpolation_t * u15_local
        return query_x * local[0] + query_y * local[1] + query_z * local[2]


    @wp.func
    def vector_carrier_inside(
        position: wp.vec3f,
        origin: wp.vec3f,
        query_x: wp.vec3f,
        query_y: wp.vec3f,
        query_z: wp.vec3f,
        first_center: wp.vec3f,
        spacing: wp.vec3f,
        nx: int,
        ny: int,
        nz: int,
        interpolation_t: float,
        interpolation_mode: int,
    ) -> bool:
        if interpolation_mode == 1:
            return teacher_flow_inside(position, first_center, spacing, nx, ny, nz)
        if interpolation_t <= 0.0 or interpolation_t >= 1.0:
            return teacher_flow_inside(position, first_center, spacing, nx, ny, nz)
        relative = position - origin
        q_local = wp.vec3f(
            wp.dot(relative, query_x),
            wp.dot(relative, query_y),
            wp.dot(relative, query_z),
        )
        angle15 = 15.0 * 3.141592653589793 / 180.0
        x15 = wp.vec3f(wp.cos(angle15), 0.0, -wp.sin(angle15))
        y15 = wp.vec3f(0.0, 1.0, 0.0)
        z15 = wp.vec3f(wp.sin(angle15), 0.0, wp.cos(angle15))
        p0 = origin + wp.vec3f(q_local[0], q_local[1], q_local[2])
        p15 = origin + x15 * q_local[0] + y15 * q_local[1] + z15 * q_local[2]
        return teacher_flow_inside(p0, first_center, spacing, nx, ny, nz) and teacher_flow_inside(p15, first_center, spacing, nx, ny, nz)


    @wp.kernel
    def _sample_vector_carrier_kernel(
        values0: wp.array(dtype=wp.vec3f),
        values15: wp.array(dtype=wp.vec3f),
        positions: wp.array(dtype=wp.vec3f),
        output: wp.array(dtype=wp.vec3f),
        inside: wp.array(dtype=wp.bool),
        origin: wp.vec3f,
        query_x: wp.vec3f,
        query_y: wp.vec3f,
        query_z: wp.vec3f,
        first_center: wp.vec3f,
        spacing: wp.vec3f,
        nx: int,
        ny: int,
        nz: int,
        interpolation_t: float,
        interpolation_mode: int,
    ):
        index = wp.tid()
        position = positions[index]
        inside[index] = vector_carrier_inside(
            position,
            origin,
            query_x,
            query_y,
            query_z,
            first_center,
            spacing,
            nx,
            ny,
            nz,
            interpolation_t,
            interpolation_mode,
        )
        if inside[index]:
            output[index] = vector_carrier_sample(
                values0,
                values15,
                position,
                origin,
                query_x,
                query_y,
                query_z,
                first_center,
                spacing,
                nx,
                ny,
                nz,
                interpolation_t,
                interpolation_mode,
            )
        else:
            output[index] = wp.vec3f(0.0, 0.0, 0.0)


def sample_vector_carrier_warp(
    model: VectorCarrierModel,
    points_world: np.ndarray,
    angle_deg: float,
    *,
    device: str = "cuda:0",
    interpolation: str = COROTATING_VECTOR_INTERP,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the vector carrier on Warp for CPU/GPU agreement checks."""

    if wp is None:
        raise RuntimeError("Warp is not available")
    if interpolation not in {COROTATING_VECTOR_INTERP, WORLD_LINEAR_DIAGNOSTIC}:
        raise VectorCarrierError(f"unsupported interpolation method: {interpolation}")
    wp.init()
    device_obj = wp.get_device(device)
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    values0 = wp.array(model.flat_anchor_0_vec3f, dtype=wp.vec3f, device=device_obj)
    values15 = wp.array(model.flat_anchor_15_vec3f, dtype=wp.vec3f, device=device_obj)
    point_array = wp.array(points, dtype=wp.vec3f, device=device_obj)
    output = wp.zeros(len(points), dtype=wp.vec3f, device=device_obj)
    inside = wp.zeros(len(points), dtype=wp.bool, device=device_obj)
    x_axis, y_axis, z_axis = _rotation_axes(angle_deg)
    wp.launch(
        kernel=_sample_vector_carrier_kernel,
        dim=len(points),
        inputs=[
            values0,
            values15,
            point_array,
            output,
            inside,
            wp.vec3f(*np.asarray(model.nozzle_origin_m, dtype=np.float32)),
            wp.vec3f(*np.asarray(x_axis, dtype=np.float32)),
            wp.vec3f(*np.asarray(y_axis, dtype=np.float32)),
            wp.vec3f(*np.asarray(z_axis, dtype=np.float32)),
            wp.vec3f(*np.asarray(model.first_center_m, dtype=np.float32)),
            wp.vec3f(*np.asarray(model.spacing_m, dtype=np.float32)),
            model.nx,
            model.ny,
            model.nz,
            float(angle_deg / ANGLE_MAX_DEG),
            0 if interpolation == COROTATING_VECTOR_INTERP else 1,
        ],
        device=device_obj,
    )
    wp.synchronize_device(device_obj)
    return output.numpy(), inside.numpy().astype(bool)
