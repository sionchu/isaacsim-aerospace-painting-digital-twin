"""Portable full-panel serpentine planner and film-thickness contract.

This module intentionally has no Isaac Sim imports.  It owns the deterministic
panel sampling, local surface frame, pass-spacing search, finite-surface S2
quadrature, and the WFT/optional illustrative DFT statistics used by the
native adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .s2_runtime import (
    FiniteSurfaceAccumulator,
    S2Runtime,
    S2RuntimeModel,
    StructuredSurfaceGrid,
    surface_frame,
)


DEFAULT_Y_RANGE = (-1.5, 1.5)
DEFAULT_Z_RANGE = (1.65, 2.55)
DEFAULT_ANALYSIS_SHAPE = (121, 37)
DEFAULT_FAN_INCIDENCE_DEG = 7.5
DEFAULT_STAND_OFF_M = 0.24
DEFAULT_MASS_FLOW_KG_S = 1.0e-4
DEFAULT_SPEEDS_M_S = (0.25, 0.5, 0.75, 1.0)
DEFAULT_OVERLAPS = (0.40, 0.50, 0.60, 0.65)
DEFAULT_FPS = 30.0


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unit(value: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(value), dtype=float)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or norm <= 1.0e-12:
        raise ValueError("expected a finite non-zero 3-vector")
    return vector / norm


def _tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _panel_points_from_usda(path: Path) -> np.ndarray:
    """Read only the point3f array from the existing generic panel USD."""

    text = path.read_text(encoding="utf-8")
    start = text.find("point3f[] points = [")
    if start < 0:
        raise ValueError(f"panel USD has no point3f[] points array: {path}")
    uniform = text.find("uniform token", start)
    if uniform < 0:
        raise ValueError(f"panel USD point array has no terminator: {path}")
    end = text.rfind("]", start, uniform)
    if end < 0:
        raise ValueError(f"panel USD point array is malformed: {path}")
    block = text[start:end]
    matches = re.findall(
        r"\(([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?),\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?),\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)",
        block,
    )
    points = np.asarray(matches, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 16:
        raise ValueError(f"panel USD point array is unexpectedly small: {path}")
    return points


def _interpolator(points: np.ndarray):
    try:
        from scipy.interpolate import LinearNDInterpolator
    except ImportError as exc:  # pragma: no cover - native workstation dependency
        raise RuntimeError("full-panel planning requires scipy for the existing panel interpolation") from exc
    return LinearNDInterpolator(points[:, 1:3], points[:, 0], fill_value=np.nan)


@dataclass
class PanelSurface:
    """A dense structured view of the actual generic panel subregion."""

    y_values: np.ndarray
    z_values: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    tangent_long: np.ndarray
    tangent_cross: np.ndarray
    area_weights_m2: np.ndarray
    source_path: str
    source_sha256: str

    @classmethod
    def from_usda(
        cls,
        path: str | Path,
        *,
        y_range: tuple[float, float] = DEFAULT_Y_RANGE,
        z_range: tuple[float, float] = DEFAULT_Z_RANGE,
        shape: tuple[int, int] = DEFAULT_ANALYSIS_SHAPE,
        normal_step_m: float = 0.12,
    ) -> "PanelSurface":
        asset = Path(path)
        points = _panel_points_from_usda(asset)
        interpolate_x = _interpolator(points)
        y_values = np.linspace(float(y_range[0]), float(y_range[1]), int(shape[0]))
        z_values = np.linspace(float(z_range[0]), float(z_range[1]), int(shape[1]))
        position_grid = np.empty((len(y_values), len(z_values), 3), dtype=float)
        normal_grid = np.empty_like(position_grid)
        long_grid = np.empty_like(position_grid)
        cross_grid = np.empty_like(position_grid)

        def sample(y: float, z: float) -> tuple[np.ndarray, np.ndarray]:
            x = float(interpolate_x(float(y), float(z)))
            if not np.isfinite(x):
                raise ValueError(f"panel interpolation left the mesh at y={y}, z={z}")
            x_plus = float(interpolate_x(float(y) + normal_step_m, float(z)))
            x_minus = float(interpolate_x(float(y) - normal_step_m, float(z)))
            z_plus = float(interpolate_x(float(y), float(z) + normal_step_m))
            z_minus = float(interpolate_x(float(y), float(z) - normal_step_m))
            if not all(np.isfinite(value) for value in (x_plus, x_minus, z_plus, z_minus)):
                raise ValueError(f"panel finite-difference normal left the mesh at y={y}, z={z}")
            dy = (x_plus - x_minus) / (2.0 * normal_step_m)
            dz = (z_plus - z_minus) / (2.0 * normal_step_m)
            normal = _unit((-1.0, dy, dz))
            return np.asarray((x, y, z), dtype=float), normal

        for iy, y in enumerate(y_values):
            for iz, z in enumerate(z_values):
                position, normal = sample(float(y), float(z))
                position_grid[iy, iz] = position
                normal_grid[iy, iz] = normal
                # The derivatives are used only to orient the local frame and
                # overscan; a centred finite difference keeps the helper and
                # planner on the same panel geometry contract.
                p_y0 = sample(float(y) - min(normal_step_m * 0.05, 0.005), float(z))[0]
                p_y1 = sample(float(y) + min(normal_step_m * 0.05, 0.005), float(z))[0]
                p_z0 = sample(float(y), float(z) - min(normal_step_m * 0.05, 0.005))[0]
                p_z1 = sample(float(y), float(z) + min(normal_step_m * 0.05, 0.005))[0]
                long_grid[iy, iz] = _unit(p_y1 - p_y0)
                cross_grid[iy, iz] = _unit(p_z1 - p_z0)
        grid = StructuredSurfaceGrid.from_structured(
            position_grid.reshape(-1, 3), normal_grid.reshape(-1, 3), (len(y_values), len(z_values))
        )
        return cls(
            y_values=y_values,
            z_values=z_values,
            positions=position_grid,
            normals=normal_grid,
            tangent_long=long_grid,
            tangent_cross=cross_grid,
            area_weights_m2=grid.area_weights_m2.reshape(position_grid.shape[:2]),
            source_path=asset.as_posix(),
            source_sha256=sha256_file(asset),
        )

    @classmethod
    def synthetic(cls, *, shape: tuple[int, int] = (9, 7)) -> "PanelSurface":
        """Small deterministic curved surface used by portable unit tests."""

        y_values = np.linspace(-0.4, 0.4, shape[0])
        z_values = np.linspace(-0.3, 0.3, shape[1])
        positions = np.asarray([[(0.08 * y * y), y, z] for y in y_values for z in z_values]).reshape(shape[0], shape[1], 3)
        normals = np.asarray([[(-1.0, 0.16 * y, 0.0) for z in z_values] for y in y_values], dtype=float)
        normals /= np.linalg.norm(normals, axis=2, keepdims=True)
        tangent_long = np.asarray([[_unit((0.16 * y, 1.0, 0.0)) for z in z_values] for y in y_values])
        tangent_cross = np.broadcast_to(np.asarray((0.0, 0.0, 1.0)), positions.shape).copy()
        grid = StructuredSurfaceGrid.from_structured(positions.reshape(-1, 3), normals.reshape(-1, 3), shape)
        return cls(
            y_values=y_values,
            z_values=z_values,
            positions=positions,
            normals=normals,
            tangent_long=tangent_long,
            tangent_cross=tangent_cross,
            area_weights_m2=grid.area_weights_m2.reshape(shape),
            source_path="synthetic-test-surface",
            source_sha256="synthetic",
        )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.positions.shape[:2])

    @property
    def total_area_m2(self) -> float:
        return float(np.sum(self.area_weights_m2))

    def sample(self, y: float, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        iy = int(np.clip(np.searchsorted(self.y_values, float(y)), 1, len(self.y_values) - 1))
        iz = int(np.clip(np.searchsorted(self.z_values, float(z)), 1, len(self.z_values) - 1))
        y0, y1 = self.y_values[iy - 1], self.y_values[iy]
        z0, z1 = self.z_values[iz - 1], self.z_values[iz]
        ty = 0.0 if y1 == y0 else (float(y) - y0) / (y1 - y0)
        tz = 0.0 if z1 == z0 else (float(z) - z0) / (z1 - z0)

        def bilinear(values: np.ndarray) -> np.ndarray:
            a = values[iy - 1, iz - 1] * (1.0 - ty) + values[iy, iz - 1] * ty
            b = values[iy - 1, iz] * (1.0 - ty) + values[iy, iz] * ty
            return a * (1.0 - tz) + b * tz

        position = bilinear(self.positions)
        normal = _unit(bilinear(self.normals))
        tangent_long = _unit(bilinear(self.tangent_long))
        tangent_cross = _unit(bilinear(self.tangent_cross))
        return position, normal, tangent_long, tangent_cross

    def frame(self, y: float, z: float, incidence_deg: float = DEFAULT_FAN_INCIDENCE_DEG):
        position, normal, tangent_long, tangent_cross = self.sample(y, z)
        # Choose u so that v = cross(w, u) follows increasing-y travel.
        candidate = tangent_cross
        w_inward = -normal
        if float(np.dot(np.cross(w_inward, candidate), tangent_long)) < 0.0:
            candidate = -candidate
        spray_axis = math.cos(math.radians(incidence_deg)) * w_inward + math.sin(math.radians(incidence_deg)) * candidate
        frame = surface_frame(normal, candidate, spray_axis, minor_tolerance_deg=1.0)
        return position, normal, tangent_long, frame


@dataclass(frozen=True)
class PathPose:
    time_s: float
    y: float
    z: float
    surface_position: np.ndarray
    normal: np.ndarray
    tangent_long: np.ndarray
    frame: Any
    spray_on: bool
    pass_index: int | None
    overscan_offset_m: float


@dataclass
class FullPanelPlan:
    surface: PanelSurface
    fan_incidence_deg: float
    stand_off_m: float
    mass_flow_kg_s: float
    overlap_fraction: float
    requested_spacing_m: float
    spacing_m: float
    speed_m_s: float
    overscan_m: float
    passes: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    duration_s: float
    plan_version: str = "full_panel_plan_v1"

    @property
    def pass_count(self) -> int:
        return len(self.passes)

    @property
    def spray_time_s(self) -> float:
        return float(sum(float(item["duration_s"]) for item in self.passes))

    @property
    def path_length_m(self) -> float:
        return float(sum(float(item["length_m"]) for item in self.passes))

    def _segment_at(self, time_s: float) -> dict[str, Any]:
        t = float(np.clip(time_s, 0.0, self.duration_s))
        for segment in self.segments:
            if t <= float(segment["end_s"]) + 1.0e-12:
                return segment
        return self.segments[-1]

    def pose_at(self, time_s: float) -> PathPose:
        segment = self._segment_at(time_s)
        start = float(segment["start_s"])
        end = float(segment["end_s"])
        fraction = 0.0 if end <= start else float(np.clip((float(time_s) - start) / (end - start), 0.0, 1.0))
        kind = str(segment["kind"])
        if kind == "pass":
            direction = int(segment["direction"])
            span = float(self.surface.y_values[-1] - self.surface.y_values[0])
            s = -self.overscan_m + fraction * (span + 2.0 * self.overscan_m)
            base = float(np.clip(s, 0.0, span))
            y = float(self.surface.y_values[0] + base) if direction > 0 else float(self.surface.y_values[-1] - base)
            z = float(segment["z"])
            position, normal, tangent_long, frame = self.surface.frame(y, z, self.fan_incidence_deg)
            offset = float(direction * (s - base))
            surface_position = position + tangent_long * offset
            return PathPose(float(time_s), y, z, surface_position, normal, tangent_long, frame, True, int(segment["pass_index"]), offset)
        if kind == "step":
            y = float(segment["y0"] + fraction * (segment["y1"] - segment["y0"]))
            z = float(segment["z0"] + fraction * (segment["z1"] - segment["z0"]))
            y_clamped = float(np.clip(y, self.surface.y_values[0], self.surface.y_values[-1]))
            z_clamped = float(np.clip(z, self.surface.z_values[0], self.surface.z_values[-1]))
            position, normal, tangent_long, frame = self.surface.frame(y_clamped, z_clamped, self.fan_incidence_deg)
            return PathPose(float(time_s), y_clamped, z_clamped, position, normal, tangent_long, frame, False, None, 0.0)
        if kind == "depart":
            last_pass = self.passes[-1]
            z = float(last_pass["z"])
            y = float(self.surface.y_values[0] if int(last_pass["direction"]) < 0 else self.surface.y_values[-1])
        else:
            first_pass = self.passes[0]
            z = float(first_pass["z"])
            y = float(self.surface.y_values[-1] if int(first_pass["direction"]) < 0 else self.surface.y_values[0])
        position, normal, tangent_long, frame = self.surface.frame(y, z, self.fan_incidence_deg)
        return PathPose(float(time_s), y, z, position, normal, tangent_long, frame, False, None, 0.0)

    def to_dict(self, *, model: S2RuntimeModel | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.plan_version,
            "surface": {
                "source_path": self.surface.source_path,
                "source_sha256": self.surface.source_sha256,
                "shape": list(self.surface.shape),
                "y_range_m": [float(self.surface.y_values[0]), float(self.surface.y_values[-1])],
                "z_range_m": [float(self.surface.z_values[0]), float(self.surface.z_values[-1])],
                "area_m2": self.surface.total_area_m2,
                "positions_m": self.surface.positions.reshape(-1, 3).tolist(),
                "normals": self.surface.normals.reshape(-1, 3).tolist(),
                "area_weights_m2": self.surface.area_weights_m2.reshape(-1).tolist(),
            },
            "process": {
                "fan_incidence_deg": self.fan_incidence_deg,
                "stand_off_m": self.stand_off_m,
                "mass_flow_kg_s": self.mass_flow_kg_s,
                "speed_m_s": self.speed_m_s,
                "overlap_fraction": self.overlap_fraction,
                "requested_spacing_m": self.requested_spacing_m,
                "spacing_m": self.spacing_m,
                "overscan_m": self.overscan_m,
                "path_length_m": self.path_length_m,
                "spray_time_s": self.spray_time_s,
                "cycle_duration_s": self.duration_s,
                "pass_count": self.pass_count,
            },
            "passes": self.passes,
            "segments": self.segments,
        }
        if model is not None:
            payload["s2_model"] = {
                "model_id": model.model_id,
                "model_sha256": model.model_sha256,
                "openfoam_version": model.openfoam_version,
            }
        return payload


def _build_segments(
    surface: PanelSurface,
    *,
    spacing_m: float,
    overlap_fraction: float,
    speed_m_s: float,
    overscan_m: float,
    fan_incidence_deg: float,
    stand_off_m: float,
    mass_flow_kg_s: float,
) -> FullPanelPlan:
    z_min, z_max = float(surface.z_values[0]), float(surface.z_values[-1])
    y_min, y_max = float(surface.y_values[0]), float(surface.y_values[-1])
    span_z = z_max - z_min
    pass_count = max(2, int(math.ceil(span_z / spacing_m)) + 1)
    z_values = np.linspace(z_min, z_max, pass_count)
    actual_spacing = float(z_values[1] - z_values[0])
    passes: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    cursor = 0.0
    # A short non-spraying approach keeps the native motion continuous.
    approach = 2.0
    segments.append({"kind": "approach", "start_s": cursor, "end_s": cursor + approach, "spray_on": False})
    cursor += approach
    for index, z in enumerate(z_values):
        direction = 1 if index % 2 == 0 else -1
        length = (y_max - y_min) + 2.0 * overscan_m
        duration = length / float(speed_m_s)
        start_s = cursor
        end_s = cursor + duration
        passes.append({
            "pass_index": index,
            "z": float(z),
            "direction": direction,
            "length_m": float(length),
            "duration_s": float(duration),
            "start_s": start_s,
            "end_s": end_s,
        })
        segments.append({
            "kind": "pass",
            "pass_index": index,
            "z": float(z),
            "direction": direction,
            "start_s": start_s,
            "end_s": end_s,
            "spray_on": True,
        })
        cursor = end_s
        if index < len(z_values) - 1:
            # Step-over follows the edge endpoint in the same overscan frame.
            step_duration = actual_spacing / float(speed_m_s)
            next_direction = -direction
            endpoint_y = (y_max if direction > 0 else y_min) + overscan_m * direction
            segments.append({
                "kind": "step",
                "start_s": cursor,
                "end_s": cursor + step_duration,
                "y0": float(endpoint_y),
                "y1": float(endpoint_y),
                "z0": float(z),
                "z1": float(z_values[index + 1]),
                "spray_on": False,
                "from_pass": index,
                "to_pass": index + 1,
                "next_direction": next_direction,
            })
            cursor += step_duration
    segments.append({"kind": "depart", "start_s": cursor, "end_s": cursor + approach, "spray_on": False})
    cursor += approach
    return FullPanelPlan(
        surface=surface,
        fan_incidence_deg=fan_incidence_deg,
        stand_off_m=stand_off_m,
        mass_flow_kg_s=mass_flow_kg_s,
        overlap_fraction=overlap_fraction,
        requested_spacing_m=float(spacing_m),
        spacing_m=actual_spacing,
        speed_m_s=float(speed_m_s),
        overscan_m=float(overscan_m),
        passes=passes,
        segments=segments,
        duration_s=float(cursor),
    )


def plan_full_panel(
    surface: PanelSurface,
    model: S2RuntimeModel,
    *,
    fan_incidence_deg: float = DEFAULT_FAN_INCIDENCE_DEG,
    stand_off_m: float = DEFAULT_STAND_OFF_M,
    mass_flow_kg_s: float = DEFAULT_MASS_FLOW_KG_S,
    overlaps: Iterable[float] = DEFAULT_OVERLAPS,
    speeds_m_s: Iterable[float] = DEFAULT_SPEEDS_M_S,
    overscan_sigma_multiplier: float = 1.5,
    target_duration_s: tuple[float, float] = (90.0, 150.0),
    fps: float = DEFAULT_FPS,
) -> tuple[FullPanelPlan, dict[str, Any], list[dict[str, Any]]]:
    model.validate_operating_point(stand_off_m=stand_off_m, air_velocity_m_s=model.air_velocity_m_s, mass_flow_kg_s=mass_flow_kg_s)
    moments = model.predict(fan_incidence_deg)
    sigma_u = math.sqrt(float(moments.covariance_uv_m2[0, 0]))
    fwhm_u = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_u
    sigma_v = math.sqrt(float(moments.covariance_uv_m2[1, 1]))
    candidates: list[dict[str, Any]] = []
    for overlap in overlaps:
        overlap = float(overlap)
        requested_spacing = fwhm_u * (1.0 - overlap)
        overscan = overscan_sigma_multiplier * sigma_v
        for speed in speeds_m_s:
            speed = float(speed)
            if not (0.25 <= speed <= 1.0):
                continue
            plan = _build_segments(
                surface,
                spacing_m=requested_spacing,
                overlap_fraction=overlap,
                speed_m_s=speed,
                overscan_m=overscan,
                fan_incidence_deg=fan_incidence_deg,
                stand_off_m=stand_off_m,
                mass_flow_kg_s=mass_flow_kg_s,
            )
            metrics = simulate_plan(plan, model, fps=fps)
            duration = plan.duration_s
            in_window = target_duration_s[0] <= duration <= target_duration_s[1]
            candidates.append({
                "overlap_fraction": overlap,
                "speed_m_s": speed,
                "requested_spacing_m": requested_spacing,
                "duration_s": duration,
                "in_duration_window": in_window,
                "cv": metrics["wft"]["cv"],
                "mean_wft_um": metrics["wft"]["mean_um"],
                "painted_fraction": metrics["wft"]["painted_fraction"],
                "pass_count": plan.pass_count,
                "plan": plan,
                "metrics": metrics,
            })
    if not candidates:
        raise RuntimeError("planner produced no candidates")
    in_window = [row for row in candidates if row["in_duration_window"]]
    pool = in_window or candidates
    chosen = min(pool, key=lambda row: (float(row["cv"]), abs(float(row["duration_s"]) - 120.0), float(row["overlap_fraction"]), float(row["speed_m_s"])))
    selected = chosen["plan"]
    summary = {
        "schema_version": "full_panel_planner_metrics_v1",
        "selected": {key: value for key, value in chosen.items() if key not in {"plan", "metrics"}},
        "s2": {
            "model_id": model.model_id,
            "model_sha256": model.model_sha256,
            "fan_incidence_deg": fan_incidence_deg,
            "sigma_u_m": sigma_u,
            "sigma_v_m": sigma_v,
            "fwhm_u_m": fwhm_u,
            "stand_off_m": stand_off_m,
            "mass_flow_kg_s": mass_flow_kg_s,
        },
        "surface": {"source_path": surface.source_path, "source_sha256": surface.source_sha256, "area_m2": surface.total_area_m2, "shape": list(surface.shape)},
        "candidates": [{key: value for key, value in row.items() if key not in {"plan", "metrics"}} for row in candidates],
        "selected_metrics": chosen["metrics"],
    }
    return selected, summary, candidates


def _stats(values_um: np.ndarray, masses: np.ndarray, area_weights: np.ndarray, density_kg_m3: float) -> dict[str, Any]:
    values = np.asarray(values_um, dtype=float).reshape(-1)
    masses = np.asarray(masses, dtype=float).reshape(-1)
    areas = np.asarray(area_weights, dtype=float).reshape(-1)
    total_area = float(np.sum(areas))
    mean = float(np.sum(masses) / total_area / density_kg_m3 * 1.0e6) if total_area > 0.0 else 0.0
    positive = values[values > 0.0]
    weighted_var = float(np.sum(areas * (values - mean) ** 2) / total_area) if total_area > 0.0 else 0.0
    cv = math.sqrt(max(weighted_var, 0.0)) / mean if mean > 0.0 else 0.0
    band = (values >= 0.8 * mean) & (values <= 1.2 * mean) if mean > 0.0 else np.zeros_like(values, dtype=bool)
    threshold = values >= 0.10 * mean if mean > 0.0 else np.zeros_like(values, dtype=bool)
    return {
        "min_um": float(np.min(values)) if len(values) else 0.0,
        "max_um": float(np.max(values)) if len(values) else 0.0,
        "mean_um": mean,
        "std_um": float(math.sqrt(max(weighted_var, 0.0))),
        "cv": float(cv),
        "p05_um": float(np.percentile(positive, 5.0)) if len(positive) else 0.0,
        "p50_um": float(np.percentile(positive, 50.0)) if len(positive) else 0.0,
        "p95_um": float(np.percentile(positive, 95.0)) if len(positive) else 0.0,
        "fraction_within_plus_minus_20pct": float(np.mean(band)),
        "painted_fraction": float(np.mean(threshold)),
        "raw_nonzero_fraction": float(np.mean(values > 0.0)),
        "min_nonzero_um": float(np.min(positive)) if len(positive) else 0.0,
        "total_mass_kg": float(np.sum(masses)),
        "area_m2": total_area,
    }


def film_statistics(
    cumulative_mass_kg: np.ndarray,
    area_weights_m2: np.ndarray,
    *,
    liquid_density_kg_m3: float,
    volume_solids_fraction: float | None = None,
) -> dict[str, Any]:
    masses = np.asarray(cumulative_mass_kg, dtype=float)
    areas = np.asarray(area_weights_m2, dtype=float)
    if masses.shape != areas.shape or not np.all(areas > 0.0):
        raise ValueError("mass and area arrays must have the same positive shape")
    areal_mass = masses / areas
    wft_um = areal_mass / float(liquid_density_kg_m3) * 1.0e6
    result: dict[str, Any] = {
        "areal_mass_kg_m2": areal_mass.tolist(),
        "wft_um": _stats(wft_um, masses, areas, float(liquid_density_kg_m3)),
        "liquid_density_kg_m3": float(liquid_density_kg_m3),
        "wft_is_estimated": True,
    }
    if volume_solids_fraction is not None:
        fraction = float(volume_solids_fraction)
        dft_um = wft_um * fraction
        result["dft_um"] = _stats(dft_um, masses * fraction, areas, float(liquid_density_kg_m3))
        result["dft_um"]["assumed_volume_solids_fraction"] = fraction
        result["dft_um"]["label"] = f"Illustrative DFT estimate — assumed {fraction * 100.0:.0f}% volume solids"
    return result


def simulate_plan(plan: FullPanelPlan, model: S2RuntimeModel, *, fps: float = DEFAULT_FPS, liquid_density_kg_m3: float = 1000.0, volume_solids_fraction: float | None = None) -> dict[str, Any]:
    grid = StructuredSurfaceGrid.from_structured(
        plan.surface.positions.reshape(-1, 3), plan.surface.normals.reshape(-1, 3), plan.surface.shape
    )
    accumulator = FiniteSurfaceAccumulator(grid)
    runtime = S2Runtime(model)
    total_frames = int(round(plan.duration_s * float(fps))) + 1
    dt = plan.duration_s / max(total_frames - 1, 1)
    for frame in range(total_frames):
        pose = plan.pose_at(frame * dt)
        step = runtime.step(
            dt_s=dt,
            incidence_angle_deg=plan.fan_incidence_deg,
            stand_off_m=plan.stand_off_m,
            air_velocity_m_s=model.air_velocity_m_s,
            mass_flow_kg_s=plan.mass_flow_kg_s,
            spray_on=pose.spray_on,
        )
        if pose.spray_on:
            accumulator.apply(step, target_position=pose.surface_position, frame=pose.frame)
    film = film_statistics(accumulator.grid.cumulative_mass_kg, accumulator.grid.area_weights_m2, liquid_density_kg_m3=liquid_density_kg_m3, volume_solids_fraction=volume_solids_fraction)
    return {
        "runtime": runtime.as_dict(),
        "finite_surface": accumulator.as_dict(),
        "wft": film["wft_um"],
        "film": film,
        "frames": total_frames,
        "fps": float(fps),
        "cycle_duration_s": plan.duration_s,
    }


def save_plan(plan: FullPanelPlan, summary: dict[str, Any], path: str | Path, *, model: S2RuntimeModel) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = plan.to_dict(model=model)
    payload["planner_summary"] = summary
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
