"""Portable S2 deposition runtime used by the native Isaac Sim adapter.

The runtime owns the calibrated model contract, surface-local frame checks,
structured curved-surface accumulation, and the injected/deposited/overspray
ledger.  It deliberately has no Isaac Sim imports so the numerical contract
can be tested with the normal project Python environment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .cfd_calibrated_kernel import GaussianMoments, interpolate_moments


_MODEL_HASH_KEY = "model_sha256"
_FRAME_EPS = 1.0e-12


def _unit(vector: Iterable[float], name: str) -> np.ndarray:
    value = np.asarray(tuple(vector), dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    norm = float(np.linalg.norm(value))
    if norm <= _FRAME_EPS:
        raise ValueError(f"{name} must be non-zero")
    return value / norm


def _canonical_json(payload: dict[str, Any]) -> bytes:
    """Serialize model data deterministically for its artifact hash."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class SurfaceFrame:
    """Surface-local frame and the measured incidence components."""

    normal: np.ndarray
    w_inward: np.ndarray
    u_fan_major: np.ndarray
    v_fan_minor: np.ndarray
    incidence_u_deg: float
    incidence_v_deg: float

    def __post_init__(self) -> None:
        arrays = (self.normal, self.w_inward, self.u_fan_major, self.v_fan_minor)
        for value in arrays:
            value = np.asarray(value, dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("surface frame axes must be finite length-3 vectors")
        object.__setattr__(self, "normal", _unit(self.normal, "normal"))
        object.__setattr__(self, "w_inward", _unit(self.w_inward, "w_inward"))
        object.__setattr__(self, "u_fan_major", _unit(self.u_fan_major, "u_fan_major"))
        object.__setattr__(self, "v_fan_minor", _unit(self.v_fan_minor, "v_fan_minor"))


def surface_frame(
    surface_normal: Iterable[float],
    fan_major_axis: Iterable[float],
    spray_axis: Iterable[float] | None = None,
    *,
    minor_tolerance_deg: float = 1.0,
) -> SurfaceFrame:
    """Build the right-handed surface frame and guard the tool orientation.

    The process direction is ``w = -normal``.  ``u`` is the fan-major axis
    projected onto the surface tangent plane and ``v = cross(w, u)``.  When a
    measured spray axis is supplied, its in-plane incidence and its forbidden
    ``v`` component are returned; a non-trivial minor component is rejected.
    """

    normal = _unit(surface_normal, "surface_normal")
    w_inward = -normal
    major = np.asarray(tuple(fan_major_axis), dtype=float)
    if major.shape != (3,) or not np.all(np.isfinite(major)):
        raise ValueError("fan_major_axis must be a finite length-3 vector")
    major -= float(np.dot(major, w_inward)) * w_inward
    u = _unit(major, "fan_major_axis tangent projection")
    v = _unit(np.cross(w_inward, u), "fan_minor_axis")
    incidence_u_deg = 0.0
    incidence_v_deg = 0.0
    if spray_axis is not None:
        axis = _unit(spray_axis, "spray_axis")
        denominator = float(np.dot(axis, w_inward))
        incidence_u_deg = float(np.degrees(np.arctan2(np.dot(axis, u), denominator)))
        incidence_v_deg = float(np.degrees(np.arctan2(np.dot(axis, v), denominator)))
        if abs(incidence_v_deg) > float(minor_tolerance_deg):
            raise ValueError(
                "spray axis has a forbidden fan-minor incidence component: "
                f"{incidence_v_deg:.6g} deg"
            )
    return SurfaceFrame(normal, w_inward, u, v, incidence_u_deg, incidence_v_deg)


@dataclass(frozen=True)
class S2RuntimeModel:
    """Validated, frozen S2 v1 endpoint model loaded from JSON."""

    model_id: str
    model_type: str
    openfoam_version: str
    minimum_angle_deg: float
    maximum_angle_deg: float
    stand_off_m: float
    air_velocity_m_s: float
    mass_flow_kg_s: float
    canonical_injected_mass_kg: float
    endpoint_zero: GaussianMoments
    endpoint_fifteen: GaussianMoments
    source_hashes: dict[str, str]
    model_sha256: str
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path, *, verify_hash: bool = True) -> "S2RuntimeModel":
        artifact = Path(path)
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "s2_runtime_model_v1":
            raise ValueError("unsupported S2 runtime model schema")
        recorded_hash = str(payload.get(_MODEL_HASH_KEY, ""))
        if verify_hash:
            without_hash = dict(payload)
            without_hash.pop(_MODEL_HASH_KEY, None)
            actual_hash = _sha256_bytes(_canonical_json(without_hash))
            if recorded_hash != actual_hash:
                raise ValueError(
                    f"S2 model hash mismatch: recorded={recorded_hash}, actual={actual_hash}"
                )
        domain = payload["calibrated_incidence_deg"]
        minimum = float(domain[0])
        maximum = float(domain[1])
        if minimum != 0.0 or maximum <= minimum:
            raise ValueError("S2 runtime model must start at 0 degrees")
        operating = payload["operating_point"]
        endpoints = payload["endpoints"]
        zero = GaussianMoments(
            float(endpoints["0_deg"]["deposited_mass_kg"]),
            tuple(endpoints["0_deg"]["centroid_uv_m"]),
            np.asarray(endpoints["0_deg"]["covariance_uv_m2"], dtype=float),
        )
        fifteen = GaussianMoments(
            float(endpoints["15_deg"]["deposited_mass_kg"]),
            tuple(endpoints["15_deg"]["centroid_uv_m"]),
            np.asarray(endpoints["15_deg"]["covariance_uv_m2"], dtype=float),
        )
        return cls(
            model_id=str(payload["model_id"]),
            model_type=str(payload["model_type"]),
            openfoam_version=str(payload["teacher"]["openfoam_version"]),
            minimum_angle_deg=minimum,
            maximum_angle_deg=maximum,
            stand_off_m=float(operating["stand_off_m"]),
            air_velocity_m_s=float(operating["air_velocity_m_s"]),
            mass_flow_kg_s=float(operating["mass_flow_kg_s"]),
            canonical_injected_mass_kg=float(operating["canonical_injected_mass_kg"]),
            endpoint_zero=zero,
            endpoint_fifteen=fifteen,
            source_hashes=dict(payload.get("source_hashes", {})),
            model_sha256=recorded_hash,
            payload=payload,
        )

    def predict(self, incidence_angle_deg: float) -> GaussianMoments:
        """Predict endpoint-interpolated moments, rejecting extrapolation."""

        theta = float(incidence_angle_deg)
        if theta < self.minimum_angle_deg or theta > self.maximum_angle_deg:
            raise ValueError(
                f"incidence_angle_deg must be within "
                f"[{self.minimum_angle_deg:g}, {self.maximum_angle_deg:g}] degrees"
            )
        return interpolate_moments(
            theta,
            self.endpoint_zero,
            self.endpoint_fifteen,
            maximum_angle_deg=self.maximum_angle_deg,
        )

    def transfer_efficiency(self, incidence_angle_deg: float) -> float:
        """Return the bounded deposited/injected fraction for one angle."""

        moments = self.predict(incidence_angle_deg)
        raw = moments.deposited_mass_kg / self.canonical_injected_mass_kg
        return float(np.clip(raw, 0.0, 1.0))

    def verify_sources(self, repo_root: str | Path) -> None:
        """Verify the evidence files recorded by the deterministic package."""

        root = Path(repo_root)
        paths = self.payload.get("source_paths", {})
        for key, expected in self.source_hashes.items():
            relative = paths.get(key)
            if not relative:
                raise ValueError(f"S2 model has no source path for {key}")
            source = root / relative
            if not source.is_file():
                raise ValueError(f"S2 model source is missing: {source}")
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"S2 model source hash mismatch for {key}: {actual} != {expected}")

    def validate_operating_point(
        self,
        *,
        stand_off_m: float,
        air_velocity_m_s: float,
        mass_flow_kg_s: float,
        tolerance: float = 1.0e-9,
    ) -> None:
        values = (
            ("stand_off_m", stand_off_m, self.stand_off_m),
            ("air_velocity_m_s", air_velocity_m_s, self.air_velocity_m_s),
            ("mass_flow_kg_s", mass_flow_kg_s, self.mass_flow_kg_s),
        )
        for name, actual, expected in values:
            if not np.isclose(float(actual), float(expected), rtol=tolerance, atol=tolerance):
                raise ValueError(f"S2 fixed operating point mismatch for {name}: {actual} != {expected}")


@dataclass(frozen=True)
class RuntimeStep:
    frame: int
    dt_s: float
    incidence_angle_deg: float
    spray_on: bool
    injected_mass_kg: float
    deposited_mass_kg: float
    overspray_mass_kg: float
    moments: GaussianMoments | None
    events: tuple[str, ...]


class S2Runtime:
    """Stateful timestep adapter for one calibrated surface run."""

    def __init__(self, model: S2RuntimeModel) -> None:
        self.model = model
        self.frame = 0
        self.injected_mass_kg = 0.0
        self.deposited_mass_kg = 0.0
        self.overspray_mass_kg = 0.0
        self.spray_on_time_s = 0.0
        self.out_of_domain_steps = 0
        self.events: list[dict[str, Any]] = []

    def step(
        self,
        *,
        dt_s: float,
        incidence_angle_deg: float,
        stand_off_m: float,
        air_velocity_m_s: float,
        mass_flow_kg_s: float,
        spray_on: bool,
    ) -> RuntimeStep:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt_s must be finite and non-negative")
        self.model.validate_operating_point(
            stand_off_m=stand_off_m,
            air_velocity_m_s=air_velocity_m_s,
            mass_flow_kg_s=mass_flow_kg_s,
        )
        injected = float(mass_flow_kg_s) * dt if spray_on else 0.0
        if not spray_on or dt == 0.0:
            step = RuntimeStep(self.frame, dt, float(incidence_angle_deg), bool(spray_on), 0.0, 0.0, 0.0, None, ())
            self.frame += 1
            return step
        self.spray_on_time_s += dt
        try:
            prediction = self.model.predict(incidence_angle_deg)
        except ValueError as exc:
            self.out_of_domain_steps += 1
            event = f"out_of_domain:{exc}"
            self.events.append({"frame": self.frame, "event": event, "incidence_angle_deg": float(incidence_angle_deg)})
            self.injected_mass_kg += injected
            self.overspray_mass_kg += injected
            step = RuntimeStep(self.frame, dt, float(incidence_angle_deg), True, injected, 0.0, injected, None, (event,))
            self.frame += 1
            return step
        efficiency = self.model.transfer_efficiency(incidence_angle_deg)
        deposited = injected * efficiency
        overspray = injected - deposited
        if prediction.deposited_mass_kg > 0.0:
            scale = deposited / prediction.deposited_mass_kg
            moments = GaussianMoments(
                deposited,
                prediction.centroid_uv_m,
                prediction.covariance_uv_m2,
            )
        else:
            scale = 0.0
            moments = GaussianMoments(0.0, prediction.centroid_uv_m, prediction.covariance_uv_m2)
        _ = scale  # kept explicit for the mass-normalised moment contract
        self.injected_mass_kg += injected
        self.deposited_mass_kg += deposited
        self.overspray_mass_kg += overspray
        step = RuntimeStep(self.frame, dt, float(incidence_angle_deg), True, injected, deposited, overspray, moments, ())
        self.frame += 1
        return step

    def as_dict(self) -> dict[str, Any]:
        residual = self.injected_mass_kg - self.deposited_mass_kg - self.overspray_mass_kg
        return {
            "injected_kg": self.injected_mass_kg,
            "deposited_kg": self.deposited_mass_kg,
            "overspray_kg": self.overspray_mass_kg,
            "balance_error_kg": residual,
            "transfer_efficiency": self.deposited_mass_kg / self.injected_mass_kg if self.injected_mass_kg else 0.0,
            "spray_on_time_s": self.spray_on_time_s,
            "out_of_domain_steps": self.out_of_domain_steps,
            "events": list(self.events),
        }


@dataclass
class StructuredSurfaceGrid:
    """Structured curved-surface samples with geometry-derived area weights."""

    positions: np.ndarray
    normals: np.ndarray
    shape: tuple[int, int]
    area_weights_m2: np.ndarray
    cumulative_mass_kg: np.ndarray

    @classmethod
    def from_structured(
        cls,
        positions: np.ndarray,
        normals: np.ndarray,
        shape: tuple[int, int],
    ) -> "StructuredSurfaceGrid":
        points = np.asarray(positions, dtype=float)
        normal_values = np.asarray(normals, dtype=float)
        if points.shape != (int(np.prod(shape)), 3) or normal_values.shape != points.shape:
            raise ValueError("positions/normals must be flattened arrays matching shape")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normal_values)):
            raise ValueError("surface samples must be finite")
        ny, nz = map(int, shape)
        if ny < 2 or nz < 2:
            raise ValueError("structured surface requires at least a 2x2 grid")
        area = np.zeros(len(points), dtype=float)
        grid = points.reshape((ny, nz, 3))
        def tri(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        def index(i: int, j: int) -> int:
            return i * nz + j
        for i in range(ny - 1):
            for j in range(nz - 1):
                p00, p10 = grid[i, j], grid[i + 1, j]
                p01, p11 = grid[i, j + 1], grid[i + 1, j + 1]
                t0 = tri(p00, p10, p11)
                t1 = tri(p00, p11, p01)
                for vertex in (index(i, j), index(i + 1, j), index(i + 1, j + 1)):
                    area[vertex] += t0 / 3.0
                for vertex in (index(i, j), index(i + 1, j + 1), index(i, j + 1)):
                    area[vertex] += t1 / 3.0
        if not np.all(area > 0.0):
            raise ValueError("surface area weights must be positive")
        return cls(points, normal_values, (ny, nz), area, np.zeros(len(points), dtype=float))

    @property
    def total_area_m2(self) -> float:
        return float(np.sum(self.area_weights_m2))

    @property
    def integrated_mass_kg(self) -> float:
        return float(np.sum(self.cumulative_mass_kg))

    def deposit(
        self,
        *,
        target_position: Iterable[float],
        frame: SurfaceFrame,
        moments: GaussianMoments,
    ) -> float:
        """Accumulate one mass-normalised S2 footprint and return its mass."""

        target = np.asarray(tuple(target_position), dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_position must be finite length-3")
        displacement = self.positions - target
        uv = np.column_stack((displacement @ frame.u_fan_major, displacement @ frame.v_fan_minor))
        weights = np.asarray(moments.density(uv), dtype=float) * self.area_weights_m2
        normalisation = float(np.sum(weights))
        if normalisation <= 0.0 or not np.isfinite(normalisation):
            raise ValueError("surface grid does not intersect the S2 footprint")
        contribution = moments.deposited_mass_kg * weights / normalisation
        self.cumulative_mass_kg += contribution
        return float(np.sum(contribution))

    def triangle_faces(self) -> np.ndarray:
        """Return a deterministic triangulation suitable for one USD mesh."""

        ny, nz = self.shape
        faces: list[tuple[int, int, int]] = []
        for i in range(ny - 1):
            for j in range(nz - 1):
                a = i * nz + j
                b = (i + 1) * nz + j
                c = (i + 1) * nz + j + 1
                d = i * nz + j + 1
                faces.extend(((a, b, c), (a, c, d)))
        return np.asarray(faces, dtype=np.int32)


class RuntimeSurfaceAccumulator:
    """Surface map + ledger coupling for runtime integration."""

    def __init__(self, grid: StructuredSurfaceGrid) -> None:
        self.grid = grid
        self.deposited_mass_kg = 0.0
        self.out_of_domain_mass_kg = 0.0

    def apply(self, step: RuntimeStep, *, target_position: Iterable[float], frame: SurfaceFrame) -> None:
        if step.moments is None or step.deposited_mass_kg <= 0.0:
            self.out_of_domain_mass_kg += step.overspray_mass_kg
            return
        deposited = self.grid.deposit(target_position=target_position, frame=frame, moments=step.moments)
        self.deposited_mass_kg += deposited

    def as_dict(self) -> dict[str, float]:
        return {
            "deposited_kg": self.deposited_mass_kg,
            "surface_integrated_kg": self.grid.integrated_mass_kg,
            "surface_integration_error_kg": self.grid.integrated_mass_kg - self.deposited_mass_kg,
            "rejected_or_overspray_kg": self.out_of_domain_mass_kg,
        }
