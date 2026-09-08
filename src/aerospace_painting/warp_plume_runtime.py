"""Rolling Warp plume batches for the native painting-process adapter.

The helper owns only numeric state.  It deliberately has no Isaac Sim imports;
the native adapter supplies a frozen world frame for each emitted batch and
publishes the returned world-space point positions to one USD ``Points`` prim.
Each batch reuses the validated W1.3 fixed-grid full-vector carrier and the
W1 Schiller--Naumann transport kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import warp_spray as _warp_spray
from .warp_spray import WarpCaseConfig, _frame_axes, make_injection_samples

wp = _warp_spray.wp
mesh_query_ray = _warp_spray.mesh_query_ray
_make_target_mesh = _warp_spray._make_target_mesh
_advance_particles = getattr(_warp_spray, "_advance_particles", None)
from .warp_vector_carrier import VectorCarrierModel, load_model_artifacts


class WarpPlumeUnavailable(RuntimeError):
    """Raised when native CUDA Warp execution is requested but unavailable."""


def batch_mass_kg(mass_flow_kg_s: float, cadence_hz: float) -> float:
    """Return the exact represented mass of one quasi-steady emission batch."""

    flow = float(mass_flow_kg_s)
    cadence = float(cadence_hz)
    if not np.isfinite(flow) or flow <= 0.0:
        raise ValueError("mass_flow_kg_s must be positive and finite")
    if not np.isfinite(cadence) or cadence <= 0.0:
        raise ValueError("cadence_hz must be positive and finite")
    return flow / cadence


def _unit(vector: Iterable[float], name: str) -> np.ndarray:
    value = np.asarray(tuple(vector), dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must be non-zero")
    return value / norm


def _orthonormalize(u_axis: Iterable[float], v_axis: Iterable[float], w_axis: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize the frame without changing its handedness."""

    u = _unit(u_axis, "u_axis")
    w = _unit(w_axis, "w_axis")
    u = u - float(np.dot(u, w)) * w
    u = _unit(u, "u_axis tangent projection")
    v = _unit(v_axis, "v_axis")
    if abs(float(np.dot(v, w))) > 2.0e-5 or abs(float(np.dot(v, u))) > 2.0e-5:
        raise ValueError("batch frame axes must be mutually orthogonal")
    handed = float(np.dot(np.cross(u, v), w))
    if handed < 0.999:
        raise ValueError("batch frame axes must be right-handed")
    return u, v, w


def benchmark_to_world(points_m: np.ndarray, frame: "WarpBatchFrame", nozzle_origin_m: Iterable[float] = (0.0, 0.0, 0.002)) -> np.ndarray:
    """Map W1.3 benchmark coordinates into one frozen world frame."""

    values = np.asarray(points_m, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("points_m must end in a length-three dimension")
    relative = values - np.asarray(tuple(nozzle_origin_m), dtype=np.float64)
    return (
        np.asarray(frame.origin_world_m, dtype=np.float64)
        + relative[..., 0, None] * frame.u_world
        + relative[..., 1, None] * frame.v_world
        + relative[..., 2, None] * frame.w_world
    )


def world_to_benchmark(points_m: np.ndarray, frame: "WarpBatchFrame", nozzle_origin_m: Iterable[float] = (0.0, 0.0, 0.002)) -> np.ndarray:
    """Inverse of :func:`benchmark_to_world` for portable frame tests."""

    values = np.asarray(points_m, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("points_m must end in a length-three dimension")
    relative = values - np.asarray(frame.origin_world_m, dtype=np.float64)
    local = np.stack(
        (
            np.einsum("...i,i->...", relative, frame.u_world),
            np.einsum("...i,i->...", relative, frame.v_world),
            np.einsum("...i,i->...", relative, frame.w_world),
        ),
        axis=-1,
    )
    return local + np.asarray(tuple(nozzle_origin_m), dtype=np.float64)


@dataclass(frozen=True)
class WarpBatchFrame:
    """The immutable world frame captured at one emission instant."""

    origin_world_m: np.ndarray
    target_world_m: np.ndarray
    u_world: np.ndarray
    v_world: np.ndarray
    w_world: np.ndarray
    incidence_deg: float
    stand_off_m: float

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_world_m, dtype=np.float64)
        target = np.asarray(self.target_world_m, dtype=np.float64)
        if origin.shape != (3,) or target.shape != (3,) or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(target)):
            raise ValueError("batch frame points must be finite length-three vectors")
        if not 0.0 <= float(self.incidence_deg) <= 15.0:
            raise ValueError("batch incidence must be within [0, 15] degrees")
        if float(self.stand_off_m) <= 0.0 or not np.isfinite(float(self.stand_off_m)):
            raise ValueError("batch stand-off must be positive")
        u, v, w = _orthonormalize(self.u_world, self.v_world, self.w_world)
        object.__setattr__(self, "origin_world_m", origin)
        object.__setattr__(self, "target_world_m", target)
        object.__setattr__(self, "u_world", u)
        object.__setattr__(self, "v_world", v)
        object.__setattr__(self, "w_world", w)
        object.__setattr__(self, "incidence_deg", float(self.incidence_deg))
        object.__setattr__(self, "stand_off_m", float(self.stand_off_m))

    @classmethod
    def from_surface_frame(
        cls,
        *,
        origin_world_m: Iterable[float],
        target_world_m: Iterable[float],
        surface_frame: Any,
        incidence_deg: float,
        stand_off_m: float,
    ) -> "WarpBatchFrame":
        return cls(
            origin_world_m=np.asarray(tuple(origin_world_m), dtype=np.float64),
            target_world_m=np.asarray(tuple(target_world_m), dtype=np.float64),
            u_world=np.asarray(surface_frame.u_fan_major, dtype=np.float64),
            v_world=np.asarray(surface_frame.v_fan_minor, dtype=np.float64),
            w_world=np.asarray(surface_frame.w_inward, dtype=np.float64),
            incidence_deg=float(incidence_deg),
            stand_off_m=float(stand_off_m),
        )

    @property
    def target_plane_center_world_m(self) -> np.ndarray:
        # W1's benchmark origin is z=0.002 m and its collision plane is z=stand-off.
        return np.asarray(self.origin_world_m) + (self.stand_off_m - 0.002) * self.w_world

    @property
    def target_plane_error_m(self) -> float:
        return float(np.linalg.norm(self.target_plane_center_world_m - self.target_world_m))


@dataclass
class _Batch:
    batch_id: int
    frame: WarpBatchFrame
    created_time_s: float
    config: WarpCaseConfig
    samples_mass_kg: np.ndarray
    positions: Any
    previous_positions: Any
    velocities: Any
    diameters: Any
    represented_mass: Any
    bin_id: Any
    release_time: Any
    alive: Any
    released: Any
    deposited: Any
    escaped: Any
    carrier_out_of_field: Any
    deposition_mass: Any
    last_integrated_time_s: float
    initial_tool_origin_m: np.ndarray
    initial_tool_u: np.ndarray
    initial_tool_v: np.ndarray
    initial_tool_w: np.ndarray
    max_tcp_translation_m: float = 0.0
    max_tcp_orientation_deg: float = 0.0


@dataclass
class WarpPlumeRuntime:
    """Rolling fixed-grid W1.3 batches used by the native adapter."""

    model_path: Path
    repo_root: Path
    mass_flow_kg_s: float
    stand_off_m: float
    particle_count_per_bin: int = 500
    cadence_hz: float = 15.0
    particle_dt_s: float = 5.0e-5
    maximum_flight_s: float = 0.06
    device: str = "cuda:0"
    vector_carrier: VectorCarrierModel = field(init=False)
    model_payload: dict[str, Any] = field(init=False)
    _mesh: Any = field(init=False, repr=False)
    _carrier_values_0: Any = field(init=False, repr=False)
    _carrier_values_15: Any = field(init=False, repr=False)
    _carrier_first_center: np.ndarray = field(init=False, repr=False)
    _carrier_spacing: np.ndarray = field(init=False, repr=False)
    _carrier_nx: int = field(init=False, repr=False)
    _carrier_ny: int = field(init=False, repr=False)
    _carrier_nz: int = field(init=False, repr=False)
    _active: list[_Batch] = field(default_factory=list, init=False, repr=False)
    _closed: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _next_batch_id: int = field(default=1, init=False, repr=False)
    _next_emit_time_s: float | None = field(default=None, init=False, repr=False)
    _last_time_s: float = field(default=0.0, init=False, repr=False)
    _mass_emitted_kg: float = field(default=0.0, init=False, repr=False)
    _rejected_batches: int = field(default=0, init=False, repr=False)
    _rejected_mass_kg: float = field(default=0.0, init=False, repr=False)
    _rejection_reasons: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _max_origin_offset_m: float = field(default=0.0, init=False, repr=False)
    _max_target_plane_error_m: float = field(default=0.0, init=False, repr=False)
    _max_active_particles: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if wp is None or mesh_query_ray is None or _advance_particles is None:
            raise WarpPlumeUnavailable("NVIDIA Warp is not importable in this Python environment")
        if not str(self.device).startswith("cuda"):
            raise ValueError("Warp plume runtime requires a CUDA device")
        if int(self.particle_count_per_bin) != 500:
            raise ValueError("W2 uses the frozen 500 particles per droplet bin")
        if not np.isclose(float(self.cadence_hz), 15.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("W2 cadence is fixed at 15 Hz")
        if not np.isclose(float(self.particle_dt_s), 5.0e-5, rtol=0.0, atol=1.0e-12):
            raise ValueError("W2 particle dt is fixed at 5e-5 s")
        if not np.isclose(float(self.maximum_flight_s), 0.06, rtol=0.0, atol=1.0e-12):
            raise ValueError("W2 maximum flight time is fixed at 0.06 s")
        if float(self.stand_off_m) <= 0.0:
            raise ValueError("stand-off must be positive")
        wp.init()
        if wp.get_cuda_device_count() < 1:
            raise WarpPlumeUnavailable("Warp reports no usable CUDA device")
        self.model_path = Path(self.model_path)
        self.repo_root = Path(self.repo_root)
        self.vector_carrier, self.model_payload = load_model_artifacts(
            self.model_path,
            repo_root=self.repo_root,
            verify_sources=True,
        )
        if self.model_payload.get("interpolation_method") != "FIXED_GRID_LOCAL_VECTOR_INTERP":
            raise ValueError("W2 requires the validated W1.3 fixed-grid vector carrier")
        generic_config = WarpCaseConfig(
            incidence_angle_deg=0.0,
            particle_count_per_bin=1,
            stand_off_m=self.stand_off_m,
            particle_dt_s=self.particle_dt_s,
            maximum_duration_s=self.maximum_flight_s,
            mass_flow_kg_s=self.mass_flow_kg_s,
            injection_duration_s=1.0 / self.cadence_hz,
        )
        device_obj = wp.get_device(self.device)
        self._mesh = _make_target_mesh(generic_config, self.device)
        self._carrier_values_0 = wp.array(self.vector_carrier.flat_anchor_0_vec3f, dtype=wp.vec3f, device=device_obj)
        self._carrier_values_15 = wp.array(self.vector_carrier.flat_anchor_15_vec3f, dtype=wp.vec3f, device=device_obj)
        self._carrier_first_center = np.asarray(self.vector_carrier.first_center_m, dtype=np.float32)
        self._carrier_spacing = np.asarray(self.vector_carrier.spacing_m, dtype=np.float32)
        self._carrier_nx, self._carrier_ny, self._carrier_nz = self.vector_carrier.nx, self.vector_carrier.ny, self.vector_carrier.nz

    @property
    def cadence_period_s(self) -> float:
        return 1.0 / float(self.cadence_hz)

    @property
    def batch_mass_kg(self) -> float:
        return batch_mass_kg(self.mass_flow_kg_s, self.cadence_hz)

    def _make_batch(self, frame: WarpBatchFrame, created_time_s: float) -> _Batch:
        config = WarpCaseConfig(
            incidence_angle_deg=frame.incidence_deg,
            particle_count_per_bin=self.particle_count_per_bin,
            particle_dt_s=self.particle_dt_s,
            maximum_duration_s=self.maximum_flight_s,
            stand_off_m=self.stand_off_m,
            mass_flow_kg_s=self.mass_flow_kg_s,
            # This controls the exact represented mass.  The quasi-steady W2
            # approximation releases every parcel at the batch timestamp.
            injection_duration_s=self.cadence_period_s,
        )
        samples = make_injection_samples(config)
        samples["release_time"][:] = 0.0
        device_obj = wp.get_device(self.device)
        n = config.particle_count
        return _Batch(
            batch_id=self._next_batch_id,
            frame=frame,
            created_time_s=float(created_time_s),
            config=config,
            samples_mass_kg=np.asarray(samples["represented_mass"], dtype=np.float64),
            positions=wp.array(samples["position"], dtype=wp.vec3f, device=device_obj),
            previous_positions=wp.array(samples["position"], dtype=wp.vec3f, device=device_obj),
            velocities=wp.array(samples["velocity"], dtype=wp.vec3f, device=device_obj),
            diameters=wp.array(samples["diameter"], dtype=wp.float32, device=device_obj),
            represented_mass=wp.array(samples["represented_mass"].astype(np.float32), dtype=wp.float32, device=device_obj),
            bin_id=wp.array(samples["bin_id"], dtype=wp.int32, device=device_obj),
            release_time=wp.array(samples["release_time"], dtype=wp.float32, device=device_obj),
            alive=wp.zeros(n, dtype=wp.bool, device=device_obj),
            released=wp.zeros(n, dtype=wp.bool, device=device_obj),
            deposited=wp.zeros(n, dtype=wp.bool, device=device_obj),
            escaped=wp.zeros(n, dtype=wp.bool, device=device_obj),
            carrier_out_of_field=wp.zeros(n, dtype=wp.bool, device=device_obj),
            deposition_mass=wp.zeros(config.target_resolution * config.target_resolution, dtype=wp.float32, device=device_obj),
            last_integrated_time_s=float(created_time_s),
            initial_tool_origin_m=np.asarray(frame.origin_world_m, dtype=np.float64).copy(),
            initial_tool_u=np.asarray(frame.u_world, dtype=np.float64).copy(),
            initial_tool_v=np.asarray(frame.v_world, dtype=np.float64).copy(),
            initial_tool_w=np.asarray(frame.w_world, dtype=np.float64).copy(),
        )

    def _launch_step(self, batch: _Batch, simulation_time_s: float, dt_s: float) -> None:
        x_axis, y_axis, z_axis = _frame_axes(batch.frame.incidence_deg)
        origin = np.asarray(batch.config.nozzle_origin_m, dtype=np.float32)
        gravity = np.asarray(batch.config.gravity_m_s2, dtype=np.float32)
        device_obj = wp.get_device(self.device)
        wp.launch(
            kernel=_advance_particles,
            dim=batch.config.particle_count,
            inputs=[
                batch.positions,
                batch.previous_positions,
                batch.velocities,
                batch.diameters,
                batch.represented_mass,
                batch.bin_id,
                batch.release_time,
                batch.alive,
                batch.released,
                batch.deposited,
                batch.escaped,
                batch.carrier_out_of_field,
                batch.deposition_mass,
                self._mesh.id,
                wp.vec3f(*origin),
                wp.vec3f(*x_axis),
                wp.vec3f(*y_axis),
                wp.vec3f(*z_axis),
                float(simulation_time_s - batch.created_time_s),
                float(dt_s),
                float(batch.config.air_field.u0_m_s),
                float(batch.config.air_field.sigma0_m),
                float(batch.config.air_field.sigma_slope),
                float(batch.config.air_field.decay_z0_m),
                float(batch.config.air_field.decay_exponent),
                float(batch.config.air_density_kg_m3),
                float(batch.config.air_dynamic_viscosity_pa_s),
                float(batch.config.liquid_density_kg_m3),
                wp.vec3f(*gravity),
                float(batch.config.target_extent_m),
                float(batch.config.stand_off_m + 0.05),
                int(batch.config.target_resolution),
                self._carrier_values_0,
                self._carrier_values_15,
                wp.vec3f(*self._carrier_first_center),
                wp.vec3f(*self._carrier_spacing),
                int(self._carrier_nx),
                int(self._carrier_ny),
                int(self._carrier_nz),
                float(batch.frame.incidence_deg / 15.0),
                4,
            ],
            device=device_obj,
        )

    def _advance_batch_to(self, batch: _Batch, end_time_s: float) -> None:
        target = min(float(end_time_s), batch.created_time_s + self.maximum_flight_s)
        while batch.last_integrated_time_s < target - 1.0e-12:
            step = min(self.particle_dt_s, target - batch.last_integrated_time_s)
            self._launch_step(batch, batch.last_integrated_time_s, step)
            batch.last_integrated_time_s += step

    def _close_batch(self, batch: _Batch, *, reason: str) -> None:
        # All host copies happen after the caller synchronizes one device stream.
        masses = batch.samples_mass_kg
        deposited = batch.deposited.numpy().astype(bool)
        escaped = batch.escaped.numpy().astype(bool)
        alive = batch.alive.numpy().astype(bool)
        released = batch.released.numpy().astype(bool)
        carrier_out = batch.carrier_out_of_field.numpy().astype(bool)
        deposited_mass = float(np.sum(masses[deposited]))
        escaped_mass = float(np.sum(masses[escaped]))
        shutdown_mask = ~(deposited | escaped)
        shutdown_mass = float(np.sum(masses[shutdown_mask]))
        expected = float(np.sum(masses))
        residual = expected - deposited_mass - escaped_mass - shutdown_mass
        batch_map_mass = float(np.sum(batch.deposition_mass.numpy().astype(np.float64)))
        self._closed.append(
            {
                "batch_id": int(batch.batch_id),
                "created_time_s": float(batch.created_time_s),
                "incidence_deg": float(batch.frame.incidence_deg),
                "expected_mass_kg": expected,
                "deposited_mass_kg": deposited_mass,
                "escaped_mass_kg": escaped_mass,
                "live_shutdown_mass_kg": shutdown_mass,
                "mass_residual_kg": residual,
                "map_mass_kg": batch_map_mass,
                "released_particle_count": int(np.count_nonzero(released)),
                "deposited_particle_count": int(np.count_nonzero(deposited)),
                "escaped_particle_count": int(np.count_nonzero(escaped)),
                "live_shutdown_particle_count": int(np.count_nonzero(alive & shutdown_mask)),
                "unreleased_particle_count": int(np.count_nonzero((~released) & shutdown_mask)),
                "carrier_out_of_field_particle_count": int(np.count_nonzero(carrier_out)),
                "carrier_out_of_field_mass_kg": float(np.sum(masses[carrier_out])),
                "closure_reason": reason,
                "max_tcp_translation_m": float(batch.max_tcp_translation_m),
                "max_tcp_orientation_deg": float(batch.max_tcp_orientation_deg),
                "target_plane_error_m": float(batch.frame.target_plane_error_m),
            }
        )

    def emit_batch(
        self,
        frame: WarpBatchFrame,
        time_s: float,
        *,
        actual_tool_origin_m: Iterable[float] | None = None,
        actual_tool_axes: tuple[Iterable[float], Iterable[float], Iterable[float]] | None = None,
    ) -> bool:
        """Emit one batch, rejecting invalid incidence/frame inputs."""

        try:
            if frame.incidence_deg < 0.0 or frame.incidence_deg > 15.0:
                raise ValueError("incidence outside W1.3 domain")
            batch = self._make_batch(frame, float(time_s))
        except (ValueError, TypeError) as exc:
            self._rejected_batches += 1
            self._rejected_mass_kg += self.batch_mass_kg
            key = str(exc).split(":", 1)[0]
            self._rejection_reasons[key] = self._rejection_reasons.get(key, 0) + 1
            return False
        self._next_batch_id += 1
        if actual_tool_axes is not None:
            batch.initial_tool_u, batch.initial_tool_v, batch.initial_tool_w = _orthonormalize(*actual_tool_axes)
        self._active.append(batch)
        self._mass_emitted_kg += self.batch_mass_kg
        self._max_target_plane_error_m = max(self._max_target_plane_error_m, frame.target_plane_error_m)
        if actual_tool_origin_m is not None:
            offset = float(np.linalg.norm(np.asarray(tuple(actual_tool_origin_m), dtype=np.float64) - frame.origin_world_m))
            self._max_origin_offset_m = max(self._max_origin_offset_m, offset)
        return True

    def observe_live_tool(self, time_s: float, origin_world_m: Iterable[float], u_world: Iterable[float], v_world: Iterable[float], w_world: Iterable[float]) -> None:
        """Measure actual TCP motion against every still-live frozen batch."""

        origin = np.asarray(tuple(origin_world_m), dtype=np.float64)
        u, v, w = _orthonormalize(u_world, v_world, w_world)
        for batch in self._active:
            age = float(time_s - batch.created_time_s)
            if age < -1.0e-12 or age > self.maximum_flight_s + 1.0e-9:
                continue
            batch.max_tcp_translation_m = max(batch.max_tcp_translation_m, float(np.linalg.norm(origin - batch.initial_tool_origin_m)))
            dots = np.clip(
                [float(np.dot(batch.initial_tool_u, u)), float(np.dot(batch.initial_tool_v, v)), float(np.dot(batch.initial_tool_w, w))],
                -1.0,
                1.0,
            )
            batch.max_tcp_orientation_deg = max(batch.max_tcp_orientation_deg, float(np.degrees(np.max(np.arccos(dots)))))

    def update(
        self,
        time_s: float,
        spray_on: bool,
        frame: WarpBatchFrame,
        *,
        actual_tool_origin_m: Iterable[float] | None = None,
        actual_tool_axes: tuple[Iterable[float], Iterable[float], Iterable[float]] | None = None,
    ) -> None:
        """Advance all active batches and emit due spray-on batches."""

        current = float(time_s)
        if current < self._last_time_s - 1.0e-9:
            raise ValueError("plume time must be monotonic")
        self._advance_to(current)
        if spray_on:
            if self._next_emit_time_s is None:
                self._next_emit_time_s = current
            while self._next_emit_time_s <= current + 1.0e-10:
                self.emit_batch(
                    frame,
                    self._next_emit_time_s,
                    actual_tool_origin_m=actual_tool_origin_m,
                    actual_tool_axes=actual_tool_axes,
                )
                self._next_emit_time_s += self.cadence_period_s
            # Integrate newly emitted particles up to the current render time.
            self._advance_to(current)
        else:
            self._next_emit_time_s = None
        observe_axes = actual_tool_axes if actual_tool_axes is not None else (frame.u_world, frame.v_world, frame.w_world)
        self.observe_live_tool(current, actual_tool_origin_m if actual_tool_origin_m is not None else frame.origin_world_m, *observe_axes)
        self._last_time_s = current
        active_count = sum(int(np.count_nonzero(batch.alive.numpy())) for batch in self._active)
        self._max_active_particles = max(self._max_active_particles, active_count)

    def _advance_to(self, time_s: float) -> None:
        if not self._active:
            return
        for batch in self._active:
            self._advance_batch_to(batch, time_s)
        wp.synchronize_device(wp.get_device(self.device))
        remaining: list[_Batch] = []
        for batch in self._active:
            if time_s >= batch.created_time_s + self.maximum_flight_s - 1.0e-10:
                self._close_batch(batch, reason="max_flight")
            else:
                remaining.append(batch)
        self._active = remaining

    def active_world_positions(self) -> np.ndarray:
        """Return only currently alive particles in world coordinates."""

        if not self._active:
            return np.empty((0, 3), dtype=np.float32)
        wp.synchronize_device(wp.get_device(self.device))
        chunks: list[np.ndarray] = []
        for batch in self._active:
            alive = batch.alive.numpy().astype(bool)
            if not np.any(alive):
                continue
            local = batch.positions.numpy()[alive]
            chunks.append(benchmark_to_world(local, batch.frame).astype(np.float32))
        if not chunks:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def shutdown(self, time_s: float | None = None) -> None:
        """Close all batches at the end of a native run."""

        if time_s is not None:
            self._advance_to(float(time_s))
        else:
            wp.synchronize_device(wp.get_device(self.device))
        for batch in self._active:
            self._close_batch(batch, reason="runtime_shutdown")
        self._active.clear()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mass, motion, and W1.3 provenance summary."""

        expected = float(self._mass_emitted_kg)
        deposited = float(sum(row["deposited_mass_kg"] for row in self._closed))
        escaped = float(sum(row["escaped_mass_kg"] for row in self._closed))
        shutdown = float(sum(row["live_shutdown_mass_kg"] for row in self._closed))
        rejected = float(self._rejected_mass_kg)
        balance = expected - deposited - escaped - shutdown
        return {
            "expected_emitted_mass_kg": expected,
            "deposited_mass_kg": deposited,
            "escaped_mass_kg": escaped,
            "live_shutdown_mass_kg": shutdown,
            "rejected_mass_kg": rejected,
            "balance_error_kg": float(balance),
            "relative_balance_error": float(balance / max(expected, 1.0e-30)),
            "batch_count": len(self._closed) + len(self._active),
            "closed_batch_count": len(self._closed),
            "active_batch_count": len(self._active),
            "rejected_batch_count": self._rejected_batches,
            "rejection_reasons": dict(self._rejection_reasons),
            "batch_mass_kg": self.batch_mass_kg,
            "cadence_hz": float(self.cadence_hz),
            "particle_count_per_bin": int(self.particle_count_per_bin),
            "particle_count_per_batch": int(self.particle_count_per_bin * 5),
            "particle_dt_s": float(self.particle_dt_s),
            "maximum_flight_s": float(self.maximum_flight_s),
            "mass_fractions": [0.10, 0.20, 0.30, 0.25, 0.15],
            "max_active_particles": int(self._max_active_particles),
            "max_tcp_translation_m": float(max((row["max_tcp_translation_m"] for row in self._closed), default=0.0)),
            "max_tcp_orientation_deg": float(max((row["max_tcp_orientation_deg"] for row in self._closed), default=0.0)),
            "max_target_plane_error_m": float(self._max_target_plane_error_m),
            "max_visual_origin_offset_m": float(self._max_origin_offset_m),
            "carrier": {
                "model_id": str(self.vector_carrier.model_id),
                "interpolation_method": str(self.vector_carrier.interpolation_method),
                "model_sha256": str(self.model_payload.get("model_sha256", "")),
                "npz_sha256": str(self.model_payload.get("npz_sha256", "")),
                "source_commit": str(self.model_payload.get("source_commit", "")),
                "openfoam_version": str(self.model_payload.get("openfoam_version", "")),
                "grid_dimensions_nx_ny_nz": list(self.vector_carrier.dimensions_nx_ny_nz),
            },
            "closed_batches": list(self._closed),
        }
