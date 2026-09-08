"""GPU Lagrangian W1 spray transport implemented with NVIDIA Warp.

This module is intentionally a static flat-plate validator.  It has no Isaac
Sim imports and does not alter the moving painting runtime.  Importing it on a
machine without Warp remains safe; CUDA execution then raises a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from .warp_air_field import AirFieldParameters


try:  # Warp is optional for the portable test suite.
    import warp as wp
    from warp._src.lang import mesh_query_ray
except ImportError:  # pragma: no cover - exercised only on non-Isaac hosts.
    wp = None  # type: ignore[assignment]
    mesh_query_ray = None  # type: ignore[assignment]


class WarpUnavailable(RuntimeError):
    """Raised when a CUDA Warp run is requested without the optional runtime."""


@dataclass(frozen=True)
class WarpCaseConfig:
    incidence_angle_deg: float
    particle_count_per_bin: int
    particle_dt_s: float = 5.0e-5
    maximum_duration_s: float = 0.06
    stand_off_m: float = 0.24
    nozzle_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.002)
    target_extent_m: float = 0.15
    target_resolution: int = 24
    cone_half_angle_deg: float = 22.0
    initial_speed_m_s: float = 12.0
    mass_flow_kg_s: float = 1.0e-4
    injection_duration_s: float = 0.01
    liquid_density_kg_m3: float = 1000.0
    air_density_kg_m3: float = 1.1995094678554101
    air_dynamic_viscosity_pa_s: float = 1.8094681577601565e-5
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    droplet_bins_m: tuple[float, ...] = (4.0e-5, 7.0e-5, 1.0e-4, 1.5e-4, 2.2e-4)
    mass_fractions: tuple[float, ...] = (0.10, 0.20, 0.30, 0.25, 0.15)
    air_field: AirFieldParameters = AirFieldParameters(18.0, 0.05, 0.3, 0.05, 1.0)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.incidence_angle_deg) <= 15.0:
            raise ValueError("W1 incidence must be within [0, 15] degrees")
        if int(self.particle_count_per_bin) < 1:
            raise ValueError("particle_count_per_bin must be positive")
        if self.particle_dt_s <= 0.0 or self.maximum_duration_s <= 0.0:
            raise ValueError("particle_dt_s and maximum_duration_s must be positive")
        if self.target_resolution < 2 or self.target_extent_m <= 0.0:
            raise ValueError("target grid configuration is invalid")
        if len(self.droplet_bins_m) != 5 or len(self.mass_fractions) != 5:
            raise ValueError("W1 requires exactly five droplet bins and fractions")
        if not np.isclose(sum(self.mass_fractions), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("mass fractions must sum to one")

    @property
    def expected_injected_mass_kg(self) -> float:
        return float(self.mass_flow_kg_s * self.injection_duration_s)

    @property
    def particle_count(self) -> int:
        return int(self.particle_count_per_bin) * len(self.droplet_bins_m)

    @property
    def target_cell_size_m(self) -> float:
        return 2.0 * float(self.target_extent_m) / float(self.target_resolution)


def _radical_inverse(index: int, base: int = 2) -> float:
    value = 0.0
    factor = 1.0 / float(base)
    n = int(index) + 1
    while n > 0:
        n, remainder = divmod(n, base)
        value += remainder * factor
        factor /= float(base)
    return value


def _frame_axes(angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = np.deg2rad(float(angle_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return (
        np.asarray((c, 0.0, -s), dtype=np.float32),
        np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
        np.asarray((s, 0.0, c), dtype=np.float32),
    )


def make_injection_samples(config: WarpCaseConfig) -> dict[str, np.ndarray]:
    """Create deterministic stratified cone samples and exact bin masses."""

    n_bin = int(config.particle_count_per_bin)
    radius = 0.5 * 8.0e-4
    half_angle = np.deg2rad(config.cone_half_angle_deg)
    x_axis, y_axis, z_axis = _frame_axes(config.incidence_angle_deg)
    origin = np.asarray(config.nozzle_origin_m, dtype=np.float32)
    positions = np.empty((config.particle_count, 3), dtype=np.float32)
    velocities = np.empty_like(positions)
    diameters = np.empty(config.particle_count, dtype=np.float32)
    represented_mass = np.empty(config.particle_count, dtype=np.float64)
    release_time = np.empty(config.particle_count, dtype=np.float32)
    bin_id = np.empty(config.particle_count, dtype=np.int32)

    cursor = 0
    for bin_index, (diameter, fraction) in enumerate(zip(config.droplet_bins_m, config.mass_fractions)):
        mass_per_particle = config.expected_injected_mass_kg * float(fraction) / float(n_bin)
        for local_index in range(n_bin):
            global_index = cursor + local_index
            radial_u = (local_index + 0.5) / float(n_bin)
            radial_v = _radical_inverse(local_index + 17 * bin_index, 2)
            radial = radius * math.sqrt(radial_u)
            phi = 2.0 * math.pi * radial_v
            # v2606 ConeNozzleInjection samples the cone angle uniformly
            # between thetaInner and thetaOuter, independently of the disc
            # radius.  A separate prime-base radical inverse avoids coupling
            # this angular draw to the radial/azimuthal stratification.
            angle_u = _radical_inverse(local_index + 29 * bin_index, 3)
            alpha = half_angle * angle_u
            local_position = radial * np.asarray((math.cos(phi), math.sin(phi), 0.0), dtype=np.float32)
            local_direction = np.asarray(
                (math.sin(alpha) * math.cos(phi), math.sin(alpha) * math.sin(phi), math.cos(alpha)),
                dtype=np.float32,
            )
            positions[global_index] = origin + local_position[0] * x_axis + local_position[1] * y_axis
            velocities[global_index] = config.initial_speed_m_s * (
                local_direction[0] * x_axis + local_direction[1] * y_axis + local_direction[2] * z_axis
            )
            diameters[global_index] = float(diameter)
            represented_mass[global_index] = mass_per_particle
            release_time[global_index] = config.injection_duration_s * (local_index + 0.5) / float(n_bin)
            bin_id[global_index] = bin_index
        cursor += n_bin

    return {
        "position": positions,
        "velocity": velocities,
        "diameter": diameters,
        "represented_mass": represented_mass,
        "release_time": release_time,
        "bin_id": bin_id,
    }


if wp is not None:

    @wp.func
    def _air_velocity(
        position: wp.vec3f,
        origin: wp.vec3f,
        x_axis: wp.vec3f,
        y_axis: wp.vec3f,
        z_axis: wp.vec3f,
        u0: float,
        sigma0: float,
        sigma_slope: float,
        decay_z0: float,
        decay_exponent: float,
    ) -> wp.vec3f:
        relative = position - origin
        x_local = wp.dot(relative, x_axis)
        y_local = wp.dot(relative, y_axis)
        z_local = wp.max(wp.dot(relative, z_axis), 0.0)
        sigma = sigma0 + sigma_slope * z_local
        decay = wp.pow(decay_z0 / (z_local + decay_z0), decay_exponent)
        gaussian = wp.exp(-0.5 * ((x_local / sigma) * (x_local / sigma) + (y_local / sigma) * (y_local / sigma)))
        return z_axis * (u0 * decay * gaussian)


    @wp.func
    def _sphere_drag_acceleration(
        velocity: wp.vec3f,
        air_velocity: wp.vec3f,
        diameter: float,
        air_density: float,
        air_viscosity: float,
        liquid_density: float,
        gravity: wp.vec3f,
    ) -> wp.vec3f:
        relative = air_velocity - velocity
        relative_speed = wp.length(relative)
        if relative_speed <= 1.0e-8:
            return gravity
        reynolds = air_density * diameter * relative_speed / air_viscosity
        safe_re = wp.max(reynolds, 1.0e-8)
        schiller = 24.0 / safe_re * (1.0 + 0.15 * wp.pow(safe_re, 0.687))
        cd = wp.where(reynolds <= 1000.0, schiller, 0.44)
        drag_factor = 0.75 * cd * air_density * relative_speed / (liquid_density * diameter)
        return drag_factor * relative + gravity


    @wp.kernel
    def _advance_particles(
        positions: wp.array(dtype=wp.vec3f),
        previous_positions: wp.array(dtype=wp.vec3f),
        velocities: wp.array(dtype=wp.vec3f),
        diameters: wp.array(dtype=wp.float32),
        represented_mass: wp.array(dtype=wp.float32),
        bin_id: wp.array(dtype=wp.int32),
        release_time: wp.array(dtype=wp.float32),
        alive: wp.array(dtype=wp.bool),
        released: wp.array(dtype=wp.bool),
        deposited: wp.array(dtype=wp.bool),
        escaped: wp.array(dtype=wp.bool),
        deposition_mass: wp.array(dtype=wp.float32),
        mesh_id: wp.uint64,
        origin: wp.vec3f,
        x_axis: wp.vec3f,
        y_axis: wp.vec3f,
        z_axis: wp.vec3f,
        simulation_time: float,
        dt: float,
        u0: float,
        sigma0: float,
        sigma_slope: float,
        decay_z0: float,
        decay_exponent: float,
        air_density: float,
        air_viscosity: float,
        liquid_density: float,
        gravity: wp.vec3f,
        target_extent: float,
        max_z: float,
        map_resolution: int,
    ):
        index = wp.tid()
        # Keep the discrete teacher bin as an explicit GPU state array.  All
        # canonical bins are non-negative; the guard also makes accidental
        # uninitialised labels fail closed without changing valid particles.
        if bin_id[index] < 0:
            return
        if not released[index]:
            if simulation_time + dt < release_time[index]:
                return
            released[index] = True
            alive[index] = True
            previous_positions[index] = positions[index]

        if not alive[index]:
            return

        previous = positions[index]
        previous_positions[index] = previous
        air = _air_velocity(
            previous,
            origin,
            x_axis,
            y_axis,
            z_axis,
            u0,
            sigma0,
            sigma_slope,
            decay_z0,
            decay_exponent,
        )
        acceleration = _sphere_drag_acceleration(
            velocities[index],
            air,
            diameters[index],
            air_density,
            air_viscosity,
            liquid_density,
            gravity,
        )
        velocity = velocities[index] + acceleration * dt
        candidate = previous + velocity * dt
        segment = candidate - previous
        segment_length = wp.length(segment)
        if segment_length > 1.0e-10:
            direction = segment / segment_length
            query = mesh_query_ray(mesh_id, previous, direction, segment_length, -1)
            if query.result:
                hit = previous + direction * query.t
                if (
                    hit[0] >= -target_extent
                    and hit[0] <= target_extent
                    and hit[1] >= -target_extent
                    and hit[1] <= target_extent
                ):
                    positions[index] = hit
                    velocities[index] = velocity
                    deposited[index] = True
                    alive[index] = False
                    cell_size = 2.0 * target_extent / float(map_resolution)
                    u_index = int(wp.floor((hit[0] + target_extent) / cell_size))
                    v_index = int(wp.floor((hit[1] + target_extent) / cell_size))
                    u_index = wp.min(wp.max(u_index, 0), map_resolution - 1)
                    v_index = wp.min(wp.max(v_index, 0), map_resolution - 1)
                    cell_index = v_index * map_resolution + u_index
                    wp.atomic_add(deposition_mass, cell_index, represented_mass[index])
                    return

        positions[index] = candidate
        velocities[index] = velocity
        if (
            candidate[0] < -target_extent
            or candidate[0] > target_extent
            or candidate[1] < -target_extent
            or candidate[1] > target_extent
            or candidate[2] < -0.01
            or candidate[2] > max_z
        ):
            escaped[index] = True
            alive[index] = False


@dataclass
class WarpCaseResult:
    """Host-side result and evidence for one static Warp case."""

    incidence_angle_deg: float
    particle_count: int
    particle_count_per_bin: int
    map_mass_kg: np.ndarray
    map_density_kg_m2: np.ndarray
    hit_positions_m: np.ndarray
    represented_mass_kg: np.ndarray
    deposited_flags: np.ndarray
    escaped_flags: np.ndarray
    alive_flags: np.ndarray
    released_flags: np.ndarray
    performance: dict[str, Any]
    mass_ledger: dict[str, float]


def _make_target_mesh(config: WarpCaseConfig, device: str):
    points = np.asarray(
        [
            (-config.target_extent_m, -config.target_extent_m, config.stand_off_m),
            (config.target_extent_m, -config.target_extent_m, config.stand_off_m),
            (config.target_extent_m, config.target_extent_m, config.stand_off_m),
            (-config.target_extent_m, config.target_extent_m, config.stand_off_m),
        ],
        dtype=np.float32,
    )
    indices = np.asarray((0, 1, 2, 0, 2, 3), dtype=np.int32)
    return wp.Mesh(
        points=wp.array(points, dtype=wp.vec3f, device=device),
        indices=wp.array(indices, dtype=wp.int32, device=device),
    )


def run_warp_case(config: WarpCaseConfig, *, device: str = "cuda:0") -> WarpCaseResult:
    """Execute one actual GPU Warp transport case."""

    if wp is None or mesh_query_ray is None:
        raise WarpUnavailable("NVIDIA Warp is not importable in this Python environment")
    if not str(device).startswith("cuda"):
        raise ValueError("W1 validation requires a CUDA device")
    wp.init()
    if wp.get_cuda_device_count() < 1:
        raise WarpUnavailable("Warp reports no usable CUDA device")

    samples = make_injection_samples(config)
    n = config.particle_count
    device_obj = wp.get_device(device)
    mesh = _make_target_mesh(config, device)
    positions = wp.array(samples["position"], dtype=wp.vec3f, device=device_obj)
    previous_positions = wp.array(samples["position"], dtype=wp.vec3f, device=device_obj)
    velocities = wp.array(samples["velocity"], dtype=wp.vec3f, device=device_obj)
    diameters = wp.array(samples["diameter"], dtype=wp.float32, device=device_obj)
    represented_mass = wp.array(samples["represented_mass"].astype(np.float32), dtype=wp.float32, device=device_obj)
    bin_id = wp.array(samples["bin_id"], dtype=wp.int32, device=device_obj)
    release_time = wp.array(samples["release_time"], dtype=wp.float32, device=device_obj)
    alive = wp.zeros(n, dtype=wp.bool, device=device_obj)
    released = wp.zeros(n, dtype=wp.bool, device=device_obj)
    deposited = wp.zeros(n, dtype=wp.bool, device=device_obj)
    escaped = wp.zeros(n, dtype=wp.bool, device=device_obj)
    deposition_mass = wp.zeros(config.target_resolution * config.target_resolution, dtype=wp.float32, device=device_obj)
    x_axis, y_axis, z_axis = _frame_axes(config.incidence_angle_deg)
    origin = np.asarray(config.nozzle_origin_m, dtype=np.float32)
    gravity = np.asarray(config.gravity_m_s2, dtype=np.float32)
    step_count = int(math.ceil(config.maximum_duration_s / config.particle_dt_s))
    launch_times: list[float] = []
    start = time.perf_counter()
    for step_index in range(step_count):
        simulation_time = step_index * config.particle_dt_s
        launch_start = time.perf_counter()
        wp.launch(
            kernel=_advance_particles,
            dim=n,
            inputs=[
                positions,
                previous_positions,
                velocities,
                diameters,
                represented_mass,
                bin_id,
                release_time,
                alive,
                released,
                deposited,
                escaped,
                deposition_mass,
                mesh.id,
                wp.vec3f(*origin),
                wp.vec3f(*x_axis),
                wp.vec3f(*y_axis),
                wp.vec3f(*z_axis),
                float(simulation_time),
                float(config.particle_dt_s),
                float(config.air_field.u0_m_s),
                float(config.air_field.sigma0_m),
                float(config.air_field.sigma_slope),
                float(config.air_field.decay_z0_m),
                float(config.air_field.decay_exponent),
                float(config.air_density_kg_m3),
                float(config.air_dynamic_viscosity_pa_s),
                float(config.liquid_density_kg_m3),
                wp.vec3f(*gravity),
                float(config.target_extent_m),
                float(config.stand_off_m + 0.05),
                int(config.target_resolution),
            ],
            device=device_obj,
        )
        wp.synchronize_device(device_obj)
        launch_times.append(time.perf_counter() - launch_start)
    total_wall = time.perf_counter() - start

    host_positions = positions.numpy()
    host_mass = samples["represented_mass"]
    host_deposited = deposited.numpy().astype(bool)
    host_escaped = escaped.numpy().astype(bool)
    host_alive = alive.numpy().astype(bool)
    host_released = released.numpy().astype(bool)
    host_map_mass = deposition_mass.numpy().astype(np.float64)
    area = config.target_cell_size_m**2
    host_map_density = host_map_mass / area
    expected_mass = float(np.sum(host_mass))
    deposited_mass = float(np.sum(host_mass[host_deposited]))
    escaped_mass = float(np.sum(host_mass[host_escaped]))
    live_mass = float(np.sum(host_mass[host_alive]))
    residual = expected_mass - deposited_mass - escaped_mass - live_mass
    performance = {
        "device": str(device_obj),
        "particle_count": n,
        "particle_count_per_bin": config.particle_count_per_bin,
        "substeps": step_count,
        "kernel_launch_count": step_count,
        "particle_dt_s": config.particle_dt_s,
        "simulated_duration_s": config.maximum_duration_s,
        "wall_seconds": total_wall,
        "mean_step_compute_seconds": float(np.mean(launch_times)),
        "p95_step_compute_seconds": float(np.percentile(launch_times, 95.0)),
        "max_step_compute_seconds": float(np.max(launch_times)),
    }
    ledger = {
        "expected_injected_mass_kg": expected_mass,
        "particle_injected_mass_kg": expected_mass,
        "deposited_mass_kg": deposited_mass,
        "escaped_mass_kg": escaped_mass,
        "live_mass_kg": live_mass,
        "mass_residual_kg": residual,
        "relative_balance_error": residual / max(expected_mass, 1.0e-30),
        "transfer_efficiency": deposited_mass / max(expected_mass, 1.0e-30),
        "released_particle_count": int(np.count_nonzero(host_released)),
        "deposited_particle_count": int(np.count_nonzero(host_deposited)),
        "escaped_particle_count": int(np.count_nonzero(host_escaped)),
        "live_particle_count": int(np.count_nonzero(host_alive)),
    }
    return WarpCaseResult(
        incidence_angle_deg=config.incidence_angle_deg,
        particle_count=n,
        particle_count_per_bin=config.particle_count_per_bin,
        map_mass_kg=host_map_mass,
        map_density_kg_m2=host_map_density,
        hit_positions_m=host_positions,
        represented_mass_kg=host_mass,
        deposited_flags=host_deposited,
        escaped_flags=host_escaped,
        alive_flags=host_alive,
        released_flags=host_released,
        performance=performance,
        mass_ledger=ledger,
    )
