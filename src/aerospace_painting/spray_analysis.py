"""Portable contracts for synchronized spray-analysis visualization.

The module contains no Isaac Sim or Warp imports.  It only defines the
deterministic view, trail, and timeline bookkeeping used by the native capture
adapter, so provenance and synchronization can be regression-tested on a
regular Python interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


DEPOSITION_VIEWS = ("s2", "warp", "compare")


def validate_deposition_view(value: str) -> str:
    view = str(value).lower()
    if view not in DEPOSITION_VIEWS:
        raise ValueError(f"deposition view must be one of {DEPOSITION_VIEWS}")
    return view


@dataclass
class ParticleTrailHistory:
    """Short histories keyed by actual Warp particle IDs."""

    max_particles: int = 200
    history_frames: int = 6
    samples: dict[int, list[tuple[float, np.ndarray]]] = field(default_factory=dict)

    def update(self, particle_ids: Iterable[int], positions_world_m: np.ndarray, time_s: float) -> None:
        ids = np.asarray(tuple(particle_ids), dtype=np.int64)
        positions = np.asarray(positions_world_m, dtype=np.float64)
        if positions.shape != (len(ids), 3) or not np.all(np.isfinite(positions)):
            raise ValueError("trail positions must be finite [N, 3]")
        if self.max_particles < 1 or self.history_frames < 2:
            raise ValueError("trail history limits must be positive")
        if len(ids) > self.max_particles:
            chosen = np.linspace(0, len(ids) - 1, self.max_particles, dtype=int)
            ids = ids[chosen]
            positions = positions[chosen]
        for particle_id, position in zip(ids.tolist(), positions):
            history = self.samples.setdefault(int(particle_id), [])
            history.append((float(time_s), np.asarray(position, dtype=np.float64).copy()))
            self.samples[int(particle_id)] = history[-self.history_frames :]
        cutoff = float(time_s) - max(0.25, self.history_frames / 30.0)
        self.samples = {
            particle_id: values
            for particle_id, values in self.samples.items()
            if values and values[-1][0] >= cutoff
        }

    def lines(self) -> list[np.ndarray]:
        return [
            np.asarray([position for _, position in values], dtype=np.float64)
            for values in self.samples.values()
            if len(values) >= 2
        ]

    def record_segment(
        self,
        particle_id: int,
        previous_position_world_m: Iterable[float],
        position_world_m: Iterable[float],
        time_s: float,
    ) -> None:
        """Record one actual Warp integration segment for a hit particle.

        Hit particles can leave the active set before the next render frame.
        The segment is still an actual pair of Warp positions, so retaining it
        briefly makes the impact trajectory visible without inventing points.
        """

        previous = np.asarray(tuple(previous_position_world_m), dtype=np.float64)
        current = np.asarray(tuple(position_world_m), dtype=np.float64)
        if previous.shape != (3,) or current.shape != (3,) or not np.all(np.isfinite((previous, current))):
            raise ValueError("trail segment positions must be finite length-three vectors")
        particle_id = int(particle_id)
        history = self.samples.setdefault(particle_id, [])
        if not history:
            history.append((float(time_s) - 1.0e-6, previous.copy()))
        history.append((float(time_s), current.copy()))
        self.samples[particle_id] = history[-self.history_frames :]
        cutoff = float(time_s) - max(0.25, self.history_frames / 30.0)
        self.samples = {
            sample_id: values
            for sample_id, values in self.samples.items()
            if values and values[-1][0] >= cutoff
        }


@dataclass
class TimelineSync:
    """Maximum observed offsets between one simulated timeline and its layers."""

    max_warp_to_s2_time_offset_s: float = 0.0
    max_hit_to_overlay_time_offset_s: float = 0.0
    max_overlay_to_simulated_time_offset_s: float = 0.0
    observation_count: int = 0

    def observe(
        self,
        *,
        simulated_time_s: float,
        s2_time_s: float,
        overlay_time_s: float,
        warp_emission_time_s: float | None = None,
        hit_time_s: Iterable[float] = (),
    ) -> None:
        sim = float(simulated_time_s)
        s2 = float(s2_time_s)
        overlay = float(overlay_time_s)
        self.max_overlay_to_simulated_time_offset_s = max(
            self.max_overlay_to_simulated_time_offset_s,
            abs(overlay - sim),
        )
        if warp_emission_time_s is not None:
            self.max_warp_to_s2_time_offset_s = max(
                self.max_warp_to_s2_time_offset_s,
                abs(float(warp_emission_time_s) - s2),
            )
        for hit_time in hit_time_s:
            self.max_hit_to_overlay_time_offset_s = max(
                self.max_hit_to_overlay_time_offset_s,
                abs(float(hit_time) - overlay),
            )
        self.observation_count += 1

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "max_warp_to_s2_time_offset_s": float(self.max_warp_to_s2_time_offset_s),
            "max_hit_to_overlay_time_offset_s": float(self.max_hit_to_overlay_time_offset_s),
            "max_overlay_to_simulated_time_offset_s": float(self.max_overlay_to_simulated_time_offset_s),
            "observation_count": int(self.observation_count),
            "same_simulated_timeline": self.observation_count > 0 and self.max_overlay_to_simulated_time_offset_s <= 1.0e-12,
        }


def visual_layer_mass_contract() -> dict[str, dict[str, bool]]:
    """Declare which display-only layers are forbidden from changing ledgers."""

    return {
        "cfd_reference_flow": {"visual_only": True, "adds_mass": False, "adds_deposition": False},
        "warp_particle_trails": {"visual_only": True, "adds_mass": False, "adds_deposition": False},
        "warp_impact_markers": {"visual_only": True, "adds_mass": False, "adds_deposition": False},
        "warp_diagnostic_hit_map": {"visual_only": True, "adds_mass": False, "adds_deposition": False},
    }
