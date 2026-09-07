"""Portable contract for the externally composed Isaac Sim scene."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneContract:
    base_scene: Path
    robot_reference_policy: str = "official Isaac Sim asset by URI or local asset root"
    aircraft_policy: str = "generic course-owned curved surface only"

    @classmethod
    def from_environment(cls) -> "SceneContract":
        value = os.environ.get("AEROSPACE_PAINTING_BASE_USD")
        if not value:
            raise RuntimeError(
                "Set AEROSPACE_PAINTING_BASE_USD to a locally composed painting base scene. "
                "Raw NVIDIA or third-party assets are intentionally not distributed here."
            )
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return cls(base_scene=path)
