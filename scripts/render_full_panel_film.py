"""Render the final full-panel WFT/mass/DFT analysis artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "air_assisted" / "full_panel_film"
MEDIA_DIR = ROOT / "media" / "air_assisted_spray"


def _save_map(path: Path, y: np.ndarray, z: np.ndarray, values: np.ndarray, title: str, colorbar: str, cmap: str = "viridis") -> None:
    figure, axis = plt.subplots(figsize=(13.333, 7.5), dpi=144)
    mesh = axis.scatter(y, z, c=values, s=10, cmap=cmap, edgecolors="none")
    axis.set_xlabel("Panel long coordinate y [m]")
    axis.set_ylabel("Panel cross-track coordinate z [m]")
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    figure.colorbar(mesh, ax=axis, label=colorbar)
    figure.tight_layout()
    figure.savefig(path, dpi=144)
    plt.close(figure)


def main() -> int:
    metrics_path = RESULT_DIR / "full_panel_metrics.json"
    data_path = RESULT_DIR / "full_panel_surface_data.npz"
    if not metrics_path.is_file() or not data_path.is_file():
        print("RENDER_BLOCKED missing native full-panel metrics or surface data", file=sys.stderr)
        return 2
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    arrays = np.load(data_path)
    positions = np.asarray(arrays["positions_m"], dtype=float)
    area = np.asarray(arrays["area_weights_m2"], dtype=float)
    mass = np.asarray(arrays["cumulative_mass_kg"], dtype=float)
    wft = np.asarray(arrays["wft_um"], dtype=float)
    density = float(metrics["estimated_wft"]["liquid_density_kg_m3"])
    areal_mass = np.divide(mass, area, out=np.zeros_like(mass), where=area > 0.0)
    dft = wft * 0.50
    y = positions[:, 1]
    z = positions[:, 2]
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    _save_map(MEDIA_DIR / "full_panel_wft_map.png", y, z, wft, "Full-panel estimated wet-film thickness", "Estimated WFT [µm]")
    _save_map(MEDIA_DIR / "full_panel_mass_density_map.png", y, z, areal_mass, "Full-panel deposited mass density", "Deposited mass density [kg/m²]", cmap="magma")
    _save_map(MEDIA_DIR / "full_panel_dft_demo_map.png", y, z, dft, "Illustrative DFT estimate — assumed 50% volume solids", "Illustrative DFT estimate [µm]", cmap="cividis")
    figure, axis = plt.subplots(figsize=(11, 7), dpi=144)
    axis.hist(wft[wft > 0.0], bins=40, color="#2f74a8", alpha=0.9, edgecolor="white")
    axis.axvline(float(metrics["estimated_wft"]["stats"]["mean_um"]), color="#d78b2b", linewidth=2.0, label="area-weighted mean")
    axis.set_xlabel("Estimated WFT [µm]")
    axis.set_ylabel("Analysis vertices")
    axis.set_title("Full-panel estimated WFT distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(MEDIA_DIR / "full_panel_wft_histogram.png", dpi=144)
    plt.close(figure)
    print(f"RENDER_PASS density={density:g} vertices={len(wft)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
