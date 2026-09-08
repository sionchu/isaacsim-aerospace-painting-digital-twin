"""Build the deterministic, Isaac-free full-panel film plan."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerospace_painting.full_panel_planner import (  # noqa: E402
    DEFAULT_ANALYSIS_SHAPE,
    DEFAULT_FAN_INCIDENCE_DEG,
    DEFAULT_MASS_FLOW_KG_S,
    DEFAULT_OVERLAPS,
    DEFAULT_SPEEDS_M_S,
    DEFAULT_STAND_OFF_M,
    PanelSurface,
    S2RuntimeModel,
    plan_full_panel,
    save_plan,
)


def _yaml_scalar(path: Path, key: str, default: float) -> float:
    match = re.search(rf"^\s*{re.escape(key)}:\s*([0-9]+(?:\.[0-9]+)?)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    return float(match.group(1)) if match else float(default)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, default=ROOT.parent / "assets" / "course" / "generic_aircraft" / "generic_panel.usda")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "s2_air_assisted_v1.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "full_panel_film_demo.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "air_assisted" / "full_panel_film")
    args = parser.parse_args()
    config_text = args.config.read_text(encoding="utf-8")
    density = _yaml_scalar(ROOT / "configs" / "air_assisted_spray.yaml", "density_kg_m3", 1000.0)
    incidence = _yaml_scalar(args.config, "fan_incidence_deg", DEFAULT_FAN_INCIDENCE_DEG)
    stand_off = _yaml_scalar(args.config, "stand_off_m", DEFAULT_STAND_OFF_M)
    mass_flow = _yaml_scalar(args.config, "mass_flow_kg_s", DEFAULT_MASS_FLOW_KG_S)
    solids = _yaml_scalar(args.config, "volume_solids_fraction", 0.50)
    _ = config_text
    model = S2RuntimeModel.load(args.model)
    surface = PanelSurface.from_usda(args.asset, shape=DEFAULT_ANALYSIS_SHAPE)
    try:
        surface.source_path = args.asset.resolve().relative_to(ROOT.parent.resolve()).as_posix()
    except ValueError:
        surface.source_path = args.asset.name
    plan, summary, _ = plan_full_panel(
        surface,
        model,
        fan_incidence_deg=incidence,
        stand_off_m=stand_off,
        mass_flow_kg_s=mass_flow,
        overlaps=DEFAULT_OVERLAPS,
        speeds_m_s=DEFAULT_SPEEDS_M_S,
    )
    summary["configuration"] = {
        "path": args.config.relative_to(ROOT).as_posix(),
        "liquid_density_kg_m3": density,
        "liquid_density_source": "configs/air_assisted_spray.yaml",
        "illustrative_dft_enabled": True,
        "volume_solids_fraction": solids,
        "wft_primary": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_plan(plan, summary, args.output_dir / "full_panel_plan.json", model=model)
    (args.output_dir / "planner_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PLANNER_PASS shape={surface.shape} area_m2={surface.total_area_m2:.9f}")
    print(f"PLANNER_SELECTED overlap={plan.overlap_fraction:.3f} spacing_m={plan.spacing_m:.9f} speed_m_s={plan.speed_m_s:.3f} passes={plan.pass_count} cycle_s={plan.duration_s:.6f}")
    selected = summary["selected_metrics"]
    print(json.dumps({"wft": selected["wft"], "finite_surface": selected["finite_surface"], "runtime": selected["runtime"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
