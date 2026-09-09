"""Capture a synchronized CFD → Warp → impact → S2 WFT analysis view.

This adapter reuses the validated full-panel plan and native PaintingCycle.
OpenFOAM remains an offline reference field, Warp remains diagnostic
Lagrangian transport, and S2 remains the authoritative deposited-mass/WFT
layer.  The script adds only synchronized visualization and evidence capture.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = ROOT.parent / "visual_prototypes"
SCENE_PATH = VISUAL_ROOT / "scenes" / "aircraft_painting_cell.usda"
PANEL_ASSET = ROOT.parent / "assets" / "course" / "generic_aircraft" / "generic_panel.usda"
MODEL_PATH = ROOT / "models" / "s2_air_assisted_v1.json"
PLAN_PATH = ROOT / "results" / "air_assisted" / "full_panel_film" / "full_panel_plan.json"
DENSITY_SOURCE_PATH = ROOT / "configs" / "air_assisted_spray.yaml"
WARP_MODEL_PATH = ROOT / "models" / "warp_vector_carrier_v3.json"
FLOW_NPZ = ROOT / "models" / "openfoam_flow_field_7p5deg_v1.npz"
FLOW_JSON = ROOT / "models" / "openfoam_flow_field_7p5deg_v1.json"
MEDIA_DIR = ROOT / "media" / "air_assisted_spray"
RESULT_DIR = ROOT / "results" / "air_assisted" / "spray_analysis"
FRAME_DIR = RESULT_DIR / "_frames"
FPS = 30
RESOLUTION = (1920, 1080)
TARGET_VIDEO_SECONDS = 85.0
ROOT_PRIM = "/World/AerospacePaintingCell"
RAIL_PRIM = ROOT_PRIM + "/LinearTrack/Carriage"
ROBOT_PRIM = ROOT_PRIM + "/LinearTrack/Carriage/PaintRobot/FANUC"
FLANGE_PRIM = ROBOT_PRIM + "/Asset/J6_link/flange"
PROCESS_TOOL_PRIM = FLANGE_PRIM + "/ProcessTool"
SPRAY_TOOL_PRIM = PROCESS_TOOL_PRIM + "/SprayGun"
PANEL_PRIM = "/World/Aircraft/Panel/Asset"
WFT_OVERLAY_PRIM = ROOT_PRIM + "/FilmThicknessOverlay"
WARP_PLUME_PRIM = ROOT_PRIM + "/WarpSprayPlume"
WARP_GUIDE_PRIM = ROOT_PRIM + "/WarpSprayPlumeGuide"
FLOW_ROOT_PRIM = ROOT_PRIM + "/SynchronizedOpenFOAMFlow"
FLOW_VECTORS_PRIM = FLOW_ROOT_PRIM + "/VelocityVectors"
FLOW_STREAMLINES_PRIM = FLOW_ROOT_PRIM + "/Streamlines"
TRAILS_PRIM = ROOT_PRIM + "/WarpParticleTrails"
IMPACTS_PRIM = ROOT_PRIM + "/WarpImpactMarkers"
HIT_MAP_PRIM = ROOT_PRIM + "/WarpDiagnosticHitDensity"
CLOSEUP_CAMERA_PATH = "/World/Cameras/synchronized_spray_closeup"


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(array.max()),
    }


def _set_visible(prim, visible: bool) -> None:
    from pxr import UsdGeom

    value = prim.GetPrim() if hasattr(prim, "GetPrim") else prim
    UsdGeom.Imageable(value).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)


def _capture(viewport, path: Path, simulation_app) -> None:
    from omni.kit.viewport.utility import capture_viewport_to_file

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    capture_viewport_to_file(viewport, file_path=str(path))
    for _ in range(100):
        simulation_app.update()
        if path.exists() and path.stat().st_size > 0:
            return
    raise RuntimeError(f"viewport capture did not complete: {path}")


def _create_points(stage, path: str, *, color: tuple[float, float, float], width_m: float, source: str, opacity: float = 0.9):
    from pxr import Sdf, UsdGeom

    points = UsdGeom.Points.Define(stage, path)
    points.CreatePointsAttr([])
    points.CreateWidthsAttr([])
    points.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([color])
    points.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(opacity)])
    points.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    prim = points.GetPrim()
    prim.CreateAttribute("visual:source", Sdf.ValueTypeNames.String).Set(source)
    prim.CreateAttribute("visual:only", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("visual:addsMass", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("visual:addsDeposition", Sdf.ValueTypeNames.Bool).Set(False)
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return points


def _write_points(points, positions: np.ndarray, *, width_m: float, visible: bool = True) -> None:
    from pxr import UsdGeom

    values = np.asarray(positions, dtype=np.float32).reshape((-1, 3))
    points.GetPointsAttr().Set(values.tolist() if len(values) else [])
    points.GetWidthsAttr().Set([float(width_m)] * len(values))
    _set_visible(points, visible and len(values) > 0)


def _create_visual_particle_instancer(stage, path: str, *, color: tuple[float, float, float], radius_m: float, source: str, opacity: float = 1.0):
    """Create renderer-portable emissive spheres for display-only positions.

    RTX point widths are not consistently visible when a point lies on or
    just inside the panel surface.  A batched PointInstancer keeps the same
    actual positions while giving impact/particle markers a small volume that
    remains legible in the native capture.
    """

    from pxr import Gf, Sdf, UsdGeom, UsdShade

    instancer = UsdGeom.PointInstancer.Define(stage, path)
    prototype = UsdGeom.Sphere.Define(stage, path + "/Prototype")
    prototype.CreateRadiusAttr().Set(float(radius_m))
    material = UsdShade.Material.Define(stage, path + "/Material")
    shader = UsdShade.Shader.Define(stage, path + "/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(0.8 * np.asarray(color))))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prototype.GetPrim()).Bind(material)
    instancer.CreatePrototypesRel().SetTargets([prototype.GetPath()])
    instancer.CreatePositionsAttr([])
    instancer.CreateProtoIndicesAttr([])
    instancer.CreateScalesAttr([])
    prim = instancer.GetPrim()
    prim.CreateAttribute("visual:source", Sdf.ValueTypeNames.String).Set(source)
    prim.CreateAttribute("visual:only", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("visual:addsMass", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("visual:addsDeposition", Sdf.ValueTypeNames.Bool).Set(False)
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return instancer


def _write_instanced_markers(instancer, positions: np.ndarray, *, visible: bool = True) -> None:
    from pxr import UsdGeom

    values = np.asarray(positions, dtype=np.float32).reshape((-1, 3))
    instancer.GetPositionsAttr().Set(values.tolist() if len(values) else [])
    instancer.GetProtoIndicesAttr().Set([0] * len(values))
    instancer.GetScalesAttr().Set([(1.0, 1.0, 1.0)] * len(values))
    _set_visible(instancer, visible and len(values) > 0)


def _create_trails(stage):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    curves = UsdGeom.BasisCurves.Define(stage, TRAILS_PRIM)
    curves.CreatePointsAttr([])
    curves.CreateCurveVertexCountsAttr([])
    curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curves.CreateWidthsAttr().Set([0.0022])
    curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curves.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set([(1.0, 0.52, 0.12)])
    curves.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    material = UsdShade.Material.Define(stage, "/World/Looks/ActualWarpParticleTrails")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/ActualWarpParticleTrails/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.30, 0.03))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.18, 0.01))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(material)
    prim = curves.GetPrim()
    prim.CreateAttribute("warp:source", Sdf.ValueTypeNames.String).Set("actual Warp particle positions, recorded per analysis frame")
    prim.CreateAttribute("warp:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("warp:addsMass", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("warp:addsDeposition", Sdf.ValueTypeNames.Bool).Set(False)
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return curves


def _write_trails(curves, lines: list[np.ndarray], *, visible: bool = True) -> int:
    from pxr import UsdGeom

    valid = [np.asarray(line, dtype=np.float32) for line in lines if len(line) >= 2]
    points = np.concatenate(valid, axis=0) if valid else np.empty((0, 3), dtype=np.float32)
    curves.GetPointsAttr().Set(points.tolist() if len(points) else [])
    curves.GetCurveVertexCountsAttr().Set([int(len(line)) for line in valid])
    _set_visible(curves, visible and bool(valid))
    return int(sum(max(len(line) - 1, 0) for line in valid))


def _create_flow_geometry(stage, field, *, roi_min_m, roi_max_m, vector_count: int, streamline_count: int):
    from aerospace_painting.openfoam_flow_field import clip_flow_roi, deterministic_streamlines
    from pxr import UsdGeom
    from run_isaac_cfd_flow_visualization import _create_batched_line_mesh, _create_basis_curves

    points, velocities, speeds, _ = clip_flow_roi(field, roi_min_m, roi_max_m)
    if len(points) == 0:
        raise RuntimeError("CFD ROI contains no solved field samples")
    count = min(int(vector_count), len(points))
    sample_indices = np.linspace(0, len(points) - 1, count, dtype=int)
    vector_lines: list[np.ndarray] = []
    vector_colors: list[tuple[float, float, float]] = []
    lo, hi = float(speeds.min()), float(speeds.max())
    for index in sample_indices:
        start = np.asarray(points[index], dtype=np.float64)
        velocity = np.asarray(velocities[index], dtype=np.float64)
        speed = float(speeds[index])
        direction = velocity / max(float(np.linalg.norm(velocity)), 1.0e-12)
        length = 0.0015 + 0.0017 * speed
        q = float(np.clip((speed - lo) / max(hi - lo, 1.0e-9), 0.0, 1.0))
        vector_lines.append(np.asarray((start, start + direction * length), dtype=np.float64))
        vector_colors.append((0.05 + 0.10 * q, 0.48 + 0.40 * q, 0.90 + 0.08 * q))

    seed_x = np.linspace(float(roi_min_m[0]) * 0.65, float(roi_max_m[0]) * 0.65, 5)
    seed_y = np.linspace(float(roi_min_m[1]) * 0.65, float(roi_max_m[1]) * 0.65, 5)
    seeds = np.asarray([(x, y, max(float(roi_min_m[2]) + 0.005, 0.006)) for x in seed_x for y in seed_y], dtype=float)
    raw_streamlines = deterministic_streamlines(field, seeds, step_m=0.005, max_steps=52)
    streamlines_local: list[np.ndarray] = []
    for line in raw_streamlines:
        inside = np.all((line >= np.asarray(roi_min_m)[None, :]) & (line <= np.asarray(roi_max_m)[None, :]), axis=1)
        clipped = line[inside]
        if len(clipped) >= 2:
            streamlines_local.append(clipped)
    streamlines_local = streamlines_local[: int(streamline_count)]
    streamline_colors = [(0.04, 0.92, 0.98)] * len(streamlines_local)

    root = UsdGeom.Xform.Define(stage, FLOW_ROOT_PRIM)
    transform_op = root.AddTransformOp()
    vectors = _create_basis_curves(stage, FLOW_VECTORS_PRIM, vector_lines, vector_colors, 0.0040, "OpenFOAM v2606 solved U in clipped process ROI")
    vector_mesh = _create_batched_line_mesh(stage, FLOW_VECTORS_PRIM + "Mesh", vector_lines, vector_colors, 0.0040, "OpenFOAM v2606 solved U in clipped process ROI", arrowheads=True)
    streamlines = _create_basis_curves(stage, FLOW_STREAMLINES_PRIM, streamlines_local, streamline_colors, 0.0058, "OpenFOAM v2606 solved U streamlines in clipped process ROI")
    streamline_mesh = _create_batched_line_mesh(stage, FLOW_STREAMLINES_PRIM + "Mesh", streamlines_local, streamline_colors, 0.0058, "OpenFOAM v2606 solved U streamlines in clipped process ROI")
    for prim in (root, vectors, vector_mesh, streamlines, streamline_mesh):
        _set_visible(prim, True)
    return {
        "root": root,
        "transform_op": transform_op,
        "vectors": vectors,
        "vector_mesh": vector_mesh,
        "streamlines": streamlines,
        "streamline_mesh": streamline_mesh,
        "vector_count": len(vector_lines),
        "streamline_count": len(streamlines_local),
        "roi_min_m": tuple(float(value) for value in roi_min_m),
        "roi_max_m": tuple(float(value) for value in roi_max_m),
    }


def _create_closeup_camera(stage):
    from pxr import UsdGeom

    camera = UsdGeom.Camera.Define(stage, CLOSEUP_CAMERA_PATH)
    camera.CreateFocalLengthAttr().Set(44.0)
    camera.CreateHorizontalApertureAttr().Set(36.0)
    camera.CreateVerticalApertureAttr().Set(20.25)
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    return camera, xform.AddTransformOp()


def _update_closeup_camera(camera_op, origin_world_m, frame) -> None:
    from pxr import Gf

    origin = np.asarray(origin_world_m, dtype=float)
    u = np.asarray(frame.u_world, dtype=float)
    v = np.asarray(frame.v_world, dtype=float)
    w = np.asarray(frame.w_world, dtype=float)
    # Keep the same behind-the-nozzle composition as the validated CFD still:
    # the full clipped ROI (nozzle to panel) remains in frame instead of being
    # occluded by the tool or pushed outside the close-up crop.
    target = origin + 0.12 * w
    eye = origin - 0.62 * w + 0.48 * u + 0.72 * v
    view = target - eye
    view /= max(float(np.linalg.norm(view)), 1.0e-12)
    right = np.cross(view, v)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    up = np.cross(right, view)
    up /= max(float(np.linalg.norm(up)), 1.0e-12)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRow(0, Gf.Vec4d(float(right[0]), float(right[1]), float(right[2]), 0.0))
    matrix.SetRow(1, Gf.Vec4d(float(up[0]), float(up[1]), float(up[2]), 0.0))
    matrix.SetRow(2, Gf.Vec4d(float(-view[0]), float(-view[1]), float(-view[2]), 0.0))
    matrix.SetRow(3, Gf.Vec4d(float(eye[0]), float(eye[1]), float(eye[2]), 1.0))
    camera_op.Set(matrix)


def _wft_mean_um(grid, density_kg_m3: float) -> float:
    return float(grid.integrated_mass_kg / max(grid.total_area_m2 * float(density_kg_m3), 1.0e-30) * 1.0e6)


def _phase(simulation_time_s: float, duration_s: float) -> tuple[float, str]:
    video_time = TARGET_VIDEO_SECONDS * float(simulation_time_s) / max(float(duration_s), 1.0e-9)
    if video_time < 8.0:
        return video_time, "process"
    if video_time < 20.0:
        return video_time, "flow"
    if video_time < 40.0:
        return video_time, "combined_closeup"
    if video_time < 70.0:
        return video_time, "combined_wide"
    return video_time, "result"


def _set_analysis_visibility(phase: str, deposition_view: str, flow, warp_points, trails, impacts, hit_map, overlay, guide) -> None:
    flow_on = phase in {"flow", "combined_closeup"}
    combined_on = phase in {"combined_closeup", "combined_wide"}
    _set_visible(flow["root"], flow_on)
    _set_visible(flow["vectors"], flow_on)
    _set_visible(flow["vector_mesh"], flow_on)
    _set_visible(flow["streamlines"], flow_on)
    _set_visible(flow["streamline_mesh"], flow_on)
    _set_visible(warp_points, phase in {"process", "combined_closeup", "combined_wide"})
    _set_visible(trails, combined_on)
    _set_visible(impacts, combined_on)
    _set_visible(hit_map, deposition_view in {"warp", "compare"} and phase in {"combined_closeup", "combined_wide", "result"})
    _set_visible(overlay, deposition_view in {"s2", "compare"} and phase in {"combined_closeup", "combined_wide", "result"})
    # The guide is never enabled in this technical view.
    if guide is not None and guide.IsValid():
        _set_visible(guide, False)


def _annotate(path: Path, *, phase: str, deposition_view: str, pass_index: int, pass_count: int, spray_on: bool, deposited_mass_kg: float, mean_wft_um: float, stand_off_m: float, incidence_deg: float, flow_report: dict, active_particles: int, hit_marker_count: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    bold_path = Path("C:/Windows/Fonts/seguisb.ttf")
    try:
        body = ImageFont.truetype(str(font_path), 21) if font_path.exists() else ImageFont.load_default()
        title = ImageFont.truetype(str(bold_path), 28) if bold_path.exists() else body
        small = ImageFont.truetype(str(font_path), 17) if font_path.exists() else body
    except Exception:
        body = title = small = ImageFont.load_default()
    phase_label = {
        "process": "PROCESS FRAME",
        "flow": "CFD FLOW DIRECTION",
        "combined_closeup": "ANALYSIS_VIEW — CLOSE-UP",
        "combined_wide": "ANALYSIS_VIEW — FULL PANEL",
        "result": "RESULT — ESTIMATED WFT",
    }[phase]
    lines = [
        phase_label,
        "OPENFOAM REFERENCE FLOW (offline)",
        "Warp GPU Droplet Transport | actual particles only | guide OFF",
        "S2 Estimated WFT | deposition-view: " + deposition_view,
        f"Stand-off: {stand_off_m:.3f} m   Incidence: {incidence_deg:.1f} deg   Spray: {'ON' if spray_on else 'OFF'}",
        f"Pass: {pass_index}/{pass_count}   Deposited mass: {deposited_mass_kg * 1.0e3:.4f} g   Current mean WFT: {mean_wft_um:.3f} um",
        f"CFD_DIRECTION_PASS | axial mean/min/max: {flow_report['mean_axial_velocity_m_s']:.2f}/{flow_report['min_axial_velocity_m_s']:.2f}/{flow_report['max_axial_velocity_m_s']:.2f} m/s",
        f"Active Warp particles: {active_particles}   Impact markers: {hit_marker_count}",
    ]
    if phase == "combined_closeup":
        lines.insert(1, "Close-up: nozzle → CFD streamlines → actual particles → impacts → S2 WFT")
    heights = [title.getbbox(line)[3] - title.getbbox(line)[1] if i == 0 else body.getbbox(line)[3] - body.getbbox(line)[1] for i, line in enumerate(lines)]
    left, top = 34, 30
    gap = 8
    panel_height = sum(heights) + gap * (len(lines) - 1) + 28
    panel_width = min(image.width - 68, max(930, max(int(draw.textlength(line, font=title if i == 0 else body)) for i, line in enumerate(lines)) + 44))
    draw.rounded_rectangle((left, top, left + panel_width, top + panel_height), radius=12, fill=(4, 16, 25, 220), outline=(65, 189, 207, 230), width=2)
    y = top + 13
    for i, line in enumerate(lines):
        use_font = title if i == 0 else body
        color = (226, 249, 253, 255) if i == 0 else (207, 238, 244, 255)
        draw.text((left + 20, y), line, font=use_font, fill=color)
        y += heights[i] + gap
    legend_y = image.height - 66
    draw.rounded_rectangle((34, legend_y, 520, image.height - 28), radius=9, fill=(4, 16, 25, 205), outline=(65, 189, 207, 190), width=2)
    draw.text((52, legend_y + 8), "cyan = OpenFOAM U   orange = Warp   warm tint = S2 WFT", font=small, fill=(200, 241, 246, 255))
    image.convert("RGB").save(path)


def _encode_video(captured: list[Path], sequence_dir: Path, target: Path, target_seconds: float) -> dict[str, object]:
    if not captured:
        return {"path": None, "duration_s": None, "codec": None}
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"path": None, "duration_s": None, "codec": None, "error": "ffmpeg not found"}
    # The sequence directory is generated output.  Clear stale frames from a
    # prior smoke/crashed run so ffmpeg cannot append them to the new video.
    if sequence_dir.exists():
        shutil.rmtree(sequence_dir)
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(captured):
        shutil.copy2(source, sequence_dir / f"frame_{index:06d}.png")
    source_duration = max(len(captured) / FPS, 1.0 / FPS)
    time_scale = float(target_seconds) / source_duration
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", str(sequence_dir / "frame_%06d.png"),
        "-vf", f"setpts={time_scale:.8f}*PTS,format=yuv420p",
        "-t", f"{target_seconds:.3f}", "-r", str(FPS), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return {"path": None, "duration_s": None, "codec": None, "error": result.stderr.strip()}
    return {"path": target.relative_to(ROOT).as_posix(), "duration_s": float(target_seconds), "codec": "H.264"}


def run(args: argparse.Namespace) -> int:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(launch_config={"headless": bool(args.headless), "width": RESOLUTION[0], "height": RESOLUTION[1]})
    started = time.perf_counter()
    warp_runtime = None
    try:
        import omni.usd
        from pxr import UsdGeom

        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(ROOT / "scripts"))
        sys.path.insert(0, str(VISUAL_ROOT))
        from aerospace_painting.full_panel_planner import PanelSurface, film_statistics
        from aerospace_painting.openfoam_flow_field import OpenFOAMFlowField, evaluate_cfd_direction
        from aerospace_painting.s2_runtime import FiniteSurfaceAccumulator, S2Runtime, S2RuntimeModel, StructuredSurfaceGrid, surface_frame
        from aerospace_painting.spray_analysis import ParticleTrailHistory, TimelineSync, validate_deposition_view, visual_layer_mass_contract
        from aerospace_painting.warp_plume_runtime import WarpBatchFrame, WarpPlumeRuntime
        from run_isaac_cfd_flow_visualization import _create_hud, _set_flow_transform
        from run_isaac_full_panel_painting import (
            FullPanelPaintingCycle,
            _create_overlay,
            _create_warp_points,
            _density_from_config,
            _load_plan,
            _translation,
            _vec3,
            _write_overlay,
            _write_warp_points,
        )

        deposition_view = validate_deposition_view(args.deposition_view)
        field = OpenFOAMFlowField.load(FLOW_NPZ, FLOW_JSON)
        model = S2RuntimeModel.load(MODEL_PATH)
        surface = PanelSurface.from_usda(PANEL_ASSET)
        surface.source_path = "assets/course/generic_aircraft/generic_panel.usda"
        plan = _load_plan(model, surface)
        if args.max_seconds is not None:
            plan.duration_s = min(float(plan.duration_s), max(0.1, float(args.max_seconds)))
        liquid_density = _density_from_config(DENSITY_SOURCE_PATH)
        context = omni.usd.get_context()
        context.open_stage(str(SCENE_PATH))
        stage = None
        for _ in range(240):
            simulation_app.update()
            stage = context.get_stage()
            if stage and stage.GetPrimAtPath(ROOT_PRIM).IsValid():
                break
        if not stage or not stage.GetPrimAtPath(ROOT_PRIM).IsValid():
            raise RuntimeError("base painting stage did not load")
        required = [ROOT_PRIM, RAIL_PRIM, ROBOT_PRIM, FLANGE_PRIM, PROCESS_TOOL_PRIM, SPRAY_TOOL_PRIM, PANEL_PRIM]
        missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
        if missing:
            raise RuntimeError(f"required scene prims missing: {missing}")
        print(f"SCENE_GATE_PASS scene={SCENE_PATH}", flush=True)

        grid = StructuredSurfaceGrid.from_structured(surface.positions.reshape(-1, 3), surface.normals.reshape(-1, 3), surface.shape)
        cycle = FullPanelPaintingCycle(stage, ROOT.parent, plan)
        overlay = _create_overlay(stage, grid)
        warp_points = _create_visual_particle_instancer(
            stage,
            WARP_PLUME_PRIM + "ActualInstancer",
            color=(1.0, 0.24, 0.02),
            radius_m=0.0045,
            source="actual active NVIDIA Warp particle positions; display-only point size",
            opacity=0.98,
        )
        trails = _create_trails(stage)
        impacts = _create_visual_particle_instancer(
            stage,
            IMPACTS_PRIM,
            color=(1.0, 0.25, 0.02),
            radius_m=0.007,
            source="actual Warp mesh-hit positions; short-lived visual marker",
            opacity=0.98,
        )
        hit_map = _create_visual_particle_instancer(
            stage,
            HIT_MAP_PRIM,
            color=(1.0, 0.60, 0.05),
            radius_m=0.0045,
            source="sampled actual Warp mesh-hit positions; diagnostic projected hit density",
            opacity=0.45,
        )
        guide = stage.GetPrimAtPath(WARP_GUIDE_PRIM)
        runtime = S2Runtime(model)
        accumulator = FiniteSurfaceAccumulator(grid)
        warp_runtime = WarpPlumeRuntime(
            model_path=WARP_MODEL_PATH,
            repo_root=ROOT,
            mass_flow_kg_s=model.mass_flow_kg_s,
            stand_off_m=model.stand_off_m,
            particle_count_per_bin=int(args.warp_particles_per_bin),
            cadence_hz=float(args.warp_cadence_hz),
            particle_dt_s=5.0e-5,
            maximum_flight_s=0.06,
        )
        roi_min = (-0.06, -0.06, 0.0)
        roi_max = (0.06, 0.06, 0.24)
        total_frames = int(round(plan.duration_s * FPS)) + 1
        dt = plan.duration_s / max(total_frames - 1, 1)
        first_pose = cycle.update(stage, 0, total_frames)
        first_axis = np.asarray(cycle.base.current_spray_axis, dtype=float)
        first_frame = surface_frame(first_pose.normal, first_pose.frame.u_fan_major, first_axis, minor_tolerance_deg=89.0)
        direction_report = evaluate_cfd_direction(
            field,
            roi_min_m=roi_min,
            roi_max_m=roi_max,
            spray_axis_world=first_axis,
            u_world=first_frame.u_fan_major,
            v_world=first_frame.v_fan_minor,
            w_world=first_frame.w_inward,
        )
        if direction_report.status != "CFD_DIRECTION_PASS":
            print(f"BLOCKED_CFD_DIRECTION {json.dumps(direction_report.as_dict(), sort_keys=True)}", flush=True)
            return 2
        flow = _create_flow_geometry(stage, field, roi_min_m=roi_min, roi_max_m=roi_max, vector_count=int(args.vector_count), streamline_count=int(args.streamline_count))
        camera, camera_op = _create_closeup_camera(stage)
        _set_flow_transform(flow["transform_op"], WarpBatchFrame.from_surface_frame(origin_world_m=_translation(cycle.base.current_tcp), target_world_m=cycle.base.current_surface_point, surface_frame=first_frame, incidence_deg=first_frame.incidence_u_deg, stand_off_m=model.stand_off_m))
        viewport_window = __import__("omni.kit.viewport.utility", fromlist=["create_viewport_window"]).create_viewport_window(name="SynchronizedSprayAnalysisViewport", width=RESOLUTION[0], height=RESOLUTION[1], camera_path="/World/Cameras/process_painting")
        viewport = viewport_window.viewport_api if viewport_window else __import__("omni.kit.viewport.utility", fromlist=["get_active_viewport"]).get_active_viewport()
        if viewport is None:
            raise RuntimeError("native viewport is unavailable")
        viewport.resolution = RESOLUTION
        hud = _create_hud(viewport_window, mode="combined", slice_y_m=0.0)
        if FRAME_DIR.exists():
            shutil.rmtree(FRAME_DIR)
        FRAME_DIR.mkdir(parents=True, exist_ok=True)

        trail_history = ParticleTrailHistory(max_particles=200, history_frames=6)
        timeline = TimelineSync()
        sampled_hit_positions: list[np.ndarray] = []
        active_particle_counts: list[int] = []
        impact_marker_counts: list[int] = []
        trail_segment_counts: list[int] = []
        cfd_seconds: list[float] = []
        warp_seconds: list[float] = []
        impact_seconds: list[float] = []
        overlay_seconds: list[float] = []
        analysis_seconds: list[float] = []
        captured: list[Path] = []
        closeup_written = False
        combined_written = False
        progressive_written = False
        capture_index = 0
        spray_off_batch_violations = 0
        timeline_samples: list[dict[str, object]] = []
        impact_markers: list[tuple[float, np.ndarray]] = []
        print(f"CFD_DIRECTION_PASS {json.dumps(direction_report.as_dict(), sort_keys=True)}", flush=True)

        for frame_index in range(total_frames):
            simulation_time_s = plan.duration_s * frame_index / max(total_frames - 1, 1)
            video_time, phase = _phase(simulation_time_s, plan.duration_s)
            analysis_start = time.perf_counter()
            pose = first_pose if frame_index == 0 else cycle.update(stage, frame_index, total_frames)
            actual_axis = np.asarray(cycle.base.current_spray_axis, dtype=float)
            measured_frame = surface_frame(pose.normal, pose.frame.u_fan_major, actual_axis, minor_tolerance_deg=3.0)
            nozzle = _translation(cycle.base.current_tcp)
            warp_frame = WarpBatchFrame.from_surface_frame(
                origin_world_m=nozzle,
                target_world_m=cycle.base.current_surface_point,
                surface_frame=measured_frame,
                incidence_deg=measured_frame.incidence_u_deg,
                stand_off_m=model.stand_off_m,
            )
            cfd_start = time.perf_counter()
            _set_flow_transform(flow["transform_op"], warp_frame)
            _update_closeup_camera(camera_op, nozzle, warp_frame)
            cfd_seconds.append(time.perf_counter() - cfd_start)

            s2_start = time.perf_counter()
            step = runtime.step(
                dt_s=dt,
                incidence_angle_deg=plan.fan_incidence_deg,
                stand_off_m=model.stand_off_m,
                air_velocity_m_s=model.air_velocity_m_s,
                mass_flow_kg_s=model.mass_flow_kg_s,
                spray_on=bool(pose.spray_on),
            )
            if pose.spray_on:
                accumulator.apply(step, target_position=cycle.base.current_surface_point, frame=pose.frame)
            _write_overlay(overlay, grid, liquid_density)
            overlay_seconds.append(time.perf_counter() - s2_start)
            current_mass = float(accumulator.surface_captured_mass_kg)
            current_wft = _wft_mean_um(grid, liquid_density)

            warp_start = time.perf_counter()
            previous_batch_count = int(warp_runtime.as_dict()["batch_count"])
            warp_runtime.update(
                simulation_time_s,
                bool(pose.spray_on),
                warp_frame,
                actual_tool_origin_m=nozzle,
                actual_tool_axes=(_vec3(cycle.base.current_tcp, (1.0, 0.0, 0.0)), _vec3(cycle.base.current_tcp, (0.0, 1.0, 0.0)), _vec3(cycle.base.current_tcp, (0.0, 0.0, 1.0))),
            )
            new_batch_count = int(warp_runtime.as_dict()["batch_count"])
            if not pose.spray_on and new_batch_count > previous_batch_count:
                spray_off_batch_violations += 1
            particle_ids, particle_positions = warp_runtime.active_particles()
            # Actual Warp positions are sub-pixel at native scale; this is a
            # display-only point-size increase, not a change to transport.
            _write_instanced_markers(warp_points, particle_positions, visible=True)
            trail_history.update(particle_ids, particle_positions, simulation_time_s)
            warp_seconds.append(time.perf_counter() - warp_start)

            impact_start = time.perf_counter()
            new_hits = warp_runtime.drain_hit_events()
            for event in new_hits:
                position = np.asarray(event.position_world_m, dtype=np.float32)
                impact_markers.append((simulation_time_s, position))
                sampled_hit_positions.append(position)
                if event.previous_position_world_m is not None:
                    trail_history.record_segment(
                        int(event.batch_id) * 1_000_000 + int(event.particle_index),
                        event.previous_position_world_m,
                        event.position_world_m,
                        float(event.time_s),
                    )
            impact_markers = [(timestamp, position) for timestamp, position in impact_markers if simulation_time_s - timestamp <= 0.20]
            _write_instanced_markers(impacts, np.asarray([position for _, position in impact_markers], dtype=np.float32), visible=True)
            # Retain a bounded, deterministic diagnostic map made only of actual
            # hits.  Keep the display batch deliberately small: the complete
            # hit ledger remains in the runtime/result artifact, while updating
            # a growing 12k-instance USD PointInstancer every simulation frame
            # can exhaust RTX resources during a long native capture.
            if new_hits:
                hit_map_positions = np.asarray(sampled_hit_positions[-2500:], dtype=np.float32)
                _write_instanced_markers(hit_map, hit_map_positions, visible=True)
            trail_segments = _write_trails(trails, trail_history.lines(), visible=True)
            impact_seconds.append(time.perf_counter() - impact_start)
            active_particle_counts.append(int(len(particle_positions)))
            impact_marker_counts.append(int(len(impact_markers)))
            trail_segment_counts.append(int(trail_segments))

            warp_emission_time = simulation_time_s if new_batch_count > previous_batch_count else None
            hit_times = [event.time_s for event in new_hits]
            timeline.observe(
                simulated_time_s=simulation_time_s,
                s2_time_s=simulation_time_s,
                overlay_time_s=simulation_time_s,
                warp_emission_time_s=warp_emission_time,
                hit_time_s=hit_times,
            )
            if len(timeline_samples) < 64 and (new_hits or frame_index in {0, total_frames - 1}):
                timeline_samples.append({
                    "spray_timestamp_s": simulation_time_s,
                    "warp_emission_timestamp_s": warp_emission_time,
                    "warp_hit_timestamps_s": hit_times[:8],
                    "s2_deposition_timestamp_s": simulation_time_s,
                    "wft_overlay_timestamp_s": simulation_time_s,
                })

            _set_analysis_visibility(phase, deposition_view, flow, warp_points, trails, impacts, hit_map, overlay, guide)
            if phase in {"flow", "combined_closeup"}:
                viewport.camera_path = CLOSEUP_CAMERA_PATH
            else:
                viewport.camera_path = "/World/Cameras/process_painting"
            # Keep a bounded capture cadence for the native renderer.  The
            # technical combined view still contains actual particles/trails/
            # hits in the sampled frames; forcing an image capture on every
            # live-particle frame creates avoidable GPU pressure over 28 passes.
            if args.capture_media and (frame_index % max(int(args.capture_every), 1) == 0 or frame_index == total_frames - 1):
                capture_path = FRAME_DIR / f"rgb_{capture_index:06d}.png"
                _capture(viewport, capture_path, simulation_app)
                _annotate(
                    capture_path,
                    phase=phase,
                    deposition_view=deposition_view,
                    pass_index=int(pose.pass_index) if pose.pass_index is not None else 0,
                    pass_count=int(plan.pass_count),
                    spray_on=bool(pose.spray_on),
                    deposited_mass_kg=current_mass,
                    mean_wft_um=current_wft,
                    stand_off_m=model.stand_off_m,
                    incidence_deg=measured_frame.incidence_u_deg,
                    flow_report=direction_report.as_dict(),
                    active_particles=len(particle_positions),
                    hit_marker_count=len(impact_markers),
                )
                captured.append(capture_path)
                capture_index += 1
                if phase == "combined_closeup" and video_time >= 32.0 and not closeup_written and (len(particle_positions) > 0 or trail_segments > 0):
                    shutil.copy2(capture_path, MEDIA_DIR / "isaac_spray_analysis_closeup.png")
                    closeup_written = True
                if phase == "combined_closeup" and video_time >= 31.0 and not combined_written and (len(particle_positions) > 0 or trail_segments > 0 or len(impact_markers) > 0):
                    shutil.copy2(capture_path, MEDIA_DIR / "isaac_spray_analysis_combined.png")
                    combined_written = True
                if phase == "combined_wide" and video_time >= 55.0 and not progressive_written:
                    shutil.copy2(capture_path, MEDIA_DIR / "isaac_progressive_wft.png")
                    progressive_written = True
            else:
                simulation_app.update()
            analysis_seconds.append(time.perf_counter() - analysis_start)
            if frame_index % (FPS * 10) == 0:
                print(f"SPRAY_ANALYSIS_FRAME frame={frame_index} time_s={simulation_time_s:.3f} video_s={video_time:.2f} phase={phase} pass={pose.pass_index} spray={int(pose.spray_on)} mass_kg={current_mass:.9g} mean_wft_um={current_wft:.6g}", flush=True)

        warp_runtime.shutdown(plan.duration_s)
        final_hits = warp_runtime.drain_hit_events()
        for event in final_hits:
            sampled_hit_positions.append(np.asarray(event.position_world_m, dtype=np.float32))
        warp_ledger = warp_runtime.as_dict()
        hit_bounds = None
        if sampled_hit_positions:
            hit_array = np.asarray(sampled_hit_positions, dtype=np.float64)
            hit_bounds = {"min_world_m": hit_array.min(axis=0).tolist(), "max_world_m": hit_array.max(axis=0).tolist(), "sample_count": int(len(hit_array))}
        finite_ledger = accumulator.as_dict()
        wft = film_statistics(grid.cumulative_mass_kg, grid.area_weights_m2, liquid_density_kg_m3=liquid_density, volume_solids_fraction=0.50)
        native_wft = wft["wft_um"]
        wft_reverse_mass = native_wft["mean_um"] * liquid_density * 1.0e-6 * grid.total_area_m2
        gates = {
            "cfd_direction": direction_report.status == "CFD_DIRECTION_PASS",
            "roi_clipped": flow["vector_count"] <= 120 and flow["streamline_count"] <= 30,
            "guide_off": not (guide and guide.IsValid() and guide.GetAttribute("visibility").Get() == "inherited"),
            "s2_combined_mass_closure": abs(finite_ledger["combined_closure_error_kg"]) <= 1.0e-12,
            "warp_mass_closed": abs(float(warp_ledger["balance_error_kg"])) <= 1.0e-12,
            "wft_reverse_mass": abs(wft_reverse_mass - finite_ledger["surface_captured_kg"]) / max(finite_ledger["surface_captured_kg"], 1.0e-18) <= 1.0e-12,
            "same_simulated_timeline": bool(timeline.as_dict()["same_simulated_timeline"]),
            "spray_off_no_new_batches": spray_off_batch_violations == 0,
            "actual_hit_samples": int(warp_ledger.get("actual_hit_event_count", 0)) > 0,
            "media_closeup": closeup_written,
            "media_combined": combined_written,
            "media_progressive": progressive_written,
        }
        media = {
            "closeup": "media/air_assisted_spray/isaac_spray_analysis_closeup.png" if closeup_written else None,
            "combined": "media/air_assisted_spray/isaac_spray_analysis_combined.png" if combined_written else None,
            "progressive_wft": "media/air_assisted_spray/isaac_progressive_wft.png" if progressive_written else None,
        }
        if args.capture_media:
            video = _encode_video(captured, FRAME_DIR / "sequence", MEDIA_DIR / "isaac_spray_analysis_demo.mp4", TARGET_VIDEO_SECONDS)
        else:
            video = {"path": None, "duration_s": None, "codec": None}
        media["video"] = video.get("path")
        status = "SYNCHRONIZED_SPRAY_ANALYSIS_VALIDATED" if all(gates.values()) and video.get("path") else "PARTIAL_SPRAY_VISUALIZATION"
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        metrics = {
            "schema_version": "synchronized_spray_analysis_v1",
            "status": status,
            "authority": {
                "openfoam": "offline OpenFOAM v2606 reference flow",
                "warp": "GPU Lagrangian droplet transport and actual mesh-hit diagnostics",
                "s2": "authoritative deposited-mass and Estimated WFT model",
                "isaac_sim": "synchronized visualization and native process execution",
            },
            "cfd": {
                "artifact": "models/openfoam_flow_field_7p5deg_v1.npz",
                "manifest": "models/openfoam_flow_field_7p5deg_v1.json",
                "source_case": field.source_case,
                "latest_time_s": field.latest_time_s,
                "openfoam_version": field.openfoam_version,
                "source_u_sha256": field.source_u_sha256,
                "source_c_sha256": field.source_c_sha256,
                "artifact_sha256": field.artifact_sha256,
                "direction": direction_report.as_dict(),
                "roi_bounds_benchmark_m": {"min": list(roi_min), "max": list(roi_max)},
                "vectors": flow["vector_count"],
                "streamlines": flow["streamline_count"],
                "mapping": "+X→u fan-major, +Y→v travel/fan-minor, +Z→w inward process direction",
            },
            "warp": {
                "actual_particles_only_in_technical_view": True,
                "guide_enabled_in_technical_view": False,
                "trails": {"representative_particles": 200, "history_frames": 6, "max_segments": max(trail_segment_counts, default=0)},
                "impact_markers": {"source": "NVIDIA Warp actual mesh hits", "visual_only": True, "max_visible": max(impact_marker_counts, default=0), "sample_stride": int(warp_runtime.hit_event_stride)},
                "sampled_hit_bounds_world_m": hit_bounds,
                "ledger": warp_ledger,
            },
            "deposition": {
                "view": deposition_view,
                "s2_authority": True,
                "warp_hit_map": "diagnostic projected hit density; not an alternate thickness authority",
                "surface_area_m2": grid.total_area_m2,
                "pass_count": plan.pass_count,
                "final_mean_wft_um": native_wft["mean_um"],
                "final_p05_um": native_wft["p05_um"],
                "final_p95_um": native_wft["p95_um"],
                "final_cv": native_wft["cv"],
                "wft_formula": "deposited mass / (surface area × configured liquid density)",
                "wft_is_model_derived": True,
            },
            "synchronization": {
                **timeline.as_dict(),
                "spray_off_batch_violations": spray_off_batch_violations,
                "timeline_samples": timeline_samples,
                "fields": ["spray_timestamp_s", "warp_emission_timestamp_s", "warp_hit_timestamps_s", "s2_deposition_timestamp_s", "wft_overlay_timestamp_s"],
            },
            "performance": {
                "cfd_visualization_update_s": _summary(cfd_seconds),
                "warp_update_s": _summary(warp_seconds),
                "impact_visualization_update_s": _summary(impact_seconds),
                "wft_overlay_update_s": _summary(overlay_seconds),
                "combined_analysis_view_update_s": _summary(analysis_seconds),
                "peak_active_warp_particles": max(active_particle_counts, default=0),
                "impact_marker_count_peak": max(impact_marker_counts, default=0),
                "trail_segment_count_peak": max(trail_segment_counts, default=0),
                "wall_seconds": time.perf_counter() - started,
            },
            "mass_ledgers": {
                "s2_runtime": runtime.as_dict(),
                "s2_finite_surface": finite_ledger,
                "warp": warp_ledger,
                "visual_layers": visual_layer_mass_contract(),
                "gates": gates,
            },
            "media": media,
            "capture": {"resolution": list(RESOLUTION), "fps": FPS, "duration_s": video.get("duration_s"), "codec": video.get("codec"), "capture_every_sim_frames": int(args.capture_every)},
            "fidelity_boundary": "The OpenFOAM field is solved offline and visualized in the current local tangent process frame. Warp positions and mesh hits are diagnostic transport evidence. S2 remains authoritative for progressive deposited mass and Estimated WFT; WFT is model-derived, not measured or production-qualified.",
        }
        (RESULT_DIR / "spray_analysis_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"SPRAY_ANALYSIS_{'PASS' if status == 'SYNCHRONIZED_SPRAY_ANALYSIS_VALIDATED' else 'PARTIAL'} status={status} vectors={flow['vector_count']} streamlines={flow['streamline_count']} hits={warp_ledger.get('actual_hit_event_count', 0)} final_wft_um={native_wft['mean_um']:.6g}", flush=True)
        print(f"SPRAY_ANALYSIS_MEDIA {json.dumps(media, sort_keys=True)}", flush=True)
        return 0 if status == "SYNCHRONIZED_SPRAY_ANALYSIS_VALIDATED" else 2
    except Exception as exc:
        print(f"SPRAY_ANALYSIS_ERROR {type(exc).__name__}: {exc}", flush=True)
        return 2
    finally:
        if warp_runtime is not None:
            try:
                warp_runtime.shutdown()
            except Exception:
                pass
        simulation_app.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--capture-media", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None, help="diagnostic limit; omit for the complete 28-pass plan")
    parser.add_argument("--deposition-view", choices=("s2", "warp", "compare"), default="compare")
    parser.add_argument("--capture-every", type=int, default=6)
    parser.add_argument("--vector-count", type=int, default=80)
    parser.add_argument("--streamline-count", type=int, default=20)
    parser.add_argument("--warp-particles-per-bin", type=int, default=500)
    parser.add_argument("--warp-cadence-hz", type=float, default=15.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
