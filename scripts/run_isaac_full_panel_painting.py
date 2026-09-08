"""Native Isaac Sim full-panel painting and estimated WFT runtime.

The existing PaintingCycle scene/robot helper is reused, while the portable
full-panel plan supplies the serpentine path and the S2 finite-surface ledger.
The old 24 s W2 entry point is intentionally untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = ROOT.parent / "visual_prototypes"
SCENE_PATH = VISUAL_ROOT / "scenes" / "aircraft_painting_cell.usda"
PANEL_ASSET = ROOT.parent / "assets" / "course" / "generic_aircraft" / "generic_panel.usda"
MODEL_PATH = ROOT / "models" / "s2_air_assisted_v1.json"
PLAN_PATH = ROOT / "results" / "air_assisted" / "full_panel_film" / "full_panel_plan.json"
CONFIG_PATH = ROOT / "configs" / "full_panel_film_demo.yaml"
DENSITY_SOURCE_PATH = ROOT / "configs" / "air_assisted_spray.yaml"
WARP_MODEL_PATH = ROOT / "models" / "warp_vector_carrier_v3.json"
MEDIA_DIR = ROOT / "media" / "air_assisted_spray"
RESULT_DIR = ROOT / "results" / "air_assisted" / "full_panel_film"
FRAME_DIR = RESULT_DIR / "_frames"
FPS = 30
RESOLUTION = (1920, 1080)
ROOT_PRIM = "/World/AerospacePaintingCell"
RAIL_PRIM = "/World/AerospacePaintingCell/LinearTrack/Carriage"
ROBOT_PRIM = "/World/AerospacePaintingCell/LinearTrack/Carriage/PaintRobot/FANUC"
FLANGE_PRIM = "/World/AerospacePaintingCell/LinearTrack/Carriage/PaintRobot/FANUC/Asset/J6_link/flange"
PROCESS_TOOL_PRIM = FLANGE_PRIM + "/ProcessTool"
SPRAY_TOOL_PRIM = PROCESS_TOOL_PRIM + "/SprayGun"
PANEL_PRIM = "/World/Aircraft/Panel/Asset"
WFT_OVERLAY_PRIM = "/World/AerospacePaintingCell/FilmThicknessOverlay"
WARP_PLUME_PRIM = "/World/AerospacePaintingCell/WarpSprayPlume"
WARP_GUIDE_PRIM = "/World/AerospacePaintingCell/WarpSprayPlumeGuide"
SCENE_DISPLAY_PATH = "../visual_prototypes/scenes/aircraft_painting_cell.usda"
MODEL_DISPLAY_PATH = "models/s2_air_assisted_v1.json"
PLAN_DISPLAY_PATH = "results/air_assisted/full_panel_film/full_panel_plan.json"
WARP_MODEL_DISPLAY_PATH = "models/warp_vector_carrier_v3.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _density_from_config(path: Path) -> float:
    match = re.search(r"^\s*density_kg_m3:\s*([0-9]+(?:\.[0-9]+)?)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise ValueError(f"liquid density is missing from {path}")
    return float(match.group(1))


def _load_plan(model, surface):
    from aerospace_painting.full_panel_planner import FullPanelPlan

    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    process = payload["process"]
    recorded_sha = payload["surface"]["source_sha256"]
    if recorded_sha != surface.source_sha256:
        raise RuntimeError(f"full-panel plan geometry hash mismatch: {recorded_sha} != {surface.source_sha256}")
    model_ref = payload.get("s2_model", {})
    if model_ref.get("model_sha256") != model.model_sha256:
        raise RuntimeError("full-panel plan S2 model hash does not match the loaded model")
    return FullPanelPlan(
        surface=surface,
        fan_incidence_deg=float(process["fan_incidence_deg"]),
        stand_off_m=float(process["stand_off_m"]),
        mass_flow_kg_s=float(process["mass_flow_kg_s"]),
        overlap_fraction=float(process["overlap_fraction"]),
        requested_spacing_m=float(process["requested_spacing_m"]),
        spacing_m=float(process["spacing_m"]),
        speed_m_s=float(process["speed_m_s"]),
        overscan_m=float(process["overscan_m"]),
        passes=list(payload["passes"]),
        segments=list(payload["segments"]),
        duration_s=float(process["cycle_duration_s"]),
    )


def _vec3(matrix, vector):
    import numpy as np
    from pxr import Gf

    value = matrix.TransformDir(Gf.Vec3d(*vector))
    array = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(array))
    return array / norm


def _translation(matrix):
    import numpy as np

    return np.asarray(matrix.ExtractTranslation(), dtype=float)


def _create_overlay(stage, grid):
    from pxr import Sdf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, WFT_OVERLAY_PRIM)
    # A slight outward offset keeps the engineering overlay readable without
    # altering the geometry used by the mass ledger.
    points = grid.positions + grid.normals * 0.0015
    mesh.CreatePointsAttr(points.tolist())
    faces = grid.triangle_faces()
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    colors = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    opacities = mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.vertex)
    colors.Set([(0.04, 0.10, 0.18)] * len(grid.positions))
    opacities.Set([0.0] * len(grid.positions))
    prim = mesh.GetPrim()
    prim.CreateAttribute("film:metric", Sdf.ValueTypeNames.String).Set("estimated_wet_film_thickness_um")
    prim.CreateAttribute("film:primary", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("film:measured", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("film:source", Sdf.ValueTypeNames.String).Set("S2 finite-surface deposited mass / configured liquid density")
    return mesh


def _write_overlay(mesh, grid, liquid_density_kg_m3: float):
    import numpy as np

    wft_um = np.divide(
        grid.cumulative_mass_kg,
        grid.area_weights_m2,
        out=np.zeros_like(grid.cumulative_mass_kg),
        where=grid.area_weights_m2 > 0.0,
    ) / float(liquid_density_kg_m3) * 1.0e6
    maximum = float(np.percentile(wft_um[wft_um > 0.0], 98.0)) if np.any(wft_um > 0.0) else 1.0
    normalized = np.clip(wft_um / max(maximum, 1.0e-9), 0.0, 1.0)
    colors = [(float(0.05 + 0.90 * q), float(0.18 + 0.52 * (1.0 - q)), float(0.68 * (1.0 - q))) for q in normalized]
    opacities = [float(0.18 + 0.72 * q) if value > 0.0 else 0.0 for q, value in zip(normalized, wft_um)]
    mesh.GetDisplayColorPrimvar().Set(colors)
    mesh.GetDisplayOpacityPrimvar().Set(opacities)


def _create_warp_points(stage):
    from pxr import Sdf, UsdGeom

    points = UsdGeom.Points.Define(stage, WARP_PLUME_PRIM)
    points.CreatePointsAttr([])
    points.CreateWidthsAttr([])
    points.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(0.96, 0.72, 0.22)])
    points.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([0.86])
    points.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    prim = points.GetPrim()
    prim.CreateAttribute("warp:source", Sdf.ValueTypeNames.String).Set("W1.3 fixed-grid vector carrier")
    prim.CreateAttribute("warp:visualScale_m", Sdf.ValueTypeNames.Float).Set(0.003)
    prim.CreateAttribute("warp:positionsOnly", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("warp:frameFrozenPerBatch", Sdf.ValueTypeNames.Bool).Set(True)
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return points


def _write_warp_points(points_prim, positions):
    from pxr import UsdGeom

    points_prim.GetPointsAttr().Set(positions.tolist() if len(positions) else [])
    points_prim.GetWidthsAttr().Set([0.003] * len(positions))
    UsdGeom.Imageable(points_prim.GetPrim()).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if len(positions) else UsdGeom.Tokens.invisible)


def _create_plume_guide(stage):
    """Create a restrained visual guide for the live Warp transport layer.

    The guide is deliberately separate from ``WarpSprayPlume``: the latter is
    the actual W1.3 particle point cloud, while this translucent envelope keeps
    the sub-pixel CUDA points legible in a 1920x1080 review capture.  It adds no
    mass, force, or deposition state.
    """

    from pxr import Sdf, UsdGeom

    # Use deliberately oversized diagnostic points rather than a surface mesh:
    # this remains legible in a 1920x1080 review capture even when the live
    # W1.3 points are sub-pixel.  It is not part of the physical transport.
    points = UsdGeom.Points.Define(stage, WARP_GUIDE_PRIM)
    points.CreatePointsAttr([])
    points.CreateWidthsAttr([])
    colors = points.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    opacities = points.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant)
    colors.Set([(1.0, 0.32, 0.04)])
    opacities.Set([0.0])
    prim = points.GetPrim()
    prim.CreateAttribute("warp:diagnosticVisualGuide", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("warp:addsMass", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("warp:source", Sdf.ValueTypeNames.String).Set("W1.3 live point-cloud readability guide")
    prim.CreateAttribute("warp:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return points


def _write_plume_guide(mesh, *, origin, frame, stand_off_m: float, spray_on: bool):
    import numpy as np
    from pxr import UsdGeom

    if not spray_on:
        mesh.GetDisplayOpacityPrimvar().Set([0.0])
        UsdGeom.Imageable(mesh.GetPrim()).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        return
    origin = np.asarray(origin, dtype=float)
    u = np.asarray(frame.u_world, dtype=float)
    v = np.asarray(frame.v_world, dtype=float)
    w = np.asarray(frame.w_world, dtype=float)
    # Keep the readability guide entirely in the air gap so the translucent
    # fan is not hidden by the panel/WFT overlay.  It remains diagnostic-only:
    # the actual W1.3 point cloud and ledgers are unchanged.
    span = min(0.18, float(stand_off_m) * 0.75)
    guide_points = []
    for axial in np.linspace(0.02, span, 12):
        fraction = (axial - 0.02) / max(span - 0.02, 1.0e-9)
        radius = 0.018 + 0.067 * fraction
        for index in range(12):
            angle = 2.0 * math.pi * index / 12.0
            guide_points.append(origin + axial * w + radius * (math.cos(angle) * u + math.sin(angle) * v))
    mesh.GetPointsAttr().Set(guide_points)
    mesh.GetWidthsAttr().Set([0.012] * len(guide_points))
    mesh.GetDisplayOpacityPrimvar().Set([0.24])
    UsdGeom.Imageable(mesh.GetPrim()).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)


def _render_capture_setup(frame_dir: Path):
    from pxr import Sdf
    from omni.kit.viewport.utility import create_viewport_window, get_active_viewport

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    window = create_viewport_window(name="FullPanelFilmViewport", width=RESOLUTION[0], height=RESOLUTION[1], camera_path=Sdf.Path("/World/Cameras/process_painting"))
    viewport = window.viewport_api if window else get_active_viewport()
    if viewport is None:
        raise RuntimeError("native viewport is unavailable for full-panel evidence capture")
    viewport.camera_path = Sdf.Path("/World/Cameras/process_painting")
    viewport.resolution = RESOLUTION
    return viewport


def _encode_media(captured: list[Path], frame_dir: Path) -> dict[str, str | None]:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    if not captured:
        return {"hero": None, "final": None, "video": None}
    ordered = sorted(captured)
    hero = MEDIA_DIR / "isaac_full_panel_film_hero.png"
    final = MEDIA_DIR / "isaac_full_panel_film_final.png"
    shutil.copy2(ordered[min(len(ordered) - 1, int(len(ordered) * 0.10))], hero)
    shutil.copy2(ordered[-1], final)
    sequence_dir = frame_dir / "sequence"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(ordered):
        shutil.copy2(source, sequence_dir / f"frame_{index:06d}.png")
    video = MEDIA_DIR / "isaac_full_panel_film_demo.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"hero": hero.relative_to(ROOT).as_posix(), "final": final.relative_to(ROOT).as_posix(), "video": None}
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", str(sequence_dir / "frame_%06d.png"),
        "-vf", "tpad=stop_mode=clone:stop_duration=8,format=yuv420p",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"MEDIA_FFMPEG_ERROR {result.stderr.strip()}", flush=True)
        video_path = None
    else:
        video_path = video.relative_to(ROOT).as_posix()
    return {"hero": hero.relative_to(ROOT).as_posix(), "final": final.relative_to(ROOT).as_posix(), "video": video_path}


class FullPanelPaintingCycle:
    """Reuse the validated PaintingCycle arm/scene helper with the saved plan."""

    def __init__(self, stage, root_path: Path, plan):
        from aerospace_process_motion import PaintingCycle
        from pxr import UsdGeom

        self.base = PaintingCycle(stage, str(root_path), stand_off_m=plan.stand_off_m)
        self.stage = stage
        self.plan = plan
        self.duration = plan.duration_s
        # The base helper's 61x19 geometric-coverage mesh is not the film
        # metric; leave it inactive so the single dense overlay is authoritative.
        if hasattr(self.base, "coverage_mesh"):
            self.base.coverage_mesh.GetPrim().SetActive(False)
        self.base.cone.Set(UsdGeom.Tokens.invisible)

    def update(self, stage, index: int, total: int):
        import numpy as np
        from pxr import Gf, Usd, UsdGeom
        from process_motion import translation

        time_s = self.duration * index / max(total - 1, 1)
        pose = self.plan.pose_at(time_s)
        self.base.carriage.Set(translation((0.0, pose.y + 0.3, 0.0)))
        direction = pose.frame.w_inward * math.cos(math.radians(self.plan.fan_incidence_deg)) + pose.frame.u_fan_major * math.sin(math.radians(self.plan.fan_incidence_deg))
        direction = direction / np.linalg.norm(direction)
        tcp, error = self.base.arm.move_tcp(pose.surface_position - direction * self.plan.stand_off_m, direction, pose.frame.v_fan_minor)
        nozzle = np.asarray(tcp.ExtractTranslation(), dtype=float)
        axis = np.asarray(tcp.TransformDir(Gf.Vec3d(1, 0, 0)), dtype=float)
        axis /= np.linalg.norm(axis)
        self.base.current_surface_point = np.asarray(pose.surface_position, dtype=float)
        self.base.current_surface_normal = np.asarray(pose.normal, dtype=float)
        self.base.current_tcp = Gf.Matrix4d(tcp)
        self.base.current_spray_axis = axis
        self.base.current_spray_on = bool(pose.spray_on)
        self.base.current_incidence_deg = float(self.plan.fan_incidence_deg)
        self.base.cone.Set(UsdGeom.Tokens.inherited if pose.spray_on else UsdGeom.Tokens.invisible, Usd.TimeCode.Default())
        event = "SPRAY_PASS" if pose.spray_on else "STEP_OVER"
        if self.base.last_event != event:
            print(f"FULL_PANEL_EVENT time_s={time_s:.3f} event={event} pass={pose.pass_index}", flush=True)
            self.base.last_event = event
        self.base.records.append({
            "frame": index,
            "time": time_s,
            "event": event,
            "pass_index": pose.pass_index,
            "spray_on": bool(pose.spray_on),
            "tcp_error_m": float(error),
            "stand_off_error_m": float(abs(np.linalg.norm(nozzle - pose.surface_position) - self.plan.stand_off_m)),
            "incidence_commanded_deg": float(self.plan.fan_incidence_deg),
            "joint_degrees": list(map(float, self.base.arm.q)),
        })
        return pose


def run(args: argparse.Namespace) -> int:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(launch_config={"headless": bool(args.headless), "width": RESOLUTION[0], "height": RESOLUTION[1]})
    started = time.perf_counter()
    try:
        import numpy as np
        import omni.usd
        from pxr import UsdGeom

        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(VISUAL_ROOT))
        from aerospace_painting.full_panel_planner import PanelSurface, film_statistics
        from aerospace_painting.s2_runtime import FiniteSurfaceAccumulator, S2Runtime, S2RuntimeModel, StructuredSurfaceGrid, surface_frame
        from aerospace_painting.warp_plume_runtime import WarpBatchFrame, WarpPlumeRuntime

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
        plume_guide = _create_plume_guide(stage)
        runtime = S2Runtime(model)
        accumulator = FiniteSurfaceAccumulator(grid)
        warp_runtime = None
        warp_points = None
        if not args.no_warp:
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
            warp_points = _create_warp_points(stage)
        capture = None
        if not args.no_media:
            capture = _render_capture_setup(FRAME_DIR)
        total_frames = int(round(plan.duration_s * FPS)) + 1
        dt = plan.duration_s / max(total_frames - 1, 1)
        performance_start = time.perf_counter()
        s2_seconds = []
        warp_seconds = []
        measured_incidence = []
        measured_minor = []
        stand_offs = []
        tcp_errors = []
        violation_count = 0
        captured_paths: list[Path] = []
        capture_index = 0
        for frame in range(total_frames):
            time_s = plan.duration_s * frame / max(total_frames - 1, 1)
            pose = cycle.update(stage, frame, total_frames)
            actual_axis = np.asarray(cycle.base.current_spray_axis, dtype=float)
            try:
                measured_frame = surface_frame(pose.normal, pose.frame.u_fan_major, actual_axis, minor_tolerance_deg=3.0)
            except ValueError:
                violation_count += 1
                measured_frame = surface_frame(pose.normal, pose.frame.u_fan_major, actual_axis, minor_tolerance_deg=89.0)
            measured_incidence.append(float(measured_frame.incidence_u_deg))
            measured_minor.append(float(measured_frame.incidence_v_deg))
            nozzle = _translation(cycle.base.current_tcp)
            stand_offs.append(float(np.linalg.norm(nozzle - cycle.base.current_surface_point)))
            tcp_errors.append(float(cycle.base.records[-1]["tcp_error_m"]))
            s2_start = time.perf_counter()
            step = runtime.step(
                dt_s=dt,
                incidence_angle_deg=plan.fan_incidence_deg,
                stand_off_m=model.stand_off_m,
                air_velocity_m_s=model.air_velocity_m_s,
                mass_flow_kg_s=model.mass_flow_kg_s,
                spray_on=bool(pose.spray_on),
            )
            s2_seconds.append(time.perf_counter() - s2_start)
            if pose.spray_on:
                accumulator.apply(step, target_position=cycle.base.current_surface_point, frame=pose.frame)
            _write_overlay(overlay, grid, liquid_density)
            if warp_runtime is not None:
                warp_start = time.perf_counter()
                tool_x = _vec3(cycle.base.current_tcp, (1.0, 0.0, 0.0))
                tool_y = _vec3(cycle.base.current_tcp, (0.0, 1.0, 0.0))
                tool_z = _vec3(cycle.base.current_tcp, (0.0, 0.0, 1.0))
                warp_frame = WarpBatchFrame.from_surface_frame(
                    origin_world_m=nozzle,
                    target_world_m=cycle.base.current_surface_point,
                    surface_frame=measured_frame,
                    incidence_deg=measured_frame.incidence_u_deg,
                    stand_off_m=model.stand_off_m,
                )
                warp_runtime.update(time_s, bool(pose.spray_on), warp_frame, actual_tool_origin_m=nozzle, actual_tool_axes=(tool_x, tool_y, tool_z))
                _write_warp_points(warp_points, warp_runtime.active_world_positions())
                warp_seconds.append(time.perf_counter() - warp_start)
                _write_plume_guide(plume_guide, origin=nozzle, frame=warp_frame, stand_off_m=model.stand_off_m, spray_on=bool(pose.spray_on))
            else:
                _write_plume_guide(plume_guide, origin=nozzle, frame=WarpBatchFrame.from_surface_frame(origin_world_m=nozzle, target_world_m=cycle.base.current_surface_point, surface_frame=pose.frame, incidence_deg=plan.fan_incidence_deg, stand_off_m=model.stand_off_m), stand_off_m=model.stand_off_m, spray_on=bool(pose.spray_on))
            if capture is not None and (frame < 10 * FPS or frame % 3 == 0 or frame == total_frames - 1):
                from omni.kit.viewport.utility import capture_viewport_to_file

                capture_path = FRAME_DIR / f"rgb_{capture_index:06d}.png"
                capture_viewport_to_file(capture, file_path=str(capture_path))
                capture_index += 1
                for _ in range(20):
                    simulation_app.update()
                    if capture_path.exists() and capture_path.stat().st_size > 0:
                        break
                captured_paths.append(capture_path)
            else:
                simulation_app.update()
            if frame % (FPS * 5) == 0:
                print(f"FULL_PANEL_FRAME frame={frame} time_s={time_s:.3f} pass={pose.pass_index} spray={int(pose.spray_on)}", flush=True)
        if warp_runtime is not None:
            warp_runtime.shutdown(plan.duration_s)
            _write_warp_points(warp_points, warp_runtime.active_world_positions())
        performance_elapsed = time.perf_counter() - performance_start
        if capture is not None:
            media = _encode_media(captured_paths, FRAME_DIR)
        else:
            media = {"hero": None, "final": None, "video": None}
        runtime_ledger = runtime.as_dict()
        finite_ledger = accumulator.as_dict()
        wft = film_statistics(grid.cumulative_mass_kg, grid.area_weights_m2, liquid_density_kg_m3=liquid_density, volume_solids_fraction=0.50)
        planner_summary = json.loads(PLAN_PATH.read_text(encoding="utf-8"))["planner_summary"]
        planned_metrics = planner_summary["selected_metrics"]
        native_wft = wft["wft_um"]
        planner_compare = {
            "surface_captured_mass_relative_error": abs(finite_ledger["surface_captured_kg"] - planned_metrics["finite_surface"]["surface_captured_kg"]) / max(planned_metrics["finite_surface"]["surface_captured_kg"], 1.0e-18),
            "mean_wft_relative_error": abs(native_wft["mean_um"] - planned_metrics["wft"]["mean_um"]) / max(planned_metrics["wft"]["mean_um"], 1.0e-18),
            "cv_absolute_error": abs(native_wft["cv"] - planned_metrics["wft"]["cv"]),
        }
        planner_runtime_drift = planner_compare["surface_captured_mass_relative_error"] > 0.10 or planner_compare["mean_wft_relative_error"] > 0.10 or planner_compare["cv_absolute_error"] > 0.05
        wft_reverse_mass = native_wft["mean_um"] * liquid_density * 1.0e-6 * grid.total_area_m2
        gates = {
            "s2_injected_equals_plane_plus_process_overspray": abs(runtime_ledger["injected_kg"] - runtime_ledger["deposited_kg"] - runtime_ledger["overspray_kg"]) <= 1.0e-12,
            "plane_equals_surface_plus_edge_loss": abs(finite_ledger["plane_deposited_kg"] - finite_ledger["surface_captured_kg"] - finite_ledger["geometric_edge_loss_kg"]) <= 1.0e-12,
            "combined_mass_closure": abs(finite_ledger["combined_closure_error_kg"]) <= 1.0e-12,
            "wft_reverse_mass": abs(wft_reverse_mass - finite_ledger["surface_captured_kg"]) / max(finite_ledger["surface_captured_kg"], 1.0e-18) <= 1.0e-12,
            "planner_runtime_agreement": not planner_runtime_drift,
            "minor_coherence": violation_count == 0,
        }
        status = "FULL_PANEL_FILM_ESTIMATION_VALIDATED" if all(gates.values()) else ("BLOCKED_PLANNER_RUNTIME_DRIFT" if planner_runtime_drift else "BLOCKED_NATIVE_MOTION" if violation_count else "BLOCKED_MASS_ACCOUNTING")
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            RESULT_DIR / "full_panel_surface_data.npz",
            positions_m=grid.positions,
            normals=grid.normals,
            area_weights_m2=grid.area_weights_m2,
            cumulative_mass_kg=grid.cumulative_mass_kg,
            wft_um=np.divide(grid.cumulative_mass_kg, grid.area_weights_m2) / liquid_density * 1.0e6,
        )
        def _summary(values):
            return {"min": float(np.min(values)) if values else None, "max": float(np.max(values)) if values else None, "mean": float(np.mean(values)) if values else None, "p95": float(np.percentile(values, 95.0)) if values else None}
        warp_ledger = warp_runtime.as_dict() if warp_runtime is not None else None
        if warp_ledger is not None:
            warp_ledger["visual_guide"] = {"prim": WARP_GUIDE_PRIM, "adds_mass": False, "purpose": "legibility only for sub-pixel live Warp points"}
        metrics = {
            "status": status,
            "schema_version": "full_panel_film_metrics_v1",
            "isaac_sim": {"launcher": "native Isaac Sim python.bat", "scene": SCENE_DISPLAY_PATH},
            "scene": {"root": ROOT_PRIM, "rail": RAIL_PRIM, "robot": ROBOT_PRIM, "flange": FLANGE_PRIM, "process_tool": PROCESS_TOOL_PRIM, "spray_tool": SPRAY_TOOL_PRIM, "workpiece": PANEL_PRIM, "film_overlay": WFT_OVERLAY_PRIM, "warp_plume": WARP_PLUME_PRIM, "warp_visual_guide": WARP_GUIDE_PRIM},
            "plan": {"path": PLAN_DISPLAY_PATH, "fan_incidence_deg": plan.fan_incidence_deg, "stand_off_m": plan.stand_off_m, "overlap_fraction": plan.overlap_fraction, "spacing_m": plan.spacing_m, "requested_spacing_m": plan.requested_spacing_m, "speed_m_s": plan.speed_m_s, "overscan_m": plan.overscan_m, "pass_count": plan.pass_count, "path_length_m": plan.path_length_m, "spray_time_s": plan.spray_time_s, "cycle_duration_s": plan.duration_s},
            "surface": {"shape": list(surface.shape), "grid_vertices": int(len(grid.positions)), "area_m2": grid.total_area_m2, "source_path": surface.source_path, "source_sha256": surface.source_sha256},
            "model": {"path": MODEL_DISPLAY_PATH, "model_id": model.model_id, "model_sha256": model.model_sha256, "openfoam_version": model.openfoam_version, "stand_off_m": model.stand_off_m, "mass_flow_kg_s": model.mass_flow_kg_s},
            "process_contract": {"accumulation_mode": "FINITE_SURFACE", "normalised_footprint_regression_preserved": True, "s2_authority": "7.5 degree interpolated S2 Gaussian moments", "warp_layer": "W1.3 fixed-grid full-vector diagnostic", "wft_primary": True},
            "mass_ledger": {"s2_runtime": runtime_ledger, "finite_surface": finite_ledger, "gates": gates},
            "estimated_wft": {"source": "cumulative surface deposited mass / actual surface area / configured liquid density", "liquid_density_kg_m3": liquid_density, "density_source_path": "configs/air_assisted_spray.yaml", "density_source_sha256": _sha256(DENSITY_SOURCE_PATH), "stats": native_wft, "reverse_mass_kg": wft_reverse_mass, "label": "Estimated WFT — derived from deposited mass and configured liquid density; not measured"},
            "illustrative_dft": {"enabled": True, "volume_solids_fraction": 0.50, "stats": wft["dft_um"], "label": "Illustrative DFT estimate — assumed 50% volume solids; synthetic_demo_only"},
            "planner_vs_native": planner_compare,
            "warp_visual_layer": warp_ledger,
            "motion": {"measured_incidence_deg": _summary(measured_incidence), "measured_minor_incidence_deg": _summary(measured_minor), "stand_off_m": _summary(stand_offs), "tcp_error_m": _summary(tcp_errors), "minor_coherence_violations": violation_count},
            "performance": {"simulated_duration_s": plan.duration_s, "frame_count": total_frames, "wall_seconds": performance_elapsed, "runtime_fps": total_frames / performance_elapsed if performance_elapsed else None, "real_time_ratio": plan.duration_s / performance_elapsed if performance_elapsed else None, "s2_update": _summary(s2_seconds), "warp_update": _summary(warp_seconds), "warp_peak_active_particles": warp_ledger.get("max_active_particles") if warp_ledger else None, "warp_parcel_histories": warp_ledger.get("closed_batch_count") if warp_ledger else 0, "grid_vertices": int(len(grid.positions))},
            "media": media,
            "capture": {"resolution": list(RESOLUTION), "source_frame_count": len(captured_paths), "source_framerate_hz": FPS, "time_compressed": True, "final_overlay_hold_s": 8.0},
            "fidelity_boundary": "The full-panel demo predicts surface deposited mass and estimated wet-film thickness from the CFD-calibrated S2 deposition model while visualizing GPU Lagrangian droplet transport with NVIDIA Warp. WFT is derived from deposited mass and configured liquid density, not measured; DFT is an illustrative assumption, not validated paint data. The moving Warp layer remains a quasi-steady local tangent-patch visualization.",
            "generated_at_epoch_s": time.time(),
        }
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "full_panel_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"FULL_PANEL_{'PASS' if status == 'FULL_PANEL_FILM_ESTIMATION_VALIDATED' else 'BLOCKED'} status={status} frames={total_frames} mass={finite_ledger['surface_captured_kg']:.9g} mean_wft_um={native_wft['mean_um']:.6g}", flush=True)
        print(f"MEDIA hero={media['hero']} final={media['final']} video={media['video']}", flush=True)
        return 0 if status == "FULL_PANEL_FILM_ESTIMATION_VALIDATED" else 2
    except Exception as exc:
        print(f"FULL_PANEL_NATIVE_ERROR {type(exc).__name__}: {exc}", flush=True)
        return 2
    finally:
        simulation_app.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-media", action="store_true")
    parser.add_argument("--no-warp", action="store_true")
    parser.add_argument("--warp-particles-per-bin", type=int, default=500)
    parser.add_argument("--warp-cadence-hz", type=float, default=15.0)
    parser.add_argument("--max-seconds", type=float, default=None, help="diagnostic capture limit; do not use for the canonical full-panel run")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
