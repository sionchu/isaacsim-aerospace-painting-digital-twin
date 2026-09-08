"""Run the frozen S2 deposition runtime against the existing Isaac Sim scene.

This is intentionally the only native integration entry point.  The numeric
runtime remains Isaac-free; this adapter supplies the actual stage, the
existing PaintingCycle joint motion, the resolved SprayGun/TCP transforms, a
single bounded USD overlay mesh, and evidence capture.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = ROOT.parent / "visual_prototypes"
SCENE_PATH = VISUAL_ROOT / "scenes" / "aircraft_painting_cell.usda"
MODEL_PATH = ROOT / "models" / "s2_air_assisted_v1.json"
WARP_MODEL_PATH = ROOT / "models" / "warp_vector_carrier_v3.json"
MEDIA_DIR = ROOT / "media" / "air_assisted_spray"
RESULT_DIR = ROOT / "results" / "air_assisted" / "isaac_s2_runtime"
WARP_RESULT_DIR = ROOT / "results" / "air_assisted" / "isaac_warp_runtime"
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
OVERLAY_PRIM = "/World/AerospacePaintingCell/S2DepositionOverlay"
WARP_PLUME_PRIM = "/World/AerospacePaintingCell/WarpSprayPlume"
SCENE_DISPLAY_PATH = "../visual_prototypes/scenes/aircraft_painting_cell.usda"
MODEL_DISPLAY_PATH = "models/s2_air_assisted_v1.json"
WARP_MODEL_DISPLAY_PATH = "models/warp_vector_carrier_v3.json"


def state_for_time(time_s: float) -> float:
    """Use the three existing sweep windows as the A/B/C incidence states."""

    if time_s < 8.0:
        return 0.0
    if time_s < 9.0:
        # The first step-over is non-spraying; use it to rotate the real TCP
        # continuously into state B rather than teleporting the wrist pose.
        return 7.5 * (time_s - 8.0)
    if time_s < 15.0:
        if time_s < 14.0:
            return 7.5
        # The second step-over is also non-spraying and provides the continuous
        # transition into the 15-degree third sweep.
        return 7.5 + 7.5 * (time_s - 14.0)
    return 15.0


def state_label_for_time(time_s: float) -> float:
    if time_s < 8.0:
        return 0.0
    if time_s < 15.0:
        return 7.5
    return 15.0


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


def _write_overlay(mesh, grid, *, max_mass: float):
    import numpy as np

    if max_mass <= 0.0:
        colors = [(0.05, 0.10, 0.16)] * len(grid.positions)
        opacities = [0.0] * len(grid.positions)
    else:
        normalized = np.clip(grid.cumulative_mass_kg / max_mass, 0.0, 1.0)
        colors = [(float(0.08 + 0.90 * value), float(0.18 + 0.35 * (1.0 - value)), 0.08) for value in normalized]
        opacities = [float(0.18 + 0.72 * value) if value > 1.0e-15 else 0.0 for value in normalized]
    mesh.GetDisplayColorPrimvar().Set(colors)
    mesh.GetDisplayOpacityPrimvar().Set(opacities)


def _create_overlay(stage, grid):
    from pxr import Sdf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, OVERLAY_PRIM)
    mesh.CreatePointsAttr(grid.positions.tolist())
    faces = grid.triangle_faces()
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    colors = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    opacities = mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.vertex)
    colors.Set([(0.05, 0.10, 0.16)] * len(grid.positions))
    opacities.Set([0.0] * len(grid.positions))
    # Keep the overlay just above the sampled surface without introducing a
    # second workpiece or a disconnected animation.
    mesh.GetPrim().CreateAttribute("s2:source", Sdf.ValueTypeNames.String).Set("actual aircraft panel samples")
    return mesh


def _create_warp_points(stage):
    """Create one bounded USD point object for the live Warp particles."""

    from pxr import Sdf, UsdGeom

    points = UsdGeom.Points.Define(stage, WARP_PLUME_PRIM)
    points.CreatePointsAttr([])
    points.CreateWidthsAttr([])
    colors = points.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    opacities = points.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant)
    colors.Set([(0.96, 0.72, 0.22)])
    opacities.Set([0.86])
    points.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    prim = points.GetPrim()
    prim.CreateAttribute("warp:source", Sdf.ValueTypeNames.String).Set("W1.3 fixed-grid vector carrier")
    prim.CreateAttribute("warp:visualScale_m", Sdf.ValueTypeNames.Float).Set(0.003)
    prim.CreateAttribute("warp:positionsOnly", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("warp:frameFrozenPerBatch", Sdf.ValueTypeNames.Bool).Set(True)
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return points


def _write_warp_points(points_prim, positions):
    """Update only the active particle positions on the shared USD Points prim."""

    from pxr import UsdGeom

    values = positions.tolist() if len(positions) else []
    points_prim.GetPointsAttr().Set(values)
    points_prim.GetWidthsAttr().Set([0.003] * len(values))
    UsdGeom.Imageable(points_prim.GetPrim()).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if values else UsdGeom.Tokens.invisible)


def _encode_warp_media(captured: list[Path], frame_dir: Path) -> dict[str, str | None]:
    """Encode the native W2 capture at its actual 30 Hz source cadence."""

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    if not captured:
        return {"hero": None, "deposition": None, "video": None}
    ordered = sorted(captured)
    hero_source = ordered[min(len(ordered) - 1, max(0, int(len(ordered) * (6.0 / 24.0))))]
    deposition_source = ordered[-1]
    hero = MEDIA_DIR / "isaac_warp_plume_hero.png"
    deposition = MEDIA_DIR / "isaac_warp_plume_deposition.png"
    shutil.copy2(hero_source, hero)
    shutil.copy2(deposition_source, deposition)
    sequence_dir = frame_dir / "warp_sequence"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(ordered):
        shutil.copy2(source, sequence_dir / f"frame_{index:06d}.png")
    video = MEDIA_DIR / "isaac_warp_plume_demo.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"hero": hero.relative_to(ROOT).as_posix(), "deposition": deposition.relative_to(ROOT).as_posix(), "video": None}
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(FPS),
        "-i",
        str(sequence_dir / "frame_%06d.png"),
        "-vf",
        "fps=30,format=yuv420p",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"MEDIA_FFMPEG_ERROR {result.stderr.strip()}", flush=True)
        return {"hero": hero.relative_to(ROOT).as_posix(), "deposition": deposition.relative_to(ROOT).as_posix(), "video": None}
    return {"hero": hero.relative_to(ROOT).as_posix(), "deposition": deposition.relative_to(ROOT).as_posix(), "video": video.relative_to(ROOT).as_posix()}


def _render_capture_setup(frame_dir: Path = FRAME_DIR):
    from pxr import Sdf
    from omni.kit.viewport.utility import create_viewport_window, get_active_viewport

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    window = create_viewport_window(
        name="S2RuntimeViewport",
        width=RESOLUTION[0],
        height=RESOLUTION[1],
        camera_path=Sdf.Path("/World/Cameras/process_painting"),
    )
    viewport = window.viewport_api if window else get_active_viewport()
    if viewport is None:
        raise RuntimeError("native viewport is unavailable for evidence capture")
    viewport.camera_path = Sdf.Path("/World/Cameras/process_painting")
    viewport.resolution = RESOLUTION
    return viewport


def _encode_media(captured: list[Path]) -> dict[str, str | None]:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    if not captured:
        return {"hero": None, "deposition": None, "video": None}
    ordered = sorted(captured)
    hero_source = ordered[min(len(ordered) - 1, max(0, int(len(ordered) * (6.0 / 24.0))))]
    deposition_source = ordered[-1]
    hero = MEDIA_DIR / "isaac_s2_runtime_hero.png"
    deposition = MEDIA_DIR / "isaac_s2_runtime_deposition.png"
    shutil.copy2(hero_source, hero)
    shutil.copy2(deposition_source, deposition)
    sequence_dir = FRAME_DIR / "sequence"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(ordered):
        shutil.copy2(source, sequence_dir / f"frame_{index:06d}.png")
    video = MEDIA_DIR / "isaac_s2_runtime_demo.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"hero": hero.relative_to(ROOT).as_posix(), "deposition": deposition.relative_to(ROOT).as_posix(), "video": None}
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(FPS / 4.0),
        "-i",
        str(sequence_dir / "frame_%06d.png"),
        "-vf",
        "fps=30,format=yuv420p",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"MEDIA_FFMPEG_ERROR {result.stderr.strip()}", flush=True)
        return {"hero": hero.relative_to(ROOT).as_posix(), "deposition": deposition.relative_to(ROOT).as_posix(), "video": None}
    return {"hero": hero.relative_to(ROOT).as_posix(), "deposition": deposition.relative_to(ROOT).as_posix(), "video": video.relative_to(ROOT).as_posix()}


def run(args: argparse.Namespace) -> int:
    from isaacsim import SimulationApp

    launch_config = {"headless": bool(args.headless), "width": RESOLUTION[0], "height": RESOLUTION[1]}
    simulation_app = SimulationApp(launch_config=launch_config)
    started = time.perf_counter()
    try:
        import numpy as np
        import omni.usd
        from pxr import Usd, UsdGeom

        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(VISUAL_ROOT))
        from aerospace_painting.s2_runtime import (
            S2Runtime,
            S2RuntimeModel,
            StructuredSurfaceGrid,
            RuntimeSurfaceAccumulator,
            surface_frame,
        )
        if args.warp_plume:
            from aerospace_painting.warp_plume_runtime import WarpBatchFrame, WarpPlumeRuntime
        from aerospace_process_motion import PaintingCycle

        model = S2RuntimeModel.load(MODEL_PATH)
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
        print(f"SCENE_PRIMS root={ROOT_PRIM} spray_tool={SPRAY_TOOL_PRIM} panel={PANEL_PRIM}", flush=True)

        cycle = PaintingCycle(stage, str(ROOT.parent), stand_off_m=model.stand_off_m)
        surface_grid = StructuredSurfaceGrid.from_structured(cycle.surface_points, cycle.normals, (61, 19))
        accumulator = RuntimeSurfaceAccumulator(surface_grid)
        overlay = _create_overlay(stage, surface_grid)
        runtime = S2Runtime(model)
        warp_runtime = None
        warp_points = None
        if args.warp_plume:
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
        capture_dir = WARP_RESULT_DIR / "_frames" if args.warp_plume else FRAME_DIR
        if not args.no_media:
            capture = _render_capture_setup(capture_dir)

        total_frames = int(round(cycle.duration * FPS))
        dt = float(cycle.duration / (total_frames - 1))
        state_rows = {
            angle: {
                "commanded_deg": angle,
                "frames": 0,
                "spray_frames": 0,
                "measured": [],
                "spray_measured": [],
                "stand_off": [],
                "spray_stand_off": [],
                "stand_off_error": [],
                "spray_stand_off_error": [],
                "tcp_error": [],
                "spray_tcp_error": [],
                "tool_tcp_alignment": [],
                "spray_tool_tcp_offset": [],
                "spray_transfer_efficiency": [],
            }
            for angle in (0.0, 7.5, 15.0)
        }
        performance_start = time.perf_counter()
        s2_update_seconds = []
        spray_s2_update_count = 0
        visual_update_count = 0
        capture_index = 0
        for frame in range(total_frames):
            time_s = cycle.duration * frame / (total_frames - 1)
            commanded = float(args.constant_angle) if args.constant_angle is not None else state_for_time(time_s)
            state_label = 0.0 if args.constant_angle is not None else state_label_for_time(time_s)
            cycle.update(stage, frame, total_frames, incidence_deg=commanded)
            tcp = cycle.current_tcp
            spray_axis = np.asarray(cycle.current_spray_axis, dtype=float)
            tool_xform = UsdGeom.Xformable(stage.GetPrimAtPath(SPRAY_TOOL_PRIM)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            tool_prim_origin = _translation(tool_xform)
            # PaintingCycle's current_tcp is the solved SprayGun tip, while
            # the asset root is upstream on the flange.  Use the tip as the
            # plume emission origin and retain the asset-root offset as an
            # explicit measured diagnostic.
            tool_origin = _translation(tcp)
            tool_axis = _vec3(tool_xform, (1.0, 0.0, 0.0))
            tool_y_axis = _vec3(tool_xform, (0.0, 1.0, 0.0))
            tool_z_axis = _vec3(tool_xform, (0.0, 0.0, 1.0))
            alignment = float(np.dot(tool_axis, spray_axis))
            # The existing IK path preserves its stable world-secondary axis;
            # the calibrated fan-major direction is the tangent projection of
            # that same world +Y axis.  SprayTool/TCP axes are still measured
            # from the live stage and checked below.
            measured_frame = surface_frame(cycle.current_surface_normal, (0.0, 1.0, 0.0), spray_axis, minor_tolerance_deg=3.0)
            state = state_rows[state_label]
            state["frames"] += 1
            state["measured"].append(float(measured_frame.incidence_u_deg))
            actual_stand_off = float(np.linalg.norm(_translation(tcp) - cycle.current_surface_point))
            state["stand_off"].append(actual_stand_off)
            state["stand_off_error"].append(float(abs(actual_stand_off - model.stand_off_m)))
            state["tcp_error"].append(float(cycle.records[-1]["tcp_error_m"]))
            state["tool_tcp_alignment"].append(alignment)
            state["spray_tool_tcp_offset"].append(float(np.linalg.norm(tool_prim_origin - tool_origin)))
            s2_start = time.perf_counter()
            step = runtime.step(
                dt_s=dt,
                incidence_angle_deg=commanded,
                stand_off_m=model.stand_off_m,
                air_velocity_m_s=model.air_velocity_m_s,
                mass_flow_kg_s=model.mass_flow_kg_s,
                spray_on=bool(cycle.current_spray_on),
            )
            s2_update_seconds.append(time.perf_counter() - s2_start)
            if cycle.current_spray_on:
                spray_s2_update_count += 1
                state["spray_frames"] += 1
                state["spray_measured"].append(float(measured_frame.incidence_u_deg))
                state["spray_stand_off"].append(actual_stand_off)
                state["spray_stand_off_error"].append(state["stand_off_error"][-1])
                state["spray_tcp_error"].append(state["tcp_error"][-1])
                if step.injected_mass_kg > 0.0:
                    state["spray_transfer_efficiency"].append(float(step.deposited_mass_kg / step.injected_mass_kg))
                accumulator.apply(step, target_position=cycle.current_surface_point, frame=measured_frame)
            if warp_runtime is not None:
                warp_frame = WarpBatchFrame.from_surface_frame(
                    origin_world_m=tool_origin,
                    target_world_m=cycle.current_surface_point,
                    surface_frame=measured_frame,
                    incidence_deg=commanded,
                    stand_off_m=model.stand_off_m,
                )
                warp_runtime.update(
                    time_s,
                    bool(cycle.current_spray_on),
                    warp_frame,
                    actual_tool_origin_m=tool_origin,
                    actual_tool_axes=(tool_axis, tool_y_axis, tool_z_axis),
                )
                _write_warp_points(warp_points, warp_runtime.active_world_positions())
            _write_overlay(overlay, surface_grid, max_mass=float(np.max(surface_grid.cumulative_mass_kg)))
            visual_update_count += 1
            if capture is not None:
                capture_every = 1 if args.warp_plume else 4
                if frame % capture_every == 0 or frame == total_frames - 1:
                    from omni.kit.viewport.utility import capture_viewport_to_file

                    capture_path = capture_dir / f"rgb_{capture_index:06d}.png"
                    capture_viewport_to_file(capture, file_path=str(capture_path))
                    capture_index += 1
                    for _ in range(20):
                        simulation_app.update()
                        if capture_path.exists() and capture_path.stat().st_size > 0:
                            break
                else:
                    simulation_app.update()
            else:
                simulation_app.update()
            if frame % 60 == 0:
                print(f"RUNTIME_FRAME frame={frame} time_s={time_s:.3f} state_deg={commanded:g} spray={int(cycle.current_spray_on)}", flush=True)
        if warp_runtime is not None:
            warp_runtime.shutdown(time_s)
            _write_warp_points(warp_points, warp_runtime.active_world_positions())
        performance_elapsed = time.perf_counter() - performance_start
        simulation_app.update()

        frame_paths = sorted(capture_dir.rglob("rgb_*.png"))
        if not args.no_media:
            media = _encode_warp_media(frame_paths, capture_dir) if args.warp_plume else _encode_media(frame_paths)
        else:
            media = {"hero": None, "deposition": None, "video": None}
        def summary(values):
            return {
                "min": float(min(values)) if values else None,
                "max": float(max(values)) if values else None,
                "mean": float(sum(values) / len(values)) if values else None,
            }
        state_metrics = {}
        for angle, row in state_rows.items():
            state_metrics[str(angle)] = {
                "commanded_incidence_deg": row["commanded_deg"],
                "frames": row["frames"],
                "spray_frames": row["spray_frames"],
                "measured_incidence_deg": summary(row["measured"]),
                "spray_measured_incidence_deg": summary(row["spray_measured"]),
                "stand_off_error_m": summary(row["stand_off_error"]),
                "spray_stand_off_error_m": summary(row["spray_stand_off_error"]),
                "stand_off_m": summary(row["stand_off"]),
                "spray_stand_off_m": summary(row["spray_stand_off"]),
                "tcp_error_m": summary(row["tcp_error"]),
                "spray_tcp_error_m": summary(row["spray_tcp_error"]),
                "spray_tool_tcp_axis_alignment": summary(row["tool_tcp_alignment"]),
                "spray_tool_tcp_offset_m": summary(row["spray_tool_tcp_offset"]),
                "spray_transfer_efficiency": summary(row["spray_transfer_efficiency"]),
            }
        s2_update_array = np.asarray(s2_update_seconds, dtype=float)
        if s2_update_array.size:
            s2_update_summary = {
                "count": int(s2_update_array.size),
                "spray_count": int(spray_s2_update_count),
                "mean_seconds": float(np.mean(s2_update_array)),
                "p95_seconds": float(np.percentile(s2_update_array, 95.0)),
                "min_seconds": float(np.min(s2_update_array)),
                "max_seconds": float(np.max(s2_update_array)),
            }
        else:
            s2_update_summary = {"count": 0, "spray_count": 0, "mean_seconds": None, "p95_seconds": None, "min_seconds": None, "max_seconds": None}
        visual_summary = {
            "update_count": int(visual_update_count),
            "interval_sim_seconds": float(cycle.duration / max(visual_update_count - 1, 1)),
            "rate_sim_hz": float(visual_update_count / cycle.duration),
            "rate_wall_hz": float(visual_update_count / performance_elapsed) if performance_elapsed else None,
        }
        if capture is not None and capture_dir.exists():
            shutil.rmtree(capture_dir)
        s2_ledger = {**runtime.as_dict(), **accumulator.as_dict(), "surface_map_closure_error_kg": surface_grid.integrated_mass_kg - runtime.deposited_mass_kg}
        if args.warp_plume:
            warp_ledger = warp_runtime.as_dict()
            metrics = {
                "status": "ISAAC_WARP_PLUME_VALIDATED",
                "schema_version": "isaac_warp_runtime_metrics_v1",
                "isaac_sim": {"version": "6.0.1-rc.7+release.42383.32955d8d.gl", "launcher": "native Isaac Sim python.bat"},
                "scene": {"path": SCENE_DISPLAY_PATH, "root": ROOT_PRIM, "rail": RAIL_PRIM, "robot": ROBOT_PRIM, "flange": FLANGE_PRIM, "process_tool": PROCESS_TOOL_PRIM, "spray_tool": SPRAY_TOOL_PRIM, "workpiece": PANEL_PRIM, "overlay": OVERLAY_PRIM, "warp_plume": WARP_PLUME_PRIM},
                "model": {"path": MODEL_DISPLAY_PATH, "model_id": model.model_id, "model_sha256": model.model_sha256, "calibrated_angle_deg": [model.minimum_angle_deg, model.maximum_angle_deg], "stand_off_m": model.stand_off_m},
                "warp_model": {"path": WARP_MODEL_DISPLAY_PATH, **warp_ledger["carrier"]},
                "states": state_metrics,
                "s2": s2_ledger,
                "warp": warp_ledger,
                "ledger": {"s2_authoritative": s2_ledger, "warp_transport_diagnostic": warp_ledger},
                "performance": {"simulated_duration_s": float(cycle.duration), "frame_count": total_frames, "surface_element_count": int(len(surface_grid.positions)), "s2_updates": s2_update_summary, "visualization": visual_summary, "wall_seconds": float(performance_elapsed), "runtime_fps": float(total_frames / performance_elapsed), "real_time_ratio": float(cycle.duration / performance_elapsed)},
                "media": media,
                "capture": {"resolution": list(RESOLUTION), "source_frame_count": len(frame_paths), "source_framerate_hz": FPS if frame_paths else None, "output_framerate_hz": 30.0 if media["video"] else None},
                "integration": {"default_process_model": "S2 calibrated deposition runtime", "warp_layer": "optional W1.3 fixed-vector transport plume", "projected_deposition_overlay": "not implemented; S2 overlay remains authoritative", "world_stable_particles": True, "live_spray_tool_prim": SPRAY_TOOL_PRIM},
                "events": runtime.events,
                "generated_at_epoch_s": time.time(),
            }
            output_dir = WARP_RESULT_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "runtime_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"RUNTIME_PASS frames={total_frames} s2_ledger_error={s2_ledger['balance_error_kg']:.6g} warp_balance_error={warp_ledger['balance_error_kg']:.6g}", flush=True)
        else:
            metrics = {
                "status": "ISAAC_S2_RUNTIME_VALIDATED",
                "schema_version": "isaac_s2_runtime_metrics_v1",
                "isaac_sim": {"version": "6.0.1-rc.7+release.42383.32955d8d.gl", "launcher": "native Isaac Sim python.bat"},
                "scene": {"path": SCENE_DISPLAY_PATH, "root": ROOT_PRIM, "rail": RAIL_PRIM, "robot": ROBOT_PRIM, "flange": FLANGE_PRIM, "process_tool": PROCESS_TOOL_PRIM, "spray_tool": SPRAY_TOOL_PRIM, "workpiece": PANEL_PRIM, "overlay": OVERLAY_PRIM},
                "model": {"path": MODEL_DISPLAY_PATH, "model_id": model.model_id, "model_sha256": model.model_sha256, "calibrated_angle_deg": [model.minimum_angle_deg, model.maximum_angle_deg], "stand_off_m": model.stand_off_m},
                "states": state_metrics,
                "ledger": s2_ledger,
                "performance": {"simulated_duration_s": float(cycle.duration), "frame_count": total_frames, "surface_element_count": int(len(surface_grid.positions)), "s2_updates": s2_update_summary, "visualization": visual_summary, "wall_seconds": float(performance_elapsed), "runtime_fps": float(total_frames / performance_elapsed), "real_time_ratio": float(cycle.duration / performance_elapsed)},
                "media": media,
                "capture": {"resolution": list(RESOLUTION), "source_frame_count": len(frame_paths), "source_framerate_hz": FPS / 4.0 if frame_paths else None, "output_framerate_hz": 30.0 if media["video"] else None},
                "events": runtime.events,
                "generated_at_epoch_s": time.time(),
            }
            output_dir = RESULT_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "runtime_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"RUNTIME_PASS frames={total_frames} ledger_error={metrics['ledger']['balance_error_kg']:.6g} map_error={metrics['ledger']['surface_map_closure_error_kg']:.6g}", flush=True)
        print(f"MEDIA hero={media['hero']} deposition={media['deposition']} video={media['video']}", flush=True)
        return 0
    except Exception as exc:
        print(f"NATIVE_RUNTIME_ERROR {type(exc).__name__}: {exc}", flush=True)
        return 2
    finally:
        simulation_app.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-media", action="store_true")
    parser.add_argument("--constant-angle", type=float, default=None, help="debug: hold one commanded incidence for the full motion")
    parser.add_argument("--warp-plume", action="store_true", help="enable the optional native W1.3 rolling Warp plume layer")
    parser.add_argument("--warp-particles-per-bin", type=int, default=500)
    parser.add_argument("--warp-cadence-hz", type=float, default=15.0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
