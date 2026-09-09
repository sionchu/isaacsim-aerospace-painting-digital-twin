"""Visualize the solved OpenFOAM 7.5-degree field in native Isaac Sim.

This entry point is deliberately separate from the validated painting physics
runner.  It loads the compact offline field, places one local reference frame
at the current SprayGun/target patch, and captures technical view modes without
solving CFD or changing S2/Warp process inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = ROOT.parent / "visual_prototypes"
SCENE_PATH = VISUAL_ROOT / "scenes" / "aircraft_painting_cell.usda"
PANEL_ASSET = ROOT.parent / "assets" / "course" / "generic_aircraft" / "generic_panel.usda"
MODEL_PATH = ROOT / "models" / "s2_air_assisted_v1.json"
PLAN_PATH = ROOT / "results" / "air_assisted" / "full_panel_film" / "full_panel_plan.json"
FLOW_NPZ = ROOT / "models" / "openfoam_flow_field_7p5deg_v1.npz"
FLOW_JSON = ROOT / "models" / "openfoam_flow_field_7p5deg_v1.json"
FULL_PANEL_DATA = ROOT / "results" / "air_assisted" / "full_panel_film" / "full_panel_surface_data.npz"
FULL_PANEL_METRICS = ROOT / "results" / "air_assisted" / "full_panel_film" / "full_panel_metrics.json"
MEDIA_DIR = ROOT / "media" / "air_assisted_spray"
RESULT_DIR = ROOT / "results" / "air_assisted" / "cfd_flow_visualization"
FPS = 30
RESOLUTION = (1920, 1080)
ROOT_PRIM = "/World/AerospacePaintingCell"
RAIL_PRIM = "/World/AerospacePaintingCell/LinearTrack/Carriage"
ROBOT_PRIM = "/World/AerospacePaintingCell/LinearTrack/Carriage/PaintRobot/FANUC"
FLANGE_PRIM = "/World/AerospacePaintingCell/LinearTrack/Carriage/PaintRobot/FANUC/Asset/J6_link/flange"
PROCESS_TOOL_PRIM = FLANGE_PRIM + "/ProcessTool"
SPRAY_TOOL_PRIM = PROCESS_TOOL_PRIM + "/SprayGun"
PANEL_PRIM = "/World/Aircraft/Panel/Asset"
CFD_ROOT_PRIM = ROOT_PRIM + "/OpenFOAMReferenceFlow"
CFD_VECTORS_PRIM = CFD_ROOT_PRIM + "/VelocityVectors"
CFD_SLICE_PRIM = CFD_ROOT_PRIM + "/VelocityMagnitudeSlice"
CFD_STREAMLINES_PRIM = CFD_ROOT_PRIM + "/Streamlines"
CFD_HUD_ROOT = ROOT_PRIM + "/CFDFlowHUD"
WFT_OVERLAY_PRIM = ROOT_PRIM + "/FilmThicknessOverlay"
WARP_PLUME_PRIM = ROOT_PRIM + "/WarpSprayPlume"
WARP_GUIDE_PRIM = ROOT_PRIM + "/WarpSprayPlumeGuide"


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {"min": float(array.min()), "mean": float(array.mean()), "p95": float(np.percentile(array, 95.0)), "max": float(array.max())}


def _color(value: float, low: float, high: float) -> tuple[float, float, float]:
    q = float(np.clip((value - low) / max(high - low, 1.0e-9), 0.0, 1.0))
    # Restrained cyan/blue engineering scale; the Warp plume stays warm/orange.
    return (0.04 + 0.10 * q, 0.35 + 0.48 * q, 0.78 + 0.18 * q)


def _set_visible(prim, visible: bool) -> None:
    from pxr import UsdGeom

    imageable = UsdGeom.Imageable(prim.GetPrim() if hasattr(prim, "GetPrim") else prim)
    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)


def _create_basis_curves(stage, path: str, lines: list[np.ndarray], colors: list[tuple[float, float, float]], width_m: float, source: str):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    curve = UsdGeom.BasisCurves.Define(stage, path)
    if lines:
        points = np.concatenate(lines, axis=0)
        counts = [int(len(line)) for line in lines]
        curve.CreatePointsAttr(points.tolist())
        curve.CreateCurveVertexCountsAttr(counts)
        curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        curve.CreateWidthsAttr().Set([float(width_m)])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set([tuple(map(float, color)) for color in colors])
    else:
        curve.CreatePointsAttr([])
        curve.CreateCurveVertexCountsAttr([])
    curve.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    # RTX does not consistently render an unbound BasisCurves prim from its
    # displayColor alone.  Give the batched lines a small emissive engineering
    # material while retaining per-line display colors for renderers that
    # support the primvar.
    material_name = "OpenFOAMReferenceFlowStreamlines" if "streamline" in source.lower() else "OpenFOAMReferenceFlowVectors"
    material_path = f"/World/Looks/{material_name}"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    base_color = (0.08, 0.82, 0.98) if "streamline" in source.lower() else (0.06, 0.56, 0.90)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(0.35 * np.asarray(base_color))))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)
    prim = curve.GetPrim()
    prim.CreateAttribute("cfd:source", Sdf.ValueTypeNames.String).Set(source)
    prim.CreateAttribute("cfd:offlineReference", Sdf.ValueTypeNames.Bool).Set(True)
    return curve


def _create_batched_line_mesh(stage, path: str, lines: list[np.ndarray], colors: list[tuple[float, float, float]], width_m: float, source: str, *, arrowheads: bool = False):
    """Create a renderer-portable batched ribbon mesh for technical lines.

    RTX 6.0.1 can retain a BasisCurves prim in the stage while omitting it
    from a headless render product on some driver/material combinations.  A
    single double-sided mesh made of narrow, oriented quads is the fallback
    geometry: it remains one batched object (not one prim per cell), preserves
    the same source-derived line coordinates, and is visible in both viewport
    and file capture paths.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, path)
    vertices: list[tuple[float, float, float]] = []
    faces: list[int] = []
    face_colors: list[tuple[float, float, float]] = []
    half_width = 0.5 * float(width_m)

    def add_ribbon(p0: np.ndarray, p1: np.ndarray, color: tuple[float, float, float], arrow: bool) -> None:
        delta = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            return
        direction = delta / length
        reference = np.asarray((0.0, 1.0, 0.0), dtype=float)
        side = np.cross(direction, reference)
        if float(np.linalg.norm(side)) <= 1.0e-7:
            side = np.cross(direction, np.asarray((1.0, 0.0, 0.0), dtype=float))
        side /= max(float(np.linalg.norm(side)), 1.0e-9)
        side *= half_width
        base = len(vertices)
        vertices.extend((tuple(p0 - side), tuple(p0 + side), tuple(p1 + side), tuple(p1 - side)))
        faces.extend((base, base + 1, base + 2, base + 3))
        face_colors.append(color)
        if arrow:
            head_len = min(0.007, max(length * 0.32, 0.002))
            head_base = np.asarray(p1, dtype=float) - direction * head_len
            head_side = side * 2.8
            tri_base = len(vertices)
            vertices.extend((tuple(p1), tuple(head_base - head_side), tuple(head_base + head_side)))
            faces.extend((tri_base, tri_base + 1, tri_base + 2))
            face_colors.append(color)

    for line, color in zip(lines, colors):
        points = np.asarray(line, dtype=float)
        for start, end in zip(points[:-1], points[1:]):
            add_ribbon(start, end, color, arrowheads and len(points) == 2)

    mesh.CreatePointsAttr(vertices)
    # The faces list stores indices in groups of four for ribbons and groups of
    # three for vector arrowheads; rebuild counts while adding the geometry to
    # keep the USD arrays unambiguous.
    counts: list[int] = []
    cursor = 0
    for line in lines:
        points = np.asarray(line, dtype=float)
        segment_count = max(len(points) - 1, 0)
        counts.extend([4] * segment_count)
        if arrowheads and len(points) == 2 and segment_count:
            counts.append(3)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(face_colors)
    material_name = "OpenFOAMReferenceFlowStreamlineMesh" if "streamline" in source.lower() else "OpenFOAMReferenceFlowVectorMesh"
    material_path = f"/World/Looks/{material_name}"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    base_color = (0.08, 0.82, 0.98) if "streamline" in source.lower() else (0.06, 0.56, 0.90)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(0.45 * np.asarray(base_color))))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.30)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    prim = mesh.GetPrim()
    prim.CreateAttribute("cfd:source", Sdf.ValueTypeNames.String).Set(source)
    prim.CreateAttribute("cfd:offlineReference", Sdf.ValueTypeNames.Bool).Set(True)
    return mesh


def _create_flow_geometry(stage, field, *, vector_count: int = 512, streamline_count: int = 25):
    from pxr import Sdf, UsdGeom, UsdShade
    from aerospace_painting.openfoam_flow_field import deterministic_streamlines

    root = UsdGeom.Xform.Define(stage, CFD_ROOT_PRIM)
    transform_op = root.AddTransformOp()

    flat_centres = field.grid_centres_m.reshape(-1, 3).astype(np.float64)
    flat_velocity = field.velocity_m_s.reshape(-1, 3).astype(np.float64)
    flat_speed = field.speed_magnitude_m_s.reshape(-1).astype(float)
    count = min(int(vector_count), len(flat_centres))
    sample_indices = np.linspace(0, len(flat_centres) - 1, count, dtype=int)
    vector_lines: list[np.ndarray] = []
    vector_colors: list[tuple[float, float, float]] = []
    for index in sample_indices:
        start = flat_centres[index]
        velocity = flat_velocity[index]
        speed = float(flat_speed[index])
        direction = velocity / max(float(np.linalg.norm(velocity)), 1.0e-9)
        length = 0.0015 + 0.0020 * speed
        vector_lines.append(np.asarray((start, start + direction * length), dtype=np.float64))
        vector_colors.append(_color(speed, float(flat_speed.min()), float(flat_speed.max())))
    vectors = _create_basis_curves(stage, CFD_VECTORS_PRIM, vector_lines, vector_colors, 0.0045, "OpenFOAM v2606 solved U")
    vector_mesh = _create_batched_line_mesh(stage, CFD_VECTORS_PRIM + "Mesh", vector_lines, vector_colors, 0.0045, "OpenFOAM v2606 solved U", arrowheads=True)

    nz, ny, nx = field.velocity_m_s.shape[:3]
    slice_centres, slice_speeds, y_index = field.magnitude_slice(axis="y", coordinate_m=0.0)
    y_values = field.grid_centres_m[0, :, 0, 1]
    slice_centres = slice_centres.astype(np.float64)
    slice_speeds = slice_speeds.astype(float)
    slice_points = slice_centres.reshape(-1, 3)
    faces: list[int] = []
    for iz in range(nz - 1):
        for ix in range(nx - 1):
            base = iz * nx + ix
            faces.extend((base, base + 1, base + nx + 1, base + nx))
    mesh = UsdGeom.Mesh.Define(stage, CFD_SLICE_PRIM)
    mesh.CreatePointsAttr(slice_points.tolist())
    mesh.CreateFaceVertexCountsAttr([4] * ((nz - 1) * (nx - 1)))
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set([_color(value, float(flat_speed.min()), float(flat_speed.max())) for value in slice_speeds.reshape(-1)])
    mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([0.24])
    # Bind eight transparent speed bands to face subsets.  This keeps the
    # actual |U|-derived engineering colormap while avoiding an RTX driver
    # dependency on an unavailable UsdPrimvarReader_color3f shader node.
    slice_min = float(slice_speeds.min())
    slice_max = float(slice_speeds.max())
    cell_speeds = 0.25 * (slice_speeds[:-1, :-1] + slice_speeds[1:, :-1] + slice_speeds[:-1, 1:] + slice_speeds[1:, 1:])
    normalized_cells = np.clip((cell_speeds - slice_min) / max(slice_max - slice_min, 1.0e-9), 0.0, 1.0)
    for band in range(8):
        lower = band / 8.0
        upper = (band + 1) / 8.0
        cell_indices = np.flatnonzero((normalized_cells >= lower) & (normalized_cells <= upper if band == 7 else normalized_cells < upper)).astype(int).tolist()
        subset = UsdGeom.Subset.Define(stage, f"{CFD_SLICE_PRIM}/SpeedBand{band:02d}")
        subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr().Set("materialBind")
        subset.CreateIndicesAttr().Set(cell_indices)
        material_path = f"/World/Looks/OpenFOAMReferenceFlowSliceBand{band:02d}"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        band_speed = slice_min + (band + 0.5) / 8.0 * (slice_max - slice_min)
        band_color = _color(band_speed, slice_min, slice_max)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(band_color)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(tuple(0.08 * np.asarray(band_color)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.24)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.render)
    slice_prim = mesh.GetPrim()
    slice_prim.CreateAttribute("cfd:source", Sdf.ValueTypeNames.String).Set("OpenFOAM v2606 solved |U|")
    slice_prim.CreateAttribute("cfd:plane", Sdf.ValueTypeNames.String).Set(f"fan-major / spray-axis, y={float(y_values[y_index]):.6f} m")

    seeds = []
    seed_axis = np.linspace(-0.04, 0.04, int(round(math.sqrt(streamline_count))))
    for x in seed_axis:
        for y in seed_axis:
            seeds.append((float(x), float(y), 0.006))
    seeds = np.asarray(seeds[:streamline_count], dtype=float)
    streamlines_local = deterministic_streamlines(field, seeds, step_m=0.005, max_steps=48)
    streamline_colors = [(0.08, 0.86, 0.98)] * len(streamlines_local)
    streamlines = _create_basis_curves(stage, CFD_STREAMLINES_PRIM, streamlines_local, streamline_colors, 0.0055, "OpenFOAM v2606 solved U streamline integration")
    streamline_mesh = _create_batched_line_mesh(stage, CFD_STREAMLINES_PRIM + "Mesh", streamlines_local, streamline_colors, 0.0055, "OpenFOAM v2606 solved U streamline integration")
    for prim in (vectors.GetPrim(), vector_mesh.GetPrim(), mesh.GetPrim(), streamlines.GetPrim(), streamline_mesh.GetPrim()):
        _set_visible(prim, True)
    return {
        "root": root,
        "transform_op": transform_op,
        "vectors": vectors,
        "vector_mesh": vector_mesh,
        "slice": mesh,
        "streamlines": streamlines,
        "streamline_mesh": streamline_mesh,
        "vector_count": len(vector_lines),
        "slice_vertices": int(len(slice_points)),
        "slice_cells": int((nz - 1) * (nx - 1)),
        "slice_y_m": float(y_values[y_index]),
        "slice_speed_stats": {"min_m_s": float(slice_speeds.min()), "mean_m_s": float(slice_speeds.mean()), "max_m_s": float(slice_speeds.max())},
        "streamline_count": len(streamlines_local),
        "streamline_seed_rule": "5x5 deterministic seeds at z=0.006 m, truncated to target count",
    }


def _set_flow_transform(transform_op, frame) -> None:
    from pxr import Gf

    u = np.asarray(frame.u_world, dtype=float)
    v = np.asarray(frame.v_world, dtype=float)
    w = np.asarray(frame.w_world, dtype=float)
    origin = np.asarray(frame.origin_world_m, dtype=float)
    benchmark_origin = np.asarray((0.0, 0.0, 0.002), dtype=float)
    translation = origin - benchmark_origin[0] * u - benchmark_origin[1] * v - benchmark_origin[2] * w
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRow(0, Gf.Vec4d(float(u[0]), float(v[0]), float(w[0]), 0.0))
    matrix.SetRow(1, Gf.Vec4d(float(u[1]), float(v[1]), float(w[1]), 0.0))
    matrix.SetRow(2, Gf.Vec4d(float(u[2]), float(v[2]), float(w[2]), 0.0))
    matrix.SetRow(3, Gf.Vec4d(float(translation[0]), float(translation[1]), float(translation[2]), 1.0))
    transform_op.Set(matrix)


def _create_cfd_camera(stage, origin_world_m, frame) -> str:
    """Create a close technical camera aimed through the local process patch."""
    from pxr import Gf, UsdGeom

    path = "/World/Cameras/cfd_flow"
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr().Set(42.0)
    camera.CreateHorizontalApertureAttr().Set(36.0)
    camera.CreateVerticalApertureAttr().Set(20.25)
    origin = np.asarray(origin_world_m, dtype=float)
    u = np.asarray(frame.u_world, dtype=float)
    v = np.asarray(frame.v_world, dtype=float)
    w = np.asarray(frame.w_world, dtype=float)
    target = origin + 0.12 * w
    # Stand behind the nozzle and look down the inward process direction.  A
    # small fan-major offset keeps the slice and vector field readable while
    # retaining the real panel/tool context in the frame.
    eye = origin - 0.62 * w + 0.48 * u + 0.72 * v
    view = target - eye
    view = view / max(float(np.linalg.norm(view)), 1.0e-9)
    right = np.cross(view, v)
    right = right / max(float(np.linalg.norm(right)), 1.0e-9)
    up = np.cross(right, view)
    up = up / max(float(np.linalg.norm(up)), 1.0e-9)
    # USD camera local -Z is the viewing direction.  The stage uses row-vector
    # transforms, so each camera basis vector is written as a matrix row.
    look_at = Gf.Matrix4d(1.0)
    look_at.SetRow(0, Gf.Vec4d(float(right[0]), float(right[1]), float(right[2]), 0.0))
    look_at.SetRow(1, Gf.Vec4d(float(up[0]), float(up[1]), float(up[2]), 0.0))
    look_at.SetRow(2, Gf.Vec4d(float(-view[0]), float(-view[1]), float(-view[2]), 0.0))
    look_at.SetRow(3, Gf.Vec4d(float(eye[0]), float(eye[1]), float(eye[2]), 1.0))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(look_at)
    return path


class _HudOverlay:
    """Viewport-scene HUD that is available in Isaac Sim 6.0.1.

    Isaac Sim 6.0.1 does not expose ``UsdGeom.Text``.  ``omni.ui.scene`` is
    the native viewport-overlay API and is rendered by the viewport capture
    path, so it keeps the labels in the technical stills without adding a
    machine-specific USD asset.
    """

    def __init__(self, viewport_window, *, slice_y_m: float):
        import omni.ui as ui
        from omni.ui import scene as sc

        self.viewport_window = viewport_window
        self.slice_y_m = float(slice_y_m)
        self.scene_view = sc.SceneView(aspect_ratio_policy=sc.AspectRatioPolicy.STRETCH)
        with viewport_window.get_frame("cfd_flow_hud"):
            with ui.Frame(width=ui.Fraction(1), height=ui.Fraction(1), opaque_for_mouse_events=False):
                # The scene view is attached to the viewport below.  Screen
                # coordinates make the overlay independent of the robot pose.
                self.scene_view = sc.SceneView(aspect_ratio_policy=sc.AspectRatioPolicy.STRETCH)
        viewport_window.viewport_api.add_scene_view(self.scene_view)
        self.set_mode("flow")

    def _build(self, mode: str) -> None:
        import omni.ui as ui
        from omni.ui import scene as sc

        lines = {
            "process": ["PROCESS FRAME", "S2 + Warp process view"],
            "flow": [
                "OPENFOAM REFERENCE FLOW - LOCAL TANGENT FRAME",
                "OpenFOAM v2606 | Reference field: 7.5 deg | Cells: 27,648",
                "Offline CFD -> Isaac visualization",
                "u = fan-major   v = travel/fan-minor   w = process direction",
                "Stand-off: 0.240 m   Incidence: 7.5 deg",
            ],
            "combined": [
                "COMBINED: OFFLINE CFD FLOW + WARP DROPLETS",
                "OpenFOAM v2606 | Reference field: 7.5 deg | Cells: 27,648",
                "Local tangent-frame mapping | CFD solve remains offline",
            ],
            "slice": [
                "AIR VELOCITY MAGNITUDE [m/s]",
                "OpenFOAM v2606 reference field | y ~= 0 slice",
            ],
            "result": ["RESULT FRAME", "Estimated WFT overlay | model-derived, not measured"],
        }[mode]
        self.scene_view.scene.clear()
        with self.scene_view.scene:
            sc.Screen()
            with sc.Transform(scale_to=sc.Space.NDC, transform=sc.Matrix44.get_translation_matrix(-0.74, 0.40, 0.0)):
                sc.Rectangle(width=0.84, height=0.22 + 0.035 * len(lines), color=0xA0081724)
                with sc.Transform(transform=sc.Matrix44.get_translation_matrix(0.02, 0.085, 0.0)):
                    for index, line in enumerate(lines):
                        with sc.Transform(transform=sc.Matrix44.get_translation_matrix(0.0, -0.035 * index, 0.0)):
                            sc.Label(line, size=18 if index == 0 else 14, color=0xE6F5FCFF, alignment=ui.Alignment.LEFT)
            if mode in {"flow", "combined", "slice"}:
                with sc.Transform(scale_to=sc.Space.NDC, transform=sc.Matrix44.get_translation_matrix(-0.74, -0.30, 0.0)):
                    sc.Rectangle(width=0.40, height=0.095, color=0xA0081724)
                    with sc.Transform(transform=sc.Matrix44.get_translation_matrix(0.02, 0.025, 0.0)):
                        sc.Label("Air velocity magnitude [m/s]", size=14, color=0x59E0FAFF, alignment=ui.Alignment.LEFT)
                        with sc.Transform(transform=sc.Matrix44.get_translation_matrix(0.0, -0.035, 0.0)):
                            sc.Label(f"Slice plane: y ~= {self.slice_y_m:.4f} m", size=12, color=0xB9EAF2FF, alignment=ui.Alignment.LEFT)

    def set_mode(self, mode: str) -> None:
        self._build(mode)

    def destroy(self) -> None:
        try:
            self.scene_view.scene.clear()
            self.viewport_window.viewport_api.remove_scene_view(self.scene_view)
        except Exception:
            pass


def _create_hud(viewport_window, *, mode: str, slice_y_m: float):
    hud = _HudOverlay(viewport_window, slice_y_m=slice_y_m)
    hud.set_mode(mode)
    return hud


def _legacy_create_hud(stage, camera_world, *, mode: str, field, frame, slice_y_m: float):
    """Compatibility shim retained for callers that imported the old helper."""
    from pxr import UsdGeom

    hud = UsdGeom.Xform.Define(stage, CFD_HUD_ROOT)
    lines = {
        "process": "PROCESS FRAME\nS2 + Warp process view",
        "flow": "OPENFOAM REFERENCE FLOW — LOCAL TANGENT FRAME\nOpenFOAM v2606 | Reference field: 7.5° | Cells: 27,648\nOffline CFD → Isaac visualization\nu = fan-major   v = travel/fan-minor   w = process direction\nStand-off: 0.240 m   Incidence: 7.5°",
        "combined": "COMBINED: OFFLINE CFD FLOW + WARP DROPLETS\nOpenFOAM v2606 | Reference field: 7.5° | Cells: 27,648\nLocal tangent-frame mapping | CFD solve remains offline",
        "slice": "AIR VELOCITY MAGNITUDE [m/s]\nOpenFOAM v2606 reference field | y≈0 slice",
        "result": "RESULT FRAME\nEstimated WFT overlay | model-derived, not measured",
    }
    return hud, hud, hud


def _set_mode(mode: str, geometry: dict, warp_points, plume_guide, overlay, hud_overlay) -> None:
    flow_visible = mode in {"flow", "combined", "slice"}
    _set_visible(geometry["root"].GetPrim(), flow_visible)
    _set_visible(geometry["vectors"].GetPrim(), mode in {"flow", "combined"})
    _set_visible(geometry["slice"].GetPrim(), mode in {"flow", "combined", "slice"})
    _set_visible(geometry["streamlines"].GetPrim(), mode in {"flow", "combined"})
    if warp_points is not None:
        _set_visible(warp_points.GetPrim(), mode in {"process", "combined"})
    if plume_guide is not None:
        _set_visible(plume_guide.GetPrim(), mode in {"process", "combined"})
    if overlay is not None:
        _set_visible(overlay.GetPrim(), mode == "result")
    hud_overlay.set_mode(mode)


def _capture(viewport, path: Path, simulation_app) -> None:
    from omni.kit.viewport.utility import capture_viewport_to_file

    path.parent.mkdir(parents=True, exist_ok=True)
    # The capture helper is asynchronous.  Remove a previous still first so
    # a stale file cannot be mistaken for completion of the new render.
    if path.exists():
        path.unlink()
    capture_viewport_to_file(viewport, file_path=str(path))
    for _ in range(90):
        simulation_app.update()
        if path.exists() and path.stat().st_size > 0:
            return
    raise RuntimeError(f"viewport capture did not complete: {path}")


def _annotate_capture(path: Path, mode: str, field, geometry: dict, stand_off_m: float, incidence_deg: float) -> None:
    """Add the compact engineering HUD to the captured technical still.

    The native viewport renderer captures the scene render product, while
    ``omni.ui.scene`` overlays are interactive UI and are not included in that
    headless file capture.  The annotation is therefore applied to the
    already-rendered still with Pillow; it does not alter any USD or physics.
    """
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    bold_path = Path("C:/Windows/Fonts/seguisb.ttf")
    try:
        body = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default()
        title = ImageFont.truetype(str(bold_path), 29) if bold_path.exists() else body
        small = ImageFont.truetype(str(font_path), 19) if font_path.exists() else body
    except Exception:
        body = title = small = ImageFont.load_default()

    if mode == "flow":
        lines = [
            "OPENFOAM REFERENCE FLOW - LOCAL TANGENT FRAME",
            "OpenFOAM v2606 | Reference field: 7.5 deg | Cells: 27,648",
            "Offline CFD -> Isaac visualization",
            "u = fan-major   v = travel/fan-minor   w = process direction",
            f"Stand-off: {stand_off_m:.3f} m   Incidence: {incidence_deg:.1f} deg",
        ]
    elif mode == "combined":
        lines = [
            "COMBINED - OFFLINE CFD FLOW + WARP DROPLETS",
            "Cyan: OpenFOAM vectors / slice / streamlines",
            "Orange: Warp W1.3 droplet transport",
            "Local tangent-frame mapping | CFD solve remains offline",
        ]
    elif mode == "slice":
        lines = [
            "AIR VELOCITY MAGNITUDE [m/s]",
            f"OpenFOAM v2606 reference field | y ~= {geometry['slice_y_m']:.4f} m",
            f"Slice range: {geometry['slice_speed_stats']['min_m_s']:.2f} - {geometry['slice_speed_stats']['max_m_s']:.2f} m/s",
        ]
    elif mode == "process":
        lines = ["PROCESS FRAME", "S2 + Warp process view"]
    else:
        lines = ["RESULT FRAME", "Estimated WFT overlay | model-derived, not measured"]

    left, top = 42, 38
    line_gap = 11
    heights = [title.getbbox(line)[3] - title.getbbox(line)[1] if index == 0 else body.getbbox(line)[3] - body.getbbox(line)[1] for index, line in enumerate(lines)]
    panel_height = sum(heights) + line_gap * max(len(lines) - 1, 0) + 34
    panel_width = min(image.width - 84, max(720, max(int(draw.textlength(line, font=title if index == 0 else body)) for index, line in enumerate(lines)) + 46))
    draw.rounded_rectangle((left, top, left + panel_width, top + panel_height), radius=12, fill=(5, 18, 28, 214), outline=(63, 173, 194, 210), width=2)
    y = top + 17
    for index, line in enumerate(lines):
        use_font = title if index == 0 else body
        draw.text((left + 22, y), line, font=use_font, fill=(225, 247, 252, 255))
        y += heights[index] + line_gap

    if mode in {"flow", "combined", "slice"}:
        legend_left = 42
        legend_top = image.height - 104
        legend_width = 480 if mode != "slice" else 610
        draw.rounded_rectangle((legend_left, legend_top, legend_left + legend_width, image.height - 36), radius=10, fill=(5, 18, 28, 205), outline=(63, 173, 194, 180), width=2)
        draw.text((legend_left + 18, legend_top + 12), "Air velocity magnitude [m/s]", font=small, fill=(150, 239, 252, 255))
        bar_x0, bar_x1 = legend_left + 18, legend_left + legend_width - 18
        bar_y0, bar_y1 = legend_top + 45, legend_top + 59
        for index in range(bar_x1 - bar_x0):
            q = index / max(bar_x1 - bar_x0 - 1, 1)
            draw.line((bar_x0 + index, bar_y0, bar_x0 + index, bar_y1), fill=(int(10 + 30 * q), int(80 + 145 * q), int(220 + 30 * q), 255), width=1)
        draw.text((bar_x0, bar_y1 + 3), f"{field.stats['min_m_s']:.2f}", font=small, fill=(205, 233, 240, 255))
        max_text = f"{field.stats['max_m_s']:.2f}"
        draw.text((bar_x1 - draw.textlength(max_text, font=small), bar_y1 + 3), max_text, font=small, fill=(205, 233, 240, 255))
    image.convert("RGB").save(path)


def run(args: argparse.Namespace) -> int:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(launch_config={"headless": bool(args.headless), "width": RESOLUTION[0], "height": RESOLUTION[1]})
    wall_start = time.perf_counter()
    warp_runtime = None
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(ROOT / "scripts"))
        sys.path.insert(0, str(VISUAL_ROOT))
        from aerospace_painting.openfoam_flow_field import OpenFOAMFlowField
        from aerospace_painting.s2_runtime import S2RuntimeModel, StructuredSurfaceGrid, surface_frame
        from run_isaac_full_panel_painting import (
            FullPanelPaintingCycle,
            _create_overlay,
            _create_plume_guide,
            _create_warp_points,
            _density_from_config,
            _load_plan,
            _translation,
            _vec3,
            _write_overlay,
            _write_plume_guide,
            _write_warp_points,
        )
        from aerospace_painting.warp_plume_runtime import WarpBatchFrame, WarpPlumeRuntime

        field = OpenFOAMFlowField.load(FLOW_NPZ, FLOW_JSON)
        model = S2RuntimeModel.load(MODEL_PATH)
        surface = __import__("aerospace_painting.full_panel_planner", fromlist=["PanelSurface"]).PanelSurface.from_usda(PANEL_ASSET)
        surface.source_path = "assets/course/generic_aircraft/generic_panel.usda"
        plan = _load_plan(model, surface)
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
        frame_total = int(round(plan.duration_s * FPS)) + 1
        frame_index = int(np.clip(round(float(args.frame_time_s) / plan.duration_s * (frame_total - 1)), 0, frame_total - 1))
        pose = cycle.update(stage, frame_index, frame_total)
        actual_axis = np.asarray(cycle.base.current_spray_axis, dtype=float)
        try:
            measured_frame = surface_frame(pose.normal, pose.frame.u_fan_major, actual_axis, minor_tolerance_deg=3.0)
        except ValueError:
            measured_frame = surface_frame(pose.normal, pose.frame.u_fan_major, actual_axis, minor_tolerance_deg=89.0)
        nozzle = _translation(cycle.base.current_tcp)
        flow_frame = WarpBatchFrame.from_surface_frame(
            origin_world_m=nozzle,
            target_world_m=cycle.base.current_surface_point,
            surface_frame=measured_frame,
            incidence_deg=measured_frame.incidence_u_deg,
            stand_off_m=model.stand_off_m,
        )
        geometry = _create_flow_geometry(stage, field, vector_count=args.vector_count, streamline_count=args.streamline_count)
        _set_flow_transform(geometry["transform_op"], flow_frame)
        cfd_camera_path = _create_cfd_camera(stage, nozzle, flow_frame)

        overlay = _create_overlay(stage, grid)
        if FULL_PANEL_DATA.exists():
            with np.load(FULL_PANEL_DATA, allow_pickle=False) as data:
                grid.cumulative_mass_kg[:] = np.asarray(data["cumulative_mass_kg"], dtype=float)
        _write_overlay(overlay, grid, _density_from_config(ROOT / "configs" / "air_assisted_spray.yaml"))

        warp_points = None
        plume_guide = None
        warp_update_seconds: list[float] = []
        if args.view in {"process", "combined"} or args.capture_stills:
            warp_runtime = WarpPlumeRuntime(
                model_path=ROOT / "models" / "warp_vector_carrier_v3.json",
                repo_root=ROOT,
                mass_flow_kg_s=model.mass_flow_kg_s,
                stand_off_m=model.stand_off_m,
                particle_count_per_bin=500,
                cadence_hz=15.0,
                particle_dt_s=5.0e-5,
                maximum_flight_s=0.06,
            )
            warp_points = _create_warp_points(stage)
            plume_guide = _create_plume_guide(stage)
            tool_axes = (_vec3(cycle.base.current_tcp, (1.0, 0.0, 0.0)), _vec3(cycle.base.current_tcp, (0.0, 1.0, 0.0)), _vec3(cycle.base.current_tcp, (0.0, 0.0, 1.0)))
            for time_s in (0.0, 0.03, 0.06):
                update_start = time.perf_counter()
                warp_runtime.update(time_s, True, flow_frame, actual_tool_origin_m=nozzle, actual_tool_axes=tool_axes)
                warp_update_seconds.append(time.perf_counter() - update_start)
            _write_warp_points(warp_points, warp_runtime.active_world_positions())
            _write_plume_guide(plume_guide, origin=nozzle, frame=flow_frame, stand_off_m=model.stand_off_m, spray_on=True)

        viewport_window = __import__("omni.kit.viewport.utility", fromlist=["create_viewport_window"]).create_viewport_window(name="CFDFlowViewport", width=RESOLUTION[0], height=RESOLUTION[1], camera_path="/World/Cameras/process_painting")
        viewport = viewport_window.viewport_api if viewport_window else __import__("omni.kit.viewport.utility", fromlist=["get_active_viewport"]).get_active_viewport()
        if viewport is None:
            raise RuntimeError("native viewport is unavailable")
        viewport.camera_path = "/World/Cameras/process_painting"
        viewport.resolution = RESOLUTION
        hud_overlay = _create_hud(viewport_window, mode=args.view, slice_y_m=geometry["slice_y_m"])
        _set_mode(args.view, geometry, warp_points, plume_guide, overlay, hud_overlay)

        cfd_update_seconds: list[float] = []
        for _ in range(30):
            update_start = time.perf_counter()
            _set_flow_transform(geometry["transform_op"], flow_frame)
            simulation_app.update()
            cfd_update_seconds.append(time.perf_counter() - update_start)

        outputs: dict[str, str] = {}
        capture_modes = [args.view]
        if args.capture_stills:
            capture_modes = ["process", "flow", "combined", "slice", "result"]
        output_map = {
            "process": MEDIA_DIR / "isaac_cfd_process_frame.png",
            "flow": MEDIA_DIR / "isaac_cfd_flow_field.png",
            "combined": MEDIA_DIR / "isaac_cfd_warp_combined.png",
            "slice": MEDIA_DIR / "isaac_cfd_flow_slice.png",
            "result": MEDIA_DIR / "isaac_cfd_result_frame.png",
        }
        for mode in capture_modes:
            _set_mode(mode, geometry, warp_points, plume_guide, overlay, hud_overlay)
            viewport.camera_path = cfd_camera_path if mode in {"flow", "combined", "slice"} else "/World/Cameras/process_painting"
            # Camera and visibility changes are asynchronous in the headless
            # viewport; allow the render product to settle before capture so
            # each still corresponds to its requested view mode.
            for _ in range(8):
                simulation_app.update()
            target = output_map[mode]
            _capture(viewport, target, simulation_app)
            _annotate_capture(target, mode, field, geometry, model.stand_off_m, measured_frame.incidence_u_deg)
            outputs[mode] = target.relative_to(ROOT).as_posix()
        if warp_runtime is not None:
            warp_runtime.shutdown(0.06)
        hud_overlay.destroy()
        full_metrics = json.loads(FULL_PANEL_METRICS.read_text(encoding="utf-8")) if FULL_PANEL_METRICS.exists() else {}
        cfd_metrics = {
            "schema_version": "isaac_cfd_flow_visualization_v1",
            "status": "ISAAC_CFD_FLOW_VISUALIZATION_VALIDATED",
            "view_modes": {"process": "robot + Warp plume + minimal HUD", "flow": "robot + CFD vectors + slice + streamlines", "combined": "robot + CFD vectors/streamlines + Warp plume", "result": "final WFT overlay + summary"},
            "field_artifact": {"npz": "models/openfoam_flow_field_7p5deg_v1.npz", "manifest": "models/openfoam_flow_field_7p5deg_v1.json", "artifact_sha256": field.artifact_sha256, "source_case": field.source_case, "source_u_sha256": field.source_u_sha256, "source_c_sha256": field.source_c_sha256, "openfoam_version": field.openfoam_version, "latest_time_s": field.latest_time_s, "angle_deg": field.angle_deg, "dimensions_nx_ny_nz": list(field.dimensions_nx_ny_nz), "cells": field.cell_count, "bounds_min_m": list(field.bounds_min_m), "bounds_max_m": list(field.bounds_max_m), "spacing_m": list(field.spacing_m), "speed_magnitude_m_s": field.stats},
            "geometry": {"vector_count": geometry["vector_count"], "slice_plane": f"fan-major / spray-axis, y≈{geometry['slice_y_m']:.6f} m", "slice_vertices": geometry["slice_vertices"], "slice_cells": geometry["slice_cells"], "slice_speed_m_s": geometry["slice_speed_stats"], "streamline_count": geometry["streamline_count"], "streamline_seed_rule": geometry["streamline_seed_rule"], "mapping": "+X→u fan-major, +Y→v travel/fan-minor, +Z→w inward process direction", "update_method": "precomputed local geometry + rigid transform per render update"},
            "process_frame": {"stand_off_m": model.stand_off_m, "incidence_deg": float(measured_frame.incidence_u_deg), "origin_world_m": nozzle.tolist()},
            "performance": {"cfd_visualization_update_s": _summary(cfd_update_seconds), "warp_update_s": _summary(warp_update_seconds), "s2_update_s": {"source": "validated full-panel metrics; no S2 step executed by this visual-only capture", **(full_metrics.get("performance", {}).get("s2_update", {}) or {})}, "combined_view_wall_seconds": time.perf_counter() - wall_start},
            "media": outputs,
            "semantics": {"cfd_solve": "offline OpenFOAM v2606 reference field", "mapping": "quasi-steady local tangent frame", "online_cfd": False, "warp_visual": "existing explicit W1.3 droplet transport; warm/orange", "cfd_visual": "cool cyan/blue vectors, slice, and streamlines"},
        }
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "cfd_flow_visualization_metrics.json").write_text(json.dumps(cfd_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"CFD_FLOW_PASS vectors={geometry['vector_count']} slice_vertices={geometry['slice_vertices']} streamlines={geometry['streamline_count']} artifact={field.artifact_sha256}", flush=True)
        print(f"CFD_MEDIA {json.dumps(outputs, sort_keys=True)}", flush=True)
        return 0
    except Exception as exc:
        print(f"CFD_FLOW_ERROR {type(exc).__name__}: {exc}", flush=True)
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
    parser.add_argument("--view", choices=("process", "flow", "combined", "result"), default="combined")
    parser.add_argument("--capture-stills", action="store_true", help="capture the flow, combined, and slice technical stills")
    parser.add_argument("--frame-time-s", type=float, default=10.0, help="validated plan time used for the local process frame")
    parser.add_argument("--vector-count", type=int, default=512)
    parser.add_argument("--streamline-count", type=int, default=25)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
