#!/usr/bin/env python3
"""16:9 Widescreen Marble Music Template Library (Decoupled from 9:16 Vertical Pipeline).

Provides three distinct, mathematically zero-clipping 16:9 concert choreographies:
1. `serpentine_descent`: Continuous T4-Marbles-style Left<->Right downward serpentine track
   with stepped 3D walnut terraces, sweeping switchback chutes, and close-follow chase cam.
2. `diamond_loop`: Seamless 4-phrase 3D Diamond Loop where descending edges fall naturally
   under gravity and ascending edges launch off animated brass spring-piston actuators,
   closing back seamlessly to the top apex.
3. `horological_wheel`: Rotating Clockwork Music-Box Turntable where the whole radial
   instrument wheel turns 360 deg per cycle while the marble always bounces downward
   inside the right-hand gravity quadrant, returning cleanly via an outer brass scoop.
"""

from __future__ import annotations

import json
import math
import os

import bpy
import bmesh
from mathutils import Euler, Matrix, Quaternion, Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
LIB_BLEND = os.path.join(ASSETS_DIR, "instrument_library_v2.blend")
MANIFEST_JSON = os.path.join(ASSETS_DIR, "instrument_manifest_v2.json")

FPS = 60
GRAVITY = 14.0
MARBLE_RADIUS = 0.092
FLIGHT_Y = -0.10
SUB_Y_OFFSET = 0.11       # Pushes v3.0 bodies strictly behind flight lane (Y >= +0.14)
ANVIL_LEN = 0.28
ANVIL_WID_Y = 0.18
ANVIL_THICK = 0.034
RAIL_HALF_GAP = 0.052
RAIL_PIPE_R = 0.012
INST_DISPLAY_TILT_DEG = 20.0

LEGACY_TRACK_TO_GM = {
    1: 115, 2: 0, 3: 13, 4: 46, 5: 12, 6: 24,
    7: 14, 8: 73, 9: 56, 10: 40, 11: 21, 12: 80,
}

_BOX_MESH_CACHE: dict[tuple[float, float, float, float], bpy.types.Mesh] = {}


# ---------------------------------------------------------------------------
# Basic Scene, Material & Geometry Helpers
# ---------------------------------------------------------------------------

def set_socket(node, name: str, value) -> None:
    if node is not None and name in node.inputs:
        node.inputs[name].default_value = value


def get_eevee_engine() -> str:
    available = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    for cand in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if cand in available:
            return cand
    return next(iter(available))


def clear_scene() -> None:
    _BOX_MESH_CACHE.clear()
    bpy.ops.wm.read_factory_settings(use_empty=True)


def get_or_create_collection(name: str, parent=None) -> bpy.types.Collection:
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    if parent is None:
        bpy.context.scene.collection.children.link(col)
    else:
        parent.children.link(col)
    return col


def make_pbr_material(
    name: str,
    base_color: tuple[float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.42,
    coat: float = 0.0,
    emit_color: tuple[float, float, float] | None = None,
    emit_strength: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    set_socket(bsdf, "Base Color", (*base_color[:3], 1.0))
    set_socket(bsdf, "Metallic", metallic)
    set_socket(bsdf, "Roughness", roughness)
    set_socket(bsdf, "Coat Weight", coat)
    if emit_color is not None and emit_strength > 0.0:
        set_socket(bsdf, "Emission Color", (*emit_color[:3], 1.0))
        set_socket(bsdf, "Emission Strength", emit_strength)
    return mat


def create_beveled_box_mesh(name: str, length: float, width_y: float, thickness: float, bevel_w: float = 0.008) -> bpy.types.Mesh:
    key = (round(length, 4), round(width_y, 4), round(thickness, 4), round(bevel_w, 4))
    if key in _BOX_MESH_CACHE:
        return _BOX_MESH_CACHE[key].copy()
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(length, width_y, thickness), verts=bm.verts)
    bmesh.ops.bevel(
        bm, geom=list(bm.edges), offset=bevel_w, segments=3, profile=0.5, affect="EDGES", clamp_overlap=True
    )
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    _BOX_MESH_CACHE[key] = mesh
    return mesh.copy()


def create_tube_polyline(name: str, pts: list[list[float]], radius: float, mat, collection) -> bpy.types.Object:
    curve = bpy.data.curves.new(name + "_Curve", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 18
    curve.bevel_depth = radius
    curve.bevel_resolution = 8
    curve.use_fill_caps = True
    sp = curve.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for k, p in enumerate(pts):
        sp.points[k].co = (p[0], p[1], p[2], 1.0)
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def create_cylinder_between(name: str, p_a: Vector, p_b: Vector, radius: float, mat, collection, verts: int = 16):
    diff = p_b - p_a
    dist = diff.length
    if dist < 1e-4:
        return None
    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=verts, radius1=radius, radius2=radius, depth=dist
    )
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = 0.5 * (p_a + p_b)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(diff.normalized())
    obj.data.materials.append(mat)
    return obj


# ---------------------------------------------------------------------------
# Static v3.0 Instrument Asset Loader
# ---------------------------------------------------------------------------

def load_static_instrument_library(
    lib_path: str = LIB_BLEND,
    manifest_path: str = MANIFEST_JSON,
) -> tuple[dict[int, dict], dict[str, bpy.types.Material], dict]:
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Missing instrument library asset: {lib_path}")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    by_prog_manifest = manifest.get("by_program", {})

    with bpy.data.libraries.load(lib_path, link=False) as (data_from, data_to):
        data_to.collections = [c for c in data_from.collections if c.startswith("Lib_")]
        data_to.materials = [m for m in data_from.materials if m.startswith("LibMat_")]

    mats = {m.name: m for m in bpy.data.materials if m.name.startswith("LibMat_")}
    for m in mats.values():
        m.use_fake_user = True

    templates: dict[int, dict] = {}
    temp_cols = []
    for col in list(bpy.data.collections):
        if not col.name.startswith("Lib_"):
            continue
        temp_cols.append(col)
        parts = col.name.split("_")
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].startswith("GM") and parts[2][2:].isdigit():
            t_idx = int(parts[1])
            gm_prog = int(parts[2][2:])
        elif len(parts) >= 2 and parts[1].isdigit():
            t_idx = int(parts[1])
            gm_prog = LEGACY_TRACK_TO_GM.get(t_idx, t_idx)
        else:
            continue

        core_obj = None
        sub_objs = []
        for obj in col.objects:
            if obj.data is not None:
                obj.data.use_fake_user = True
            if obj.name.startswith("LibCore_") or obj.name.endswith("_Core"):
                core_obj = obj
            else:
                sub_objs.append(obj)

        m_info = by_prog_manifest.get(str(gm_prog), {})
        entry = {
            "index": t_idx,
            "gm_program": gm_prog,
            "name": m_info.get("name", col.name),
            "anim_family": m_info.get("anim_family", "generic_strike"),
            "glow_color": tuple(m_info.get("glow_color", (0.95, 0.72, 0.28))),
            "subs": [
                (
                    o.name,
                    o.data,
                    list(o.data.materials) if o.data else [],
                    o.location.copy(),
                    o.rotation_euler.copy(),
                    o.scale.copy(),
                )
                for o in sub_objs
                if o.data is not None
            ],
        }
        templates[gm_prog] = entry
        templates.setdefault(-1000 - t_idx, entry)

    for col in temp_cols:
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)

    return templates, mats, manifest


EXPLICIT_INSTRUMENT_ALIASES = {
    "piano": 0,
    "acoustic grand piano": 0,
    "grand piano": 0,
    "marimba": 12,
    "xylophone": 13,
    "tubular bells": 14,
    "guitar": 24,
    "acoustic guitar": 24,
    "harp": 46,
    "castanets": 115,
    "woodblock": 115,
}


def resolve_gm_program(instrument_name: str, track_idx: int, manifest: dict) -> int:
    name_l = instrument_name.strip().lower()
    if name_l in EXPLICIT_INSTRUMENT_ALIASES:
        return EXPLICIT_INSTRUMENT_ALIASES[name_l]
    aliases = manifest.get("name_aliases", {})
    if name_l in aliases:
        return int(aliases[name_l])
    for gp, info in manifest.get("by_program", {}).items():
        if str(info.get("name", "")).strip().lower() == name_l:
            return int(gp)
    return int(LEGACY_TRACK_TO_GM.get(int(track_idx), 0))


# ---------------------------------------------------------------------------
# Smooth Splines & Ballistic Kinematics
# ---------------------------------------------------------------------------

def hermite_curve(p0: Vector, m0: Vector, p1: Vector, m1: Vector, steps: int) -> list[Vector]:
    pts = []
    for s in range(steps + 1):
        u = s / steps
        u2, u3 = u * u, u * u * u
        pts.append(
            p0 * (2 * u3 - 3 * u2 + 1)
            + m0 * (u3 - 2 * u2 + u)
            + p1 * (-2 * u3 + 3 * u2)
            + m1 * (u3 - u2)
        )
    return pts


def catmull_rom_chain(waypoints: list[Vector], samples_per_seg: int = 18) -> list[Vector]:
    if len(waypoints) < 2:
        return list(waypoints)
    pts_ext = [waypoints[0] - (waypoints[1] - waypoints[0])] + list(waypoints) + [
        waypoints[-1] + (waypoints[-1] - waypoints[-2])
    ]
    out: list[Vector] = []
    for i in range(1, len(pts_ext) - 2):
        p0, p1, p2, p3 = pts_ext[i - 1], pts_ext[i], pts_ext[i + 1], pts_ext[i + 2]
        seg = hermite_curve(p1, 0.5 * (p2 - p0), p2, 0.5 * (p3 - p1), samples_per_seg)
        if out:
            seg = seg[1:]
        out.extend(seg)
    return out


def resample_by_arc_length(pts: list[Vector], n_out: int) -> list[Vector]:
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + (pts[i] - pts[i - 1]).length)
    total = max(1e-6, cum[-1])
    out: list[Vector] = []
    j = 0
    for k in range(n_out):
        target = total * (k / max(1, n_out - 1))
        while j + 1 < len(cum) - 1 and cum[j + 1] < target:
            j += 1
        seg_l = max(1e-8, cum[j + 1] - cum[j])
        frac = max(0.0, min(1.0, (target - cum[j]) / seg_l))
        out.append(pts[j].lerp(pts[j + 1], frac))
    return out


def eval_arc(arc_pts: list[Vector], u_frac: float) -> Vector:
    idx_f = max(0.0, min(len(arc_pts) - 1.0, u_frac * (len(arc_pts) - 1)))
    i0 = int(math.floor(idx_f))
    i1 = min(len(arc_pts) - 1, i0 + 1)
    return arc_pts[i0].lerp(arc_pts[i1], idx_f - i0)


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def solve_ballistic_pos(p0: Vector, p1: Vector, dt: float, u: float, g: float = GRAVITY) -> Vector:
    t = u * dt
    vz0 = ((p1.z - p0.z) / max(1e-5, dt)) + 0.5 * g * dt
    vx = (p1.x - p0.x) / max(1e-5, dt)
    vy = (p1.y - p0.y) / max(1e-5, dt)
    return Vector((
        p0.x + vx * t,
        p0.y + vy * t,
        p0.z + vz0 * t - 0.5 * g * t * t,
    ))


def _eval_segment(seg: dict, t: float, g: float = GRAVITY) -> Vector:
    t0, t1 = float(seg["t0"]), float(seg["t1"])
    dt = max(1e-5, t1 - t0)
    stype = seg["type"]
    if stype == "ballistic":
        u = max(0.0, min(1.0, (t - t0) / dt))
        return solve_ballistic_pos(Vector(seg["p0"]), Vector(seg["p1"]), dt, u, g=g)
    if stype == "rail_drop":
        rail_end = float(seg["rail_end_t"])
        arc = [Vector(p) for p in seg["arc_pts"]]
        if t <= rail_end:
            u = smoothstep((t - t0) / max(1e-5, rail_end - t0))
            u_acc = 0.25 * u + 0.75 * (u * u)
            return eval_arc(arc, u_acc)
        u_drop = (t - rail_end) / max(1e-5, t1 - rail_end)
        return solve_ballistic_pos(arc[-1], Vector(seg["p_target"]), t1 - rail_end, u_drop, g=g)
    if stype == "loop_chute":
        ldt = float(seg["launch_dt"])
        ddt = float(seg["drop_dt"])
        arc = [Vector(p) for p in seg["arc_pts"]]
        t_enter = t0 + ldt
        t_exit = t1 - ddt
        if t <= t_enter:
            u = (t - t0) / max(1e-5, ldt)
            return solve_ballistic_pos(Vector(seg["p0"]), arc[0], ldt, u, g=g)
        if t >= t_exit:
            u = (t - t_exit) / max(1e-5, ddt)
            return solve_ballistic_pos(arc[-1], Vector(seg["p1"]), ddt, u, g=g)
        u = (t - t_enter) / max(1e-5, t_exit - t_enter)
        return eval_arc(arc, 0.85 * u + 0.15 * smoothstep(u))
    if stype == "launch_rail":
        ldt = float(seg["launch_dt"])
        arc = [Vector(p) for p in seg["arc_pts"]]
        t_enter = t0 + ldt
        if t <= t_enter:
            u = (t - t0) / max(1e-5, ldt)
            return solve_ballistic_pos(Vector(seg["p0"]), arc[0], ldt, u, g=g)
        u = (t - t_enter) / max(1e-5, t1 - t_enter)
        return eval_arc(arc, min(1.0, u * 0.92))
    return Vector(seg.get("p0", [0.0, 0.0, 0.0]))


def build_parallel_rail_geometry(
    name_prefix: str,
    spline_pts: list[Vector],
    rail_mat,
    brass_mat,
    dark_mat,
    col_rails: bpy.types.Collection,
) -> dict:
    """Twin brass gauge rails cradling bottom belt of marble + coaxial flared end cuffs."""
    left_pts, right_pts, tie_coords = [], [], []
    n_pts = len(spline_pts)
    for idx, p in enumerate(spline_pts):
        p_prev = spline_pts[max(0, idx - 1)]
        p_next = spline_pts[min(n_pts - 1, idx + 1)]
        tang = p_next - p_prev
        tang = tang.normalized() if tang.length > 1e-6 else Vector((1.0, 0.0, 0.0))
        binorm = tang.cross(Vector((0.0, 1.0, 0.0)))
        binorm = binorm.normalized() if binorm.length > 1e-6 else Vector((0.0, 0.0, 1.0))
        # Ensure normal faces outward/upward relative to horizontal movement on flatter slopes
        cradle = p - binorm * (MARBLE_RADIUS * 0.72)
        left_pts.append([cradle.x, cradle.y - RAIL_HALF_GAP, cradle.z])
        right_pts.append([cradle.x, cradle.y + RAIL_HALF_GAP, cradle.z])
        if idx % 7 == 3 or idx == 0 or idx == n_pts - 1:
            tie_coords.append((Vector(left_pts[-1]), Vector(right_pts[-1])))

    create_tube_polyline(f"{name_prefix}_RailFront", left_pts, RAIL_PIPE_R, rail_mat, col_rails)
    create_tube_polyline(f"{name_prefix}_RailBack", right_pts, RAIL_PIPE_R, rail_mat, col_rails)
    for k, (pa, pb) in enumerate(tie_coords):
        tie_mat = brass_mat if k % 3 == 0 else dark_mat
        create_cylinder_between(f"{name_prefix}_Tie_{k:02d}", pa, pb, RAIL_PIPE_R * 0.60, tie_mat, col_rails, verts=10)

    for tag, idx_end, idx_nbr in (("EntryCollar", 0, 1), ("DropNozzle", -1, -2)):
        c_pos = 0.5 * (Vector(left_pts[idx_end]) + Vector(right_pts[idx_end]))
        tang = spline_pts[idx_nbr] - spline_pts[idx_end]
        if idx_end == -1:
            tang = -tang
        tang = tang.normalized() if tang.length > 1e-5 else Vector((0.0, 0.0, -1.0))
        tmesh = bpy.data.meshes.new(f"{name_prefix}_{tag}Mesh")
        tbm = bmesh.new()
        bmesh.ops.create_cone(
            tbm, cap_ends=False, cap_tris=False, segments=20,
            radius1=MARBLE_RADIUS * 0.70, radius2=MARBLE_RADIUS * 0.86, depth=0.040,
        )
        bmesh.ops.solidify(tbm, geom=list(tbm.faces), thickness=0.005)
        tbm.to_mesh(tmesh)
        tbm.free()
        for p in tmesh.polygons:
            p.use_smooth = True
        tobj = bpy.data.objects.new(f"{name_prefix}_{tag}", tmesh)
        col_rails.objects.link(tobj)
        tobj.location = c_pos
        tobj.rotation_mode = "QUATERNION"
        tobj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(tang)
        tobj.data.materials.append(brass_mat)

    return {
        "name": name_prefix,
        "left": left_pts,
        "right": right_pts,
        "center": [[p.x, p.y, p.z] for p in spline_pts],
    }


# ---------------------------------------------------------------------------
# Anti-Clipping Jitter Solver & Dynamic Strict 3D Verifier
# ---------------------------------------------------------------------------

def _measure_template_footprint(tpl: dict, inst_scale: float = 1.0) -> tuple[list[float], list[float]]:
    """Measure local (min_xyz, max_xyz) of a v3.0 instrument mounted behind SUB_Y_OFFSET at tilt."""
    xs, ys, zs = [], [], []
    tilt_m = Matrix.Rotation(math.radians(INST_DISPLAY_TILT_DEG), 3, "X")
    piv = Vector((0.0, SUB_Y_OFFSET, 0.0))
    for _sn, sdata, _sm, sloc, srot, sscale in tpl["subs"]:
        if not hasattr(sdata, "vertices") or len(sdata.vertices) == 0:
            continue
        sub_rot_m = Euler(srot, "XYZ").to_matrix()
        stride = max(1, len(sdata.vertices) // 32)
        for vi in range(0, len(sdata.vertices), stride):
            v = sdata.vertices[vi]
            vs = Vector((v.co.x * sscale.x, v.co.y * sscale.y, v.co.z * sscale.z)) * inst_scale
            p = piv + (tilt_m @ (sloc * inst_scale + (sub_rot_m @ vs)))
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)
    if not xs:
        return [-0.25, 0.08, -0.10], [0.25, 0.35, 0.25]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def resolve_station_jitter(
    stations: list[dict],
    templates: dict[int, dict],
    manifest: dict,
    min_gap: float = 0.12,
    allow_z_jitter: bool = True,
) -> None:
    """Iteratively jitter (X, Z) using true rotated AABBs so zero v3.0 bodies or anvils overlap."""
    n = len(stations)
    for _it in range(42):
        aabbs = compute_station_aabbs_at_transform(stations, templates, manifest)
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                g = aabb_gap(aabbs[i]["min"], aabbs[i]["max"], aabbs[j]["min"], aabbs[j]["max"])
                if g < min_gap:
                    moved = True
                    ci, cj = Vector(stations[i]["center"]), Vector(stations[j]["center"])
                    dx_pair = cj.x - ci.x
                    dz_pair = cj.z - ci.z
                    # Push apart primarily along dominant separation axis in XZ (with clean jitter)
                    if allow_z_jitter and abs(dz_pair) > abs(dx_pair) * 0.75:
                        sx_dir = -1.0 if ci.x <= cj.x else 1.0
                        sz_dir = -1.0 if ci.z <= cj.z else 1.0
                        shift_x, shift_z = 0.045, 0.075
                    else:
                        sx_dir = -1.0 if ci.x <= cj.x else 1.0
                        sz_dir = (1.0 if ci.z >= cj.z else -1.0) if allow_z_jitter else 0.0
                        shift_x, shift_z = 0.080, (0.045 if allow_z_jitter else 0.0)
                    for idx_s, sgn in ((i, 1.0), (j, -1.0)):
                        stations[idx_s]["center"][0] += sgn * sx_dir * shift_x
                        stations[idx_s]["center"][2] += sgn * sz_dir * shift_z
                        stations[idx_s]["contact"][0] = stations[idx_s]["center"][0]
                        stations[idx_s]["contact"][2] = stations[idx_s]["center"][2] + ANVIL_THICK * 0.5
        if not moved:
            break


def compute_station_aabbs_at_transform(
    stations: list[dict],
    templates: dict[int, dict],
    manifest: dict,
    world_rot: Matrix | None = None,
    world_origin: Vector | None = None,
) -> list[dict]:
    """Compute exact 3D world-space AABBs (split anvil_box + inst_box) under optional wheel rotation."""
    aabbs = []
    tilt_m = Matrix.Rotation(math.radians(INST_DISPLAY_TILT_DEG), 3, "X")
    piv = Vector((0.0, SUB_Y_OFFSET, 0.0))

    for st in stations:
        gm = resolve_gm_program(st["instrument"], st["track_idx"], manifest)
        tpl = templates.get(gm) or templates.get(-1000 - st["track_idx"]) or next(iter(templates.values()))
        u = Vector(st["axis_u"]).normalized()
        v = Vector(st["axis_v"]).normalized()
        n = Vector(st["axis_n"]).normalized()
        st_rot = Matrix(((u.x, v.x, n.x), (u.y, v.y, n.y), (u.z, v.z, n.z)))
        st_pos = Vector(st["center"])
        sc = float(st.get("inst_scale", 1.0))

        wheel_upright = bool(st.get("wheel_upright")) and world_rot is not None
        wc = world_origin or Vector((0, 0, 0))
        rotated_center = (wc + (world_rot @ (st_pos - wc))) if wheel_upright else st_pos

        def to_world(local_p: Vector) -> Vector:
            if wheel_upright:
                return rotated_center + (st_rot @ local_p)
            p = st_pos + (st_rot @ local_p)
            if world_rot is not None:
                p = wc + (world_rot @ (p - wc))
            return p

        hx = float(st.get("anvil_len", ANVIL_LEN)) * 0.5
        hy, hz = ANVIL_WID_Y * 0.5, ANVIL_THICK * 0.5
        anvil_pts = [
            to_world(Vector((sx * hx, sy * hy, sz * hz)))
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ]

        inst_pts = []
        for _sn, sdata, _sm, sloc, srot, sscale in tpl["subs"]:
            if not hasattr(sdata, "vertices") or len(sdata.vertices) == 0:
                continue
            sub_rot_m = Euler(srot, "XYZ").to_matrix()
            stride = max(1, len(sdata.vertices) // 48)
            for vi in range(0, len(sdata.vertices), stride):
                vert = sdata.vertices[vi]
                vs = Vector((vert.co.x * sscale.x, vert.co.y * sscale.y, vert.co.z * sscale.z)) * sc
                loc_inst = piv + (tilt_m @ (sloc * sc + (sub_rot_m @ vs)))
                inst_pts.append(to_world(loc_inst))

        all_pts = anvil_pts + inst_pts
        aabbs.append({
            "station_id": st["id"],
            "label": f"{st['instrument']}({st['pitch_name']})",
            "anvil_min": [min(p.x for p in anvil_pts), min(p.y for p in anvil_pts), min(p.z for p in anvil_pts)],
            "anvil_max": [max(p.x for p in anvil_pts), max(p.y for p in anvil_pts), max(p.z for p in anvil_pts)],
            "inst_min": [min(p.x for p in inst_pts), min(p.y for p in inst_pts), min(p.z for p in inst_pts)] if inst_pts else [0, 0, 0],
            "inst_max": [max(p.x for p in inst_pts), max(p.y for p in inst_pts), max(p.z for p in inst_pts)] if inst_pts else [0, 0, 0],
            "min": [min(p.x for p in all_pts), min(p.y for p in all_pts), min(p.z for p in all_pts)],
            "max": [max(p.x for p in all_pts), max(p.y for p in all_pts), max(p.z for p in all_pts)],
        })
    return aabbs


def point_to_aabb_distance(p: Vector, bmin: list[float], bmax: list[float]) -> float:
    dx = max(bmin[0] - p.x, 0.0, p.x - bmax[0])
    dy = max(bmin[1] - p.y, 0.0, p.y - bmax[1])
    dz = max(bmin[2] - p.z, 0.0, p.z - bmax[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def aabb_gap(a_min: list[float], a_max: list[float], b_min: list[float], b_max: list[float]) -> float:
    dx = max(b_min[0] - a_max[0], a_min[0] - b_max[0], 0.0)
    dy = max(b_min[1] - a_max[1], a_min[1] - b_max[1], 0.0)
    dz = max(b_min[2] - a_max[2], a_min[2] - b_max[2], 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _ensure_rail_clearance(
    wp: list[Vector],
    aabbs: list[dict],
    min_clear: float = 0.22,
) -> list[Vector]:
    """Push rail waypoints cleanly outside every station AABB in XZ before spline resampling."""
    out = [p.copy() for p in wp]
    for _it in range(16):
        changed = False
        arc = resample_by_arc_length(catmull_rom_chain(out, 10), 60)
        worst_idx, worst_d, worst_box = -1, min_clear, None
        for idx, p in enumerate(arc):
            for b in aabbs:
                d = point_to_aabb_distance(p, b["min"], b["max"])
                if d < worst_d:
                    worst_d = d
                    worst_box = b
                    worst_idx = min(len(out) - 1, round(idx * (len(out) - 1) / max(1, len(arc) - 1)))
        if worst_box is None:
            break
        changed = True
        bmin, bmax = worst_box["min"], worst_box["max"]
        bc = Vector((0.5 * (bmin[0] + bmax[0]), FLIGHT_Y, 0.5 * (bmin[2] + bmax[2])))
        w = out[worst_idx]
        push = Vector((w.x - bc.x, 0.0, w.z - bc.z))
        if push.length < 0.05:
            push = Vector((0.0, 0.0, 1.0))
        else:
            push.normalize()
        # Keep endpoints comfortably above rather than diving below anvils
        if worst_idx in (0, len(out) - 1):
            push = (push + Vector((0.0, 0.0, 1.2))).normalized()
        out[worst_idx] = w + push * 0.24
        if not changed:
            break
    return out


def verify_zero_intersections(
    plan: dict,
    templates: dict[int, dict],
    manifest: dict,
    rails_geom: list[dict],
) -> None:
    """Strictly verify zero overlaps across static stations, fixed rails, and dynamic trajectories."""
    stations = plan["stations"]
    wheel_cfg = plan.get("wheel")
    aabbs0 = compute_station_aabbs_at_transform(stations, templates, manifest)

    # 1. Pairwise station overlap check (in their shared frame)
    min_pair_gap = 999.0
    for i in range(len(aabbs0)):
        for j in range(i + 1, len(aabbs0)):
            g = aabb_gap(aabbs0[i]["min"], aabbs0[i]["max"], aabbs0[j]["min"], aabbs0[j]["max"])
            min_pair_gap = min(min_pair_gap, g)
            if g < 0.04:
                raise RuntimeError(
                    f"[INTERSECTION:{plan['template']}] Stations {aabbs0[i]['label']} & {aabbs0[j]['label']} overlap (gap={g:.4f}m)"
                )

    # 2. Fixed rails vs stations (across full 360 wheel rotation if rotating!)
    min_rail_clear = 999.0
    angles = [2.0 * math.pi * k / 36.0 for k in range(36)] if wheel_cfg else [0.0]
    for ang in angles:
        if wheel_cfg:
            wrot = Matrix.Rotation(ang, 3, "Y")
            worig = Vector(wheel_cfg["center"])
            boxes = compute_station_aabbs_at_transform(stations, templates, manifest, world_rot=wrot, world_origin=worig)
        else:
            boxes = aabbs0
        for rg in rails_geom:
            for side in ("left", "right"):
                for pt in rg[side][::2]:
                    pv = Vector(pt)
                    for b in boxes:
                        d_a = point_to_aabb_distance(pv, b["anvil_min"], b["anvil_max"]) - RAIL_PIPE_R
                        d_i = point_to_aabb_distance(pv, b["inst_min"], b["inst_max"]) - RAIL_PIPE_R
                        d = min(d_a, d_i)
                        min_rail_clear = min(min_rail_clear, d)
                        if d < 0.03:
                            raise RuntimeError(
                                f"[INTERSECTION:{plan['template']}] Rail {rg['name']} clips {b['label']} at {pt} (ang={math.degrees(ang):.0f}d, clear={d:.4f}m)"
                            )

    # 3. Marble trajectory vs all non-target anvils & all instrument bodies
    min_m_clear = 999.0
    bounces = plan["bounces"]
    dur_s = float(plan.get("duration_s", 0.0))
    is_loop = plan["template"] in ("diamond_loop", "horological_wheel")
    for smp in plan["dense_samples"][::2]:
        mp = Vector(smp["position"])
        t = float(smp["t"])
        if wheel_cfg:
            ang = float(wheel_cfg["omega"]) * t
            boxes = compute_station_aabbs_at_transform(
                stations, templates, manifest,
                world_rot=Matrix.Rotation(ang, 3, "Y"),
                world_origin=Vector(wheel_cfg["center"]),
            )
        else:
            boxes = aabbs0
        near = {
            b["station_id"] for b in bounces
            if abs(float(b["t"]) - t) <= 0.14
            or (is_loop and abs(float(b["t"]) + dur_s - t) <= 0.14)
        }
        for b in boxes:
            di = point_to_aabb_distance(mp, b["inst_min"], b["inst_max"]) - MARBLE_RADIUS
            min_m_clear = min(min_m_clear, di)
            if di < 0.015:
                raise RuntimeError(
                    f"[INTERSECTION:{plan['template']}] Marble clips body {b['label']} at t={t:.3f}s (clear={di:.4f}m)"
                )
            if b["station_id"] not in near:
                da = point_to_aabb_distance(mp, b["anvil_min"], b["anvil_max"]) - MARBLE_RADIUS
                if da < 0.015:
                    raise RuntimeError(
                        f"[INTERSECTION:{plan['template']}] Marble clips non-target anvil {b['label']} at t={t:.3f}s (clear={da:.4f}m)"
                    )

    print(
        f"[ZeroClip OK · {plan['template']}] stations={len(stations)}, rails={len(rails_geom)} | "
        f"min_station_gap={min_pair_gap:.3f}m, min_rail_clear={min_rail_clear:.3f}m, "
        f"min_marble_clear={min_m_clear:.3f}m"
    )


# ---------------------------------------------------------------------------
# Note Clustering Helpers (Share Station on Fast Repeat Notes <= 0.24s)
# ---------------------------------------------------------------------------

def load_notes_slice(notes_json_path: str, max_note_t: float = 10.0) -> list[dict]:
    with open(notes_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    notes = raw["notes"] if isinstance(raw, dict) and "notes" in raw else raw
    return sorted(
        [n for n in notes if float(n["start_time"]) <= max_note_t + 1e-6],
        key=lambda n: float(n["start_time"]),
    )


def group_notes_into_phrases(notes: list[dict], gap_thresh: float = 0.65) -> list[list[tuple[int, dict]]]:
    phrases: list[list[tuple[int, dict]]] = []
    cur: list[tuple[int, dict]] = []
    for idx, n in enumerate(notes):
        if cur and (float(n["start_time"]) - float(cur[-1][1]["start_time"])) > gap_thresh:
            phrases.append(cur)
            cur = []
        cur.append((idx, n))
    if cur:
        phrases.append(cur)
    return phrases


def _cluster_phrase_stations(phrase: list[tuple[int, dict]]) -> list[list[tuple[int, dict]]]:
    """Rule 4: 1 note = 1 dedicated board. Never strike the same board twice."""
    return [[item] for item in phrase]


def _update_bounces_from_stations(stations: list[dict], bounces: list[dict]) -> None:
    """Refresh bounce world impact points after anti-clipping jitter moved station centers."""
    st_by_id = {s["id"]: s for s in stations}
    for b in bounces:
        st = st_by_id[b["station_id"]]
        n_dir = Vector(st["axis_n"]).normalized()
        offset_u = float(b.get("anvil_hit_u", 0.0)) * Vector(st["axis_u"]).normalized()
        c_pos = Vector(st["center"]) + offset_u + n_dir * (ANVIL_THICK * 0.5 + MARBLE_RADIUS)
        b["marble_pos"] = [c_pos.x, c_pos.y, c_pos.z]
        b["station_center"] = list(st["center"])
        # Sync station's hit record contact
        for h in st["hits"]:
            if h["note_index"] == b["note_index"]:
                h["contact"] = [c_pos.x, c_pos.y, c_pos.z]


def _finalize_trajectory(
    template_name: str,
    stations: list[dict],
    bounces: list[dict],
    rails_plan: list[dict],
    segments: list[dict],
    t_end: float,
    pre_roll: float,
    wheel_cfg: dict | None = None,
    camera_mode: str = "chase",
) -> dict:
    total_frames = int(math.ceil(t_end * FPS))
    dense_samples: list[dict] = []
    for sub_i in range(total_frames * 2 + 1):
        t = min(t_end, sub_i / (FPS * 2.0))
        seg = next((s for s in segments if s["t0"] - 1e-7 <= t <= s["t1"] + 1e-7), segments[-1])
        pos = _eval_segment(seg, t)
        dense_samples.append({
            "frame_f": t * FPS,
            "t": round(t, 5),
            "position": [round(pos.x, 5), round(pos.y, 5), round(pos.z, 5)],
        })
    return {
        "schema": "concert_16x9_template_v2",
        "template": template_name,
        "camera_mode": camera_mode,
        "fps": FPS,
        "duration_s": round(t_end, 4),
        "total_frames": total_frames,
        "pre_roll_s": pre_roll,
        "gravity": GRAVITY,
        "marble_radius": MARBLE_RADIUS,
        "wheel": wheel_cfg,
        "stations": stations,
        "bounces": bounces,
        "rails_plan": rails_plan,
        "frame_samples": dense_samples[::2],
        "dense_samples": dense_samples,
    }


# ---------------------------------------------------------------------------
# TEMPLATE 1: Serpentine Descent ("左右并往下 · T4 Marbles 蛇形连续跟随")
# ---------------------------------------------------------------------------

def plan_serpentine_descent(
    notes: list[dict],
    templates: dict[int, dict],
    manifest: dict,
    pre_roll: float = 0.55,
    post_roll: float = 0.60,
) -> dict:
    phrases = group_notes_into_phrases(notes, gap_thresh=0.65)
    stations: list[dict] = []
    bounces: list[dict] = []
    sid = 0

    # Place each phrase as an alternating diagonal run sweeping Left->Right then Right->Left descending in Z
    y = FLIGHT_Y
    run_drop = 1.85
    z_cursor = 2.55
    min_x, max_x = -2.35, 2.35

    for p_idx, phrase in enumerate(phrases):
        clusters = _cluster_phrase_stations(phrase)
        going_right = (p_idx % 2 == 0)
        nc = len(clusters)
        z_start = z_cursor
        z_end = z_start - 0.90
        tier_min_x = min_x - 0.60 * p_idx
        tier_max_x = max_x + 0.60 * p_idx
        for c_i, cl in enumerate(clusters):
            frac = c_i / max(1, nc - 1) if nc > 1 else 0.5
            cx = (tier_min_x + frac * (tier_max_x - tier_min_x)) if going_right else (tier_max_x - frac * (tier_max_x - tier_min_x))
            # Organic pitch-weighted gentle vertical modulation around diagonal slope
            n0 = cl[0][1]
            pitch_off = ((int(n0["note"]) - 72) / 12.0) * 0.08
            cz = z_start + frac * (z_end - z_start) + pitch_off
            axis_u = [1.0, 0.0, 0.0]
            axis_n = [0.0, 0.0, 1.0]
            anvil_len = ANVIL_LEN

            st = {
                "id": sid,
                "tier": p_idx,
                "going_right": going_right,
                "track_idx": int(n0.get("track_idx", 1)),
                "instrument": n0["instrument"],
                "note": int(n0["note"]),
                "pitch_name": n0.get("pitch_name", f"M{n0['note']}"),
                "center": [cx, y, cz],
                "contact": [cx, y, cz + ANVIL_THICK * 0.5],
                "axis_u": axis_u,
                "axis_v": [0.0, 1.0, 0.0],
                "axis_n": axis_n,
                "anvil_len": anvil_len,
                "inst_scale": 0.90,
                "spring_actuator": False,
                "hits": [],
            }
            for sub_k, (note_idx, n) in enumerate(cl):
                t_vid = pre_roll + float(n["start_time"])
                f_hit = int(round(t_vid * FPS))
                hit_u = 0.0 if len(cl) == 1 else (-0.12 + 0.24 * (sub_k / max(1, len(cl) - 1))) * (1.0 if going_right else -1.0)
                st["hits"].append({
                    "note_index": note_idx,
                    "frame_hit": f_hit,
                    "t": t_vid,
                    "velocity": float(n.get("velocity", 110)),
                    "duration": float(n.get("duration", 0.25)),
                    "note": int(n["note"]),
                    "contact": [cx, y, cz],
                })
                bounces.append({
                    "note_index": note_idx,
                    "station_id": sid,
                    "track_idx": st["track_idx"],
                    "instrument": st["instrument"],
                    "note": int(n["note"]),
                    "pitch_name": st["pitch_name"],
                    "t": t_vid,
                    "frame": f_hit,
                    "anvil_hit_u": hit_u,
                    "going_right": going_right,
                })
            stations.append(st)
            sid += 1
        z_cursor = z_end - run_drop * 0.55

    # Relax any overlapping neighboring stations with jitter solver
    resolve_station_jitter(stations, templates, manifest, min_gap=0.25, allow_z_jitter=True)
    _update_bounces_from_stations(stations, bounces)
    bounces.sort(key=lambda b: b["t"])

    rails_plan, segments = [], []

    # Pure ballistic opening drop onto Note 0 (Rule 1: zero tubes)
    p0 = Vector(bounces[0]["marble_pos"])
    p_open_start = Vector((p0.x - 1.15, y, p0.z + 0.95))
    segments.append({
        "type": "ballistic", "t0": 0.0, "t1": pre_roll,
        "p0": [p_open_start.x, p_open_start.y, p_open_start.z],
        "p1": [p0.x, p0.y, p0.z],
    })

    for k in range(len(bounces) - 1):
        b0, b1 = bounces[k], bounces[k + 1]
        t0, t1 = float(b0["t"]), float(b1["t"])
        pa, pb = Vector(b0["marble_pos"]), Vector(b1["marble_pos"])
        segments.append({
            "type": "ballistic", "t0": t0, "t1": t1,
            "p0": [pa.x, pa.y, pa.z], "p1": [pb.x, pb.y, pb.z],
        })

    # Pure ballistic closing hop off the final board
    p_end = Vector(bounces[-1]["marble_pos"])
    t_end_n = float(bounces[-1]["t"])
    t_fin = t_end_n + post_roll
    p_exit_end = Vector((p_end.x - 1.25, y, p_end.z - 0.55))
    segments.append({
        "type": "ballistic", "t0": t_end_n, "t1": t_fin,
        "p0": [p_end.x, p_end.y, p_end.z],
        "p1": [p_exit_end.x, p_exit_end.y, p_exit_end.z],
    })

    return _finalize_trajectory(
        "serpentine_descent", stations, bounces, rails_plan, segments, t_fin, pre_roll, camera_mode="chase"
    )


# ---------------------------------------------------------------------------
# TEMPLATE 2: Diamond 4-Phrase Closed Loop ("四小节菱形闭环回原点")
# ---------------------------------------------------------------------------

def plan_diamond_loop(
    notes: list[dict],
    templates: dict[int, dict],
    manifest: dict,
    pre_roll: float = 0.0,
) -> dict:
    """4 musical phrases mapped onto a 3D Diamond (descending Right/Bottom + catapult climb Left/Top) closing to frame 0."""
    phrases = group_notes_into_phrases(notes, gap_thresh=0.65)
    # Collapse into 4 structural edges (merge extras if >4)
    while len(phrases) > 4:
        phrases[-2].extend(phrases[-1])
        phrases.pop()

    # True 16:9 widescreen diamond with gentle slope & wide corner stagger so horizontal boards never clip climbing arcs
    corners = [
        (Vector((0.72, FLIGHT_Y, 1.48)), Vector((3.30, FLIGHT_Y, 0.52)), False),    # Edge 0: Down-Right gravity
        (Vector((4.30, FLIGHT_Y, -0.52)), Vector((0.72, FLIGHT_Y, -1.48)), False),  # Edge 1: Down-Left gravity
        (Vector((-0.72, FLIGHT_Y, -1.46)), Vector((-5.55, FLIGHT_Y, -0.52)), True), # Edge 2: Climb-Left spring
        (Vector((-4.65, FLIGHT_Y, 0.96)), Vector((-0.25, FLIGHT_Y, 1.46)), True),   # Edge 3: Climb-Right spring (gentle slope)
    ]

    stations, bounces = [], []
    sid = 0
    for p_idx, phrase in enumerate(phrases[:4]):
        p_a, p_b, is_climb = corners[p_idx]
        clusters = _cluster_phrase_stations(phrase)
        nc = len(clusters)
        edge_dir = (p_b - p_a).normalized()
        for c_i, cl in enumerate(clusters):
            frac = c_i / max(1, nc - 1) if nc > 1 else 0.5
            # Subtle alternating depth/height stagger preserving monotonic horizontal progress
            perp = Vector((-edge_dir.z, 0.0, edge_dir.x))
            jitter = ((c_i % 2) * 2 - 1) * 0.05 * perp
            pos = p_a.lerp(p_b, frac) + jitter
            n0 = cl[0][1]
            # Rule 2 & 3: strictly horizontal board facing UP, long edge along X facing audience
            st = {
                "id": sid,
                "tier": p_idx,
                "track_idx": int(n0.get("track_idx", 1)),
                "instrument": n0["instrument"],
                "note": int(n0["note"]),
                "pitch_name": n0.get("pitch_name", f"M{n0['note']}"),
                "center": [pos.x, pos.y, pos.z],
                "contact": [pos.x, pos.y, pos.z + ANVIL_THICK * 0.5],
                "axis_u": [1.0, 0.0, 0.0],
                "axis_v": [0.0, 1.0, 0.0],
                "axis_n": [0.0, 0.0, 1.0],
                "anvil_len": ANVIL_LEN,
                "inst_scale": 0.72,
                "spring_actuator": is_climb,
                "hits": [],
            }
            for sub_k, (note_idx, n) in enumerate(cl):
                t_vid = float(n["start_time"])
                f_hit = int(round(t_vid * FPS))
                hit_u = 0.0 if len(cl) == 1 else (-0.11 + 0.22 * (sub_k / max(1, len(cl) - 1)))
                st["hits"].append({
                    "note_index": note_idx, "frame_hit": f_hit, "t": t_vid,
                    "velocity": float(n.get("velocity", 110)), "duration": float(n.get("duration", 0.25)),
                    "note": int(n["note"]), "contact": [pos.x, pos.y, pos.z],
                })
                bounces.append({
                    "note_index": note_idx, "station_id": sid, "track_idx": st["track_idx"],
                    "instrument": st["instrument"], "note": int(n["note"]), "pitch_name": st["pitch_name"],
                    "t": t_vid, "frame": f_hit, "anvil_hit_u": hit_u,
                })
            stations.append(st)
            sid += 1

    resolve_station_jitter(stations, templates, manifest, min_gap=0.14, allow_z_jitter=True)
    _update_bounces_from_stations(stations, bounces)
    bounces.sort(key=lambda b: b["t"])
    aabbs_stat = compute_station_aabbs_at_transform(stations, templates, manifest)

    rails_plan, segments = [], []
    # Wrap end->start seam seamlessly with pure ballistic bounces (Rule 1: zero tubes)
    t_loop = float(bounces[-1]["t"]) + 0.92
    for k in range(len(bounces)):
        b0 = bounces[k]
        b1 = bounces[(k + 1) % len(bounces)]
        t0 = float(b0["t"])
        t1 = float(b1["t"]) if k + 1 < len(bounces) else t_loop
        pa, pb = Vector(b0["marble_pos"]), Vector(b1["marble_pos"])
        segments.append({
            "type": "ballistic", "t0": t0, "t1": t1,
            "p0": [pa.x, pa.y, pa.z], "p1": [pb.x, pb.y, pb.z],
        })

    return _finalize_trajectory(
        "diamond_loop", stations, bounces, rails_plan, segments, t_loop, 0.0, camera_mode="chase_wide"
    )


# ---------------------------------------------------------------------------
# TEMPLATE 3: Horological Wheel ("旋转钟表水车圆盘 · 物理重力三重无缝循环")
# ---------------------------------------------------------------------------

def plan_horological_wheel(
    notes: list[dict],
    templates: dict[int, dict],
    manifest: dict,
    t_loop: float = 11.20,
) -> dict:
    """True 360-deg Horological Clockwork Wheel: alternating Outer/Inner dual rings (0.70m separation) turning CCW."""
    wheel_center = Vector((-0.20, FLIGHT_Y, -0.35))
    omega = 2.0 * math.pi / t_loop    # CW spin (+Y rot) sweeps top-arc stations down-right immediately after strike!
    phrases = group_notes_into_phrases(notes, gap_thresh=0.65)
    ring_radii = [3.12, 2.10]          # Even phrases on Outer Rim (3.12m), Odd on Inner Ring (2.10m): 1.02m radial clearance!
    stations, bounces = [], []
    sid = 0

    for p_idx, phrase in enumerate(phrases):
        clusters = _cluster_phrase_stations(phrase)
        nc = len(clusters)
        rad_base = ring_radii[p_idx % 2]
        for c_i, cl in enumerate(clusters):
            frac = c_i / max(1, nc - 1) if nc > 1 else 0.45
            # Clean 60-deg top-arch contact zone (60 deg -> 120 deg): stays above wheel & separates inner/outer rings!
            theta_w = math.radians(60.0 + frac * 60.0)
            rad = rad_base + (0.08 if (sid % 2 == 0) else -0.08)
            world_contact = Vector((
                wheel_center.x + rad * math.cos(theta_w),
                FLIGHT_Y,
                wheel_center.z + rad * math.sin(theta_w),
            ))
            t_hit_ref = float(cl[0][1]["start_time"])
            ang_t = omega * t_hit_ref
            inv_rot = Matrix.Rotation(-ang_t, 3, "Y")
            local_center = wheel_center + (inv_rot @ (world_contact - wheel_center))

            st = {
                "id": sid,
                "tier": p_idx,
                "track_idx": int(cl[0][1].get("track_idx", 1)),
                "instrument": cl[0][1]["instrument"],
                "note": int(cl[0][1]["note"]),
                "pitch_name": cl[0][1].get("pitch_name", f"M{cl[0][1]['note']}"),
                "center": [local_center.x, local_center.y, local_center.z],
                "contact": [local_center.x, local_center.y, local_center.z + ANVIL_THICK * 0.5],
                "axis_u": [1.0, 0.0, 0.0],
                "axis_v": [0.0, 1.0, 0.0],
                "axis_n": [0.0, 0.0, 1.0],
                "anvil_len": ANVIL_LEN,
                "inst_scale": 0.42,
                "spring_actuator": False,
                "wheel_local": True,
                "wheel_upright": True,
                "wheel_center": [wheel_center.x, wheel_center.y, wheel_center.z],
                "wheel_omega": omega,
                "wheel_t_loop": t_loop,
                "hits": [],
            }
            for sub_k, (note_idx, n) in enumerate(cl):
                t_vid = float(n["start_time"])
                f_hit = int(round(t_vid * FPS))
                st["hits"].append({
                    "note_index": note_idx, "frame_hit": f_hit, "t": t_vid,
                    "velocity": float(n.get("velocity", 110)), "duration": float(n.get("duration", 0.25)),
                    "note": int(n["note"]), "contact": [local_center.x, local_center.y, local_center.z],
                })
                bounces.append({
                    "note_index": note_idx, "station_id": sid, "track_idx": st["track_idx"],
                    "instrument": st["instrument"], "note": int(n["note"]), "pitch_name": st["pitch_name"],
                    "t": t_vid, "frame": f_hit,
                })
            stations.append(st)
            sid += 1

    # Guarantee zero overlap in rotating wheel-local frame, then sync exact world bounce positions!
    resolve_station_jitter(stations, templates, manifest, min_gap=0.10, allow_z_jitter=True)
    st_by_id = {s["id"]: s for s in stations}
    for b in bounces:
        st = st_by_id[b["station_id"]]
        lc = Vector(st["center"])
        rot_t = Matrix.Rotation(omega * float(b["t"]), 3, "Y")
        w_anvil = wheel_center + (rot_t @ (lc - wheel_center))
        w_mpos = w_anvil + Vector((0.0, 0.0, ANVIL_THICK * 0.5 + MARBLE_RADIUS))
        b["marble_pos"] = [w_mpos.x, w_mpos.y, w_mpos.z]

    bounces.sort(key=lambda b: b["t"])
    rails_plan, segments = [], []

    # Pure ballistic bounces everywhere (Rule 1: zero tubes)
    for k in range(len(bounces)):
        b0 = bounces[k]
        b1 = bounces[(k + 1) % len(bounces)]
        t0 = float(b0["t"])
        t1 = float(b1["t"]) if k + 1 < len(bounces) else t_loop
        pa, pb = Vector(b0["marble_pos"]), Vector(b1["marble_pos"])
        segments.append({
            "type": "ballistic", "t0": t0, "t1": t1,
            "p0": [pa.x, pa.y, pa.z], "p1": [pb.x, pb.y, pb.z],
        })

    wheel_cfg = {
        "center": [wheel_center.x, wheel_center.y, wheel_center.z],
        "radius": 3.28,
        "rings": [2.10, 3.12],
        "omega": omega,
        "t_loop": t_loop,
    }
    return _finalize_trajectory(
        "horological_wheel", stations, bounces, rails_plan, segments, t_loop, 0.0,
        wheel_cfg=wheel_cfg, camera_mode="wheel_focus",
    )


TEMPLATE_PLANNERS = {
    "serpentine_descent": plan_serpentine_descent,
    "diamond_loop": plan_diamond_loop,
    "horological_wheel": plan_horological_wheel,
}


# ---------------------------------------------------------------------------
# Architectural Environment Builders per Template
# ---------------------------------------------------------------------------

def build_concert_world(scene: bpy.types.Scene) -> None:
    world = bpy.data.worlds.new("WidescreenStudioWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    lp = nt.nodes.new("ShaderNodeLightPath")

    bg_diff = nt.nodes.new("ShaderNodeBackground")
    bg_diff.inputs["Color"].default_value = (0.015, 0.017, 0.025, 1.0)
    bg_diff.inputs["Strength"].default_value = 0.45

    bg_gloss = nt.nodes.new("ShaderNodeBackground")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Normal"], sep.inputs["Vector"])

    r_top = nt.nodes.new("ShaderNodeValToRGB")
    r_top.color_ramp.elements[0].position = 0.30
    r_top.color_ramp.elements[0].color = (0.008, 0.009, 0.014, 1.0)
    r_top.color_ramp.elements[1].position = 0.52
    r_top.color_ramp.elements[1].color = (4.2, 3.8, 3.1, 1.0)
    nt.links.new(sep.outputs["Z"], r_top.inputs["Fac"])

    r_side = nt.nodes.new("ShaderNodeValToRGB")
    r_side.color_ramp.elements[0].position = 0.55
    r_side.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    r_side.color_ramp.elements[1].position = 0.72
    r_side.color_ramp.elements[1].color = (1.8, 2.5, 3.5, 1.0)
    nt.links.new(sep.outputs["X"], r_side.inputs["Fac"])

    add_rgb = nt.nodes.new("ShaderNodeMix")
    add_rgb.data_type = "RGBA"
    add_rgb.blend_type = "ADD"
    add_rgb.inputs["Factor"].default_value = 1.0
    nt.links.new(r_top.outputs["Color"], add_rgb.inputs["A"])
    nt.links.new(r_side.outputs["Color"], add_rgb.inputs["B"])

    nt.links.new(add_rgb.outputs["Result"], bg_gloss.inputs["Color"])
    nt.links.new(lp.outputs["Is Glossy Ray"], mix_sh.inputs["Fac"])
    nt.links.new(bg_diff.outputs["Background"], mix_sh.inputs[1])
    nt.links.new(bg_gloss.outputs["Background"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])


def setup_bloom_compositor(scene: bpy.types.Scene) -> None:
    ng = bpy.data.node_groups.new("ConcertBloom16x9", "CompositorNodeTree")
    scene.compositing_node_group = ng
    scene.render.use_compositing = True
    if not any(s.name == "Image" and s.in_out == "OUTPUT" for s in ng.interface.items_tree):
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = ng.nodes.new("CompositorNodeRLayers")
    glare = ng.nodes.new("CompositorNodeGlare")
    out = ng.nodes.new("NodeGroupOutput")
    set_socket(glare, "Type", "Bloom")
    set_socket(glare, "Quality", "High")
    set_socket(glare, "Threshold", 1.25)
    set_socket(glare, "Smoothness", 0.35)
    set_socket(glare, "Strength", 0.40)
    set_socket(glare, "Size", 0.55)
    ng.links.new(rl.outputs["Image"], glare.inputs["Image"])
    ng.links.new(glare.outputs["Image"], out.inputs["Image"])


def create_lighting_rig(col_stage: bpy.types.Collection, z_center: float = 0.0) -> None:
    def add_area(name, loc, rot_deg, energy, sx, sy, col):
        ld = bpy.data.lights.new(name + "Data", type="AREA")
        ld.energy = energy
        ld.shape = "RECTANGLE"
        ld.size = sx
        ld.size_y = sy
        ld.color = col
        lo = bpy.data.objects.new(name, ld)
        col_stage.objects.link(lo)
        lo.location = loc
        lo.rotation_euler = tuple(math.radians(a) for a in rot_deg)
        return lo

    add_area("KeySoftboxWarm", (0.0, -4.8, z_center + 4.0), (50.0, 0.0, 0.0), 420.0, 7.5, 3.0, (1.0, 0.94, 0.85))
    add_area("CoolRimRight", (5.2, -2.5, z_center + 1.5), (68.0, 0.0, 48.0), 170.0, 3.6, 5.5, (0.70, 0.86, 1.0))
    add_area("WarmFillLeft", (-5.2, -2.5, z_center - 0.5), (92.0, 0.0, -50.0), 150.0, 3.6, 5.5, (1.0, 0.78, 0.56))


def create_stage_backdrop_and_props(plan: dict, col_stage: bpy.types.Collection, mats: dict) -> bpy.types.Object | None:
    """Build low-contrast matte acoustic hall shell + subtle slats so marble & instruments pop cleanly."""
    wall_mat = make_pbr_material("StageObsidianShell", (0.010, 0.011, 0.014), metallic=0.02, roughness=0.92)
    slat_mat = make_pbr_material("StageDarkWalnutSlat", (0.020, 0.015, 0.012), metallic=0.02, roughness=0.78, coat=0.0)
    brass_trim_mat = make_pbr_material("StageDimBrassTrim", (0.16, 0.12, 0.06), metallic=0.65, roughness=0.58)
    walnut_mat = mats.get("LibMat_Walnut") or make_pbr_material("StageWalnut", (0.14, 0.08, 0.04), roughness=0.38)
    brass_mat = mats.get("LibMat_Brass") or make_pbr_material("StageBrass", (0.82, 0.62, 0.24), metallic=0.95, roughness=0.22)
    cove_mat = make_pbr_material(
        "StageWarmCove", (0.95, 0.78, 0.50), emit_color=(1.0, 0.80, 0.48), emit_strength=1.10
    )

    zs = [b["marble_pos"][2] for b in plan["bounces"]]
    z_mid = 0.5 * (min(zs) + max(zs)) if zs else 0.0
    z_span = max(12.5, (max(zs) - min(zs)) + 9.0) if zs else 12.5

    # Deep dark acoustic shell behind instruments (Y >= +0.96)
    smesh = bpy.data.meshes.new("AcousticBackdropShell")
    bm = bmesh.new()
    sx, sz = 32, 6
    grid = []
    for iz in range(sz + 1):
        zc = (z_mid - z_span * 0.5) + (iz / sz) * z_span
        row = []
        for ix in range(sx + 1):
            xf = (ix / sx) * 2.0 - 1.0
            xc = xf * 7.8
            yc = 1.24 - 0.14 * (1.0 - xf * xf)
            row.append(bm.verts.new((xc, yc, zc)))
        grid.append(row)
    for iz in range(sz):
        for ix in range(sx):
            bm.faces.new((grid[iz][ix], grid[iz][ix + 1], grid[iz + 1][ix + 1], grid[iz + 1][ix]))
    bm.to_mesh(smesh)
    bm.free()
    for p in smesh.polygons:
        p.use_smooth = True
    bobj = bpy.data.objects.new("AcousticBackdropShell", smesh)
    col_stage.objects.link(bobj)
    bobj.data.materials.append(wall_mat)

    # Subtle low-contrast acoustic walnut flutes (no bright vertical brass streaks behind marble!)
    for idx_flute, fx in enumerate([k * 0.85 for k in range(-7, 8)]):
        rm = create_beveled_box_mesh(f"WallFlute_{idx_flute}", 0.10, 0.08, z_span * 0.94, bevel_w=0.008)
        ro = bpy.data.objects.new(f"WallFlute_{idx_flute}", rm)
        col_stage.objects.link(ro)
        ro.location = (fx, 0.98, z_mid)
        ro.data.materials.append(slat_mat)

    # Subtle brushed-brass horizontal tier guide rails behind active shelves
    if plan["template"] == "serpentine_descent":
        seen_tiers = sorted({st.get("tier", 0) for st in plan["stations"]})
        for t_idx in seen_tiers:
            tzs = [st["center"][2] for st in plan["stations"] if st.get("tier", 0) == t_idx]
            if tzs:
                z_bar = sum(tzs) / len(tzs)
                bar_m = create_beveled_box_mesh(f"TierSpine_{t_idx}", 9.2, 0.04, 0.028, bevel_w=0.004)
                bar_o = bpy.data.objects.new(f"TierSpine_{t_idx}", bar_m)
                col_stage.objects.link(bar_o)
                bar_o.location = (0.0, 0.80, z_bar - 0.01)
                bar_o.data.materials.append(brass_trim_mat)

    # Outer walnut acoustic wings on far edges (|X| >= 4.4)
    for side in (-1, 1):
        for k in range(3):
            rx = side * (4.5 + k * 0.44)
            rm = create_beveled_box_mesh(f"WingRib_{side}_{k}", 0.15, 0.20, z_span * 0.94, bevel_w=0.01)
            ro = bpy.data.objects.new(f"WingRib_{side}_{k}", rm)
            col_stage.objects.link(ro)
            ro.location = (rx, 0.72, z_mid)
            ro.data.materials.append(walnut_mat)

    tname = plan["template"]
    if tname == "serpentine_descent":
        # Subtle warm brass guide trim along outer wings + opening marquee above Run 0
        top_z = max(zs) + 0.95
        bpy.ops.object.text_add(location=(0.0, 0.56, top_z))
        txt = bpy.context.active_object
        txt.name = "SerpTitle"
        txt.data.body = "LUCA  ·  MARBLE SYMPHONY"
        txt.data.align_x = "CENTER"
        txt.data.align_y = "CENTER"
        txt.data.size = 0.22
        txt.data.extrude = 0.016
        txt.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        txt.data.materials.append(brass_mat)
        for c in list(txt.users_collection):
            c.objects.unlink(txt)
        col_stage.objects.link(txt)
        return None

    if tname == "diamond_loop":
        # Luminous subtle brass diamond halo frame behind the diamond track (Y = +0.64)
        d_pts = [
            [0.0, 0.64, 2.15],
            [4.28, 0.64, 0.0],
            [0.0, 0.64, -2.15],
            [-4.28, 0.64, 0.0],
            [0.0, 0.64, 2.15],
        ]
        create_tube_polyline("DiamondBackFrame", d_pts, 0.022, brass_mat, col_stage)
        create_tube_polyline("DiamondGlowTrim", [[p[0] * 0.94, 0.68, p[2] * 0.94] for p in d_pts], 0.009, cove_mat, col_stage)
        return None

    if tname == "horological_wheel":
        wcfg = plan["wheel"]
        wc = Vector(wcfg["center"])
        r_ring = float(wcfg["radius"])
        root = bpy.data.objects.new("ClockworkWheelRoot", None)
        col_stage.objects.link(root)
        root.location = wc
        root.rotation_mode = "XYZ"

        # Monumental Clockwork Turntable Wheel lying behind instruments (Y_local = +0.52) with controlled satin bronze so instruments pop
        y_wheel = 0.52
        outer_pts = [
            [r_ring * math.cos(2.0 * math.pi * i / 72), y_wheel, r_ring * math.sin(2.0 * math.pi * i / 72)]
            for i in range(73)
        ]
        create_tube_polyline("WheelOuterRim", outer_pts, 0.032, brass_trim_mat, col_stage).parent = root
        for r_idx, r_sub in enumerate(wcfg.get("rings", [r_ring * 0.62])):
            sub_pts = [
                [r_sub * math.cos(2.0 * math.pi * i / 72), y_wheel, r_sub * math.sin(2.0 * math.pi * i / 72)]
                for i in range(73)
            ]
            rmat = walnut_mat if r_idx % 2 else brass_trim_mat
            create_tube_polyline(f"WheelTrackRing_{r_idx:02d}", sub_pts, 0.016, rmat, col_stage).parent = root
        for s_i in range(12):
            ang = 2.0 * math.pi * s_i / 12.0
            pa = Vector((0.35 * math.cos(ang), y_wheel, 0.35 * math.sin(ang)))
            pb = Vector((r_ring * math.cos(ang), y_wheel, r_ring * math.sin(ang)))
            spk = create_cylinder_between(f"WheelSpoke_{s_i:02d}", pa, pb, 0.014, brass_trim_mat, col_stage, verts=12)
            if spk:
                spk.parent = root

        # Central stepped clockwork brass hub
        bpy.ops.mesh.primitive_cylinder_add(vertices=36, radius=0.44, depth=0.14, location=(0.0, y_wheel, 0.0))
        hub = bpy.context.active_object
        hub.name = "ClockworkHub"
        hub.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        hub.data.materials.append(walnut_mat)
        for c in list(hub.users_collection):
            c.objects.unlink(hub)
        col_stage.objects.link(hub)
        hub.parent = root

        bpy.ops.mesh.primitive_cylinder_add(vertices=36, radius=0.26, depth=0.18, location=(0.0, y_wheel - 0.02, 0.0))
        hub_cap = bpy.context.active_object
        hub_cap.name = "ClockworkHubCap"
        hub_cap.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        hub_cap.data.materials.append(brass_mat)
        for c in list(hub_cap.users_collection):
            c.objects.unlink(hub_cap)
        col_stage.objects.link(hub_cap)
        hub_cap.parent = root

        # Keyframe exact 360 deg perpetual rotation frame-by-frame matching wcfg['omega'] sign (Blender 5.x Action safe!)
        t_loop = float(wcfg["t_loop"])
        omega = float(wcfg["omega"])
        total_f = int(round(t_loop * FPS))
        for f in range(total_f + 1):
            ang = omega * (f / FPS)
            root.rotation_euler = (0.0, ang, 0.0)
            root.keyframe_insert(data_path="rotation_euler", frame=f)
        return root

    return None


# ---------------------------------------------------------------------------
# Instrument Spawner (With Optional Animated Spring Piston Actuator)
# ---------------------------------------------------------------------------

def _should_highlight_part(sub_name: str) -> bool:
    tokens = (
        "_WKey_", "_BKey_", "_ToneBar_", "_SteelBar_", "_Str_", "_HarpStr_",
        "_BellCup", "_GongDish", "_GongBoss", "_Chime", "_MalletHead", "_MalletYarn",
        "_MalletBall", "_HammerHead", "_Clapper", "_Knob", "_Pad", "_VU_", "_Strip",
    )
    return any(tok in sub_name for tok in tokens)


def _keyframe_emission(em_sock, f_hit: int, base_v: float, peak_v: float, decay_f: int, bsdf=None, glow_color=None):
    if em_sock is None:
        return
    if bsdf is not None and glow_color is not None:
        set_socket(bsdf, "Emission Color", (*glow_color[:3], 1.0))
    em_sock.default_value = base_v
    em_sock.keyframe_insert(data_path="default_value", frame=max(0, f_hit - 1))
    em_sock.default_value = peak_v
    em_sock.keyframe_insert(data_path="default_value", frame=f_hit)
    em_sock.default_value = base_v + 0.35 * (peak_v - base_v)
    em_sock.keyframe_insert(data_path="default_value", frame=f_hit + min(2, decay_f - 1))
    em_sock.default_value = base_v
    em_sock.keyframe_insert(data_path="default_value", frame=f_hit + max(3, decay_f))


def _keyframe_part_transform(part: dict, keyframes: list, pivot: Vector | None = None) -> None:
    sobj = part["obj"]
    orig_loc: Vector = part["orig_loc"]
    orig_rot_mat = part["orig_rot"].to_matrix()
    orig_scale: Vector = part["orig_scale"]
    piv = pivot if pivot is not None else orig_loc
    sobj.rotation_mode = "XYZ"
    for frame, rot_deg, trans, scale_mul in keyframes:
        rx, ry, rz = (math.radians(a) for a in rot_deg)
        r_delta = Euler((rx, ry, rz), "XYZ").to_matrix()
        sobj.location = piv + (r_delta @ (orig_loc - piv)) + Vector(trans)
        sobj.rotation_euler = (r_delta @ orig_rot_mat).to_euler("XYZ")
        sobj.scale = Vector((orig_scale.x * scale_mul[0], orig_scale.y * scale_mul[1], orig_scale.z * scale_mul[2]))
        f_c = max(0, int(frame))
        sobj.keyframe_insert(data_path="location", frame=f_c)
        sobj.keyframe_insert(data_path="rotation_euler", frame=f_c)
        sobj.keyframe_insert(data_path="scale", frame=f_c)


def spawn_station(
    st: dict,
    templates: dict[int, dict],
    mats: dict,
    manifest: dict,
    col_inst: bpy.types.Collection,
    col_fx: bpy.types.Collection,
    wheel_root: bpy.types.Object | None = None,
) -> None:
    sid = int(st["id"])
    gm = resolve_gm_program(st["instrument"], st["track_idx"], manifest)
    tpl = templates.get(gm) or templates.get(-1000 - st["track_idx"]) or next(iter(templates.values()))
    anim_family = str(tpl.get("anim_family", "generic_strike"))
    glow_color = tuple(tpl.get("glow_color", (1.0, 0.78, 0.30)))
    inst_scale = float(st.get("inst_scale", 1.0))
    anvil_len = float(st.get("anvil_len", ANVIL_LEN))

    u = Vector(st["axis_u"]).normalized()
    v = Vector(st["axis_v"]).normalized()
    n = Vector(st["axis_n"]).normalized()
    rot_mat = Matrix(((u.x, v.x, n.x), (u.y, v.y, n.y), (u.z, v.z, n.z)))
    base_pos = Vector(st["center"])

    # Station Base Mount: Ferris-wheel upright gondola orbit when wheel_upright=True (Rule 2: always facing UP)
    mount = bpy.data.objects.new(f"Mount_{sid:02d}", None)
    col_inst.objects.link(mount)
    if wheel_root is not None and st.get("wheel_upright"):
        wc = Vector(st["wheel_center"])
        omega = float(st["wheel_omega"])
        total_f = int(round(float(st["wheel_t_loop"]) * FPS))
        for f in range(total_f + 1):
            rot_f = Matrix.Rotation(omega * (f / FPS), 3, "Y")
            mount.location = wc + (rot_f @ (base_pos - wc))
            mount.keyframe_insert(data_path="location", frame=f)
        mount.location = base_pos
    elif wheel_root is not None:
        mount.parent = wheel_root
        mount.location = base_pos - wheel_root.location
    else:
        mount.location = base_pos
    mount.rotation_mode = "QUATERNION"
    mount.rotation_quaternion = rot_mat.to_quaternion()

    # Compact beveled dark acoustic Anvil Saddle (NO bright neon bar!)
    anvil_mesh = create_beveled_box_mesh(f"AnvilMesh_{sid:02d}", anvil_len, ANVIL_WID_Y, ANVIL_THICK, bevel_w=0.008)
    anvil_obj = bpy.data.objects.new(f"Anvil_{sid:02d}_{st['instrument']}_{st['pitch_name']}", anvil_mesh)
    col_inst.objects.link(anvil_obj)
    anvil_obj.parent = mount
    anvil_mat = make_pbr_material(
        f"AnvilMat_{sid:02d}",
        base_color=(0.11, 0.075, 0.055) if st["track_idx"] == 1 else (0.06, 0.07, 0.10),
        metallic=0.58, roughness=0.36, coat=0.25,
    )
    anvil_obj.data.materials.append(anvil_mat)

    cap_mesh = create_beveled_box_mesh(f"CapMesh_{sid:02d}", anvil_len * 0.42, 0.10, 0.008, bevel_w=0.002)
    cap_obj = bpy.data.objects.new(f"Cap_{sid:02d}", cap_mesh)
    col_inst.objects.link(cap_obj)
    cap_obj.parent = anvil_obj
    cap_obj.location = (0.0, 0.0, ANVIL_THICK * 0.5 + 0.002)
    cap_mat = make_pbr_material(f"CapMat_{sid:02d}", base_color=(0.88, 0.66, 0.26), metallic=0.92, roughness=0.18)
    cap_obj.data.materials.append(cap_mat)

    # Rear support arm linking anvil cleanly to backdrop or spinning wheel plane
    arm_len_y = 0.96 if wheel_root is None else 0.54
    arm_obj = create_cylinder_between(
        f"BackArm_{sid:02d}",
        Vector((0.0, 0.06, -0.01)),
        Vector((0.0, arm_len_y, -0.01)),
        0.014,
        make_pbr_material(f"ArmMat_{sid:02d}", (0.14, 0.11, 0.09), metallic=0.78, roughness=0.34),
        col_inst,
        verts=12,
    )
    if arm_obj is not None:
        arm_obj.parent = mount

    # Optional telescoping brass Spring Piston Catapult for climb edges in Diamond Loop
    if st.get("spring_actuator"):
        p_mesh = create_beveled_box_mesh(f"Piston_{sid:02d}", 0.08, 0.08, 0.16, bevel_w=0.008)
        p_obj = bpy.data.objects.new(f"PistonActuator_{sid:02d}", p_mesh)
        col_inst.objects.link(p_obj)
        p_obj.parent = mount
        p_obj.location = (0.0, 0.0, -0.09)
        p_obj.data.materials.append(cap_mat)

    # Sculpted v3.0 Instrument Showcase tilted toward viewer behind anvil
    piv_obj = bpy.data.objects.new(f"InstTiltPivot_{sid:02d}", None)
    col_inst.objects.link(piv_obj)
    piv_obj.parent = anvil_obj
    piv_obj.location = (0.0, SUB_Y_OFFSET, 0.0)
    piv_obj.rotation_mode = "XYZ"
    piv_obj.rotation_euler = (math.radians(INST_DISPLAY_TILT_DEG), 0.0, 0.0)

    spawned_subs = []
    local_glow_cache = {}
    for sub_name, sub_data, sub_mats, sub_loc, sub_rot, sub_scale in tpl["subs"]:
        sobj = bpy.data.objects.new(f"{sub_name}_{sid:02d}", sub_data)
        col_inst.objects.link(sobj)
        sobj.parent = piv_obj
        sc_loc = sub_loc * inst_scale
        sc_scale = sub_scale * inst_scale
        sobj.location = sc_loc
        sobj.rotation_mode = "XYZ"
        sobj.rotation_euler = sub_rot
        sobj.scale = sc_scale
        if not sobj.data.materials and sub_mats:
            for sm in sub_mats:
                sobj.data.materials.append(sm)

        em_sock, bsdf_sub, base_em = None, None, 0.0
        if _should_highlight_part(sub_name) and sub_mats and len(sobj.material_slots) > 0:
            orig_m = sub_mats[0]
            ckey = f"{orig_m.name}|{'VU' if '_VU_' in sub_name else 'P'}"
            if ckey not in local_glow_cache:
                mc = orig_m.copy()
                mc.name = f"PartM_{sid:02d}_{len(local_glow_cache):02d}"
                bsdf_c = next((nd for nd in mc.node_tree.nodes if nd.type == "BSDF_PRINCIPLED"), None)
                esock_c = bsdf_c.inputs.get("Emission Strength") if bsdf_c else None
                b_em = float(esock_c.default_value) if esock_c is not None else 0.0
                if "_VU_" in sub_name and esock_c is not None:
                    b_em = 0.08
                local_glow_cache[ckey] = (mc, bsdf_c, esock_c, b_em)
            mc, bsdf_sub, em_sock, base_em = local_glow_cache[ckey]
            sobj.material_slots[0].link = "OBJECT"
            sobj.material_slots[0].material = mc

        spawned_subs.append({
            "obj": sobj, "base_name": sub_name,
            "orig_loc": sc_loc.copy(), "orig_rot": sub_rot.copy(), "orig_scale": sc_scale.copy(),
            "bsdf": bsdf_sub, "em_sock": em_sock, "base_em": base_em,
        })

    # Localized HitSpark point light (Zero ugly collision circle rings!)
    hits = sorted(st["hits"], key=lambda h: int(h["frame_hit"]))
    ldata = None
    if hits:
        ldata = bpy.data.lights.new(name=f"SparkData_{sid:02d}", type="POINT")
        ldata.color = glow_color[:3]
        ldata.energy = 0.0
        ldata.shadow_soft_size = 0.14
        if hasattr(ldata, "use_shadow"):
            ldata.use_shadow = False
        lobj = bpy.data.objects.new(f"HitSpark_{sid:02d}", ldata)
        col_fx.objects.link(lobj)
        lobj.parent = mount
        lobj.location = (0.0, -0.20, 0.22)

    anvil_bsdf = next(n for n in anvil_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    cap_bsdf = next(n for n in cap_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")

    for k, h in enumerate(hits):
        f_hit = int(h["frame_hit"])
        vel = float(h.get("velocity", 110.0))
        vel_scale = 0.65 + 0.55 * max(0.0, min(1.0, vel / 127.0))
        pitch_frac = max(0.0, min(1.0, (int(h.get("note", st["note"])) - 60) / 24.0))
        nxt_f = int(hits[k + 1]["frame_hit"]) if k + 1 < len(hits) else None
        prev_f = int(hits[k - 1]["frame_hit"]) if k > 0 else None
        max_decay = 9 if nxt_f is None else max(3, min(9, nxt_f - f_hit - 2))

        if prev_f is None or (f_hit - prev_f) > 2:
            anvil_obj.location = (0.0, 0.0, 0.0)
            anvil_obj.keyframe_insert(data_path="location", frame=max(0, f_hit - 1))
            if ldata is not None:
                ldata.energy = 0.0
                ldata.keyframe_insert(data_path="energy", frame=max(0, f_hit - 1))

        # Instrument switching & music-box wake-up anticipation (14 frames before impact)
        if k == 0 and f_hit >= 14:
            piv_obj.scale = (0.91, 0.91, 0.91)
            piv_obj.location = (0.0, SUB_Y_OFFSET, -0.016)
            piv_obj.keyframe_insert(data_path="scale", frame=0)
            piv_obj.keyframe_insert(data_path="location", frame=0)
            piv_obj.keyframe_insert(data_path="scale", frame=max(0, f_hit - 14))
            piv_obj.keyframe_insert(data_path="location", frame=max(0, f_hit - 14))
            piv_obj.scale = (1.04, 1.04, 1.04)
            piv_obj.location = (0.0, SUB_Y_OFFSET, 0.008)
            piv_obj.keyframe_insert(data_path="scale", frame=max(0, f_hit - 4))
            piv_obj.keyframe_insert(data_path="location", frame=max(0, f_hit - 4))
            piv_obj.scale = (1.0, 1.0, 1.0)
            piv_obj.location = (0.0, SUB_Y_OFFSET, 0.0)
            piv_obj.keyframe_insert(data_path="scale", frame=max(0, f_hit - 1))
            piv_obj.keyframe_insert(data_path="location", frame=max(0, f_hit - 1))

        if st.get("spring_actuator"):
            # Visible catapult spring kick UPWARD on climb stations!
            anvil_obj.location = (0.0, 0.0, -0.018 * vel_scale)
            anvil_obj.keyframe_insert(data_path="location", frame=max(0, f_hit))
            anvil_obj.location = (0.0, 0.0, 0.042 * vel_scale)
            anvil_obj.keyframe_insert(data_path="location", frame=f_hit + 2)
            anvil_obj.location = (0.0, 0.0, 0.0)
            anvil_obj.keyframe_insert(data_path="location", frame=f_hit + min(8, max_decay))
        else:
            anvil_obj.location = (0.0, 0.0, -0.014 * vel_scale)
            anvil_obj.keyframe_insert(data_path="location", frame=f_hit + 1)
            anvil_obj.location = (0.0, 0.0, 0.0)
            anvil_obj.keyframe_insert(data_path="location", frame=f_hit + min(6, max_decay))

        _keyframe_emission(cap_bsdf.inputs.get("Emission Strength"), f_hit, 0.0, 0.78 * vel_scale, max_decay, cap_bsdf, glow_color)
        _keyframe_emission(anvil_bsdf.inputs.get("Emission Strength"), f_hit, 0.0, 0.16 * vel_scale, max_decay, anvil_bsdf, glow_color)

        if ldata is not None:
            ldata.energy = 16.0 * vel_scale
            ldata.keyframe_insert(data_path="energy", frame=f_hit)
            ldata.energy = 5.5 * vel_scale
            ldata.keyframe_insert(data_path="energy", frame=f_hit + min(2, max_decay - 1))
            ldata.energy = 0.0
            ldata.keyframe_insert(data_path="energy", frame=f_hit + max_decay)

        keys = [sp for sp in spawned_subs if "_WKey_" in sp["base_name"] or "_BKey_" in sp["base_name"]]
        if keys and anim_family in ("keyboard_hinge", "keyboard_pluck"):
            ordered = sorted(keys, key=lambda p: p["orig_loc"].x)
            hk = ordered[max(0, min(len(ordered) - 1, int(round(pitch_frac * (len(ordered) - 1)))))]
            piv = Vector((hk["orig_loc"].x, hk["orig_loc"].y + 0.04 * inst_scale, hk["orig_loc"].z))
            _keyframe_part_transform(hk, [
                (f_hit - 1, (0, 0, 0), (0, 0, 0), (1, 1, 1)),
                (f_hit + 1, (-8.5 * vel_scale, 0, 0), (0, 0, -0.015 * vel_scale * inst_scale), (1, 1, 1)),
                (f_hit + min(7, max_decay), (0, 0, 0), (0, 0, 0), (1, 1, 1)),
            ], pivot=piv)

        mallets = [sp for sp in spawned_subs if any(tok in sp["base_name"] for tok in ("_Mallet", "_Clapper"))]
        if mallets:
            piv_m = Vector((0.0, 0.22 * inst_scale, 0.14 * inst_scale))
            for mp in mallets:
                _keyframe_part_transform(mp, [
                    (f_hit - 2, (-14.0, 0, 0), (0, 0, 0.018 * inst_scale), (1, 1, 1)),
                    (f_hit, (12.0 * vel_scale, 0, 0), (0, -0.022 * inst_scale, -0.015 * inst_scale), (1, 1, 1)),
                    (f_hit + min(7, max_decay), (0, 0, 0), (0, 0, 0), (1, 1, 1)),
                ], pivot=piv_m)

        for sp in spawned_subs:
            esock = sp.get("em_sock")
            if esock is None:
                continue
            b_em = sp.get("base_em", 0.0)
            p_em = 2.4 * vel_scale if "_VU_" in sp["base_name"] else min(1.30, max(0.35, 0.85 * vel_scale))
            _keyframe_emission(esock, f_hit, b_em, p_em, max_decay, sp.get("bsdf"), glow_color if b_em < 0.5 else None)


# ---------------------------------------------------------------------------
# Hero Marble & Mode-Adaptive 16:9 Cinema Follow Camera
# ---------------------------------------------------------------------------

def create_hero_marble(col_marble: bpy.types.Collection, frame_samples: list[dict], bounces: list[dict]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=44, ring_count=26, radius=MARBLE_RADIUS)
    marble = bpy.context.active_object
    marble.name = "HeroMarble_Aurora"
    for c in list(marble.users_collection):
        c.objects.unlink(marble)
    col_marble.objects.link(marble)
    for p in marble.data.polygons:
        p.use_smooth = True

    mat = make_pbr_material(
        "MarbleAuroraMat",
        base_color=(0.06, 0.88, 0.56), metallic=0.90, roughness=0.07, coat=1.0,
        emit_color=(0.12, 0.95, 0.62), emit_strength=0.18,
    )
    marble.data.materials.append(mat)
    marble.rotation_mode = "XYZ"

    hit_frames = {int(b["frame"]) for b in bounces}
    angle_acc_y = 0.0
    prev_pos = Vector(frame_samples[0]["position"])
    for smp in frame_samples:
        f = int(round(smp["frame_f"]))
        pos = Vector(smp["position"])
        angle_acc_y += (pos.x - prev_pos.x) / max(0.05, MARBLE_RADIUS)
        prev_pos = pos
        marble.location = pos
        marble.rotation_euler = (0.0, angle_acc_y, 0.0)
        marble.keyframe_insert(data_path="location", frame=f)
        marble.keyframe_insert(data_path="rotation_euler", frame=f)
        if f in hit_frames:
            marble.scale = (1.0, 1.0, 1.0)
            marble.keyframe_insert(data_path="scale", frame=max(0, f - 1))
            marble.scale = (1.025, 1.025, 0.96)
            marble.keyframe_insert(data_path="scale", frame=f)
            marble.scale = (0.988, 0.988, 1.025)
            marble.keyframe_insert(data_path="scale", frame=f + 1)
            marble.scale = (1.0, 1.0, 1.0)
            marble.keyframe_insert(data_path="scale", frame=f + 3)

    ml = bpy.data.lights.new("MarbleRimLamp", type="POINT")
    ml.color = (0.28, 0.98, 0.66)
    ml.energy = 3.5
    ml.shadow_soft_size = 0.14
    if hasattr(ml, "use_shadow"):
        ml.use_shadow = False
    mlo = bpy.data.objects.new("MarbleRimLampObj", ml)
    col_marble.objects.link(mlo)
    mlo.parent = marble
    mlo.location = (0.0, -0.06, 0.0)
    return marble


def setup_cinema_camera(scene: bpy.types.Scene, col_cam: bpy.types.Collection, plan: dict) -> None:
    """Rhythm-adaptive 16:9 cinema chase cam: oblique 3/4 depth, dynamic dolly, look-ahead & auto-focus DOF."""
    cam_data = bpy.data.cameras.new("ConcertCinemaCamData")
    mode = plan["camera_mode"]
    if mode == "chase":
        cam_data.lens = 38.0
    elif mode == "chase_wide":
        cam_data.lens = 32.0
    else:
        cam_data.lens = 42.0
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.05
    cam_data.clip_end = 120.0
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 6.3
    cam_obj = bpy.data.objects.new("ConcertCinemaCam", cam_data)
    col_cam.objects.link(cam_obj)
    scene.camera = cam_obj

    tgt = bpy.data.objects.new("CamFocusTarget", None)
    col_cam.objects.link(tgt)
    cns = cam_obj.constraints.new(type="TRACK_TO")
    cns.target = tgt
    cns.track_axis = "TRACK_NEGATIVE_Z"
    cns.up_axis = "UP_Y"

    samples = plan["frame_samples"]
    bounces = plan["bounces"]
    n_smp = len(samples)
    total_f = plan["total_frames"]

    # Map each bounce frame to local phrase spacing (dt) & velocity for rhythm-driven dolly + impact shake
    bounce_info: list[tuple[int, float, float]] = []
    for i, b in enumerate(bounces):
        bf = int(b["frame"])
        dt_next = (float(bounces[i + 1]["t"]) - float(b["t"])) if i + 1 < len(bounces) else 0.50
        dt_prev = (float(b["t"]) - float(bounces[i - 1]["t"])) if i > 0 else 0.50
        vel = float(b.get("velocity", 110.0))
        bounce_info.append((bf, min(dt_next, dt_prev), vel))
    strong_hit_frames = {bf for bf, _, vel in bounce_info if vel >= 95.0}

    cam_pos_smooth = Vector(samples[0]["position"])
    tgt_pos_smooth = Vector(samples[0]["position"])
    dolly_smooth = 1.0
    base_alpha = 0.11 if mode == "chase" else (0.09 if mode == "chase_wide" else 0.10)

    for idx, smp in enumerate(samples):
        f = int(round(smp["frame_f"]))
        mp = Vector(smp["position"])
        t_cur = float(smp["t"])

        # Determine rhythm density: push in on dense runs (dt <= 0.35s), pull back on high-arc rests (dt >= 0.70s)
        local_dt = 0.45
        for i in range(len(bounces) - 1):
            if float(bounces[i]["t"]) <= t_cur <= float(bounces[i + 1]["t"]):
                local_dt = float(bounces[i + 1]["t"]) - float(bounces[i]["t"])
                break
        target_dolly = 0.88 if local_dt <= 0.35 else (1.20 if local_dt >= 0.70 else 1.0)
        dolly_smooth = dolly_smooth + 0.08 * (target_dolly - dolly_smooth)

        ahead_steps = 14 if local_dt <= 0.35 else 22
        ahead_idx = min(n_smp - 1, idx + ahead_steps)
        mp_ahead = Vector(samples[ahead_idx]["position"])

        # Settle slightly around impact frames so contact is tack-steady
        near_hit = any(abs(f - bf) <= 3 for bf, _, _ in bounce_info)
        alpha = base_alpha * (0.65 if near_hit else 1.0)
        cam_pos_smooth = cam_pos_smooth.lerp(mp, alpha)
        tgt_pos_smooth = tgt_pos_smooth.lerp(0.60 * mp + 0.40 * mp_ahead, alpha * 1.15)

        if f % 2 == 0 or idx == n_smp - 1:
            shake_z = 0.0
            if f in strong_hit_frames:
                shake_z = -0.012
            elif (f - 2) in strong_hit_frames:
                shake_z = 0.005

            if mode == "chase":
                cam_loc = Vector((0.76 * cam_pos_smooth.x + 0.65, -4.10 * dolly_smooth, cam_pos_smooth.z + 0.48 + shake_z))
                tgt_loc = Vector((0.80 * tgt_pos_smooth.x, 0.14, tgt_pos_smooth.z - 0.06))
            elif mode == "chase_wide":
                cam_loc = Vector((0.28 * cam_pos_smooth.x + 0.70, -8.10 * dolly_smooth, 0.28 * cam_pos_smooth.z + 0.55 + shake_z))
                tgt_loc = Vector((0.32 * tgt_pos_smooth.x, 0.12, 0.28 * tgt_pos_smooth.z))
            else:
                # Wheel oblique 3/4 focus cam: ~7% marble height in close runs, pulls back smoothly on 1.2s high arcs
                cam_loc = Vector((0.58 * cam_pos_smooth.x + 0.95, -7.85 * dolly_smooth, 0.58 * cam_pos_smooth.z + 0.92 + shake_z))
                tgt_loc = Vector((0.64 * tgt_pos_smooth.x - 0.06, 0.10, 0.64 * tgt_pos_smooth.z + 0.28))

            cam_obj.location = cam_loc
            tgt.location = tgt_loc
            cam_data.dof.focus_distance = max(1.0, (cam_loc - mp).length)

            f_key = total_f if idx == n_smp - 1 else f
            cam_obj.keyframe_insert(data_path="location", frame=f_key)
            tgt.keyframe_insert(data_path="location", frame=f_key)
            cam_data.dof.keyframe_insert(data_path="focus_distance", frame=f_key)
