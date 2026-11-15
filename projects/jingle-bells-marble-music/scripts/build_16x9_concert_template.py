#!/usr/bin/env python3
"""16:9 (1920x1080 @ 60fps) Standalone Closed-Loop Widescreen Marble Concert Pipeline.

Completely decoupled from the 9:16 vertical pipeline:
- Zero imports from `blender_render_marble_run.py` or `plan_physics_trajectory.py`.
- Reads only static 3D instrument assets (`instrument_library_v2.blend` & `instrument_manifest_v2.json`)
  and note JSONs (defaulting to the first 10.0s of `sample_14s_notes.json`).
- Pitch-ordered two-tier panoramic concert stage (Upper Percussion Gallery & Lower Piano Gallery)
  so neighboring v3.0 instruments never overlap horizontally or vertically.
- Cyclical left-right ballistic bounces for active melodies paired with dedicated obstacle-free
  brass return loops (Overhead Crown Arch, Left 360 Coaster Loop, Left-Lower Pendulum Bowl,
  Mid-Stage Mezzanine Halo Loop, and Right-Flank Ascension Return) across phrase gaps.
- Strict 3D AABB numerical verifier (`verify_zero_intersections`) checking instrument spacing,
  rail-vs-instrument clearance, and marble-vs-non-target-instrument clearance before rendering.
- Refined acoustic strike aesthetic: compact dark walnut/brass Strike Saddle + velocity-coupled
  v3.0 part actuation + localized transient HitSpark point light (ZERO flat ImpactRing circles,
  ZERO giant glowing striker planks).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import bpy
import bmesh
from mathutils import Euler, Matrix, Quaternion, Vector

# ---------------------------------------------------------------------------
# Paths & Standalone 16:9 Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
RENDERS_DIR = os.path.join(PROJECT_DIR, "renders")

LIB_BLEND = os.path.join(ASSETS_DIR, "instrument_library_v2.blend")
MANIFEST_JSON = os.path.join(ASSETS_DIR, "instrument_manifest_v2.json")

FPS = 60
GRAVITY = 24.0
MARBLE_RADIUS = 0.095
FLIGHT_Y = -0.10          # Open front performance lane (Y < +0.02 guarantees clean depth cushion)
SUB_Y_OFFSET = 0.09       # Pushes v3.0 instrument body strictly to Y >= +0.12
ANVIL_LEN = 0.36
ANVIL_WID_Y = 0.22
ANVIL_THICK = 0.036
RAIL_HALF_GAP = 0.068
RAIL_PIPE_R = 0.012
BALLISTIC_MAX_DT = 0.65
PRE_ROLL = 0.60
POST_ROLL = 0.55

INST_DISPLAY_TILT_DEG = 20.0   # Tilts rear +Y upward so camera clearly reads piano keys & lids
UPPER_Z = 0.68
LOWER_Z = -0.74
STATION_XS = [-2.20, -1.10, 0.00, 1.10, 2.20]

LEGACY_TRACK_TO_GM = {
    1: 115, 2: 0, 3: 13, 4: 46, 5: 12, 6: 24,
    7: 14, 8: 73, 9: 56, 10: 40, 11: 21, 12: 80,
}

# Reusable datablock caches for fast procedural geometry generation
_BOX_MESH_CACHE: dict[tuple[float, float, float], bpy.types.Mesh] = {}


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


# ---------------------------------------------------------------------------
# Static v3.0 Instrument Asset Loader (Reads .blend/.json only)
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
            "core_mat": core_obj.data.materials[0] if (core_obj and core_obj.data and core_obj.data.materials) else None,
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


def resolve_gm_program(instrument_name: str, track_idx: int, manifest: dict) -> int:
    name_l = instrument_name.strip().lower()
    aliases = manifest.get("name_aliases", {})
    if name_l in aliases:
        return int(aliases[name_l])
    for gp, info in manifest.get("by_program", {}).items():
        if str(info.get("name", "")).strip().lower() == name_l:
            return int(gp)
    return int(LEGACY_TRACK_TO_GM.get(int(track_idx), 0))


# ---------------------------------------------------------------------------
# Procedural Materials & Architectural Geometry Helpers
# ---------------------------------------------------------------------------

def make_pbr_material(
    name: str,
    base_color: tuple[float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.45,
    coat: float = 0.0,
    emit_color: tuple[float, float, float] | None = None,
    emit_strength: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
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
        bm,
        geom=list(bm.edges),
        offset=bevel_w,
        segments=3,
        profile=0.5,
        affect="EDGES",
        clamp_overlap=True,
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
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=verts,
        radius1=radius,
        radius2=radius,
        depth=dist,
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
# Smooth Ballistic & Closed-Loop Curve Builders (Strictly Obstacle-Clear)
# ---------------------------------------------------------------------------

def hermite_curve(p0: Vector, m0: Vector, p1: Vector, m1: Vector, steps: int) -> list[Vector]:
    pts = []
    for s in range(steps + 1):
        u = s / steps
        u2, u3 = u * u, u * u * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2
        pts.append(p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11)
    return pts


def catmull_rom_chain(waypoints: list[Vector], samples_per_seg: int = 18) -> list[Vector]:
    """Generate a C1 continuous spatial curve passing cleanly through all control waypoints."""
    if len(waypoints) < 2:
        return list(waypoints)
    pts_ext = [waypoints[0] - (waypoints[1] - waypoints[0])] + list(waypoints) + [
        waypoints[-1] + (waypoints[-1] - waypoints[-2])
    ]
    out: list[Vector] = []
    for i in range(1, len(pts_ext) - 2):
        p0, p1, p2, p3 = pts_ext[i - 1], pts_ext[i], pts_ext[i + 1], pts_ext[i + 2]
        m1 = 0.5 * (p2 - p0)
        m2 = 0.5 * (p3 - p1)
        seg = hermite_curve(p1, m1, p2, m2, samples_per_seg)
        if out:
            seg = seg[1:]
        out.extend(seg)
    return out


def resample_by_arc_length(pts: list[Vector], n_out: int) -> tuple[list[Vector], list[float]]:
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
    return out, cum


def eval_catmull_speed_profile(arc_pts: list[Vector], u_frac: float) -> Vector:
    idx_f = max(0.0, min(len(arc_pts) - 1.0, u_frac * (len(arc_pts) - 1)))
    i0 = int(math.floor(idx_f))
    i1 = min(len(arc_pts) - 1, i0 + 1)
    return arc_pts[i0].lerp(arc_pts[i1], idx_f - i0)


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def build_parallel_rail_geometry(
    name_prefix: str,
    spline_pts: list[Vector],
    rail_mat,
    brass_mat,
    dark_mat,
    col_rails: bpy.types.Collection,
) -> dict:
    """Create twin front/back brass rails + architectural tie rings cleanly offset around marble path."""
    left_pts = []
    right_pts = []
    tie_coords = []
    n_pts = len(spline_pts)
    for idx, p in enumerate(spline_pts):
        p_prev = spline_pts[max(0, idx - 1)]
        p_next = spline_pts[min(n_pts - 1, idx + 1)]
        tang = (p_next - p_prev)
        if tang.length < 1e-6:
            tang = Vector((1.0, 0.0, 0.0))
        else:
            tang.normalize()
        # Lateral in-plane normal perpendicular to tangent and Y viewing axis
        up_world = Vector((0.0, 1.0, 0.0))
        binorm = tang.cross(up_world)
        if binorm.length < 1e-6:
            binorm = Vector((0.0, 0.0, 1.0))
        else:
            binorm.normalize()
        offset_y = Vector((0.0, RAIL_HALF_GAP, 0.0))
        # Narrower twin gauge cradling bottom hemisphere of marble (~52 deg contact angle)
        cradle = p - binorm * (MARBLE_RADIUS * 0.72)
        gauge = 0.052
        left_pts.append([cradle.x, cradle.y - gauge, cradle.z])
        right_pts.append([cradle.x, cradle.y + gauge, cradle.z])
        if idx % 7 == 3 or idx == 0 or idx == n_pts - 1:
            tie_coords.append((
                Vector(left_pts[-1]),
                Vector(right_pts[-1]),
            ))

    create_tube_polyline(f"{name_prefix}_RailFront", left_pts, RAIL_PIPE_R, rail_mat, col_rails)
    create_tube_polyline(f"{name_prefix}_RailBack", right_pts, RAIL_PIPE_R, rail_mat, col_rails)
    for k, (pa, pb) in enumerate(tie_coords):
        tie_mat = brass_mat if k % 3 == 0 else dark_mat
        create_cylinder_between(f"{name_prefix}_Tie_{k:02d}", pa, pb, RAIL_PIPE_R * 0.62, tie_mat, col_rails, verts=10)

    # Coaxial flared brass cuffs aligned smoothly to rail exit/entry tangents (NO flat floating coins!)
    for tag, idx_end, idx_nbr in (("EntryCollar", 0, 1), ("DropNozzle", -1, -2)):
        c_pos = 0.5 * (Vector(left_pts[idx_end]) + Vector(right_pts[idx_end]))
        tang = (spline_pts[idx_nbr] - spline_pts[idx_end])
        if idx_end == -1:
            tang = -tang
        tang = tang.normalized() if tang.length > 1e-5 else Vector((0.0, 0.0, -1.0))
        tmesh = bpy.data.meshes.new(f"{name_prefix}_{tag}Mesh")
        tbm = bmesh.new()
        bmesh.ops.create_cone(
            tbm,
            cap_ends=False,
            cap_tris=False,
            segments=20,
            radius1=MARBLE_RADIUS * 0.72,
            radius2=MARBLE_RADIUS * 0.88,
            depth=0.044,
        )
        bmesh.ops.solidify(tbm, geom=list(tbm.faces), thickness=0.006)
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

    # Dark bronze rear wall standoffs anchoring brass loops strictly outside instrument columns
    n_struts = max(2, min(3, n_pts // 55))
    for s_i in range(n_struts):
        u_idx = int(round((s_i + 0.5) / n_struts * (n_pts - 1)))
        p_rail_back = Vector(right_pts[u_idx])
        if abs(p_rail_back.x) > 2.76 or p_rail_back.z > 1.24 or abs(p_rail_back.z) < 0.34:
            p_wall = Vector((p_rail_back.x, 0.90, p_rail_back.z))
            create_cylinder_between(
                f"{name_prefix}_Anchor_{s_i:02d}",
                p_rail_back,
                p_wall,
                0.008,
                dark_mat,
                col_rails,
                verts=10,
            )

    return {
        "name": name_prefix,
        "left": left_pts,
        "right": right_pts,
        "center": [[p.x, p.y, p.z] for p in spline_pts],
    }


# ---------------------------------------------------------------------------
# Strict 3D AABB Zero-Intersection Verifier
# ---------------------------------------------------------------------------

def compute_station_world_aabbs(
    stations: list[dict],
    templates: dict[int, dict],
    manifest: dict,
) -> list[dict]:
    """Compute exact 3D world-space AABB for every stationary instrument sculpture + strike anvil."""
    aabbs = []
    for st in stations:
        gm = resolve_gm_program(st["instrument"], st["track_idx"], manifest)
        tpl = templates.get(gm) or templates.get(-1000 - st["track_idx"]) or next(iter(templates.values()))
        u = Vector(st["axis_u"]).normalized()
        v = Vector(st["axis_v"]).normalized()
        n = Vector(st["axis_n"]).normalized()
        rot_mat = Matrix(((u.x, v.x, n.x), (u.y, v.y, n.y), (u.z, v.z, n.z)))
        origin = Vector(st["center"])

        # Core anvil saddle vertices
        hx, hy, hz = ANVIL_LEN * 0.5, ANVIL_WID_Y * 0.5, ANVIL_THICK * 0.5
        anvil_pts = [
            origin + (rot_mat @ Vector((sx * hx, sy * hy, sz * hz)))
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
        inst_pts = []
        tilt_m = Matrix.Rotation(math.radians(INST_DISPLAY_TILT_DEG), 3, "X")
        pivot_off = Vector((0.0, SUB_Y_OFFSET, 0.0))
        for _sn, sdata, _sm, sloc, srot, sscale in tpl["subs"]:
            if not hasattr(sdata, "vertices") or len(sdata.vertices) == 0:
                continue
            sub_rot_m = Euler(srot, "XYZ").to_matrix()
            stride = max(1, len(sdata.vertices) // 64)
            for vi in range(0, len(sdata.vertices), stride):
                vert = sdata.vertices[vi]
                v_scaled = Vector((vert.co.x * sscale.x, vert.co.y * sscale.y, vert.co.z * sscale.z))
                local_inst = pivot_off + (tilt_m @ (sloc + (sub_rot_m @ v_scaled)))
                inst_pts.append(origin + (rot_mat @ local_inst))

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


def verify_zero_intersections(
    aabbs: list[dict],
    rails_geom: list[dict],
    samples: list[dict],
    bounces: list[dict],
) -> None:
    """Strictly validate zero instrument overlap, zero rail clipping, and zero marble clipping."""
    # 1. Pairwise station instrument AABB check (full occupied bounding boxes)
    min_pair_gap = 999.0
    for i in range(len(aabbs)):
        for j in range(i + 1, len(aabbs)):
            gap = aabb_gap(aabbs[i]["min"], aabbs[i]["max"], aabbs[j]["min"], aabbs[j]["max"])
            min_pair_gap = min(min_pair_gap, gap)
            if gap < 0.06:
                raise RuntimeError(
                    f"[INTERSECTION] Station {aabbs[i]['label']} overlaps {aabbs[j]['label']} (gap={gap:.4f}m)"
                )

    # 2. Rail polyline vs REAL solids (actual anvil box + actual v3.0 instrument body box)
    min_rail_clear = 999.0
    for rg in rails_geom:
        for side_key in ("left", "right"):
            for pt in rg[side_key]:
                pv = Vector(pt)
                for box in aabbs:
                    d_anvil = point_to_aabb_distance(pv, box["anvil_min"], box["anvil_max"]) - RAIL_PIPE_R
                    d_inst = point_to_aabb_distance(pv, box["inst_min"], box["inst_max"]) - RAIL_PIPE_R
                    d = min(d_anvil, d_inst)
                    min_rail_clear = min(min_rail_clear, d)
                    if d < 0.05:
                        raise RuntimeError(
                            f"[INTERSECTION] Rail {rg['name']} ({side_key}) clips {box['label']} at {pt} (clear={d:.4f}m)"
                        )

    # 3. Marble trajectory vs REAL solids (v3.0 body AABBs + non-target anvil AABBs)
    min_marble_inst_clear = 999.0
    for smp in samples:
        mp = Vector(smp["position"])
        t = float(smp["t"])
        near_targets = {
            b["station_id"] for b in bounces if abs(float(b["t"]) - t) <= 0.14
        }
        for box in aabbs:
            d_inst = point_to_aabb_distance(mp, box["inst_min"], box["inst_max"]) - MARBLE_RADIUS
            min_marble_inst_clear = min(min_marble_inst_clear, d_inst)
            if d_inst < 0.02:
                raise RuntimeError(
                    f"[INTERSECTION] Marble clips instrument body {box['label']} at t={t:.3f}s (clear={d_inst:.4f}m)"
                )
            if box["station_id"] not in near_targets:
                d_anvil = point_to_aabb_distance(mp, box["anvil_min"], box["anvil_max"]) - MARBLE_RADIUS
                if d_anvil < 0.02:
                    raise RuntimeError(
                        f"[INTERSECTION] Marble clips non-target anvil {box['label']} at t={t:.3f}s (clear={d_anvil:.4f}m)"
                    )

    print(
        f"[ZeroClip OK] stations={len(aabbs)}, rails={len(rails_geom)}, samples={len(samples)} | "
        f"min_station_gap={min_pair_gap:.3f}m, min_rail_clear={min_rail_clear:.3f}m, "
        f"min_marble_inst_clear={min_marble_inst_clear:.3f}m"
    )


# ---------------------------------------------------------------------------
# 16:9 Closed-Loop Trajectory Planner (First 10s Test Notes)
# ---------------------------------------------------------------------------

def build_stations_for_notes(notes_10s: list[dict]) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    """Map notes cleanly onto two vertically separated horizontal pitch tiers (5 Upper + 5 Lower)."""
    tier1_pitches = sorted({int(n["note"]) for n in notes_10s if int(n.get("track_idx", 1)) == 1})
    tier2_pitches = sorted({int(n["note"]) for n in notes_10s if int(n.get("track_idx", 1)) == 2})

    stations: list[dict] = []
    lookup: dict[tuple[int, int], dict] = {}
    sid = 0

    for tier_idx, (trk, plist, z_base, default_inst) in enumerate([
        (1, tier1_pitches, UPPER_Z, "Castanets"),
        (2, tier2_pitches, LOWER_Z, "Piano"),
    ]):
        n_p = len(plist)
        xs = STATION_XS if n_p == 5 else [
            -2.20 + 4.40 * (i / max(1, n_p - 1)) for i in range(n_p)
        ]
        for i, midi in enumerate(plist):
            sample_n = next(
                (n for n in notes_10s if int(n.get("track_idx", 1)) == trk and int(n["note"]) == midi),
                None,
            )
            inst_name = sample_n["instrument"] if sample_n else default_inst
            p_name = sample_n.get("pitch_name", f"M{midi}") if sample_n else f"M{midi}"
            cx = xs[i]
            cz = z_base + 0.12 * ((cx / 2.20) ** 2)
            contact = [cx, FLIGHT_Y, cz + ANVIL_THICK * 0.5]
            center = [cx, FLIGHT_Y, cz]
            st = {
                "id": sid,
                "tier": tier_idx,
                "track_idx": trk,
                "instrument": inst_name,
                "note": midi,
                "pitch_name": p_name,
                "center": center,
                "contact": contact,
                "axis_u": [1.0, 0.0, 0.0],
                "axis_v": [0.0, 1.0, 0.0],
                "axis_n": [0.0, 0.0, 1.0],
                "hits": [],
            }
            stations.append(st)
            lookup[(trk, midi)] = st
            sid += 1

    return stations, lookup


def solve_ballistic_pos(p0: Vector, p1: Vector, dt: float, u: float) -> Vector:
    """Exact 3D projectile trajectory from p0 to p1 in duration dt under GRAVITY along -Z."""
    t = u * dt
    vz0 = ((p1.z - p0.z) / max(1e-5, dt)) + 0.5 * GRAVITY * dt
    vx = (p1.x - p0.x) / max(1e-5, dt)
    vy = (p1.y - p0.y) / max(1e-5, dt)
    return Vector((
        p0.x + vx * t,
        p0.y + vy * t,
        p0.z + vz0 * t - 0.5 * GRAVITY * t * t,
    ))


def plan_loop_waypoints(seg_idx: int, p0: Vector, p1: Vector, dt: float, n_next: dict) -> tuple[list[Vector], float, float]:
    """Design clean, sculptural, non-intersecting closed-loop return curves for long gaps (dt > 0.65s).

    Returns (rail_waypoints, air_launch_dt, air_drop_dt) where:
    - During `[0, air_launch_dt]`, the marble rebounds smoothly in open air off `p0` to `rail_waypoints[0]`.
    - During `[air_launch_dt, dt - air_drop_dt]`, it rides the curved brass return loop `rail_waypoints`.
    - During `[dt - air_drop_dt, dt]`, it exits cleanly out of the drop nozzle `rail_waypoints[-1]` in free-fall onto `p1`.
    """
    y = FLIGHT_Y
    # Loop #1: Note 4 (Upper E5 at X=0.0) -> Note 5 (Upper G5 at X=+2.20), dt=1.225s
    # Overhead Crown Orbital Arch curving high above Upper Castanets (Z <= 1.66, well below Brand Title!)
    if p0.z > 0.0 and p1.z > 0.0:
        launch_dt = 0.14
        drop_dt = 0.16
        w_start = Vector((p0.x + 0.18, y, p0.z + 0.42))
        w_end = Vector((p1.x, y, p1.z + 0.44))
        wp = [
            w_start,
            Vector((p0.x + 0.48, y, 1.44)),
            Vector((1.00, y, 1.66)),
            Vector((1.72, y, 1.65)),
            Vector((p1.x - 0.05, y + 0.02, p1.z + 0.64)),
            w_end,
        ]
        return wp, launch_dt, drop_dt

    # Loop #2: Upper C5 (X=-2.20) -> Lower G4 (X=-2.20), dt=1.224s
    # Full 360-degree Left-Flank Helical Coaster Loop (depth-separated in Y so zero rail self-crossing!)
    if p0.z > 0.0 and p1.z < 0.0:
        launch_dt = 0.12
        drop_dt = 0.14
        cx, cz, r_loop = -3.22, 0.14, 0.48
        wp = [
            Vector((p0.x - 0.50, y, p0.z + 0.22)),
            Vector((cx + r_loop * 0.22, y + 0.04, cz + r_loop * 0.96)),
            Vector((cx - r_loop * 0.96, y + 0.02, cz + r_loop * 0.32)),
            Vector((cx - r_loop * 0.92, y - 0.03, cz - r_loop * 0.55)),
            Vector((cx + r_loop * 0.35, y - 0.08, cz - r_loop * 0.94)),
            Vector((cx + r_loop * 0.94, y - 0.11, cz + r_loop * 0.15)),
            Vector((cx + r_loop * 0.12, y - 0.11, cz + r_loop * 0.84)),
            Vector((cx - r_loop * 0.48, y - 0.08, cz - r_loop * 0.25)),
            Vector((-2.78, y - 0.04, -0.26)),
            Vector((p1.x, y, p1.z + 0.36)),
        ]
        return wp, launch_dt, drop_dt

    # Loop #3: Lower G4 (X=-2.20) -> Lower G4 (X=-2.20), dt=0.918s
    # Left-Lower Pendulum Loop Bowl clearly separated from Loop #2 crossover
    if abs(p0.x - (-2.20)) < 0.25 and p0.z < 0.0:
        launch_dt = 0.12
        drop_dt = 0.16
        wp = [
            Vector((-2.92, y - 0.02, -0.58)),
            Vector((-3.46, y - 0.04, -0.84)),
            Vector((-3.48, y - 0.08, -1.28)),
            Vector((-3.02, y - 0.11, -1.28)),
            Vector((-2.98, y - 0.08, -0.76)),
            Vector((-2.92, y - 0.02, -0.44)),
        ]
        return wp, launch_dt, drop_dt

    # Loop #4: Lower A4 (X=-1.10) -> Lower A4 (X=-1.10), dt=1.225s
    # Mid-Stage Mezzanine Helical Halo Loop (smooth depth corkscrew avoids self-intersection!)
    launch_dt = 0.14
    drop_dt = 0.16
    cx, cz, r_mid = -0.52, 0.10, 0.34
    wp = [
        Vector((p0.x + 0.14, y, -0.24)),
        Vector((cx - r_mid * 0.85, y + 0.03, cz + r_mid * 0.45)),
        Vector((cx, y + 0.06, cz + r_mid)),
        Vector((cx + r_mid, y + 0.02, cz)),
        Vector((cx, y - 0.06, cz - r_mid)),
        Vector((cx - r_mid * 0.90, y - 0.08, cz - r_mid * 0.18)),
        Vector((p1.x, y, -0.18)),
    ]
    return wp, launch_dt, drop_dt


def plan_16x9_closed_loop_physics(notes_json_path: str, max_note_t: float = 10.0) -> dict:
    with open(notes_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_notes = raw["notes"] if isinstance(raw, dict) and "notes" in raw else raw
    notes_10s = sorted(
        [n for n in raw_notes if float(n["start_time"]) <= max_note_t + 1e-6],
        key=lambda n: float(n["start_time"]),
    )
    stations, lookup = build_stations_for_notes(notes_10s)

    bounces: list[dict] = []
    for k, n in enumerate(notes_10s):
        trk = int(n.get("track_idx", 1))
        midi = int(n["note"])
        st = lookup[(trk, midi)]
        t_vid = PRE_ROLL + float(n["start_time"])
        f_hit = int(round(t_vid * FPS))
        contact = Vector(st["contact"]) + Vector((0.0, 0.0, MARBLE_RADIUS))
        st["hits"].append({
            "note_index": k,
            "frame_hit": f_hit,
            "t": t_vid,
            "velocity": float(n.get("velocity", 110)),
            "duration": float(n.get("duration", 0.25)),
            "note": midi,
            "contact": [contact.x, contact.y, contact.z],
        })
        bounces.append({
            "note_index": k,
            "station_id": st["id"],
            "track_idx": trk,
            "instrument": st["instrument"],
            "note": midi,
            "pitch_name": st["pitch_name"],
            "t": t_vid,
            "t_note": float(n["start_time"]),
            "frame": f_hit,
            "marble_pos": [contact.x, contact.y, contact.z],
            "station_center": list(st["center"]),
        })

    rails_plan: list[dict] = []
    segments: list[dict] = []

    # 1. Opening Pre-Roll Loop from Upper-Left Launch Turret -> Upper E5 (bounces[0])
    p_n0 = Vector(bounces[0]["marble_pos"])
    open_drop_dt = 0.15
    open_wp = [
        Vector((-3.18, FLIGHT_Y, 1.38)),
        Vector((-2.62, FLIGHT_Y, 1.56)),
        Vector((-1.65, FLIGHT_Y, 1.48)),
        Vector((-0.65, FLIGHT_Y, 1.34)),
        Vector((p_n0.x, FLIGHT_Y, p_n0.z + 0.36)),
    ]
    open_arc, _ = resample_by_arc_length(catmull_rom_chain(open_wp, samples_per_seg=18), 160)
    rails_plan.append({"name": "Loop0_OpeningLaunch", "waypoints": open_wp, "arc_pts": open_arc})
    segments.append({
        "type": "rail_drop",
        "t0": 0.0,
        "t1": PRE_ROLL,
        "rail_end_t": PRE_ROLL - open_drop_dt,
        "arc_pts": open_arc,
        "p_target": p_n0,
    })

    # 2. All Note-to-Note Transitions
    loop_count = 0
    for k in range(len(bounces) - 1):
        b0, b1 = bounces[k], bounces[k + 1]
        p0 = Vector(b0["marble_pos"])
        p1 = Vector(b1["marble_pos"])
        t0, t1 = float(b0["t"]), float(b1["t"])
        dt = t1 - t0
        if dt <= BALLISTIC_MAX_DT:
            segments.append({
                "type": "ballistic",
                "t0": t0,
                "t1": t1,
                "p0": p0,
                "p1": p1,
            })
        else:
            loop_count += 1
            wp, launch_dt, drop_dt = plan_loop_waypoints(k, p0, p1, dt, bounces[k + 1])
            arc_pts, _ = resample_by_arc_length(catmull_rom_chain(wp, samples_per_seg=16), 200)
            rails_plan.append({"name": f"Loop{loop_count}_Bridge", "waypoints": wp, "arc_pts": arc_pts})
            segments.append({
                "type": "loop_chute",
                "t0": t0,
                "t1": t1,
                "p0": p0,
                "p1": p1,
                "launch_dt": launch_dt,
                "drop_dt": drop_dt,
                "arc_pts": arc_pts,
            })

    # 3. Closing Perpetual Return Tail (Right-Flank Ascension Track starting smoothly rightward out of A4)
    p_last = Vector(bounces[-1]["marble_pos"])
    t_last = float(bounces[-1]["t"])
    t_end = t_last + POST_ROLL
    close_launch_dt = 0.15
    close_wp = [
        Vector((0.20, FLIGHT_Y, -0.18)),
        Vector((1.35, FLIGHT_Y, -0.12)),
        Vector((2.78, FLIGHT_Y, -0.04)),
        Vector((3.22, FLIGHT_Y, 0.58)),
        Vector((2.92, FLIGHT_Y, 1.38)),
    ]
    close_arc, _ = resample_by_arc_length(catmull_rom_chain(close_wp, samples_per_seg=16), 160)
    rails_plan.append({"name": "Loop5_ClosingReturn", "waypoints": close_wp, "arc_pts": close_arc})
    segments.append({
        "type": "launch_rail",
        "t0": t_last,
        "t1": t_end,
        "launch_dt": close_launch_dt,
        "p0": p_last,
        "arc_pts": close_arc,
    })

    # Dense 60fps + sub-frame sampling at 120Hz for collision safety and clean keyframing
    total_frames = int(math.ceil(t_end * FPS))
    samples: list[dict] = []
    for sub_i in range(total_frames * 2 + 1):
        t = min(t_end, sub_i / (FPS * 2.0))
        seg = next(
            (s for s in segments if s["t0"] - 1e-7 <= t <= s["t1"] + 1e-7),
            segments[-1],
        )
        pos = _eval_segment(seg, t)
        samples.append({
            "frame_f": t * FPS,
            "t": round(t, 5),
            "position": [round(pos.x, 5), round(pos.y, 5), round(pos.z, 5)],
        })

    frame_samples = samples[::2]
    return {
        "schema": "concert_16x9_closed_loop_v1",
        "fps": FPS,
        "duration_s": round(t_end, 4),
        "total_frames": total_frames,
        "pre_roll_s": PRE_ROLL,
        "gravity": GRAVITY,
        "marble_radius": MARBLE_RADIUS,
        "stations": stations,
        "bounces": bounces,
        "rails_plan": [
            {"name": r["name"], "points": [[p.x, p.y, p.z] for p in r["arc_pts"]]}
            for r in rails_plan
        ],
        "frame_samples": frame_samples,
        "dense_samples": samples,
    }


def _eval_segment(seg: dict, t: float) -> Vector:
    t0, t1 = float(seg["t0"]), float(seg["t1"])
    dt = max(1e-5, t1 - t0)
    stype = seg["type"]
    if stype == "ballistic":
        u = max(0.0, min(1.0, (t - t0) / dt))
        return solve_ballistic_pos(seg["p0"], seg["p1"], dt, u)
    if stype == "rail_drop":
        rail_end = float(seg["rail_end_t"])
        arc = seg["arc_pts"]
        if t <= rail_end:
            u = smoothstep((t - t0) / max(1e-5, rail_end - t0))
            # Lead gently faster as gravity slides marble down launch ramp
            u_accel = 0.25 * u + 0.75 * (u * u)
            return eval_catmull_speed_profile(arc, u_accel)
        u_drop = (t - rail_end) / max(1e-5, t1 - rail_end)
        return solve_ballistic_pos(arc[-1], seg["p_target"], t1 - rail_end, u_drop)
    if stype == "loop_chute":
        ldt = float(seg["launch_dt"])
        ddt = float(seg["drop_dt"])
        arc = seg["arc_pts"]
        t_enter = t0 + ldt
        t_exit = t1 - ddt
        if t <= t_enter:
            u = (t - t0) / max(1e-5, ldt)
            return solve_ballistic_pos(seg["p0"], arc[0], ldt, u)
        if t >= t_exit:
            u = (t - t_exit) / max(1e-5, ddt)
            return solve_ballistic_pos(arc[-1], seg["p1"], ddt, u)
        u = (t - t_enter) / max(1e-5, t_exit - t_enter)
        # Smooth ease in/out inside coaster loop
        return eval_catmull_speed_profile(arc, 0.85 * u + 0.15 * smoothstep(u))
    if stype == "launch_rail":
        ldt = float(seg["launch_dt"])
        arc = seg["arc_pts"]
        t_enter = t0 + ldt
        if t <= t_enter:
            u = (t - t0) / max(1e-5, ldt)
            return solve_ballistic_pos(seg["p0"], arc[0], ldt, u)
        u = (t - t_enter) / max(1e-5, t1 - t_enter)
        return eval_catmull_speed_profile(arc, min(1.0, u * 0.88))
    return seg.get("p0", Vector((0, 0, 0)))


# ---------------------------------------------------------------------------
# Environment, Refined Concert Hall, Instrument Rigging & Animation
# ---------------------------------------------------------------------------

def build_concert_world(scene: bpy.types.Scene) -> None:
    world = bpy.data.worlds.new("ConcertStageWorld16x9")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    lp = nt.nodes.new("ShaderNodeLightPath")

    bg_diff = nt.nodes.new("ShaderNodeBackground")
    bg_diff.inputs["Color"].default_value = (0.015, 0.016, 0.024, 1.0)
    bg_diff.inputs["Strength"].default_value = 0.42

    bg_gloss = nt.nodes.new("ShaderNodeBackground")
    bg_gloss.inputs["Strength"].default_value = 1.0
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Normal"], sep.inputs["Vector"])

    # Warm overhead key studio reflection strip
    r_top = nt.nodes.new("ShaderNodeValToRGB")
    r_top.color_ramp.elements[0].position = 0.32
    r_top.color_ramp.elements[0].color = (0.008, 0.009, 0.014, 1.0)
    r_top.color_ramp.elements[1].position = 0.52
    r_top.color_ramp.elements[1].color = (4.4, 3.9, 3.2, 1.0)
    nt.links.new(sep.outputs["Z"], r_top.inputs["Fac"])

    # Cool sapphire side reflection rim
    r_side = nt.nodes.new("ShaderNodeValToRGB")
    r_side.color_ramp.elements[0].position = 0.56
    r_side.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    r_side.color_ramp.elements[1].position = 0.72
    r_side.color_ramp.elements[1].color = (2.0, 2.7, 3.6, 1.0)
    nt.links.new(sep.outputs["X"], r_side.inputs["Fac"])

    add1 = nt.nodes.new("ShaderNodeMix")
    add1.data_type = "RGBA"
    add1.blend_type = "ADD"
    add1.inputs["Factor"].default_value = 1.0
    nt.links.new(r_top.outputs["Color"], add1.inputs["A"])
    nt.links.new(r_side.outputs["Color"], add1.inputs["B"])

    nt.links.new(add1.outputs["Result"], bg_gloss.inputs["Color"])
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
    set_socket(glare, "Smoothness", 0.38)
    set_socket(glare, "Strength", 0.42)
    set_socket(glare, "Size", 0.58)
    ng.links.new(rl.outputs["Image"], glare.inputs["Image"])
    ng.links.new(glare.outputs["Image"], out.inputs["Image"])


def create_panoramic_concert_hall(col_stage: bpy.types.Collection, mats: dict[str, bpy.types.Material]) -> None:
    """Clean, warm architectural concert shell with smooth walnut & brass framing (no busy grids)."""
    wall_mat = make_pbr_material("StageObsidianBackdrop", (0.018, 0.019, 0.026), metallic=0.04, roughness=0.72)
    walnut_mat = mats.get("LibMat_Walnut") or make_pbr_material("StageWalnut", (0.14, 0.08, 0.04), roughness=0.38)
    brass_mat = mats.get("LibMat_Brass") or make_pbr_material("StageBrass", (0.86, 0.65, 0.24), metallic=0.95, roughness=0.22)
    cove_emit = make_pbr_material(
        "StageWarmCove", (0.95, 0.78, 0.50), metallic=0.0, roughness=0.2,
        emit_color=(1.0, 0.80, 0.48), emit_strength=1.15,
    )

    # 1. Clean curved concave backdrop shell well behind ALL instruments (Y >= +0.86)
    smesh = bpy.data.meshes.new("PanoramicBackdropMesh")
    bm = bmesh.new()
    segs_x, segs_z = 36, 4
    w_span, z_bot, z_top = 12.0, -3.2, 3.8
    grid_v = []
    for iz in range(segs_z + 1):
        zf = iz / segs_z
        zc = z_bot + zf * (z_top - z_bot)
        row = []
        for ix in range(segs_x + 1):
            xf = (ix / segs_x) * 2.0 - 1.0
            xc = xf * (w_span * 0.5)
            yc = 1.02 - 0.14 * (1.0 - xf * xf)  # gently curved cyclorama behind Y=+0.88
            row.append(bm.verts.new((xc, yc, zc)))
        grid_v.append(row)
    for iz in range(segs_z):
        for ix in range(segs_x):
            bm.faces.new((grid_v[iz][ix], grid_v[iz][ix + 1], grid_v[iz + 1][ix + 1], grid_v[iz + 1][ix]))
    bm.to_mesh(smesh)
    bm.free()
    for p in smesh.polygons:
        p.use_smooth = True
    bobj = bpy.data.objects.new("PanoramicBackdrop", smesh)
    col_stage.objects.link(bobj)
    bobj.data.materials.append(wall_mat)

    # 2. Horizontal Stage Decks supporting Upper & Lower Tiers cleanly from behind (Y in [+0.20, +0.84])
    for name, z_deck, width_x in (
        ("UpperTierDeck", UPPER_Z - 0.14, 5.80),
        ("LowerTierDeck", LOWER_Z - 0.14, 5.80),
        ("LowerStagePlinth", -1.58, 7.60),
    ):
        dmesh = create_beveled_box_mesh(name + "Mesh", width_x, 0.56, 0.072, bevel_w=0.012)
        dobj = bpy.data.objects.new(name, dmesh)
        col_stage.objects.link(dobj)
        dobj.location = (0.0, 0.46, z_deck)
        dobj.data.materials.append(walnut_mat)

        # Thin warm brass edge molding along front lip of shelf
        trim = create_beveled_box_mesh(name + "BrassMesh", width_x, 0.018, 0.016, bevel_w=0.003)
        tobj = bpy.data.objects.new(name + "_BrassTrim", trim)
        col_stage.objects.link(tobj)
        tobj.location = (0.0, 0.18, z_deck + 0.028)
        tobj.data.materials.append(brass_mat)

    # 3. Outer Proscenium Walnut & Brass Columns framing the widescreen stage (|X| = 3.95, never obstructing track)
    for side in (-1, 1):
        px = side * 3.95
        pmesh = create_beveled_box_mesh(f"ProsceniumPillar_{side}", 0.24, 0.42, 4.40, bevel_w=0.018)
        pobj = bpy.data.objects.new(f"ProsceniumPillar_{side}", pmesh)
        col_stage.objects.link(pobj)
        pobj.location = (px, 0.42, 0.28)
        pobj.data.materials.append(walnut_mat)

        cmesh = create_beveled_box_mesh(f"ProsceniumCove_{side}", 0.018, 0.018, 3.90, bevel_w=0.002)
        cobj = bpy.data.objects.new(f"ProsceniumCove_{side}", cmesh)
        col_stage.objects.link(cobj)
        cobj.location = (px - side * 0.125, 0.22, 0.28)
        cobj.data.materials.append(cove_emit)

    # 4. Warm Top Architectural Cove & Brass Marquee "LUCA MARBLE SYMPHONY" high above Upper Loop (Z=+2.16)
    arch_m = create_beveled_box_mesh("UpperCoveLightMesh", 7.60, 0.025, 0.020, bevel_w=0.003)
    arch_o = bpy.data.objects.new("UpperCoveLight", arch_m)
    col_stage.objects.link(arch_o)
    arch_o.location = (0.0, 0.44, 2.44)
    arch_o.data.materials.append(cove_emit)

    bpy.ops.object.text_add(location=(0.0, 0.48, 2.16))
    txt = bpy.context.active_object
    txt.name = "MarqueeBrandTitle"
    txt.data.body = "LUCA  ·  MARBLE SYMPHONY"
    txt.data.align_x = "CENTER"
    txt.data.align_y = "CENTER"
    txt.data.size = 0.20
    txt.data.extrude = 0.016
    txt.data.bevel_depth = 0.003
    txt.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    txt.data.materials.append(brass_mat)
    for c in list(txt.users_collection):
        c.objects.unlink(txt)
    col_stage.objects.link(txt)


def create_station_pedestal_and_pitch_tag(
    st: dict,
    col_stage: bpy.types.Collection,
    brass_mat,
    walnut_mat,
    pedestal_mat,
) -> None:
    """Mount small physical support strut from deck + subtle brass pitch tag on front apron."""
    x, y, z = st["center"]
    z_deck = (UPPER_Z if st["tier"] == 0 else LOWER_Z) - 0.14
    h_strut = max(0.05, (z - z_deck) - ANVIL_THICK * 0.4)
    if h_strut > 0.04:
        pmesh = create_beveled_box_mesh(f"PedestalStrut_{st['id']:02d}", 0.18, 0.16, h_strut, bevel_w=0.008)
        pobj = bpy.data.objects.new(f"PedestalStrut_{st['id']:02d}", pmesh)
        col_stage.objects.link(pobj)
        pobj.location = (x, y + 0.18, z_deck + 0.036 + h_strut * 0.5)
        pobj.data.materials.append(pedestal_mat)

    # Tiny crisp brass pitch plaque on the lower front rim of the saddle (never blocking marble flight)
    bpy.ops.object.text_add(location=(x, y - 0.118, z - 0.006))
    t_obj = bpy.context.active_object
    t_obj.name = f"PitchTag_{st['id']:02d}_{st['pitch_name']}"
    t_obj.data.body = st["pitch_name"]
    t_obj.data.align_x = "CENTER"
    t_obj.data.align_y = "CENTER"
    t_obj.data.size = 0.056
    t_obj.data.extrude = 0.004
    t_obj.rotation_euler = (math.radians(82.0), 0.0, 0.0)
    t_obj.data.materials.append(brass_mat)
    for c in list(t_obj.users_collection):
        c.objects.unlink(t_obj)
    col_stage.objects.link(t_obj)


# ---------------------------------------------------------------------------
# Refined Bespoke Instrument Actuation & Multi-Hit Station Spawner (NO RINGS)
# ---------------------------------------------------------------------------

def _filter_parts(spawned_subs: list[dict], *tokens: str) -> list[dict]:
    return [sp for sp in spawned_subs if any(tok in sp["base_name"] for tok in tokens)]


def _pick_pitch_part(parts: list[dict], pitch_frac: float) -> dict | None:
    if not parts:
        return None
    ordered = sorted(parts, key=lambda p: p["orig_loc"].x)
    idx = max(0, min(len(ordered) - 1, int(round(pitch_frac * (len(ordered) - 1)))))
    return ordered[idx]


def _should_highlight_part(sub_name: str) -> bool:
    tokens = (
        "_WKey_", "_BKey_", "_ToneBar_", "_SteelBar_", "_Str_", "_HarpStr_",
        "_BellCup", "_GongDish", "_GongBoss", "_Chime", "_MalletHead", "_MalletYarn",
        "_MalletBall", "_HammerHead", "_Clapper", "_Knob", "_Pad", "_VU_", "_Strip",
    )
    return any(tok in sub_name for tok in tokens)


def _keyframe_part_transform(
    part: dict,
    keyframes: list[tuple[int, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    pivot: Vector | None = None,
) -> None:
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


def spawn_concert_station(
    st: dict,
    templates: dict[int, dict],
    mats: dict[str, bpy.types.Material],
    manifest: dict,
    col_inst: bpy.types.Collection,
    col_fx: bpy.types.Collection,
) -> None:
    sid = int(st["id"])
    gm = resolve_gm_program(st["instrument"], st["track_idx"], manifest)
    tpl = templates.get(gm) or templates.get(-1000 - st["track_idx"]) or next(iter(templates.values()))
    anim_family = str(tpl.get("anim_family", "generic_strike"))
    glow_color = tuple(tpl.get("glow_color", (1.0, 0.78, 0.30)))

    # 1. Compact Satin Walnut-Bronze Strike Anvil (ZERO neon brick!)
    anvil_mesh = create_beveled_box_mesh(f"AnvilMesh_{sid:02d}", ANVIL_LEN, ANVIL_WID_Y, ANVIL_THICK, bevel_w=0.008)
    anvil_obj = bpy.data.objects.new(f"Station_{sid:02d}_{st['instrument']}_{st['pitch_name']}", anvil_mesh)
    col_inst.objects.link(anvil_obj)

    anvil_mat = make_pbr_material(
        f"AnvilMat_{sid:02d}",
        base_color=(0.11, 0.08, 0.06) if st["track_idx"] == 1 else (0.06, 0.07, 0.09),
        metallic=0.62,
        roughness=0.26,
        coat=0.35,
    )
    anvil_obj.data.materials.clear()
    anvil_obj.data.materials.append(anvil_mat)

    # Small recessed satin-brass contact cap inlaid flush on top of anvil
    cap_mesh = create_beveled_box_mesh(f"CapMesh_{sid:02d}", 0.14, 0.10, 0.008, bevel_w=0.002)
    cap_obj = bpy.data.objects.new(f"StrikeCap_{sid:02d}", cap_mesh)
    col_inst.objects.link(cap_obj)
    cap_obj.parent = anvil_obj
    cap_obj.location = (0.0, 0.0, ANVIL_THICK * 0.5 + 0.002)
    cap_mat = make_pbr_material(
        f"CapMat_{sid:02d}",
        base_color=(0.88, 0.66, 0.26),
        metallic=0.92,
        roughness=0.18,
    )
    cap_obj.data.materials.append(cap_mat)

    # 2. Spawn sculpted v3.0 instrument tilted cleanly toward camera on InstDisplayPivot behind anvil
    inst_pivot = bpy.data.objects.new(f"InstDisplayPivot_{sid:02d}", None)
    col_inst.objects.link(inst_pivot)
    inst_pivot.parent = anvil_obj
    inst_pivot.location = (0.0, SUB_Y_OFFSET, 0.0)
    inst_pivot.rotation_mode = "XYZ"
    inst_pivot.rotation_euler = (math.radians(INST_DISPLAY_TILT_DEG), 0.0, 0.0)

    spawned_subs: list[dict] = []
    local_glow_cache: dict[str, tuple] = {}
    for sub_name, sub_data, sub_mats, sub_loc, sub_rot, sub_scale in tpl["subs"]:
        sobj = bpy.data.objects.new(f"{sub_name}_{sid:02d}", sub_data)
        col_inst.objects.link(sobj)
        sobj.parent = inst_pivot
        sobj.location = sub_loc
        sobj.rotation_mode = "XYZ"
        sobj.rotation_euler = sub_rot
        sobj.scale = sub_scale
        if not sobj.data.materials and sub_mats:
            for sm in sub_mats:
                sobj.data.materials.append(sm)

        em_sock, bsdf_sub, base_em = None, None, 0.0
        if _should_highlight_part(sub_name) and sub_mats and len(sobj.material_slots) > 0:
            orig_m = sub_mats[0]
            ckey = f"{orig_m.name}|{'VU' if '_VU_' in sub_name else 'P'}"
            if ckey not in local_glow_cache:
                mc = orig_m.copy()
                mc.name = f"StPart_{sid:02d}_{len(local_glow_cache):02d}"
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
            "obj": sobj,
            "base_name": sub_name,
            "orig_loc": sub_loc.copy(),
            "orig_rot": sub_rot.copy(),
            "orig_scale": sub_scale.copy(),
            "bsdf": bsdf_sub,
            "em_sock": em_sock,
            "base_em": base_em,
        })

    base_pos = Vector(st["center"])
    anvil_obj.location = base_pos
    anvil_obj.rotation_mode = "QUATERNION"
    anvil_obj.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))

    # 3. Localized transient Point Light Spark beside station (replaces ugly flat ImpactRing!)
    hits = sorted(st["hits"], key=lambda h: int(h["frame_hit"]))
    if hits:
        ldata = bpy.data.lights.new(name=f"HitSparkData_{sid:02d}", type="POINT")
        ldata.color = glow_color[:3]
        ldata.energy = 0.0
        ldata.shadow_soft_size = 0.14
        if hasattr(ldata, "use_shadow"):
            ldata.use_shadow = False
        lobj = bpy.data.objects.new(f"HitSpark_{sid:02d}", ldata)
        col_fx.objects.link(lobj)
        c_vec = Vector(st["contact"])
        lobj.location = Vector((c_vec.x, c_vec.y - 0.18, c_vec.z + 0.18))
    else:
        ldata = None

    # 4. Keyframe subtle crisp mechanical feedback + tactile anvil cushion on each hit
    anvil_bsdf = next(n for n in anvil_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    cap_bsdf = next(n for n in cap_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")

    for k, h in enumerate(hits):
        f_hit = int(h["frame_hit"])
        vel = float(h.get("velocity", 110.0))
        vel_scale = 0.65 + 0.55 * max(0.0, min(1.0, vel / 127.0))
        midi_note = int(h.get("note", st["note"]))
        pitch_frac = max(0.0, min(1.0, (midi_note - 60) / 24.0))
        nxt_f = int(hits[k + 1]["frame_hit"]) if k + 1 < len(hits) else None
        prev_f = int(hits[k - 1]["frame_hit"]) if k > 0 else None
        max_decay = 9 if nxt_f is None else max(3, min(9, nxt_f - f_hit - 2))

        if prev_f is None or (f_hit - prev_f) > 2:
            anvil_obj.location = base_pos
            anvil_obj.keyframe_insert(data_path="location", frame=max(0, f_hit - 1))
            if ldata is not None:
                ldata.energy = 0.0
                ldata.keyframe_insert(data_path="energy", frame=max(0, f_hit - 1))

        # Subtle cushioned anvil compression (no wild giant slab rotation!)
        anvil_obj.location = base_pos - Vector((0.0, 0.0, 0.012 * vel_scale))
        anvil_obj.keyframe_insert(data_path="location", frame=f_hit + 1)
        anvil_obj.location = base_pos
        anvil_obj.keyframe_insert(data_path="location", frame=f_hit + min(6, max_decay))

        # Delicate brass cap glint (peak 0.65 — refined metallic sheen, zero washed-out bloom)
        _keyframe_emission(cap_bsdf.inputs.get("Emission Strength"), f_hit, 0.0, 0.72 * vel_scale, max_decay, cap_bsdf, glow_color)
        _keyframe_emission(anvil_bsdf.inputs.get("Emission Strength"), f_hit, 0.0, 0.14 * vel_scale, max_decay, anvil_bsdf, glow_color)

        # Warm transient localized Point Light spark lighting both marble and v3.0 instrument silhouette
        if ldata is not None:
            ldata.energy = 14.5 * vel_scale
            ldata.keyframe_insert(data_path="energy", frame=f_hit)
            ldata.energy = 5.0 * vel_scale
            ldata.keyframe_insert(data_path="energy", frame=f_hit + min(2, max_decay - 1))
            ldata.energy = 0.0
            ldata.keyframe_insert(data_path="energy", frame=f_hit + max_decay)

        # Bespoke instrument part animations (piano keys, castanet mallets/shells, bars)
        keys = _filter_parts(spawned_subs, "_WKey_", "_BKey_")
        if keys and anim_family in ("keyboard_hinge", "keyboard_pluck"):
            hk = _pick_pitch_part(keys, pitch_frac)
            if hk:
                piv = Vector((hk["orig_loc"].x, hk["orig_loc"].y + 0.04, hk["orig_loc"].z))
                _keyframe_part_transform(hk, [
                    (f_hit - 1, (0, 0, 0), (0, 0, 0), (1, 1, 1)),
                    (f_hit + 1, (-8.0 * vel_scale, 0, 0), (0, 0, -0.015 * vel_scale), (1, 1, 1)),
                    (f_hit + min(7, max_decay), (0, 0, 0), (0, 0, 0), (1, 1, 1)),
                ], pivot=piv)

        mallets = _filter_parts(spawned_subs, "_Mallet", "_MalletHead", "_MalletYarn", "_MalletBall", "_Clapper")
        if mallets:
            piv_m = Vector((0.0, 0.22, 0.14))
            for mp in mallets:
                _keyframe_part_transform(mp, [
                    (f_hit - 2, (-14.0, 0, 0), (0, 0, 0.018), (1, 1, 1)),
                    (f_hit, (12.0 * vel_scale, 0, 0), (0, -0.022, -0.015), (1, 1, 1)),
                    (f_hit + min(7, max_decay), (0, 0, 0), (0, 0, 0), (1, 1, 1)),
                ], pivot=piv_m)

        # Crisp localized sub-part glow accents (VU meters + highlighted keys/bars capped smoothly)
        for sp in spawned_subs:
            esock = sp.get("em_sock")
            if esock is None:
                continue
            b_em = sp.get("base_em", 0.0)
            p_em = 2.4 * vel_scale if "_VU_" in sp["base_name"] else min(1.35, max(0.4, 0.9 * vel_scale))
            _keyframe_emission(esock, f_hit, b_em, p_em, max_decay, sp.get("bsdf"), glow_color if b_em < 0.5 else None)


# ---------------------------------------------------------------------------
# Marble, Cinema Panorama Camera & Lighting Setup
# ---------------------------------------------------------------------------

def create_aurora_marble(col_marble: bpy.types.Collection, frame_samples: list[dict], bounces: list[dict]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=44, ring_count=26, radius=MARBLE_RADIUS)
    marble = bpy.context.active_object
    marble.name = "HeroMarble_AuroraEmerald"
    for c in list(marble.users_collection):
        c.objects.unlink(marble)
    col_marble.objects.link(marble)
    for p in marble.data.polygons:
        p.use_smooth = True

    mat = make_pbr_material(
        "MarbleAuroraMat",
        base_color=(0.06, 0.88, 0.56),
        metallic=0.90,
        roughness=0.07,
        coat=1.0,
        emit_color=(0.12, 0.95, 0.62),
        emit_strength=0.18,
    )
    marble.data.materials.append(mat)
    marble.rotation_mode = "XYZ"

    # Keyframe every frame + natural squash/stretch impact bounce pulse
    hit_frames = {int(b["frame"]) for b in bounces}
    angle_acc_y = 0.0
    prev_pos = Vector(frame_samples[0]["position"])
    for smp in frame_samples:
        f = int(round(smp["frame_f"]))
        pos = Vector(smp["position"])
        dx = pos.x - prev_pos.x
        angle_acc_y += dx / max(0.05, MARBLE_RADIUS)
        prev_pos = pos

        marble.location = pos
        marble.rotation_euler = (0.0, angle_acc_y, 0.0)
        marble.keyframe_insert(data_path="location", frame=f)
        marble.keyframe_insert(data_path="rotation_euler", frame=f)

        # Subtle kinetic squash right on hit frame (pure volume-preserving squash-and-stretch)
        if f in hit_frames:
            marble.scale = (1.10, 1.10, 0.82)
            marble.keyframe_insert(data_path="scale", frame=f)
            marble.scale = (0.94, 0.94, 1.12)
            marble.keyframe_insert(data_path="scale", frame=f + 2)
            marble.scale = (1.0, 1.0, 1.0)
            marble.keyframe_insert(data_path="scale", frame=f + 5)

    # Subtle trailing warm emerald glow lamp moving right with marble
    m_light = bpy.data.lights.new("MarbleGlowLightData", type="POINT")
    m_light.color = (0.28, 0.98, 0.66)
    m_light.energy = 3.2
    m_light.shadow_soft_size = 0.14
    if hasattr(m_light, "use_shadow"):
        m_light.use_shadow = False
    ml_obj = bpy.data.objects.new("MarbleGlowLight", m_light)
    col_marble.objects.link(ml_obj)
    ml_obj.parent = marble
    ml_obj.location = (0.0, -0.06, 0.0)
    return marble


def create_concert_lighting(col_stage: bpy.types.Collection) -> None:
    def add_area(name, loc, rot_deg, energy, size, size_y, color):
        ld = bpy.data.lights.new(name + "Data", type="AREA")
        ld.energy = energy
        ld.shape = "RECTANGLE"
        ld.size = size
        ld.size_y = size_y
        ld.color = color
        lo = bpy.data.objects.new(name, ld)
        col_stage.objects.link(lo)
        lo.location = loc
        lo.rotation_euler = tuple(math.radians(a) for a in rot_deg)
        return lo

    # Key softbox overhead-front illuminating piano keys, castanets, and brass rails cleanly
    add_area("KeySoftboxWarm", (0.0, -4.6, 3.6), (48.0, 0.0, 0.0), 340.0, 6.2, 2.2, (1.0, 0.93, 0.84))
    # Cool steel-blue rim fill from right-upper side
    add_area("CoolRimRight", (4.8, -2.8, 1.8), (68.0, 0.0, 48.0), 150.0, 3.2, 3.2, (0.70, 0.86, 1.0))
    # Warm gold accent fill from left-lower side catching brass loops
    add_area("WarmFillLeft", (-4.8, -3.0, -0.4), (96.0, 0.0, -52.0), 130.0, 3.0, 2.4, (1.0, 0.78, 0.56))


def setup_panoramic_camera(scene: bpy.types.Scene, col_cam: bpy.types.Collection, frame_samples: list[dict], total_frames: int) -> None:
    """Gentle widescreen 16:9 panoramic cinema camera keeping full closed-loop stage framed cleanly."""
    cam_data = bpy.data.cameras.new("ConcertPanCamData")
    cam_data.lens = 32.0
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.05
    cam_data.clip_end = 100.0
    cam_obj = bpy.data.objects.new("ConcertPanCam", cam_data)
    col_cam.objects.link(cam_obj)
    scene.camera = cam_obj

    tgt = bpy.data.objects.new("CamTarget", None)
    col_cam.objects.link(tgt)
    cns = cam_obj.constraints.new(type="TRACK_TO")
    cns.target = tgt
    cns.track_axis = "TRACK_NEGATIVE_Z"
    cns.up_axis = "UP_Y"

    # Subtle gentle pan tracking active marble without losing the panoramic closed-loop machine
    step = 6
    for idx in range(0, len(frame_samples), step):
        smp = frame_samples[idx]
        f = int(round(smp["frame_f"]))
        mp = Vector(smp["position"])
        prog = f / max(1, total_frames)
        # Keep camera smoothly sweeping slightly with the marble while framing the entire 7.4m machine
        cam_x = 0.22 * mp.x
        cam_z = 0.22 + 0.14 * mp.z + 0.05 * math.sin(prog * 2.0 * math.pi)
        cam_y = -7.45 + 0.16 * math.sin(prog * math.pi)
        cam_obj.location = (cam_x, cam_y, cam_z)
        tgt.location = (0.12 * mp.x, 0.08, 0.22 + 0.08 * mp.z)
        cam_obj.keyframe_insert(data_path="location", frame=f)
        tgt.keyframe_insert(data_path="location", frame=f)

    # Ensure last frame keyframed cleanly
    last_mp = Vector(frame_samples[-1]["position"])
    cam_obj.location = (0.22 * last_mp.x, -7.45, 0.22 + 0.14 * last_mp.z)
    tgt.location = (0.12 * last_mp.x, 0.08, 0.22 + 0.08 * last_mp.z)
    cam_obj.keyframe_insert(data_path="location", frame=total_frames)
    tgt.keyframe_insert(data_path="location", frame=total_frames)


def build_and_render_16x9(
    notes_json: str,
    physics_out: str,
    blend_out: str,
    render_frames_dir: str | None = None,
    still_frames: list[int] | None = None,
    render_anim: bool = False,
    max_note_t: float = 10.0,
) -> dict:
    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = get_eevee_engine()
    if hasattr(scene.render, "use_sequencer"):
        scene.render.use_sequencer = False
    scene.render.fps = FPS
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    if scene.render.engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") and hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 32
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = True

    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.10

    templates, mats, manifest = load_static_instrument_library()
    plan = plan_16x9_closed_loop_physics(notes_json, max_note_t=max_note_t)

    # Compute real v3.0 world-space AABBs and build geometry
    col_stage = get_or_create_collection("Stage_16x9")
    col_inst = get_or_create_collection("Instruments_16x9")
    col_rails = get_or_create_collection("ReturnLoops_16x9")
    col_fx = get_or_create_collection("ImpactSparks_16x9")
    col_marble = get_or_create_collection("Marble_16x9")
    col_cam = get_or_create_collection("Camera_16x9")

    build_concert_world(scene)
    setup_bloom_compositor(scene)
    create_concert_lighting(col_stage)
    create_panoramic_concert_hall(col_stage, mats)

    rail_mat = make_pbr_material("BrassCoasterRail", (0.72, 0.52, 0.20), metallic=0.95, roughness=0.22, coat=0.3)
    tie_dark = make_pbr_material("RailTieDarkBronze", (0.14, 0.10, 0.08), metallic=0.75, roughness=0.32)
    ped_mat = make_pbr_material("StationPedestalBronze", (0.12, 0.09, 0.07), metallic=0.60, roughness=0.35)
    brass_mat = mats.get("LibMat_Brass") or rail_mat
    walnut_mat = mats.get("LibMat_Walnut") or ped_mat

    for st in plan["stations"]:
        create_station_pedestal_and_pitch_tag(st, col_stage, brass_mat, walnut_mat, ped_mat)
        spawn_concert_station(st, templates, mats, manifest, col_inst, col_fx)

    rails_geom = []
    for rp in plan["rails_plan"]:
        pts_v = [Vector(p) for p in rp["points"]]
        rails_geom.append(
            build_parallel_rail_geometry(rp["name"], pts_v, rail_mat, brass_mat, tie_dark, col_rails)
        )

    # Run mathematical zero-intersection verification BEFORE rendering a single pixel!
    aabbs = compute_station_world_aabbs(plan["stations"], templates, manifest)
    verify_zero_intersections(aabbs, rails_geom, plan["dense_samples"], plan["bounces"])

    create_aurora_marble(col_marble, plan["frame_samples"], plan["bounces"])
    setup_panoramic_camera(scene, col_cam, plan["frame_samples"], plan["total_frames"])

    scene.frame_start = 0
    scene.frame_end = plan["total_frames"]

    os.makedirs(os.path.dirname(physics_out), exist_ok=True)
    with open(physics_out, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    os.makedirs(os.path.dirname(blend_out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend_out)
    print(f"[Saved 16:9 Scene] {blend_out} ({plan['total_frames']} frames @ {FPS}fps = {plan['duration_s']}s)")

    if render_frames_dir and still_frames:
        os.makedirs(render_frames_dir, exist_ok=True)
        for sf in still_frames:
            f_clamped = max(0, min(plan["total_frames"], int(sf)))
            scene.frame_set(f_clamped)
            scene.render.filepath = os.path.join(render_frames_dir, f"check_{f_clamped:04d}.png")
            bpy.ops.render.render(write_still=True)
            print(f"[Rendered Check Frame] {scene.render.filepath}")

    if render_frames_dir and render_anim:
        os.makedirs(render_frames_dir, exist_ok=True)
        scene.render.filepath = os.path.join(render_frames_dir, "frame_")
        bpy.ops.render.render(animation=True)

    return plan


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="Standalone 16:9 Closed-Loop Marble Concert Pipeline")
    ap.add_argument("--notes", default=os.path.join(ARTIFACTS_DIR, "sample_14s_notes.json"))
    ap.add_argument("--physics-out", default=os.path.join(ARTIFACTS_DIR, "concert_16x9_loop_10s_physics.json"))
    ap.add_argument("--blend-out", default=os.path.join(RENDERS_DIR, "concert_16x9_loop_10s.blend"))
    ap.add_argument("--frames-dir", default=os.path.join(RENDERS_DIR, "frames_16x9_loop_10s"))
    ap.add_argument("--max-note-t", type=float, default=10.0)
    ap.add_argument("--stills", default="", help="Comma-separated frame numbers to render for visual check")
    ap.add_argument("--render-anim", action="store_true", help="Render full 60fps PNG sequence")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    still_list = [int(x.strip()) for x in args.stills.split(",") if x.strip()] if args.stills else None
    build_and_render_16x9(
        notes_json=args.notes,
        physics_out=args.physics_out,
        blend_out=args.blend_out,
        render_frames_dir=args.frames_dir if (still_list or args.render_anim) else None,
        still_frames=still_list,
        render_anim=args.render_anim,
        max_note_t=args.max_note_t,
    )
