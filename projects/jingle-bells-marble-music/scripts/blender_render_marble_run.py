#!/usr/bin/env python3
"""Blender 5.3 scene builder & renderer for trajectory-first Marble Music (v5 — 64-Instrument Library).

Features:
- Pulls all 64 sculpted GM instrument assemblies & procedural PBR shaders from
  the reusable Asset Library `assets/instrument_library_v2.blend` + `instrument_manifest_v2.json`.
- Preserves full child object transforms (location/rotation/scale) & multi-material slots.
- Supports all 64 GM instruments + full backward-compatibility mapping for legacy 12-track plans.
- Per-part mechanical & expressive hit animations (keys, mallets, bars, bellows, slides, bows,
  valves, woodwind keys, plucked strings, gong/chime pendulums, synth VU/LED waveforms).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import bpy
import bmesh
from mathutils import Euler, Matrix, Quaternion, Vector


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "assets"))
DEFAULT_LIB_BLEND = os.path.join(ASSETS_DIR, "instrument_library_v2.blend")
DEFAULT_MANIFEST_JSON = os.path.join(ASSETS_DIR, "instrument_manifest_v2.json")

# Backward-compatibility map: legacy 1..12 track indices -> General MIDI program
LEGACY_TRACK_TO_GM: dict[int, int] = {
    1: 115,  # Castanets -> Woodblock
    2: 0,    # Piano -> Acoustic Grand Piano
    3: 11,   # Vibraphone -> Vibraphone
    4: 14,   # TubularBells -> Tubular Bells
    5: 12,   # Marimba -> Marimba
    6: 108,  # Kalimba -> Kalimba
    7: 73,   # Flute -> Flute
    8: 13,   # Xylophone -> Xylophone
    9: 79,   # Ocarina -> Ocarina
    10: 9,   # Glockenspiel -> Glockenspiel
    11: 114, # SteelDrums -> Steel Drums
    12: 24,  # Guitar -> Nylon Guitar
}

LEGACY_NAME_TO_GM: dict[str, int] = {
    "castanets": 115,
    "castanet": 115,
    "piano": 0,
    "grand_piano": 0,
    "vibraphone": 11,
    "tubular_bells": 14,
    "tubularbells": 14,
    "marimba": 12,
    "kalimba": 108,
    "flute": 73,
    "xylophone": 13,
    "ocarina": 79,
    "glockenspiel": 9,
    "steel_drums": 114,
    "steeldrums": 114,
    "steel_drum": 114,
    "guitar": 24,
}


# -------------------- Helper Utilities --------------------

def clear_scene():
    _BOX_MESH_CACHE.clear()
    _CYL_MESH_CACHE.clear()
    bpy.ops.wm.read_factory_settings(use_empty=True)


def get_or_create_collection(name: str, parent=None):
    if name in bpy.data.collections:
        col = bpy.data.collections[name]
    else:
        col = bpy.data.collections.new(name)
        target = parent if parent is not None else bpy.context.scene.collection
        target.children.link(col)
    return col


def set_socket(node, name: str, value):
    if name in node.inputs:
        node.inputs[name].default_value = value


def get_eevee_engine() -> str:
    enum_items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    if "BLENDER_EEVEE_NEXT" in enum_items:
        return "BLENDER_EEVEE_NEXT"
    return "BLENDER_EEVEE"


def ensure_instrument_library(lib_path: str):
    if os.path.exists(lib_path):
        return
    if os.path.basename(lib_path) == "instrument_library_v2.blend":
        builder = os.path.join(SCRIPT_DIR, "build_instrument_library_v2.py")
    else:
        builder = os.path.join(SCRIPT_DIR, "build_instrument_library.py")
    if not os.path.exists(builder):
        raise FileNotFoundError(f"Instrument library missing and builder not found: {lib_path}")
    subprocess.run([bpy.app.binary_path, "-b", "-P", builder], check=True)


def load_instrument_manifest(manifest_path: str | None = None) -> dict:
    mpath = manifest_path or DEFAULT_MANIFEST_JSON
    if not os.path.exists(mpath):
        return {}
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_instrument_library(
    lib_path: str = DEFAULT_LIB_BLEND,
    manifest_path: str | None = None,
) -> tuple[dict[int, dict], dict[str, bpy.types.Material], dict]:
    """Append all 64 (or 12 legacy) instrument assemblies & LibMat_* PBR shaders."""
    ensure_instrument_library(lib_path)
    manifest = load_instrument_manifest(manifest_path)
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
        # Support both `Lib_{idx:02d}_GM{gm:03d}_{Name}` (v2/v3) and `Lib_{idx:02d}_{Name}` (legacy)
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
        tpl_entry = {
            "index": t_idx,
            "gm_program": gm_prog,
            "name": m_info.get("name", col.name),
            "anim_family": m_info.get("anim_family", "generic_strike"),
            "glow_color": tuple(m_info.get("glow_color", (1.0, 0.78, 0.30))),
            "core_mesh": core_obj.data if core_obj else None,
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
        templates[gm_prog] = tpl_entry
        # Also allow lookup by catalog index or negative legacy key via secondary mapping
        templates.setdefault(-1000 - t_idx, tpl_entry)

    # Remove imported template staging objects/collections while keeping fake-user datablocks intact
    for col in temp_cols:
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)

    return templates, mats, manifest


def resolve_board_gm_program(board: dict, manifest: dict | None = None) -> int:
    """Resolve General MIDI program (0..115) from explicit gm_program, instrument name, or legacy track_idx."""
    if "gm_program" in board and board["gm_program"] is not None:
        return int(board["gm_program"])
    inst_name = str(board.get("instrument", "")).strip().lower().replace(" ", "_")
    inst_type = str(board.get("instrument_type", "")).strip().lower().replace(" ", "_")
    if manifest:
        for gm_str, meta in manifest.get("by_program", {}).items():
            mname = str(meta.get("name", "")).strip().lower().replace(" ", "_")
            if inst_name == mname or inst_type == mname or inst_type == meta.get("variant"):
                return int(gm_str)
    for key in (inst_type, inst_name):
        if key in LEGACY_NAME_TO_GM:
            return LEGACY_NAME_TO_GM[key]
    t_idx = int(board.get("track_idx", 1))
    return LEGACY_TRACK_TO_GM.get(t_idx, 0)


def get_board_material_and_glow(
    board: dict,
    mats: dict[str, bpy.types.Material],
    templates: dict[int, dict] | None = None,
    manifest: dict | None = None,
) -> tuple[bpy.types.Material, tuple[float, float, float], float]:
    """Select pitch-accurate material from Asset Library + calibrated hit glow color."""
    gm = resolve_board_gm_program(board, manifest)
    midi_note = int(board.get("note", 60))
    pc = midi_note % 12
    tpl = (templates or {}).get(gm)

    # Piano / Harpsichord / Electric Piano sharp vs natural key contrast
    if gm in (0, 2, 4, 6):
        if pc in {1, 3, 6, 8, 10} and "LibMat_Ebony" in mats:
            return mats["LibMat_Ebony"].copy(), (0.35, 0.85, 1.0), 4.2
        if "LibMat_Ivory" in mats:
            return mats["LibMat_Ivory"].copy(), (0.75, 0.94, 1.0), 2.8

    # Xylophone jewel-tone pitch palette using rich gloss copies
    if gm == 13 and "LibMat_XyloGold" in mats:
        xylo_cols = [
            (0.88, 0.14, 0.14),
            (0.94, 0.44, 0.08),
            (0.94, 0.76, 0.10),
            (0.12, 0.72, 0.34),
            (0.12, 0.56, 0.90),
            (0.50, 0.22, 0.88),
        ]
        jc = xylo_cols[midi_note % len(xylo_cols)]
        xm = mats["LibMat_XyloGold"].copy()
        bsdf = next(n for n in xm.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        set_socket(bsdf, "Base Color", (*jc, 1.0))
        return xm, jc, 3.8

    core_mat = tpl["core_mat"] if (tpl and tpl.get("core_mat")) else mats.get("LibMat_Walnut") or next(iter(mats.values()))
    glow_c = tpl["glow_color"] if tpl else (1.0, 0.78, 0.30)
    return core_mat.copy(), glow_c, 3.6


_BOX_MESH_CACHE: dict[tuple[float, float, float], bpy.types.Mesh] = {}
_CYL_MESH_CACHE: dict[tuple[float, float, int, str], bpy.types.Mesh] = {}


def fix_mesh_normals(
    obj: bpy.types.Object,
    angle_deg: float = 35.0,
    weighted: bool = True,
    remove_doubles: bool = False,
):
    """Apply Blender's native Recalculate Outside, Smooth by Angle, and Weighted Normal."""
    scene_col = bpy.context.scene.collection
    temp_linked = False
    if obj.name not in bpy.context.view_layer.objects:
        scene_col.objects.link(obj)
        temp_linked = True

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    if remove_doubles:
        bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.customdata_custom_splitnormals_clear()
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle_deg))

    if weighted:
        wn = obj.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
        wn.mode = "FACE_AREA"
        wn.weight = 100
        wn.keep_sharp = True
        bpy.ops.object.modifier_apply(modifier=wn.name)

    if temp_linked:
        scene_col.objects.unlink(obj)


def create_cylinder_between(name: str, p_a, p_b, radius: float, mat, collection, vertices: int = 18):
    a = Vector(p_a)
    b = Vector(p_b)
    diff = b - a
    L = diff.length
    if L < 1e-6:
        return None
    mat_key = mat.name if mat is not None else "None"
    cache_key = (round(float(L), 4), round(float(radius), 4), int(vertices), mat_key)
    mesh = _CYL_MESH_CACHE.get(cache_key)
    if mesh is None:
        mesh = bpy.data.meshes.new(f"CylProto_{cache_key[0]}_{cache_key[1]}_{vertices}")
        bm = bmesh.new()
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            cap_tris=False,
            segments=vertices,
            radius1=radius,
            radius2=radius,
            depth=L,
        )
        bm.to_mesh(mesh)
        bm.free()
        tmp_obj = bpy.data.objects.new("__TmpCyl", mesh)
        fix_mesh_normals(tmp_obj, angle_deg=35.0, weighted=True)
        bpy.data.objects.remove(tmp_obj, do_unlink=True)
        mesh.materials.append(mat)
        _CYL_MESH_CACHE[cache_key] = mesh

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = (a + b) * 0.5
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(diff.normalized())
    return obj


def create_hard_beveled_box(name: str, length: float, width_y: float, thickness: float) -> bpy.types.Mesh:
    """Build 3-segment beveled box with Blender native Smooth by Angle + Weighted Normal."""
    cache_key = (round(float(length), 4), round(float(width_y), 4), round(float(thickness), 4))
    proto = _BOX_MESH_CACHE.get(cache_key)
    if proto is None:
        proto = bpy.data.meshes.new(f"BoxProto_{cache_key[0]}_{cache_key[1]}_{cache_key[2]}")
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(length, width_y, thickness), verts=bm.verts)
        bmesh.ops.bevel(
            bm,
            geom=list(bm.edges),
            offset=0.0055,
            segments=5,
            profile=0.5,
            affect="EDGES",
            clamp_overlap=True,
            harden_normals=False,
        )
        bm.to_mesh(proto)
        bm.free()
        tmp_obj = bpy.data.objects.new("__TmpBox", proto)
        fix_mesh_normals(tmp_obj, angle_deg=35.0, weighted=True)
        bpy.data.objects.remove(tmp_obj, do_unlink=True)
        _BOX_MESH_CACHE[cache_key] = proto

    mesh = proto.copy()
    mesh.name = name
    return mesh


def create_pipe_polyline(name: str, pts: list[list[float]], radius: float, mat, collection):
    curve = bpy.data.curves.new(name + "_curve", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 8
    curve.bevel_depth = float(radius)
    curve.bevel_resolution = 12
    curve.use_fill_caps = True

    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(pts) - 1)
    for k, pt in enumerate(pts):
        bp = spline.bezier_points[k]
        bp.co = (pt[0], pt[1], pt[2])
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    spline.use_smooth = True

    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    obj.data.materials.append(mat)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target="MESH")
    fix_mesh_normals(obj, angle_deg=60.0, weighted=True, remove_doubles=True)
    return obj


def create_wall_flange(name: str, pos, radius: float, mat, collection):
    return create_cylinder_between(
        name,
        (pos[0], pos[1] - 0.014, pos[2]),
        (pos[0], pos[1] + 0.002, pos[2]),
        radius,
        mat,
        collection,
        vertices=20,
    )


def build_studio_reflection_world(scene: bpy.types.Scene):
    """Dark studio diffuse surround + 3D Normal-mapped studio softbox panels for crisp metal reflections."""
    world = bpy.data.worlds.new("DarkAcousticStudioWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    lp = nt.nodes.new("ShaderNodeLightPath")

    bg_diff = nt.nodes.new("ShaderNodeBackground")
    bg_diff.inputs["Color"].default_value = (0.016, 0.020, 0.028, 1.0)
    bg_diff.inputs["Strength"].default_value = 0.45

    # Procedural Studio Softbox Rig sampled ONLY by Glossy rays in 3D Normal space
    bg_gloss = nt.nodes.new("ShaderNodeBackground")
    bg_gloss.inputs["Strength"].default_value = 1.0

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Normal"], sep.inputs["Vector"])

    # Overhead key softbox: Z > 0.38 -> crisp bright warm-white panel
    r_top = nt.nodes.new("ShaderNodeValToRGB")
    r_top.color_ramp.elements[0].position = 0.34
    r_top.color_ramp.elements[0].color = (0.004, 0.005, 0.008, 1.0)
    r_top.color_ramp.elements[1].position = 0.48
    r_top.color_ramp.elements[1].color = (4.2, 4.0, 3.7, 1.0)
    nt.links.new(sep.outputs["Z"], r_top.inputs["Fac"])

    # Side cool softbox rim strip: X > 0.62 -> cool cyan-white edge panel
    r_side = nt.nodes.new("ShaderNodeValToRGB")
    r_side.color_ramp.elements[0].position = 0.58
    r_side.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    r_side.color_ramp.elements[1].position = 0.72
    r_side.color_ramp.elements[1].color = (2.2, 2.8, 3.8, 1.0)
    nt.links.new(sep.outputs["X"], r_side.inputs["Fac"])

    # Opposite warm accent strip: X < -0.66
    inv_x = nt.nodes.new("ShaderNodeMath")
    inv_x.operation = "MULTIPLY"
    inv_x.inputs[1].default_value = -1.0
    nt.links.new(sep.outputs["X"], inv_x.inputs[0])

    r_left = nt.nodes.new("ShaderNodeValToRGB")
    r_left.color_ramp.elements[0].position = 0.64
    r_left.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    r_left.color_ramp.elements[1].position = 0.78
    r_left.color_ramp.elements[1].color = (3.4, 2.7, 1.9, 1.0)
    nt.links.new(inv_x.outputs["Value"], r_left.inputs["Fac"])

    add1 = nt.nodes.new("ShaderNodeMix")
    add1.data_type = "RGBA"
    add1.blend_type = "ADD"
    add1.inputs["Factor"].default_value = 1.0
    nt.links.new(r_top.outputs["Color"], add1.inputs["A"])
    nt.links.new(r_side.outputs["Color"], add1.inputs["B"])

    add2 = nt.nodes.new("ShaderNodeMix")
    add2.data_type = "RGBA"
    add2.blend_type = "ADD"
    add2.inputs["Factor"].default_value = 1.0
    nt.links.new(add1.outputs["Result"], add2.inputs["A"])
    nt.links.new(r_left.outputs["Color"], add2.inputs["B"])

    nt.links.new(add2.outputs["Result"], bg_gloss.inputs["Color"])
    nt.links.new(lp.outputs["Is Glossy Ray"], mix_sh.inputs["Fac"])
    nt.links.new(bg_diff.outputs["Background"], mix_sh.inputs[1])
    nt.links.new(bg_gloss.outputs["Background"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])


def setup_bloom_compositor(scene: bpy.types.Scene):
    """Configure Blender 5.3 Compositor with subtle optical Bloom glow on hit flashes."""
    ng = bpy.data.node_groups.new("StudioBloomCompositor", "CompositorNodeTree")
    scene.compositing_node_group = ng
    scene.render.use_compositing = True

    if not any(s.name == "Image" and s.in_out == "OUTPUT" for s in ng.interface.items_tree):
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    rl = ng.nodes.new("CompositorNodeRLayers")
    glare = ng.nodes.new("CompositorNodeGlare")
    out = ng.nodes.new("NodeGroupOutput")

    set_socket(glare, "Type", "Bloom")
    set_socket(glare, "Quality", "High")
    set_socket(glare, "Threshold", 1.15)
    set_socket(glare, "Smoothness", 0.35)
    set_socket(glare, "Strength", 0.55)
    set_socket(glare, "Size", 0.65)

    ng.links.new(rl.outputs["Image"], glare.inputs["Image"])
    ng.links.new(glare.outputs["Image"], out.inputs["Image"])


def create_dark_acoustic_stage_wall(
    z_min: float,
    z_max: float,
    wall_y: float,
    felt_mat,
    slat_mat,
    trim_mat,
    collection,
    phrase_breaks_z: list[float] | None = None,
):
    """Low-contrast matte charcoal backplate + outer-edge walnut slats + brass phrase dividers."""
    span_z = max(24.0, (z_max - z_min) + 18.0)
    center_z = 0.5 * (z_max + z_min)

    # 1. Matte charcoal backplate spanning full background (recedes quietly behind instruments)
    fmesh = bpy.data.meshes.new("AcousticWallFelt")
    fbm = bmesh.new()
    bmesh.ops.create_grid(fbm, x_segments=1, y_segments=1, size=1.0)
    bmesh.ops.transform(fbm, matrix=Matrix.Rotation(math.radians(90.0), 4, "X"), verts=fbm.verts)
    bmesh.ops.scale(fbm, vec=(14.0, 1.0, span_z * 0.5), verts=fbm.verts)
    bmesh.ops.translate(fbm, vec=(0.0, wall_y + 0.032, center_z), verts=fbm.verts)
    fbm.to_mesh(fmesh)
    fbm.free()
    fobj = bpy.data.objects.new("AcousticWallFelt", fmesh)
    collection.objects.link(fobj)
    fobj.data.materials.append(felt_mat)

    # 2. Subdued smoked-walnut slats ONLY on outer left/right flanks (|x| >= 1.92), keeping center clean
    smesh = bpy.data.meshes.new("AcousticWoodSlats")
    sbm = bmesh.new()
    slat_w = 0.14
    slat_d = 0.020
    pitch_x = 0.24
    edge_indices = list(range(-14, -7)) + list(range(8, 15))
    for kx in edge_indices:
        xc = kx * pitch_x
        res = bmesh.ops.create_cube(sbm, size=1.0)
        verts = res["verts"]
        bmesh.ops.scale(sbm, vec=(slat_w, slat_d, span_z), verts=verts)
        bmesh.ops.translate(sbm, vec=(xc, wall_y + 0.020, center_z), verts=verts)
    bmesh.ops.bevel(
        sbm,
        geom=list(sbm.edges),
        offset=0.003,
        segments=1,
        profile=0.5,
        affect="EDGES",
        clamp_overlap=True,
    )
    sbm.to_mesh(smesh)
    sbm.free()
    sobj = bpy.data.objects.new("AcousticWoodSlats", smesh)
    collection.objects.link(sobj)
    sobj.data.materials.append(slat_mat)

    # 3. Subtle brass-reveal phrase divider bars at instrument transitions
    tmesh = bpy.data.meshes.new("WallBrassTrims")
    tbm = bmesh.new()
    div_zs = phrase_breaks_z if phrase_breaks_z else []
    for zc in div_zs:
        res = bmesh.ops.create_cube(tbm, size=1.0)
        verts = res["verts"]
        bmesh.ops.scale(tbm, vec=(6.4, 0.012, 0.014), verts=verts)
        bmesh.ops.translate(tbm, vec=(0.0, wall_y + 0.014, zc), verts=verts)
    tbm.to_mesh(tmesh)
    tbm.free()
    tobj = bpy.data.objects.new("WallBrassTrims", tmesh)
    collection.objects.link(tobj)
    tobj.data.materials.append(trim_mat)

    return fobj, sobj, tobj


def build_phrase_clusters(boards: list[dict], max_cluster: int = 6) -> list[list[dict]]:
    """Group consecutive boards by instrument phrase for shared modular mounting spines."""
    clusters: list[list[dict]] = []
    cur: list[dict] = []
    for b in boards:
        if (
            not cur
            or (b["track_idx"] == cur[-1]["track_idx"] and len(cur) < max_cluster)
        ):
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    if cur:
        clusters.append(cur)
    return clusters


def create_modular_phrase_racks(
    clusters: list[list[dict]],
    wall_y: float,
    spine_y: float,
    walnut_mat,
    brass_mat,
    bronze_mat,
    collection,
) -> dict[int, float]:
    """Create shared walnut & brass spine beams at midground Y=0.34 for each phrase cluster.
    Returns mapping board_index -> spine_x so board rear pins connect cleanly to the rack."""
    board_to_spine_x: dict[int, float] = {}
    for c_idx, grp in enumerate(clusters):
        zs = [b["center"][2] for b in grp]
        xs = [b["center"][0] for b in grp]
        z_top = max(zs) + 0.16
        z_bot = min(zs) - 0.16
        spine_h = max(0.32, z_top - z_bot)
        z_mid = 0.5 * (z_top + z_bot)
        spine_x = max(-0.75, min(0.75, sum(xs) / len(xs)))

        for b in grp:
            board_to_spine_x[int(b["index"])] = spine_x

        # Walnut spine beam (midground mechanical music-box backbone)
        spine_mesh = create_hard_beveled_box(f"SpineBeamMesh_{c_idx:03d}", 0.072, 0.056, spine_h)
        spine_obj = bpy.data.objects.new(f"PhraseSpine_{c_idx:03d}", spine_mesh)
        collection.objects.link(spine_obj)
        spine_obj.data.materials.append(walnut_mat)
        spine_obj.location = (spine_x, spine_y, z_mid)

        # Warm brass accent collars at top and bottom of each module spine
        for tag, zc in (("Top", z_top - 0.04), ("Bot", z_bot + 0.04)):
            cap_mesh = create_hard_beveled_box(f"SpineCapMesh_{c_idx:03d}_{tag}", 0.092, 0.070, 0.032)
            cap_obj = bpy.data.objects.new(f"PhraseCap_{c_idx:03d}_{tag}", cap_mesh)
            collection.objects.link(cap_obj)
            cap_obj.data.materials.append(brass_mat)
            cap_obj.location = (spine_x, spine_y, zc)

            # Heavy bronze wall standoff anchoring spine module to matte backplate
            create_cylinder_between(
                f"SpineStandoff_{c_idx:03d}_{tag}",
                (spine_x, spine_y + 0.025, zc),
                (spine_x, wall_y + 0.02, zc),
                0.020,
                bronze_mat,
                collection,
                vertices=16,
            )
            create_wall_flange(
                f"SpineFlange_{c_idx:03d}_{tag}",
                (spine_x, wall_y, zc),
                0.044,
                brass_mat,
                collection,
            )

    return board_to_spine_x


def create_shockwave_ring_mesh(name: str = "ShockwaveRingMesh") -> bpy.types.Mesh:
    """Thin beveled circular halo ring lying in XY plane with normal +Z."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    segs = 64
    r_in = 0.088
    r_out = 0.114
    in_verts = []
    out_verts = []
    for k in range(segs):
        ang = 2.0 * math.pi * k / segs
        ca, sa = math.cos(ang), math.sin(ang)
        in_verts.append(bm.verts.new((r_in * ca, r_in * sa, 0.0)))
        out_verts.append(bm.verts.new((r_out * ca, r_out * sa, 0.0)))
    for k in range(segs):
        kn = (k + 1) % segs
        bm.faces.new((in_verts[k], out_verts[k], out_verts[kn], in_verts[kn]))
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    return mesh


def make_shockwave_material(name: str, glow_color) -> tuple[bpy.types.Material, bpy.types.Node]:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*glow_color[:3], 1.0)
    em.inputs["Strength"].default_value = 0.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat, em


def load_note_metadata(physics_json: str) -> list[dict]:
    """Load optional MIDI velocity & duration from sibling *_notes.json if present."""
    candidates = []
    if physics_json.endswith("_physics.json"):
        candidates.append(physics_json.replace("_physics.json", "_notes.json"))
    for cand in candidates:
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("notes", [])
    return []


def plan_lookahead_camera(
    samples: list[dict],
    boards: list[dict],
    fps: int,
    half_window: int = 24,
) -> tuple[list[float], list[float], list[float]]:
    """Compute smoothed look-ahead target (X, Z) and dynamic camera Y distance.

    - Standard bounce phrases: Y ~ -3.45 @ 48mm -> marble ~8.1% of frame width.
    - Wide U-rail transitions (gap > 0.55s): smoothly pulls back to Y ~ -4.10.
    - Target biases 38% toward the next upcoming hit contact within 0.85s so viewers
      can anticipate the next 1-2 impacts before they happen.
    """
    n = len(samples)
    raw_tx: list[float] = []
    raw_ty_cam: list[float] = []
    raw_tz: list[float] = []

    b_idx = 0
    nb = len(boards)
    for s in samples:
        f = int(s["frame"])
        mx, _, mz = s["position"]
        while b_idx < nb and int(boards[b_idx]["frame_hit"]) < f:
            b_idx += 1

        if b_idx < nb:
            nxt = boards[b_idx]
            dt_next = (int(nxt["frame_hit"]) - f) / max(1, fps)
            prev_f = int(boards[b_idx - 1]["frame_hit"]) if b_idx > 0 else 0
            gap_dur = (int(nxt["frame_hit"]) - prev_f) / max(1, fps)

            # Look-ahead bias toward next landing point
            la_w = max(0.15, min(0.42, 0.42 - 0.25 * max(0.0, dt_next)))
            nx, _, nz = nxt["contact"]
            tx = (1.0 - la_w) * mx + la_w * nx
            tz = (1.0 - la_w) * mz + la_w * nz - 0.14

            # Pull back smoothly during wide U-rail glide loops
            y_cam = -4.10 if gap_dur > 0.55 else -3.45
        else:
            tx, tz, y_cam = mx, mz - 0.14, -3.45

        raw_tx.append(tx)
        raw_ty_cam.append(y_cam)
        raw_tz.append(tz)

    out_x: list[float] = []
    out_y: list[float] = []
    out_z: list[float] = []
    weights = [math.exp(-0.5 * (k / (half_window * 0.45)) ** 2) for k in range(-half_window, half_window + 1)]
    for i in range(n):
        ax = ay = az = wsum = 0.0
        for idx_w, off in enumerate(range(-half_window, half_window + 1)):
            j = max(0, min(n - 1, i + off))
            w = weights[idx_w]
            ax += raw_tx[j] * w
            ay += raw_ty_cam[j] * w
            az += raw_tz[j] * w
            wsum += w
        out_x.append(ax / wsum)
        out_y.append(ay / wsum)
        out_z.append(az / wsum)
    return out_x, out_y, out_z


# -------------------- Opening 3D Brand Plaque --------------------

def create_opening_3d_brand_text(
    anchor_x: float,
    wall_y: float,
    anchor_z: float,
    brass_mat,
    dark_mat,
    collection,
):
    """Embed physical 3D extruded 'LUCA MARBLE MUSIC' channel branding on the opening wall.

    Mounted proud of the matte acoustic wall so studio spotlights cast real 3D shadows
    and the watermark is physically part of the scene geometry (resistant to cropping).
    """
    specs = [
        ("BrandTitle3D", "LUCA MARBLE MUSIC", 0.105, 0.018, 0.003, anchor_z, brass_mat),
        ("BrandHandle3D", "@LucaMarbleMusic", 0.064, 0.012, 0.002, anchor_z - 0.135, brass_mat),
    ]
    objs = []
    for name, text, size, extrude, bevel, z_pos, mat in specs:
        curve = bpy.data.curves.new(name=f"{name}_Curve", type="FONT")
        curve.body = text
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.size = size
        curve.extrude = extrude
        curve.bevel_depth = bevel
        curve.bevel_resolution = 3
        curve.space_character = 1.06

        obj = bpy.data.objects.new(name, curve)
        obj.location = (anchor_x, wall_y - 0.028, z_pos)
        obj.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
        if mat is not None:
            obj.data.materials.append(mat)
        collection.objects.link(obj)
        objs.append(obj)

    # Subtle dark bevelled backing bar behind the 3D lettering for crisp contrast & shadow catch
    backing_w = 1.48
    backing_h = 0.34
    backing_d = 0.016
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(anchor_x, wall_y - backing_d * 0.5, anchor_z - 0.062))
    backing = bpy.context.active_object
    backing.name = "BrandPlaqueBacking"
    backing.scale = (backing_w, backing_d, backing_h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if dark_mat is not None:
        backing.data.materials.append(dark_mat)
    for c in list(backing.users_collection):
        c.objects.unlink(backing)
    collection.objects.link(backing)
    objs.append(backing)
    return objs


# -------------------- 64-Instrument Per-Part Hit Animation & Spawning --------------------

_HIGHLIGHT_TOKEN_HINTS = (
    "_WKey_", "_BKey_", "_ToneBar_", "_SteelBar_", "_CombTooth_", "_Tine_",
    "_Petal_", "_GongDish", "_GongBoss", "_VU_", "_SqStep_", "_SqVert_",
    "_SawRamp_", "_SawDrop_", "_PadOuter", "_PadInner", "_PadCore",
    "_ScreenGlass", "_Bell", "_BigBell", "_WideBell", "_UpBell", "_FlareBell",
    "_PearBell", "_Str_", "_HarpStr_", "_PluckedStr", "_MelodyStr_", "_DroneStr_",
    "_Pipe_", "_FanPipe_", "_Drawbar_", "_ModWheel", "_MalletHead", "_MalletYarn",
    "_MalletBall", "_MalletFleece",
)


def _should_highlight_part(sub_name: str, anim_family: str) -> bool:
    for tok in _HIGHLIGHT_TOKEN_HINTS:
        if tok in sub_name:
            return True
    if anim_family == "ocarina_pulse" and ("Ocarina" in sub_name or "Body" in sub_name):
        return True
    return False


def _filter_parts(spawned_subs: list[dict], *tokens: str) -> list[dict]:
    out = []
    for p in spawned_subs:
        nm = p["base_name"]
        if any(t in nm for t in tokens):
            out.append(p)
    return out


def _pick_pitch_matched_part(parts: list[dict], pitch_frac: float) -> dict | None:
    if not parts:
        return None
    sorted_parts = sorted(parts, key=lambda p: p["orig_loc"].x)
    idx = min(len(sorted_parts) - 1, max(0, int(round(pitch_frac * (len(sorted_parts) - 1)))))
    return sorted_parts[idx]


def keyframe_part_transform(
    part: dict,
    keyframes: list[tuple[int, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    pivot: Vector | None = None,
) -> None:
    """Keyframe a spawned child part around an arbitrary local pivot point.

    Each keyframe tuple is: (frame, rot_deg_xyz, trans_xyz, scale_mul_xyz).
    """
    sobj = part["obj"]
    orig_loc: Vector = part["orig_loc"]
    orig_rot_mat = part["orig_rot"].to_matrix()
    orig_scale: Vector = part["orig_scale"]
    piv = pivot if pivot is not None else orig_loc

    sobj.rotation_mode = "XYZ"
    for frame, rot_deg, trans, scale_mul in keyframes:
        rx, ry, rz = (math.radians(a) for a in rot_deg)
        r_delta = Euler((rx, ry, rz), "XYZ").to_matrix()
        loc = piv + (r_delta @ (orig_loc - piv)) + Vector(trans)
        sobj.location = loc
        sobj.rotation_euler = (r_delta @ orig_rot_mat).to_euler("XYZ")
        sobj.scale = Vector((
            orig_scale.x * scale_mul[0],
            orig_scale.y * scale_mul[1],
            orig_scale.z * scale_mul[2],
        ))
        f_clamped = max(0, int(frame))
        sobj.keyframe_insert(data_path="location", frame=f_clamped)
        sobj.keyframe_insert(data_path="rotation_euler", frame=f_clamped)
        sobj.keyframe_insert(data_path="scale", frame=f_clamped)


def keyframe_emission_pulse(
    em_sock,
    f_hit: int,
    base_val: float,
    peak_val: float,
    decay_frames: int,
    bsdf=None,
    glow_color: tuple[float, float, float] | None = None,
) -> None:
    if em_sock is None:
        return
    if bsdf is not None and glow_color is not None:
        set_socket(bsdf, "Emission Color", (*glow_color, 1.0))
    em_sock.default_value = base_val
    em_sock.keyframe_insert(data_path="default_value", frame=max(0, f_hit - 1))
    em_sock.default_value = peak_val
    em_sock.keyframe_insert(data_path="default_value", frame=f_hit)
    em_sock.default_value = base_val + 0.38 * (peak_val - base_val)
    em_sock.keyframe_insert(data_path="default_value", frame=f_hit + 2)
    em_sock.default_value = base_val
    em_sock.keyframe_insert(data_path="default_value", frame=f_hit + max(4, decay_frames))


def _create_shockwave_ring(
    col_fx,
    ring_mesh,
    idx: int,
    contact_pos: Vector,
    rot_mat: Matrix,
    glow_color: tuple[float, float, float],
    f_hit: int,
    vel_scale: float,
) -> None:
    n = Vector((rot_mat[0][2], rot_mat[1][2], rot_mat[2][2])).normalized()
    ring_obj = bpy.data.objects.new(f"ImpactRing_{idx:03d}", ring_mesh)
    col_fx.objects.link(ring_obj)
    rmat, r_em = make_shockwave_material(f"RingMat_{idx:03d}", glow_color)
    ring_obj.material_slots[0].link = "OBJECT"
    ring_obj.material_slots[0].material = rmat

    ring_obj.location = contact_pos + n * 0.006
    ring_obj.rotation_mode = "QUATERNION"
    ring_obj.rotation_quaternion = rot_mat.to_quaternion()

    ring_obj.hide_viewport = True
    ring_obj.hide_render = True
    ring_obj.scale = (0.35, 0.35, 0.35)
    ring_obj.keyframe_insert(data_path="hide_viewport", frame=max(0, f_hit - 1))
    ring_obj.keyframe_insert(data_path="hide_render", frame=max(0, f_hit - 1))
    ring_obj.keyframe_insert(data_path="scale", frame=max(0, f_hit - 1))

    ring_obj.hide_viewport = False
    ring_obj.hide_render = False
    ring_obj.scale = (0.42, 0.42, 0.42)
    ring_obj.keyframe_insert(data_path="hide_viewport", frame=f_hit)
    ring_obj.keyframe_insert(data_path="hide_render", frame=f_hit)
    ring_obj.keyframe_insert(data_path="scale", frame=f_hit)

    ring_obj.scale = (1.42, 1.42, 1.42)
    ring_obj.keyframe_insert(data_path="scale", frame=f_hit + 5)

    ring_obj.hide_viewport = True
    ring_obj.hide_render = True
    ring_obj.keyframe_insert(data_path="hide_viewport", frame=f_hit + 6)
    ring_obj.keyframe_insert(data_path="hide_render", frame=f_hit + 6)

    r_em_sock = r_em.inputs["Strength"]
    r_em_sock.default_value = 0.0
    r_em_sock.keyframe_insert(data_path="default_value", frame=max(0, f_hit - 1))
    r_em_sock.default_value = 8.5 * vel_scale
    r_em_sock.keyframe_insert(data_path="default_value", frame=f_hit)
    r_em_sock.default_value = 0.0
    r_em_sock.keyframe_insert(data_path="default_value", frame=f_hit + 5)


def animate_instrument_hit(
    core_obj: bpy.types.Object,
    spawned_subs: list[dict],
    board: dict,
    mat: bpy.types.Material,
    glow_color: tuple[float, float, float],
    hit_glow: float,
    anim_family: str,
    vel_scale: float,
    note_dur: float,
    fps: int,
    rot_mat: Matrix,
) -> None:
    """Apply bespoke velocity-coupled kinetic & emissive hit feedback to both Core bar and v3.0 parts."""
    f_hit = int(board["frame_hit"])
    midi_note = int(board.get("note", 60))
    pitch_frac = max(0.0, min(1.0, (midi_note - 45) / 36.0))
    base_pos = Vector(board["center"])
    u = Vector(board["axis_u"]).normalized()
    n = Vector(board["axis_n"]).normalized()
    base_quat = rot_mat.to_quaternion()

    core_obj.keyframe_insert(data_path="location", frame=max(0, f_hit))
    core_obj.keyframe_insert(data_path="rotation_quaternion", frame=max(0, f_hit))

    # 1. Core Strike Bar Kinetic Motion
    if anim_family in ("keyboard_hinge", "keyboard_pluck", "organ_drawbars", "organ_pipes", "bellows_accordion"):
        press_quat = (rot_mat @ Matrix.Rotation(math.radians(6.2 * vel_scale), 3, "X")).to_quaternion()
        rebound_quat = (rot_mat @ Matrix.Rotation(math.radians(-0.8), 3, "X")).to_quaternion()
        core_obj.location = base_pos - n * (0.015 * vel_scale)
        core_obj.rotation_quaternion = press_quat
        core_obj.keyframe_insert(data_path="location", frame=f_hit + 1)
        core_obj.keyframe_insert(data_path="rotation_quaternion", frame=f_hit + 1)
        core_obj.location = base_pos + n * 0.002
        core_obj.rotation_quaternion = rebound_quat
        core_obj.keyframe_insert(data_path="location", frame=f_hit + 5)
        core_obj.keyframe_insert(data_path="rotation_quaternion", frame=f_hit + 5)
        core_obj.location = base_pos
        core_obj.rotation_quaternion = base_quat
        core_obj.keyframe_insert(data_path="location", frame=f_hit + 9)
        core_obj.keyframe_insert(data_path="rotation_quaternion", frame=f_hit + 9)
        decay_frames = max(12, min(24, int(note_dur * fps * 1.1)))
    elif anim_family in ("woodblock_clapper",):
        snap_quat = (rot_mat @ Matrix.Rotation(math.radians(4.8 * vel_scale), 3, "X")).to_quaternion()
        core_obj.location = base_pos - n * (0.020 * vel_scale)
        core_obj.rotation_quaternion = snap_quat
        core_obj.keyframe_insert(data_path="location", frame=f_hit + 1)
        core_obj.keyframe_insert(data_path="rotation_quaternion", frame=f_hit + 1)
        core_obj.location = base_pos
        core_obj.rotation_quaternion = base_quat
        core_obj.keyframe_insert(data_path="location", frame=f_hit + 5)
        core_obj.keyframe_insert(data_path="rotation_quaternion", frame=f_hit + 5)
        decay_frames = 8
    elif anim_family in ("tubular_pendulum", "mallet_bars", "celesta_bars", "gong_pendulum", "steelpan_petals"):
        core_obj.location = base_pos - n * (0.018 * vel_scale)
        core_obj.keyframe_insert(data_path="location", frame=f_hit + 1)
        core_obj.location = base_pos
        core_obj.keyframe_insert(data_path="location", frame=f_hit + 7)
        for k_off, deg in ((2, 3.0), (6, -1.9), (11, 1.0), (17, -0.4), (22, 0.0)):
            sway_q = (rot_mat @ Matrix.Rotation(math.radians(deg * vel_scale), 3, "Y")).to_quaternion()
            core_obj.rotation_quaternion = sway_q
            core_obj.keyframe_insert(data_path="rotation_quaternion", frame=f_hit + k_off)
        decay_frames = max(18, min(32, int(note_dur * fps * 1.4)))
    elif anim_family in ("plucked_strings", "electric_guitar", "harp_strings", "mallet_strings", "kalimba_tines", "bowed_strings"):
        for k_off, amp in ((1, -0.016), (3, 0.008), (5, -0.006), (7, 0.003), (10, -0.002), (12, 0.0)):
            core_obj.location = base_pos + n * (amp * vel_scale)
            core_obj.keyframe_insert(data_path="location", frame=f_hit + k_off)
        decay_frames = max(14, min(26, int(note_dur * fps * 1.1)))
    else:
        core_obj.location = base_pos - n * (0.020 * vel_scale)
        core_obj.keyframe_insert(data_path="location", frame=f_hit + 1)
        core_obj.location = base_pos
        core_obj.keyframe_insert(data_path="location", frame=f_hit + 7)
        decay_frames = max(12, min(24, int(note_dur * fps * 1.0)))

    # 2. Core Material Emission Pulse
    bsdf = next(nd for nd in mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    keyframe_emission_pulse(
        bsdf.inputs.get("Emission Strength"),
        f_hit,
        base_val=0.0,
        peak_val=hit_glow * vel_scale * 1.45,
        decay_frames=decay_frames,
        bsdf=bsdf,
        glow_color=glow_color,
    )

    # 3. Bespoke Sculpted v3.0 Sub-Part Kinetic Animations + Per-Part Glow
    keys = _filter_parts(spawned_subs, "_WKey_", "_BKey_")
    if anim_family in ("keyboard_hinge", "keyboard_pluck") and keys:
        hero_key = _pick_pitch_matched_part(keys, pitch_frac)
        if hero_key is not None:
            piv = Vector((hero_key["orig_loc"].x, hero_key["orig_loc"].y + 0.045, hero_key["orig_loc"].z + 0.005))
            keyframe_part_transform(hero_key, [
                (f_hit - 1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 1, (-7.2 * vel_scale, 0.0, 0.0), (0.0, 0.0, -0.014 * vel_scale), (1.0, 1.0, 1.0)),
                (f_hit + 5, (0.8, 0.0, 0.0), (0.0, 0.0, 0.002), (1.0, 1.0, 1.0)),
                (f_hit + 10, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ], pivot=piv)

    bars = _filter_parts(spawned_subs, "_ToneBar_", "_SteelBar_")
    if anim_family in ("mallet_bars", "celesta_bars") and bars:
        hero_bar = _pick_pitch_matched_part(bars, pitch_frac)
        if hero_bar is not None:
            keyframe_part_transform(hero_bar, [
                (f_hit - 1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 1, (0.0, 0.0, 0.0), (0.0, 0.0, -0.015 * vel_scale), (1.0, 1.0, 1.0)),
                (f_hit + 3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.008 * vel_scale), (1.0, 1.0, 1.0)),
                (f_hit + 6, (0.0, 0.0, 0.0), (0.0, 0.0, -0.004 * vel_scale), (1.0, 1.0, 1.0)),
                (f_hit + 10, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ])

    mallets = _filter_parts(spawned_subs, "_Mallet", "_MalletHead", "_MalletYarn", "_MalletBall", "_MalletFleece")
    if mallets:
        piv_m = Vector((0.0, 0.22, 0.14))
        for mp in mallets:
            keyframe_part_transform(mp, [
                (f_hit - 3, (-14.0, 0.0, 0.0), (0.0, 0.0, 0.020), (1.0, 1.0, 1.0)),
                (f_hit, (12.0 * vel_scale, 0.0, 0.0), (0.0, -0.025, -0.018), (1.0, 1.0, 1.0)),
                (f_hit + 7, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ], pivot=piv_m)

    if anim_family == "music_box":
        for drum in _filter_parts(spawned_subs, "_PinDrum", "_GearWheel"):
            keyframe_part_transform(drum, [
                (f_hit - 1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 12, (36.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ])
        comb_tooth = _pick_pitch_matched_part(_filter_parts(spawned_subs, "_CombTooth_"), pitch_frac)
        if comb_tooth:
            keyframe_part_transform(comb_tooth, [
                (f_hit, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 1, (-10.0, 0.0, 0.0), (0.0, 0.0, -0.010), (1.0, 1.0, 1.0)),
                (f_hit + 4, (6.0, 0.0, 0.0), (0.0, 0.0, 0.005), (1.0, 1.0, 1.0)),
                (f_hit + 9, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ])

    if anim_family == "tubular_pendulum":
        chimes = _filter_parts(spawned_subs, "_Chime", "_Tube")
        hero_chime = _pick_pitch_matched_part(chimes, pitch_frac) or (chimes[0] if chimes else None)
        if hero_chime:
            piv_top = Vector((hero_chime["orig_loc"].x, hero_chime["orig_loc"].y, 0.28))
            keyframe_part_transform(hero_chime, [
                (f_hit, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 2, (4.8 * vel_scale, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 7, (-3.2 * vel_scale, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 14, (1.4 * vel_scale, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 22, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ], pivot=piv_top)

    if anim_family == "bellows_accordion":
        for bel in _filter_parts(spawned_subs, "_Bellow", "_Fold"):
            keyframe_part_transform(bel, [
                (f_hit - 1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 2, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.78, 1.0, 1.05)),
                (f_hit + 7, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.06, 1.0, 0.98)),
                (f_hit + 13, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ])

    if anim_family in ("plucked_strings", "electric_guitar", "harp_strings", "mallet_strings", "keyboard_pluck", "bowed_strings"):
        strs = _filter_parts(spawned_subs, "_Str_", "_HarpStr_", "_PluckedStr", "_MelodyStr_", "_DroneStr_")
        hero_str = _pick_pitch_matched_part(strs, pitch_frac) or (strs[0] if strs else None)
        for sp in ([hero_str] if (hero_str and len(strs) > 2) else strs[:3]):
            if sp is None:
                continue
            keyframe_part_transform(sp, [
                (f_hit, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 1, (0.0, 0.0, 0.0), (0.0, -0.008 * vel_scale, 0.008 * vel_scale), (1.0, 1.0, 1.0)),
                (f_hit + 3, (0.0, 0.0, 0.0), (0.0, 0.006 * vel_scale, -0.006 * vel_scale), (1.0, 1.0, 1.0)),
                (f_hit + 5, (0.0, 0.0, 0.0), (0.0, -0.003 * vel_scale, 0.003 * vel_scale), (1.0, 1.0, 1.0)),
                (f_hit + 9, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ])

    if anim_family == "bowed_strings":
        for bow in _filter_parts(spawned_subs, "_BowStick", "_BowHair", "_CelloBow", "_BassBow", "_GoldBow"):
            keyframe_part_transform(bow, [
                (f_hit - 2, (0.0, 0.0, 0.0), (-0.065, 0.0, -0.025), (1.0, 1.0, 1.0)),
                (f_hit + 2, (0.0, 3.5, 0.0), (0.085 * vel_scale, 0.0, 0.032), (1.0, 1.0, 1.0)),
                (f_hit + 8, (0.0, -2.5, 0.0), (-0.045, 0.0, -0.015), (1.0, 1.0, 1.0)),
                (f_hit + 14, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ])

    if anim_family == "gong_pendulum":
        gong_piv = Vector((0.0, 0.25, 0.30))
        for gp in _filter_parts(spawned_subs, "_GongDish", "_GongBoss"):
            keyframe_part_transform(gp, [
                (f_hit, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 2, (-7.5 * vel_scale, 0.0, 0.0), (0.0, 0.022, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 8, (5.2 * vel_scale, 0.0, 0.0), (0.0, -0.014, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 16, (-2.2 * vel_scale, 0.0, 0.0), (0.0, 0.006, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 24, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ], pivot=gong_piv)

    if anim_family == "trombone_slide":
        for sl in _filter_parts(spawned_subs, "_OuterSlide"):
            keyframe_part_transform(sl, [
                (f_hit - 1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 2, (0.0, 0.0, 0.0), (-0.085 * vel_scale, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 8, (0.0, 0.0, 0.0), (0.018, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 13, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ])

    if anim_family in ("brass_valves", "woodwind_keys", "brass_fan", "organ_drawbars"):
        valves = _filter_parts(spawned_subs, "_Valve_", "_ValveCap_", "_Key_", "_Drawbar_")
        if valves:
            vpick = _pick_pitch_matched_part(valves, pitch_frac)
            for vp in ([vpick] if vpick else valves[:2]):
                dz = -0.022 * vel_scale if "_Valve" in vp["base_name"] else 0.0
                dy = -0.014 if "_Key_" in vp["base_name"] else (-0.032 if "_Drawbar_" in vp["base_name"] else 0.0)
                keyframe_part_transform(vp, [
                    (f_hit - 1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                    (f_hit + 1, (0.0, 0.0, 0.0), (0.0, dy, dz), (1.0, 1.0, 1.0)),
                    (f_hit + 8, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ])

    if anim_family == "bagpipe_bag":
        for bag in _filter_parts(spawned_subs, "_Bag"):
            keyframe_part_transform(bag, [
                (f_hit - 1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 2, (0.0, 0.0, 0.0), (0.0, 0.012, 0.0), (0.94, 0.86, 0.94)),
                (f_hit + 10, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ])

    if anim_family == "steelpan_petals":
        petal = _pick_pitch_matched_part(_filter_parts(spawned_subs, "_Petal_"), pitch_frac)
        if petal:
            keyframe_part_transform(petal, [
                (f_hit, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 1, (0.0, 0.0, 0.0), (0.0, 0.008, 0.0), (1.16, 1.16, 1.16)),
                (f_hit + 8, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ])

    if anim_family in ("synth_euclid", "synth_saw", "synth_pad"):
        for wp in _filter_parts(spawned_subs, "_SqStep_", "_SawRamp_", "_PadOuter", "_PadInner", "_PadCore"):
            peak_sc = 1.0 + 0.16 * vel_scale
            keyframe_part_transform(wp, [
                (f_hit - 1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 1, (0.0, 0.0, 0.0), (0.0, -0.006, 0.0), (peak_sc, 1.0, peak_sc)),
                (f_hit + 10, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ])
        for wh in _filter_parts(spawned_subs, "_PitchWheel", "_ModWheel"):
            keyframe_part_transform(wh, [
                (f_hit - 1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 1, (-18.0 * vel_scale, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                (f_hit + 9, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ])

    # 4. Per-Part Selective Emissive Pulse (VU LED ladder scaled by velocity + active part glow)
    vu_parts = sorted(_filter_parts(spawned_subs, "_VU_"), key=lambda p: p["orig_loc"].z)
    vu_lit = max(1, int(round(len(vu_parts) * min(1.0, vel_scale * 0.85)))) if vu_parts else 0
    for idx_v, vp in enumerate(vu_parts):
        esock = vp.get("em_sock")
        if esock is None:
            continue
        if idx_v < vu_lit:
            keyframe_emission_pulse(esock, f_hit, base_val=0.8, peak_val=7.5 * vel_scale, decay_frames=12)
        else:
            keyframe_emission_pulse(esock, f_hit, base_val=0.15, peak_val=0.35, decay_frames=6)

    for sp in spawned_subs:
        if "_VU_" in sp["base_name"]:
            continue
        esock = sp.get("em_sock")
        if esock is None:
            continue
        base_em = sp.get("base_em", 0.0)
        peak_em = max(base_em + 1.5, hit_glow * vel_scale * (1.8 if "led" in sp.get("mat_name", "").lower() else 1.15))
        keyframe_emission_pulse(
            esock,
            f_hit,
            base_val=base_em,
            peak_val=peak_em,
            decay_frames=decay_frames,
            bsdf=sp.get("bsdf"),
            glow_color=glow_color if base_em < 0.5 else None,
        )


def spawn_instrument_board(
    board: dict,
    templates: dict[int, dict],
    lib_mats: dict[str, bpy.types.Material],
    manifest: dict,
    b_col: bpy.types.Collection,
    col_fx: bpy.types.Collection,
    ring_mesh: bpy.types.Mesh,
    notes_meta: list[dict],
    fps: int,
) -> tuple[bpy.types.Object, list[dict]]:
    """Instantiate core collision bar + sculpted v3.0 assembly & keyframe its hit response."""
    idx = int(board["index"])
    gm = resolve_board_gm_program(board, manifest)
    tpl = templates.get(gm) or templates.get(-1000 - int(board.get("track_idx", 1))) or next(iter(templates.values()))
    anim_family = str(tpl.get("anim_family", "generic_strike"))

    core_mesh = create_hard_beveled_box(
        f"BoardMesh_{idx:03d}", float(board["length"]), float(board["width_y"]), float(board["thickness"])
    )
    inst_label = str(board.get("instrument", tpl.get("name", "Instrument"))).replace(" ", "")
    obj = bpy.data.objects.new(f"Board_{idx:03d}_{inst_label}_{board.get('pitch_name', '')}", core_mesh)
    b_col.objects.link(obj)

    mat, glow_color, hit_glow = get_board_material_and_glow(board, lib_mats, templates=templates, manifest=manifest)
    mat.name = f"BoardMat_{idx:03d}"
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    spawned_subs: list[dict] = []
    # Keep object-linked highlight copies per board so emissive hit pulses stay localized to THIS board
    local_glow_cache: dict[str, tuple[bpy.types.Material, object, object, float]] = {}

    for sub_name, sub_data, sub_mats, sub_loc, sub_rot, sub_scale in tpl["subs"]:
        sobj = bpy.data.objects.new(f"{sub_name}_{idx:03d}", sub_data)
        b_col.objects.link(sobj)
        sobj.parent = obj
        sobj.location = sub_loc
        sobj.rotation_mode = "XYZ"
        sobj.rotation_euler = sub_rot
        sobj.scale = sub_scale

        # Preserve full multi-material slots from v3.0 asset
        if not sobj.data.materials and sub_mats:
            for sm in sub_mats:
                sobj.data.materials.append(sm)

        em_sock = None
        bsdf_sub = None
        base_em = 0.0
        mat_name = sub_mats[0].name if sub_mats else ""

        if _should_highlight_part(sub_name, anim_family) and sub_mats and len(sobj.material_slots) > 0:
            orig_m = sub_mats[0]
            cache_key = f"{orig_m.name}|{'VU' if '_VU_' in sub_name else 'GEN'}"
            if "_VU_" in sub_name or cache_key not in local_glow_cache:
                m_copy = orig_m.copy()
                m_copy.name = f"PartGlow_{idx:03d}_{len(local_glow_cache):02d}"
                bsdf_c = next((nd for nd in m_copy.node_tree.nodes if nd.type == "BSDF_PRINCIPLED"), None)
                esock_c = bsdf_c.inputs.get("Emission Strength") if bsdf_c else None
                b_em = float(esock_c.default_value) if esock_c is not None else 0.0
                if "_VU_" in sub_name and esock_c is not None:
                    b_em = 0.15 if idx >= 0 else b_em
                local_glow_cache[cache_key] = (m_copy, bsdf_c, esock_c, b_em)
            m_copy, bsdf_sub, em_sock, base_em = local_glow_cache[cache_key]
            sobj.material_slots[0].link = "OBJECT"
            sobj.material_slots[0].material = m_copy

        spawned_subs.append({
            "obj": sobj,
            "base_name": sub_name,
            "orig_loc": sub_loc.copy(),
            "orig_rot": sub_rot.copy(),
            "orig_scale": sub_scale.copy(),
            "bsdf": bsdf_sub,
            "em_sock": em_sock,
            "base_em": base_em,
            "mat_name": mat_name,
        })

    u = Vector(board["axis_u"]).normalized()
    v = Vector(board["axis_v"]).normalized()
    n = Vector(board["axis_n"]).normalized()
    rot_mat = Matrix((
        (u.x, v.x, n.x),
        (u.y, v.y, n.y),
        (u.z, v.z, n.z),
    ))
    obj.location = Vector(board["center"])
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = rot_mat.to_quaternion()

    f_hit = int(board["frame_hit"])
    n_meta = notes_meta[idx] if idx < len(notes_meta) else {}
    vel = float(board.get("velocity", n_meta.get("velocity", 100.0)))
    note_dur = float(board.get("duration", n_meta.get("duration", 0.25)))
    vel_scale = 0.55 + 0.65 * max(0.0, min(1.0, vel / 127.0))

    animate_instrument_hit(
        obj,
        spawned_subs,
        board,
        mat,
        glow_color,
        hit_glow,
        anim_family,
        vel_scale,
        note_dur,
        fps,
        rot_mat,
    )
    _create_shockwave_ring(
        col_fx,
        ring_mesh,
        idx,
        Vector(board["contact"]),
        rot_mat,
        glow_color,
        f_hit,
        vel_scale,
    )
    return obj, spawned_subs


# -------------------- Main Scene Builder --------------------

def build_scene(
    physics_json: str,
    out_blend: str | None = None,
    frames_dir: str | None = None,
    audio_path: str | None = None,
    lib_blend: str = DEFAULT_LIB_BLEND,
    manifest_json: str = DEFAULT_MANIFEST_JSON,
    max_frame: int | None = None,
):
    with open(physics_json, "r", encoding="utf-8") as f:
        plan = json.load(f)

    assert plan["validation"]["ok"], f"Physics plan failed validation: {plan['validation']}"

    clear_scene()
    scene = bpy.context.scene
    col = scene.collection

    # Load reusable 64-Instrument Asset Library (.blend + manifest)
    templates, lib_mats, manifest = load_instrument_library(lib_blend, manifest_json)

    fps = int(plan["fps"])
    total_frames = int(plan["total_frames"])
    if max_frame is not None and max_frame > 0:
        total_frames = min(total_frames, int(max_frame))
    radius = float(plan["marble_radius"])
    # Deep v3.0 sculptures reach Y ~ +0.58; push back wall & spine safely behind them
    wall_y = max(float(plan.get("back_wall_y", 0.52)), 0.78)
    boards = plan["boards"]
    rails = plan["rails"]
    samples = [s for s in plan["samples"] if int(s["frame"]) <= total_frames]
    notes_meta = load_note_metadata(physics_json)

    # Configure Render, EEVEE Raytracing & Color Management
    scene.render.engine = get_eevee_engine()
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    scene.frame_start = 0
    scene.frame_end = total_frames
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        scene.render.filepath = os.path.join(os.path.abspath(frames_dir), "frame_")

    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 28
        if hasattr(scene.eevee, "use_shadows"):
            scene.eevee.use_shadows = True
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = True
        if hasattr(scene.eevee, "gi_cubemap_resolution"):
            scene.eevee.gi_cubemap_resolution = "2048"
        if hasattr(scene.eevee, "ray_tracing_options"):
            rto = scene.eevee.ray_tracing_options
            if hasattr(rto, "resolution_scale"):
                rto.resolution_scale = "1"
            if hasattr(rto, "screen_trace_quality"):
                rto.screen_trace_quality = 0.85

    scene.view_settings.view_transform = "AgX" if "AgX" in [
        i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items
    ] else "Filmic"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        try:
            scene.view_settings.look = "Medium High Contrast"
        except Exception:
            pass
    scene.view_settings.exposure = -0.05

    build_studio_reflection_world(scene)
    setup_bloom_compositor(scene)

    # Shared studio hardware materials from Asset Library
    wall_felt_mat = lib_mats["LibMat_WallFelt"]
    wood_slat_mat = lib_mats["LibMat_WoodSlatDark"]
    brass_trim_mat = lib_mats.get("LibMat_WallRevealDark", lib_mats["LibMat_WallBrassTrim"])
    chrome_mat = lib_mats["LibMat_ChromeRail"]
    hardware_dark_mat = lib_mats["LibMat_HardwareDark"]
    marble_mat = lib_mats["LibMat_EmeraldMarble"]

    # Clean organized collections for Blender Outliner
    col_marble = get_or_create_collection("01_Marble", col)
    col_instruments = get_or_create_collection("02_Instruments", col)
    col_rails = get_or_create_collection("03_GuideRails", col)
    col_stage = get_or_create_collection("04_StudioStage", col)
    col_fx = get_or_create_collection("05_ImpactFX", col)

    # Dark Matte Stage Wall + Outer Edge Slats + Brass Phrase Dividers
    all_z = [s["position"][2] for s in samples]
    phrase_breaks_z: list[float] = []
    for k_b in range(1, len(boards)):
        if boards[k_b]["track_idx"] != boards[k_b - 1]["track_idx"]:
            phrase_breaks_z.append(0.5 * (boards[k_b]["center"][2] + boards[k_b - 1]["center"][2]))

    create_dark_acoustic_stage_wall(
        min(all_z),
        max(all_z),
        wall_y,
        wall_felt_mat,
        wood_slat_mat,
        brass_trim_mat,
        col_stage,
        phrase_breaks_z=phrase_breaks_z,
    )

    if samples and boards:
        brand_x = 0.5 * (float(samples[0]["position"][0]) + float(boards[0]["center"][0]))
        brand_z = float(samples[0]["position"][2]) + 0.34
    else:
        brand_x = 0.0
        brand_z = max(all_z) + 0.35
    create_opening_3d_brand_text(
        brand_x,
        wall_y,
        brand_z,
        brass_trim_mat,
        wood_slat_mat,
        col_stage,
    )

    # Modular Phrase Mounting Spines pushed cleanly behind v3.0 assemblies
    spine_y = wall_y - 0.14
    phrase_clusters = build_phrase_clusters(boards, max_cluster=6)
    brass_accent_mat = lib_mats["LibMat_WallBrassTrim"]
    board_to_spine_x = create_modular_phrase_racks(
        phrase_clusters,
        wall_y=wall_y,
        spine_y=spine_y,
        walnut_mat=wood_slat_mat,
        brass_mat=brass_accent_mat,
        bronze_mat=hardware_dark_mat,
        collection=col_stage,
    )

    # Camera target empty + localized tracking lighting rig
    target_empty = bpy.data.objects.new("CamTarget", None)
    col_stage.objects.link(target_empty)

    # Warm Hero Spotlight tracking marble zone (diagonal wall shadows + natural dark vignette)
    hero_data = bpy.data.lights.new("HeroWarmSpot", type="SPOT")
    hero_data.energy = 820.0
    hero_data.color = (1.0, 0.90, 0.78)
    hero_data.spot_size = math.radians(58.0)
    hero_data.spot_blend = 0.55
    hero_data.shadow_soft_size = 0.22
    hero_light = bpy.data.objects.new("HeroWarmSpot", hero_data)
    col_stage.objects.link(hero_light)
    hero_light.parent = target_empty
    hero_light.location = (-2.3, -3.6, 2.5)
    h_cons = hero_light.constraints.new(type="TRACK_TO")
    h_cons.target = target_empty
    h_cons.track_axis = "TRACK_NEGATIVE_Z"
    h_cons.up_axis = "UP_Y"

    # Cool Ice-Blue Rim Area light from opposite side for crisp metallic edge highlights
    rim_data = bpy.data.lights.new("CoolRimArea", type="AREA")
    rim_data.energy = 240.0
    rim_data.size = 0.65
    rim_data.color = (0.64, 0.84, 1.0)
    rim_light = bpy.data.objects.new("CoolRimArea", rim_data)
    col_stage.objects.link(rim_light)
    rim_light.parent = target_empty
    rim_light.location = (3.4, -0.7, 1.3)
    r_cons = rim_light.constraints.new(type="TRACK_TO")
    r_cons.target = target_empty
    r_cons.track_axis = "TRACK_NEGATIVE_Z"
    r_cons.up_axis = "UP_Y"

    ring_mesh = create_shockwave_ring_mesh()
    ring_mesh.materials.append(None)
    track_cols: dict[int, bpy.types.Collection] = {}

    # 1. Instantiate Boards + v3.0 Sculpted Sub-Assemblies with Per-Part Hit Animations
    for b in boards:
        idx = int(b["index"])
        t_idx = int(b["track_idx"])
        if t_idx not in track_cols:
            inst_clean = str(b["instrument"]).replace(" ", "")
            track_cols[t_idx] = get_or_create_collection(f"Track_{t_idx:02d}_{inst_clean}", col_instruments)
        b_col = track_cols[t_idx]

        spawn_instrument_board(
            b,
            templates,
            lib_mats,
            manifest,
            b_col,
            col_fx,
            ring_mesh,
            notes_meta,
            fps,
        )

        # Connect board mounting pins cleanly into the midground modular spine rack
        sx = board_to_spine_x.get(idx, b["center"][0])
        for p_idx, pin in enumerate(b["pins"]):
            spine_anchor = (sx, spine_y, pin["a"][2])
            mid_joint = (pin["a"][0], spine_y, pin["a"][2])
            create_cylinder_between(
                f"BoardPin_{idx:03d}_{p_idx}",
                pin["a"],
                mid_joint,
                pin["radius"] * 0.85,
                hardware_dark_mat,
                b_col,
                vertices=24,
            )
            create_cylinder_between(
                f"BoardArm_{idx:03d}_{p_idx}",
                mid_joint,
                spine_anchor,
                pin["radius"] * 0.72,
                brass_accent_mat,
                b_col,
                vertices=24,
            )

    # 2. Slim Restrained Bronze Guide Rails + Sparse U-Ties & 2 Rear Wall Struts
    rail_bronze_mat = lib_mats.get("LibMat_ResonatorBronze", chrome_mat)
    for r in rails:
        rid = r["id"]
        create_pipe_polyline(f"{rid}_front", r["pipe_front"], r["pipe_radius"], rail_bronze_mat, col_rails)
        create_pipe_polyline(f"{rid}_back", r["pipe_back"], r["pipe_radius"], rail_bronze_mat, col_rails)

        for t_idx, tie_pts in enumerate(r.get("ties", [])):
            create_pipe_polyline(f"{rid}_tie_{t_idx:02d}", tie_pts, 0.0085, hardware_dark_mat, col_rails)

        pts_back = r["pipe_back"]
        num_pts = len(pts_back)
        for frac in (0.22, 0.78):
            pk = pts_back[min(num_pts - 1, max(0, int(round(frac * (num_pts - 1)))))]
            a_pin = (pk[0], pk[1] + r["pipe_radius"] * 0.55, pk[2])
            b_pin = (pk[0], wall_y + 0.02, pk[2])
            create_cylinder_between(
                f"{rid}_pin_{int(frac*100)}",
                a_pin,
                b_pin,
                0.011,
                hardware_dark_mat,
                col_rails,
                vertices=24,
            )
            create_wall_flange(
                f"{rid}_flange_{int(frac*100)}",
                (pk[0], wall_y, pk[2]),
                0.026,
                hardware_dark_mat,
                col_rails,
            )

    # 3. Marble + Quaternion Rolling Spin + Emerald Glow Core Light (pulses on each hit)
    marble_mesh = bpy.data.meshes.new("MarbleMesh")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=80, v_segments=48, radius=radius)
    bm.to_mesh(marble_mesh)
    bm.free()
    for poly in marble_mesh.polygons:
        poly.use_smooth = True

    marble = bpy.data.objects.new("EmeraldMarble", marble_mesh)
    col_marble.objects.link(marble)
    marble.data.materials.append(marble_mat)
    marble.rotation_mode = "QUATERNION"

    # Green point light attached to the marble: flashes brightly on each impact for tactile punch
    mlight_data = bpy.data.lights.new("MarbleCoreGlow", type="POINT")
    mlight_data.energy = 1.6
    mlight_data.color = (0.20, 1.0, 0.50)
    mlight_data.shadow_soft_size = 0.08
    mlight = bpy.data.objects.new("MarbleCoreGlow", mlight_data)
    col_marble.objects.link(mlight)
    mlight.parent = marble

    mlight_data.keyframe_insert(data_path="energy", frame=0)
    for b in boards:
        fh = int(b["frame_hit"])
        if fh <= total_frames:
            mlight_data.energy = 1.6
            mlight_data.keyframe_insert(data_path="energy", frame=max(0, fh - 1))
            mlight_data.energy = 4.4
            mlight_data.keyframe_insert(data_path="energy", frame=fh)
            mlight_data.energy = 1.6
            mlight_data.keyframe_insert(data_path="energy", frame=fh + 4)

    spin_quat = Quaternion((1.0, 0.0, 0.0, 0.0))
    for idx_s, s in enumerate(samples):
        f = int(s["frame"])
        pos = Vector(s["position"])
        marble.location = pos
        marble.keyframe_insert(data_path="location", frame=f)
        if idx_s > 0:
            prev_pos = Vector(samples[idx_s - 1]["position"])
            dp = pos - prev_pos
            step_dist = dp.length
            if step_dist > 1e-6:
                rot_axis = Vector((0.0, 0.0, 1.0)).cross(dp)
                if rot_axis.length < 1e-5:
                    rot_axis = Vector((0.0, 1.0, 0.0))
                else:
                    rot_axis.normalize()
                dq = Quaternion(rot_axis, step_dist / radius)
                spin_quat = (dq @ spin_quat).normalized()
        marble.rotation_quaternion = spin_quat
        marble.keyframe_insert(data_path="rotation_quaternion", frame=f)

    if marble.animation_data and marble.animation_data.action:
        act = marble.animation_data.action
        fcurves = getattr(act, "fcurves", None)
        if fcurves is None and hasattr(act, "layers"):
            fcurves = []
            for layer in act.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        fcurves.extend(bag.fcurves)
        if fcurves:
            for fc in fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"

    # 4. Tighter 3/4 Oblique Look-Ahead Camera (marble ~8.1% width, U-loop pullback, reveals 3D depth)
    cam_data = bpy.data.cameras.new("MainCamera")
    cam_data.lens = 48.0
    cam_data.sensor_width = 36.0
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100.0
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = marble
    cam_data.dof.aperture_fstop = 4.0
    cam = bpy.data.objects.new("MainCamera", cam_data)
    col_stage.objects.link(cam)
    scene.camera = cam

    c_cons = cam.constraints.new(type="TRACK_TO")
    c_cons.target = target_empty
    c_cons.track_axis = "TRACK_NEGATIVE_Z"
    c_cons.up_axis = "UP_Y"

    cam_tx, cam_ys, cam_tz = plan_lookahead_camera(samples, boards, fps, half_window=24)
    for idx_s, s in enumerate(samples):
        f = int(s["frame"])
        tx = cam_tx[idx_s]
        yc = cam_ys[idx_s]
        tz = cam_tz[idx_s]
        cam.location = (0.78 + 0.35 * tx, yc, tz + 0.38)
        cam.keyframe_insert(data_path="location", frame=f)
        target_empty.location = (0.28 * tx, 0.05, tz - 0.08)
        target_empty.keyframe_insert(data_path="location", frame=f)

    # 5. Attach Pre-roll Synced Audio to Blender VSE
    if audio_path and os.path.exists(audio_path):
        scene.sequence_editor_create()
        seq = scene.sequence_editor
        strips = getattr(seq, "strips", None) or getattr(seq, "sequences", None)
        if strips is not None:
            strips.new_sound("MarbleAudio", os.path.abspath(audio_path), channel=1, frame_start=0)
        scene.use_audio = True
        scene.sync_mode = "AUDIO_SYNC"

    # 6. Configure Rendered/Material Viewport Defaults for Blender GUI Preview
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "RENDERED"
                        if hasattr(space, "region_3d") and space.region_3d is not None:
                            space.region_3d.view_perspective = "CAMERA"

    if out_blend:
        os.makedirs(os.path.dirname(os.path.abspath(out_blend)), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend))
        print(f"Saved Blender scene: {out_blend} (frames 0..{total_frames}, {len(boards)} boards, {len(rails)} rails)")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    max_frame = None
    if "--max-frame" in args:
        mi = args.index("--max-frame")
        max_frame = int(args[mi + 1])
        args = args[:mi] + args[mi + 2:]
    physics_json = args[0] if len(args) >= 1 else "projects/jingle-bells-marble-music/artifacts/sample_14s_physics.json"
    out_blend = args[1] if len(args) >= 2 else "projects/jingle-bells-marble-music/renders/sample_14s_physics.blend"
    frames_dir = args[2] if len(args) >= 3 else "projects/jingle-bells-marble-music/renders/frames"
    audio_path = args[3] if len(args) >= 4 else None
    build_scene(physics_json, out_blend=out_blend, frames_dir=frames_dir, audio_path=audio_path, max_frame=max_frame)
