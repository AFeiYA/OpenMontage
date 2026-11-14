#!/usr/bin/env python3
"""Builds the reusable Blender 5.3 Asset Library for Virtual 3D Marble Music instruments.

Outputs:
- projects/jingle-bells-marble-music/assets/blender_assets.cats.txt
- projects/jingle-bells-marble-music/assets/instrument_library.blend

Each of the 12 instruments is stored as a cataloged Blender Asset Collection (`Lib_01_Castanets` ..
`Lib_12_Guitar`) with:
- Exact unit-length (1.0u x 0.24u x 0.072u) collision-safe strike bar mesh (`LibCore_XX`).
- Sculpted acoustic sub-assembly (`LibSub_XX_*`) strictly mounted in Y >= +0.13 behind the
  marble corridor so scaling X to any note length preserves 240Hz collision clearance.
- Rich procedural PBR shader materials (`LibMat_*`) marked as Blender Assets for reuse.
"""

from __future__ import annotations

import math
import os
import sys

import bpy
import bmesh
from mathutils import Matrix, Vector


ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets")
)
LIB_BLEND_PATH = os.path.join(ASSETS_DIR, "instrument_library.blend")
CATS_FILE_PATH = os.path.join(ASSETS_DIR, "blender_assets.cats.txt")

# Fixed deterministic UUIDs for Blender Asset Browser catalogs
CAT_UUIDS = {
    "percussion": "a1b2c3d4-1111-4000-8000-000000000001",
    "keyboards": "a1b2c3d4-1111-4000-8000-000000000002",
    "melodic": "a1b2c3d4-1111-4000-8000-000000000003",
    "materials": "a1b2c3d4-1111-4000-8000-000000000004",
    "hardware": "a1b2c3d4-1111-4000-8000-000000000005",
}


def write_asset_catalogs():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    lines = [
        "# This is an Asset Catalog Definition file for Blender.",
        "#",
        "# Empty lines and lines starting with `#` will be ignored.",
        "# The first non-ignored line should be the version indicator.",
        "VERSION 1",
        "",
        f"{CAT_UUIDS['percussion']}:Instruments/Percussion:Percussion",
        f"{CAT_UUIDS['keyboards']}:Instruments/Keyboards & Bells:Keyboards-Bells",
        f"{CAT_UUIDS['melodic']}:Instruments/Melodic & Pluck:Melodic-Pluck",
        f"{CAT_UUIDS['materials']}:Materials/Studio PBR:Studio-PBR",
        f"{CAT_UUIDS['hardware']}:Stage & Hardware:Stage-Hardware",
        "",
    ]
    with open(CATS_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def set_socket(node, name: str, value):
    if name in node.inputs:
        node.inputs[name].default_value = value


def mark_asset(id_block, cat_key: str, description: str, tags: list[str]):
    id_block.asset_mark()
    ad = id_block.asset_data
    ad.catalog_id = CAT_UUIDS[cat_key]
    ad.author = "OpenMontage Studio"
    ad.description = description
    for t in tags:
        ad.tags.new(t, skip_if_exists=True)


def new_principled_material(
    name: str,
    base_color=(0.5, 0.5, 0.5),
    metallic=0.0,
    roughness=0.35,
    coat=0.0,
    coat_roughness=0.08,
    ior=1.45,
    anisotropic=0.0,
    emission_color=None,
    emission_strength=0.0,
    transmission=0.0,
):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    set_socket(bsdf, "Base Color", (*base_color[:3], 1.0))
    set_socket(bsdf, "Metallic", float(metallic))
    set_socket(bsdf, "Roughness", float(roughness))
    set_socket(bsdf, "IOR", float(ior))
    set_socket(bsdf, "Anisotropic", float(anisotropic))
    set_socket(bsdf, "Coat Weight", float(coat))
    set_socket(bsdf, "Coat Roughness", float(coat_roughness))
    set_socket(bsdf, "Transmission Weight", float(transmission))
    ec = emission_color if emission_color is not None else base_color
    set_socket(bsdf, "Emission Color", (*ec[:3], 1.0))
    set_socket(bsdf, "Emission Strength", float(emission_strength))
    return mat, nt, bsdf


def add_wood_grain_nodes(mat, c_dark, c_light, scale=6.0, bump_dist=0.0025, direction="Z", vec_scale=(1.5, 10.0, 4.0)):
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = vec_scale
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = direction
    wave.inputs["Scale"].default_value = scale
    wave.inputs["Distortion"].default_value = 5.5
    wave.inputs["Detail"].default_value = 3.5

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (*c_dark, 1.0)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (*c_light, 1.0)

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = bump_dist

    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], wave.inputs["Vector"])
    nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def add_brushed_metal_nodes(mat, r_min=0.11, r_max=0.22, scale=65.0, bump_dist=0.0005):
    """Fine directional micro-brush lines in Roughness + subtle Bump (keeps metallic albedo clean)."""
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 42.0, 42.0)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.65

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.25
    mr.inputs["From Max"].default_value = 0.75
    mr.inputs["To Min"].default_value = r_min
    mr.inputs["To Max"].default_value = r_max

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = bump_dist

    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def add_hammered_metal_nodes(mat, r_min=0.09, r_max=0.24):
    """High-frequency micro-hammered Voronoi dimples in Bump + Roughness (no camo albedo blotches)."""
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "F1"
    vor.inputs["Scale"].default_value = 140.0
    vor.inputs["Randomness"].default_value = 0.90

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.0
    mr.inputs["From Max"].default_value = 0.75
    mr.inputs["To Min"].default_value = r_min
    mr.inputs["To Max"].default_value = r_max

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.26
    bump.inputs["Distance"].default_value = 0.0016

    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    nt.links.new(vor.outputs["Distance"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    nt.links.new(vor.outputs["Distance"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def fix_mesh_normals(obj: bpy.types.Object, angle_deg: float = 35.0, weighted: bool = True):
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


def create_box_obj(name: str, dims, offset=(0.0, 0.0, 0.0), bevel=0.005, mat=None, col=None):
    mesh = bpy.data.meshes.new(name + "Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=dims, verts=bm.verts)
    if bevel > 0.0:
        bmesh.ops.bevel(
            bm,
            geom=list(bm.edges),
            offset=bevel,
            segments=3,
            profile=0.5,
            affect="EDGES",
            clamp_overlap=True,
            harden_normals=False,
        )
    bmesh.ops.translate(bm, vec=offset, verts=bm.verts)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    if col is not None:
        col.objects.link(obj)
    fix_mesh_normals(obj, angle_deg=35.0, weighted=True)
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def create_cyl_obj(name: str, radius: float, depth: float, center=(0.0, 0.0, 0.0), rot_euler=(0.0, 0.0, 0.0), mat=None, col=None, segs=28):
    mesh = bpy.data.meshes.new(name + "Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=segs, radius1=radius, radius2=radius, depth=depth
    )
    if any(abs(a) > 1e-6 for a in rot_euler):
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(rot_euler[0], 3, "X") @ Matrix.Rotation(rot_euler[1], 3, "Y") @ Matrix.Rotation(rot_euler[2], 3, "Z"), verts=bm.verts)
    bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    if col is not None:
        col.objects.link(obj)
    fix_mesh_normals(obj, angle_deg=35.0, weighted=True)
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def build_library():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    write_asset_catalogs()

    scene = bpy.context.scene
    root_col = scene.collection
    mat_col = bpy.data.collections.new("00_Materials_Palette")
    root_col.children.link(mat_col)

    # 1. Build Shared Studio & Hardware PBR Materials ("Mechanical Music Box" Aesthetic)
    mats: dict[str, bpy.types.Material] = {}

    # Satin Brushed Steel/Nickel Guide Rails (roughness=0.20 so rails stay restrained, letting Marble & Instruments pop)
    m, _, _ = new_principled_material("LibMat_ChromeRail", (0.64, 0.65, 0.68), metallic=0.96, roughness=0.20, anisotropic=0.68, coat=0.0)
    mark_asset(m, "hardware", "Satin brushed nickel-steel guide rail", ["satin", "rail", "metal"])
    mats["chrome_rail"] = m

    # Dark Aged Bronze for modular spine brackets, clamps & mounting pins
    m, _, _ = new_principled_material("LibMat_HardwareDark", (0.095, 0.075, 0.056), metallic=0.88, roughness=0.34, anisotropic=0.50, coat=0.0)
    mark_asset(m, "hardware", "Dark aged bronze for brackets, pins & wall flanges", ["bronze", "hardware"])
    mats["hardware_dark"] = m

    # Low-contrast matte warm-charcoal acoustic backdrop (recedes cleanly behind instruments)
    m, _, _ = new_principled_material("LibMat_WallFelt", (0.017, 0.016, 0.016), metallic=0.0, roughness=0.95)
    mark_asset(m, "hardware", "Low-contrast matte warm-charcoal acoustic backdrop", ["wall", "matte", "dark"])
    mats["wall_felt"] = m

    # Subdued smoked walnut for outer stage slats & modular spine beams (50% lower contrast)
    m, _, _ = new_principled_material("LibMat_WoodSlatDark", (0.020, 0.014, 0.010), metallic=0.0, roughness=0.72, coat=0.02, coat_roughness=0.35)
    add_wood_grain_nodes(m, (0.015, 0.010, 0.007), (0.024, 0.017, 0.012), scale=2.6, bump_dist=0.0008, direction="X", vec_scale=(6.0, 6.0, 0.30))
    mark_asset(m, "hardware", "Subdued smoked walnut architectural frame wood", ["walnut", "wood", "slat"])
    mats["wood_slat"] = m

    # Warm Brushed Music-Box Brass
    m, _, _ = new_principled_material("LibMat_WallBrassTrim", (0.82, 0.60, 0.22), metallic=0.98, roughness=0.18, anisotropic=0.60)
    mark_asset(m, "hardware", "Warm brushed music-box brass trim & brackets", ["brass", "trim"])
    mats["brass_trim"] = m

    m, _, _ = new_principled_material("LibMat_WallRevealDark", (0.14, 0.10, 0.055), metallic=0.72, roughness=0.36)
    mark_asset(m, "hardware", "Muted aged brass phrase divider reveal", ["bronze", "wall", "reveal"])
    mats["wall_reveal"] = m

    m, _, _ = new_principled_material("LibMat_ResonatorBronze", (0.28, 0.21, 0.14), metallic=0.96, roughness=0.22, anisotropic=0.62, coat=0.0)
    mark_asset(m, "materials", "Warm brushed bronze resonator pipe metal", ["bronze", "resonator"])
    mats["resonator_bronze"] = m

    m, _, _ = new_principled_material("LibMat_PorcelainCream", (0.92, 0.88, 0.79), metallic=0.0, roughness=0.13, coat=0.82, coat_roughness=0.03)
    mark_asset(m, "materials", "Warm ivory porcelain accent trim", ["ceramic", "ivory"])
    mats["porcelain_cream"] = m

    # Crystal Emerald Swirl Marble Shader
    m, nt, bsdf = new_principled_material(
        "LibMat_EmeraldMarble",
        (0.01, 0.42, 0.16),
        metallic=0.04,
        roughness=0.02,
        coat=1.0,
        coat_roughness=0.01,
        ior=1.56,
        transmission=0.25,
        emission_color=(0.02, 0.45, 0.18),
        emission_strength=0.04,
    )
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.8
    noise.inputs["Detail"].default_value = 4.5
    noise.inputs["Distortion"].default_value = 1.7
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.34
    ramp.color_ramp.elements[0].color = (0.002, 0.13, 0.045, 1.0)
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = (0.025, 0.62, 0.24, 1.0)
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    mark_asset(m, "materials", "Luminous crystal emerald swirl glass marble", ["marble", "emerald", "glass"])
    mats["emerald_marble"] = m

    # 2. Build 12 Instrument PBR Materials
    # 01 Castanets (Rich Walnut)
    m, _, _ = new_principled_material("LibMat_01_Castanets", (0.42, 0.18, 0.08), metallic=0.0, roughness=0.30, coat=0.45, coat_roughness=0.08, emission_color=(1.0, 0.62, 0.22))
    add_wood_grain_nodes(m, (0.24, 0.09, 0.035), (0.56, 0.27, 0.11), scale=7.0)
    mark_asset(m, "materials", "Castanets lacquered walnut grain", ["castanets", "wood"])
    mats["inst_01"] = m

    # 02 Piano (Ivory + Ebony)
    m, _, _ = new_principled_material("LibMat_02_PianoIvory", (0.94, 0.92, 0.87), metallic=0.0, roughness=0.14, coat=0.80, coat_roughness=0.04, emission_color=(0.70, 0.92, 1.0))
    mark_asset(m, "materials", "Glossy concert piano ivory key", ["piano", "ivory"])
    mats["inst_02_white"] = m

    m, _, _ = new_principled_material("LibMat_02_PianoEbony", (0.025, 0.026, 0.032), metallic=0.02, roughness=0.14, coat=0.85, coat_roughness=0.04, emission_color=(0.35, 0.78, 1.0))
    mark_asset(m, "materials", "Glossy concert piano ebony key", ["piano", "ebony"])
    mats["inst_02_black"] = m

    # 03 Vibraphone (Brushed Ice Aluminum)
    m, _, _ = new_principled_material("LibMat_03_Vibraphone", (0.78, 0.85, 0.94), metallic=0.98, roughness=0.16, anisotropic=0.72, coat=0.0, emission_color=(0.50, 0.85, 1.0))
    add_brushed_metal_nodes(m, r_min=0.11, r_max=0.22)
    mark_asset(m, "materials", "Brushed ice-blue vibraphone tone bar", ["vibraphone", "aluminum"])
    mats["inst_03"] = m

    # 04 Tubular Bells (Polished Burnished Brass)
    m, _, _ = new_principled_material("LibMat_04_TubularBells", (0.94, 0.74, 0.28), metallic=1.0, roughness=0.12, anisotropic=0.68, coat=0.0, emission_color=(1.0, 0.82, 0.28))
    add_brushed_metal_nodes(m, r_min=0.08, r_max=0.18)
    mark_asset(m, "materials", "Polished gold-brass tubular chime metal", ["brass", "tubular_bells"])
    mats["inst_04"] = m

    # 05 Marimba (Deep Padauk Rosewood)
    m, _, _ = new_principled_material("LibMat_05_Marimba", (0.36, 0.10, 0.05), metallic=0.0, roughness=0.32, coat=0.40, coat_roughness=0.10, emission_color=(1.0, 0.45, 0.16))
    add_wood_grain_nodes(m, (0.18, 0.04, 0.02), (0.48, 0.14, 0.06), scale=8.0)
    mark_asset(m, "materials", "Deep crimson rosewood marimba tone bar", ["marimba", "rosewood"])
    mats["inst_05"] = m

    # 06 Kalimba (Tempered Spring Steel)
    m, _, _ = new_principled_material("LibMat_06_Kalimba", (0.70, 0.74, 0.82), metallic=0.98, roughness=0.14, anisotropic=0.72, coat=0.0, emission_color=(0.65, 0.90, 1.0))
    add_brushed_metal_nodes(m, r_min=0.10, r_max=0.20)
    mark_asset(m, "materials", "Spring steel kalimba tine alloy", ["kalimba", "steel"])
    mats["inst_06"] = m

    # 07 Flute (Concert Sterling Silver)
    m, _, _ = new_principled_material("LibMat_07_Flute", (0.92, 0.94, 0.97), metallic=1.0, roughness=0.055, coat=0.0, emission_color=(0.68, 0.96, 1.0))
    mark_asset(m, "materials", "Mirror concert silver flute finish", ["flute", "silver"])
    mats["inst_07"] = m

    # 08 Xylophone Rainbow Jewel-Tone Lacquers (6 tones)
    xylo_palette = [
        ("Red", (0.88, 0.14, 0.14)),
        ("Amber", (0.94, 0.44, 0.08)),
        ("Gold", (0.94, 0.76, 0.10)),
        ("Emerald", (0.12, 0.72, 0.34)),
        ("Azure", (0.12, 0.56, 0.90)),
        ("Violet", (0.50, 0.22, 0.88)),
    ]
    for cname, col in xylo_palette:
        m, _, _ = new_principled_material(f"LibMat_08_Xylo_{cname}", col, metallic=0.04, roughness=0.20, coat=0.70, coat_roughness=0.06, emission_color=col)
        mark_asset(m, "materials", f"Xylophone jewel-lacquer {cname}", ["xylophone", cname.lower()])
        mats[f"inst_08_{cname.lower()}"] = m

    # 09 Ocarina (Cobalt Ceramic Glaze)
    m, _, _ = new_principled_material("LibMat_09_Ocarina", (0.14, 0.38, 0.80), metallic=0.02, roughness=0.11, coat=0.92, coat_roughness=0.025, emission_color=(0.38, 0.78, 1.0))
    mark_asset(m, "materials", "Cobalt glossy ceramic ocarina glaze", ["ocarina", "ceramic"])
    mats["inst_09"] = m

    # 10 Glockenspiel (Mirror Steel Chrome)
    m, _, _ = new_principled_material("LibMat_10_Glockenspiel", (0.88, 0.91, 0.95), metallic=1.0, roughness=0.045, coat=0.0, emission_color=(0.82, 0.95, 1.0))
    mark_asset(m, "materials", "Mirror steel glockenspiel chime plate", ["glockenspiel", "chrome"])
    mats["inst_10"] = m

    # 11 Steel Drums (Hammered Gunmetal Steel)
    m, _, _ = new_principled_material("LibMat_11_SteelDrums", (0.56, 0.62, 0.70), metallic=0.98, roughness=0.16, anisotropic=0.40, coat=0.0, emission_color=(0.40, 0.88, 1.0))
    add_hammered_metal_nodes(m, r_min=0.08, r_max=0.22)
    mark_asset(m, "materials", "Hand-hammered steel pan Voronoi bump surface", ["steel_drums", "hammered"])
    mats["inst_11"] = m

    # 12 Guitar (Golden Spruce Acoustic)
    m, _, _ = new_principled_material("LibMat_12_Guitar", (0.78, 0.50, 0.18), metallic=0.0, roughness=0.26, coat=0.60, coat_roughness=0.06, emission_color=(1.0, 0.74, 0.26))
    add_wood_grain_nodes(m, (0.58, 0.34, 0.10), (0.88, 0.60, 0.24), scale=9.0)
    mark_asset(m, "materials", "Golden Sitka spruce acoustic guitar soundboard", ["guitar", "spruce"])
    mats["inst_12"] = m

    # 3. Build 12 Cataloged Instrument Assembly Assets (Unit Width = 1.0u, Core + Safe Rear Y >= +0.13 SubAssembly)
    # Convention: local origin at (0,0,0), core box = (1.0, 0.24, 0.072). Rear sub-parts live at Y >= +0.125 so X can scale freely
    # and the Y in [-0.12, +0.12] / +Z trajectory space stays 100% collision-free!
    inst_defs = [
        (1, "Castanets", "percussion", mats["inst_01"], "Lacquered walnut castanet clapper bar with rear brass hinge"),
        (2, "Piano", "keyboards", mats["inst_02_white"], "Concert piano strike key with rear ebony damper bed"),
        (3, "Vibraphone", "keyboards", mats["inst_03"], "Brushed aluminum vibraphone bar with twin bronze resonators"),
        (4, "TubularBells", "keyboards", mats["inst_04"], "Burnished brass strike block with suspended chime tubes"),
        (5, "Marimba", "percussion", mats["inst_05"], "Rosewood marimba tone bar with dark bronze resonators"),
        (6, "Kalimba", "melodic", mats["inst_06"], "Spring-steel tine bar mounted on walnut soundbox"),
        (7, "Flute", "melodic", mats["inst_07"], "Concert silver strike bar with rear keyed flute tube"),
        (8, "Xylophone", "percussion", mats["inst_08_gold"], "Jewel-lacquer xylophone tone bar with twin resonator pipes"),
        (9, "Ocarina", "melodic", mats["inst_09"], "Cobalt glazed ceramic bar with porcelain tone chamber"),
        (10, "Glockenspiel", "keyboards", mats["inst_10"], "Mirror chrome glockenspiel plate with dark anvil cradle"),
        (11, "SteelDrums", "percussion", mats["inst_11"], "Hammered steel pan strike surface with crescent collar"),
        (12, "Guitar", "melodic", mats["inst_12"], "Golden spruce soundboard block with ebony bridge & rosette"),
    ]

    for idx, name, cat, core_mat, desc in inst_defs:
        col_name = f"Lib_{idx:02d}_{name}"
        icol = bpy.data.collections.new(col_name)
        root_col.children.link(icol)

        # Core exact collision-box strike bar (unit length 1.0)
        core = create_box_obj(
            f"LibCore_{idx:02d}_{name}",
            (1.0, 0.24, 0.072),
            offset=(0.0, 0.0, 0.0),
            bevel=0.006,
            mat=core_mat,
            col=icol,
        )

        # Instrument-specific protruding silhouettes (strictly outside marble top-corridor: either Y >= +0.125 or Z <= -0.042)
        if idx == 1:  # Castanets: twin clapper lower walnut shell underneath + gleaming brass spring hinge & side ears
            create_box_obj(f"LibSub_{idx:02d}_LowerShell", (0.29, 0.22, 0.048), offset=(0.0, 0.025, -0.068), bevel=0.018, mat=mats["wood_slat"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_HingePin", 0.022, 0.34, center=(0.0, 0.145, -0.018), rot_euler=(0.0, math.pi / 2, 0.0), mat=mats["brass_trim"], col=icol)
            create_box_obj(f"LibSub_{idx:02d}_BrassClasp", (0.14, 0.09, 0.038), offset=(0.0, 0.145, -0.052), bevel=0.005, mat=mats["brass_trim"], col=icol)
        elif idx == 2:  # Piano: raised twin Ebony sharp keys behind/above ivory key + warm walnut keybed & brass damper arm below
            create_box_obj(f"LibSub_{idx:02d}_SharpL", (0.064, 0.115, 0.058), offset=(-0.095, 0.152, 0.046), bevel=0.005, mat=mats["inst_02_black"], col=icol)
            create_box_obj(f"LibSub_{idx:02d}_SharpR", (0.064, 0.115, 0.058), offset=(0.095, 0.152, 0.046), bevel=0.005, mat=mats["inst_02_black"], col=icol)
            create_box_obj(f"LibSub_{idx:02d}_KeyBed", (0.34, 0.18, 0.046), offset=(0.0, 0.095, -0.064), bevel=0.006, mat=mats["inst_01"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_DamperRod", 0.012, 0.30, center=(0.0, 0.14, -0.030), rot_euler=(0.0, math.pi / 2, 0.0), mat=mats["brass_trim"], col=icol, segs=16)
        elif idx == 3:  # Vibraphone: brushed bronze cradle + twin hanging long brass resonator tubes underneath
            create_box_obj(f"LibSub_{idx:02d}_Rail", (0.30, 0.14, 0.032), offset=(0.0, 0.08, -0.056), bevel=0.004, mat=mats["hardware_dark"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_ResL", 0.036, 0.30, center=(-0.070, 0.095, -0.195), mat=mats["brass_trim"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_ResR", 0.036, 0.22, center=(0.070, 0.095, -0.155), mat=mats["resonator_bronze"], col=icol)
        elif idx == 4:  # Tubular Bells: brass suspension yoke + 3 long staggered vertical brass chime pipes hanging prominently below
            create_box_obj(f"LibSub_{idx:02d}_Rack", (0.30, 0.10, 0.032), offset=(0.0, 0.145, 0.020), bevel=0.005, mat=mats["hardware_dark"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_Chime1", 0.028, 0.44, center=(-0.085, 0.10, -0.255), mat=mats["inst_04"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_Chime2", 0.028, 0.36, center=(0.0, 0.10, -0.215), mat=mats["inst_04"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_Chime3", 0.028, 0.28, center=(0.085, 0.10, -0.175), mat=mats["inst_04"], col=icol)
        elif idx == 5:  # Marimba: warm mahogany cradle + twin flared bronze resonator pipes underneath
            create_box_obj(f"LibSub_{idx:02d}_Cradle", (0.30, 0.14, 0.038), offset=(0.0, 0.08, -0.060), bevel=0.006, mat=mats["wood_slat"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_PipeL", 0.038, 0.28, center=(-0.065, 0.095, -0.185), mat=mats["resonator_bronze"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_PipeR", 0.038, 0.21, center=(0.065, 0.095, -0.150), mat=mats["resonator_bronze"], col=icol)
        elif idx == 6:  # Kalimba: walnut soundbox underneath + 5 fanned silver-steel tines protruding forward/up behind Y
            create_box_obj(f"LibSub_{idx:02d}_Soundbox", (0.28, 0.18, 0.072), offset=(0.0, 0.085, -0.075), bevel=0.010, mat=mats["inst_01"], col=icol)
            for tk, tx in enumerate((-0.09, -0.045, 0.0, 0.045, 0.09)):
                create_box_obj(f"LibSub_{idx:02d}_Tine_{tk}", (0.024, 0.095, 0.012), offset=(tx, 0.168, 0.028), bevel=0.002, mat=mats["inst_06"], col=icol)
        elif idx == 7:  # Flute: long silver keyed flute body extending visibly wide behind Y + raised brass key cups
            create_cyl_obj(f"LibSub_{idx:02d}_Tube", 0.028, 0.56, center=(0.0, 0.165, 0.010), rot_euler=(0.0, math.pi / 2, 0.0), mat=mats["inst_07"], col=icol)
            for kp, kx in enumerate((-0.14, -0.05, 0.05, 0.14)):
                create_cyl_obj(f"LibSub_{idx:02d}_Key_{kp}", 0.018, 0.022, center=(kx, 0.165, 0.042), mat=mats["brass_trim"], col=icol, segs=18)
        elif idx == 8:  # Xylophone: warm walnut bridge + twin staggered brass-bronze resonator tubes below
            create_box_obj(f"LibSub_{idx:02d}_Bridge", (0.30, 0.13, 0.030), offset=(0.0, 0.08, -0.055), bevel=0.004, mat=mats["hardware_dark"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_Res1", 0.032, 0.24, center=(-0.06, 0.095, -0.165), mat=mats["resonator_bronze"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_Res2", 0.032, 0.18, center=(0.06, 0.095, -0.135), mat=mats["brass_trim"], col=icol)
        elif idx == 9:  # Ocarina: sculpted ivory porcelain tone chamber below/behind + twin brass tone ports
            create_box_obj(f"LibSub_{idx:02d}_Chamber", (0.28, 0.16, 0.088), offset=(0.0, 0.095, -0.078), bevel=0.022, mat=mats["porcelain_cream"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_PortL", 0.022, 0.024, center=(-0.060, 0.165, 0.034), mat=mats["brass_trim"], col=icol, segs=18)
            create_cyl_obj(f"LibSub_{idx:02d}_PortR", 0.022, 0.024, center=(0.060, 0.165, 0.034), mat=mats["brass_trim"], col=icol, segs=18)
        elif idx == 10:  # Glockenspiel: dark bronze anvil frame underneath + brass mounting posts
            create_box_obj(f"LibSub_{idx:02d}_Anvil", (0.30, 0.15, 0.048), offset=(0.0, 0.085, -0.064), bevel=0.007, mat=mats["hardware_dark"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_StudL", 0.017, 0.030, center=(-0.09, 0.155, 0.024), mat=mats["brass_trim"], col=icol, segs=16)
            create_cyl_obj(f"LibSub_{idx:02d}_StudR", 0.017, 0.030, center=(0.09, 0.155, 0.024), mat=mats["brass_trim"], col=icol, segs=16)
        elif idx == 11:  # Steel Drums: broad tilted steel pan skirt underneath + central brass tuning boss
            create_cyl_obj(f"LibSub_{idx:02d}_PanRim", 0.16, 0.09, center=(0.0, 0.11, -0.085), rot_euler=(math.radians(12), 0.0, 0.0), mat=mats["hardware_dark"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_Boss", 0.028, 0.032, center=(0.0, 0.16, 0.022), mat=mats["brass_trim"], col=icol, segs=18)
        else:  # Guitar: wide warm spruce soundboard apron below + circular brass rosette + ebony bridge + 4 taut brass strings
            create_box_obj(f"LibSub_{idx:02d}_BodyApron", (0.36, 0.065, 0.22), offset=(0.0, 0.125, -0.145), bevel=0.012, mat=mats["inst_12"], col=icol)
            create_cyl_obj(f"LibSub_{idx:02d}_Rosette", 0.056, 0.018, center=(0.0, 0.092, -0.135), rot_euler=(math.pi / 2, 0.0, 0.0), mat=mats["brass_trim"], col=icol, segs=28)
            create_cyl_obj(f"LibSub_{idx:02d}_Soundhole", 0.040, 0.022, center=(0.0, 0.090, -0.135), rot_euler=(math.pi / 2, 0.0, 0.0), mat=mats["inst_02_black"], col=icol, segs=24)
            create_box_obj(f"LibSub_{idx:02d}_Bridge", (0.22, 0.030, 0.028), offset=(0.0, 0.092, -0.225), bevel=0.004, mat=mats["inst_02_black"], col=icol)
            for sk, sx in enumerate((-0.054, -0.018, 0.018, 0.054)):
                create_cyl_obj(f"LibSub_{idx:02d}_Str_{sk}", 0.004, 0.24, center=(sx, 0.082, -0.125), mat=mats["brass_trim"], col=icol, segs=10)

        mark_asset(icol, cat, desc, ["instrument", name.lower()])

    # Keep all 21 materials clearly linked via fake_user so importing by name never loses any palette variant
    for m in mats.values():
        m.use_fake_user = True

    os.makedirs(ASSETS_DIR, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=LIB_BLEND_PATH)
    print(f"Saved Blender Instrument Asset Library: {LIB_BLEND_PATH} ({len(inst_defs)} instrument collections, {len(mats)} PBR materials)")


if __name__ == "__main__":
    build_library()
