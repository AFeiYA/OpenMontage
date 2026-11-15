#!/usr/bin/env python3
"""Build the v2.0 64-Instrument Asset Library (.blend) for Luca Marble Music.

Each of the 64 General MIDI instruments gets its own bespoke, individually
recognizable 3D silhouette (no shared generic placeholders between Sitar,
Koto, Shamisen, Bassoon, Trombone, French Horn, Bagpipe, etc.), built in
canonical local space:
  - Hit/collision bar: X in [-0.50, +0.50], Y in [-0.12, +0.12], Z in [-0.036, +0.036]
  - Bespoke 3D instrument sculpture: Y >= +0.15 behind the hit bar toward the stage wall,
    facing -Y toward the 3/4 oblique camera, mounted like museum wall exhibits.

Outputs:
  - projects/jingle-bells-marble-music/assets/instrument_library_v2.blend
  - projects/jingle-bells-marble-music/assets/instrument_manifest_v2.json
"""

from __future__ import annotations

import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Euler, Vector

OUT_BLEND = "projects/jingle-bells-marble-music/assets/instrument_library_v2.blend"
OUT_MANIFEST = "projects/jingle-bells-marble-music/assets/instrument_manifest_v2.json"
OUT_CATS = "projects/jingle-bells-marble-music/assets/blender_assets.cats.txt"

CAT_UUIDS = {
    "percussion": "a1b2c3d4-1111-4000-8000-000000000001",
    "keyboards": "a1b2c3d4-1111-4000-8000-000000000002",
    "melodic": "a1b2c3d4-1111-4000-8000-000000000003",
    "materials": "a1b2c3d4-1111-4000-8000-000000000004",
    "hardware": "a1b2c3d4-1111-4000-8000-000000000005",
    "strings": "a1b2c3d4-1111-4000-8000-000000000006",
    "brass": "a1b2c3d4-1111-4000-8000-000000000007",
    "woodwinds": "a1b2c3d4-1111-4000-8000-000000000008",
}

FAMILY_CAT_KEY = {
    1: "keyboards",
    2: "melodic",
    3: "strings",
    4: "brass",
    5: "woodwinds",
    6: "percussion",
}

# Animation family & signature emissive hit-glow color per GM program
ANIM_PROFILE_BY_GM = {
    # Family 1: Keyboards & Mallets
    0: ("keyboard_hinge", (0.72, 0.94, 1.00)),
    2: ("keyboard_hinge", (0.62, 0.88, 1.00)),
    4: ("keyboard_hinge", (0.48, 0.86, 1.00)),
    6: ("keyboard_pluck", (1.00, 0.78, 0.34)),
    8: ("celesta_bars", (0.76, 0.96, 1.00)),
    9: ("mallet_bars", (0.82, 0.96, 1.00)),
    10: ("music_box", (1.00, 0.84, 0.42)),
    11: ("mallet_bars", (0.48, 0.88, 1.00)),
    12: ("mallet_bars", (1.00, 0.50, 0.22)),
    13: ("mallet_bars", (1.00, 0.76, 0.24)),
    14: ("tubular_pendulum", (1.00, 0.84, 0.32)),
    15: ("mallet_strings", (0.98, 0.72, 0.30)),
    # Family 2: Organs, Accordion, Guitars & Harp
    16: ("organ_drawbars", (1.00, 0.62, 0.24)),
    19: ("organ_pipes", (0.96, 0.80, 0.42)),
    21: ("bellows_accordion", (0.95, 0.38, 0.28)),
    22: ("reed_plates", (0.78, 0.94, 1.00)),
    24: ("plucked_strings", (1.00, 0.74, 0.28)),
    25: ("plucked_strings", (1.00, 0.66, 0.24)),
    27: ("electric_guitar", (0.42, 0.86, 1.00)),
    30: ("electric_guitar", (1.00, 0.26, 0.18)),
    32: ("plucked_strings", (0.96, 0.58, 0.22)),
    35: ("electric_guitar", (0.85, 0.46, 0.20)),
    45: ("plucked_strings", (0.98, 0.72, 0.34)),
    46: ("harp_strings", (1.00, 0.84, 0.40)),
    # Family 3: Bowed Strings & Choir
    40: ("bowed_strings", (1.00, 0.62, 0.24)),
    41: ("bowed_strings", (0.96, 0.52, 0.22)),
    42: ("bowed_strings", (0.92, 0.44, 0.18)),
    43: ("bowed_strings", (0.86, 0.36, 0.16)),
    48: ("bowed_strings", (1.00, 0.74, 0.32)),
    52: ("choir_pipes", (0.64, 0.90, 1.00)),
    53: ("choir_pipes", (0.76, 0.82, 1.00)),
    55: ("gong_pendulum", (1.00, 0.68, 0.22)),
    # Family 4: Brass & Saxophones
    56: ("brass_valves", (1.00, 0.82, 0.28)),
    57: ("trombone_slide", (1.00, 0.76, 0.26)),
    58: ("brass_valves", (0.96, 0.70, 0.24)),
    59: ("brass_valves", (1.00, 0.54, 0.24)),
    60: ("brass_valves", (1.00, 0.80, 0.32)),
    61: ("brass_fan", (1.00, 0.82, 0.34)),
    64: ("woodwind_keys", (1.00, 0.80, 0.34)),
    65: ("woodwind_keys", (1.00, 0.78, 0.30)),
    66: ("woodwind_keys", (1.00, 0.74, 0.28)),
    # Family 5: Woodwinds & Flutes
    68: ("woodwind_keys", (0.82, 0.92, 1.00)),
    69: ("woodwind_keys", (0.92, 0.76, 0.48)),
    70: ("woodwind_keys", (0.88, 0.54, 0.28)),
    71: ("woodwind_keys", (0.78, 0.90, 1.00)),
    72: ("flute_air", (0.78, 0.98, 1.00)),
    73: ("flute_air", (0.64, 0.96, 1.00)),
    74: ("flute_air", (0.96, 0.72, 0.36)),
    75: ("pan_flute", (0.98, 0.78, 0.34)),
    77: ("flute_air", (0.94, 0.76, 0.38)),
    78: ("flute_air", (0.82, 0.96, 1.00)),
    79: ("ocarina_pulse", (0.35, 0.80, 1.00)),
    # Family 6: World Ethnic & Synth
    104: ("plucked_strings", (1.00, 0.68, 0.26)),
    105: ("plucked_strings", (1.00, 0.84, 0.42)),
    106: ("plucked_strings", (0.98, 0.86, 0.60)),
    107: ("harp_strings", (0.98, 0.74, 0.34)),
    108: ("kalimba_tines", (0.60, 0.94, 1.00)),
    109: ("bagpipe_bag", (0.38, 0.88, 0.56)),
    111: ("woodwind_keys", (1.00, 0.72, 0.28)),
    114: ("steelpan_petals", (0.38, 0.90, 1.00)),
    115: ("woodblock_clapper", (1.00, 0.64, 0.24)),
    80: ("synth_euclid", (0.15, 0.96, 0.62)),
    81: ("synth_saw", (1.00, 0.65, 0.16)),
    88: ("synth_pad", (0.20, 0.86, 1.00)),
}


def write_asset_catalogs() -> None:
    os.makedirs(os.path.dirname(os.path.abspath(OUT_CATS)), exist_ok=True)
    lines = [
        "# This is an Asset Catalog Definition file for Blender.",
        "#",
        "# Empty lines and lines starting with `#` will be ignored.",
        "# The first non-ignored line should be the version indicator.",
        "VERSION 1",
        "",
        f"{CAT_UUIDS['percussion']}:Instruments/Percussion & World:Percussion",
        f"{CAT_UUIDS['keyboards']}:Instruments/Keyboards & Mallets:Keyboards-Mallets",
        f"{CAT_UUIDS['melodic']}:Instruments/Organs, Guitars & Harp:Melodic-Pluck",
        f"{CAT_UUIDS['materials']}:Materials/Studio PBR:Studio-PBR",
        f"{CAT_UUIDS['hardware']}:Stage & Hardware:Stage-Hardware",
        f"{CAT_UUIDS['strings']}:Instruments/Strings & Choir:Strings-Choir",
        f"{CAT_UUIDS['brass']}:Instruments/Brass & Saxophones:Brass-Saxophones",
        f"{CAT_UUIDS['woodwinds']}:Instruments/Woodwinds & Flutes:Woodwinds-Flutes",
        "",
    ]
    with open(OUT_CATS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _mark_asset(id_block, cat_key: str, description: str, tags: list[str]) -> None:
    id_block.asset_mark()
    ad = id_block.asset_data
    ad.catalog_id = CAT_UUIDS[cat_key]
    ad.author = "OpenMontage Studio"
    ad.description = description
    for t in tags:
        ad.tags.new(t, skip_if_exists=True)


# ---------------------------------------------------------------------------
# Material Palette
# ---------------------------------------------------------------------------

def _make_pbr(
    name: str,
    color,
    metallic: float = 0.0,
    roughness: float = 0.35,
    emission=None,
    emission_strength: float = 0.0,
    coat: float = 0.0,
    coat_roughness: float = 0.08,
    ior: float = 1.45,
    anisotropic: float = 0.0,
    transmission: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = ior
    if "Anisotropic" in bsdf.inputs:
        bsdf.inputs["Anisotropic"].default_value = anisotropic
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
    if "Coat Roughness" in bsdf.inputs:
        bsdf.inputs["Coat Roughness"].default_value = coat_roughness
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    if emission is not None and "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (*emission[:3], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    mat.use_fake_user = True
    return mat


def _add_wood_grain_nodes(mat, c_dark, c_light, scale=2.6, bump_dist=0.0008, vec_scale=(6.0, 6.0, 0.30)):
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = vec_scale
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
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


def _add_swirl_marble_nodes(mat, c_dark, c_light):
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.8
    noise.inputs["Detail"].default_value = 4.5
    noise.inputs["Distortion"].default_value = 1.7
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.34
    ramp.color_ramp.elements[0].color = (*c_dark, 1.0)
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = (*c_light, 1.0)
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])


def build_materials() -> dict[str, bpy.types.Material]:
    mats = {
        "ivory": _make_pbr("LibMat_Ivory", (0.93, 0.90, 0.84), 0.0, 0.22, coat=0.65, coat_roughness=0.05),
        "ebony": _make_pbr("LibMat_Ebony", (0.04, 0.04, 0.045), 0.0, 0.28, coat=0.70, coat_roughness=0.05),
        "walnut": _make_pbr("LibMat_Walnut", (0.22, 0.11, 0.055), 0.0, 0.34, coat=0.35),
        "rosewood": _make_pbr("LibMat_Rosewood", (0.28, 0.08, 0.05), 0.0, 0.28, coat=0.40),
        "mahogany": _make_pbr("LibMat_Mahogany", (0.34, 0.10, 0.045), 0.0, 0.26, coat=0.45),
        "spruce": _make_pbr("LibMat_Spruce", (0.76, 0.56, 0.32), 0.0, 0.35, coat=0.45),
        "amber_wood": _make_pbr("LibMat_AmberWood", (0.64, 0.32, 0.09), 0.0, 0.22, coat=0.60),
        "sunburst": _make_pbr("LibMat_Sunburst", (0.48, 0.14, 0.04), 0.0, 0.20, coat=0.65),
        "bamboo": _make_pbr("LibMat_Bamboo", (0.72, 0.58, 0.28), 0.0, 0.38),
        "brass": _make_pbr("LibMat_Brass", (0.86, 0.67, 0.24), 0.95, 0.18, anisotropic=0.65),
        "antique_brass": _make_pbr("LibMat_AntiqueBrass", (0.58, 0.44, 0.18), 0.90, 0.28, anisotropic=0.55),
        "bronze": _make_pbr("LibMat_Bronze", (0.68, 0.44, 0.22), 0.92, 0.22, anisotropic=0.55),
        "copper": _make_pbr("LibMat_Copper", (0.82, 0.42, 0.24), 0.94, 0.20, anisotropic=0.60),
        "chrome": _make_pbr("LibMat_Chrome", (0.88, 0.90, 0.93), 0.98, 0.12, anisotropic=0.50),
        "brushed_alu": _make_pbr("LibMat_BrushedAlu", (0.72, 0.75, 0.79), 0.92, 0.26, anisotropic=0.70),
        "steel_dark": _make_pbr("LibMat_SteelDark", (0.14, 0.15, 0.17), 0.85, 0.32, anisotropic=0.45),
        "felt_red": _make_pbr("LibMat_FeltRed", (0.55, 0.05, 0.06), 0.0, 0.85),
        "crimson": _make_pbr("LibMat_Crimson", (0.56, 0.06, 0.08), 0.05, 0.25, coat=0.55),
        "cobalt": _make_pbr("LibMat_Cobalt", (0.08, 0.24, 0.58), 0.05, 0.18, coat=0.80),
        "celeste_blue": _make_pbr("LibMat_CelesteBlue", (0.18, 0.36, 0.54), 0.12, 0.26, coat=0.50),
        "pearl": _make_pbr("LibMat_Pearl", (0.85, 0.86, 0.88), 0.10, 0.16, coat=0.75),
        "olympic_white": _make_pbr("LibMat_OlympicWhite", (0.91, 0.89, 0.84), 0.05, 0.20, coat=0.65),
        "tortoiseshell": _make_pbr("LibMat_Tortoiseshell", (0.32, 0.08, 0.03), 0.10, 0.22, coat=0.70),
        "xylo_gold": _make_pbr("LibMat_XyloGold", (0.74, 0.48, 0.16), 0.05, 0.26, coat=0.55),
        "synth_dark": _make_pbr("LibMat_SynthDark", (0.06, 0.07, 0.09), 0.25, 0.30),
        "led_emerald": _make_pbr(
            "LibMat_LedEmerald", (0.08, 0.92, 0.55), 0.0, 0.15,
            emission=(0.10, 0.95, 0.58), emission_strength=3.2,
        ),
        "led_yellow": _make_pbr(
            "LibMat_LedYellow", (0.95, 0.88, 0.12), 0.0, 0.15,
            emission=(0.98, 0.90, 0.15), emission_strength=3.2,
        ),
        "led_red": _make_pbr(
            "LibMat_LedRed", (0.95, 0.12, 0.12), 0.0, 0.15,
            emission=(0.98, 0.15, 0.15), emission_strength=3.2,
        ),
        "led_amber": _make_pbr(
            "LibMat_LedAmber", (0.98, 0.62, 0.12), 0.0, 0.15,
            emission=(0.98, 0.65, 0.15), emission_strength=3.2,
        ),
        "led_cyan": _make_pbr(
            "LibMat_LedCyan", (0.15, 0.82, 0.98), 0.0, 0.15,
            emission=(0.18, 0.85, 1.00), emission_strength=3.2,
        ),
    }

    # Shared Stage & Hardware PBR Materials (required by both vertical & 16:9 theatre renderers)
    chrome_rail = _make_pbr("LibMat_ChromeRail", (0.64, 0.65, 0.68), metallic=0.96, roughness=0.20, anisotropic=0.68)
    _mark_asset(chrome_rail, "hardware", "Satin brushed nickel-steel guide rail", ["satin", "rail", "metal"])
    mats["chrome_rail"] = chrome_rail

    hardware_dark = _make_pbr("LibMat_HardwareDark", (0.095, 0.075, 0.056), metallic=0.88, roughness=0.34, anisotropic=0.50)
    _mark_asset(hardware_dark, "hardware", "Dark aged bronze for brackets, pins & wall flanges", ["bronze", "hardware"])
    mats["hardware_dark"] = hardware_dark

    wall_felt = _make_pbr("LibMat_WallFelt", (0.017, 0.016, 0.016), metallic=0.0, roughness=0.95)
    _mark_asset(wall_felt, "hardware", "Low-contrast matte warm-charcoal acoustic backdrop", ["wall", "matte", "dark"])
    mats["wall_felt"] = wall_felt

    wood_slat = _make_pbr("LibMat_WoodSlatDark", (0.020, 0.014, 0.010), metallic=0.0, roughness=0.72, coat=0.02, coat_roughness=0.35)
    _add_wood_grain_nodes(wood_slat, (0.015, 0.010, 0.007), (0.024, 0.017, 0.012))
    _mark_asset(wood_slat, "hardware", "Subdued smoked walnut architectural frame wood", ["walnut", "wood", "slat"])
    mats["wood_slat"] = wood_slat

    brass_trim = _make_pbr("LibMat_WallBrassTrim", (0.82, 0.60, 0.22), metallic=0.98, roughness=0.18, anisotropic=0.60)
    _mark_asset(brass_trim, "hardware", "Warm brushed music-box brass trim & brackets", ["brass", "trim"])
    mats["brass_trim"] = brass_trim

    wall_reveal = _make_pbr("LibMat_WallRevealDark", (0.14, 0.10, 0.055), metallic=0.72, roughness=0.36)
    _mark_asset(wall_reveal, "hardware", "Muted aged brass phrase divider reveal", ["bronze", "wall", "reveal"])
    mats["wall_reveal"] = wall_reveal

    emerald_marble = _make_pbr(
        "LibMat_EmeraldMarble", (0.01, 0.42, 0.16), metallic=0.04, roughness=0.02,
        coat=1.0, coat_roughness=0.01, ior=1.56, transmission=0.25,
        emission=(0.02, 0.45, 0.18), emission_strength=0.04,
    )
    _add_swirl_marble_nodes(emerald_marble, (0.002, 0.13, 0.045), (0.025, 0.62, 0.24))
    _mark_asset(emerald_marble, "materials", "Luminous crystal emerald swirl glass marble", ["marble", "emerald", "glass"])
    mats["emerald_marble"] = emerald_marble

    gold_marble = _make_pbr(
        "LibMat_GoldMarble", (0.72, 0.42, 0.04), metallic=0.10, roughness=0.025,
        coat=1.0, coat_roughness=0.01, ior=1.58, transmission=0.18,
        emission=(0.95, 0.56, 0.12), emission_strength=0.08,
    )
    _add_swirl_marble_nodes(gold_marble, (0.34, 0.12, 0.01), (0.98, 0.68, 0.16))
    _mark_asset(gold_marble, "materials", "Molten amber-gold swirl concert marble", ["marble", "gold", "glass"])
    mats["gold_marble"] = gold_marble

    sapphire_marble = _make_pbr(
        "LibMat_SapphireMarble", (0.04, 0.24, 0.68), metallic=0.06, roughness=0.02,
        coat=1.0, coat_roughness=0.01, ior=1.56, transmission=0.22,
        emission=(0.14, 0.52, 1.00), emission_strength=0.06,
    )
    _add_swirl_marble_nodes(sapphire_marble, (0.01, 0.06, 0.28), (0.14, 0.58, 0.98))
    _mark_asset(sapphire_marble, "materials", "Sapphire crystal swirl bass/synth marble", ["marble", "sapphire", "glass"])
    mats["sapphire_marble"] = sapphire_marble

    for k, m in mats.items():
        if k not in {"chrome_rail", "hardware_dark", "wall_felt", "wood_slat", "brass_trim", "wall_reveal", "emerald_marble", "gold_marble", "sapphire_marble"}:
            _mark_asset(m, "materials", f"Instrument studio PBR material ({m.name})", ["instrument", "pbr", k])

    return mats


# ---------------------------------------------------------------------------
# Mesh Primitives
# ---------------------------------------------------------------------------

def _finish_mesh(obj: bpy.types.Object, angle_deg: float = 34.0) -> None:
    if obj.type != "MESH":
        return
    mesh = obj.data
    if hasattr(mesh, "polygons") and len(mesh.polygons) > 0:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()
        for p in mesh.polygons:
            p.use_smooth = True
    if not any(m.type == "EDGE_SPLIT" for m in obj.modifiers):
        es = obj.modifiers.new(name="EdgeSplit", type="EDGE_SPLIT")
        es.split_angle = math.radians(angle_deg)
        es.use_edge_angle = True


def _link(col, obj: bpy.types.Object) -> bpy.types.Object:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def add_box(
    col,
    name: str,
    size: tuple[float, float, float],
    loc: tuple[float, float, float],
    mat: bpy.types.Material,
    bevel: float = 0.004,
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel > 0.0:
        bv = obj.modifiers.new(name="Bevel", type="BEVEL")
        bv.width = bevel
        bv.segments = 2
        bv.limit_method = "ANGLE"
    obj.data.materials.append(mat)
    _finish_mesh(obj, 38.0)
    return _link(col, obj)


def add_cyl(
    col,
    name: str,
    r1: float,
    r2: float,
    depth: float,
    loc: tuple[float, float, float],
    mat: bpy.types.Material,
    segs: int = 24,
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(
        vertices=segs, radius1=r1, radius2=r2, depth=depth, location=loc, rotation=rot
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    _finish_mesh(obj, 42.0)
    return _link(col, obj)


def add_sphere(
    col,
    name: str,
    radius: float,
    loc: tuple[float, float, float],
    mat: bpy.types.Material,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24, ring_count=16, radius=radius, location=loc, rotation=rot
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    _finish_mesh(obj, 60.0)
    return _link(col, obj)


def add_torus_arc(
    col,
    name: str,
    major_r: float,
    minor_r: float,
    loc: tuple[float, float, float],
    mat: bpy.types.Material,
    arc_deg: float = 360.0,
    rot: tuple[float, float, float] = (math.pi / 2, 0.0, 0.0),
    segs: int = 24,
) -> bpy.types.Object:
    curve = bpy.data.curves.new(name=f"{name}_Curve", type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = minor_r
    curve.bevel_resolution = 4
    curve.resolution_u = 8

    sp = curve.splines.new("POLY")
    steps = max(8, int(segs * (arc_deg / 360.0)))
    sp.points.add(steps - 1 if arc_deg >= 359.0 else steps)
    for i in range(len(sp.points)):
        ang = math.radians(arc_deg * (i / max(1, steps)))
        sp.points[i].co = (major_r * math.cos(ang), major_r * math.sin(ang), 0.0, 1.0)
    if arc_deg >= 359.0:
        sp.use_cyclic_u = True

    obj = bpy.data.objects.new(name, curve)
    obj.location = loc
    obj.rotation_euler = rot
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def add_tube_pts(
    col,
    name: str,
    pts: list[tuple[float, float, float]],
    radius: float,
    mat: bpy.types.Material,
    cyclic: bool = False,
) -> bpy.types.Object:
    curve = bpy.data.curves.new(name=f"{name}_Curve", type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    curve.resolution_u = 8
    sp = curve.splines.new("NURBS")
    sp.points.add(len(pts) - 1)
    for i, pt in enumerate(pts):
        sp.points[i].co = (pt[0], pt[1], pt[2], 1.0)
    sp.use_endpoint_u = not cyclic
    sp.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def add_extruded_profile(
    col,
    name: str,
    pts_2d: list[tuple[float, float]],
    depth: float,
    loc: tuple[float, float, float],
    mat: bpy.types.Material,
    bevel: float = 0.004,
    smooth: bool = True,
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    curve = bpy.data.curves.new(name=f"{name}_Curve", type="CURVE")
    curve.dimensions = "2D"
    curve.fill_mode = "BOTH"
    curve.extrude = depth * 0.5
    curve.bevel_depth = bevel
    curve.bevel_resolution = 3
    curve.resolution_u = 16
    sp = curve.splines.new("NURBS" if smooth else "POLY")
    sp.points.add(len(pts_2d) - 1)
    for i, (x, z) in enumerate(pts_2d):
        sp.points[i].co = (x, z, 0.0, 1.0)
    sp.use_cyclic_u = True
    if smooth:
        sp.order_u = min(4, len(pts_2d))
    obj = bpy.data.objects.new(name, curve)
    obj.location = loc
    obj.rotation_euler = (math.pi / 2.0 + rot[0], rot[1], rot[2])
    obj.data.materials.append(mat)
    col.objects.link(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target="MESH")
    obj = bpy.context.active_object
    _finish_mesh(obj, angle_deg=34.0)
    return obj


def add_extruded_horiz_profile(
    col,
    name: str,
    pts_xy: list[tuple[float, float]],
    height_z: float,
    loc: tuple[float, float, float],
    mat: bpy.types.Material,
    bevel: float = 0.004,
    smooth: bool = True,
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    """Extrude a 2D (x, y) contour vertically along local Z (for piano lids, harpsichords, etc.)."""
    return add_extruded_profile(
        col, name, pts_xy, height_z, loc, mat, bevel=bevel, smooth=smooth,
        rot=(-math.pi / 2.0 + rot[0], rot[1], rot[2]),
    )


def add_lathe(
    col,
    name: str,
    profile_rz: list[tuple[float, float]],
    loc: tuple[float, float, float],
    mat: bpy.types.Material,
    segs: int = 32,
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cap_bottom: bool = False,
    cap_top: bool = False,
) -> bpy.types.Object:
    """Revolve a 2D (radius, z) profile into a smooth watertight mesh."""
    bm = bmesh.new()
    rings = []
    for r, z in profile_rz:
        rr = max(1e-4, float(r))
        ring = []
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs
            ring.append(bm.verts.new((rr * math.cos(ang), rr * math.sin(ang), float(z))))
        rings.append(ring)
    for i in range(len(rings) - 1):
        ra, rb = rings[i], rings[i + 1]
        for s in range(segs):
            sn = (s + 1) % segs
            bm.faces.new((ra[s], ra[sn], rb[sn], rb[s]))
    if cap_bottom and rings:
        bm.faces.new(list(reversed(rings[0])))
    if cap_top and rings:
        bm.faces.new(rings[-1])
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = loc
    obj.rotation_euler = rot
    obj.data.materials.append(mat)
    col.objects.link(obj)
    _finish_mesh(obj, angle_deg=48.0)
    return obj


def _catmull_rom_4d(p0: Vector, p1: Vector, p2: Vector, p3: Vector, t: float) -> Vector:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def add_var_tube(
    col,
    name: str,
    pts_xyzr: list[tuple[float, float, float, float]],
    mat: bpy.types.Material,
    segs: int = 16,
    substeps: int = 5,
    cap_ends: bool = True,
) -> bpy.types.Object:
    """Sweep a smooth variable-radius circular tube along a 3D Catmull-Rom path."""
    from mathutils import Vector as _V

    ctrl = [_V((x, y, z, r)) for x, y, z, r in pts_xyzr]
    if len(ctrl) < 2:
        raise ValueError("add_var_tube needs at least 2 points")
    ext = [ctrl[0] + (ctrl[0] - ctrl[1])] + ctrl + [ctrl[-1] + (ctrl[-1] - ctrl[-2])]
    sampled: list[tuple[Vector, float]] = []
    for i in range(1, len(ext) - 2):
        steps = substeps if i < len(ext) - 3 else substeps + 1
        for k in range(steps):
            t = k / float(substeps)
            v = _catmull_rom_4d(ext[i - 1], ext[i], ext[i + 1], ext[i + 2], t)
            sampled.append((Vector((v.x, v.y, v.z)), max(0.0015, float(v.w))))

    bm = bmesh.new()
    rings = []
    n_pts = len(sampled)
    prev_t = None
    norm = None
    for i, (pos, r) in enumerate(sampled):
        if i == 0:
            tangent = (sampled[1][0] - pos).normalized()
        elif i == n_pts - 1:
            tangent = (pos - sampled[i - 1][0]).normalized()
        else:
            tangent = (sampled[i + 1][0] - sampled[i - 1][0]).normalized()
        if tangent.length < 1e-6:
            tangent = Vector((1.0, 0.0, 0.0))

        if norm is None:
            ref = Vector((0.0, 1.0, 0.0)) if abs(tangent.y) < 0.85 else Vector((0.0, 0.0, 1.0))
            norm = tangent.cross(ref).normalized()
        else:
            q = prev_t.rotation_difference(tangent)
            norm = (q @ norm).normalized()
        binorm = tangent.cross(norm).normalized()
        prev_t = tangent

        ring = []
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs
            offset = (math.cos(ang) * norm + math.sin(ang) * binorm) * r
            ring.append(bm.verts.new(pos + offset))
        rings.append(ring)

    for i in range(len(rings) - 1):
        ra, rb = rings[i], rings[i + 1]
        for s in range(segs):
            sn = (s + 1) % segs
            bm.faces.new((ra[s], ra[sn], rb[sn], rb[s]))
    if cap_ends and rings:
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(mat)
    col.objects.link(obj)
    _finish_mesh(obj, angle_deg=55.0)
    return obj


def add_piano_keybed(
    col,
    prefix: str,
    width: float,
    key_depth: float,
    y_front: float,
    z_top: float,
    mats: dict,
    white_count: int = 14,
    inverted: bool = False,
) -> None:
    """Real separated white keys + authentic 2-3 octave grouped sharp keys."""
    nat_mat = mats["ebony"] if inverted else mats["ivory"]
    sh_mat = mats["ivory"] if inverted else mats["ebony"]
    step = width / white_count
    kw = step * 0.84
    cy = y_front + key_depth * 0.5
    add_box(col, f"{prefix}_KeybedRail", (width + 0.02, key_depth + 0.015, 0.014), (0.0, cy, z_top - 0.008), mats["ebony"], bevel=0.001)
    for i in range(white_count):
        kx = -width * 0.5 + (i + 0.5) * step
        add_box(col, f"{prefix}_WKey_{i}", (kw, key_depth, 0.014), (kx, cy, z_top + 0.002), nat_mat, bevel=0.0012)
        if i < white_count - 1 and (i % 7) in (0, 1, 3, 4, 5):
            sx = kx + step * 0.5
            add_box(col, f"{prefix}_BKey_{i}", (step * 0.54, key_depth * 0.62, 0.018), (sx, cy + key_depth * 0.16, z_top + 0.012), sh_mat, bevel=0.0012)


def add_slotted_bars(
    col,
    prefix: str,
    count: int,
    total_w: float,
    bar_y: float,
    bar_z: float,
    mat: bpy.types.Material,
    len_left: float = 0.22,
    len_right: float = 0.13,
    thickness: float = 0.018,
) -> None:
    """Graduated trapezoidal percussion tone bars (long bass on left -> short treble on right)."""
    step = total_w / count
    for i in range(count):
        t = i / max(1, count - 1)
        x = -total_w * 0.5 + (i + 0.5) * step
        w = step * (0.86 - 0.14 * t)
        blen = len_left * (1.0 - t) + len_right * t
        add_box(col, f"{prefix}_ToneBar_{i}", (w, blen, thickness), (x, bar_y, bar_z), mat, bevel=0.002)


def add_stand_brackets(
    col,
    prefix: str,
    xs: tuple[float, ...],
    y0: float,
    z0: float,
    y1: float,
    z1: float,
    mat: bpy.types.Material,
    radius: float = 0.010,
) -> None:
    """Clean milled mounting arms linking front striker bar (y0, z0) to rear/upper instrument."""
    for i, x in enumerate(xs):
        add_var_tube(col, f"{prefix}_Arm_{i}", [(x, y0, z0, radius), (x, (y0 + y1) * 0.5, z0, radius * 0.9), (x, y1, z1, radius)], mat, segs=12, substeps=3)


def add_s_fhole(col, name: str, x: float, y: float, z: float, scale: float, mat: bpy.types.Material, rot_z: float = 0.0) -> None:
    """Authentic italic S-shaped F-hole silhouette on bowed string tops."""
    pts = [
        (-0.042, 0.012), (-0.025, 0.020), (-0.006, 0.010), (0.012, -0.006),
        (0.032, -0.016), (0.046, -0.006), (0.036, 0.006), (0.020, -0.003),
        (0.002, 0.008), (-0.018, 0.022), (-0.036, 0.020),
    ]
    scaled = [(px * scale, pz * scale) for px, pz in pts]
    add_extruded_profile(col, name, scaled, 0.010, (x, y, z), mat, bevel=0.001, smooth=True, rot=(0.0, rot_z, 0.0))


def add_figure8_body(
    col,
    name: str,
    scale_x: float,
    scale_z: float,
    depth: float,
    loc: tuple[float, float, float],
    mat: bpy.types.Material,
    bevel: float = 0.005,
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    raw = [
        (-0.24, 0.00), (-0.23, 0.10), (-0.16, 0.16), (-0.07, 0.14),
        (0.00, 0.085), (0.07, 0.12), (0.14, 0.13), (0.20, 0.08),
        (0.22, 0.00), (0.20, -0.08), (0.14, -0.13), (0.07, -0.12),
        (0.00, -0.085), (-0.07, -0.14), (-0.16, -0.16), (-0.23, -0.10),
    ]
    pts = [(x * scale_x, z * scale_z) for x, z in raw]
    return add_extruded_profile(col, name, pts, depth, loc, mat, bevel=bevel, smooth=True, rot=rot)


def build_core_bar(col, prefix: str, mat: bpy.types.Material) -> bpy.types.Object:
    return add_box(col, f"{prefix}_Core", (1.0, 0.24, 0.072), (0.0, 0.0, 0.0), mat, bevel=0.0055)


# ---------------------------------------------------------------------------
# Family 1: Keyboards & Mallets (01..12)
# ---------------------------------------------------------------------------

def build_grand_piano(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["ebony"])
    # 14 distinct white + black piano keys behind core front lip + authentic S-curved grand body
    add_box(col, f"{p}_FallboardFelt", (0.94, 0.015, 0.014), (0.0, 0.125, 0.044), mats["felt_red"], bevel=0.001)
    add_piano_keybed(col, p, 0.90, 0.095, 0.130, 0.036, mats, white_count=14)
    # S-curved bentside piano rim + hollow interior rim wall
    rim_xy = [
        (-0.46, 0.225), (-0.46, 0.52), (-0.36, 0.56), (-0.18, 0.52),
        (-0.02, 0.42), (0.16, 0.36), (0.36, 0.35), (0.46, 0.31), (0.46, 0.225),
    ]
    add_extruded_horiz_profile(col, f"{p}_GrandBody", rim_xy, 0.095, (0.0, 0.0, 0.02), mats["ebony"], bevel=0.006, smooth=True)
    # Gold cast-iron harp plate & diagonal strings + propped open curved lid + brass stick
    add_extruded_horiz_profile(
        col, f"{p}_HarpPlate",
        [(-0.40, 0.245), (-0.40, 0.48), (-0.20, 0.48), (0.05, 0.37), (0.40, 0.32), (0.40, 0.245)],
        0.014, (0.0, 0.0, 0.072), mats["brass"], bevel=0.002, smooth=True,
    )
    for i in range(6):
        t = i / 5.0
        x0 = -0.34 + t * 0.62
        add_var_tube(col, f"{p}_Str_{i}", [(x0, 0.245, 0.082, 0.0025), (x0 - 0.08 * (1.0 - t), 0.46 - 0.15 * t, 0.082, 0.0025)], mats["chrome"], segs=6, substeps=2)
    add_extruded_horiz_profile(col, f"{p}_OpenLid", rim_xy, 0.018, (0.0, 0.03, 0.18), mats["ebony"], bevel=0.004, smooth=True, rot=(-0.38, 0.0, 0.0))
    add_cyl(col, f"{p}_PropStick", 0.007, 0.007, 0.18, (0.30, 0.28, 0.14), mats["brass"], rot=(0.25, 0.0, 0.0))


def build_electric_grand(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["steel_dark"])
    add_piano_keybed(col, p, 0.88, 0.090, 0.130, 0.036, mats, white_count=14)
    # CP-80 sloped trapezoidal harp housing + gold iron frame + magnetic pickup bar + flight-case chrome corners
    add_box(col, f"{p}_CaseRim", (0.90, 0.16, 0.08), (0.0, 0.30, 0.03), mats["ebony"], bevel=0.005)
    add_extruded_profile(
        col, f"{p}_CP80HarpBack",
        [(-0.42, 0.04), (-0.36, 0.25), (0.38, 0.14), (0.42, 0.04)],
        0.045, (0.0, 0.36, 0.0), mats["ebony"], bevel=0.004, smooth=False,
    )
    add_box(col, f"{p}_GoldFrame", (0.72, 0.022, 0.025), (-0.02, 0.32, 0.11), mats["brass"], bevel=0.002, rot=(0.0, -0.14, 0.0))
    for i in range(6):
        add_cyl(col, f"{p}_HarpStr_{i}", 0.0028, 0.0028, 0.62 - i * 0.05, (-0.05, 0.30, 0.08 + i * 0.019), mats["chrome"], segs=8, rot=(0.0, math.pi / 2 - 0.14, 0.0))
    for cx in (-0.43, 0.43):
        add_box(col, f"{p}_Corner_{cx:.2f}", (0.045, 0.16, 0.09), (cx, 0.30, 0.03), mats["chrome"], bevel=0.003)


def build_electric_piano(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["ivory"])
    add_piano_keybed(col, p, 0.88, 0.090, 0.130, 0.036, mats, white_count=14)
    # Rhodes namerail with brushed alu faceplate + knobs + deep rounded matte Tolex harp dome behind keys
    add_box(col, f"{p}_Faceplate", (0.90, 0.035, 0.055), (0.0, 0.228, 0.052), mats["brushed_alu"], bevel=0.002)
    for kx in (-0.34, -0.24, 0.24, 0.34):
        add_cyl(col, f"{p}_Knob_{kx:.2f}", 0.014, 0.014, 0.022, (kx, 0.205, 0.055), mats["chrome"], rot=(math.pi / 2, 0.0, 0.0))
    add_extruded_profile(
        col, f"{p}_RhodesDome",
        [(-0.45, 0.02), (-0.43, 0.15), (0.0, 0.175), (0.43, 0.15), (0.45, 0.02)],
        0.18, (0.0, 0.33, 0.02), mats["ebony"], bevel=0.008, smooth=True,
    )
    add_box(col, f"{p}_LogoStrip", (0.28, 0.015, 0.020), (0.0, 0.232, 0.105), mats["chrome"], bevel=0.002)
    for sx in (-0.46, 0.46):
        add_box(col, f"{p}_Handle_{sx:.2f}", (0.024, 0.11, 0.085), (sx, 0.31, 0.06), mats["chrome"], bevel=0.003)


def build_harpsichord(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    # Inverted authentic double manual (ebony naturals + ivory sharps!)
    add_piano_keybed(col, f"{p}_LowerMan", 0.86, 0.075, 0.125, 0.036, mats, white_count=14, inverted=True)
    add_piano_keybed(col, f"{p}_UpperMan", 0.82, 0.065, 0.205, 0.054, mats, white_count=13, inverted=True)
    # Asymmetric triangular pointed harpsichord wing body extending far into +Y on bass side
    wing_xy = [(-0.44, 0.27), (-0.44, 0.56), (-0.28, 0.58), (0.12, 0.42), (0.44, 0.34), (0.44, 0.27)]
    add_extruded_horiz_profile(col, f"{p}_WingBody", wing_xy, 0.090, (0.0, 0.0, 0.03), mats["rosewood"], bevel=0.005, smooth=False)
    add_extruded_horiz_profile(col, f"{p}_Soundboard", [(-0.40, 0.28), (-0.40, 0.53), (-0.27, 0.54), (0.39, 0.32), (0.39, 0.28)], 0.015, (0.0, 0.0, 0.075), mats["spruce"], bevel=0.002, smooth=False)
    add_cyl(col, f"{p}_SoundRosette", 0.045, 0.045, 0.018, (-0.08, 0.36, 0.084), mats["brass"], segs=24)
    for i in range(6):
        t = i / 5.0
        x = -0.35 + t * 0.65
        add_cyl(col, f"{p}_Str_{i}", 0.0025, 0.0025, 0.24 - 0.14 * t, (x, 0.40 - 0.07 * t, 0.088), mats["brass"], segs=8, rot=(math.pi / 2, 0.0, 0.0))
    add_extruded_horiz_profile(col, f"{p}_WingLid", wing_xy, 0.014, (0.0, 0.02, 0.18), mats["rosewood"], bevel=0.003, smooth=False, rot=(-0.34, 0.0, 0.0))


def build_celesta(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["celeste_blue"])
    add_piano_keybed(col, p, 0.84, 0.080, 0.125, 0.036, mats, white_count=12)
    # Upright mahogany Celesta cabinet behind keyboard with acoustic window showing gleaming steel resonators
    add_box(col, f"{p}_Cabinet", (0.86, 0.14, 0.24), (0.0, 0.275, 0.13), mats["mahogany"], bevel=0.006)
    add_box(col, f"{p}_WindowRecess", (0.72, 0.035, 0.11), (0.0, 0.205, 0.14), mats["ebony"], bevel=0.003)
    for i in range(9):
        add_box(col, f"{p}_SteelBar_{i}", (0.052, 0.022, 0.085), (-0.28 + i * 0.07, 0.195, 0.14), mats["chrome"], bevel=0.002)
    add_box(col, f"{p}_MusicDesk", (0.48, 0.020, 0.11), (0.0, 0.24, 0.29), mats["walnut"], bevel=0.003, rot=(-0.22, 0.0, 0.0))
    add_box(col, f"{p}_DeskLedge", (0.50, 0.030, 0.015), (0.0, 0.22, 0.24), mats["brass"], bevel=0.002)


def build_glockenspiel(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    # Trapezoidal crimson velvet resonator case behind strike bar + 10 graduated tapered chrome bars + crossed mallets
    add_extruded_horiz_profile(
        col, f"{p}_TrapCase",
        [(-0.46, 0.14), (-0.46, 0.42), (0.46, 0.32), (0.46, 0.14)],
        0.055, (0.0, 0.0, 0.025), mats["crimson"], bevel=0.005, smooth=False,
    )
    add_slotted_bars(col, p, 10, 0.84, 0.25, 0.062, mats["chrome"], len_left=0.22, len_right=0.12, thickness=0.016)
    for i in range(10):
        t = i / 9.0
        x = -0.378 + i * 0.084
        bl = 0.22 * (1.0 - t) + 0.12 * t
        add_cyl(col, f"{p}_PinF_{i}", 0.003, 0.003, 0.020, (x, 0.25 - bl * 0.35, 0.064), mats["brass"])
        add_cyl(col, f"{p}_PinB_{i}", 0.003, 0.003, 0.020, (x, 0.25 + bl * 0.35, 0.064), mats["brass"])
    add_cyl(col, f"{p}_Mallet1", 0.005, 0.005, 0.30, (-0.15, 0.22, 0.15), mats["bamboo"], rot=(0.0, 0.55, 0.0))
    add_sphere(col, f"{p}_MalletHead1", 0.020, (-0.072, 0.22, 0.278), mats["brass"])
    add_cyl(col, f"{p}_Mallet2", 0.005, 0.005, 0.30, (0.15, 0.22, 0.15), mats["bamboo"], rot=(0.0, -0.55, 0.0))
    add_sphere(col, f"{p}_MalletHead2", 0.020, (0.072, 0.22, 0.278), mats["brass"])


def build_music_box(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    # Open brass clockwork plate behind bar: pinned brass drum + 16-tooth stepped comb + gear + ratchet winder
    add_box(col, f"{p}_ClockPlate", (0.86, 0.18, 0.036), (0.0, 0.23, 0.048), mats["brushed_alu"], bevel=0.004)
    add_cyl(col, f"{p}_PinDrum", 0.052, 0.052, 0.52, (-0.08, 0.26, 0.115), mats["brass"], rot=(0.0, math.pi / 2, 0.0))
    add_box(col, f"{p}_CombSpine", (0.54, 0.035, 0.024), (-0.08, 0.175, 0.075), mats["chrome"], bevel=0.002)
    for i in range(15):
        x = -0.325 + i * 0.035
        h = 0.070 - i * 0.0025
        add_box(col, f"{p}_CombTooth_{i}", (0.016, 0.014, h), (x, 0.195, 0.082), mats["chrome"], bevel=0.001, rot=(math.pi / 2, 0.0, 0.0))
    add_cyl(col, f"{p}_GearWheel", 0.058, 0.058, 0.018, (0.26, 0.25, 0.115), mats["brass"], segs=20, rot=(math.pi / 2, 0.0, 0.0))
    add_cyl(col, f"{p}_WinderShaft", 0.010, 0.010, 0.10, (0.36, 0.25, 0.115), mats["brass"], rot=(0.0, math.pi / 2, 0.0))
    add_torus_arc(col, f"{p}_WinderBow", 0.036, 0.008, (0.42, 0.25, 0.115), mats["brass"], rot=(0.0, math.pi / 2, 0.0))


def build_vibraphone(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["steel_dark"])
    # Graduated trapezoid brushed-alu bars + stepped brass tubes + butterfly damper rod & motor pulley
    add_slotted_bars(col, p, 11, 0.92, 0.23, 0.046, mats["brushed_alu"], len_left=0.24, len_right=0.14)
    for i in range(7):
        t = i / 6.0
        tx = -0.36 + i * 0.12
        tl = 0.28 - t * 0.14
        add_cyl(col, f"{p}_VibeTube_{i}", 0.024, 0.024, tl, (tx, 0.23, 0.02 - tl * 0.5), mats["brass"])
    add_cyl(col, f"{p}_DamperRod", 0.008, 0.008, 0.86, (0.0, 0.23, 0.018), mats["chrome"], rot=(0.0, math.pi / 2, 0.0))
    add_box(col, f"{p}_DamperBar", (0.84, 0.026, 0.014), (0.0, 0.23, -0.002), mats["felt_red"], bevel=0.002)
    add_cyl(col, f"{p}_MotorHousing", 0.042, 0.042, 0.06, (-0.46, 0.22, -0.02), mats["chrome"], rot=(0.0, math.pi / 2, 0.0))


def build_marimba(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    # True tapered rosewood bars (long bass on left -> short treble on right) + bronze curved resonator arc
    add_slotted_bars(col, p, 11, 0.92, 0.24, 0.052, mats["rosewood"], len_left=0.28, len_right=0.14)
    for i in range(8):
        t = i / 7.0
        tx = -0.37 + i * 0.106
        tl = 0.34 - t * 0.19
        add_cyl(col, f"{p}_MarimbaTube_{i}", 0.023, 0.023, tl, (tx, 0.24, 0.02 - tl * 0.5), mats["bronze"])
    add_var_tube(col, f"{p}_ArchRail", [(-0.42, 0.22, -0.34, 0.013), (-0.15, 0.22, -0.24, 0.013), (0.15, 0.22, -0.18, 0.013), (0.42, 0.22, -0.13, 0.013)], mats["walnut"])
    add_cyl(col, f"{p}_MalletShaft", 0.005, 0.005, 0.32, (0.20, 0.20, 0.12), mats["bamboo"], rot=(0.0, -0.55, 0.0))
    add_sphere(col, f"{p}_MalletYarn", 0.025, (0.116, 0.20, 0.256), mats["felt_red"])


def build_xylophone(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["ebony"])
    # Compact dual-row tabletop padauk xylophone + steeper taper + shorter angled resonators
    add_box(col, f"{p}_XyloRail", (0.88, 0.12, 0.030), (0.0, 0.22, 0.040), mats["ebony"], bevel=0.003)
    add_slotted_bars(col, p, 10, 0.86, 0.23, 0.062, mats["xylo_gold"], len_left=0.20, len_right=0.11, thickness=0.016)
    for i in range(6):
        tx = -0.30 + i * 0.12
        tl = 0.17 - i * 0.018
        add_cyl(col, f"{p}_XyloTube_{i}", 0.020, 0.020, tl, (tx, 0.22, 0.01 - tl * 0.5), mats["antique_brass"])
    add_cyl(col, f"{p}_Mallet", 0.005, 0.005, 0.28, (-0.18, 0.20, 0.13), mats["bamboo"], rot=(0.0, 0.55, 0.0))
    add_sphere(col, f"{p}_MalletBall", 0.020, (-0.107, 0.20, 0.249), mats["ivory"])


def build_tubular_bells(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["brass"])
    add_box(col, f"{p}_GantryTop", (0.84, 0.05, 0.032), (0.0, 0.22, 0.16), mats["steel_dark"], bevel=0.003)
    add_cyl(col, f"{p}_PostL", 0.014, 0.014, 0.34, (-0.40, 0.22, 0.00), mats["steel_dark"])
    add_cyl(col, f"{p}_PostR", 0.014, 0.014, 0.34, (0.40, 0.22, 0.00), mats["steel_dark"])
    for i in range(7):
        x = -0.30 + i * 0.10
        l = 0.34 - i * 0.028
        add_cyl(col, f"{p}_Cord_{i}", 0.004, 0.004, 0.10, (x, 0.22, 0.11), mats["crimson"])
        add_lathe(col, f"{p}_Chime_{i}", [(0.021, -l), (0.021, 0.0), (0.025, 0.012), (0.006, 0.026)], (x, 0.22, 0.06), mats["brass"], segs=20, cap_bottom=True, cap_top=True)
    add_box(col, f"{p}_DamperBeam", (0.76, 0.025, 0.020), (0.0, 0.22, -0.14), mats["felt_red"], bevel=0.002)
    add_cyl(col, f"{p}_HammerHandle", 0.007, 0.007, 0.26, (0.35, 0.175, 0.10), mats["walnut"], rot=(0.0, -0.45, 0.0))
    add_cyl(col, f"{p}_RawhideHead", 0.022, 0.022, 0.065, (0.29, 0.175, 0.21), mats["ivory"], rot=(0.0, math.pi / 2 - 0.45, 0.0))


def build_dulcimer(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["spruce"])
    # Detached behind/above front bar on twin brass supports so 100% of trapezoidal soundboard shows!
    add_stand_brackets(col, p, (-0.22, 0.22), 0.11, 0.01, 0.22, 0.08, mats["brass"])
    add_extruded_profile(
        col, f"{p}_TrapezoidBox",
        [(-0.44, 0.06), (-0.28, 0.28), (0.28, 0.28), (0.44, 0.06)],
        0.055, (0.0, 0.24, 0.0), mats["walnut"], bevel=0.005, smooth=False,
    )
    for rx in (-0.18, 0.18):
        add_cyl(col, f"{p}_Rosette_{rx:.2f}", 0.042, 0.042, 0.060, (rx, 0.232, 0.17), mats["brass"], segs=24, rot=(math.pi / 2, 0.0, 0.0))
        add_cyl(col, f"{p}_Hole_{rx:.2f}", 0.028, 0.028, 0.064, (rx, 0.228, 0.17), mats["ebony"], segs=24, rot=(math.pi / 2, 0.0, 0.0))
    add_box(col, f"{p}_BridgeL", (0.020, 0.068, 0.18), (-0.08, 0.21, 0.17), mats["ivory"], bevel=0.002)
    add_box(col, f"{p}_BridgeR", (0.020, 0.068, 0.18), (0.09, 0.21, 0.17), mats["ivory"], bevel=0.002)
    for i in range(6):
        add_cyl(col, f"{p}_Str_{i}", 0.0025, 0.0025, 0.66 - i * 0.04, (0.0, 0.20, 0.10 + i * 0.028), mats["chrome"], segs=8, rot=(0.0, math.pi / 2, 0.0))
    add_cyl(col, f"{p}_Hammer1", 0.005, 0.005, 0.24, (-0.18, 0.19, 0.26), mats["bamboo"], rot=(0.0, 0.55, 0.0))
    add_cyl(col, f"{p}_Hammer2", 0.005, 0.005, 0.24, (0.18, 0.19, 0.26), mats["bamboo"], rot=(0.0, -0.55, 0.0))


# ---------------------------------------------------------------------------
# Family 2: Organs, Accordions, Guitars & Harp (13..24)
# ---------------------------------------------------------------------------

def build_drawbar_organ(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    add_stand_brackets(col, p, (-0.30, 0.30), 0.10, 0.015, 0.22, 0.075, mats["brass"], radius=0.010)
    # Hammond B3 dual-manual wooden console above/behind Core
    add_box(col, f"{p}_ConsoleCheekL", (0.045, 0.20, 0.25), (-0.39, 0.21, 0.175), mats["mahogany"], bevel=0.004)
    add_box(col, f"{p}_ConsoleCheekR", (0.045, 0.20, 0.25), (0.39, 0.21, 0.175), mats["mahogany"], bevel=0.004)
    add_box(col, f"{p}_ConsoleBack", (0.76, 0.04, 0.25), (0.0, 0.30, 0.175), mats["mahogany"], bevel=0.004)
    # Two stacked organ manuals facing front
    add_piano_keybed(col, f"{p}_LowerManual", 0.72, 0.072, 0.135, 0.088, mats, white_count=12)
    add_piano_keybed(col, f"{p}_UpperManual", 0.72, 0.072, 0.185, 0.145, mats, white_count=12)
    # Tilted drawbar fascia with 9 authentic stepped Hammond drawbars
    add_box(col, f"{p}_DrawbarFascia", (0.75, 0.08, 0.095), (0.0, 0.25, 0.235), mats["walnut"], bevel=0.004, rot=(-0.38, 0.0, 0.0))
    for i in range(9):
        x = -0.28 + i * 0.07
        pull = 0.015 + ((i * 3) % 5) * 0.012
        tip_mat = mats["ivory"] if i in (0, 2, 3, 5, 7) else (mats["ebony"] if i in (1, 4, 6) else mats["copper"])
        add_box(col, f"{p}_DrawShaft_{i}", (0.014, pull + 0.02, 0.010), (x, 0.22 - pull * 0.5, 0.23), mats["chrome"], bevel=0.001)
        add_box(col, f"{p}_DrawTip_{i}", (0.032, 0.026, 0.020), (x, 0.20 - pull, 0.225), tip_mat, bevel=0.002)
    add_box(col, f"{p}_LeslieSwitch", (0.04, 0.02, 0.024), (-0.34, 0.135, 0.12), mats["chrome"], bevel=0.002)
    # Leslie rotary speaker cabinet cleanly slung underneath Core
    add_box(col, f"{p}_LeslieCab", (0.52, 0.10, 0.13), (0.0, 0.22, -0.115), mats["walnut"], bevel=0.005)
    for l in range(4):
        add_box(col, f"{p}_Louver_{l}", (0.42, 0.022, 0.014), (0.0, 0.168, -0.075 - l * 0.025), mats["ebony"], bevel=0.001)


def build_church_organ(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["ivory"])
    add_stand_brackets(col, p, (-0.30, 0.30), 0.10, 0.015, 0.22, 0.055, mats["brass"], radius=0.010)
    # Carved oak windchest & 3 Gothic pipe towers
    add_box(col, f"{p}_Windchest", (0.86, 0.11, 0.065), (0.0, 0.22, 0.068), mats["walnut"], bevel=0.005)
    for tx, th in ((-0.41, 0.29), (0.0, 0.34), (0.41, 0.29)):
        w = 0.065 if tx != 0.0 else 0.050
        if tx != 0.0:
            add_box(col, f"{p}_Tower_{tx:.2f}", (w, 0.10, th), (tx, 0.225, 0.08 + th * 0.5), mats["walnut"], bevel=0.004)
            add_cyl(col, f"{p}_Cap_{tx:.2f}", 0.048, 0.008, 0.055, (tx, 0.225, 0.095 + th), mats["brass"], segs=6)
    # 9 Organ pipes with lathed conical feet and gold bay mouths (tall center flat + symmetrical flats)
    heights = [0.22, 0.18, 0.15, 0.21, 0.28, 0.21, 0.15, 0.18, 0.22]
    for i, h in enumerate(heights):
        x = -0.32 + i * 0.08
        r = 0.017 + (0.007 if i in (0, 4, 8) else 0.0)
        add_lathe(
            col, f"{p}_PipeFoot_{i}",
            [(0.005, 0.0), (0.011, 0.015), (r, 0.055), (r, 0.062)],
            (x, 0.20, 0.10), mats["brass"], segs=18,
        )
        add_cyl(col, f"{p}_Pipe_{i}", r, r, h, (x, 0.20, 0.162 + h * 0.5), mats["chrome"], segs=20)
        add_box(col, f"{p}_BayMouth_{i}", (r * 1.4, 0.022, 0.026), (x, 0.20 - r * 0.85, 0.162), mats["ebony"], bevel=0.001)
        add_box(col, f"{p}_UpperLip_{i}", (r * 1.5, 0.016, 0.012), (x, 0.20 - r * 0.90, 0.178), mats["brass"], bevel=0.001)
    # Compact detached console manual below Core
    add_piano_keybed(col, f"{p}_PedalConsole", 0.44, 0.06, 0.18, -0.085, mats, white_count=8)


def build_accordion(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["crimson"])
    add_stand_brackets(col, p, (-0.28, 0.28), 0.10, 0.015, 0.22, 0.08, mats["chrome"], radius=0.010)
    # Elevated & showcased completely above Core
    z_c = 0.175
    y_c = 0.22
    # Left: Crimson pearl treble case with piano keyboard facing front + silver grille
    add_box(col, f"{p}_TrebleCase", (0.22, 0.15, 0.22), (-0.27, y_c, z_c), mats["crimson"], bevel=0.008)
    add_box(col, f"{p}_TrebleGrille", (0.14, 0.026, 0.17), (-0.30, y_c - 0.072, z_c), mats["chrome"], bevel=0.004)
    for g in range(4):
        add_box(col, f"{p}_GrilleSlot_{g}", (0.016, 0.028, 0.13), (-0.345 + g * 0.03, y_c - 0.075, z_c), mats["ebony"], bevel=0.001)
    # Vertical piano keys along inner treble cheek
    for k in range(7):
        kz = (z_c - 0.075) + k * 0.025
        add_box(col, f"{p}_WKey_{k}", (0.052, 0.042, 0.021), (-0.195, y_c - 0.068, kz), mats["ivory"], bevel=0.001)
        if k in (0, 1, 3, 4, 5):
            add_box(col, f"{p}_BKey_{k}", (0.030, 0.026, 0.013), (-0.206, y_c - 0.082, kz + 0.012), mats["ebony"], bevel=0.001)
    # Center: Continuous zig-zag accordion bellows folds
    zig_pts = []
    for i in range(15):
        zx = -0.15 + i * 0.021
        zy = -0.052 if i % 2 == 0 else -0.072
        zig_pts.append((zx, zy))
    for i in range(14, -1, -1):
        zx = -0.15 + i * 0.021
        zy = 0.052 if i % 2 == 0 else 0.072
        zig_pts.append((zx, zy))
    add_extruded_horiz_profile(col, f"{p}_Bellows", zig_pts, 0.19, (0.0, y_c, z_c - 0.095), mats["ivory"], bevel=0.001, smooth=False)
    for i in range(7):
        px = -0.129 + i * 0.042
        add_box(col, f"{p}_PleatTape_{i}", (0.012, 0.136, 0.192), (px, y_c, z_c), mats["crimson"], bevel=0.001)
        add_box(col, f"{p}_CornerT_{i}", (0.018, 0.140, 0.014), (px, y_c, z_c + 0.095), mats["chrome"], bevel=0.001)
        add_box(col, f"{p}_CornerB_{i}", (0.018, 0.140, 0.014), (px, y_c, z_c - 0.095), mats["chrome"], bevel=0.001)
    # Right: Bass box with 4x3 pearl buttons & leather strap
    add_box(col, f"{p}_BassCase", (0.17, 0.15, 0.22), (0.245, y_c, z_c), mats["crimson"], bevel=0.008)
    for row in range(4):
        for col_b in range(3):
            bx = 0.205 + col_b * 0.036
            bz = (z_c - 0.06) + row * 0.040
            add_cyl(col, f"{p}_BassBtn_{row}_{col_b}", 0.008, 0.008, 0.018, (bx, y_c - 0.076, bz), mats["pearl"], rot=(math.pi / 2, 0.0, 0.0))
    add_box(col, f"{p}_BassStrap", (0.032, 0.024, 0.19), (0.335, y_c, z_c), mats["mahogany"], bevel=0.003)


def build_harmonica(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["steel_dark"])
    add_stand_brackets(col, p, (-0.28, 0.28), 0.10, 0.015, 0.22, 0.125, mats["brass"], radius=0.010)
    # Marine Band 10-hole blues harp: solid domed mirror-chrome shells tilted forward so both comb holes & curved engraving face camera
    tilt = -0.45
    y_h, z_h = 0.22, 0.165
    # Closed solid upper & lower arched chrome covers
    add_extruded_profile(
        col, f"{p}_CoverTop",
        [(-0.39, 0.020), (-0.37, 0.048), (-0.30, 0.074), (0.30, 0.074), (0.37, 0.048), (0.39, 0.020)],
        0.105, (0.0, y_h, z_h), mats["chrome"], bevel=0.004, smooth=True, rot=(tilt, 0.0, 0.0),
    )
    add_extruded_profile(
        col, f"{p}_CoverBot",
        [(-0.39, -0.020), (-0.37, -0.048), (-0.30, -0.066), (0.30, -0.066), (0.37, -0.048), (0.39, -0.020)],
        0.105, (0.0, y_h, z_h), mats["chrome"], bevel=0.004, smooth=True, rot=(tilt, 0.0, 0.0),
    )
    add_box(col, f"{p}_ReedPlateT", (0.81, 0.114, 0.012), (0.0, y_h, z_h + 0.018), mats["brass"], bevel=0.001, rot=(tilt, 0.0, 0.0))
    add_box(col, f"{p}_ReedPlateB", (0.81, 0.114, 0.012), (0.0, y_h, z_h - 0.018), mats["brass"], bevel=0.001, rot=(tilt, 0.0, 0.0))
    add_box(col, f"{p}_PearwoodComb", (0.80, 0.110, 0.032), (0.0, y_h, z_h), mats["mahogany"], bevel=0.002, rot=(tilt, 0.0, 0.0))
    for i in range(10):
        add_box(col, f"{p}_Hole_{i}", (0.042, 0.030, 0.022), (-0.315 + i * 0.070, y_h - 0.048, z_h), mats["ebony"], bevel=0.001, rot=(tilt, 0.0, 0.0))
    for rx in (-0.355, 0.355):
        add_cyl(col, f"{p}_CoverBolt_{rx:.3f}", 0.013, 0.013, 0.14, (rx, y_h, z_h), mats["brass"], rot=(tilt, 0.0, 0.0))


def _add_guitar_frets_and_strings(col, p: str, y_face: float, z_c: float, x_bridge: float, x_nut: float, n_str: int, str_mat, fret_mat, span_z: float = 0.034):
    # Brass/silver fret wires along fingerboard
    for f in range(9):
        fx = x_nut - 0.03 - f * ((x_nut - 0.06 - (x_bridge + 0.18)) / 8.0)
        if fx > x_bridge + 0.12:
            add_box(col, f"{p}_Fret_{f}", (0.004, 0.008, span_z), (fx, y_face, z_c), fret_mat, bevel=0.0005)
    # Individual strings
    length = x_nut - x_bridge
    mid_x = 0.5 * (x_nut + x_bridge)
    for i in range(n_str):
        sz = z_c + (i - (n_str - 1) * 0.5) * (span_z * 0.85 / max(1, n_str - 1))
        add_cyl(col, f"{p}_Str_{i}", 0.0022 if n_str >= 6 else 0.0038, 0.0022 if n_str >= 6 else 0.0038, length, (mid_x, y_face - 0.004, sz), str_mat, segs=8, rot=(0.0, math.pi / 2, 0.0))


def build_nylon_guitar(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    add_stand_brackets(col, p, (-0.22, 0.12), 0.10, 0.015, 0.22, 0.10, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.185
    add_figure8_body(col, f"{p}_Body", 1.12, 1.08, 0.065, (-0.08, y_c, z_c), mats["amber_wood"], bevel=0.006)
    add_cyl(col, f"{p}_Rosette", 0.068, 0.068, 0.068, (-0.02, y_c - 0.004, z_c), mats["brass"], segs=28, rot=(math.pi / 2, 0.0, 0.0))
    add_cyl(col, f"{p}_Soundhole", 0.048, 0.048, 0.072, (-0.02, y_c - 0.006, z_c), mats["ebony"], segs=28, rot=(math.pi / 2, 0.0, 0.0))
    add_box(col, f"{p}_Bridge", (0.042, 0.074, 0.115), (-0.20, y_c - 0.015, z_c), mats["rosewood"], bevel=0.002)
    add_box(col, f"{p}_Saddle", (0.008, 0.078, 0.085), (-0.195, y_c - 0.018, z_c), mats["ivory"], bevel=0.001)
    add_box(col, f"{p}_Neck", (0.40, 0.032, 0.046), (0.24, y_c - 0.012, z_c), mats["ebony"], bevel=0.003)
    add_box(col, f"{p}_SlottedHead", (0.12, 0.028, 0.068), (0.48, y_c - 0.008, z_c), mats["walnut"], bevel=0.004)
    for row_z in (-0.024, 0.024):
        for px in (0.44, 0.48, 0.52):
            add_cyl(col, f"{p}_Peg_{row_z:.2f}_{px:.2f}", 0.007, 0.007, 0.022, (px, y_c - 0.008, z_c + row_z * 1.45), mats["ivory"])
    _add_guitar_frets_and_strings(col, p, y_c - 0.030, z_c, -0.195, 0.43, 6, mats["ivory"], mats["brass"], span_z=0.040)


def build_steel_guitar(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["spruce"])
    add_stand_brackets(col, p, (-0.22, 0.12), 0.10, 0.015, 0.22, 0.10, mats["chrome"], radius=0.009)
    y_c, z_c = 0.22, 0.185
    add_figure8_body(col, f"{p}_Body", 1.22, 1.15, 0.072, (-0.08, y_c, z_c), mats["sunburst"], bevel=0.006)
    add_extruded_profile(col, f"{p}_Pickguard", [(-0.04, -0.02), (0.04, -0.04), (0.02, -0.11), (-0.07, -0.09)], 0.076, (-0.01, y_c - 0.004, z_c), mats["tortoiseshell"], bevel=0.002, smooth=True)
    add_cyl(col, f"{p}_Rosette", 0.062, 0.062, 0.074, (-0.01, y_c - 0.004, z_c), mats["ivory"], segs=28, rot=(math.pi / 2, 0.0, 0.0))
    add_cyl(col, f"{p}_Soundhole", 0.048, 0.048, 0.078, (-0.01, y_c - 0.006, z_c), mats["ebony"], segs=28, rot=(math.pi / 2, 0.0, 0.0))
    add_box(col, f"{p}_Bridge", (0.040, 0.076, 0.118), (-0.19, y_c - 0.016, z_c), mats["ebony"], bevel=0.002)
    add_box(col, f"{p}_Neck", (0.42, 0.030, 0.042), (0.25, y_c - 0.012, z_c), mats["rosewood"], bevel=0.003)
    add_box(col, f"{p}_Headstock", (0.115, 0.028, 0.062), (0.49, y_c - 0.008, z_c), mats["walnut"], bevel=0.003)
    for sz in (-0.036, 0.036):
        for px in (0.45, 0.49, 0.53):
            add_cyl(col, f"{p}_Tuner_{sz:.2f}_{px:.2f}", 0.007, 0.007, 0.018, (px, y_c - 0.008, z_c + sz), mats["chrome"])
    _add_guitar_frets_and_strings(col, p, y_c - 0.030, z_c, -0.19, 0.44, 6, mats["bronze"], mats["chrome"], span_z=0.036)


def build_clean_electric_guitar(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    add_stand_brackets(col, p, (-0.22, 0.10), 0.10, 0.015, 0.22, 0.10, mats["chrome"], radius=0.009)
    y_c, z_c = 0.22, 0.185
    strat_pts = [
        (-0.31, 0.00), (-0.29, 0.11), (-0.18, 0.155), (-0.08, 0.11),
        (0.07, 0.165), (0.13, 0.12), (0.04, 0.04), (0.08, -0.05),
        (0.02, -0.12), (-0.08, -0.095), (-0.19, -0.155), (-0.29, -0.11),
    ]
    add_extruded_profile(col, f"{p}_StratBody", strat_pts, 0.052, (-0.08, y_c, z_c), mats["olympic_white"], bevel=0.008, smooth=True)
    add_extruded_profile(col, f"{p}_Pickguard", [(-0.20, 0.065), (-0.06, 0.095), (0.02, 0.03), (0.00, -0.075), (-0.19, -0.085)], 0.058, (-0.06, y_c - 0.004, z_c), mats["tortoiseshell"], bevel=0.002, smooth=True)
    for px in (-0.18, -0.11, -0.04):
        add_box(col, f"{p}_SC_{px:.2f}", (0.020, 0.062, 0.082), (px, y_c - 0.010, z_c), mats["ivory"], bevel=0.002)
    add_box(col, f"{p}_Bridge", (0.045, 0.066, 0.086), (-0.25, y_c - 0.010, z_c), mats["chrome"], bevel=0.002)
    add_box(col, f"{p}_Neck", (0.43, 0.028, 0.038), (0.21, y_c - 0.008, z_c), mats["spruce"], bevel=0.003)
    add_box(col, f"{p}_Headstock", (0.13, 0.028, 0.060), (0.48, y_c - 0.008, z_c + 0.015), mats["spruce"], bevel=0.003)
    for ti in range(6):
        add_cyl(col, f"{p}_Tuner_{ti}", 0.006, 0.006, 0.022, (0.43 + ti * 0.017, y_c - 0.008, z_c + 0.050), mats["chrome"])
    add_cyl(col, f"{p}_Whammy", 0.005, 0.005, 0.14, (-0.27, y_c - 0.042, z_c - 0.06), mats["chrome"], rot=(0.0, 0.45, 0.0))
    for kx in (-0.24, -0.19):
        add_cyl(col, f"{p}_VolKnob_{kx:.2f}", 0.012, 0.012, 0.022, (kx, y_c - 0.040, z_c - 0.085), mats["ivory"], rot=(math.pi / 2, 0.0, 0.0))
    _add_guitar_frets_and_strings(col, p, y_c - 0.026, z_c, -0.25, 0.42, 6, mats["chrome"], mats["brass"], span_z=0.032)


def build_distortion_guitar(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["ebony"])
    add_stand_brackets(col, p, (-0.24, 0.08), 0.10, 0.015, 0.22, 0.10, mats["crimson"], radius=0.009)
    y_c, z_c = 0.22, 0.185
    # Crisp single-piece Flying-V body profile
    v_pts = [
        (-0.34, 0.145), (-0.06, 0.085), (0.06, 0.045),
        (0.06, -0.045), (-0.06, -0.085), (-0.34, -0.145),
        (-0.26, -0.055), (-0.16, 0.0), (-0.26, 0.055),
    ]
    add_extruded_profile(col, f"{p}_FlyingV", v_pts, 0.052, (-0.05, y_c, z_c), mats["crimson"], bevel=0.005, smooth=False)
    add_box(col, f"{p}_Pickguard", (0.14, 0.056, 0.088), (-0.11, y_c - 0.004, z_c), mats["olympic_white"], bevel=0.002)
    for px in (-0.12, -0.03):
        add_box(col, f"{p}_Humbucker_{px:.2f}", (0.042, 0.062, 0.080), (px, y_c - 0.008, z_c), mats["ebony"], bevel=0.003)
    add_box(col, f"{p}_Bridge", (0.028, 0.062, 0.084), (-0.18, y_c - 0.008, z_c), mats["chrome"], bevel=0.002)
    add_box(col, f"{p}_Neck", (0.43, 0.028, 0.038), (0.21, y_c - 0.006, z_c), mats["ebony"], bevel=0.003)
    add_extruded_profile(col, f"{p}_PointHead", [(0.0, 0.022), (0.13, 0.044), (0.17, -0.012), (0.02, -0.022)], 0.028, (0.42, y_c - 0.006, z_c), mats["crimson"], bevel=0.002, smooth=False)
    for ti in range(6):
        add_cyl(col, f"{p}_Tuner_{ti}", 0.006, 0.006, 0.020, (0.44 + ti * 0.019, y_c - 0.006, z_c + 0.042), mats["chrome"])
    _add_guitar_frets_and_strings(col, p, y_c - 0.024, z_c, -0.18, 0.42, 6, mats["chrome"], mats["chrome"], span_z=0.032)


def build_acoustic_bass(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["mahogany"])
    add_stand_brackets(col, p, (-0.22, 0.12), 0.10, 0.015, 0.22, 0.10, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.185
    add_figure8_body(col, f"{p}_Body", 1.24, 1.18, 0.078, (-0.10, y_c, z_c), mats["mahogany"], bevel=0.007)
    add_cyl(col, f"{p}_Rosette", 0.068, 0.068, 0.080, (-0.01, y_c - 0.004, z_c), mats["brass"], segs=28, rot=(math.pi / 2, 0.0, 0.0))
    add_cyl(col, f"{p}_Soundhole", 0.052, 0.052, 0.084, (-0.01, y_c - 0.006, z_c), mats["ebony"], segs=28, rot=(math.pi / 2, 0.0, 0.0))
    add_box(col, f"{p}_Bridge", (0.044, 0.082, 0.112), (-0.21, y_c - 0.016, z_c), mats["ebony"], bevel=0.002)
    add_box(col, f"{p}_Neck", (0.46, 0.032, 0.044), (0.24, y_c - 0.012, z_c), mats["ebony"], bevel=0.003)
    add_box(col, f"{p}_Headstock", (0.12, 0.030, 0.068), (0.50, y_c - 0.008, z_c), mats["mahogany"], bevel=0.004)
    for sz in (-0.044, 0.044):
        for px in (0.47, 0.53):
            add_cyl(col, f"{p}_BassTuner_{sz:.2f}_{px:.2f}", 0.011, 0.011, 0.020, (px, y_c - 0.008, z_c + sz), mats["chrome"])
    _add_guitar_frets_and_strings(col, p, y_c - 0.032, z_c, -0.21, 0.46, 4, mats["bronze"], mats["brass"], span_z=0.034)


def build_fretless_bass(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    add_stand_brackets(col, p, (-0.22, 0.10), 0.10, 0.015, 0.22, 0.10, mats["chrome"], radius=0.009)
    y_c, z_c = 0.22, 0.185
    jbass_pts = [
        (-0.32, 0.00), (-0.30, 0.12), (-0.18, 0.15), (-0.06, 0.10),
        (0.08, 0.165), (0.14, 0.12), (0.03, 0.04), (0.06, -0.05),
        (0.00, -0.12), (-0.10, -0.10), (-0.22, -0.165), (-0.30, -0.12),
    ]
    add_extruded_profile(col, f"{p}_JBassBody", jbass_pts, 0.054, (-0.09, y_c, z_c), mats["walnut"], bevel=0.008, smooth=True)
    add_extruded_profile(col, f"{p}_ControlPlate", [(-0.24, -0.06), (-0.11, -0.06), (-0.11, -0.11), (-0.24, -0.11)], 0.058, (-0.09, y_c - 0.003, z_c), mats["chrome"], bevel=0.002, smooth=False)
    for px in (-0.17, -0.07):
        add_box(col, f"{p}_JPU_{px:.2f}", (0.026, 0.062, 0.088), (px, y_c - 0.008, z_c), mats["ebony"], bevel=0.003)
    add_box(col, f"{p}_Bridge", (0.044, 0.064, 0.086), (-0.26, y_c - 0.008, z_c), mats["chrome"], bevel=0.002)
    # Smooth unfretted lined ebony fingerboard (defining Fretless Bass trait)
    add_box(col, f"{p}_FretlessBoard", (0.47, 0.032, 0.040), (0.21, y_c - 0.008, z_c), mats["ebony"], bevel=0.002)
    add_box(col, f"{p}_Headstock", (0.13, 0.028, 0.064), (0.49, y_c - 0.006, z_c + 0.015), mats["spruce"], bevel=0.004)
    for ti in range(4):
        add_cyl(col, f"{p}_Tuner_{ti}", 0.011, 0.011, 0.022, (0.44 + ti * 0.028, y_c - 0.006, z_c + 0.054), mats["chrome"])
    for kx in (-0.25, -0.20, -0.15):
        add_cyl(col, f"{p}_Knob_{kx:.2f}", 0.011, 0.011, 0.022, (kx, y_c - 0.038, z_c - 0.085), mats["ebony"], rot=(math.pi / 2, 0.0, 0.0))
    # 4 thick steel bass strings with ZERO frets
    for i in range(4):
        sz = z_c + (i - 1.5) * 0.010
        add_cyl(col, f"{p}_Str_{i}", 0.0038, 0.0038, 0.69, (0.085, y_c - 0.028, sz), mats["chrome"], segs=8, rot=(0.0, math.pi / 2, 0.0))


def build_pizzicato_strings(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["spruce"])
    add_stand_brackets(col, p, (-0.18, 0.08), 0.10, 0.015, 0.22, 0.10, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.185
    add_figure8_body(col, f"{p}_Body", 0.96, 0.96, 0.052, (-0.06, y_c, z_c), mats["amber_wood"], bevel=0.005)
    add_s_fhole(col, f"{p}_FholeU", -0.06, y_c - 0.024, z_c + 0.048, 0.85, mats["ebony"], rot_z=0.0)
    add_s_fhole(col, f"{p}_FholeL", -0.06, y_c - 0.024, z_c - 0.048, 0.85, mats["ebony"], rot_z=math.pi)
    add_box(col, f"{p}_Bridge", (0.018, 0.062, 0.086), (-0.06, y_c - 0.010, z_c), mats["ivory"], bevel=0.002)
    add_box(col, f"{p}_Tailpiece", (0.11, 0.058, 0.042), (-0.19, y_c - 0.008, z_c), mats["ebony"], bevel=0.002)
    add_box(col, f"{p}_Neck", (0.30, 0.026, 0.032), (0.20, y_c - 0.006, z_c), mats["ebony"], bevel=0.002)
    add_cyl(col, f"{p}_Scroll", 0.022, 0.022, 0.034, (0.37, y_c - 0.006, z_c), mats["amber_wood"], rot=(math.pi / 2, 0.0, 0.0))
    # Top string visibly plucked into a golden chevron + clean ripple arcs above fingerboard end
    add_var_tube(col, f"{p}_PluckedStr", [(-0.18, y_c - 0.042, z_c + 0.016, 0.0025), (0.04, y_c - 0.042, z_c + 0.044, 0.0025), (0.34, y_c - 0.042, z_c + 0.014, 0.0025)], mats["brass"], segs=10, substeps=2)
    for idx in range(3):
        sz = z_c - 0.014 + idx * 0.010
        add_cyl(col, f"{p}_Str_{idx}", 0.0022, 0.0022, 0.50, (0.08, y_c - 0.042, sz), mats["chrome"], segs=8, rot=(0.0, math.pi / 2, 0.0))
    add_torus_arc(col, f"{p}_PluckWave1", 0.028, 0.0035, (0.04, y_c - 0.044, z_c + 0.044), mats["brass"], arc_deg=140.0, rot=(math.pi / 2, 0.25, 0.0))
    add_torus_arc(col, f"{p}_PluckWave2", 0.046, 0.0030, (0.04, y_c - 0.044, z_c + 0.044), mats["brass"], arc_deg=140.0, rot=(math.pi / 2, 0.25, 0.0))


def build_orchestral_harp(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["brass"])
    add_stand_brackets(col, p, (-0.30, 0.20), 0.10, 0.015, 0.22, 0.06, mats["brass"], radius=0.010)
    y_c = 0.22
    # Entire concert grand harp elevated clearly above Core
    add_box(col, f"{p}_Pedestal", (0.58, 0.09, 0.036), (-0.06, y_c, 0.055), mats["brass"], bevel=0.004)
    add_cyl(col, f"{p}_Pillar", 0.024, 0.032, 0.31, (-0.34, y_c, 0.21), mats["brass"], segs=20)
    add_cyl(col, f"{p}_Crown", 0.038, 0.024, 0.045, (-0.34, y_c, 0.375), mats["brass"], segs=20)
    add_box(col, f"{p}_Soundbox", (0.50, 0.065, 0.072), (0.02, y_c, 0.135), mats["walnut"], bevel=0.008, rot=(0.0, -0.44, 0.0))
    add_var_tube(
        col, f"{p}_HarmonicNeck",
        [(-0.34, y_c, 0.355, 0.022), (-0.12, y_c, 0.32, 0.020), (0.08, y_c, 0.26, 0.018), (0.28, y_c, 0.23, 0.016)],
        mats["brass"], segs=16, substeps=5,
    )
    for i in range(12):
        t = i / 11.0
        x = -0.28 + t * 0.50
        z_top = 0.345 - t * 0.105 - 0.025 * math.sin(t * math.pi)
        z_bot = 0.08 + t * 0.125
        h = max(0.05, z_top - z_bot)
        s_mat = mats["crimson"] if i % 4 == 0 else (mats["cobalt"] if i % 4 == 2 else mats["chrome"])
        add_cyl(col, f"{p}_HarpStr_{i}", 0.0028, 0.0028, h, (x, y_c - 0.018, 0.5 * (z_top + z_bot)), s_mat, segs=8)


# ---------------------------------------------------------------------------
# Family 3: Bowed Strings, Choir & Orchestra Hit (25..32)
# ---------------------------------------------------------------------------

def _build_bowed_string_common(col, p: str, mats: dict, body_mat, scale: float, y_c: float, z_c: float, rot_y: float = 0.0, chinrest: bool = True, endpin: bool = False, gear_tuners: bool = False, x_off: float = 0.0):
    c = math.cos(rot_y)
    s = math.sin(rot_y)
    def rpos(dx, dz):
        # Match Blender's +Y Euler rotation: x' = dx*cos + dz*sin, z' = -dx*sin + dz*cos
        return (x_off + dx * c + dz * s, y_c, z_c - dx * s + dz * c)

    add_figure8_body(col, f"{p}_Body", scale, scale, 0.048 * scale, rpos(-0.04 * scale, 0.0), body_mat, bevel=0.005, rot=(0.0, rot_y, 0.0))
    # True S-curve F-holes, tailpiece, bridge, ebony fingerboard, and sculpted scroll
    fh_y = y_c - 0.029 * scale
    px_f, _, pz_f = rpos(-0.03 * scale, 0.044 * scale)
    add_s_fhole(col, f"{p}_FholeU", px_f, fh_y, pz_f, 0.95 * scale, mats["ebony"], rot_z=rot_y)
    px_f2, _, pz_f2 = rpos(-0.03 * scale, -0.044 * scale)
    add_s_fhole(col, f"{p}_FholeL", px_f2, fh_y, pz_f2, 0.95 * scale, mats["ebony"], rot_z=rot_y + math.pi)

    bx, _, bz = rpos(-0.04 * scale, 0.0)
    add_box(col, f"{p}_Bridge", (0.016 * scale, 0.056 * scale, 0.075 * scale), (bx, y_c - 0.010 * scale, bz), mats["ivory"], bevel=0.0015, rot=(0.0, rot_y, 0.0))
    tx, _, tz = rpos(-0.18 * scale, 0.0)
    add_box(col, f"{p}_Tailpiece", (0.11 * scale, 0.052 * scale, 0.038 * scale), (tx, y_c - 0.008 * scale, tz), mats["ebony"], bevel=0.002, rot=(0.0, rot_y, 0.0))
    if chinrest:
        cx, _, cz = rpos(-0.20 * scale, -0.048 * scale)
        add_box(col, f"{p}_Chinrest", (0.085 * scale, 0.056 * scale, 0.068 * scale), (cx, y_c - 0.012 * scale, cz), mats["ebony"], bevel=0.005, rot=(0.0, rot_y, 0.0))
    nx, _, nz = rpos(0.20 * scale, 0.0)
    add_box(col, f"{p}_Neck", (0.28 * scale, 0.028 * scale, 0.030 * scale), (nx, y_c - 0.008 * scale, nz), mats["ebony"], bevel=0.002, rot=(0.0, rot_y, 0.0))
    sx, _, sz = rpos(0.36 * scale, 0.0)
    add_cyl(col, f"{p}_Pegbox", 0.018 * scale, 0.015 * scale, 0.065 * scale, (sx, y_c - 0.006 * scale, sz), body_mat, rot=(0.0, math.pi / 2 + rot_y, 0.0))
    scx, _, scz = rpos(0.39 * scale, 0.010 * scale)
    add_torus_arc(col, f"{p}_Scroll", 0.020 * scale, 0.009 * scale, (scx, y_c - 0.006 * scale, scz), body_mat, arc_deg=310.0, rot=(math.pi / 2, 0.0, 0.0))
    if gear_tuners:
        for i in range(4):
            gx, _, gz = rpos((0.32 + i * 0.018) * scale, 0.032 * scale)
            add_cyl(col, f"{p}_Gear_{i}", 0.009 * scale, 0.009 * scale, 0.018 * scale, (gx, y_c - 0.006 * scale, gz), mats["brass"])
    if endpin:
        epx, _, epz = rpos(-0.32 * scale, 0.0)
        add_cyl(col, f"{p}_Endpin", 0.006 * scale, 0.005 * scale, 0.14 * scale, (epx, y_c, epz), mats["chrome"], rot=(0.0, math.pi / 2 + rot_y, 0.0))
    # 4 taut strings from tailpiece to pegbox
    for idx in range(4):
        soff = (-0.012 + idx * 0.008) * scale
        st_x, _, st_z = rpos(0.08 * scale, soff)
        add_cyl(col, f"{p}_Str_{idx}", 0.0020 * scale, 0.0020 * scale, 0.46 * scale, (st_x, y_c - 0.036 * scale, st_z), mats["chrome"], segs=8, rot=(0.0, math.pi / 2 + rot_y, 0.0))


def build_violin(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["amber_wood"])
    add_stand_brackets(col, p, (-0.18, 0.08), 0.10, 0.015, 0.22, 0.10, mats["brass"], radius=0.009)
    _build_bowed_string_common(col, p, mats, mats["amber_wood"], scale=0.88, y_c=0.22, z_c=0.185, chinrest=True)
    # Horsehair Violin Bow with ivory frog angled cleanly across upper bout
    add_cyl(col, f"{p}_BowStick", 0.0045, 0.0045, 0.56, (0.04, 0.165, 0.23), mats["rosewood"], rot=(0.0, 1.18, 0.0))
    add_cyl(col, f"{p}_BowHair", 0.0030, 0.0030, 0.54, (0.04, 0.165, 0.222), mats["ivory"], rot=(0.0, 1.18, 0.0))


def build_viola(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["mahogany"])
    add_stand_brackets(col, p, (-0.18, 0.10), 0.10, 0.015, 0.22, 0.10, mats["brass"], radius=0.009)
    _build_bowed_string_common(col, p, mats, mats["mahogany"], scale=1.02, y_c=0.22, z_c=0.19, chinrest=True)
    add_cyl(col, f"{p}_BowStick", 0.0052, 0.0052, 0.58, (0.04, 0.162, 0.24), mats["walnut"], rot=(0.0, 1.15, 0.0))
    add_cyl(col, f"{p}_BowHair", 0.0034, 0.0034, 0.56, (0.04, 0.162, 0.231), mats["ivory"], rot=(0.0, 1.15, 0.0))


def build_cello(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["mahogany"])
    add_stand_brackets(col, p, (-0.08, 0.08), 0.10, 0.015, 0.22, 0.12, mats["brass"], radius=0.010)
    # Angled semi-upright orchestral Cello showcased cleanly above Core
    _build_bowed_string_common(col, p, mats, mats["mahogany"], scale=1.04, y_c=0.22, z_c=0.235, rot_y=-0.62, chinrest=False, endpin=True)
    add_cyl(col, f"{p}_CelloBow", 0.006, 0.006, 0.50, (-0.04, 0.16, 0.22), mats["rosewood"], rot=(0.0, 0.95, 0.0))


def build_contrabass(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    add_stand_brackets(col, p, (-0.08, 0.08), 0.10, 0.015, 0.22, 0.12, mats["brass"], radius=0.010)
    # Bold deep-bodied Contrabass angled upright on endpin cleanly above Core
    _build_bowed_string_common(col, p, mats, mats["walnut"], scale=1.14, y_c=0.225, z_c=0.250, rot_y=-0.70, chinrest=False, endpin=True, gear_tuners=True)
    add_cyl(col, f"{p}_BassBow", 0.0075, 0.0075, 0.50, (-0.05, 0.155, 0.225), mats["ebony"], rot=(0.0, 0.90, 0.0))


def build_string_ensemble(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["amber_wood"])
    add_stand_brackets(col, p, (-0.25, 0.0, 0.25), 0.10, 0.015, 0.22, 0.07, mats["brass"], radius=0.009)
    # Symphonic fan of 3 complete bowed upright instruments: Violin (left), Cello (center tall), Viola (right)
    _build_bowed_string_common(col, f"{p}_Vln", mats, mats["spruce"], scale=0.66, y_c=0.22, z_c=0.175, rot_y=-1.28, chinrest=True, x_off=-0.25)
    _build_bowed_string_common(col, f"{p}_Clo", mats, mats["mahogany"], scale=0.82, y_c=0.23, z_c=0.195, rot_y=-math.pi / 2, chinrest=False, endpin=True, x_off=0.0)
    _build_bowed_string_common(col, f"{p}_Vla", mats, mats["amber_wood"], scale=0.72, y_c=0.22, z_c=0.180, rot_y=-1.86, chinrest=True, x_off=0.25)
    # Crossed golden conductor bows framing the ensemble
    add_cyl(col, f"{p}_GoldBow1", 0.0045, 0.0045, 0.58, (-0.08, 0.16, 0.21), mats["brass"], rot=(0.0, 1.05, 0.0))
    add_cyl(col, f"{p}_GoldBow2", 0.0045, 0.0045, 0.58, (0.08, 0.16, 0.21), mats["brass"], rot=(0.0, -1.05, 0.0))


def build_choir_aahs(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["ivory"])
    add_stand_brackets(col, p, (-0.16, 0.16), 0.10, 0.015, 0.22, 0.08, mats["brass"], radius=0.009)
    # Elevated cathedral vocal rosette + golden pipe crown above Core
    y_c, z_c = 0.22, 0.195
    add_cyl(col, f"{p}_RosetteBase", 0.145, 0.145, 0.026, (0.0, y_c, z_c), mats["walnut"], segs=32, rot=(math.pi / 2, 0.0, 0.0))
    add_torus_arc(col, f"{p}_OuterHalo", 0.150, 0.013, (0.0, y_c - 0.016, z_c), mats["brass"], arc_deg=360.0)
    add_torus_arc(col, f"{p}_InnerHalo", 0.095, 0.010, (0.0, y_c - 0.020, z_c), mats["antique_brass"], arc_deg=360.0)
    add_cyl(col, f"{p}_VocalCore", 0.052, 0.052, 0.032, (0.0, y_c - 0.018, z_c), mats["pearl"], segs=24, rot=(math.pi / 2, 0.0, 0.0))
    for i in range(9):
        ang = math.radians(-64 + i * 16)
        h = 0.17 - abs(i - 4) * 0.016
        add_cyl(col, f"{p}_FanPipe_{i}", 0.0085, 0.011, h, (0.14 * math.sin(ang), y_c + 0.005, z_c + 0.10 + 0.05 * math.cos(ang)), mats["brass"], rot=(0.0, ang, 0.0))


def build_voice_oohs(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["pearl"])
    add_stand_brackets(col, p, (-0.08, 0.08), 0.10, 0.015, 0.22, 0.048, mats["chrome"], radius=0.009)
    y_c, z_c = 0.215, 0.225
    add_cyl(col, f"{p}_StandBase", 0.095, 0.095, 0.024, (0.0, y_c, 0.052), mats["synth_dark"])
    add_cyl(col, f"{p}_StandRod", 0.011, 0.011, 0.11, (0.0, y_c, 0.105), mats["chrome"])
    add_torus_arc(col, f"{p}_ShockRing", 0.105, 0.011, (0.0, y_c - 0.01, z_c), mats["copper"], arc_deg=360.0)
    for ang_deg in (45, 135, 225, 315):
        ang = math.radians(ang_deg)
        add_cyl(col, f"{p}_Strut_{ang_deg}", 0.0035, 0.0035, 0.052, (0.065 * math.cos(ang), y_c - 0.01, z_c + 0.065 * math.sin(ang)), mats["ebony"], rot=(0.0, -ang, 0.0))
    add_cyl(col, f"{p}_MicBody", 0.025, 0.025, 0.075, (0.0, y_c - 0.022, z_c - 0.025), mats["brushed_alu"])
    add_cyl(col, f"{p}_MicGrille", 0.025, 0.025, 0.062, (0.0, y_c - 0.022, z_c + 0.036), mats["brass"])
    add_cyl(col, f"{p}_MicBand", 0.028, 0.028, 0.008, (0.0, y_c - 0.022, z_c + 0.006), mats["copper"])
    add_torus_arc(col, f"{p}_WaveRing1", 0.145, 0.008, (0.0, y_c - 0.015, z_c), mats["brass"], arc_deg=360.0)
    add_torus_arc(col, f"{p}_WaveRing2", 0.182, 0.006, (0.0, y_c - 0.015, z_c), mats["antique_brass"], arc_deg=360.0)


def build_orchestra_hit(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["brass"])
    add_stand_brackets(col, p, (-0.23, 0.23), 0.10, 0.015, 0.22, 0.055, mats["steel_dark"], radius=0.010)
    # Full framed Tam-Tam Gong elevated clearly above Core using Lathed hammered dish
    y_c, z_c = 0.22, 0.205
    add_cyl(col, f"{p}_FrameLegL", 0.015, 0.015, 0.31, (-0.23, y_c, z_c), mats["steel_dark"])
    add_cyl(col, f"{p}_FrameLegR", 0.015, 0.015, 0.31, (0.23, y_c, z_c), mats["steel_dark"])
    add_box(col, f"{p}_FrameTop", (0.50, 0.036, 0.028), (0.0, y_c, z_c + 0.16), mats["steel_dark"], bevel=0.003)
    for rx in (-0.10, 0.10):
        add_cyl(col, f"{p}_Cord_{rx:.2f}", 0.0055, 0.0055, 0.07, (rx, y_c, z_c + 0.12), mats["crimson"])
    # Lathed concave Bronze Tam-Tam facing camera
    gong_prof = [(0.00, 0.016), (0.045, 0.022), (0.075, 0.008), (0.125, 0.004), (0.155, 0.010), (0.162, 0.026), (0.145, 0.026), (0.00, 0.000)]
    add_lathe(col, f"{p}_GongDish", gong_prof, (0.0, y_c + 0.01, z_c - 0.005), mats["bronze"], segs=36, rot=(-math.pi / 2, 0.0, 0.0))
    add_cyl(col, f"{p}_GongBoss", 0.045, 0.038, 0.018, (0.0, y_c - 0.012, z_c - 0.005), mats["brass"], segs=28, rot=(math.pi / 2, 0.0, 0.0))
    add_cyl(col, f"{p}_GongMallet", 0.007, 0.007, 0.26, (0.25, y_c - 0.035, z_c - 0.01), mats["walnut"], rot=(0.0, -0.45, 0.0))
    add_sphere(col, f"{p}_MalletFleece", 0.032, (0.20, y_c - 0.035, z_c + 0.10), mats["ivory"])


# ---------------------------------------------------------------------------
# Family 4: Brass & Saxophones (33..41)
# ---------------------------------------------------------------------------

def _brass_bell_profile(length: float, r_throat: float, r_rim: float):
    """Smooth exponential flare profile rotated via add_lathe from z=0 (throat) to z=length (rim)."""
    pts = []
    steps = 12
    for i in range(steps + 1):
        t = i / steps
        r = r_throat + (r_rim - r_throat) * (t ** 2.8)
        pts.append((r, t * length))
    # Rolled outer lip bead
    pts.append((r_rim + 0.007, length))
    pts.append((r_rim + 0.007, length - 0.010))
    pts.append((r_rim - 0.005, length - 0.012))
    return pts


def build_trumpet(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["brass"])
    add_stand_brackets(col, p, (-0.18, 0.14), 0.10, 0.015, 0.22, 0.11, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.19
    # Seamless leadpipe + tuning slide U-loop + flared bell + cup mouthpiece + 3 piston valves
    add_var_tube(col, f"{p}_LeadLoop", [
        (-0.36, y_c, z_c + 0.03, 0.012), (-0.18, y_c, z_c + 0.03, 0.013),
        (0.08, y_c, z_c + 0.03, 0.014), (0.14, y_c, z_c - 0.01, 0.014),
        (0.08, y_c, z_c - 0.05, 0.014), (-0.22, y_c, z_c - 0.05, 0.014),
        (-0.27, y_c, z_c - 0.025, 0.014), (-0.22, y_c, z_c + 0.005, 0.015), (0.10, y_c, z_c + 0.03, 0.018),
    ], mats["brass"], segs=14, substeps=4)
    add_lathe(col, f"{p}_Bell", _brass_bell_profile(0.24, 0.018, 0.095), (0.10, y_c, z_c + 0.03), mats["brass"], segs=28, rot=(0.0, math.pi / 2, 0.0))
    add_lathe(col, f"{p}_Mouthpiece", [(0.016, 0.0), (0.008, 0.025), (0.010, 0.06)], (-0.42, y_c, z_c + 0.03), mats["chrome"], segs=18, rot=(0.0, math.pi / 2, 0.0))
    for i in range(3):
        vx = -0.06 + i * 0.044
        add_cyl(col, f"{p}_Valve_{i}", 0.014, 0.014, 0.115, (vx, y_c, z_c - 0.005), mats["chrome"])
        add_cyl(col, f"{p}_ValveCap_{i}", 0.011, 0.011, 0.016, (vx, y_c, z_c + 0.060), mats["pearl"])


def build_trombone(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["brass"])
    add_stand_brackets(col, p, (-0.28, 0.04), 0.10, 0.015, 0.22, 0.155, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.19
    # Continuous chrome front U-slide + rear bell U-loop + flared bell + counterweight
    add_var_tube(col, f"{p}_OuterSlide", [
        (0.06, y_c, z_c + 0.035, 0.012), (-0.38, y_c, z_c + 0.035, 0.012),
        (-0.43, y_c, z_c, 0.012), (-0.38, y_c, z_c - 0.035, 0.012), (0.08, y_c, z_c - 0.035, 0.012),
    ], mats["chrome"], segs=14, substeps=4)
    add_var_tube(col, f"{p}_BellSection", [
        (-0.08, y_c, z_c + 0.035, 0.013), (-0.16, y_c, z_c + 0.055, 0.014),
        (-0.14, y_c, z_c + 0.085, 0.015), (0.14, y_c, z_c + 0.075, 0.020),
    ], mats["brass"], segs=14, substeps=4)
    add_lathe(col, f"{p}_Bell", _brass_bell_profile(0.26, 0.020, 0.120), (0.14, y_c, z_c + 0.075), mats["brass"], segs=30, rot=(0.0, math.pi / 2, 0.0))
    add_cyl(col, f"{p}_Counterweight", 0.026, 0.026, 0.022, (-0.15, y_c, z_c + 0.070), mats["bronze"], rot=(math.pi / 2, 0.0, 0.0))
    for bx in (-0.34, -0.02):
        add_cyl(col, f"{p}_Brace_{bx:.2f}", 0.006, 0.006, 0.072, (bx, y_c, z_c), mats["brass"])


def build_tuba(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["antique_brass"])
    add_stand_brackets(col, p, (-0.14, 0.10), 0.10, 0.015, 0.22, 0.07, mats["antique_brass"], radius=0.010)
    y_c, z_c = 0.22, 0.185
    # Monumental coiled upright Tuba bugle terminating in a huge lathed upward-flaring bell
    add_var_tube(col, f"{p}_OuterBugle", [
        (-0.18, y_c, z_c + 0.06, 0.016), (-0.23, y_c, z_c - 0.06, 0.022),
        (-0.04, y_c, z_c - 0.125, 0.032), (0.14, y_c, z_c - 0.06, 0.040),
        (0.15, y_c, z_c + 0.04, 0.048),
    ], mats["antique_brass"], segs=18, substeps=5)
    add_var_tube(col, f"{p}_InnerBranch", [
        (-0.12, y_c - 0.01, z_c + 0.04, 0.014), (-0.14, y_c - 0.01, z_c - 0.05, 0.018),
        (0.02, y_c - 0.01, z_c - 0.07, 0.022), (0.08, y_c - 0.01, z_c + 0.02, 0.026),
    ], mats["brass"], segs=14, substeps=4)
    add_lathe(col, f"{p}_BigBell", _brass_bell_profile(0.20, 0.048, 0.145), (0.15, y_c, z_c + 0.04), mats["brass"], segs=32, rot=(0.0, 0.12, 0.0))
    for i in range(4):
        add_cyl(col, f"{p}_Rotor_{i}", 0.013, 0.013, 0.028, (-0.10 + i * 0.034, y_c - 0.032, z_c + 0.01), mats["chrome"], rot=(math.pi / 2, 0.0, 0.0))


def build_muted_trumpet(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["copper"])
    add_stand_brackets(col, p, (-0.18, 0.14), 0.10, 0.015, 0.22, 0.11, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.19
    add_var_tube(col, f"{p}_LeadLoop", [
        (-0.37, y_c, z_c + 0.03, 0.012), (0.04, y_c, z_c + 0.03, 0.015),
        (0.10, y_c, z_c - 0.01, 0.014), (0.04, y_c, z_c - 0.045, 0.014),
        (-0.22, y_c, z_c - 0.045, 0.014), (-0.25, y_c, z_c - 0.01, 0.014),
        (-0.18, y_c, z_c + 0.02, 0.016), (0.05, y_c, z_c + 0.03, 0.018),
    ], mats["copper"], segs=14, substeps=4)
    add_lathe(col, f"{p}_Bell", _brass_bell_profile(0.21, 0.018, 0.088), (0.05, y_c, z_c + 0.03), mats["copper"], segs=28, rot=(0.0, math.pi / 2, 0.0))
    # Contoured silver aluminum Harmon / Wah-Wah mute seated cleanly in the bell mouth
    mute_prof = [(0.022, 0.0), (0.076, 0.055), (0.068, 0.095), (0.025, 0.115), (0.025, 0.140), (0.038, 0.145), (0.038, 0.155)]
    add_lathe(col, f"{p}_HarmonMute", mute_prof, (0.21, y_c, z_c + 0.03), mats["chrome"], segs=26, rot=(0.0, math.pi / 2, 0.0))
    for i in range(3):
        add_cyl(col, f"{p}_Valve_{i}", 0.013, 0.013, 0.105, (-0.11 + i * 0.042, y_c, z_c - 0.005), mats["brass"])


def build_french_horn(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["brass"])
    add_stand_brackets(col, p, (-0.14, 0.14), 0.10, 0.015, 0.22, 0.08, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.195
    # Multi-turn circular orchestral horn coils + wide flared horn bell pointing down-right
    pts_coil = []
    for k in range(25):
        ang = math.radians(30 + k * 24)
        rad = 0.135 - k * 0.0018
        pts_coil.append((-0.05 + rad * math.cos(ang), y_c + (k % 2) * 0.004, z_c + rad * math.sin(ang), 0.013 + k * 0.0007))
    add_var_tube(col, f"{p}_HornSpiral", pts_coil, mats["brass"], segs=14, substeps=3)
    add_lathe(col, f"{p}_WideBell", _brass_bell_profile(0.22, 0.030, 0.138), (0.06, y_c, z_c - 0.02), mats["brass"], segs=30, rot=(0.0, 1.35, 0.0))
    for i in range(3):
        add_box(col, f"{p}_Paddle_{i}", (0.024, 0.018, 0.052), (-0.09 + i * 0.036, y_c - 0.028, z_c), mats["chrome"], bevel=0.002)


def build_brass_section(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["brass"])
    y_c, z_c = 0.22, 0.195
    for idx, (ox, oz, r_bell, tilt, mat) in enumerate((
        (-0.24, 0.04, 0.072, -0.25, mats["brass"]),
        (0.00, 0.01, 0.092, -0.05, mats["copper"]),
        (0.24, -0.02, 0.105, 0.18, mats["antique_brass"]),
    )):
        ang = 0.72 + tilt
        # Match bell start tangent cleanly: bell points along (+sin(ang), 0, +cos(ang))
        dx, dz = math.sin(ang), math.cos(ang)
        bx, bz = ox + 0.02, z_c + oz + 0.01
        add_stand_brackets(col, f"{p}_B{idx}", (ox - 0.06,), 0.10, 0.015, y_c, bz - 0.05 * dz, mats["brass"], radius=0.008)
        add_var_tube(col, f"{p}_Lead_{idx}", [
            (bx - 0.18 * dx, y_c, bz - 0.18 * dz, 0.011),
            (bx - 0.08 * dx, y_c, bz - 0.08 * dz, 0.014),
            (bx, y_c, bz, 0.018),
        ], mat, segs=12, substeps=3)
        add_lathe(col, f"{p}_Bell_{idx}", _brass_bell_profile(0.16, 0.018, r_bell), (bx, y_c, bz), mat, segs=24, rot=(0.0, ang, 0.0))


def build_soprano_sax(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["brass"])
    add_stand_brackets(col, p, (-0.20, 0.18), 0.10, 0.015, 0.22, 0.11, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.19
    # Sleek straight conical Soprano Sax with true flared bell & black ebonite mouthpiece
    add_cyl(col, f"{p}_StraightBody", 0.012, 0.032, 0.50, (-0.04, y_c, z_c + 0.02), mats["brass"], rot=(0.0, math.pi / 2 + 0.08, 0.0))
    add_lathe(col, f"{p}_FlareBell", _brass_bell_profile(0.15, 0.032, 0.074), (0.21, y_c, z_c - 0.00), mats["brass"], segs=26, rot=(0.0, math.pi / 2 + 0.08, 0.0))
    add_cyl(col, f"{p}_Mouthpiece", 0.012, 0.007, 0.085, (-0.33, y_c, z_c + 0.044), mats["ebony"], rot=(0.0, math.pi / 2 + 0.08, 0.0))
    for i in range(6):
        tx = -0.18 + i * 0.068
        tz = z_c + 0.03 - i * 0.0055
        add_cyl(col, f"{p}_Pearl_{i}", 0.012, 0.012, 0.012, (tx, y_c - 0.025, tz), mats["pearl"], rot=(math.pi / 2, 0.0, 0.0))


def build_alto_sax(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["brass"])
    add_stand_brackets(col, p, (0.04,), 0.10, 0.015, 0.22, 0.085, mats["brass"], radius=0.010)
    y_c, z_c = 0.22, 0.195
    # One continuous seamless Alto Sax spline from mouthpiece neck -> body -> U-bow -> up into lathed bell
    add_var_tube(col, f"{p}_SaxTube", [
        (-0.22, y_c, z_c + 0.145, 0.010), (-0.13, y_c, z_c + 0.135, 0.013),
        (-0.08, y_c, z_c + 0.085, 0.017), (-0.04, y_c, z_c - 0.020, 0.025),
        (-0.01, y_c, z_c - 0.095, 0.030), (0.04, y_c, z_c - 0.120, 0.032),
        (0.09, y_c, z_c - 0.090, 0.034), (0.11, y_c, z_c - 0.010, 0.038),
    ], mats["brass"], segs=18, substeps=5)
    add_lathe(col, f"{p}_UpBell", _brass_bell_profile(0.14, 0.038, 0.092), (0.11, y_c, z_c - 0.010), mats["brass"], segs=28, rot=(0.0, 0.18, 0.0))
    add_cyl(col, f"{p}_Mouthpiece", 0.008, 0.012, 0.06, (-0.25, y_c, z_c + 0.148), mats["ebony"], rot=(0.0, math.pi / 2, 0.0))
    for i in range(5):
        add_cyl(col, f"{p}_Pearl_{i}", 0.011, 0.011, 0.012, (-0.075 + i * 0.014, y_c - 0.028, z_c + 0.06 - i * 0.032), mats["pearl"], rot=(math.pi / 2, 0.0, 0.0))


def build_tenor_sax(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["antique_brass"])
    add_stand_brackets(col, p, (0.05,), 0.10, 0.015, 0.22, 0.080, mats["brass"], radius=0.010)
    y_c, z_c = 0.22, 0.195
    # Distinct Tenor Sax with deep goose-neck S-crook + wider bore + larger lathed bell
    add_var_tube(col, f"{p}_TenorTube", [
        (-0.26, y_c, z_c + 0.125, 0.011), (-0.19, y_c, z_c + 0.155, 0.013),
        (-0.12, y_c, z_c + 0.145, 0.015), (-0.08, y_c, z_c + 0.080, 0.020),
        (-0.04, y_c, z_c - 0.025, 0.029), (-0.01, y_c, z_c - 0.105, 0.036),
        (0.05, y_c, z_c - 0.130, 0.038), (0.11, y_c, z_c - 0.095, 0.042),
        (0.14, y_c, z_c - 0.010, 0.045),
    ], mats["antique_brass"], segs=18, substeps=5)
    add_lathe(col, f"{p}_UpBell", _brass_bell_profile(0.16, 0.045, 0.108), (0.14, y_c, z_c - 0.010), mats["brass"], segs=28, rot=(0.0, 0.20, 0.0))
    add_cyl(col, f"{p}_Mouthpiece", 0.009, 0.013, 0.065, (-0.29, y_c, z_c + 0.120), mats["ebony"], rot=(0.0, math.pi / 2 + 0.2, 0.0))
    for i in range(5):
        add_cyl(col, f"{p}_Pearl_{i}", 0.012, 0.012, 0.012, (-0.072 + i * 0.015, y_c - 0.032, z_c + 0.05 - i * 0.034), mats["pearl"], rot=(math.pi / 2, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Family 5: Woodwinds & Flutes (42..52)
# ---------------------------------------------------------------------------

def build_oboe(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["ebony"])
    y_c, z_c, tilt = 0.22, 0.155, 0.14
    add_stand_brackets(col, f"{p}_L", (-0.20,), 0.10, 0.015, y_c, z_c + 0.20 * math.tan(tilt) - 0.01, mats["chrome"], radius=0.008)
    add_stand_brackets(col, f"{p}_R", (0.18,), 0.10, 0.015, y_c, z_c - 0.18 * math.tan(tilt) - 0.01, mats["chrome"], radius=0.008)
    # Slender conical grenadilla body + silver ring keys + lathed flared bell + double cane reed staple
    add_cyl(col, f"{p}_Body", 0.016, 0.025, 0.56, (-0.02, y_c, z_c), mats["ebony"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_lathe(col, f"{p}_Bell", _brass_bell_profile(0.11, 0.025, 0.048), (0.26, y_c, z_c - 0.039), mats["ebony"], segs=24, rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_BellRing", 0.050, 0.050, 0.014, (0.365, y_c, z_c - 0.054), mats["chrome"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_KeyRod", 0.005, 0.005, 0.44, (-0.01, y_c - 0.024, z_c), mats["chrome"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_StapleReed", 0.005, 0.009, 0.095, (-0.34, y_c, z_c + 0.045), mats["brass"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_box(col, f"{p}_CaneReed", (0.038, 0.007, 0.014), (-0.40, y_c, z_c + 0.054), mats["bamboo"], bevel=0.001, rot=(0.0, tilt, 0.0))
    for i in range(7):
        t = -0.20 + i * 0.068
        add_cyl(col, f"{p}_Key_{i}", 0.014, 0.014, 0.014, (t, y_c - 0.026, z_c - t * math.tan(tilt)), mats["chrome"], rot=(math.pi / 2, 0.0, 0.0))


def build_english_horn(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["ebony"])
    y_c, z_c, tilt = 0.22, 0.155, 0.12
    add_stand_brackets(col, f"{p}_L", (-0.20,), 0.10, 0.015, y_c, z_c + 0.20 * math.tan(tilt) - 0.01, mats["chrome"], radius=0.008)
    add_stand_brackets(col, f"{p}_R", (0.18,), 0.10, 0.015, y_c, z_c - 0.18 * math.tan(tilt) - 0.01, mats["chrome"], radius=0.008)
    # Slightly heavier conical bore + signature lathed pear-shaped Liebesfuss bell + curved silver S-bocal
    add_cyl(col, f"{p}_Body", 0.018, 0.028, 0.56, (-0.04, y_c, z_c), mats["ebony"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    pear_prof = [(0.028, 0.0), (0.048, 0.032), (0.054, 0.068), (0.036, 0.105), (0.040, 0.120)]
    add_lathe(col, f"{p}_PearBell", pear_prof, (0.24, y_c, z_c - 0.034), mats["ebony"], segs=26, rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_IvoryCollar", 0.036, 0.036, 0.016, (0.235, y_c, z_c - 0.033), mats["ivory"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_var_tube(col, f"{p}_Bocal", [(-0.31, y_c, z_c + 0.036, 0.007), (-0.37, y_c, z_c + 0.072, 0.006), (-0.43, y_c, z_c + 0.058, 0.005)], mats["chrome"], segs=12, substeps=4)
    for i in range(6):
        t = -0.20 + i * 0.072
        add_cyl(col, f"{p}_Key_{i}", 0.015, 0.015, 0.015, (t, y_c - 0.028, z_c - t * math.tan(tilt)), mats["chrome"], rot=(math.pi / 2, 0.0, 0.0))


def build_bassoon(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["mahogany"])
    y_c, z_c, tilt = 0.22, 0.175, 0.16
    add_stand_brackets(col, f"{p}_L", (-0.20,), 0.10, 0.015, y_c, z_c - 0.02 + 0.20 * math.tan(tilt), mats["brass"], radius=0.009)
    add_stand_brackets(col, f"{p}_R", (0.18,), 0.10, 0.015, y_c, z_c - 0.04 - 0.18 * math.tan(tilt), mats["brass"], radius=0.009)
    # Authentic twin parallel mahogany bores joined in rounded chrome boot + ivory bell crown + long silver S-bocal
    add_cyl(col, f"{p}_WingJoint", 0.020, 0.025, 0.56, (-0.02, y_c, z_c + 0.026), mats["mahogany"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_LongJoint", 0.024, 0.032, 0.66, (0.04, y_c, z_c - 0.032), mats["mahogany"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_var_tube(col, f"{p}_BootTurn", [
        (-0.28, y_c, z_c + 0.070, 0.021), (-0.34, y_c, z_c + 0.046, 0.028), (-0.28, y_c, z_c + 0.020, 0.024)
    ], mats["chrome"], segs=14, substeps=3)
    add_cyl(col, f"{p}_IvoryCrown", 0.038, 0.038, 0.036, (0.37, y_c, z_c - 0.085), mats["ivory"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_var_tube(col, f"{p}_SilverBocal", [(-0.28, y_c, z_c + 0.072, 0.007), (-0.36, y_c, z_c + 0.125, 0.006), (-0.43, y_c, z_c + 0.095, 0.005)], mats["chrome"], segs=12, substeps=4)
    for i in range(6):
        t = -0.16 + i * 0.085
        add_cyl(col, f"{p}_Key_{i}", 0.014, 0.014, 0.015, (t, y_c - 0.028, z_c - t * math.tan(tilt)), mats["chrome"], rot=(math.pi / 2, 0.0, 0.0))


def build_clarinet(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["ebony"])
    y_c, z_c, tilt = 0.22, 0.155, 0.14
    add_stand_brackets(col, f"{p}_L", (-0.20,), 0.10, 0.015, y_c, z_c + 0.20 * math.tan(tilt) - 0.01, mats["chrome"], radius=0.008)
    add_stand_brackets(col, f"{p}_R", (0.18,), 0.10, 0.015, y_c, z_c - 0.18 * math.tan(tilt) - 0.01, mats["chrome"], radius=0.008)
    # Cylindrical ebony body + silver tenon rings + tapered beak mouthpiece w/ brass ligature + true lathed flared bell
    add_cyl(col, f"{p}_CylBody", 0.022, 0.024, 0.56, (-0.02, y_c, z_c), mats["ebony"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_lathe(col, f"{p}_Bell", _brass_bell_profile(0.13, 0.025, 0.068), (0.25, y_c, z_c - 0.038), mats["ebony"], segs=26, rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_BellRing", 0.070, 0.070, 0.016, (0.375, y_c, z_c - 0.056), mats["chrome"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    for rx in (-0.23, 0.04, 0.25):
        add_cyl(col, f"{p}_Ring_{rx:.2f}", 0.027, 0.027, 0.014, (rx, y_c, z_c - rx * math.tan(tilt)), mats["chrome"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_Mouthpiece", 0.021, 0.010, 0.095, (-0.34, y_c, z_c + 0.045), mats["ebony"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_Ligature", 0.020, 0.020, 0.024, (-0.32, y_c, z_c + 0.042), mats["brass"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    for i in range(7):
        t = -0.18 + i * 0.066
        add_cyl(col, f"{p}_Key_{i}", 0.015, 0.015, 0.015, (t, y_c - 0.026, z_c - t * math.tan(tilt)), mats["chrome"], rot=(math.pi / 2, 0.0, 0.0))


def build_piccolo(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    add_stand_brackets(col, p, (-0.18, 0.18), 0.10, 0.015, 0.22, 0.09, mats["chrome"], radius=0.008)
    # Miniature crimson-velvet cradle holding half-size sterling silver head + dark ebony piccolo body
    y_c, z_c, tilt = 0.22, 0.135, -0.08
    add_box(col, f"{p}_VelvetTray", (0.64, 0.085, 0.028), (0.0, y_c, z_c - 0.036), mats["felt_red"], bevel=0.004)
    add_cyl(col, f"{p}_SilverHead", 0.015, 0.015, 0.16, (-0.18, y_c, z_c + 0.015), mats["chrome"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_box(col, f"{p}_LipPlate", (0.044, 0.022, 0.018), (-0.19, y_c - 0.018, z_c + 0.016), mats["chrome"], bevel=0.002)
    add_cyl(col, f"{p}_CorkTenon", 0.014, 0.014, 0.016, (-0.09, y_c, z_c + 0.008), mats["brass"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_EbonyBody", 0.014, 0.012, 0.34, (0.09, y_c, z_c - 0.008), mats["ebony"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    for i in range(6):
        t = -0.02 + i * 0.052
        add_cyl(col, f"{p}_Key_{i}", 0.011, 0.011, 0.013, (t, y_c - 0.018, z_c - t * math.tan(tilt)), mats["chrome"], rot=(math.pi / 2, 0.0, 0.0))


def build_flute(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    add_stand_brackets(col, p, (-0.25, 0.25), 0.10, 0.015, 0.22, 0.125, mats["chrome"], radius=0.008)
    y_c, z_c = 0.22, 0.145
    # Full-length silver Concert C Flute with B-footjoint, embossed lip plate & open-hole keys
    add_cyl(col, f"{p}_SilverTube", 0.018, 0.018, 0.84, (0.0, y_c, z_c), mats["chrome"], rot=(0.0, math.pi / 2, 0.0))
    add_box(col, f"{p}_LipPlate", (0.068, 0.028, 0.022), (-0.27, y_c - 0.020, z_c), mats["chrome"], bevel=0.003)
    add_cyl(col, f"{p}_Embouchure", 0.010, 0.010, 0.016, (-0.27, y_c - 0.024, z_c), mats["ebony"], rot=(math.pi / 2, 0.0, 0.0))
    add_cyl(col, f"{p}_Crown", 0.021, 0.021, 0.022, (-0.425, y_c, z_c), mats["chrome"], rot=(0.0, math.pi / 2, 0.0))
    for jx in (-0.16, 0.26):
        add_cyl(col, f"{p}_JointRing_{jx:.2f}", 0.021, 0.021, 0.018, (jx, y_c, z_c), mats["brushed_alu"], rot=(0.0, math.pi / 2, 0.0))
    add_cyl(col, f"{p}_KeyRod", 0.006, 0.006, 0.56, (0.09, y_c - 0.022, z_c), mats["brushed_alu"], rot=(0.0, math.pi / 2, 0.0))
    for i in range(8):
        t = -0.11 + i * 0.066
        add_cyl(col, f"{p}_KeyCup_{i}", 0.014, 0.014, 0.014, (t, y_c - 0.024, z_c), mats["pearl"], rot=(math.pi / 2, 0.0, 0.0))


def build_recorder(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    y_c, z_c, tilt = 0.22, 0.155, 0.12
    add_stand_brackets(col, f"{p}_L", (-0.18,), 0.10, 0.015, y_c, z_c + 0.18 * math.tan(tilt) - 0.01, mats["ivory"], radius=0.008)
    add_stand_brackets(col, f"{p}_R", (0.18,), 0.10, 0.015, y_c, z_c - 0.18 * math.tan(tilt) - 0.01, mats["ivory"], radius=0.008)
    # Warm pearwood Baroque Recorder with 3 ivory joint rings, slanted beak & rectangular labium windway
    add_cyl(col, f"{p}_TaperingBody", 0.027, 0.018, 0.62, (0.02, y_c, z_c), mats["amber_wood"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_Beak", 0.027, 0.011, 0.11, (-0.33, y_c, z_c + 0.042), mats["amber_wood"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_box(col, f"{p}_LabiumWindow", (0.032, 0.022, 0.016), (-0.25, y_c - 0.020, z_c + 0.030), mats["ebony"], bevel=0.001, rot=(0.0, tilt, 0.0))
    for rx in (-0.21, 0.16, 0.32):
        add_cyl(col, f"{p}_IvoryRing_{rx:.2f}", 0.031, 0.031, 0.018, (rx, y_c, z_c - rx * math.tan(tilt)), mats["ivory"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    for i in range(7):
        t = -0.12 + i * 0.056
        add_cyl(col, f"{p}_Hole_{i}", 0.0075, 0.0075, 0.022, (t, y_c - 0.024, z_c - t * math.tan(tilt)), mats["ebony"], rot=(math.pi / 2, 0.0, 0.0))


def build_pan_flute(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["bamboo"])
    add_stand_brackets(col, p, (-0.22, 0.22), 0.10, 0.015, 0.22, 0.08, mats["mahogany"], radius=0.009)
    y_c, z_c = 0.22, 0.175
    # Curved Romanian Syrinx panpipe tilted slightly forward so dark open beveled tops are unmistakably visible
    tilt_x = -0.24
    for i in range(10):
        t = (i - 4.5) / 4.5
        x = -0.29 + i * 0.064
        h = 0.26 - i * 0.017
        dy = -0.028 * (t * t)
        add_cyl(col, f"{p}_Pan_{i}", 0.020, 0.018, h, (x, y_c + dy, z_c - i * 0.008), mats["bamboo"], rot=(tilt_x, 0.0, 0.0))
        add_cyl(col, f"{p}_Bore_{i}", 0.013, 0.013, 0.015, (x, y_c + dy - math.sin(tilt_x) * (h * 0.49), z_c - i * 0.008 + math.cos(tilt_x) * (h * 0.49)), mats["ebony"], rot=(tilt_x, 0.0, 0.0))
    add_box(col, f"{p}_BindStrap", (0.65, 0.062, 0.036), (0.0, y_c - 0.010, z_c + 0.02), mats["mahogany"], bevel=0.005, rot=(tilt_x, 0.0, 0.0))


def build_shakuhachi(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["bamboo"])
    add_stand_brackets(col, p, (-0.22, 0.22), 0.10, 0.015, 0.22, 0.125, mats["bronze"], radius=0.008)
    y_c, z_c = 0.22, 0.155
    # Thick golden madake bamboo shaft with raised ring nodes + gnarled flared root bell + red lacquer utaguchi bevel
    add_cyl(col, f"{p}_BambooTube", 0.027, 0.033, 0.68, (-0.02, y_c, z_c), mats["bamboo"], rot=(0.0, math.pi / 2, 0.0))
    add_lathe(col, f"{p}_RootFlare", _brass_bell_profile(0.11, 0.033, 0.056), (0.32, y_c, z_c), mats["bamboo"], segs=22, rot=(0.0, math.pi / 2, 0.0))
    add_cyl(col, f"{p}_UtaguchiLip", 0.028, 0.022, 0.025, (-0.37, y_c, z_c), mats["crimson"], rot=(0.0, math.pi / 2, 0.0))
    for nx in (-0.20, 0.02, 0.22):
        add_cyl(col, f"{p}_Node_{nx:.2f}", 0.036, 0.036, 0.020, (nx, y_c, z_c), mats["bronze"], rot=(0.0, math.pi / 2, 0.0))
    for i in range(4):
        add_cyl(col, f"{p}_ToneHole_{i}", 0.010, 0.010, 0.036, (-0.08 + i * 0.078, y_c - 0.024, z_c), mats["ebony"], rot=(math.pi / 2, 0.0, 0.0))


def build_whistle(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["brushed_alu"])
    add_stand_brackets(col, p, (-0.22, 0.22), 0.10, 0.015, 0.22, 0.115, mats["brass"], radius=0.008)
    y_c, z_c = 0.22, 0.135
    add_cyl(col, f"{p}_TinTube", 0.017, 0.016, 0.64, (0.04, y_c, z_c), mats["chrome"], rot=(0.0, math.pi / 2, 0.0))
    add_cyl(col, f"{p}_FippleBody", 0.021, 0.014, 0.145, (-0.32, y_c, z_c), mats["cobalt"], rot=(0.0, math.pi / 2, 0.0))
    add_box(col, f"{p}_FippleWindow", (0.022, 0.024, 0.016), (-0.265, y_c - 0.016, z_c), mats["ebony"], bevel=0.001)
    for i in range(6):
        add_cyl(col, f"{p}_Hole_{i}", 0.0075, 0.0075, 0.022, (-0.08 + i * 0.065, y_c - 0.016, z_c), mats["ebony"], rot=(math.pi / 2, 0.0, 0.0))


def build_ocarina(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["cobalt"])
    add_stand_brackets(col, p, (-0.10, 0.10), 0.10, 0.015, 0.22, 0.10, mats["ivory"], radius=0.008)
    y_c, z_c = 0.22, 0.185
    # Sculpted glazed ceramic 12-hole sweet-potato Ocarina + smooth tapered thumb spout + tone holes
    add_sphere(col, f"{p}_CeramicBody", 0.125, (-0.04, y_c, z_c), mats["cobalt"], scale=(1.58, 0.52, 0.92), rot=(0.25, 0.0, 0.0))
    add_var_tube(col, f"{p}_MouthSpout", [
        (0.05, y_c, z_c + 0.03, 0.042), (0.14, y_c - 0.015, z_c + 0.055, 0.030), (0.23, y_c - 0.028, z_c + 0.075, 0.022)
    ], mats["cobalt"], segs=16, substeps=3)
    add_cyl(col, f"{p}_IvoryLip", 0.024, 0.024, 0.018, (0.23, y_c - 0.028, z_c + 0.075), mats["ivory"], rot=(0.0, math.pi / 2 - 0.24, 0.0))
    for hx, hz in ((-0.13, 0.022), (-0.07, 0.032), (-0.01, 0.026), (-0.10, -0.028), (-0.03, -0.028)):
        add_cyl(col, f"{p}_Hole_{hx:.2f}", 0.013, 0.013, 0.028, (hx, y_c - 0.058, z_c + hz), mats["ebony"], rot=(math.pi / 2 + 0.25, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Family 6: World Ethnic & Synth (53..64)
# ---------------------------------------------------------------------------

def build_sitar(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["amber_wood"])
    add_stand_brackets(col, f"{p}_L", (-0.22,), 0.10, 0.015, 0.22, 0.10, mats["brass"], radius=0.009)
    add_stand_brackets(col, f"{p}_R", (0.18,), 0.10, 0.015, 0.22, 0.185, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.185
    # Unobstructed ribbed amber kaddu gourd + floral ivory inlay + broad teak neck w/ raised brass pardas + secondary upper gourd
    add_sphere(col, f"{p}_MainGourd", 0.145, (-0.20, y_c, z_c - 0.01), mats["amber_wood"], scale=(1.15, 0.55, 0.92))
    for i in range(4):
        add_cyl(col, f"{p}_Rib_{i}", 0.008, 0.008, 0.24, (-0.20, y_c - 0.045, z_c + 0.025 - i * 0.024), mats["walnut"], rot=(0.0, math.pi / 2, 0.0))
    add_cyl(col, f"{p}_IvoryInlay", 0.075, 0.075, 0.02, (-0.20, y_c - 0.055, z_c - 0.01), mats["ivory"], segs=24, rot=(math.pi / 2, 0.0, 0.0))
    add_box(col, f"{p}_SitarNeck", (0.54, 0.034, 0.052), (0.12, y_c, z_c + 0.015), mats["walnut"], bevel=0.004)
    add_sphere(col, f"{p}_TopGourd", 0.072, (0.34, y_c + 0.025, z_c - 0.06), mats["rosewood"], scale=(1.0, 0.55, 1.0))
    # Iconic arched brass sitar frets (pardas)
    for i in range(8):
        fx = -0.05 + i * 0.055
        add_torus_arc(col, f"{p}_Parda_{i}", 0.032, 0.004, (fx, y_c - 0.015, z_c + 0.015), mats["brass"], arc_deg=140.0, rot=(0.0, math.pi / 2, math.pi / 2))
    for i in range(5):
        add_cyl(col, f"{p}_Peg_{i}", 0.008, 0.012, 0.055, (0.14 + i * 0.055, y_c - 0.01, z_c + 0.060), mats["ivory"])
    for idx in range(4):
        add_cyl(col, f"{p}_Str_{idx}", 0.0020, 0.0020, 0.62, (0.04, y_c - 0.032, z_c + 0.002 + idx * 0.008), mats["chrome"], segs=8, rot=(0.0, math.pi / 2, 0.0))


def build_banjo(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    add_stand_brackets(col, f"{p}_L", (-0.16,), 0.10, 0.015, 0.22, 0.08, mats["chrome"], radius=0.009)
    add_stand_brackets(col, f"{p}_R", (0.14,), 0.10, 0.015, 0.22, 0.185, mats["chrome"], radius=0.009)
    y_c, z_c = 0.22, 0.19
    add_cyl(col, f"{p}_ChromeRim", 0.155, 0.155, 0.048, (-0.12, y_c, z_c), mats["chrome"], segs=32, rot=(math.pi / 2, 0.0, 0.0))
    add_cyl(col, f"{p}_VellumHead", 0.138, 0.138, 0.054, (-0.12, y_c - 0.005, z_c), mats["ivory"], segs=32, rot=(math.pi / 2, 0.0, 0.0))
    add_box(col, f"{p}_Tailpiece", (0.045, 0.058, 0.038), (-0.27, y_c - 0.006, z_c), mats["chrome"], bevel=0.002)
    add_box(col, f"{p}_MapleBridge", (0.016, 0.060, 0.075), (-0.14, y_c - 0.010, z_c), mats["spruce"], bevel=0.002)
    for bi in range(10):
        bang = math.radians(bi * 36)
        add_cyl(col, f"{p}_Bracket_{bi}", 0.0055, 0.0055, 0.046, (-0.12 + 0.155 * math.cos(bang), y_c, z_c + 0.155 * math.sin(bang)), mats["chrome"])
    add_box(col, f"{p}_Neck", (0.42, 0.028, 0.036), (0.22, y_c - 0.008, z_c), mats["walnut"], bevel=0.003)
    add_box(col, f"{p}_Headstock", (0.095, 0.026, 0.054), (0.46, y_c - 0.008, z_c), mats["walnut"], bevel=0.003)
    add_cyl(col, f"{p}_FifthPeg", 0.009, 0.009, 0.034, (0.14, y_c - 0.008, z_c + 0.030), mats["ivory"])
    _add_guitar_frets_and_strings(col, p, y_c - 0.026, z_c, -0.25, 0.43, 5, mats["chrome"], mats["brass"], span_z=0.030)


def build_shamisen(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["ebony"])
    add_stand_brackets(col, f"{p}_L", (-0.20,), 0.10, 0.015, 0.22, 0.10, mats["ivory"], radius=0.009)
    add_stand_brackets(col, f"{p}_R", (0.14,), 0.10, 0.015, 0.22, 0.180, mats["ivory"], radius=0.009)
    y_c, z_c = 0.22, 0.185
    # Red Karin square-rounded dou body + crisp white stretched skin + 3 silk strings + ivory bachi plectrum
    add_box(col, f"{p}_DouBody", (0.22, 0.070, 0.20), (-0.19, y_c, z_c), mats["rosewood"], bevel=0.020)
    add_box(col, f"{p}_SkinFace", (0.19, 0.074, 0.17), (-0.19, y_c - 0.004, z_c), mats["ivory"], bevel=0.010)
    add_box(col, f"{p}_DoKake", (0.10, 0.078, 0.10), (-0.23, y_c - 0.008, z_c - 0.04), mats["crimson"], bevel=0.006)
    add_box(col, f"{p}_KomaBridge", (0.014, 0.040, 0.072), (-0.15, y_c - 0.024, z_c), mats["ivory"], bevel=0.002)
    add_box(col, f"{p}_Neck", (0.48, 0.024, 0.030), (0.15, y_c - 0.006, z_c), mats["ebony"], bevel=0.002)
    add_box(col, f"{p}_Tenjin", (0.085, 0.024, 0.040), (0.42, y_c - 0.006, z_c), mats["ebony"], bevel=0.003)
    for i, px in enumerate((0.38, 0.415, 0.45)):
        add_cyl(col, f"{p}_Itomaki_{i}", 0.009, 0.017, 0.085, (px, y_c - 0.006, z_c + (0.048 if i != 1 else -0.048)), mats["ivory"])
    for idx in range(3):
        add_cyl(col, f"{p}_SilkStr_{idx}", 0.0025, 0.0025, 0.64, (0.06, y_c - 0.032, z_c + (idx - 1) * 0.009), mats["amber_wood"], segs=8, rot=(0.0, math.pi / 2, 0.0))
    add_extruded_profile(col, f"{p}_Bachi", [(-0.055, 0.0), (0.055, 0.0), (0.016, 0.085), (-0.016, 0.085)], 0.014, (-0.18, y_c - 0.048, z_c + 0.075), mats["ivory"], bevel=0.002, smooth=False)


def build_koto(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["spruce"])
    add_stand_brackets(col, p, (-0.30, 0.30), 0.10, 0.015, 0.22, 0.09, mats["rosewood"], radius=0.009)
    y_c, z_c = 0.22, 0.16
    tilt = -0.32
    add_box(col, f"{p}_KotoBody", (0.86, 0.085, 0.16), (0.0, y_c, z_c), mats["spruce"], bevel=0.014, rot=(tilt, 0.0, 0.0))
    add_box(col, f"{p}_EndTrimL", (0.05, 0.092, 0.168), (-0.41, y_c, z_c), mats["rosewood"], bevel=0.004, rot=(tilt, 0.0, 0.0))
    add_box(col, f"{p}_EndTrimR", (0.05, 0.092, 0.168), (0.41, y_c, z_c), mats["rosewood"], bevel=0.004, rot=(tilt, 0.0, 0.0))
    for s in range(7):
        dz = 0.050 - s * 0.0165
        add_cyl(col, f"{p}_KotoStr_{s}", 0.0024, 0.0024, 0.78, (0.0, y_c - 0.048, z_c + dz), mats["chrome"], segs=8, rot=(0.0, math.pi / 2, 0.0))
    for i in range(9):
        x = -0.32 + i * 0.08
        dz = 0.045 - i * 0.011
        add_box(col, f"{p}_JiBridge_{i}", (0.014, 0.026, 0.030), (x, y_c - 0.046, z_c + dz), mats["ivory"], bevel=0.002, rot=(tilt, 0.0, 0.0))


def build_kalimba(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    add_stand_brackets(col, p, (-0.18, 0.18), 0.10, 0.015, 0.22, 0.09, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.18
    tilt = -0.14
    add_box(col, f"{p}_Soundbox", (0.52, 0.072, 0.26), (0.0, y_c, z_c), mats["walnut"], bevel=0.022, rot=(tilt, 0.0, 0.0))
    add_cyl(col, f"{p}_Rosette", 0.056, 0.056, 0.075, (0.0, y_c - 0.005, z_c - 0.035), mats["amber_wood"], segs=24, rot=(math.pi / 2 + tilt, 0.0, 0.0))
    add_cyl(col, f"{p}_Soundhole", 0.040, 0.040, 0.078, (0.0, y_c - 0.008, z_c - 0.035), mats["ebony"], segs=24, rot=(math.pi / 2 + tilt, 0.0, 0.0))
    add_box(col, f"{p}_BridgeBar", (0.46, 0.024, 0.020), (0.0, y_c - 0.044, z_c + 0.095), mats["brass"], bevel=0.002, rot=(tilt, 0.0, 0.0))
    for i in range(11):
        x = -0.20 + i * 0.040
        h = 0.165 - abs(i - 5) * 0.014
        add_box(col, f"{p}_Tine_{i}", (0.018, 0.010, h), (x, y_c - 0.048, z_c + 0.09 - h * 0.45), mats["chrome"], bevel=0.002, rot=(tilt, 0.0, 0.0))


def build_bagpipe(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["walnut"])
    add_stand_brackets(col, p, (-0.16, 0.14), 0.10, 0.015, 0.22, 0.09, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.165
    # Scottish tartan velvet oval bag + brass ribs + 3 upright ebony drones w/ silver slides & ivory mounts
    add_sphere(col, f"{p}_TartanBag", 0.135, (0.0, y_c, z_c), mats["crimson"], scale=(1.28, 0.55, 0.80))
    for rx in (-0.07, 0.07):
        add_torus_arc(col, f"{p}_BagRib_{rx:.2f}", 0.10, 0.009, (rx, y_c, z_c), mats["brass"], arc_deg=360.0)
    for i, (dx, h) in enumerate(((-0.11, 0.24), (0.0, 0.21), (0.11, 0.18))):
        add_cyl(col, f"{p}_Stock_{i}", 0.020, 0.024, 0.04, (dx, y_c, z_c + 0.075), mats["walnut"])
        add_cyl(col, f"{p}_Drone_{i}", 0.013, 0.017, h, (dx, y_c, z_c + 0.08 + h * 0.5), mats["ebony"])
        add_cyl(col, f"{p}_Slide_{i}", 0.016, 0.016, 0.024, (dx, y_c, z_c + 0.08 + h * 0.55), mats["chrome"])
        add_cyl(col, f"{p}_DroneCap_{i}", 0.023, 0.023, 0.024, (dx, y_c, z_c + 0.08 + h), mats["ivory"])
    # Continuous blowpipe on right
    add_cyl(col, f"{p}_Blowpipe", 0.010, 0.010, 0.17, (0.16, y_c - 0.03, z_c + 0.06), mats["ebony"], rot=(0.40, 0.35, 0.0))
    add_cyl(col, f"{p}_BlowTip", 0.012, 0.008, 0.03, (0.19, y_c - 0.055, z_c + 0.125), mats["ivory"], rot=(0.40, 0.35, 0.0))
    # Joined Chanter stock + pipe + flared ivory sole seamlessly aligned along the same bore axis!
    ang_c = 0.42
    c_len = 0.24
    cx0, cz0 = -0.14, z_c + 0.02
    add_cyl(col, f"{p}_Chanter", 0.013, 0.022, c_len, (cx0 - 0.5 * c_len * math.sin(ang_c), y_c - 0.01, cz0 - 0.5 * c_len * math.cos(ang_c)), mats["ebony"], rot=(0.0, ang_c, 0.0))
    add_lathe(col, f"{p}_ChanterSole", [(0.022, 0.0), (0.042, 0.038), (0.038, 0.042)], (cx0 - c_len * math.sin(ang_c), y_c - 0.01, cz0 - c_len * math.cos(ang_c)), mats["ivory"], segs=22, rot=(0.0, ang_c - math.pi, 0.0))


def build_shanai(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["rosewood"])
    y_c, z_c, tilt = 0.22, 0.155, 0.12
    add_stand_brackets(col, f"{p}_L", (-0.18,), 0.10, 0.015, y_c, z_c + 0.18 * math.tan(tilt) - 0.01, mats["brass"], radius=0.008)
    add_stand_brackets(col, f"{p}_R", (0.14,), 0.10, 0.015, y_c, z_c - 0.14 * math.tan(tilt) - 0.01, mats["brass"], radius=0.008)
    # Rosewood conical shanai + 3 brass ferrules + single smooth lathed brass pagoda bell + lip pirouette
    add_cyl(col, f"{p}_WoodCone", 0.015, 0.034, 0.50, (-0.06, y_c, z_c), mats["rosewood"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    for rx in (-0.18, -0.06, 0.06):
        add_cyl(col, f"{p}_Band_{rx:.2f}", 0.030, 0.030, 0.014, (rx, y_c, z_c - rx * math.tan(tilt)), mats["brass"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_lathe(col, f"{p}_BrassBell", _brass_bell_profile(0.18, 0.034, 0.102), (0.19, y_c, z_c - 0.030), mats["brass"], segs=28, rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_MouthDisc", 0.036, 0.036, 0.014, (-0.33, y_c, z_c + 0.032), mats["ivory"], rot=(0.0, math.pi / 2 + tilt, 0.0))
    add_cyl(col, f"{p}_CrimsonCord", 0.020, 0.020, 0.032, (-0.29, y_c, z_c + 0.028), mats["crimson"], rot=(0.0, math.pi / 2 + tilt, 0.0))


def build_steel_drums(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["chrome"])
    add_stand_brackets(col, p, (-0.13, 0.13), 0.10, 0.015, 0.22, 0.145, mats["chrome"], radius=0.009)
    y_c, z_c = 0.22, 0.185
    tilt_x = -0.58  # Tilted directly down-forward so the viewer gazes right into the dished steelpan petals!
    # Lathed steelpan: outer rolled steel skirt + deeply concaved inward-hammered chrome bowl
    pan_prof = [
        (0.00, -0.018), (0.045, -0.016), (0.115, -0.006), (0.165, 0.018),
        (0.180, 0.035), (0.186, 0.035), (0.186, -0.055), (0.175, -0.055), (0.175, 0.020),
    ]
    add_lathe(col, f"{p}_SteelpanDish", pan_prof, (0.0, y_c, z_c), mats["chrome"], segs=36, rot=(math.pi / 2 + tilt_x, 0.0, 0.0))
    # Concentric dark note grooves hammered inside the concave dish
    add_torus_arc(col, f"{p}_CenterGroove", 0.042, 0.0035, (0.0, y_c - 0.012, z_c), mats["steel_dark"], arc_deg=360.0, rot=(tilt_x, 0.0, 0.0))
    for i in range(8):
        ang = math.radians(i * 45)
        nx = 0.105 * math.cos(ang)
        dz = 0.105 * math.sin(ang) * math.cos(tilt_x)
        dy = -0.105 * math.sin(ang) * math.sin(tilt_x)
        add_torus_arc(col, f"{p}_Petal_{i}", 0.028, 0.0030, (nx, y_c + dy - 0.004, z_c + dz), mats["steel_dark"], arc_deg=360.0, rot=(tilt_x, 0.0, 0.0))


def build_woodblock(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["rosewood"])
    add_stand_brackets(col, p, (-0.16, 0.16), 0.10, 0.015, 0.22, 0.10, mats["brass"], radius=0.009)
    y_c, z_c = 0.22, 0.15
    add_cyl(col, f"{p}_BlockLow", 0.068, 0.068, 0.28, (-0.16, y_c, z_c), mats["walnut"], segs=24, rot=(0.0, math.pi / 2, 0.0))
    add_cyl(col, f"{p}_BlockHigh", 0.056, 0.056, 0.22, (0.16, y_c, z_c - 0.008), mats["rosewood"], segs=24, rot=(0.0, math.pi / 2, 0.0))
    add_box(col, f"{p}_SlitL", (0.20, 0.08, 0.022), (-0.16, y_c - 0.03, z_c), mats["ebony"], bevel=0.002)
    add_box(col, f"{p}_SlitH", (0.16, 0.08, 0.020), (0.16, y_c - 0.03, z_c - 0.008), mats["ebony"], bevel=0.002)
    add_box(col, f"{p}_Bridge", (0.16, 0.03, 0.025), (0.0, y_c, z_c - 0.004), mats["brass"], bevel=0.002)
    add_cyl(col, f"{p}_Mallet1", 0.007, 0.007, 0.24, (0.33, y_c - 0.04, z_c + 0.03), mats["spruce"], rot=(0.0, -0.55, 0.0))
    add_sphere(col, f"{p}_MalletHead", 0.018, (0.267, y_c - 0.04, z_c + 0.132), mats["ivory"])


def _add_synth_controls(col, p: str, mats: dict, accent_mat: bpy.types.Material):
    # Bracket + dark glass OLED bezel: zero overlap between Pitch/Mod wheels (left), Waveform display (center), VU ladder (right), and Jacks/Knobs (bottom apron)
    add_stand_brackets(col, p, (-0.28, 0.28), 0.10, 0.015, 0.22, 0.06, mats["brushed_alu"], radius=0.009)
    add_box(col, f"{p}_ScreenGlass", (0.50, 0.015, 0.145), (-0.01, 0.184, 0.190), mats["ebony"], bevel=0.003)
    add_box(col, f"{p}_RackEarL", (0.036, 0.068, 0.24), (-0.42, 0.22, 0.165), mats["brushed_alu"], bevel=0.003)
    add_box(col, f"{p}_RackEarR", (0.036, 0.068, 0.24), (0.42, 0.22, 0.165), mats["brushed_alu"], bevel=0.003)
    for ez in (0.08, 0.25):
        add_cyl(col, f"{p}_BoltL_{ez:.2f}", 0.0075, 0.0075, 0.02, (-0.42, 0.18, ez), mats["chrome"], rot=(math.pi / 2, 0.0, 0.0))
        add_cyl(col, f"{p}_BoltR_{ez:.2f}", 0.0075, 0.0075, 0.02, (0.42, 0.18, ez), mats["chrome"], rot=(math.pi / 2, 0.0, 0.0))
    # Left bezel: Pitch & Mod wheels cleanly isolated from OLED screen
    add_box(col, f"{p}_PitchWheel", (0.024, 0.036, 0.065), (-0.365, 0.178, 0.19), mats["chrome"], bevel=0.003)
    add_box(col, f"{p}_ModWheel", (0.024, 0.036, 0.065), (-0.320, 0.178, 0.19), accent_mat, bevel=0.003)
    # Right bezel: 6-segment LED VU Meter ladder cleanly outside OLED screen
    vu_mats = [mats["led_emerald"], mats["led_emerald"], mats["led_emerald"], mats["led_yellow"], mats["led_yellow"], mats["led_red"]]
    for vi, vmat in enumerate(vu_mats):
        vz = 0.125 + vi * 0.024
        add_box(col, f"{p}_VU_{vi}", (0.022, 0.016, 0.016), (0.335, 0.180, vz), vmat, bevel=0.001)
    # Bottom control strip: 4 Patch Jacks on left + 4 Rotary Knobs on right
    for ji in range(4):
        jx = -0.22 + ji * 0.052
        add_cyl(col, f"{p}_Jack_{ji}", 0.011, 0.011, 0.015, (jx, 0.178, 0.082), mats["chrome"], rot=(math.pi / 2, 0.0, 0.0))
        add_cyl(col, f"{p}_JackHole_{ji}", 0.006, 0.006, 0.018, (jx, 0.176, 0.082), mats["ebony"], rot=(math.pi / 2, 0.0, 0.0))
    for i in range(4):
        kx = 0.05 + i * 0.072
        add_cyl(col, f"{p}_Knob_{i}", 0.014, 0.014, 0.022, (kx, 0.175, 0.082), mats["chrome"], rot=(math.pi / 2, 0.0, 0.0))


def build_square_lead(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["synth_dark"])
    add_box(col, f"{p}_Chassis", (0.80, 0.065, 0.24), (0.0, 0.22, 0.165), mats["synth_dark"], bevel=0.006)
    _add_synth_controls(col, p, mats, mats["led_emerald"])
    for i in range(5):
        x = -0.20 + i * 0.095
        z = 0.225 if i % 2 == 0 else 0.155
        add_box(col, f"{p}_SqStep_{i}", (0.092, 0.020, 0.020), (x, 0.174, z), mats["led_emerald"], bevel=0.002)
        if i < 4:
            add_box(col, f"{p}_SqVert_{i}", (0.020, 0.020, 0.075), (x + 0.046, 0.174, 0.190), mats["led_emerald"], bevel=0.002)


def build_sawtooth_lead(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["synth_dark"])
    add_box(col, f"{p}_Chassis", (0.80, 0.065, 0.24), (0.0, 0.22, 0.165), mats["synth_dark"], bevel=0.006)
    _add_synth_controls(col, p, mats, mats["led_amber"])
    for i in range(4):
        x = -0.19 + i * 0.125
        add_box(col, f"{p}_SawRamp_{i}", (0.115, 0.020, 0.018), (x, 0.174, 0.190), mats["led_amber"], bevel=0.002, rot=(0.0, -0.58, 0.0))
        add_box(col, f"{p}_SawDrop_{i}", (0.016, 0.020, 0.082), (x + 0.052, 0.174, 0.190), mats["led_amber"], bevel=0.002)


def build_new_age_pad(col, p: str, mats: dict, _v: str):
    build_core_bar(col, p, mats["synth_dark"])
    add_box(col, f"{p}_Chassis", (0.80, 0.065, 0.24), (0.0, 0.22, 0.165), mats["synth_dark"], bevel=0.006)
    _add_synth_controls(col, p, mats, mats["led_cyan"])
    add_torus_arc(col, f"{p}_PadOuter", 0.062, 0.009, (-0.01, 0.174, 0.190), mats["led_cyan"], arc_deg=360.0)
    add_torus_arc(col, f"{p}_PadInner", 0.036, 0.008, (-0.01, 0.172, 0.190), mats["led_emerald"], arc_deg=360.0)
    add_sphere(col, f"{p}_PadCore", 0.016, (-0.01, 0.170, 0.190), mats["led_cyan"])


# ---------------------------------------------------------------------------
# Full 64-Instrument Catalog (1-to-1 bespoke builders)
# ---------------------------------------------------------------------------

CATALOG_64 = [
    # Family 1 (01..12)
    (1, 0, "Acoustic Grand Piano", 1, "Pianos & Mallets", build_grand_piano, "grand"),
    (2, 2, "Electric Grand Piano", 1, "Pianos & Mallets", build_electric_grand, "egrand"),
    (3, 4, "Electric Piano", 1, "Pianos & Mallets", build_electric_piano, "epiano"),
    (4, 6, "Harpsichord", 1, "Pianos & Mallets", build_harpsichord, "harpsichord"),
    (5, 8, "Celesta", 1, "Pianos & Mallets", build_celesta, "celesta"),
    (6, 9, "Glockenspiel", 1, "Pianos & Mallets", build_glockenspiel, "glock"),
    (7, 10, "Music Box", 1, "Pianos & Mallets", build_music_box, "musicbox"),
    (8, 11, "Vibraphone", 1, "Pianos & Mallets", build_vibraphone, "vibe"),
    (9, 12, "Marimba", 1, "Pianos & Mallets", build_marimba, "marimba"),
    (10, 13, "Xylophone", 1, "Pianos & Mallets", build_xylophone, "xylo"),
    (11, 14, "Tubular Bells", 1, "Pianos & Mallets", build_tubular_bells, "bells"),
    (12, 15, "Dulcimer", 1, "Pianos & Mallets", build_dulcimer, "dulcimer"),

    # Family 2 (13..24)
    (13, 16, "Drawbar Organ", 2, "Organs, Guitars & Harp", build_drawbar_organ, "drawbar"),
    (14, 19, "Church Organ", 2, "Organs, Guitars & Harp", build_church_organ, "church"),
    (15, 21, "Accordion", 2, "Organs, Guitars & Harp", build_accordion, "accordion"),
    (16, 22, "Harmonica", 2, "Organs, Guitars & Harp", build_harmonica, "harmonica"),
    (17, 24, "Nylon Guitar", 2, "Organs, Guitars & Harp", build_nylon_guitar, "nylon"),
    (18, 25, "Steel Guitar", 2, "Organs, Guitars & Harp", build_steel_guitar, "steel"),
    (19, 27, "Clean Electric Guitar", 2, "Organs, Guitars & Harp", build_clean_electric_guitar, "clean_e"),
    (20, 30, "Distortion Guitar", 2, "Organs, Guitars & Harp", build_distortion_guitar, "dist_e"),
    (21, 32, "Acoustic Bass", 2, "Organs, Guitars & Harp", build_acoustic_bass, "abass"),
    (22, 35, "Fretless Bass", 2, "Organs, Guitars & Harp", build_fretless_bass, "fbass"),
    (23, 45, "Pizzicato Strings", 2, "Organs, Guitars & Harp", build_pizzicato_strings, "pizz"),
    (24, 46, "Orchestral Harp", 2, "Organs, Guitars & Harp", build_orchestral_harp, "harp"),

    # Family 3 (25..32)
    (25, 40, "Violin", 3, "Strings & Choir", build_violin, "violin"),
    (26, 41, "Viola", 3, "Strings & Choir", build_viola, "viola"),
    (27, 42, "Cello", 3, "Strings & Choir", build_cello, "cello"),
    (28, 43, "Contrabass", 3, "Strings & Choir", build_contrabass, "bass"),
    (29, 48, "String Ensemble", 3, "Strings & Choir", build_string_ensemble, "ensemble"),
    (30, 52, "Choir Aahs", 3, "Strings & Choir", build_choir_aahs, "choir_a"),
    (31, 53, "Voice Oohs", 3, "Strings & Choir", build_voice_oohs, "choir_o"),
    (32, 55, "Orchestra Hit", 3, "Strings & Choir", build_orchestra_hit, "orch_hit"),

    # Family 4 (33..41)
    (33, 56, "Trumpet", 4, "Brass & Saxophones", build_trumpet, "trumpet"),
    (34, 57, "Trombone", 4, "Brass & Saxophones", build_trombone, "trombone"),
    (35, 58, "Tuba", 4, "Brass & Saxophones", build_tuba, "tuba"),
    (36, 59, "Muted Trumpet", 4, "Brass & Saxophones", build_muted_trumpet, "muted_tpt"),
    (37, 60, "French Horn", 4, "Brass & Saxophones", build_french_horn, "horn"),
    (38, 61, "Brass Section", 4, "Brass & Saxophones", build_brass_section, "brass_sec"),
    (39, 64, "Soprano Sax", 4, "Brass & Saxophones", build_soprano_sax, "soprano_sax"),
    (40, 65, "Alto Sax", 4, "Brass & Saxophones", build_alto_sax, "alto_sax"),
    (41, 66, "Tenor Sax", 4, "Brass & Saxophones", build_tenor_sax, "tenor_sax"),

    # Family 5 (42..52)
    (42, 68, "Oboe", 5, "Woodwinds & Flutes", build_oboe, "oboe"),
    (43, 69, "English Horn", 5, "Woodwinds & Flutes", build_english_horn, "ehorn"),
    (44, 70, "Bassoon", 5, "Woodwinds & Flutes", build_bassoon, "bassoon"),
    (45, 71, "Clarinet", 5, "Woodwinds & Flutes", build_clarinet, "clarinet"),
    (46, 72, "Piccolo", 5, "Woodwinds & Flutes", build_piccolo, "piccolo"),
    (47, 73, "Flute", 5, "Woodwinds & Flutes", build_flute, "flute"),
    (48, 74, "Recorder", 5, "Woodwinds & Flutes", build_recorder, "recorder"),
    (49, 75, "Pan Flute", 5, "Woodwinds & Flutes", build_pan_flute, "panflute"),
    (50, 77, "Shakuhachi", 5, "Woodwinds & Flutes", build_shakuhachi, "shakuhachi"),
    (51, 78, "Whistle", 5, "Woodwinds & Flutes", build_whistle, "whistle"),
    (52, 79, "Ocarina", 5, "Woodwinds & Flutes", build_ocarina, "ocarina"),

    # Family 6 (53..64)
    (53, 104, "Sitar", 6, "World Ethnic & Synth", build_sitar, "sitar"),
    (54, 105, "Banjo", 6, "World Ethnic & Synth", build_banjo, "banjo"),
    (55, 106, "Shamisen", 6, "World Ethnic & Synth", build_shamisen, "shamisen"),
    (56, 107, "Koto", 6, "World Ethnic & Synth", build_koto, "koto"),
    (57, 108, "Kalimba", 6, "World Ethnic & Synth", build_kalimba, "kalimba"),
    (58, 109, "Bagpipe", 6, "World Ethnic & Synth", build_bagpipe, "bagpipe"),
    (59, 111, "Shanai", 6, "World Ethnic & Synth", build_shanai, "shanai"),
    (60, 114, "Steel Drums", 6, "World Ethnic & Synth", build_steel_drums, "steel_drum"),
    (61, 115, "Woodblock", 6, "World Ethnic & Synth", build_woodblock, "woodblock"),
    (62, 80, "Square Lead", 6, "World Ethnic & Synth", build_square_lead, "square"),
    (63, 81, "Sawtooth Lead", 6, "World Ethnic & Synth", build_sawtooth_lead, "saw"),
    (64, 88, "New Age Pad", 6, "World Ethnic & Synth", build_new_age_pad, "pad"),
]


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    write_asset_catalogs()
    scene = bpy.context.scene
    mats = build_materials()

    manifest: dict[str, dict] = {}
    root_col = scene.collection

    for idx, gm, name, fam_idx, fam_name, builder, variant in CATALOG_64:
        safe = name.replace(" ", "_")
        col_name = f"Lib_{idx:02d}_GM{gm:03d}_{safe}"
        col = bpy.data.collections.new(col_name)
        root_col.children.link(col)
        prefix = f"I{idx:02d}_GM{gm:03d}"
        builder(col, prefix, mats, variant)

        anim_family, glow_color = ANIM_PROFILE_BY_GM.get(gm, ("generic_strike", (1.0, 0.75, 0.30)))
        cat_key = FAMILY_CAT_KEY.get(fam_idx, "melodic")
        _mark_asset(
            col,
            cat_key,
            f"GM {gm:03d} {name} ({fam_name}) — bespoke 3D marble music assembly",
            ["instrument", f"gm_{gm:03d}", safe.lower(), anim_family],
        )

        manifest[str(gm)] = {
            "index": idx,
            "gm_program": gm,
            "name": name,
            "family_idx": fam_idx,
            "family_name": fam_name,
            "collection": col_name,
            "variant": variant,
            "anim_family": anim_family,
            "glow_color": list(glow_color),
            "part_count": len(col.objects),
        }

    os.makedirs(os.path.dirname(os.path.abspath(OUT_BLEND)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(OUT_BLEND))
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(
            {"version": "3.0", "total_instruments": len(CATALOG_64), "by_program": manifest},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Built 64-instrument library v3.0: {OUT_BLEND} ({len(manifest)} instruments, {len(mats)} materials)")


if __name__ == "__main__":
    main()
