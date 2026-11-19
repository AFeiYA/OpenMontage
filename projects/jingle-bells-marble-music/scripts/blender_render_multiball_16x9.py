#!/usr/bin/env python3
"""16:9 Multi-Ball Parallel-Lane Blender Scene Builder.

Reads `glass_marbles_synth_3ball_16x9_physics.json` and builds a 1920x1080 60fps
Blender scene with 3 colored glass marbles in Left / Center / Right lanes,
pulsing Square Lead boards, bronze rails, look-ahead camera, and synced VSE audio.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

import bmesh
import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import blender_render_marble_run as base_scene


LOOK_NIGHT = False  # set by build_multiball_scene(look=...)


def create_colored_marble_material(name: str, rgb: list[float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    # Crystal-clear hand-blown jewel glass with Object-space helical striae so reflections shimmer & roll during spin
    mx = max(rgb) if max(rgb) > 1e-4 else 1.0
    r_n = (rgb[0] / mx) ** 1.65
    g_n = (rgb[1] / mx) ** 1.65
    b_n = (rgb[2] / mx) ** 1.65
    deep_col = (0.03 + 0.72 * r_n, 0.03 + 0.72 * g_n, 0.03 + 0.72 * b_n, 1.0)
    bright_col = (0.10 + 0.90 * r_n, 0.10 + 0.90 * g_n, 0.10 + 0.90 * b_n, 1.0)

    tc = nodes.new("ShaderNodeTexCoord")
    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "DIAGONAL"
    wave.inputs["Scale"].default_value = 2.6
    wave.inputs["Distortion"].default_value = 5.2
    wave.inputs["Detail"].default_value = 3.0

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.22
    ramp.color_ramp.elements[0].color = deep_col
    ramp.color_ramp.elements[1].position = 0.80
    ramp.color_ramp.elements[1].color = bright_col

    rgh_map = nodes.new("ShaderNodeMapRange")
    rgh_map.inputs["From Min"].default_value = 0.0
    rgh_map.inputs["From Max"].default_value = 1.0
    rgh_map.inputs["To Min"].default_value = 0.004
    rgh_map.inputs["To Max"].default_value = 0.028

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.016
    bump.inputs["Distance"].default_value = 0.002

    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["IOR"].default_value = 1.52
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 1.0
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 1.0
    if "Coat Roughness" in bsdf.inputs:
        bsdf.inputs["Coat Roughness"].default_value = 0.004
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.0
    if hasattr(mat, "thickness_mode"):
        mat.thickness_mode = "SPHERE"
    if hasattr(mat, "use_raytrace_refraction"):
        mat.use_raytrace_refraction = True
    if hasattr(mat, "use_screen_refraction"):
        mat.use_screen_refraction = True
    if hasattr(mat, "refraction_depth"):
        mat.refraction_depth = 0.21

    links.new(tc.outputs["Object"], wave.inputs["Vector"])
    links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(wave.outputs["Fac"], rgh_map.inputs["Value"])
    links.new(rgh_map.outputs["Result"], bsdf.inputs["Roughness"])
    links.new(wave.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _tileable_fbm(h: int, w: int, octaves: list[tuple[int, int, float]], seed: int = 42):
    import numpy as np

    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    amp_sum = 0.0
    yy = np.linspace(0.0, 1.0, h, endpoint=False, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, endpoint=False, dtype=np.float32)[None, :]
    for nx, ny, amp in octaves:
        grid = rng.random((ny, nx), dtype=np.float32)
        gx = xx * nx
        gy = yy * ny
        ix0 = np.floor(gx).astype(np.int32) % nx
        iy0 = np.floor(gy).astype(np.int32) % ny
        ix1 = (ix0 + 1) % nx
        iy1 = (iy0 + 1) % ny
        fx = gx - np.floor(gx)
        fy = gy - np.floor(gy)
        sx = fx * fx * (3.0 - 2.0 * fx)
        sy = fy * fy * (3.0 - 2.0 * fy)
        v00 = grid[iy0, ix0]
        v10 = grid[iy0, ix1]
        v01 = grid[iy1, ix0]
        v11 = grid[iy1, ix1]
        interp = (v00 * (1.0 - sx) + v10 * sx) * (1.0 - sy) + (v01 * (1.0 - sx) + v11 * sx) * sy
        acc += amp * interp
        amp_sum += amp
    return acc / max(1e-6, amp_sum)


def ensure_walnut_textures(tex_dir: str) -> dict[str, str]:
    """Synthesize and cache sharp seamless 768x1536 American Walnut albedo, bump, and roughness PNGs."""
    import numpy as np

    os.makedirs(tex_dir, exist_ok=True)
    paths = {
        "albedo": os.path.join(tex_dir, "walnut_albedo_v3.png"),
        "bump": os.path.join(tex_dir, "walnut_bump_v3.png"),
        "rough": os.path.join(tex_dir, "walnut_rough_v3.png"),
    }
    if all(os.path.exists(p) for p in paths.values()):
        return paths

    w, h = 768, 1536
    u = np.linspace(0.0, 1.0, w, endpoint=False, dtype=np.float32)[None, :]
    v = np.linspace(0.0, 1.0, h, endpoint=False, dtype=np.float32)[:, None]

    # Warp field for authentic cathedral arches & organic side grain (periodic across both U & V)
    warp_low = _tileable_fbm(h, w, [(4, 2, 1.0), (8, 4, 0.5), (16, 8, 0.25)], seed=101)
    warp_hi = _tileable_fbm(h, w, [(28, 10, 1.0), (56, 20, 0.45)], seed=202)
    # Straight, fine-pored American Walnut: near-vertical lines with only gentle drift (matches reference photo)
    xw = u + 0.030 * (warp_low - 0.5) + 0.0045 * (warp_hi - 0.5) + 0.004 * np.sin(2.0 * math.pi * v)
    rings = 0.5 + 0.5 * np.sin(2.0 * math.pi * 46.0 * xw)
    rings = 0.65 * rings + 0.35 * (0.5 + 0.5 * np.sin(2.0 * math.pi * 17.0 * xw + 1.3))

    # Fine elongated hardwood pores & longitudinal fiber streaks
    fibers = _tileable_fbm(h, w, [(120, 10, 1.0), (240, 20, 0.55)], seed=303)
    pores_raw = _tileable_fbm(h, w, [(256, 28, 1.0), (384, 52, 0.5)], seed=404)
    pores = np.clip((pores_raw - 0.62) / 0.30, 0.0, 1.0)
    rays = _tileable_fbm(h, w, [(6, 3, 1.0), (14, 7, 0.4)], seed=505)

    # Linear albedo mix: deep walnut pores vs rich cognac heartwood
    t = np.clip(0.50 + 0.20 * (rings - 0.5) + 0.34 * (fibers - 0.5) + 0.22 * (rays - 0.5) - 0.30 * pores, 0.0, 1.0)
    dark_lin = np.array([0.060, 0.028, 0.013], dtype=np.float32)
    light_lin = np.array([0.185, 0.098, 0.047], dtype=np.float32)
    rgb_lin = dark_lin[None, None, :] + t[:, :, None] * (light_lin - dark_lin)[None, None, :]
    rgb_srgb = np.power(np.clip(rgb_lin, 0.0, 1.0), 1.0 / 2.2)

    bump_val = np.clip(0.62 + 0.16 * rings + 0.14 * fibers - 0.55 * pores, 0.0, 1.0)
    rough_val = np.clip(0.36 + 0.10 * (1.0 - rings) + 0.26 * pores, 0.0, 1.0)

    def _save_png(arr_rgb: np.ndarray, out_path: str, img_name: str):
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = arr_rgb
        img = bpy.data.images.new(img_name, width=w, height=h, alpha=False, float_buffer=False)
        img.pixels.foreach_set(rgba.ravel())
        img.filepath_raw = out_path
        img.file_format = "PNG"
        img.save()
        bpy.data.images.remove(img)

    _save_png(rgb_srgb, paths["albedo"], "TmpWalnutAlbedo")
    _save_png(np.repeat(bump_val[:, :, None], 3, axis=2), paths["bump"], "TmpWalnutBump")
    _save_png(np.repeat(rough_val[:, :, None], 3, axis=2), paths["rough"], "TmpWalnutRough")
    return paths


def create_wood_plank_image_material(name: str, tex_paths: dict[str, str], is_smoked: bool = False) -> bpy.types.Material:
    """Fast UV-mapped hardwood shader using baked grain maps + ObjectInfo.Random per-plank diversity."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    tc = nodes.new("ShaderNodeTexCoord")
    obj_info = nodes.new("ShaderNodeObjectInfo")

    # Per-object random UV shift so every physical wood board displays a unique grain section
    uv_shift_scale = nodes.new("ShaderNodeVectorMath")
    uv_shift_scale.operation = "SCALE"
    uv_shift_scale.inputs["Scale"].default_value = 0.85

    rnd_vec = nodes.new("ShaderNodeCombineXYZ")
    rnd_y = nodes.new("ShaderNodeMath")
    rnd_y.operation = "MULTIPLY"
    rnd_y.inputs[1].default_value = 1.37

    uv_add = nodes.new("ShaderNodeVectorMath")
    uv_add.operation = "ADD"

    img_alb = bpy.data.images.load(tex_paths["albedo"], check_existing=True)
    img_alb.colorspace_settings.name = "sRGB"
    tex_alb = nodes.new("ShaderNodeTexImage")
    tex_alb.image = img_alb

    img_bmp = bpy.data.images.load(tex_paths["bump"], check_existing=True)
    img_bmp.colorspace_settings.name = "Non-Color"
    tex_bmp = nodes.new("ShaderNodeTexImage")
    tex_bmp.image = img_bmp

    img_rgh = bpy.data.images.load(tex_paths["rough"], check_existing=True)
    img_rgh.colorspace_settings.name = "Non-Color"
    tex_rgh = nodes.new("ShaderNodeTexImage")
    tex_rgh.image = img_rgh

    # Subtle per-board natural wood tone variation via ObjectInfo.Random
    val_map = nodes.new("ShaderNodeMapRange")
    val_map.inputs["From Min"].default_value = 0.0
    val_map.inputs["From Max"].default_value = 1.0
    if is_smoked:
        val_map.inputs["To Min"].default_value = 0.34
        val_map.inputs["To Max"].default_value = 0.48
    else:
        val_map.inputs["To Min"].default_value = 0.46
        val_map.inputs["To Max"].default_value = 0.86

    hsv = nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Hue"].default_value = 0.5
    hsv.inputs["Saturation"].default_value = 0.85 if is_smoked else 1.02

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14 if not is_smoked else 0.10
    bump.inputs["Distance"].default_value = 0.004

    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.42

    links.new(obj_info.outputs["Random"], rnd_vec.inputs["X"])
    links.new(obj_info.outputs["Random"], rnd_y.inputs[0])
    links.new(rnd_y.outputs["Value"], rnd_vec.inputs["Y"])
    links.new(rnd_vec.outputs["Vector"], uv_shift_scale.inputs[0])
    links.new(tc.outputs["UV"], uv_add.inputs[0])
    links.new(uv_shift_scale.outputs["Vector"], uv_add.inputs[1])

    for tex_node in (tex_alb, tex_bmp, tex_rgh):
        links.new(uv_add.outputs["Vector"], tex_node.inputs["Vector"])

    links.new(tex_alb.outputs["Color"], hsv.inputs["Color"])
    links.new(obj_info.outputs["Random"], val_map.inputs["Value"])
    links.new(val_map.outputs["Result"], hsv.inputs["Value"])
    links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex_rgh.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(tex_bmp.outputs["Color"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def create_beveled_wood_plank_obj(
    name: str,
    xc: float,
    y_back: float,
    zc: float,
    w: float,
    thick: float,
    h: float,
    horizontal_grain: bool,
    uv_scale: float,
    uv_off: tuple[float, float],
    bevel_offset: float,
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Create a real individual beveled 3D wooden board object with continuous box-mapped wood UVs."""
    bm = bmesh.new()
    res = bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(w, thick, h), verts=res["verts"])
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel_offset, segments=2, affect="EDGES", clamp_overlap=True)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    uv_layer = bm.loops.layers.uv.verify()
    for face in bm.faces:
        face.smooth = False
        nx, ny, nz = abs(face.normal.x), abs(face.normal.y), abs(face.normal.z)
        for loop in face.loops:
            vx, vy, vz = loop.vert.co
            if ny >= 0.5:
                ur, vr = vx + 0.15 * vy, vz + 0.15 * vy
            elif nx >= 0.5:
                ur, vr = vy, vz
            else:
                ur, vr = vx, vy
            if horizontal_grain:
                ur, vr = vr, ur
            loop[uv_layer].uv = (ur * uv_scale + uv_off[0], vr * uv_scale + uv_off[1])

    mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = (xc, y_back - 0.5 * thick, zc)
    obj.data.materials.append(mat)
    return obj


def create_textured_white_wall(
    wall_y: float,
    z_min: float,
    z_max: float,
    brass_mat,
    collection,
    opening_z: float = 0.56,
    finale_z: float | None = None,
):
    """Build real modular 3D wooden plank wall with horizontal hero boards, physical thickness steps & fluted slats."""
    wall_col = base_scene.get_or_create_collection("Stage_WoodWall", collection)
    span_z = max(40.0, (z_max - z_min) + 48.0)
    center_z = 0.5 * (z_max + z_min)
    z_top = center_z + span_z * 0.5
    z_bot = center_z - span_z * 0.5

    tex_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "assets", "textures"))
    tex_paths = ensure_walnut_textures(tex_dir)
    plank_mat = create_wood_plank_image_material("Mat_WalnutBoard_Main", tex_paths, is_smoked=False)
    smoked_mat = create_wood_plank_image_material("Mat_SmokedWalnutSlat", tex_paths, is_smoked=True)

    # Dark recessed shadow-gap backing plate behind all joints
    bmat = bpy.data.materials.new("WalnutSeamBackingMat")
    bmat.use_nodes = True
    bsdf_bk = next(nd for nd in bmat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    bsdf_bk.inputs["Base Color"].default_value = (0.010, 0.007, 0.006, 1.0)
    bsdf_bk.inputs["Roughness"].default_value = 0.88

    fmesh = bpy.data.meshes.new("WallShadowBacking")
    fbm = bmesh.new()
    bmesh.ops.create_grid(fbm, x_segments=1, y_segments=1, size=1.0)
    bmesh.ops.transform(fbm, matrix=Matrix.Rotation(math.radians(90.0), 4, "X"), verts=fbm.verts)
    bmesh.ops.scale(fbm, vec=(28.0, 1.0, span_z * 0.5), verts=fbm.verts)
    bmesh.ops.translate(fbm, vec=(0.0, wall_y + 0.032, center_z), verts=fbm.verts)
    fbm.to_mesh(fmesh)
    fbm.free()
    fobj = bpy.data.objects.new("WallShadowBacking", fmesh)
    wall_col.objects.link(fobj)
    fobj.data.materials.append(bmat)

    y_back = wall_y + 0.027
    gap = 0.007
    rng = random.Random(202612)

    # 1. Collect full-width Horizontal Hero Planks (Opening title, 15m carved watermarks, Finale credit)
    hero_specs: list[tuple[float, float, str]] = [(opening_z, 1.30, "opening")]
    watermark_zs: list[float] = []
    z_wm = -10.5
    while z_wm > z_min + 2.0:
        if finale_z is None or abs(z_wm - finale_z) > 1.45:
            hero_specs.append((z_wm, 1.02, "watermark"))
            watermark_zs.append(z_wm)
        z_wm -= 15.0
    if finale_z is not None:
        hero_specs.append((finale_z, 1.04, "finale"))

    hero_specs.sort(key=lambda item: item[0], reverse=True)
    for h_i, (hz, hh, tag) in enumerate(hero_specs):
        create_beveled_wood_plank_obj(
            name=f"HeroPlank_{h_i:02d}_{tag}",
            xc=0.0,
            y_back=y_back,
            zc=hz,
            w=4.28,
            thick=0.023,
            h=hh,
            horizontal_grain=True,
            uv_scale=0.55,
            uv_off=(rng.random(), rng.random()),
            bevel_offset=0.004,
            mat=plank_mat,
            collection=wall_col,
        )
    # 2. Fill vertical mosaic bays between horizontal hero planks with real staggered individual boards
    bay_edges = [z_top]
    for hz, hh, _ in hero_specs:
        bay_edges.append(hz + hh * 0.5)
        bay_edges.append(hz - hh * 0.5)
    bay_edges.append(z_bot)

    bays = [(bay_edges[k * 2 + 1], bay_edges[k * 2]) for k in range(len(bay_edges) // 2)]
    col_bounds = [(-2.14, -1.36), (-1.36, -0.52), (-0.52, 0.56), (0.56, 1.38), (1.38, 2.14)]
    col_staggers = [0.0, 0.72, 0.31, 0.94, 0.48]

    for bay_i, (bz_lo, bz_hi) in enumerate(bays):
        usable_top = bz_hi - gap
        usable_bot = bz_lo + gap
        if usable_top - usable_bot < 0.20:
            continue
        for c_i, ((x0, x1), stagger) in enumerate(zip(col_bounds, col_staggers)):
            pw = (x1 - x0) - gap
            pxc = 0.5 * (x0 + x1)
            cursor = usable_top
            first_row = True
            row_i = 0
            while cursor > usable_bot + 0.02:
                base_len = rng.uniform(1.55, 2.65)
                if first_row:
                    base_len = max(0.85, base_len - stagger)
                    first_row = False
                rem = cursor - usable_bot
                if rem <= base_len + 0.55:
                    ph = rem
                else:
                    ph = base_len - gap
                pzc = cursor - 0.5 * ph
                thick = rng.uniform(0.017, 0.023)
                create_beveled_wood_plank_obj(
                    name=f"WalnutPlank_b{bay_i:02d}_c{c_i}_r{row_i:02d}",
                    xc=pxc,
                    y_back=y_back,
                    zc=pzc,
                    w=pw,
                    thick=thick,
                    h=ph,
                    horizontal_grain=False,
                    uv_scale=0.62,
                    uv_off=(rng.random(), rng.random()),
                    bevel_offset=0.0035,
                    mat=plank_mat,
                    collection=wall_col,
                )
                cursor -= ph + gap
                row_i += 1
    # 3. Left & right fluted architectural dark-walnut 3D slats (segmented into modular 6m battens)
    slat_seg_h = 6.0
    n_slat_segs = max(1, int(math.ceil(span_z / slat_seg_h)))
    for side_s, side_tag in ((-1.0, "L"), (1.0, "R")):
        for k in range(28):
            xc = side_s * (2.28 + k * 0.105)
            z_off = (k % 3) * 1.4
            for s_i in range(n_slat_segs + 1):
                sz_hi = min(z_top, z_top + z_off - s_i * slat_seg_h)
                sz_lo = max(z_bot, z_top + z_off - (s_i + 1) * slat_seg_h + gap)
                if sz_hi - sz_lo < 0.25:
                    continue
                create_beveled_wood_plank_obj(
                    name=f"SmokedSlat_{side_tag}_{k:02d}_{s_i:02d}",
                    xc=xc,
                    y_back=wall_y + 0.032,
                    zc=0.5 * (sz_hi + sz_lo),
                    w=0.062,
                    thick=0.048,
                    h=sz_hi - sz_lo,
                    horizontal_grain=False,
                    uv_scale=0.58,
                    uv_off=(rng.random(), rng.random()),
                    bevel_offset=0.004,
                    mat=smoked_mat,
                    collection=wall_col,
                )

    # Slim vertical brushed-brass gallery border strips & side-flank brass hairline accents
    vmesh = bpy.data.meshes.new("GalleryBrassBorders")
    vbm = bmesh.new()
    for bx in (-2.165, 2.165):
        res = bmesh.ops.create_cube(vbm, size=1.0)
        bmesh.ops.scale(vbm, vec=(0.016, 0.024, span_z), verts=res["verts"])
        bmesh.ops.translate(vbm, vec=(bx, wall_y + 0.015, center_z), verts=res["verts"])
    for wz in watermark_zs:
        for sx in (-2.90, 2.90):
            res = bmesh.ops.create_cube(vbm, size=1.0)
            bmesh.ops.scale(vbm, vec=(1.45, 0.014, 0.014), verts=res["verts"])
            bmesh.ops.translate(vbm, vec=(sx, wall_y - 0.002, wz + 0.04), verts=res["verts"])
    vbm.to_mesh(vmesh)
    vbm.free()
    vobj = bpy.data.objects.new("GalleryBrassBorders", vmesh)
    wall_col.objects.link(vobj)
    vobj.data.materials.append(brass_mat)

    # 4. Carved/charred 2-line Futura typography centered onto each Horizontal Hero Watermark Plank
    relief_mat1 = bpy.data.materials.new("WalnutEngraveTitleMat")
    relief_mat1.use_nodes = True
    bsdf_r1 = next(nd for nd in relief_mat1.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    bsdf_r1.inputs["Base Color"].default_value = (0.014, 0.007, 0.003, 1.0)
    bsdf_r1.inputs["Roughness"].default_value = 0.78

    relief_mat2 = bpy.data.materials.new("WalnutEngraveSubMat")
    relief_mat2.use_nodes = True
    bsdf_r2 = next(nd for nd in relief_mat2.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    bsdf_r2.inputs["Base Color"].default_value = (0.018, 0.009, 0.004, 1.0)
    bsdf_r2.inputs["Roughness"].default_value = 0.74

    font_path = "/System/Library/Fonts/Supplemental/Futura.ttc"
    vfont = bpy.data.fonts.load(font_path) if os.path.exists(font_path) else None

    for t_idx, z_text in enumerate(watermark_zs):
        for line_i, (txt_str, sz, mat_r, dz_l) in enumerate([
            ("MARBLE CHRISTMAS", 0.23, relief_mat1, 0.14),
            ("GLASS MARBLES ON WOOD", 0.115, relief_mat2, -0.13),
        ]):
            c = bpy.data.curves.new(f"WallWatermark_{t_idx}_{line_i}_curve", type="FONT")
            if vfont is not None:
                c.font = vfont
            c.body = txt_str
            c.size = sz
            c.space_character = 1.15 if line_i == 0 else 1.25
            c.extrude = 0.0022
            c.bevel_depth = 0.0014
            c.align_x = "CENTER"
            c.align_y = "CENTER"
            wobj = bpy.data.objects.new(f"WallWatermark_{t_idx}_{line_i}", c)
            wall_col.objects.link(wobj)
            wobj.data.materials.append(mat_r)
            wobj.location = (0.0, wall_y + 0.0038, z_text + dz_l)
            wobj.rotation_euler = (math.radians(90.0), 0.0, 0.0)

    # 5. Christmas fairy-light garlands draped down both fluted flanks (warm gold / red / green bulbs)
    bulb_mesh = bpy.data.meshes.new("FairyBulbMesh")
    bbm = bmesh.new()
    bmesh.ops.create_uvsphere(bbm, u_segments=12, v_segments=8, radius=0.030)
    bbm.to_mesh(bulb_mesh)
    bbm.free()
    for poly in bulb_mesh.polygons:
        poly.use_smooth = True
    bulb_palette = [
        ("Gold", (1.0, 0.58, 0.14)),
        ("Red", (1.0, 0.03, 0.02)),
        ("Gold", (1.0, 0.58, 0.14)),
        ("Green", (0.04, 0.90, 0.15)),
    ]
    bulb_mats = []
    for bi, (bn, bc) in enumerate(bulb_palette):
        bm_ = bpy.data.materials.new(f"FairyBulbMat_{bi}_{bn}")
        bm_.use_nodes = True
        nt = bm_.node_tree
        for nd in list(nt.nodes):
            nt.nodes.remove(nd)
        o_ = nt.nodes.new("ShaderNodeOutputMaterial")
        e_ = nt.nodes.new("ShaderNodeEmission")
        e_.inputs["Color"].default_value = (bc[0], bc[1], bc[2], 1.0)
        e_.inputs["Strength"].default_value = 18.0 if LOOK_NIGHT else 14.0
        nt.links.new(e_.outputs["Emission"], o_.inputs["Surface"])
        bulb_mats.append(bm_)
    bulb_meshes = []
    for bi, bm_ in enumerate(bulb_mats):
        mcopy = bulb_mesh.copy()
        mcopy.name = f"FairyBulbMesh_{bi}"
        mcopy.materials.append(bm_)
        bulb_meshes.append(mcopy)
    wire_mat = bpy.data.materials.new("FairyWireMat")
    wire_mat.use_nodes = True
    wb = next(nd for nd in wire_mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    wb.inputs["Base Color"].default_value = (0.02, 0.07, 0.03, 1.0)
    wb.inputs["Roughness"].default_value = 0.5
    for side_s in (-1.0, 1.0):
        for strand, (x_base, amp, phase) in enumerate(((2.34, 0.10, 0.0), (2.68, 0.14, 1.9))):
            z_b = z_top
            n_b = 0
            wire_pts = []
            while z_b > z_bot:
                x_b = side_s * (x_base + amp * math.sin(z_b * 2.1 + phase))
                bo = bpy.data.objects.new(f"FairyBulb_{int(side_s)}_{strand}_{n_b:04d}", bulb_meshes[n_b % len(bulb_meshes)])
                wall_col.objects.link(bo)
                bo.location = (x_b, wall_y - 0.055, z_b)
                wire_pts.append((x_b, wall_y - 0.052, z_b))
                z_b -= 0.23 + 0.02 * (n_b % 3)
                n_b += 1
            wc = bpy.data.curves.new(f"FairyWire_{int(side_s)}_{strand}", type="CURVE")
            wc.dimensions = "3D"
            wc.bevel_depth = 0.0035
            wc.bevel_resolution = 2
            sp = wc.splines.new("POLY")
            sp.points.add(len(wire_pts) - 1)
            for pi_, p_ in enumerate(wire_pts):
                sp.points[pi_].co = (p_[0], p_[1], p_[2], 1.0)
            wo = bpy.data.objects.new(f"FairyWire_{int(side_s)}_{strand}", wc)
            wall_col.objects.link(wo)
            wo.data.materials.append(wire_mat)

    return fobj


_TITLE_BULB_MESHES: list = []


def _title_bulb_meshes():
    """Four shared emissive bulb meshes: gold, red, gold, green (classic Christmas string order)."""
    if _TITLE_BULB_MESHES:
        return _TITLE_BULB_MESHES
    base = bpy.data.meshes.new("TitleBulbBase")
    bm_ = bmesh.new()
    bmesh.ops.create_uvsphere(bm_, u_segments=14, v_segments=10, radius=1.0)
    bm_.to_mesh(base)
    bm_.free()
    for poly in base.polygons:
        poly.use_smooth = True
    for bi, (bn, bc) in enumerate((
        ("Gold", (1.0, 0.58, 0.14)),
        ("Red", (1.0, 0.03, 0.02)),
        ("Gold", (1.0, 0.58, 0.14)),
        ("Green", (0.04, 0.90, 0.15)),
    )):
        m = bpy.data.materials.new(f"TitleBulbMat_{bi}_{bn}")
        m.use_nodes = True
        nt = m.node_tree
        for nd in list(nt.nodes):
            nt.nodes.remove(nd)
        o_ = nt.nodes.new("ShaderNodeOutputMaterial")
        e_ = nt.nodes.new("ShaderNodeEmission")
        e_.inputs["Color"].default_value = (bc[0], bc[1], bc[2], 1.0)
        e_.inputs["Strength"].default_value = 12.5 if LOOK_NIGHT else 9.5
        nt.links.new(e_.outputs["Emission"], o_.inputs["Surface"])
        mesh = base.copy()
        mesh.name = f"TitleBulbMesh_{bi}"
        mesh.materials.append(m)
        _TITLE_BULB_MESHES.append(mesh)
    return _TITLE_BULB_MESHES


def _create_xmas_neon_font_material(name: str, is_multicolor: bool = True) -> bpy.types.Material:
    """Warm Christmas multi-color (Gold / Ruby / Emerald) or Champagne-Gold glowing light-tube material for vector letters."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    em = nodes.new("ShaderNodeEmission")
    if is_multicolor:
        tc = nodes.new("ShaderNodeTexCoord")
        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "X"
        wave.inputs["Scale"].default_value = 2.2
        wave.inputs["Distortion"].default_value = 0.0
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.interpolation = "CONSTANT"
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (1.0, 0.62, 0.16, 1.0)  # Warm Gold
        e1 = ramp.color_ramp.elements.new(0.26)
        e1.color = (1.0, 0.10, 0.06, 1.0)                           # Christmas Red
        e2 = ramp.color_ramp.elements.new(0.52)
        e2.color = (1.0, 0.68, 0.22, 1.0)                           # Warm Gold
        ramp.color_ramp.elements[-1].position = 0.76
        ramp.color_ramp.elements[-1].color = (0.12, 0.92, 0.24, 1.0)  # Christmas Green
        em.inputs["Strength"].default_value = 5.4
        links.new(tc.outputs["Generated"], wave.inputs["Vector"])
        links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], em.inputs["Color"])
    else:
        em.inputs["Color"].default_value = (1.0, 0.72, 0.30, 1.0)
        em.inputs["Strength"].default_value = 4.2
    links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def create_neon_title_plaque(wall_y: float, anchor_x: float, anchor_z: float, brass_mat, collection):
    """Christmas neon-channel vector typography framed by draped red/gold/green fairy-light garlands."""
    font_path = "/System/Library/Fonts/Supplemental/Futura.ttc"
    vfont = bpy.data.fonts.load(font_path) if os.path.exists(font_path) else None
    meshes = _title_bulb_meshes()
    tag = f"{anchor_z:+.2f}"

    # Dark bronze letter channel housing
    housing_mat = bpy.data.materials.new(f"TitleHousingMat_{tag}")
    housing_mat.use_nodes = True
    bsdf_h = next(nd for nd in housing_mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    bsdf_h.inputs["Base Color"].default_value = (0.04, 0.025, 0.015, 1.0)
    bsdf_h.inputs["Metallic"].default_value = 0.75
    bsdf_h.inputs["Roughness"].default_value = 0.35

    neon_main = _create_xmas_neon_font_material(f"TitleNeonMain_{tag}", is_multicolor=True)
    neon_sub = _create_xmas_neon_font_material(f"TitleNeonSub_{tag}", is_multicolor=False)

    lines = [
        ("TitleLine1", "MARBLE CHRISTMAS", 0.200, 1.14, anchor_z + 0.13, neon_main),
        ("TitleLine2", "02  GLASS MARBLES ON WOOD", 0.102, 1.20, anchor_z - 0.04, neon_sub),
        ("TitleLine3", "ARTIST  :  ZERO.SOE", 0.080, 1.24, anchor_z - 0.19, neon_sub),
    ]
    for obj_name, txt_str, size, tracking, z_pos, neon_mat in lines:
        # 1. Dark bronze outer letter channel
        c_back = bpy.data.curves.new(f"{obj_name}_{tag}_back_curve", type="FONT")
        if vfont is not None:
            c_back.font = vfont
        c_back.body = txt_str
        c_back.size = size
        c_back.space_character = tracking
        c_back.extrude = 0.0028
        c_back.bevel_depth = 0.0022
        c_back.align_x = "CENTER"
        c_back.align_y = "CENTER"
        o_back = bpy.data.objects.new(f"{obj_name}_{tag}_back", c_back)
        collection.objects.link(o_back)
        o_back.data.materials.append(housing_mat)
        o_back.location = (anchor_x, wall_y + 0.0038, z_pos)
        o_back.rotation_euler = (math.radians(90.0), 0.0, 0.0)

        # 2. Glowing festive Christmas neon core inside the letter channel
        c_glow = bpy.data.curves.new(f"{obj_name}_{tag}_glow_curve", type="FONT")
        if vfont is not None:
            c_glow.font = vfont
        c_glow.body = txt_str
        c_glow.size = size
        c_glow.space_character = tracking
        c_glow.extrude = 0.0032
        c_glow.bevel_depth = 0.0008
        if hasattr(c_glow, "offset"):
            c_glow.offset = -0.0008
        c_glow.align_x = "CENTER"
        c_glow.align_y = "CENTER"
        o_glow = bpy.data.objects.new(f"{obj_name}_{tag}_glow", c_glow)
        collection.objects.link(o_glow)
        o_glow.data.materials.append(neon_mat)
        o_glow.location = (anchor_x, wall_y + 0.0022, z_pos)
        o_glow.rotation_euler = (math.radians(90.0), 0.0, 0.0)

    # 3. Top and bottom draped Christmas fairy-light garlands framing the title
    wire_mat = bpy.data.materials.new(f"TitleGarlandWireMat_{tag}")
    wire_mat.use_nodes = True
    wb = next(nd for nd in wire_mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    wb.inputs["Base Color"].default_value = (0.02, 0.08, 0.03, 1.0)
    wb.inputs["Roughness"].default_value = 0.45

    span_w = 2.24
    n_bulbs = 23
    for g_idx, (gz_base, sag_sign) in enumerate(((anchor_z + 0.29, -1.0), (anchor_z - 0.29, 1.0))):
        wire_pts = []
        for bi in range(n_bulbs):
            u = bi / (n_bulbs - 1)
            bx = anchor_x + (u - 0.5) * span_w
            # Gentle 3-arch scalloped garland sag
            arch = abs(math.sin(u * 3.0 * math.pi))
            bz = gz_base + sag_sign * 0.032 * arch
            by = wall_y - 0.008
            wire_pts.append((bx, by + 0.003, bz))
            bo = bpy.data.objects.new(f"TitleGarlandBulb_{tag}_{g_idx}_{bi:02d}", meshes[(bi + g_idx) % 4])
            collection.objects.link(bo)
            bo.scale = (0.015, 0.015, 0.015)
            bo.location = (bx, by, bz)
        wc = bpy.data.curves.new(f"TitleGarlandWire_{tag}_{g_idx}", type="CURVE")
        wc.dimensions = "3D"
        wc.bevel_depth = 0.0028
        wc.bevel_resolution = 2
        sp = wc.splines.new("POLY")
        sp.points.add(len(wire_pts) - 1)
        for pi_, p_ in enumerate(wire_pts):
            sp.points[pi_].co = (p_[0], p_[1], p_[2], 1.0)
        wo = bpy.data.objects.new(f"TitleGarlandWire_{tag}_{g_idx}", wc)
        collection.objects.link(wo)
        wo.data.materials.append(wire_mat)


BULB_FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    ":": ["0", "1", "1", "0", "1", "1", "0"],
    ".": ["0", "0", "0", "0", "0", "1", "1"],
}


def create_bulb_title_plaque(
    wall_y: float, anchor_x: float, anchor_z: float, brass_mat, collection, compact: bool = False
):
    """v14 style: every letter is a dot-matrix of glowing red/gold/green Christmas bulbs (compact=True for Finale clearance)."""
    meshes = _title_bulb_meshes()
    # (text, pitch, bulb radius, colour mode)
    if compact:
        lines = [
            ("MARBLE CHRISTMAS", 0.031, 0.0115, "cycle"),
            ("GLASS MARBLES ON WOOD", 0.0195, 0.0074, "cycle"),
            ("ARTIST : ZERO.SOE", 0.018, 0.0068, "gold"),
        ]
        gaps = [0.065, 0.055]
    else:
        lines = [
            ("MARBLE", 0.036, 0.0135, "cycle"),
            ("CHRISTMAS", 0.036, 0.0135, "cycle"),
            ("GLASS MARBLES ON WOOD", 0.021, 0.0080, "cycle"),
            ("ARTIST : ZERO.SOE", 0.019, 0.0072, "gold"),
        ]
        gaps = [0.07, 0.085, 0.07]
    heights = [7 * ln[1] for ln in lines]
    total_h = sum(heights) + sum(gaps)
    cursor_z = anchor_z + 0.5 * total_h
    bulb_i = 0
    tag = f"{anchor_z:+.2f}"
    for li, (text, pitch, rad, mode) in enumerate(lines):
        glyphs = [None if ch == " " else BULB_FONT[ch] for ch in text]
        total_cols = sum((3 if g is None else len(g[0])) + 1 for g in glyphs) - 1
        col_cursor = 0
        top_row_center = cursor_z - 0.5 * pitch
        for g in glyphs:
            if g is None:
                col_cursor += 4
                continue
            gw = len(g[0])
            for r_i, row in enumerate(g):
                for c_i, cell in enumerate(row):
                    if cell != "1":
                        continue
                    x = anchor_x + (col_cursor + c_i - 0.5 * (total_cols - 1)) * pitch
                    z = top_row_center - r_i * pitch
                    mesh = meshes[0 if mode == "gold" else bulb_i % 4]
                    bo = bpy.data.objects.new(f"TitleBulb_{tag}_{li}_{bulb_i:04d}", mesh)
                    collection.objects.link(bo)
                    bo.scale = (rad, rad, rad)
                    bo.location = (x, wall_y + 0.004 - 0.8 * rad, z)
                    bulb_i += 1
            col_cursor += gw + 1
        cursor_z -= heights[li] + (gaps[li] if li < len(gaps) else 0.0)


def create_custom_title_plaque(
    wall_y: float, anchor_x: float, anchor_z: float, brass_mat, collection, style: str = "bulb", compact: bool = False
):
    """style='bulb' (v14 string-light dot-matrix, default) or 'neon' (v15 vector neon letters + garlands)."""
    if style == "neon":
        return create_neon_title_plaque(wall_y, anchor_x, anchor_z, brass_mat, collection)
    return create_bulb_title_plaque(wall_y, anchor_x, anchor_z, brass_mat, collection, compact=compact)


def create_smooth_pipe_polyline(name: str, pts: list[list[float]], radius: float, mat, collection):
    """C2-smooth downsampled NURBS pipe curve (eliminates dense-point jitter and tail kinks)."""
    # Downsample dense solver polylines (~89 pts -> ~18 pts) so NURBS order-4 stays silky smooth
    if len(pts) > 20:
        step = max(1, len(pts) // 16)
        sampled = pts[::step]
        if sampled[-1] != pts[-1]:
            # Replace second-to-last if too close to endpoint to prevent tail hook
            if len(sampled) >= 2 and (
                (sampled[-1][0] - pts[-1][0]) ** 2
                + (sampled[-1][1] - pts[-1][1]) ** 2
                + (sampled[-1][2] - pts[-1][2]) ** 2
            ) < 0.0025:
                sampled[-1] = pts[-1]
            else:
                sampled.append(pts[-1])
        pts_use = sampled
    else:
        pts_use = pts

    curve = bpy.data.curves.new(name + "_curve", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 32
    curve.bevel_depth = float(radius)
    curve.bevel_resolution = 16
    curve.use_fill_caps = True

    spline = curve.splines.new("NURBS")
    spline.points.add(len(pts_use) - 1)
    for k, pt in enumerate(pts_use):
        spline.points[k].co = (pt[0], pt[1], pt[2], 1.0)
    spline.use_endpoint_u = True
    spline.order_u = min(4, len(pts_use))
    spline.use_smooth = True

    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def build_multiball_scene(
    physics_json: str,
    out_blend: str,
    audio_path: str | None = None,
    frames_dir: str | None = None,
    check_frames: list[int] | None = None,
    look: str = "warm",
    title_style: str = "bulb",
) -> None:
    """look: 'warm' (Christmas room) | 'night' (marbles at night). title_style: 'bulb' (v14) | 'neon' (v15)."""
    global LOOK_NIGHT
    LOOK_NIGHT = (look == "night")
    with open(physics_json, "r", encoding="utf-8") as f:
        plan = json.load(f)

    fps = int(plan["fps"])
    total_frames = int(plan["total_frames"])
    radius = float(plan["marble_radius"])
    wall_y = float(plan["back_wall_y"])
    balls = plan["balls"]

    base_scene.clear_scene()
    _TITLE_BULB_MESHES.clear()
    scene = bpy.context.scene
    scene.render.engine = base_scene.get_eevee_engine()
    scene.frame_start = 0
    scene.frame_end = total_frames
    scene.render.fps = fps
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 24
        if hasattr(scene.eevee, "use_shadows"):
            scene.eevee.use_shadows = True
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = True
        if hasattr(scene.eevee, "ray_tracing_options"):
            scene.eevee.ray_tracing_options.resolution_scale = "1"
            scene.eevee.ray_tracing_options.screen_trace_quality = 0.5
        if hasattr(scene.eevee, "motion_blur_steps"):
            scene.eevee.motion_blur_steps = 4
        if hasattr(scene.eevee, "motion_blur_max"):
            scene.eevee.motion_blur_max = 48

    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5
    scene.render.motion_blur_position = "CENTER"

    # Color management: enum_items probing is unreliable in background mode, so assign directly (verified: AgX + look accepted)
    vs = scene.view_settings
    for tname in ("AgX", "Filmic", "Standard"):
        try:
            vs.view_transform = tname
            break
        except Exception:
            continue
    for lname in ("AgX - Medium High Contrast", "AgX - Medium Contrast"):
        try:
            vs.look = lname
            break
        except Exception:
            continue
    vs.exposure = 0.25 if LOOK_NIGHT else -0.10
    scene.display_settings.display_device = "sRGB"

    base_scene.build_studio_reflection_world(scene)
    # Warm candle-lit ambient so the whole room glows Christmas-gold instead of cold grey
    if scene.world and scene.world.use_nodes:
        for nd in scene.world.node_tree.nodes:
            if nd.type == "BACKGROUND":
                if LOOK_NIGHT:
                    nd.inputs["Color"].default_value = (0.10, 0.16, 0.38, 1.0)   # deep moonlit navy
                    nd.inputs["Strength"].default_value = 0.045
                else:
                    nd.inputs["Color"].default_value = (1.0, 0.66, 0.38, 1.0)
                    nd.inputs["Strength"].default_value = 0.30
    base_scene.setup_bloom_compositor(scene)
    # Soft festive glow around bulbs / brass / glass highlights
    if scene.compositing_node_group:
        for nd in scene.compositing_node_group.nodes:
            if nd.type == "GLARE":
                base_scene.set_socket(nd, "Threshold", 1.15 if LOOK_NIGHT else 1.6)
                base_scene.set_socket(nd, "Strength", 0.44 if LOOK_NIGHT else 0.34)
                base_scene.set_socket(nd, "Size", 0.56 if LOOK_NIGHT else 0.55)

    templates, lib_mats, manifest = base_scene.load_instrument_library()
    chrome_mat = lib_mats["LibMat_ChromeRail"]
    felt_mat = lib_mats["LibMat_WallFelt"]
    slat_mat = lib_mats["LibMat_WoodSlatDark"]
    trim_mat = lib_mats.get("LibMat_WallRevealDark", lib_mats["LibMat_WallBrassTrim"])
    hardware_dark_mat = lib_mats["LibMat_HardwareDark"]

    col_root = bpy.context.scene.collection
    col_stage = base_scene.get_or_create_collection("01_Stage", col_root)
    col_boards = base_scene.get_or_create_collection("02_Boards", col_root)
    col_rails = base_scene.get_or_create_collection("03_Rails", col_root)
    col_marbles = base_scene.get_or_create_collection("04_Marbles", col_root)
    col_fx = base_scene.get_or_create_collection("05_FX", col_root)

    all_z = [b["board_center"][2] for ball in balls for b in ball["boards"]]
    z_min = min(all_z) - 4.0
    z_max = max(all_z) + 4.0

    rail_bronze_mat = lib_mats.get("LibMat_ResonatorBronze", chrome_mat)
    def create_square_wave_inlay_mesh(name: str, span_x: float, amp_z: float, thick: float = 0.0085, depth_y: float = 0.014) -> bpy.types.Mesh:
        """Build a crisp 3D square-wave pulse trace (_|‾|_|‾|_) geometry inlaid on the bar front."""
        bm = bmesh.new()
        n_cycles = 3
        dx_step = span_x / (n_cycles * 2.0)
        x0 = -0.5 * span_x
        pts_xz = []
        for c_i in range(n_cycles):
            xa = x0 + (2 * c_i) * dx_step
            xm = xa + dx_step
            xb = xm + dx_step
            z_lo, z_hi = -amp_z, +amp_z
            if c_i == 0:
                pts_xz.append((xa, z_lo))
            pts_xz.append((xm, z_lo))
            pts_xz.append((xm, z_hi))
            pts_xz.append((xb, z_hi))
            pts_xz.append((xb, z_lo))

        for k in range(len(pts_xz) - 1):
            ax, az = pts_xz[k]
            bx, bz = pts_xz[k + 1]
            cx = 0.5 * (ax + bx)
            cz = 0.5 * (az + bz)
            seg_len = abs(bx - ax) + thick
            seg_h = abs(bz - az) + thick
            res = bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.scale(bm, vec=(seg_len, depth_y, seg_h), verts=res["verts"])
            bmesh.ops.translate(bm, vec=(cx, 0.0, cz), verts=res["verts"])

        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.0015, segments=1, affect="EDGES", clamp_overlap=True)
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        return mesh

    brass_rod_mat = lib_mats.get("LibMat_WallBrassTrim", rail_bronze_mat)

    term_zs = [bd["board_center"][2] for ball in balls for bd in ball["boards"] if bd.get("is_terminal")]
    z_end_mid = 0.5 * (min(term_zs) + max(term_zs))
    finale_z = z_end_mid - (1.32 if title_style == 'bulb' else 1.27)

    # Modular 3D walnut plank wall + horizontal hero title planks + fluted dark slats
    create_textured_white_wall(
        wall_y, z_min, z_max, brass_rod_mat, col_stage, opening_z=0.56, finale_z=finale_z
    )

    # Opening (shifted right so it clears the left entry rails) & Finale Christmas light titles
    create_custom_title_plaque(wall_y, 0.48 if title_style == 'bulb' else 0.28, 0.56, brass_rod_mat, col_stage, style=title_style, compact=False)
    create_custom_title_plaque(wall_y, 0.0, finale_z, brass_rod_mat, col_stage, style=title_style, compact=True)

    # Christmas pulse palette: ruby red / emerald green / warm gold, cycled bar by bar
    XMAS_PULSE = [
        [1.00, 0.06, 0.04],
        [0.08, 1.00, 0.22],
        [1.00, 0.62, 0.12],
    ]

    WOOD_LAYER_COLORS = [
        ((0.26, 0.14, 0.07, 1.0), (0.14, 0.07, 0.035, 1.0)),   # Layer 0: Warm Walnut
        ((0.56, 0.36, 0.16, 1.0), (0.40, 0.24, 0.095, 1.0)),   # Layer 1: Golden Maple
        ((0.18, 0.07, 0.055, 1.0), (0.09, 0.032, 0.024, 1.0)), # Layer 2: Smoked Rosewood
    ]

    def create_wood_bar_material(name: str, layer_idx: int) -> bpy.types.Material:
        c_light, c_dark = WOOD_LAYER_COLORS[layer_idx % len(WOOD_LAYER_COLORS)]
        m = bpy.data.materials.new(name=name)
        m.use_nodes = True
        nodes = m.node_tree.nodes
        links = m.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        tex_coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (1.5, 14.0, 14.0)  # grain along bar X length

        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "Z"
        wave.inputs["Scale"].default_value = 6.0
        wave.inputs["Distortion"].default_value = 4.5
        wave.inputs["Detail"].default_value = 4.0

        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.20
        ramp.color_ramp.elements[0].color = c_dark
        ramp.color_ramp.elements[1].position = 0.82
        ramp.color_ramp.elements[1].color = c_light

        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.045
        bump.inputs["Distance"].default_value = 0.005

        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.28
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.45
        if "Coat Roughness" in bsdf.inputs:
            bsdf.inputs["Coat Roughness"].default_value = 0.12

        links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
        links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(wave.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return m

    def create_electronic_pulse_material(name: str, rgb: list[float], bar_len: float) -> tuple[bpy.types.Material, bpy.types.Node, bpy.types.Node]:
        """Square-wave light trace: dim idle glow -> on impact a white-hot front sweeps outward + whole trace flashes, then decays."""
        m = bpy.data.materials.new(name=name)
        m.use_nodes = True
        nodes = m.node_tree.nodes
        links = m.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (rgb[0] * 0.30, rgb[1] * 0.30, rgb[2] * 0.30, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.10
        bsdf.inputs["Roughness"].default_value = 0.20

        tc = nodes.new("ShaderNodeTexCoord")
        sep = nodes.new("ShaderNodeSeparateXYZ")
        abs_x = nodes.new("ShaderNodeMath")
        abs_x.operation = "ABSOLUTE"
        pulse_dist = nodes.new("ShaderNodeMath")
        pulse_dist.operation = "SUBTRACT"
        pulse_abs = nodes.new("ShaderNodeMath")
        pulse_abs.operation = "ABSOLUTE"
        pulse_norm = nodes.new("ShaderNodeMath")
        pulse_norm.operation = "DIVIDE"
        pulse_norm.inputs[1].default_value = max(0.06, bar_len * 0.36)
        pulse_clamp = nodes.new("ShaderNodeMath")
        pulse_clamp.operation = "SUBTRACT"
        pulse_clamp.inputs[0].default_value = 1.0
        pulse_clamp.use_clamp = True

        pulse_amp_node = nodes.new("ShaderNodeValue")
        pulse_amp_node.outputs[0].default_value = 0.0
        pulse_radius_node = nodes.new("ShaderNodeValue")
        pulse_radius_node.outputs[0].default_value = 0.0

        front = nodes.new("ShaderNodeMath")          # travelling front * amp
        front.operation = "MULTIPLY"
        flash = nodes.new("ShaderNodeMath")          # uniform flash of the whole trace
        flash.operation = "MULTIPLY"
        flash.inputs[1].default_value = 0.40
        sum_a = nodes.new("ShaderNodeMath")
        sum_a.operation = "ADD"
        add_idle = nodes.new("ShaderNodeMath")       # low idle so the flash reads clearly
        add_idle.operation = "ADD"
        add_idle.inputs[1].default_value = 1.0 if LOOK_NIGHT else 0.55
        white_gain = nodes.new("ShaderNodeMath")     # white-hot only where the front is
        white_gain.operation = "MULTIPLY"
        white_gain.inputs[1].default_value = 0.65

        em_col = nodes.new("ShaderNodeEmission")
        em_col.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        em_hot = nodes.new("ShaderNodeEmission")
        em_hot.inputs["Color"].default_value = (1.0, 0.96, 0.82, 1.0)
        add_sh = nodes.new("ShaderNodeAddShader")
        add_sh2 = nodes.new("ShaderNodeAddShader")

        links.new(tc.outputs["Object"], sep.inputs["Vector"])
        links.new(sep.outputs["X"], abs_x.inputs[0])
        links.new(abs_x.outputs["Value"], pulse_dist.inputs[0])
        links.new(pulse_radius_node.outputs["Value"], pulse_dist.inputs[1])
        links.new(pulse_dist.outputs["Value"], pulse_abs.inputs[0])
        links.new(pulse_abs.outputs["Value"], pulse_norm.inputs[0])
        links.new(pulse_norm.outputs["Value"], pulse_clamp.inputs[1])
        links.new(pulse_clamp.outputs["Value"], front.inputs[0])
        links.new(pulse_amp_node.outputs["Value"], front.inputs[1])
        links.new(pulse_amp_node.outputs["Value"], flash.inputs[0])
        links.new(front.outputs["Value"], sum_a.inputs[0])
        links.new(flash.outputs["Value"], sum_a.inputs[1])
        links.new(sum_a.outputs["Value"], add_idle.inputs[0])
        links.new(add_idle.outputs["Value"], em_col.inputs["Strength"])
        links.new(front.outputs["Value"], white_gain.inputs[0])
        links.new(white_gain.outputs["Value"], em_hot.inputs["Strength"])
        links.new(bsdf.outputs["BSDF"], add_sh.inputs[0])
        links.new(em_col.outputs["Emission"], add_sh.inputs[1])
        links.new(add_sh.outputs["Shader"], add_sh2.inputs[0])
        links.new(em_hot.outputs["Emission"], add_sh2.inputs[1])
        links.new(add_sh2.outputs["Shader"], out.inputs["Surface"])
        return m, pulse_amp_node, pulse_radius_node

    # Spawn handcrafted wood bars (with inlaid LED strip + wall mounting rod), rails, and glass marbles
    for ball in balls:
        b_idx = int(ball["ball_idx"])
        pal = ball["palette"]
        b_col = base_scene.get_or_create_collection(f"Layer_{b_idx}_{pal['name']}", col_boards)
        body_mat = create_wood_bar_material(f"WoodBarMat_{b_idx}", b_idx)

        # Snap integer hit-frame marble positions to exact analytical hit_center & center boards laterally on hit_center
        s_by_frame = {int(s["frame"]): s["position"] for s in ball["samples"]}
        hit_frames_set = {int(bd["frame_hit"]) for bd in ball["boards"]}
        for bd in ball["boards"]:
            if bd.get("is_terminal"):
                continue
            fh = int(bd["frame_hit"])
            hc = bd.get("hit_center")
            if not hc or fh not in s_by_frame:
                continue
            cur_p = s_by_frame[fh]
            dx_h = hc[0] - cur_p[0]
            dy_h = hc[1] - cur_p[1]
            dz_h = hc[2] - cur_p[2]
            s_by_frame[fh] = [hc[0], hc[1], hc[2]]
            for adj_f, w_adj in ((fh - 1, 0.32), (fh + 1, 0.32)):
                if adj_f in s_by_frame and adj_f not in hit_frames_set:
                    ap = s_by_frame[adj_f]
                    s_by_frame[adj_f] = [ap[0] + w_adj * dx_h, ap[1] + w_adj * dy_h, ap[2] + w_adj * dz_h]

        for s in ball["samples"]:
            s["position"] = s_by_frame[int(s["frame"])]

        for bd in ball["boards"]:
            if bd.get("is_terminal"):
                continue
            fh = int(bd["frame_hit"])
            hc = bd.get("hit_center")
            if not hc:
                continue
            n_v = bd["n"]
            u_v = bd["u"]
            th_v = float(bd["thickness"])
            len_v = float(bd["length"])
            c_orig = bd["board_center"]
            c_ideal = [
                hc[0] - n_v[0] * (radius + 0.5 * th_v),
                hc[1] - n_v[1] * (radius + 0.5 * th_v),
                hc[2] - n_v[2] * (radius + 0.5 * th_v),
            ]
            for alpha_c in (0.0, 0.25, 0.50, 0.75, 1.0):
                c_cand = [
                    c_ideal[0] + alpha_c * (c_orig[0] - c_ideal[0]),
                    c_ideal[1] + alpha_c * (c_orig[1] - c_ideal[1]),
                    c_ideal[2] + alpha_c * (c_orig[2] - c_ideal[2]),
                ]
                ok_c = True
                for f_k, pos_k in s_by_frame.items():
                    if abs(f_k - fh) <= 3:
                        continue
                    dx_k = pos_k[0] - c_cand[0]
                    dy_k = pos_k[1] - c_cand[1]
                    dz_k = pos_k[2] - c_cand[2]
                    if dx_k * dx_k + dy_k * dy_k + dz_k * dz_k > (0.5 * len_v + radius + 0.05) ** 2:
                        continue
                    du_k = abs(dx_k * u_v[0] + dy_k * u_v[1] + dz_k * u_v[2]) - 0.5 * len_v
                    dn_k = abs(dx_k * n_v[0] + dy_k * n_v[1] + dz_k * n_v[2]) - 0.5 * th_v
                    if math.hypot(max(0.0, du_k), max(0.0, dn_k)) < radius + 0.006:
                        ok_c = False
                        break
                if ok_c:
                    bd["board_center"] = c_cand
                    break

        # Reusable shadowless per-ball hit-flash point light (splashes colored bounce light onto wood/marbles on impact)
        hl_data = bpy.data.lights.new(f"HitFlashLight_{b_idx}", type="POINT")
        hl_data.energy = 0.0
        hl_data.shadow_soft_size = 0.18
        hl_data.use_shadow = False
        hl_obj = bpy.data.objects.new(f"HitFlashLight_{b_idx}", hl_data)
        col_fx.objects.link(hl_obj)
        hl_data.keyframe_insert(data_path="energy", frame=0)

        for b_pos_idx, board in enumerate(ball["boards"]):
            idx = int(board["index"])
            b_len = float(board["length"])
            b_wy = float(board["width_y"]) * 0.85
            b_th = float(board["thickness"])

            core_mesh = base_scene.create_hard_beveled_box(
                f"CleanBarMesh_{b_idx}_{idx:03d}", b_len, b_wy, b_th
            )
            obj = bpy.data.objects.new(f"Bar_{b_idx}_{idx:03d}_{board.get('pitch_name', '')}", core_mesh)
            b_col.objects.link(obj)
            obj.data.materials.append(body_mat)

            # Bold 3D Square-Wave Electronic Pulse Trace (_|‾|_|‾|_) inlaid directly on natural wood front face (no black backing box)
            bar_pulse_rgb = XMAS_PULSE[(b_idx + idx) % len(XMAS_PULSE)]
            led_mesh = create_square_wave_inlay_mesh(
                f"BarWaveMesh_{b_idx}_{idx:03d}", b_len * 0.82, b_th * 0.20, thick=0.0088, depth_y=0.010
            )
            led_obj = bpy.data.objects.new(f"BarWave_{b_idx}_{idx:03d}", led_mesh)
            b_col.objects.link(led_obj)
            led_mat, p_amp_node, p_rad_node = create_electronic_pulse_material(
                f"BarPulseMat_{b_idx}_{idx:03d}", bar_pulse_rgb, b_len * 0.82
            )
            led_obj.data.materials.append(led_mat)
            led_obj.parent = obj
            led_obj.location = (0.0, -b_wy * 0.502, 0.0)

            u = Vector(board["u"]).normalized()
            v = Vector((0.0, 1.0, 0.0))
            n = Vector(board["n"]).normalized()
            rot_mat = Matrix(((u.x, v.x, n.x), (u.y, v.y, n.y), (u.z, v.z, n.z)))
            c_vec = Vector(board["board_center"])
            obj.location = c_vec
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = rot_mat.to_quaternion()

            # Structural rear mounting rod + wall flange (embeds cleanly into staggered wooden planks)
            pin_a = c_vec + Vector((0.0, b_wy * 0.42, 0.0))
            pin_b = Vector((c_vec.x, wall_y + 0.025, c_vec.z))
            base_scene.create_cylinder_between(
                f"BarPin_{b_idx}_{idx:03d}",
                pin_a,
                pin_b,
                0.011,
                hardware_dark_mat,
                b_col,
                vertices=16,
            )
            base_scene.create_wall_flange(
                f"BarFlange_{b_idx}_{idx:03d}",
                (c_vec.x, wall_y + 0.003, c_vec.z),
                0.024,
                brass_rod_mat,
                b_col,
            )

            # Hit recoil & LED emissive pulse
            fh = int(board["frame_hit"])
            vel = float(board.get("velocity", 90.0))
            vel_scale = 0.55 + 0.65 * max(0.0, min(1.0, vel / 127.0))

            # Keep board at exact contact position on hit frame fh; recoil downward only after impact (fh + 2)
            obj.location = c_vec
            obj.keyframe_insert(data_path="location", frame=max(0, fh - 1))
            obj.keyframe_insert(data_path="location", frame=fh)
            obj.location = c_vec - n * (0.018 * vel_scale)
            obj.keyframe_insert(data_path="location", frame=fh + 2)
            obj.location = c_vec
            obj.keyframe_insert(data_path="location", frame=min(total_frames, fh + 10))

            # Impact pulse: front sweeps centre->edges in 6 frames, whole trace flashes, then decays over ~16 frames
            half_span = b_len * 0.46
            p_rad_node.outputs[0].default_value = 0.0
            p_rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=max(0, fh - 1))
            p_rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=fh)
            p_rad_node.outputs[0].default_value = half_span
            p_rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=min(total_frames, fh + 7))
            p_rad_node.outputs[0].default_value = 0.0
            p_rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=min(total_frames, fh + 8))

            amp_pk = 20.0 * vel_scale
            p_amp_node.outputs[0].default_value = 0.0
            p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=max(0, fh - 2))
            p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=max(0, fh - 1))
            p_amp_node.outputs[0].default_value = amp_pk
            p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=fh)
            p_amp_node.outputs[0].default_value = 0.55 * amp_pk
            p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=min(total_frames, fh + 6))
            p_amp_node.outputs[0].default_value = 0.18 * amp_pk
            p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=min(total_frames, fh + 12))
            p_amp_node.outputs[0].default_value = 0.0
            p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=min(total_frames, fh + 18))

            if not board.get("is_terminal"):
                flash_peak = (18.0 if LOOK_NIGHT else 8.0) * vel_scale
                f_pre = max(0, fh - 1)
                f_mid = min(total_frames, fh + 4)
                f_end = min(total_frames, fh + 9)
                if b_pos_idx + 1 < len(ball["boards"]):
                    f_end = min(f_end, int(ball["boards"][b_pos_idx + 1]["frame_hit"]) - 2)
                f_mid = min(f_mid, max(fh, f_end - 1))

                hl_obj.location = (c_vec.x, c_vec.y - 0.22, c_vec.z)
                hl_data.color = tuple(bar_pulse_rgb)
                hl_data.energy = 0.0
                hl_obj.keyframe_insert(data_path="location", frame=f_pre)
                hl_data.keyframe_insert(data_path="color", frame=f_pre)
                hl_data.keyframe_insert(data_path="energy", frame=f_pre)

                hl_data.energy = flash_peak
                hl_data.keyframe_insert(data_path="energy", frame=fh)

                if f_mid > fh:
                    hl_data.energy = 0.42 * flash_peak
                    hl_data.keyframe_insert(data_path="energy", frame=f_mid)

                hl_obj.location = (c_vec.x, c_vec.y - 0.22, c_vec.z)
                hl_data.color = tuple(bar_pulse_rgb)
                hl_data.energy = 0.0
                hl_obj.keyframe_insert(data_path="location", frame=max(fh + 1, f_end))
                hl_data.keyframe_insert(data_path="color", frame=max(fh + 1, f_end))
                hl_data.keyframe_insert(data_path="energy", frame=max(fh + 1, f_end))

        for r in ball["rails"]:
            rid = r["id"]
            create_smooth_pipe_polyline(f"{rid}_front", r["pipe_front"], r["pipe_radius"], rail_bronze_mat, col_rails)
            create_smooth_pipe_polyline(f"{rid}_back", r["pipe_back"], r["pipe_radius"], rail_bronze_mat, col_rails)
            for t_idx, tie_pts in enumerate(r.get("ties", [])):
                create_smooth_pipe_polyline(f"{rid}_tie_{t_idx:02d}", tie_pts, 0.0085, hardware_dark_mat, col_rails)

            # Wall support brackets: 3 stations per rail with automatic L-bracket Z-offset to clear deeper marbles & bars
            back_pts = r["pipe_back"]
            n_bp = len(back_pts)
            stations = sorted(set([min(n_bp - 2, 4), n_bp // 2, max(2, n_bp - 5)]))
            n_rods = 0
            for k_i in stations:
                px_, py_, pz_ = back_pts[k_i]
                chosen_dz = None
                for dz_cand in (0.0, 0.15, -0.15, 0.25, -0.25, 0.35, -0.35):
                    tz_ = pz_ + dz_cand
                    blocked = False
                    for o_ball in balls:
                        for s_ in o_ball["samples"][::2]:
                            bx_, by_, bz_ = s_["position"]
                            if by_ > py_ + 0.04 and (bx_ - px_) ** 2 + (bz_ - tz_) ** 2 < (radius + 0.032) ** 2:
                                blocked = True
                                break
                        if blocked:
                            break
                        for bd_ in o_ball["boards"]:
                            bc_ = bd_["board_center"]
                            half_l = 0.55 * float(bd_["length"])
                            if bc_[1] > py_ + 0.02 and abs(bc_[0] - px_) < half_l and abs(bc_[2] - tz_) < 0.085:
                                blocked = True
                                break
                        if blocked:
                            break
                    if not blocked:
                        chosen_dz = dz_cand
                        break
                if chosen_dz is None:
                    chosen_dz = 0.0
                tz_ = pz_ + chosen_dz
                if abs(chosen_dz) > 1e-4:
                    elbow = Vector((px_, min(wall_y - 0.02, py_ + 0.06), tz_))
                    base_scene.create_cylinder_between(
                        f"{rid}_riser_{n_rods:02d}",
                        Vector((px_, py_, pz_)),
                        elbow,
                        0.0080,
                        hardware_dark_mat,
                        col_rails,
                        vertices=12,
                    )
                    rod_start = elbow
                else:
                    rod_start = Vector((px_, py_, pz_))
                base_scene.create_cylinder_between(
                    f"{rid}_support_{n_rods:02d}",
                    rod_start,
                    Vector((px_, wall_y + 0.025, tz_)),
                    0.0080,
                    hardware_dark_mat,
                    col_rails,
                    vertices=12,
                )
                base_scene.create_wall_flange(
                    f"{rid}_support_flange_{n_rods:02d}",
                    (px_, wall_y + 0.003, tz_),
                    0.020,
                    brass_rod_mat,
                    col_rails,
                )
                n_rods += 1
            print(f"rail {rid}: {n_rods} wall support rods")

        # Colored glass marble + core point light
        mmesh = bpy.data.meshes.new(f"MarbleMesh_{b_idx}")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=40, radius=radius)
        bm.to_mesh(mmesh)
        bm.free()
        for poly in mmesh.polygons:
            poly.use_smooth = True

        m_obj = bpy.data.objects.new(f"Marble_{b_idx}_{pal['name']}", mmesh)
        col_marbles.objects.link(m_obj)
        m_mat = create_colored_marble_material(f"MarbleMat_{b_idx}", pal["color"])
        m_obj.data.materials.append(m_mat)
        m_obj.rotation_mode = "QUATERNION"

        # Hand-blown cat's-eye ribbon + 3 tiny trapped air bubbles rotate WITH the marble so spin reads cleanly
        SWIRL_COLORS = [
            (0.95, 0.78, 0.34),  # Champagne Gold swirl inside Emerald marble
            (0.86, 0.18, 0.06),  # Deep Ruby-Copper swirl inside Amber marble
            (0.78, 0.94, 1.00),  # Silver-Ice Pearl swirl inside Sapphire marble
        ]
        sw_col = SWIRL_COLORS[b_idx % len(SWIRL_COLORS)]
        swirl_mat = bpy.data.materials.new(f"MarbleSwirlMat_{b_idx}")
        swirl_mat.use_nodes = True
        sb = next(nd for nd in swirl_mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
        sb.inputs["Base Color"].default_value = (sw_col[0], sw_col[1], sw_col[2], 1.0)
        sb.inputs["Metallic"].default_value = 0.55
        sb.inputs["Roughness"].default_value = 0.14
        if "Coat Weight" in sb.inputs:
            sb.inputs["Coat Weight"].default_value = 1.0
        if "Emission Color" in sb.inputs:
            sb.inputs["Emission Color"].default_value = (sw_col[0], sw_col[1], sw_col[2], 1.0)
            sb.inputs["Emission Strength"].default_value = 0.12 if LOOK_NIGHT else 0.05

        def _cat_eye_mesh(R: float):
            bmr = bmesh.new()
            n_seg = 26
            rings_v = []
            for i in range(n_seg + 1):
                t = i / n_seg
                z = (t - 0.5) * 1.36 * R
                sin_t = max(0.06, math.sin(math.pi * t))
                w_r = 0.46 * R * (sin_t ** 0.72)
                th_r = 0.042 * R * (sin_t ** 0.5) + 0.0008
                phi = math.pi * 1.1 * (t - 0.5)
                cp, sp = math.cos(phi), math.sin(phi)
                local_pts = ((w_r, th_r), (-w_r, th_r), (-w_r, -th_r), (w_r, -th_r))
                rings_v.append([bmr.verts.new((lx * cp - ly * sp, lx * sp + ly * cp, z)) for lx, ly in local_pts])
            for i in range(n_seg):
                r0, r1 = rings_v[i], rings_v[i + 1]
                for j in range(4):
                    bmr.faces.new((r0[j], r0[(j + 1) % 4], r1[(j + 1) % 4], r1[j]))
            bmr.faces.new(tuple(reversed(rings_v[0])))
            bmr.faces.new(tuple(rings_v[-1]))
            bmesh.ops.recalc_face_normals(bmr, faces=bmr.faces)
            rm = bpy.data.meshes.new(f"MarbleCatEyeMesh_{b_idx}")
            bmr.to_mesh(rm)
            bmr.free()
            for pl in rm.polygons:
                pl.use_smooth = True
            return rm

        cat_obj = bpy.data.objects.new(f"MarbleCatEye_{b_idx}", _cat_eye_mesh(radius))
        col_marbles.objects.link(cat_obj)
        cat_obj.data.materials.append(swirl_mat)
        cat_obj.parent = m_obj
        cat_obj.rotation_euler = (math.radians(24.0), math.radians(18.0), 0.0)

        bubble_mat = bpy.data.materials.new(f"MarbleBubbleMat_{b_idx}")
        bubble_mat.use_nodes = True
        bb_node = next(nd for nd in bubble_mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
        bb_node.inputs["Base Color"].default_value = (0.95, 0.98, 1.0, 1.0)
        bb_node.inputs["Metallic"].default_value = 0.25
        bb_node.inputs["Roughness"].default_value = 0.02
        if "Emission Strength" in bb_node.inputs:
            bb_node.inputs["Emission Strength"].default_value = 0.0

        for bk, (bx_u, by_u, bz_u, br_u) in enumerate((
            (0.36, 0.16, 0.22, 0.070),
            (-0.28, -0.20, 0.15, 0.055),
            (0.14, -0.26, -0.30, 0.045),
        )):
            b_mesh = bpy.data.meshes.new(f"MarbleBubbleMesh_{b_idx}_{bk}")
            bmb = bmesh.new()
            bmesh.ops.create_uvsphere(bmb, u_segments=14, v_segments=10, radius=radius * br_u)
            bmb.to_mesh(b_mesh)
            bmb.free()
            for pl in b_mesh.polygons:
                pl.use_smooth = True
            b_obj = bpy.data.objects.new(f"MarbleBubble_{b_idx}_{bk}", b_mesh)
            col_marbles.objects.link(b_obj)
            b_obj.data.materials.append(bubble_mat)
            b_obj.parent = m_obj
            b_obj.location = (radius * bx_u, radius * by_u, radius * bz_u)

        # Natural roll spin: horizontal travel rotates around Y; vertical travel adds subtle X tumble
        q_cur = Quaternion((0.92, 0.18, 0.31, 0.12))
        q_cur.normalize()
        prev_pos = None
        for s in ball["samples"]:
            f = int(s["frame"])
            px, py, pz = s["position"]
            cur_pos = Vector((px, py, pz))
            if prev_pos is not None:
                th_y = 1.05 * (cur_pos.x - prev_pos.x) / radius
                th_x = 0.38 * (cur_pos.z - prev_pos.z) / radius
                if abs(th_y) > 1e-9 or abs(th_x) > 1e-9:
                    q_cur = Quaternion((0.0, 1.0, 0.0), th_y) @ Quaternion((1.0, 0.0, 0.0), th_x) @ q_cur
                    q_cur.normalize()
            m_obj.location = cur_pos
            m_obj.rotation_quaternion = q_cur
            m_obj.keyframe_insert(data_path="location", frame=f)
            m_obj.keyframe_insert(data_path="rotation_quaternion", frame=f)
            prev_pos = cur_pos

    # Soft indoor warm Christmas lighting (diffused large-radius key shadow + shadowless warm area fill)
    key_data = bpy.data.lights.new("StudioKeyLight", type="SPOT")
    if LOOK_NIGHT:
        # Night: symmetric central warm candlelight pool illuminates left rails + center marbles + boards
        key_data.energy = 2700.0
        key_data.spot_size = math.radians(44.0)
        key_data.spot_blend = 1.0
        key_data.shadow_soft_size = 1.1
        key_data.use_shadow = True
        key_data.color = (1.0, 0.72, 0.38)
    else:
        key_data.energy = 5800.0
        key_data.spot_size = math.radians(52.0)
        key_data.spot_blend = 0.85
        key_data.shadow_soft_size = 1.45
        key_data.use_shadow = True
        key_data.color = (1.0, 0.72, 0.42)
    key_obj = bpy.data.objects.new("StudioKeyLight", key_data)
    col_stage.objects.link(key_obj)

    fill_data = bpy.data.lights.new("StudioRimLight", type="AREA")
    if LOOK_NIGHT:
        fill_data.energy = 340.0      # cool window moonlight: gives glass & wood slats a subtle cold rim
        fill_data.size = 6.0
        fill_data.use_shadow = False
        fill_data.color = (0.36, 0.54, 1.0)
    else:
        fill_data.energy = 1150.0
        fill_data.size = 7.5
        fill_data.use_shadow = False
        fill_data.color = (1.0, 0.62, 0.38)
    fill_obj = bpy.data.objects.new("StudioRimLight", fill_data)
    col_stage.objects.link(fill_obj)

    # 16:9 Look-ahead Camera with slight 3/4 angle & subtle DOF
    cam_data = bpy.data.cameras.new("MainCamera16x9")
    cam_data.lens = 42.0
    cam_data.clip_end = 250.0
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 4.2
    cam = bpy.data.objects.new("MainCamera16x9", cam_data)
    col_root.objects.link(cam)
    scene.camera = cam

    target_empty = bpy.data.objects.new("CamTarget16x9", None)
    col_root.objects.link(target_empty)
    cam_data.dof.focus_object = target_empty
    c_cons = cam.constraints.new(type="TRACK_TO")
    c_cons.target = target_empty
    c_cons.track_axis = "TRACK_NEGATIVE_Z"
    c_cons.up_axis = "UP_Y"

    # Smooth average Z across the 3 balls
    raw_z = []
    for f in range(total_frames + 1):
        zs = [b["samples"][f]["position"][2] for b in balls]
        raw_z.append(sum(zs) / len(zs))
    hw = 24
    weights = [math.exp(-0.5 * (k / (hw * 0.45)) ** 2) for k in range(-hw, hw + 1)]
    for f in range(total_frames + 1):
        az = ws = 0.0
        for idx_w, off in enumerate(range(-hw, hw + 1)):
            j = max(0, min(total_frames, f + off))
            w = weights[idx_w]
            az += w * raw_z[j]
            ws += w
        tz = az / ws
        cam.location = (0.36, -5.45, tz + 0.42)
        cam.keyframe_insert(data_path="location", frame=f)
        target_empty.location = (0.0, 0.0, tz - 0.15)
        target_empty.keyframe_insert(data_path="location", frame=f)
        if f % 10 == 0 or f == total_frames:
            if LOOK_NIGHT:
                key_obj.location = (-1.1, -4.8, tz + 2.8)
                key_dir = Vector((0.0, 0.3, tz - 0.15)) - Vector(key_obj.location)
                fill_obj.location = (3.8, -2.4, tz + 1.8)
            else:
                key_obj.location = (-3.6, -4.4, tz + 3.4)
                key_dir = Vector((0.4, 0.6, tz - 0.5)) - Vector(key_obj.location)
                fill_obj.location = (3.8, -1.8, tz + 1.4)
            key_obj.rotation_euler = key_dir.to_track_quat("-Z", "Y").to_euler()
            key_obj.keyframe_insert(data_path="location", frame=f)
            key_obj.keyframe_insert(data_path="rotation_euler", frame=f)
            fill_obj.rotation_euler = (math.radians(74.0), 0.0, math.radians(54.0))
            fill_obj.keyframe_insert(data_path="location", frame=f)

    # Attach synced audio to VSE (Blender 5.x strips)
    if audio_path and os.path.exists(audio_path):
        scene.sequence_editor_create()
        seq = scene.sequence_editor
        strips = getattr(seq, "strips", None)
        if strips is None:
            strips = getattr(seq, "sequences", None)
        if strips is not None:
            audio_start = int(round(float(plan.get("pre_roll", 0.0)) * fps))
            strips.new_sound("MarbleAudio", os.path.abspath(audio_path), channel=1, frame_start=audio_start)
        scene.use_audio = True
        scene.sync_mode = "AUDIO_SYNC"

    os.makedirs(os.path.dirname(os.path.abspath(out_blend)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend))
    print(f"Saved 16:9 multi-ball scene: {out_blend} (frames 0..{total_frames}, 3 balls)")

    if frames_dir and check_frames:
        os.makedirs(frames_dir, exist_ok=True)
        for cf in check_frames:
            scene.frame_current = cf
            scene.render.filepath = os.path.join(frames_dir, f"check_{cf:04d}.png")
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    physics_json = args[0] if len(args) >= 1 else "projects/jingle-bells-marble-music/artifacts/glass_marbles_synth_3ball_16x9_physics.json"
    out_blend = args[1] if len(args) >= 2 else "projects/jingle-bells-marble-music/renders/glass_marbles_synth_3ball_16x9.blend"
    audio_path = args[2] if len(args) >= 3 else "projects/jingle-bells-marble-music/assets/audio/glass_marbles_on_wood_full.wav"
    frames_dir = args[3] if len(args) >= 4 else "projects/jingle-bells-marble-music/renders/frames_synth_3ball_16x9_check"
    check_frames = [int(x) for x in args[4].split(",")] if len(args) >= 5 else [72, 600, 1800, 3000]
    look_arg = args[5] if len(args) >= 6 else "warm"
    title_arg = args[6] if len(args) >= 7 else "bulb"
    build_multiball_scene(physics_json, out_blend, audio_path=audio_path, frames_dir=frames_dir, check_frames=check_frames, look=look_arg, title_style=title_arg)
