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
    finale_h: float = 1.04,
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
        hero_specs.append((finale_z, finale_h, "finale"))

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
    # 3. Left & right fluted architectural dark-walnut 3D slats (segmented into modular 12m battens)
    slat_seg_h = 12.0
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

    # 4. (Watermark text replaced by Christmas bulb dot-matrix plaques built in scene assembly)

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

    return fobj, bulb_mats, hero_specs, z_bot, z_top, watermark_zs


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
    wall_y: float,
    anchor_x: float,
    anchor_z: float,
    brass_mat,
    collection,
    compact: bool = False,
    narrow_hero: bool = False,
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
    elif narrow_hero:
        lines = [
            ("MARBLE", 0.034, 0.0128, "cycle"),
            ("CHRISTMAS", 0.034, 0.0128, "cycle"),
            ("GLASS MARBLES ON WOOD", 0.0182, 0.0070, "cycle"),
            ("ARTIST : ZERO.SOE", 0.0175, 0.0068, "gold"),
        ]
        gaps = [0.058, 0.072, 0.058]
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


def create_watermark_bulb_materials(strength: float) -> list[bpy.types.Material]:
    """Four shared emissive materials (Gold, Red, Gold, Green) for mid-wall Christmas bulb watermarks."""
    mats = []
    for bi, (bn, bc) in enumerate((
        ("Gold", (1.0, 0.58, 0.14)),
        ("Red", (1.0, 0.03, 0.02)),
        ("Gold", (1.0, 0.58, 0.14)),
        ("Green", (0.04, 0.90, 0.15)),
    )):
        m = bpy.data.materials.new(f"WatermarkBulbMat_{bi}_{bn}")
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        o_ = nt.nodes.new("ShaderNodeOutputMaterial")
        e_ = nt.nodes.new("ShaderNodeEmission")
        e_.inputs["Color"].default_value = (bc[0], bc[1], bc[2], 1.0)
        e_.inputs["Strength"].default_value = strength
        nt.links.new(e_.outputs["Emission"], o_.inputs["Surface"])
        mats.append(m)
    return mats


def create_bulb_watermark_plaque(
    wall_y: float,
    anchor_x: float,
    anchor_z: float,
    collection: bpy.types.Collection,
    wm_idx: int,
    wm_mats: list[bpy.types.Material],
) -> None:
    """Mid-wall 2-line Christmas bulb dot-matrix watermark merged into 4 color-grouped smooth meshes."""
    t_bm = bmesh.new()
    bmesh.ops.create_uvsphere(t_bm, u_segments=12, v_segments=8, radius=1.0)
    unit_verts = [(float(v.co.x), float(v.co.y), float(v.co.z)) for v in t_bm.verts]
    unit_faces = [[v.index for v in f.verts] for f in t_bm.faces]
    t_bm.free()

    lines = [
        ("MARBLE CHRISTMAS", 0.0255, 0.0096),
        ("GLASS MARBLES ON WOOD", 0.0185, 0.0070),
    ]
    gaps = [0.068]
    heights = [7 * ln[1] for ln in lines]
    total_h = sum(heights) + sum(gaps)
    cursor_z = anchor_z + 0.5 * total_h

    color_bms = [bmesh.new() for _ in range(len(wm_mats))]
    bulb_i = 0
    for li, (text, pitch, rad) in enumerate(lines):
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
                    bx = anchor_x + (col_cursor + c_i - 0.5 * (total_cols - 1)) * pitch
                    by = wall_y + 0.004 - 0.8 * rad
                    bz = top_row_center - r_i * pitch
                    cbm = color_bms[bulb_i % len(wm_mats)]
                    new_v = [cbm.verts.new((ux * rad + bx, uy * rad + by, uz * rad + bz)) for ux, uy, uz in unit_verts]
                    for f_idxs in unit_faces:
                        cbm.faces.new([new_v[idx_v] for idx_v in f_idxs])
                    bulb_i += 1
            col_cursor += gw + 1
        cursor_z -= heights[li] + (gaps[li] if li < len(gaps) else 0.0)

    for bi, cbm in enumerate(color_bms):
        if not cbm.verts:
            cbm.free()
            continue
        mesh = bpy.data.meshes.new(f"WatermarkBulbs_{wm_idx:02d}_c{bi}_Mesh")
        cbm.to_mesh(mesh)
        cbm.free()
        for pl in mesh.polygons:
            pl.use_smooth = True
        mesh.materials.append(wm_mats[bi])
        obj = bpy.data.objects.new(f"WatermarkBulbs_{wm_idx:02d}_c{bi}", mesh)
        collection.objects.link(obj)


def create_custom_title_plaque(
    wall_y: float,
    anchor_x: float,
    anchor_z: float,
    brass_mat,
    collection,
    style: str = "bulb",
    compact: bool = False,
    narrow_hero: bool = False,
):
    """style='bulb' (v14 string-light dot-matrix, default) or 'neon' (v15 vector neon letters + garlands)."""
    if style == "neon":
        return create_neon_title_plaque(wall_y, anchor_x, anchor_z, brass_mat, collection)
    return create_bulb_title_plaque(
        wall_y, anchor_x, anchor_z, brass_mat, collection, compact=compact, narrow_hero=narrow_hero
    )


def create_rail_materials() -> tuple[bpy.types.Material, bpy.types.Material]:
    """Create distinct satin champagne-chrome (center melody) and antique copper-bronze (side drum) rail shaders."""
    # Center Melody: bright brushed satin champagne-chrome
    mc = bpy.data.materials.new("Mat_Rail_CenterMelody")
    mc.use_nodes = True
    nt = mc.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.88, 0.86, 0.82, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.96
    bsdf.inputs["Roughness"].default_value = 0.12
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.25
    if "Coat Roughness" in bsdf.inputs:
        bsdf.inputs["Coat Roughness"].default_value = 0.06
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 80.0
    noise.inputs["Detail"].default_value = 4.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.015
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Side Drums: warm burnished antique copper-bronze
    ms = bpy.data.materials.new("Mat_Rail_SideDrum")
    ms.use_nodes = True
    nt2 = ms.node_tree
    nt2.nodes.clear()
    out2 = nt2.nodes.new("ShaderNodeOutputMaterial")
    bsdf2 = nt2.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf2.inputs["Base Color"].default_value = (0.48, 0.22, 0.09, 1.0)
    bsdf2.inputs["Metallic"].default_value = 0.88
    bsdf2.inputs["Roughness"].default_value = 0.28
    tc2 = nt2.nodes.new("ShaderNodeTexCoord")
    mp2 = nt2.nodes.new("ShaderNodeMapping")
    mp2.inputs["Scale"].default_value = (1.0, 1.0, 6.0)
    noise2 = nt2.nodes.new("ShaderNodeTexNoise")
    noise2.inputs["Scale"].default_value = 120.0
    noise2.inputs["Detail"].default_value = 5.0
    bump2 = nt2.nodes.new("ShaderNodeBump")
    bump2.inputs["Strength"].default_value = 0.040
    bump2.inputs["Distance"].default_value = 0.003
    nt2.links.new(tc2.outputs["Object"], mp2.inputs["Vector"])
    nt2.links.new(mp2.outputs["Vector"], noise2.inputs["Vector"])
    nt2.links.new(noise2.outputs["Fac"], bump2.inputs["Height"])
    nt2.links.new(bump2.outputs["Normal"], bsdf2.inputs["Normal"])
    nt2.links.new(bsdf2.outputs["BSDF"], out2.inputs["Surface"])
    return mc, ms


def _pt_seg_dist(p: tuple[float, float, float], a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    apx, apy, apz = p[0] - a[0], p[1] - a[1], p[2] - a[2]
    d2 = abx * abx + aby * aby + abz * abz
    t = 0.0 if d2 < 1e-12 else max(0.0, min(1.0, (apx * abx + apy * aby + apz * abz) / d2))
    dx = apx - t * abx
    dy = apy - t * aby
    dz = apz - t * abz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _pt_obb_dist(
    pt: tuple[float, float, float],
    center: tuple[float, float, float],
    u: tuple[float, float, float],
    v: tuple[float, float, float],
    n: tuple[float, float, float],
    he: tuple[float, float, float],
) -> float:
    dx, dy, dz = pt[0] - center[0], pt[1] - center[1], pt[2] - center[2]
    cu = dx * u[0] + dy * u[1] + dz * u[2]
    cv = dx * v[0] + dy * v[1] + dz * v[2]
    cn = dx * n[0] + dy * n[1] + dz * n[2]
    qu = max(0.0, abs(cu) - he[0])
    qv = max(0.0, abs(cv) - he[1])
    qn = max(0.0, abs(cn) - he[2])
    return math.sqrt(qu * qu + qv * qv + qn * qn)


def build_support_obstacles(
    balls: list[dict],
    drum_balls: list[dict],
    radius: float,
    total_frames: int,
    finale_spec: dict | None = None,
) -> dict:
    """Index snapped 5-ball trajectories (Z-binned), OBB boards, and rails for reverse support search."""
    ball_tracks = []
    board_entries = []
    rail_entries = []
    z_bins_boards: dict[int, list[int]] = {}

    def _add_ball(tag: tuple[str, int], rad_v: float, b_obj: dict):
        samps = [tuple(s["position"]) for s in b_obj["samples"]]
        z_bins: dict[int, list[int]] = {}
        for f_i, p in enumerate(samps):
            z_bins.setdefault(int(math.floor(p[2])), []).append(f_i)
        ball_tracks.append((tag, rad_v, samps, z_bins))

        for bd in b_obj["boards"]:
            bidx = int(bd["index"])
            bc = tuple(bd["board_center"])
            u = tuple(bd["u"])
            n = tuple(bd["n"])
            he = (0.5 * float(bd["length"]), 0.5 * float(bd["width_y"]) * 0.85, 0.5 * float(bd["thickness"]))
            bi = len(board_entries)
            board_entries.append((tag, bidx, bc, u, n, he))
            zb0 = int(math.floor(bc[2] - he[0] - 0.15))
            zb1 = int(math.floor(bc[2] + he[0] + 0.15))
            for zb in range(zb0, zb1 + 1):
                z_bins_boards.setdefault(zb, []).append(bi)

        for r in b_obj["rails"]:
            pts = [tuple(pt) for pt in (r["pipe_front"] + r["pipe_back"])]
            zlo = min(pt[2] for pt in pts)
            zhi = max(pt[2] for pt in pts)
            rail_entries.append((tag, str(r["id"]), float(r["pipe_radius"]), zlo, zhi, [tuple(pt) for pt in r["pipe_front"]], [tuple(pt) for pt in r["pipe_back"]]))

    for b in balls:
        _add_ball(("M", int(b["ball_idx"])), radius, b)
    for d in drum_balls:
        _add_ball(("D", int(d["drum_idx"])), float(d["radius"]), d)

    if finale_spec and finale_spec.get("bridge"):
        br = finale_spec["bridge"]
        bc = tuple(float(v) for v in br["center"])
        he = (0.5 * float(br["length"]), 0.5 * float(br["width_y"]), 0.5 * float(br["thickness"]))
        bi = len(board_entries)
        board_entries.append((("BRIDGE", 0), 0, bc, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), he))
        zb0 = int(math.floor(bc[2] - he[2] - 0.15))
        zb1 = int(math.floor(bc[2] + he[2] + 0.15))
        for zb in range(zb0, zb1 + 1):
            z_bins_boards.setdefault(zb, []).append(bi)

    # Prebuild sorted (dx, dz) offset candidates
    offsets = [(0.0, 0.0)]
    for dx in (0.0, -0.14, 0.14, -0.26, 0.26, -0.38, 0.38, -0.52, 0.52, -0.68, 0.68, -0.84, 0.84):
        for dz in (0.0, 0.16, -0.16, 0.30, -0.30, 0.44, -0.44, 0.58, -0.58, 0.72, -0.72):
            if (dx, dz) == (0.0, 0.0):
                continue
            if math.hypot(dx, dz) <= 0.96:
                offsets.append((dx, dz))
    offsets.sort(key=lambda t: (1.15 * t[0] * t[0] + t[1] * t[1]))

    return {
        "ball_tracks": ball_tracks,
        "board_entries": board_entries,
        "z_bins_boards": z_bins_boards,
        "rail_entries": rail_entries,
        "offsets": offsets,
        "used_wall_anchors": [],
    }


def _segment_is_clear(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    self_tag: tuple[str, int] | None,
    self_idx: int | None,
    pin_r: float,
    obstacles: dict,
    self_rail_id: str | None = None,
    self_rail_rear_is_front: bool = False,
) -> bool:
    xlo, xhi = min(a[0], b[0]) - 0.55, max(a[0], b[0]) + 0.55
    ylo, yhi = min(a[1], b[1]) - 0.25, max(a[1], b[1]) + 0.25
    zlo, zhi = min(a[2], b[2]) - 0.55, max(a[2], b[2]) + 0.55
    zbin0, zbin1 = int(math.floor(zlo)), int(math.floor(zhi))

    # 1. All 5 moving marbles across the Z-bucket of this segment
    for tag, rad_v, samps, z_bins in obstacles["ball_tracks"]:
        req = rad_v + pin_r + 0.024
        for zb in range(zbin0, zbin1 + 1):
            for f_i in z_bins.get(zb, ()):
                p = samps[f_i]
                if self_rail_id is not None and tag == self_tag and p[1] < a[1] + 0.05:
                    continue
                if not (xlo <= p[0] <= xhi and ylo <= p[1] <= yhi and zlo <= p[2] <= zhi):
                    continue
                if _pt_seg_dist(p, a, b) < req:
                    return False

    # 2. All other boards
    n_samp = 14
    seg_pts = [
        (a[0] + (b[0] - a[0]) * k / n_samp, a[1] + (b[1] - a[1]) * k / n_samp, a[2] + (b[2] - a[2]) * k / n_samp)
        for k in range(1, n_samp + 1)
    ]
    checked_bi = set()
    board_entries = obstacles["board_entries"]
    z_bins_boards = obstacles["z_bins_boards"]
    v_ax = (0.0, 1.0, 0.0)
    for zb in range(zbin0, zbin1 + 1):
        for bi in z_bins_boards.get(zb, ()):
            if bi in checked_bi:
                continue
            checked_bi.add(bi)
            tag, bidx, bc, u, n, he = board_entries[bi]
            if tag == self_tag and bidx == self_idx:
                continue
            if not (xlo - he[0] <= bc[0] <= xhi + he[0] and ylo - he[1] <= bc[1] <= yhi + he[1] and zlo - he[0] <= bc[2] <= zhi + he[0]):
                continue
            for sp in seg_pts:
                if _pt_obb_dist(sp, bc, u, v_ax, n, he) < pin_r + 0.018:
                    return False

    # 3. All rails (checking opposite pipe of same rail when mounted on +Y-rearmost pipe)
    for _tag, rid, pipe_r, r_zlo, r_zhi, p_front, p_back in obstacles["rail_entries"]:
        if r_zhi < zlo or r_zlo > zhi:
            continue
        req_r = pipe_r + pin_r + 0.018
        if rid == self_rail_id:
            pts_chk = (p_back[::2] if self_rail_rear_is_front else p_front[::2])
        else:
            pts_chk = p_front[::2] + p_back[::2]
        for rp in pts_chk:
            if not (xlo <= rp[0] <= xhi and ylo <= rp[1] <= yhi and zlo <= rp[2] <= zhi):
                continue
            if _pt_seg_dist(rp, a, b) < req_r:
                return False
    return True


def find_reverse_wall_anchor(
    pin_a: tuple[float, float, float],
    cx: float,
    cz: float,
    self_tag: tuple[str, int] | None,
    self_idx: int | None,
    pin_r: float,
    wall_y: float,
    obstacles: dict,
    self_rail_id: str | None = None,
    self_rail_rear_is_front: bool = False,
) -> tuple[str, tuple[float, float, float], tuple[float, float, float] | None]:
    """Reverse-search a 100% obstacle-free wall anchor (wx, wall_y+0.025, wz), with 2-segment elbow fallback."""
    a = (float(pin_a[0]), float(pin_a[1]), float(pin_a[2]))
    used = obstacles["used_wall_anchors"]

    def _anchor_free(wx: float, wz: float) -> bool:
        for ux, uz in used:
            if (wx - ux) ** 2 + (wz - uz) ** 2 < 0.055 ** 2:
                return False
        return True

    # Pass 1: 1-segment straight or diagonal strut preferring non-overlapping wall flanges
    for enforce_flange in (True, False):
        for idx_c, (dx, dz) in enumerate(obstacles["offsets"]):
            wx = cx + dx
            wz = cz + dz
            if abs(wx) > 2.05:
                continue
            if enforce_flange and not _anchor_free(wx, wz):
                continue
            pin_b = (wx, wall_y + 0.025, wz)
            if _segment_is_clear(
                a, pin_b, self_tag, self_idx, pin_r, obstacles,
                self_rail_id=self_rail_id, self_rail_rear_is_front=self_rail_rear_is_front,
            ):
                used.append((wx, wz))
                mode = "straight" if idx_c == 0 else "diag"
                return mode, pin_b, None

        # Pass 2: 2-segment L-bracket elbow within current layer before piercing deeper layers
        ey = min(wall_y - 0.05, a[1] + 0.08)
        for dx, dz in obstacles["offsets"][1:]:
            wx = cx + dx
            wz = cz + dz
            if abs(wx) > 2.05:
                continue
            if enforce_flange and not _anchor_free(wx, wz):
                continue
            p_elb = (wx, ey, wz)
            pin_b = (wx, wall_y + 0.025, wz)
            if (
                _segment_is_clear(
                    a, p_elb, self_tag, self_idx, pin_r, obstacles,
                    self_rail_id=self_rail_id, self_rail_rear_is_front=self_rail_rear_is_front,
                )
                and _segment_is_clear(
                    p_elb, pin_b, self_tag, self_idx, pin_r, obstacles,
                    self_rail_id=self_rail_id, self_rail_rear_is_front=self_rail_rear_is_front,
                )
            ):
                used.append((wx, wz))
                return "elbow", pin_b, p_elb

    return "fail", (cx, wall_y + 0.025, cz), None


def create_animated_finale_title_plaque(
    wall_y: float,
    finale_spec: dict,
    col_stage: bpy.types.Collection,
    col_fx: bpy.types.Collection,
    melody_marble_objs: dict[int, bpy.types.Object],
    drum_marble_objs: dict[int, bpy.types.Object],
) -> None:
    """Build split 5-word animated GLASS MARBLES / ON WOOD bulb matrix + marble pop-morph & starburst VFX."""
    t_bm = bmesh.new()
    bmesh.ops.create_uvsphere(t_bm, u_segments=12, v_segments=8, radius=1.0)
    unit_verts = [(float(v.co.x), float(v.co.y), float(v.co.z)) for v in t_bm.verts]
    unit_faces = [[v.index for v in f.verts] for f in t_bm.faces]
    t_bm.free()

    spark_mesh = bpy.data.meshes.new("FinaleSparkMesh")
    s_bm = bmesh.new()
    bmesh.ops.create_uvsphere(s_bm, u_segments=10, v_segments=8, radius=0.015)
    s_bm.to_mesh(spark_mesh)
    s_bm.free()
    for pl in spark_mesh.polygons:
        pl.use_smooth = True
    spark_dummy_mat = bpy.data.materials.new("FinaleSparkDummyMat")
    spark_mesh.materials.append(spark_dummy_mat)

    halo_mesh = bpy.data.meshes.new("FinaleHaloRingMesh")
    h_bm = bmesh.new()
    maj_segs, min_segs = 48, 12
    maj_r, min_r = 1.0, 0.028
    ring_verts = []
    for ui in range(maj_segs):
        phi = (2.0 * math.pi * ui) / maj_segs
        cp, sp_ = math.cos(phi), math.sin(phi)
        loop_v = []
        for vj in range(min_segs):
            psi = (2.0 * math.pi * vj) / min_segs
            cs, ss = math.cos(psi), math.sin(psi)
            vx = (maj_r + min_r * cs) * cp
            vy = min_r * ss
            vz = (maj_r + min_r * cs) * sp_
            loop_v.append(h_bm.verts.new((vx, vy, vz)))
        ring_verts.append(loop_v)
    for ui in range(maj_segs):
        u_nxt = (ui + 1) % maj_segs
        for vj in range(min_segs):
            v_nxt = (vj + 1) % min_segs
            h_bm.faces.new(
                (ring_verts[ui][vj], ring_verts[u_nxt][vj], ring_verts[u_nxt][v_nxt], ring_verts[ui][v_nxt])
            )
    h_bm.to_mesh(halo_mesh)
    h_bm.free()
    for pl in halo_mesh.polygons:
        pl.use_smooth = True
    halo_dummy_mat = bpy.data.materials.new("FinaleHaloDummyMat")
    halo_mesh.materials.append(halo_dummy_mat)

    bulb_palette = [
        ("Gold", (1.0, 0.58, 0.14)),
        ("Red", (1.0, 0.03, 0.02)),
        ("Gold", (1.0, 0.58, 0.14)),
        ("Green", (0.04, 0.90, 0.15)),
    ]

    words_by_id = {w["id"]: w for w in finale_spec["words"]}
    f_celebrate = int(finale_spec["celebration_frame"])

    rows = [
        (
            float(finale_spec["z_row1"]),
            float(finale_spec["row1_pitch"]),
            float(finale_spec["row1_radius"]),
            [("GLASS", "GLASS"), (None, " "), ("MARBLE", "MARBLE"), ("S", "S")],
        ),
        (
            float(finale_spec["z_row2"]),
            float(finale_spec["row2_pitch"]),
            float(finale_spec["row2_radius"]),
            [("ON", "ON"), (None, " "), ("WOOD", "WOOD")],
        ),
    ]

    for row_z, pitch, rad, segs in rows:
        full_text = "".join(txt for _, txt in segs)
        glyphs = [None if ch == " " else BULB_FONT[ch] for ch in full_text]
        total_cols = sum((3 if g is None else len(g[0])) + 1 for g in glyphs) - 1

        char_word_ids: list[str | None] = []
        for wid, txt in segs:
            for ch in txt:
                char_word_ids.append(None if ch == " " else wid)

        word_bms: dict[str, list[bmesh.types.BMesh]] = {}
        for wid, _ in segs:
            if wid is not None:
                word_bms[wid] = [bmesh.new() for _ in range(len(bulb_palette))]

        y_plane = wall_y + 0.004 - 0.8 * rad
        col_cursor = 0
        bulb_i = 0
        for g_idx, g in enumerate(glyphs):
            if g is None:
                col_cursor += 4
                continue
            gw = len(g[0])
            wid = char_word_ids[g_idx]
            xw_orig = float(words_by_id[wid]["x_center"])
            cbms = word_bms[wid]
            for r_i, row in enumerate(g):
                for c_i, cell in enumerate(row):
                    if cell != "1":
                        continue
                    bx_local = (col_cursor + c_i - 0.5 * (total_cols - 1)) * pitch - xw_orig
                    bz_local = (3.0 - r_i) * pitch
                    cbm = cbms[bulb_i % len(bulb_palette)]
                    new_v = [cbm.verts.new((ux * rad + bx_local, uy * rad, uz * rad + bz_local)) for ux, uy, uz in unit_verts]
                    for f_idxs in unit_faces:
                        cbm.faces.new([new_v[iv] for iv in f_idxs])
                    bulb_i += 1
            col_cursor += gw + 1

        for wid, cbms in word_bms.items():
            wmeta = words_by_id[wid]
            f_m = int(wmeta["morph_frame"])
            xw = float(wmeta["x_center"])
            zw = float(wmeta["z_center"])

            root_empty = bpy.data.objects.new(f"FinaleWordRoot_{wid}", None)
            col_stage.objects.link(root_empty)
            root_empty.location = (xw, y_plane, zw)
            root_empty.scale = (0.0001, 0.0001, 0.0001)
            root_empty.hide_render = True
            root_empty.keyframe_insert(data_path="scale", frame=0)
            root_empty.keyframe_insert(data_path="hide_render", frame=0)
            root_empty.keyframe_insert(data_path="scale", frame=max(0, f_m - 1))
            root_empty.keyframe_insert(data_path="hide_render", frame=max(0, f_m - 1))

            root_empty.hide_render = False
            root_empty.scale = (0.20, 0.20, 0.20)
            root_empty.keyframe_insert(data_path="hide_render", frame=f_m)
            root_empty.keyframe_insert(data_path="scale", frame=f_m)
            root_empty.scale = (1.14, 1.14, 1.14)
            root_empty.keyframe_insert(data_path="scale", frame=f_m + 5)
            root_empty.scale = (0.97, 0.97, 0.97)
            root_empty.keyframe_insert(data_path="scale", frame=f_m + 10)
            root_empty.scale = (1.0, 1.0, 1.0)
            root_empty.keyframe_insert(data_path="scale", frame=f_m + 14)

            # Shared ShaderNodeValue driving all 4 bulb colors in lockstep for this word
            em_val_nodes = []
            for bi, (bn, bc) in enumerate(bulb_palette):
                m = bpy.data.materials.new(f"FinaleWordMat_{wid}_{bi}_{bn}")
                m.use_nodes = True
                nt = m.node_tree
                nt.nodes.clear()
                o_ = nt.nodes.new("ShaderNodeOutputMaterial")
                v_nd = nt.nodes.new("ShaderNodeValue")
                v_nd.outputs[0].default_value = 0.0
                e_ = nt.nodes.new("ShaderNodeEmission")
                e_.inputs["Color"].default_value = (bc[0], bc[1], bc[2], 1.0)
                nt.links.new(v_nd.outputs["Value"], e_.inputs["Strength"])
                nt.links.new(e_.outputs["Emission"], o_.inputs["Surface"])
                em_val_nodes.append((m, v_nd.outputs[0]))

            for vsock in (pair[1] for pair in em_val_nodes):
                vsock.default_value = 0.0
                vsock.keyframe_insert(data_path="default_value", frame=0)
                vsock.keyframe_insert(data_path="default_value", frame=max(0, f_m - 1))
                vsock.default_value = 34.0
                vsock.keyframe_insert(data_path="default_value", frame=f_m)
                vsock.default_value = 22.0
                vsock.keyframe_insert(data_path="default_value", frame=f_m + 5)
                vsock.default_value = 15.5
                vsock.keyframe_insert(data_path="default_value", frame=f_m + 10)
                vsock.default_value = 13.5
                vsock.keyframe_insert(data_path="default_value", frame=f_m + 14)
                if wid != "S" and f_celebrate > f_m + 14:
                    vsock.default_value = 13.5
                    vsock.keyframe_insert(data_path="default_value", frame=f_celebrate - 1)
                    vsock.default_value = 21.0
                    vsock.keyframe_insert(data_path="default_value", frame=f_celebrate + 2)
                    vsock.default_value = 13.5
                    vsock.keyframe_insert(data_path="default_value", frame=f_celebrate + 14)

            for bi, cbm in enumerate(cbms):
                if not cbm.verts:
                    cbm.free()
                    continue
                mesh = bpy.data.meshes.new(f"FinaleWordMesh_{wid}_c{bi}")
                cbm.to_mesh(mesh)
                cbm.free()
                for pl in mesh.polygons:
                    pl.use_smooth = True
                mesh.materials.append(em_val_nodes[bi][0])
                obj = bpy.data.objects.new(f"FinaleWord_{wid}_c{bi}", mesh)
                col_stage.objects.link(obj)
                obj.parent = root_empty
                obj.hide_render = True
                obj.keyframe_insert(data_path="hide_render", frame=0)
                obj.keyframe_insert(data_path="hide_render", frame=max(0, f_m - 1))
                obj.hide_render = False
                obj.keyframe_insert(data_path="hide_render", frame=f_m)

    # Golden "ARTIST : ZERO.SOE" credit fades in cleanly right below
    cr_txt = str(finale_spec["credit_text"])
    cr_pitch = float(finale_spec["credit_pitch"])
    cr_rad = float(finale_spec["credit_radius"])
    cr_z = float(finale_spec["z_row3"])
    cr_f0 = int(finale_spec["credit_fade_start_frame"])
    cr_f1 = int(finale_spec["credit_fade_end_frame"])

    cr_bm = bmesh.new()
    cr_glyphs = [None if ch == " " else BULB_FONT[ch] for ch in cr_txt]
    cr_cols = sum((3 if g is None else len(g[0])) + 1 for g in cr_glyphs) - 1
    col_c = 0
    for g in cr_glyphs:
        if g is None:
            col_c += 4
            continue
        gw = len(g[0])
        for r_i, row in enumerate(g):
            for c_i, cell in enumerate(row):
                if cell != "1":
                    continue
                bx = (col_c + c_i - 0.5 * (cr_cols - 1)) * cr_pitch
                by = wall_y + 0.004 - 0.8 * cr_rad
                bz = cr_z + (3.0 - r_i) * cr_pitch
                new_v = [cr_bm.verts.new((ux * cr_rad + bx, uy * cr_rad + by, uz * cr_rad + bz)) for ux, uy, uz in unit_verts]
                for f_idxs in unit_faces:
                    cr_bm.faces.new([new_v[iv] for iv in f_idxs])
        col_c += gw + 1

    cr_mat = bpy.data.materials.new("FinaleCreditGoldMat")
    cr_mat.use_nodes = True
    nt_cr = cr_mat.node_tree
    nt_cr.nodes.clear()
    o_cr = nt_cr.nodes.new("ShaderNodeOutputMaterial")
    em_cr = nt_cr.nodes.new("ShaderNodeEmission")
    em_cr.inputs["Color"].default_value = (1.0, 0.72, 0.22, 1.0)
    em_cr.inputs["Strength"].default_value = 0.0
    nt_cr.links.new(em_cr.outputs["Emission"], o_cr.inputs["Surface"])

    cr_mesh = bpy.data.meshes.new("FinaleCreditMesh")
    cr_bm.to_mesh(cr_mesh)
    cr_bm.free()
    for pl in cr_mesh.polygons:
        pl.use_smooth = True
    cr_mesh.materials.append(cr_mat)
    cr_obj = bpy.data.objects.new("FinaleCredit_ARTIST", cr_mesh)
    col_stage.objects.link(cr_obj)
    cr_obj.hide_render = True
    cr_obj.keyframe_insert(data_path="hide_render", frame=0)
    cr_obj.keyframe_insert(data_path="hide_render", frame=max(0, cr_f0 - 1))
    cr_obj.hide_render = False
    cr_obj.keyframe_insert(data_path="hide_render", frame=cr_f0)

    cr_sock = em_cr.inputs["Strength"]
    cr_sock.default_value = 0.0
    cr_sock.keyframe_insert(data_path="default_value", frame=0)
    cr_sock.keyframe_insert(data_path="default_value", frame=max(0, cr_f0 - 1))
    cr_sock.default_value = 9.5
    cr_sock.keyframe_insert(data_path="default_value", frame=cr_f1)

    # Marble pop-morph + burst flash light + 24-spark starburst + halo ring VFX per word
    for idx_w, wmeta in enumerate(finale_spec["words"]):
        wid = str(wmeta["id"])
        f_m = int(wmeta["morph_frame"])
        xw = float(wmeta["x_center"])
        zw = float(wmeta["z_center"])
        brgb = wmeta["burst_rgb"]
        mix_rgb = (0.5 * float(brgb[0]) + 0.5 * 1.0, 0.5 * float(brgb[1]) + 0.5 * 0.78, 0.5 * float(brgb[2]) + 0.5 * 0.25)

        m_obj = drum_marble_objs.get(int(wmeta["ball_idx"])) if wmeta["ball_type"] == "drum" else melody_marble_objs.get(int(wmeta["ball_idx"]))
        if m_obj is not None:
            m_obj.scale = (1.0, 1.0, 1.0)
            m_obj.hide_render = False
            m_obj.keyframe_insert(data_path="scale", frame=0)
            m_obj.keyframe_insert(data_path="hide_render", frame=0)
            m_obj.keyframe_insert(data_path="scale", frame=max(0, f_m - 2))
            m_obj.scale = (1.30, 1.30, 1.30)
            m_obj.keyframe_insert(data_path="scale", frame=f_m)
            m_obj.scale = (0.0001, 0.0001, 0.0001)
            m_obj.keyframe_insert(data_path="scale", frame=f_m + 4)
            m_obj.keyframe_insert(data_path="hide_render", frame=f_m + 4)
            m_obj.hide_render = True
            m_obj.keyframe_insert(data_path="hide_render", frame=f_m + 5)
            for ch_obj in m_obj.children:
                ch_obj.hide_render = False
                ch_obj.keyframe_insert(data_path="hide_render", frame=0)
                ch_obj.keyframe_insert(data_path="hide_render", frame=f_m + 4)
                ch_obj.hide_render = True
                ch_obj.keyframe_insert(data_path="hide_render", frame=f_m + 5)

        bl_data = bpy.data.lights.new(f"FinaleBurstLight_{wid}", type="POINT")
        bl_data.energy = 0.0
        bl_data.color = mix_rgb
        bl_data.shadow_soft_size = 0.22
        bl_data.use_shadow = False
        bl_obj = bpy.data.objects.new(f"FinaleBurstLight_{wid}", bl_data)
        col_fx.objects.link(bl_obj)
        bl_obj.location = (xw, wall_y - 0.26, zw)
        bl_data.keyframe_insert(data_path="energy", frame=0)
        bl_data.keyframe_insert(data_path="energy", frame=max(0, f_m - 1))
        bl_data.energy = 48.0
        bl_data.keyframe_insert(data_path="energy", frame=f_m + 1)
        bl_data.energy = 16.0
        bl_data.keyframe_insert(data_path="energy", frame=f_m + 6)
        bl_data.energy = 0.0
        bl_data.keyframe_insert(data_path="energy", frame=f_m + 16)

        sp_mat = bpy.data.materials.new(f"FinaleSparkMat_{wid}")
        sp_mat.use_nodes = True
        nt_sp = sp_mat.node_tree
        nt_sp.nodes.clear()
        o_sp = nt_sp.nodes.new("ShaderNodeOutputMaterial")
        em_sp = nt_sp.nodes.new("ShaderNodeEmission")
        em_sp.inputs["Color"].default_value = (mix_rgb[0], mix_rgb[1], mix_rgb[2], 1.0)
        em_sp.inputs["Strength"].default_value = 22.0
        nt_sp.links.new(em_sp.outputs["Emission"], o_sp.inputs["Surface"])

        origin = Vector((xw, wall_y - 0.12, zw))
        for sk in range(24):
            th = (2.0 * math.pi * sk) / 24.0 + 0.13 * idx_w
            sp_scale = 0.80 + 0.33 * ((sk * 5 + idx_w * 3) % 7) / 6.0
            spd_scale = 0.82 + 0.36 * ((sk * 7 + idx_w * 2) % 6) / 5.0
            z_tilt = -0.04 if (sk % 2 == 0) else 0.04
            sdir = Vector((math.cos(th), -0.25, math.sin(th) + z_tilt)).normalized()
            s_obj = bpy.data.objects.new(f"FinaleSpark_{wid}_{sk:02d}", spark_mesh)
            col_fx.objects.link(s_obj)
            s_obj.material_slots[0].link = "OBJECT"
            s_obj.material_slots[0].material = sp_mat
            s_obj.location = origin
            s_obj.scale = (0.001, 0.001, 0.001)
            s_obj.hide_render = True
            s_obj.keyframe_insert(data_path="location", frame=0)
            s_obj.keyframe_insert(data_path="scale", frame=0)
            s_obj.keyframe_insert(data_path="hide_render", frame=0)
            s_obj.keyframe_insert(data_path="location", frame=max(0, f_m - 1))
            s_obj.keyframe_insert(data_path="scale", frame=max(0, f_m - 1))
            s_obj.keyframe_insert(data_path="hide_render", frame=max(0, f_m - 1))

            s_obj.hide_render = False
            s_obj.location = origin
            s_obj.scale = (1.15 * sp_scale, 1.15 * sp_scale, 1.15 * sp_scale)
            s_obj.keyframe_insert(data_path="hide_render", frame=f_m)
            s_obj.keyframe_insert(data_path="location", frame=f_m)
            s_obj.keyframe_insert(data_path="scale", frame=f_m)

            s_obj.location = origin + sdir * (0.26 * spd_scale)
            s_obj.scale = (0.85 * sp_scale, 0.85 * sp_scale, 0.85 * sp_scale)
            s_obj.keyframe_insert(data_path="location", frame=f_m + 7)
            s_obj.keyframe_insert(data_path="scale", frame=f_m + 7)

            s_obj.location = origin + sdir * (0.44 * spd_scale) + Vector((0.0, 0.0, -0.06))
            s_obj.scale = (0.001, 0.001, 0.001)
            s_obj.keyframe_insert(data_path="location", frame=f_m + 18)
            s_obj.keyframe_insert(data_path="scale", frame=f_m + 18)
            s_obj.keyframe_insert(data_path="hide_render", frame=f_m + 18)

            s_obj.hide_render = True
            s_obj.keyframe_insert(data_path="hide_render", frame=f_m + 19)

        hr_mat = bpy.data.materials.new(f"FinaleHaloMat_{wid}")
        hr_mat.use_nodes = True
        if hasattr(hr_mat, "surface_render_method"):
            try:
                hr_mat.surface_render_method = "BLENDED"
            except Exception:
                pass
        if hasattr(hr_mat, "blend_method"):
            try:
                hr_mat.blend_method = "BLEND"
            except Exception:
                pass
        nt_hr = hr_mat.node_tree
        nt_hr.nodes.clear()
        o_hr = nt_hr.nodes.new("ShaderNodeOutputMaterial")
        tr_hr = nt_hr.nodes.new("ShaderNodeBsdfTransparent")
        em_hr = nt_hr.nodes.new("ShaderNodeEmission")
        em_hr.inputs["Color"].default_value = (mix_rgb[0], mix_rgb[1], mix_rgb[2], 1.0)
        em_hr.inputs["Strength"].default_value = 0.0
        add_hr = nt_hr.nodes.new("ShaderNodeAddShader")
        nt_hr.links.new(tr_hr.outputs["BSDF"], add_hr.inputs[0])
        nt_hr.links.new(em_hr.outputs["Emission"], add_hr.inputs[1])
        nt_hr.links.new(add_hr.outputs["Shader"], o_hr.inputs["Surface"])

        hr_str = em_hr.inputs["Strength"]
        hr_str.default_value = 0.0
        hr_str.keyframe_insert(data_path="default_value", frame=0)
        hr_str.keyframe_insert(data_path="default_value", frame=max(0, f_m - 1))
        hr_str.default_value = 30.0
        hr_str.keyframe_insert(data_path="default_value", frame=f_m)
        hr_str.default_value = 12.0
        hr_str.keyframe_insert(data_path="default_value", frame=f_m + 7)
        hr_str.default_value = 0.0
        hr_str.keyframe_insert(data_path="default_value", frame=f_m + 16)

        end_rad = 0.52 if wid == "S" else 0.42
        mid_rad = 0.28 if wid == "S" else 0.23
        hr_obj = bpy.data.objects.new(f"FinaleHaloRing_{wid}", halo_mesh)
        col_fx.objects.link(hr_obj)
        hr_obj.material_slots[0].link = "OBJECT"
        hr_obj.material_slots[0].material = hr_mat
        hr_obj.location = (xw, wall_y - 0.10, zw)
        hr_obj.scale = (0.001, 0.001, 0.001)
        hr_obj.hide_render = True
        hr_obj.keyframe_insert(data_path="scale", frame=0)
        hr_obj.keyframe_insert(data_path="hide_render", frame=0)
        hr_obj.keyframe_insert(data_path="scale", frame=max(0, f_m - 1))
        hr_obj.keyframe_insert(data_path="hide_render", frame=max(0, f_m - 1))

        hr_obj.hide_render = False
        hr_obj.scale = (0.06, 0.06, 0.06)
        hr_obj.keyframe_insert(data_path="hide_render", frame=f_m)
        hr_obj.keyframe_insert(data_path="scale", frame=f_m)

        hr_obj.scale = (mid_rad, mid_rad, mid_rad)
        hr_obj.keyframe_insert(data_path="scale", frame=f_m + 6)

        hr_obj.scale = (end_rad, end_rad, end_rad)
        hr_obj.keyframe_insert(data_path="scale", frame=f_m + 16)
        hr_obj.keyframe_insert(data_path="hide_render", frame=f_m + 16)

        hr_obj.hide_render = True
        hr_obj.keyframe_insert(data_path="hide_render", frame=f_m + 17)


def build_unified_finale_bridge(
    wall_y: float,
    bridge_spec: dict,
    f_credit_end: int,
    brass_mat: bpy.types.Material,
    hardware_dark_mat: bpy.types.Material,
    col_stage: bpy.types.Collection,
    col_fx: bpy.types.Collection,
    total_frames: int,
) -> None:
    """Build the unified Finale walnut console bridge, brass caps, LED ripple bar & rear mounting struts."""
    bc = Vector(bridge_spec["center"])
    b_len = float(bridge_spec["length"])
    b_wy = float(bridge_spec["width_y"])
    b_th = float(bridge_spec["thickness"])

    def _make_beveled_box(name: str, lx: float, wy: float, hz: float, bevel_off: float, segs: int = 4) -> bpy.types.Mesh:
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(lx, wy, hz), verts=bm.verts)
        bmesh.ops.bevel(
            bm,
            geom=list(bm.edges),
            offset=bevel_off,
            segments=segs,
            profile=0.5,
            affect="EDGES",
            clamp_overlap=True,
            harden_normals=False,
        )
        bm.to_mesh(mesh)
        bm.free()
        tmp = bpy.data.objects.new(f"__Tmp_{name}", mesh)
        base_scene.fix_mesh_normals(tmp, angle_deg=35.0, weighted=True)
        bpy.data.objects.remove(tmp, do_unlink=True)
        return mesh

    walnut_mat = create_wood_bar_material_from_colors(
        "FinaleBridgeWalnutMat", (0.24, 0.11, 0.055, 1.0), (0.10, 0.045, 0.020, 1.0)
    )
    beam_mesh = _make_beveled_box("FinaleBridgeBeamMesh", b_len, b_wy, b_th, bevel_off=0.008, segs=5)
    beam_obj = bpy.data.objects.new("FinaleBridgeBeam", beam_mesh)
    col_stage.objects.link(beam_obj)
    beam_obj.data.materials.append(walnut_mat)
    beam_obj.location = bc

    cap_mesh = _make_beveled_box("FinaleBridgeCapMesh", 0.042, b_wy + 0.012, b_th + 0.010, bevel_off=0.005, segs=4)
    for tag_c, sx in (("L", -0.5 * b_len), ("R", 0.5 * b_len)):
        cap_obj = bpy.data.objects.new(f"FinaleBridgeCap_{tag_c}", cap_mesh)
        col_stage.objects.link(cap_obj)
        cap_obj.data.materials.append(brass_mat)
        cap_obj.parent = beam_obj
        cap_obj.location = (sx, 0.0, 0.0)

    led_len = max(0.5, b_len - 0.16)
    led_mesh = _make_beveled_box("FinaleBridgeLEDMesh", led_len, 0.014, 0.018, bevel_off=0.003, segs=2)
    led_mat = bpy.data.materials.new("FinaleBridgeLEDMat")
    led_mat.use_nodes = True
    nt = led_mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.35, 0.24, 0.08, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.15
    bsdf.inputs["Roughness"].default_value = 0.18

    tc = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    hit_x_node = nodes.new("ShaderNodeValue")
    hit_x_node.outputs[0].default_value = 0.0
    rad_node = nodes.new("ShaderNodeValue")
    rad_node.outputs[0].default_value = 0.0
    amp_node = nodes.new("ShaderNodeValue")
    amp_node.outputs[0].default_value = 0.0
    idle_node = nodes.new("ShaderNodeValue")
    idle_node.outputs[0].default_value = 1.2

    dx_n = nodes.new("ShaderNodeMath")
    dx_n.operation = "SUBTRACT"
    abs_dx = nodes.new("ShaderNodeMath")
    abs_dx.operation = "ABSOLUTE"
    ring_d = nodes.new("ShaderNodeMath")
    ring_d.operation = "SUBTRACT"
    abs_ring = nodes.new("ShaderNodeMath")
    abs_ring.operation = "ABSOLUTE"
    norm_ring = nodes.new("ShaderNodeMath")
    norm_ring.operation = "DIVIDE"
    norm_ring.inputs[1].default_value = 0.26
    front_fac = nodes.new("ShaderNodeMath")
    front_fac.operation = "SUBTRACT"
    front_fac.inputs[0].default_value = 1.0
    front_fac.use_clamp = True

    front_e = nodes.new("ShaderNodeMath")
    front_e.operation = "MULTIPLY"
    hot_e = nodes.new("ShaderNodeMath")
    hot_e.operation = "MULTIPLY"
    hot_e.inputs[1].default_value = 0.85
    flash_e = nodes.new("ShaderNodeMath")
    flash_e.operation = "MULTIPLY"
    flash_e.inputs[1].default_value = 0.35
    gold_str = nodes.new("ShaderNodeMath")
    gold_str.operation = "ADD"

    em_gold = nodes.new("ShaderNodeEmission")
    em_gold.inputs["Color"].default_value = (1.0, 0.74, 0.30, 1.0)
    em_white = nodes.new("ShaderNodeEmission")
    em_white.inputs["Color"].default_value = (1.0, 0.94, 0.72, 1.0)
    add1 = nodes.new("ShaderNodeAddShader")
    add2 = nodes.new("ShaderNodeAddShader")

    links.new(tc.outputs["Object"], sep.inputs["Vector"])
    links.new(sep.outputs["X"], dx_n.inputs[0])
    links.new(hit_x_node.outputs["Value"], dx_n.inputs[1])
    links.new(dx_n.outputs["Value"], abs_dx.inputs[0])
    links.new(abs_dx.outputs["Value"], ring_d.inputs[0])
    links.new(rad_node.outputs["Value"], ring_d.inputs[1])
    links.new(ring_d.outputs["Value"], abs_ring.inputs[0])
    links.new(abs_ring.outputs["Value"], norm_ring.inputs[0])
    links.new(norm_ring.outputs["Value"], front_fac.inputs[1])
    links.new(front_fac.outputs["Value"], front_e.inputs[0])
    links.new(amp_node.outputs["Value"], front_e.inputs[1])
    links.new(front_e.outputs["Value"], hot_e.inputs[0])
    links.new(amp_node.outputs["Value"], flash_e.inputs[0])
    links.new(flash_e.outputs["Value"], gold_str.inputs[0])
    links.new(idle_node.outputs["Value"], gold_str.inputs[1])
    links.new(gold_str.outputs["Value"], em_gold.inputs["Strength"])
    links.new(hot_e.outputs["Value"], em_white.inputs["Strength"])
    links.new(bsdf.outputs["BSDF"], add1.inputs[0])
    links.new(em_gold.outputs["Emission"], add1.inputs[1])
    links.new(add1.outputs["Shader"], add2.inputs[0])
    links.new(em_white.outputs["Emission"], add2.inputs[1])
    links.new(add2.outputs["Shader"], out.inputs["Surface"])

    led_obj = bpy.data.objects.new("FinaleBridgeLED", led_mesh)
    col_stage.objects.link(led_obj)
    led_obj.data.materials.append(led_mat)
    led_obj.parent = beam_obj
    led_obj.location = (0.0, -0.5 * b_wy - 0.001, 0.0)

    idle_node.outputs[0].default_value = 1.2
    idle_node.outputs[0].keyframe_insert(data_path="default_value", frame=0)
    f_idle0 = max(0, f_credit_end - 45)
    if f_idle0 > 0:
        idle_node.outputs[0].keyframe_insert(data_path="default_value", frame=f_idle0)
    idle_node.outputs[0].default_value = 2.4
    idle_node.outputs[0].keyframe_insert(data_path="default_value", frame=max(f_idle0 + 1, f_credit_end))

    flash_data = bpy.data.lights.new("FinaleBridgeFlashLight", type="POINT")
    flash_data.energy = 0.0
    flash_data.shadow_soft_size = 0.24
    flash_data.use_shadow = False
    flash_obj = bpy.data.objects.new("FinaleBridgeFlashLight", flash_data)
    col_fx.objects.link(flash_obj)
    flash_data.keyframe_insert(data_path="energy", frame=0)

    hits = sorted(bridge_spec.get("hits", []), key=lambda h: int(h["frame_hit"]))
    if hits:
        last_hx_f = -1
        last_rad_f = -1
        last_amp_f = -1
        last_rec_f = -1
        last_lt_f = 0
        for hi, h in enumerate(hits):
            fh = int(h["frame_hit"])
            hx = float(h["x"])
            rgb = h.get("rgb", [1.0, 0.78, 0.32])
            nxt_fh = int(hits[hi + 1]["frame_hit"]) if hi + 1 < len(hits) else (total_frames + 100)

            if hi == 0:
                hit_x_node.outputs[0].default_value = hx
                hit_x_node.outputs[0].keyframe_insert(data_path="default_value", frame=0)
                last_hx_f = 0
                if fh - 1 > 0:
                    hit_x_node.outputs[0].keyframe_insert(data_path="default_value", frame=fh - 1)
                    last_hx_f = fh - 1
            if fh > last_hx_f:
                hit_x_node.outputs[0].default_value = hx
                hit_x_node.outputs[0].keyframe_insert(data_path="default_value", frame=fh)
                last_hx_f = fh
            f_hx_hold = min(total_frames, nxt_fh - 1)
            if hi + 1 < len(hits) and f_hx_hold > last_hx_f:
                hit_x_node.outputs[0].default_value = hx
                hit_x_node.outputs[0].keyframe_insert(data_path="default_value", frame=f_hx_hold)
                last_hx_f = f_hx_hold

            if hi == 0:
                rad_node.outputs[0].default_value = 0.0
                rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=0)
                last_rad_f = 0
                if fh - 1 > 0:
                    rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=fh - 1)
                    last_rad_f = fh - 1
            if fh > last_rad_f:
                rad_node.outputs[0].default_value = 0.0
                rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=fh)
                last_rad_f = fh
            f_rad_pk = min(total_frames, fh + 11, nxt_fh - 2)
            if f_rad_pk > last_rad_f:
                rad_node.outputs[0].default_value = 1.45
                rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=f_rad_pk)
                last_rad_f = f_rad_pk
            f_rad_end = min(total_frames, f_rad_pk + 1, nxt_fh - 1)
            if f_rad_end > last_rad_f:
                rad_node.outputs[0].default_value = 0.0
                rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=f_rad_end)
                last_rad_f = f_rad_end

            if hi == 0:
                amp_node.outputs[0].default_value = 0.0
                amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=0)
                last_amp_f = 0
            if fh - 1 > last_amp_f:
                amp_node.outputs[0].default_value = 0.0
                amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=fh - 1)
                last_amp_f = fh - 1
            if fh > last_amp_f:
                amp_node.outputs[0].default_value = 16.0
                amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=fh)
                last_amp_f = fh
            f_a1 = min(total_frames, fh + 7, nxt_fh - 2)
            if f_a1 > last_amp_f:
                amp_node.outputs[0].default_value = 5.5
                amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=f_a1)
                last_amp_f = f_a1
            f_a2 = min(total_frames, fh + 15, nxt_fh - 1)
            if f_a2 > last_amp_f:
                amp_node.outputs[0].default_value = 0.0
                amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=f_a2)
                last_amp_f = f_a2

            if hi == 0:
                beam_obj.location = bc
                beam_obj.keyframe_insert(data_path="location", frame=0)
                last_rec_f = 0
            if fh > last_rec_f:
                beam_obj.location = bc
                beam_obj.keyframe_insert(data_path="location", frame=fh)
                last_rec_f = fh
            f_rp = min(total_frames, fh + 2, nxt_fh - 2)
            if f_rp > last_rec_f:
                beam_obj.location = bc - Vector((0.0, 0.0, 0.010))
                beam_obj.keyframe_insert(data_path="location", frame=f_rp)
                last_rec_f = f_rp
            f_re = min(total_frames, fh + 9, nxt_fh - 1)
            if f_re > last_rec_f:
                beam_obj.location = bc
                beam_obj.keyframe_insert(data_path="location", frame=f_re)
                last_rec_f = f_re

            warm_c = (
                0.55 * float(rgb[0]) + 0.45 * 1.0,
                0.55 * float(rgb[1]) + 0.45 * 0.78,
                0.55 * float(rgb[2]) + 0.45 * 0.34,
            )
            if fh - 1 > last_lt_f:
                flash_obj.location = (hx, -0.34, bc.z + 0.08)
                flash_data.color = warm_c
                flash_data.energy = 0.0
                flash_obj.keyframe_insert(data_path="location", frame=fh - 1)
                flash_data.keyframe_insert(data_path="color", frame=fh - 1)
                flash_data.keyframe_insert(data_path="energy", frame=fh - 1)
                last_lt_f = fh - 1
            if fh > last_lt_f:
                flash_obj.location = (hx, -0.34, bc.z + 0.08)
                flash_data.color = warm_c
                flash_data.energy = 36.0
                flash_obj.keyframe_insert(data_path="location", frame=fh)
                flash_data.keyframe_insert(data_path="color", frame=fh)
                flash_data.keyframe_insert(data_path="energy", frame=fh)
                last_lt_f = fh
            f_lm = min(total_frames, fh + 4, nxt_fh - 2)
            if f_lm > last_lt_f:
                flash_data.energy = 12.0
                flash_data.keyframe_insert(data_path="energy", frame=f_lm)
                last_lt_f = f_lm
            f_le = min(total_frames, fh + 10, nxt_fh - 1)
            if f_le > last_lt_f:
                flash_data.energy = 0.0
                flash_data.keyframe_insert(data_path="energy", frame=f_le)
                last_lt_f = f_le

    # 2 symmetrical dark-steel rear mounting struts at x = ±1.02 + brass beam collars & wall flanges
    collar_mesh = bpy.data.meshes.new("FinaleBridgeCollarMesh")
    c_bm = bmesh.new()
    bmesh.ops.create_uvsphere(c_bm, u_segments=14, v_segments=10, radius=0.022)
    c_bm.to_mesh(collar_mesh)
    c_bm.free()
    for pl in collar_mesh.polygons:
        pl.use_smooth = True
    collar_mesh.materials.append(brass_mat)

    for tag_s, sx in (("L", -1.02), ("R", 1.02)):
        base_scene.create_cylinder_between(
            f"FinaleBridgeStrut_{tag_s}",
            Vector((sx, 0.115, bc.z)),
            Vector((sx, wall_y + 0.025, bc.z)),
            0.015,
            hardware_dark_mat,
            col_stage,
            vertices=16,
        )
        col_obj = bpy.data.objects.new(f"FinaleBridgeCollar_{tag_s}", collar_mesh)
        col_stage.objects.link(col_obj)
        col_obj.location = (sx, 0.5 * b_wy, bc.z)
        base_scene.create_wall_flange(
            f"FinaleBridgeFlange_{tag_s}",
            (sx, wall_y + 0.003, bc.z),
            0.028,
            brass_mat,
            col_stage,
        )


def create_smooth_pipe_polyline(name: str, pts: list[list[float]], radius: float, mat, collection):
    """C2-smooth arc-length downsampled NURBS pipe curve (scales smoothly for both short chutes & 32m guides)."""
    if len(pts) > 20:
        total_len = 0.0
        for k in range(len(pts) - 1):
            dx = pts[k + 1][0] - pts[k][0]
            dy = pts[k + 1][1] - pts[k][1]
            dz = pts[k + 1][2] - pts[k][2]
            total_len += math.sqrt(dx * dx + dy * dy + dz * dz)
        target_n = max(16, min(220, int(round(total_len / 0.26))))
        if len(pts) > target_n:
            step = max(1, len(pts) // target_n)
            sampled = pts[::step]
            if sampled[-1] != pts[-1]:
                dx = sampled[-1][0] - pts[-1][0]
                dy = sampled[-1][1] - pts[-1][1]
                dz = sampled[-1][2] - pts[-1][2]
                if dx * dx + dy * dy + dz * dz < 0.0025 and len(sampled) >= 2:
                    sampled[-1] = pts[-1]
                else:
                    sampled.append(pts[-1])
            pts_use = sampled
        else:
            pts_use = pts
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


def _eval_section_state(
    t_sec: float,
    sections: list[dict] | None,
    fixed_night_mix: float,
    has_drums: bool,
) -> tuple[float, float, float]:
    """Return smoothly blended (night_mix, lens_mm, cam_y) for the given timestamp."""
    if not sections:
        lens_d = 38.0 if has_drums else 42.0
        cam_yd = -5.62 if has_drums else -5.45
        return fixed_night_mix, lens_d, cam_yd

    trans = 1.6
    half_t = 0.5 * trans
    for i in range(len(sections) - 1):
        bnd = float(sections[i]["t1"])
        if bnd - half_t <= t_sec <= bnd + half_t:
            u = max(0.0, min(1.0, (t_sec - (bnd - half_t)) / trans))
            w = 0.5 - 0.5 * math.cos(math.pi * u)
            s0, s1 = sections[i], sections[i + 1]
            return (
                float(s0["night_mix"]) + w * (float(s1["night_mix"]) - float(s0["night_mix"])),
                float(s0["lens"]) + w * (float(s1["lens"]) - float(s0["lens"])),
                float(s0["cam_y"]) + w * (float(s1["cam_y"]) - float(s0["cam_y"])),
            )
    for s in sections:
        if t_sec <= float(s["t1"]) + 1e-6:
            return float(s["night_mix"]), float(s["lens"]), float(s["cam_y"])
    last = sections[-1]
    return float(last["night_mix"]), float(last["lens"]), float(last["cam_y"])


def _lerp_tuple(a: tuple[float, ...], b: tuple[float, ...], w: float) -> tuple[float, ...]:
    return tuple(a[k] + w * (b[k] - a[k]) for k in range(len(a)))


def create_wood_bar_material_from_colors(
    name: str,
    c_light: tuple[float, float, float, float],
    c_dark: tuple[float, float, float, float],
) -> bpy.types.Material:
    m = bpy.data.materials.new(name=name)
    m.use_nodes = True
    nodes = m.node_tree.nodes
    links = m.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.5, 14.0, 14.0)

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


def create_chime_inlay_mesh(name: str, span_x: float, bar_h: float, depth_y: float = 0.010) -> bpy.types.Mesh:
    """Straight chamfered brass-chime light bar inlaid on Keyboard harmonic blocks."""
    bm = bmesh.new()
    res = bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(span_x, depth_y, bar_h), verts=res["verts"])
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.0018, segments=1, affect="EDGES", clamp_overlap=True)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def animate_fairy_garlands(
    bulb_mats: list[bpy.types.Material],
    perc_events: list[dict],
    base_strength: float,
    total_frames: int,
) -> None:
    """Keyframe alternating fairy-light bulb material pairs to twinkle on Percussion_Lights hits."""
    if not bulb_mats or not perc_events:
        return
    em_sockets = []
    for bm in bulb_mats:
        em = next((nd for nd in bm.node_tree.nodes if nd.type == "EMISSION"), None)
        if em is not None:
            em.inputs["Strength"].default_value = base_strength
            em.inputs["Strength"].keyframe_insert(data_path="default_value", frame=0)
            em_sockets.append(em.inputs["Strength"])
        else:
            em_sockets.append(None)

    pair_evs: list[list[dict]] = [[], []]
    for i, ev in enumerate(perc_events):
        pair_evs[i % 2].append(ev)

    for p_idx, evs in enumerate(pair_evs):
        pair_socks = [s for s in em_sockets[p_idx * 2: p_idx * 2 + 2] if s is not None]
        if not pair_socks:
            continue
        last_kf = 0
        for k, ev in enumerate(evs):
            fh = int(ev["frame_hit"])
            if fh <= last_kf or fh > total_frames:
                continue
            vel_s = 0.6 + 0.5 * max(0.0, min(1.0, float(ev.get("velocity", 90)) / 127.0))
            peak_s = base_strength * (1.75 + 0.45 * vel_s)
            nxt_fh = int(evs[k + 1]["frame_hit"]) if k + 1 < len(evs) else total_frames + 10
            f_pre = fh - 1
            f_dec = min(total_frames, fh + 6, nxt_fh - 2)
            for sock in pair_socks:
                if f_pre > last_kf:
                    sock.default_value = base_strength
                    sock.keyframe_insert(data_path="default_value", frame=f_pre)
                sock.default_value = peak_s
                sock.keyframe_insert(data_path="default_value", frame=fh)
                if f_dec > fh:
                    sock.default_value = base_strength
                    sock.keyframe_insert(data_path="default_value", frame=f_dec)
            last_kf = max(fh, f_dec)


def build_wall_bass_resonators(
    wall_y: float,
    z_bot: float,
    z_top: float,
    hero_specs: list[tuple[float, float, str]],
    bass_events: list[dict],
    brass_mat: bpy.types.Material,
    col_stage: bpy.types.Collection,
    total_frames: int,
) -> None:
    """Build twin brushed-brass inlaid bass light columns along x=-1.36/+1.38 in every non-hero bay."""
    if not bass_events:
        return

    bay_edges = [z_top]
    for hz, hh, _ in sorted(hero_specs, key=lambda x: x[0], reverse=True):
        bay_edges.append(hz + hh * 0.5)
        bay_edges.append(hz - hh * 0.5)
    bay_edges.append(z_bot)
    bays = [(bay_edges[k * 2 + 1], bay_edges[k * 2]) for k in range(len(bay_edges) // 2)]

    bm_ch = bmesh.new()
    bm_glow = bmesh.new()
    seam_xs = (-1.36, 1.38)
    for bz_lo, bz_hi in bays:
        z_a = bz_lo + 0.025
        z_b = bz_hi - 0.025
        bh = z_b - z_a
        if bh < 0.35:
            continue
        zc = 0.5 * (z_a + z_b)
        for sx in seam_xs:
            res_c = bmesh.ops.create_cube(bm_ch, size=1.0)
            bmesh.ops.scale(bm_ch, vec=(0.024, 0.014, bh), verts=res_c["verts"])
            bmesh.ops.translate(bm_ch, vec=(sx, wall_y + 0.008, zc), verts=res_c["verts"])
            for cap_z in (z_a + 0.011, z_b - 0.011):
                res_k = bmesh.ops.create_cube(bm_ch, size=1.0)
                bmesh.ops.scale(bm_ch, vec=(0.034, 0.018, 0.022), verts=res_k["verts"])
                bmesh.ops.translate(bm_ch, vec=(sx, wall_y + 0.006, cap_z), verts=res_k["verts"])
            res_g = bmesh.ops.create_cube(bm_glow, size=1.0)
            bmesh.ops.scale(bm_glow, vec=(0.009, 0.008, max(0.20, bh - 0.044)), verts=res_g["verts"])
            bmesh.ops.translate(bm_glow, vec=(sx, wall_y - 0.001, zc), verts=res_g["verts"])

    m_ch = bpy.data.meshes.new("BassResonatorChannelsMesh")
    bm_ch.to_mesh(m_ch)
    bm_ch.free()
    o_ch = bpy.data.objects.new("BassResonatorChannels", m_ch)
    col_stage.objects.link(o_ch)
    o_ch.data.materials.append(brass_mat)

    g_mat = bpy.data.materials.new("BassResonatorGlowMat")
    g_mat.use_nodes = True
    nt = g_mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.35, 0.16, 0.03, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.25
    bsdf.inputs["Roughness"].default_value = 0.22
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.56, 0.12, 1.0)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = 0.55
    wave.inputs["Distortion"].default_value = 0.0
    amp_node = nt.nodes.new("ShaderNodeValue")
    amp_node.outputs[0].default_value = 0.45
    mod_map = nt.nodes.new("ShaderNodeMapRange")
    mod_map.inputs["To Min"].default_value = 0.65
    mod_map.inputs["To Max"].default_value = 1.15
    mul_s = nt.nodes.new("ShaderNodeMath")
    mul_s.operation = "MULTIPLY"
    add_sh = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    nt.links.new(wave.outputs["Fac"], mod_map.inputs["Value"])
    nt.links.new(mod_map.outputs["Result"], mul_s.inputs[0])
    nt.links.new(amp_node.outputs[0], mul_s.inputs[1])
    nt.links.new(mul_s.outputs["Value"], em.inputs["Strength"])
    nt.links.new(bsdf.outputs["BSDF"], add_sh.inputs[0])
    nt.links.new(em.outputs["Emission"], add_sh.inputs[1])
    nt.links.new(add_sh.outputs["Shader"], out.inputs["Surface"])

    m_gl = bpy.data.meshes.new("BassResonatorGlowMesh")
    bm_glow.to_mesh(m_gl)
    bm_glow.free()
    o_gl = bpy.data.objects.new("BassResonatorGlow", m_gl)
    col_stage.objects.link(o_gl)
    o_gl.data.materials.append(g_mat)

    amp_out = amp_node.outputs[0]
    idle_v = 0.45
    amp_out.default_value = idle_v
    amp_out.keyframe_insert(data_path="default_value", frame=0)
    last_kf = 0
    for k, ev in enumerate(bass_events):
        fh = int(ev["frame_hit"])
        if fh <= last_kf or fh > total_frames:
            continue
        vel_s = 0.55 + 0.65 * max(0.0, min(1.0, float(ev.get("velocity", 90)) / 127.0))
        pk = 11.5 * vel_s
        nxt_fh = int(bass_events[k + 1]["frame_hit"]) if k + 1 < len(bass_events) else total_frames + 10
        f_pre = fh - 1
        f_dec = min(total_frames, fh + 11, nxt_fh - 2)
        if f_pre > last_kf:
            amp_out.default_value = idle_v
            amp_out.keyframe_insert(data_path="default_value", frame=f_pre)
        amp_out.default_value = pk
        amp_out.keyframe_insert(data_path="default_value", frame=fh)
        if f_dec > fh:
            amp_out.default_value = idle_v
            amp_out.keyframe_insert(data_path="default_value", frame=f_dec)
        last_kf = max(fh, f_dec)


def _add_disk_layer(bm: bmesh.types.BMesh, rad: float, th: float, zc: float, mat_idx: int, segs: int = 36) -> None:
    f0 = len(bm.faces)
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=rad, radius2=rad, depth=th)
    new_verts = list({v for f in bm.faces[f0:] for v in f.verts})
    bmesh.ops.translate(bm, vec=(0.0, 0.0, zc), verts=new_verts)
    for f in bm.faces[f0:]:
        f.material_index = mat_idx
        f.smooth = True


def _add_ring_band(bm: bmesh.types.BMesh, r_in: float, r_out: float, z_lo: float, z_hi: float, mat_idx: int, segs: int = 36) -> None:
    top_in, top_out, bot_in, bot_out = [], [], [], []
    for k in range(segs):
        ang = (2.0 * math.pi * k) / segs
        ca, sa = math.cos(ang), math.sin(ang)
        top_in.append(bm.verts.new((r_in * ca, r_in * sa, z_hi)))
        top_out.append(bm.verts.new((r_out * ca, r_out * sa, z_hi)))
        bot_in.append(bm.verts.new((r_in * ca, r_in * sa, z_lo)))
        bot_out.append(bm.verts.new((r_out * ca, r_out * sa, z_lo)))
    faces = []
    for k in range(segs):
        kn = (k + 1) % segs
        faces.append(bm.faces.new((top_in[k], top_out[k], top_out[kn], top_in[kn])))
        faces.append(bm.faces.new((bot_out[k], bot_in[k], bot_in[kn], bot_out[kn])))
        faces.append(bm.faces.new((top_out[k], bot_out[k], bot_out[kn], top_out[kn])))
        faces.append(bm.faces.new((bot_in[k], top_in[k], top_in[kn], bot_in[kn])))
    for f in faces:
        f.material_index = mat_idx
        f.smooth = True


def build_drum_pad_mesh_protos(
    shell_walnut_mat: bpy.types.Material,
    head_leather_mat: bpy.types.Material,
    hoop_brass_mat: bpy.types.Material,
    cymbal_bronze_mat: bpy.types.Material,
) -> dict[str, tuple[bpy.types.Mesh, bpy.types.Mesh]]:
    """Build reusable circular drumhead/cymbal/catch meshes + perimeter LED ring meshes."""
    specs = {
        "drum_kick": (0.145, 0.052),
        "drum_snare": (0.130, 0.044),
        "drum_hat": (0.110, 0.026),
        "drum_catch": (0.210, 0.062),
    }
    protos: dict[str, tuple[bpy.types.Mesh, bpy.types.Mesh]] = {}
    for kind, (pr, pth) in specs.items():
        bm = bmesh.new()
        bm_r = bmesh.new()
        if kind == "drum_hat":
            # Spun bronze mini-cymbal flat at +0.5*pth center contact, stepped shoulder outside
            _add_disk_layer(bm, pr * 0.99, pth * 0.72, -0.08 * pth, 0)
            _add_disk_layer(bm, pr * 0.58, pth * 0.36, +0.32 * pth, 0)
            mats = [cymbal_bronze_mat]
            _add_ring_band(bm_r, pr * 0.94, pr * 1.01, -0.18 * pth, +0.16 * pth, 0)
        elif kind == "drum_catch":
            _add_disk_layer(bm, pr * 0.98, pth * 0.90, -0.05 * pth, 0)
            _add_disk_layer(bm, pr * 0.86, pth * 0.16, +0.42 * pth, 1)
            _add_ring_band(bm, pr * 0.84, pr * 1.02, +0.36 * pth, +0.76 * pth, 2)
            mats = [shell_walnut_mat, head_leather_mat, hoop_brass_mat]
            _add_ring_band(bm_r, pr * 0.95, pr * 1.025, +0.12 * pth, +0.34 * pth, 0)
        else:
            # Turned shell + parchment head + top brass tension hoop
            shell_mat = hoop_brass_mat if kind == "drum_snare" else shell_walnut_mat
            _add_disk_layer(bm, pr * 0.97, pth * 0.78, -0.11 * pth, 0)
            _add_disk_layer(bm, pr * 0.94, pth * 0.24, +0.38 * pth, 1)
            _add_ring_band(bm, pr * 0.90, pr * 1.01, +0.24 * pth, +0.52 * pth, 2)
            mats = [shell_mat, head_leather_mat, hoop_brass_mat]
            _add_ring_band(bm_r, pr * 0.95, pr * 1.02, +0.06 * pth, +0.23 * pth, 0)

        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bmesh.ops.recalc_face_normals(bm_r, faces=bm_r.faces)
        m_pad = bpy.data.meshes.new(f"Proto_{kind}_Mesh")
        bm.to_mesh(m_pad)
        bm.free()
        for mat_s in mats:
            m_pad.materials.append(mat_s)

        m_ring = bpy.data.meshes.new(f"Proto_{kind}_RingMesh")
        bm_r.to_mesh(m_ring)
        bm_r.free()
        protos[kind] = (m_pad, m_ring)
    return protos


def build_marble_assembly(
    name_tag: str,
    pal_name: str,
    marble_rgb: list[float],
    swirl_rgb: tuple[float, float, float],
    radius: float,
    samples: list[dict],
    col_marbles: bpy.types.Collection,
    swirl_em_strength: float,
) -> bpy.types.Object:
    """Create hand-blown crystal jewel marble + twisted cat's-eye ribbon + 3 bubbles + natural roll keyframes."""
    mmesh = bpy.data.meshes.new(f"MarbleMesh_{name_tag}")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=40, radius=radius)
    bm.to_mesh(mmesh)
    bm.free()
    for poly in mmesh.polygons:
        poly.use_smooth = True

    m_obj = bpy.data.objects.new(f"Marble_{name_tag}_{pal_name}", mmesh)
    col_marbles.objects.link(m_obj)
    m_mat = create_colored_marble_material(f"MarbleMat_{name_tag}", marble_rgb)
    m_obj.data.materials.append(m_mat)
    m_obj.rotation_mode = "QUATERNION"

    swirl_mat = bpy.data.materials.new(f"MarbleSwirlMat_{name_tag}")
    swirl_mat.use_nodes = True
    sb = next(nd for nd in swirl_mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    sb.inputs["Base Color"].default_value = (swirl_rgb[0], swirl_rgb[1], swirl_rgb[2], 1.0)
    sb.inputs["Metallic"].default_value = 0.55
    sb.inputs["Roughness"].default_value = 0.14
    if "Coat Weight" in sb.inputs:
        sb.inputs["Coat Weight"].default_value = 1.0
    if "Emission Color" in sb.inputs:
        sb.inputs["Emission Color"].default_value = (swirl_rgb[0], swirl_rgb[1], swirl_rgb[2], 1.0)
        sb.inputs["Emission Strength"].default_value = swirl_em_strength

    bmr = bmesh.new()
    n_seg = 26
    rings_v = []
    for i in range(n_seg + 1):
        t = i / n_seg
        z = (t - 0.5) * 1.36 * radius
        sin_t = max(0.06, math.sin(math.pi * t))
        w_r = 0.46 * radius * (sin_t ** 0.72)
        th_r = 0.042 * radius * (sin_t ** 0.5) + 0.0008
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
    rm = bpy.data.meshes.new(f"MarbleCatEyeMesh_{name_tag}")
    bmr.to_mesh(rm)
    bmr.free()
    for pl in rm.polygons:
        pl.use_smooth = True

    cat_obj = bpy.data.objects.new(f"MarbleCatEye_{name_tag}", rm)
    col_marbles.objects.link(cat_obj)
    cat_obj.data.materials.append(swirl_mat)
    cat_obj.parent = m_obj
    cat_obj.rotation_euler = (math.radians(24.0), math.radians(18.0), 0.0)

    bubble_mat = bpy.data.materials.new(f"MarbleBubbleMat_{name_tag}")
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
        b_mesh = bpy.data.meshes.new(f"MarbleBubbleMesh_{name_tag}_{bk}")
        bmb = bmesh.new()
        bmesh.ops.create_uvsphere(bmb, u_segments=14, v_segments=10, radius=radius * br_u)
        bmb.to_mesh(b_mesh)
        bmb.free()
        for pl in b_mesh.polygons:
            pl.use_smooth = True
        b_obj = bpy.data.objects.new(f"MarbleBubble_{name_tag}_{bk}", b_mesh)
        col_marbles.objects.link(b_obj)
        b_obj.data.materials.append(bubble_mat)
        b_obj.parent = m_obj
        b_obj.location = (radius * bx_u, radius * by_u, radius * bz_u)

    q_cur = Quaternion((0.92, 0.18, 0.31, 0.12))
    q_cur.normalize()
    prev_pos = None
    for s in samples:
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
    return m_obj


def build_multiball_scene(
    physics_json: str,
    out_blend: str,
    audio_path: str | None = None,
    frames_dir: str | None = None,
    check_frames: list[int] | None = None,
    look: str = "dynamic",
    title_style: str = "bulb",
) -> None:
    """look: 'dynamic' (3-Act ensemble) | 'warm' | 'night'. title_style: 'bulb' (v14) | 'neon' (v15)."""
    global LOOK_NIGHT
    with open(physics_json, "r", encoding="utf-8") as f:
        plan = json.load(f)

    sections = plan.get("sections") or []
    drum_balls = plan.get("drum_balls") or []
    has_drums = len(drum_balls) > 0
    LOOK_DYNAMIC = (look == "dynamic" and len(sections) > 0)
    LOOK_NIGHT = (look == "night")
    fixed_nm = 1.0 if LOOK_NIGHT else 0.0

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
    vs.exposure = 0.05 if LOOK_DYNAMIC else (0.25 if LOOK_NIGHT else -0.10)
    scene.display_settings.display_device = "sRGB"

    base_scene.build_studio_reflection_world(scene)
    init_nm = _eval_section_state(0.0, sections, fixed_nm, has_drums)[0] if LOOK_DYNAMIC else fixed_nm
    init_bg_col = _lerp_tuple((1.0, 0.66, 0.38, 1.0), (0.10, 0.16, 0.38, 1.0), init_nm)
    init_bg_str = 0.28 + init_nm * (0.055 - 0.28)
    world_bg_nodes = []
    if scene.world and scene.world.use_nodes:
        for nd in scene.world.node_tree.nodes:
            if nd.type == "BACKGROUND":
                world_bg_nodes.append(nd)
                if not nd.inputs["Color"].is_linked:
                    nd.inputs["Color"].default_value = init_bg_col
                nd.inputs["Strength"].default_value = init_bg_str
    base_scene.setup_bloom_compositor(scene)
    if scene.compositing_node_group:
        for nd in scene.compositing_node_group.nodes:
            if nd.type == "GLARE":
                base_scene.set_socket(nd, "Threshold", 1.30 if LOOK_DYNAMIC else (1.15 if LOOK_NIGHT else 1.6))
                base_scene.set_socket(nd, "Strength", 0.38 if LOOK_DYNAMIC else (0.44 if LOOK_NIGHT else 0.34))
                base_scene.set_socket(nd, "Size", 0.56 if (LOOK_DYNAMIC or LOOK_NIGHT) else 0.55)

    templates, lib_mats, manifest = base_scene.load_instrument_library()
    chrome_mat = lib_mats["LibMat_ChromeRail"]
    hardware_dark_mat = lib_mats["LibMat_HardwareDark"]

    col_root = bpy.context.scene.collection
    col_stage = base_scene.get_or_create_collection("01_Stage", col_root)
    col_boards = base_scene.get_or_create_collection("02_Boards", col_root)
    col_rails = base_scene.get_or_create_collection("03_Rails", col_root)
    col_marbles = base_scene.get_or_create_collection("04_Marbles", col_root)
    col_fx = base_scene.get_or_create_collection("05_FX", col_root)

    all_z = [b["board_center"][2] for ball in balls + drum_balls for b in ball["boards"]]
    finale_spec = plan.get("finale")
    if finale_spec is not None:
        finale_z = float(finale_spec["z_finale"])
        finale_h = 1.42
        extra_z = [finale_z]
        if finale_spec.get("bridge"):
            extra_z.append(float(finale_spec["bridge"]["center"][2]))
        z_min = min(all_z + extra_z) - 4.0
    else:
        term_zs = [bd["board_center"][2] for ball in balls for bd in ball["boards"] if bd.get("is_terminal")]
        z_end_mid = 0.5 * (min(term_zs) + max(term_zs))
        finale_z = z_end_mid - (1.32 if title_style == "bulb" else 1.27)
        finale_h = 1.04
        z_min = min(all_z) - 4.0
    z_max = max(all_z) + 4.0

    rail_center_melody_mat, rail_side_drum_mat = create_rail_materials()
    rail_bronze_mat = lib_mats.get("LibMat_ResonatorBronze", rail_side_drum_mat)

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

    brass_rod_mat = lib_mats.get("LibMat_WallBrassTrim", rail_side_drum_mat)
    opening_z = 0.66 if has_drums else 0.56

    # Modular 3D walnut plank wall + horizontal hero title planks + fluted dark slats
    _, wall_bulb_mats, hero_specs, wall_z_bot, wall_z_top, watermark_zs = create_textured_white_wall(
        wall_y, z_min, z_max, brass_rod_mat, col_stage, opening_z=opening_z, finale_z=finale_z, finale_h=finale_h
    )
    if title_style == "bulb" and watermark_zs:
        wm_mats = create_watermark_bulb_materials(10.5 if LOOK_NIGHT else 8.5)
        for wm_idx, z_wm in enumerate(watermark_zs):
            create_bulb_watermark_plaque(wall_y, 0.0, z_wm, col_stage, wm_idx, wm_mats)
    animate_fairy_garlands(
        wall_bulb_mats,
        plan.get("percussion_events", []),
        16.0 if LOOK_NIGHT else (14.5 if LOOK_DYNAMIC else 13.5),
        total_frames,
    )
    build_wall_bass_resonators(
        wall_y,
        wall_z_bot,
        wall_z_top,
        hero_specs,
        plan.get("bass_events", []),
        brass_rod_mat,
        col_stage,
        total_frames,
    )

    # Opening Christmas light title (and static Finale only for legacy non-ensemble plans)
    open_x = (0.10 if title_style == "bulb" else 0.05) if has_drums else (0.48 if title_style == "bulb" else 0.28)
    create_custom_title_plaque(
        wall_y, open_x, opening_z, brass_rod_mat, col_stage, style=title_style, compact=False, narrow_hero=has_drums
    )
    if finale_spec is None:
        create_custom_title_plaque(wall_y, 0.0, finale_z, brass_rod_mat, col_stage, style=title_style, compact=True)

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
    wood_kb_mat = create_wood_bar_material_from_colors(
        "WoodBarMat_Keyboard",
        (0.76, 0.58, 0.32, 1.0),
        (0.56, 0.40, 0.18, 1.0),
    )

    def create_electronic_pulse_material(name: str, rgb: list[float], bar_len: float) -> tuple[bpy.types.Material, bpy.types.Node, bpy.types.Node]:
        """Square-wave / chime light trace: dim idle glow -> sweeping white-hot impact front + whole flash -> decay."""
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

        front = nodes.new("ShaderNodeMath")
        front.operation = "MULTIPLY"
        flash = nodes.new("ShaderNodeMath")
        flash.operation = "MULTIPLY"
        flash.inputs[1].default_value = 0.40
        sum_a = nodes.new("ShaderNodeMath")
        sum_a.operation = "ADD"
        add_idle = nodes.new("ShaderNodeMath")
        add_idle.operation = "ADD"
        add_idle.inputs[1].default_value = 1.0 if LOOK_NIGHT else (0.72 if LOOK_DYNAMIC else 0.55)
        white_gain = nodes.new("ShaderNodeMath")
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

    SWIRL_COLORS = [
        (0.95, 0.78, 0.34),  # Champagne Gold swirl inside Emerald marble
        (0.86, 0.18, 0.06),  # Deep Ruby-Copper swirl inside Amber marble
        (0.78, 0.94, 1.00),  # Silver-Ice Pearl swirl inside Sapphire marble
    ]
    swirl_em = 0.12 if LOOK_NIGHT else (0.08 if LOOK_DYNAMIC else 0.05)

    # Snap ALL ball hit-frames & board_centers FIRST so reverse support search sees exact final positions
    for ball in balls:
        s_by_frame = {int(s["frame"]): s["position"] for s in ball["samples"]}
        hit_frames_set = set()
        for bd in ball["boards"]:
            for hf in (bd.get("hit_frames") or [int(bd["frame_hit"])]):
                hit_frames_set.add(int(hf))
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
            f_lo = max(0, fh - 180)
            f_hi = min(total_frames, fh + 180)
            for alpha_c in (0.0, 0.25, 0.50, 0.75, 1.0):
                c_cand = [
                    c_ideal[0] + alpha_c * (c_orig[0] - c_ideal[0]),
                    c_ideal[1] + alpha_c * (c_orig[1] - c_ideal[1]),
                    c_ideal[2] + alpha_c * (c_orig[2] - c_ideal[2]),
                ]
                ok_c = True
                for f_k in range(f_lo, f_hi + 1):
                    if abs(f_k - fh) <= 3:
                        continue
                    pos_k = s_by_frame[f_k]
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

    for d_ball in drum_balls:
        ds_by_f = {int(s["frame"]): s["position"] for s in d_ball["samples"]}
        dhit_f_set = {int(bd["frame_hit"]) for bd in d_ball["boards"]}
        for bd in d_ball["boards"]:
            if bd.get("is_terminal"):
                continue
            fh = int(bd["frame_hit"])
            hc = bd.get("hit_center")
            if not hc or fh not in ds_by_f:
                continue
            cp = ds_by_f[fh]
            dx_h, dy_h, dz_h = hc[0] - cp[0], hc[1] - cp[1], hc[2] - cp[2]
            ds_by_f[fh] = [hc[0], hc[1], hc[2]]
            for af, wa in ((fh - 1, 0.32), (fh + 1, 0.32)):
                if af in ds_by_f and af not in dhit_f_set:
                    ap = ds_by_f[af]
                    ds_by_f[af] = [ap[0] + wa * dx_h, ap[1] + wa * dy_h, ap[2] + wa * dz_h]
        for s in d_ball["samples"]:
            s["position"] = ds_by_f[int(s["frame"])]

    support_obstacles = build_support_obstacles(balls, drum_balls, radius, total_frames, finale_spec=finale_spec)
    collar_mesh = bpy.data.meshes.new("BarSwivelCollarMesh")
    col_bm = bmesh.new()
    bmesh.ops.create_uvsphere(col_bm, u_segments=12, v_segments=8, radius=0.017)
    col_bm.to_mesh(collar_mesh)
    col_bm.free()
    for pl in collar_mesh.polygons:
        pl.use_smooth = True
    collar_mesh.materials.append(brass_rod_mat)

    pin_audit = {"straight": 0, "diag": 0, "elbow": 0, "fail": 0}
    rail_rod_audit = {"straight": 0, "diag": 0, "elbow": 0, "fail": 0}
    melody_marble_objs: dict[int, bpy.types.Object] = {}
    drum_marble_objs: dict[int, bpy.types.Object] = {}

    # Spawn center melody wood bars (distinguishing Synth Square Lead vs Keyboard Chime), rails, and glass marbles
    for ball in balls:
        b_idx = int(ball["ball_idx"])
        pal = ball["palette"]
        b_col = base_scene.get_or_create_collection(f"Layer_{b_idx}_{pal['name']}", col_boards)
        c_lt, c_dk = WOOD_LAYER_COLORS[b_idx % len(WOOD_LAYER_COLORS)]
        body_mat = create_wood_bar_material_from_colors(f"WoodBarMat_{b_idx}", c_lt, c_dk)

        hl_data = bpy.data.lights.new(f"HitFlashLight_{b_idx}", type="POINT")
        hl_data.energy = 0.0
        hl_data.shadow_soft_size = 0.18
        hl_data.use_shadow = False
        hl_obj = bpy.data.objects.new(f"HitFlashLight_{b_idx}", hl_data)
        col_fx.objects.link(hl_obj)
        hl_data.keyframe_insert(data_path="energy", frame=0)

        for b_pos_idx, board in enumerate(ball["boards"]):
            idx = int(board["index"])
            is_kb = (board.get("source_stem") == "keyboard") and not board.get("is_terminal")
            b_len = float(board["length"]) * (1.04 if is_kb else 1.0)
            b_wy = float(board["width_y"]) * 0.85
            b_th = float(board["thickness"])

            core_mesh = base_scene.create_hard_beveled_box(
                f"CleanBarMesh_{b_idx}_{idx:03d}", b_len, b_wy, b_th
            )
            obj = bpy.data.objects.new(f"Bar_{b_idx}_{idx:03d}_{board.get('pitch_name', '')}", core_mesh)
            b_col.objects.link(obj)
            obj.data.materials.append(wood_kb_mat if is_kb else body_mat)

            if is_kb:
                bar_pulse_rgb = [1.00, 0.84, 0.40]
                led_mesh = create_chime_inlay_mesh(
                    f"BarChimeMesh_{b_idx}_{idx:03d}", b_len * 0.82, b_th * 0.18, depth_y=0.010
                )
            else:
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

            # Reverse-searched collision-free wall support struts + brass swivel collars + flanges
            mount_xs = [c_vec.x - 0.36, c_vec.x + 0.36] if b_len > 0.80 else [c_vec.x]
            for m_i, mx in enumerate(mount_xs):
                pin_a = Vector((mx, c_vec.y + b_wy * 0.42, c_vec.z))
                mode_p, pin_b_t, p_elb_t = find_reverse_wall_anchor(
                    (pin_a.x, pin_a.y, pin_a.z), mx, c_vec.z, ("M", b_idx), idx, 0.011, wall_y, support_obstacles
                )
                pin_audit[mode_p] = pin_audit.get(mode_p, 0) + 1
                sfx = f"_{m_i}" if len(mount_xs) > 1 else ""
                if p_elb_t is not None:
                    p_elb = Vector(p_elb_t)
                    base_scene.create_cylinder_between(
                        f"BarPin_{b_idx}_{idx:03d}{sfx}_e", pin_a, p_elb, 0.011, hardware_dark_mat, b_col, vertices=14
                    )
                    base_scene.create_cylinder_between(
                        f"BarPin_{b_idx}_{idx:03d}{sfx}", p_elb, Vector(pin_b_t), 0.011, hardware_dark_mat, b_col, vertices=16
                    )
                    e_obj = bpy.data.objects.new(f"BarElbow_{b_idx}_{idx:03d}{sfx}", collar_mesh)
                    b_col.objects.link(e_obj)
                    e_obj.location = p_elb
                else:
                    base_scene.create_cylinder_between(
                        f"BarPin_{b_idx}_{idx:03d}{sfx}", pin_a, Vector(pin_b_t), 0.011, hardware_dark_mat, b_col, vertices=16
                    )
                if mode_p in ("diag", "elbow"):
                    c_obj = bpy.data.objects.new(f"BarCollar_{b_idx}_{idx:03d}{sfx}", collar_mesh)
                    b_col.objects.link(c_obj)
                    c_obj.location = pin_a
                base_scene.create_wall_flange(
                    f"BarFlange_{b_idx}_{idx:03d}{sfx}",
                    (pin_b_t[0], wall_y + 0.003, pin_b_t[2]),
                    0.024,
                    brass_rod_mat,
                    b_col,
                )

            # Hit recoil & LED emissive pulse (supports multi-hit Finale stepping plank)
            hit_frames_list = sorted(set(int(h) for h in (board.get("hit_frames") or [int(board["frame_hit"])])))
            vel = float(board.get("velocity", 90.0))
            vel_scale = 0.55 + 0.65 * max(0.0, min(1.0, vel / 127.0))
            half_span = b_len * 0.46
            amp_pk = 20.0 * vel_scale
            for hi_i, fh in enumerate(hit_frames_list):
                nxt_hit = hit_frames_list[hi_i + 1] if hi_i + 1 < len(hit_frames_list) else (
                    int(ball["boards"][b_pos_idx + 1]["frame_hit"]) if b_pos_idx + 1 < len(ball["boards"]) else total_frames + 100
                )
                f_rec_peak = min(total_frames, fh + 2, nxt_hit - 2)
                f_rec_end = min(total_frames, fh + 10, nxt_hit - 1)
                obj.location = c_vec
                if hi_i == 0:
                    obj.keyframe_insert(data_path="location", frame=max(0, fh - 1))
                obj.keyframe_insert(data_path="location", frame=fh)
                if f_rec_peak > fh:
                    obj.location = c_vec - n * (0.018 * vel_scale)
                    obj.keyframe_insert(data_path="location", frame=f_rec_peak)
                obj.location = c_vec
                obj.keyframe_insert(data_path="location", frame=max(fh + 1, f_rec_end))

                f_rad_end = min(total_frames, fh + 7, nxt_hit - 2)
                p_rad_node.outputs[0].default_value = 0.0
                if hi_i == 0:
                    p_rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=max(0, fh - 1))
                p_rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=fh)
                p_rad_node.outputs[0].default_value = half_span
                p_rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=max(fh + 1, f_rad_end))
                p_rad_node.outputs[0].default_value = 0.0
                p_rad_node.outputs[0].keyframe_insert(data_path="default_value", frame=min(total_frames, max(fh + 2, f_rad_end + 1)))

                p_amp_node.outputs[0].default_value = 0.0
                if hi_i == 0:
                    p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=max(0, fh - 2))
                p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=max(0, fh - 1))
                p_amp_node.outputs[0].default_value = amp_pk
                p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=fh)
                f_d1 = min(total_frames, fh + 6, nxt_hit - 3)
                f_d2 = min(total_frames, fh + 12, nxt_hit - 2)
                f_d3 = min(total_frames, fh + 18, nxt_hit - 1)
                if f_d1 > fh:
                    p_amp_node.outputs[0].default_value = 0.55 * amp_pk
                    p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=f_d1)
                if f_d2 > max(fh, f_d1):
                    p_amp_node.outputs[0].default_value = 0.18 * amp_pk
                    p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=f_d2)
                p_amp_node.outputs[0].default_value = 0.0
                p_amp_node.outputs[0].keyframe_insert(data_path="default_value", frame=max(fh + 1, f_d3))

                if not board.get("is_terminal") or finale_spec is not None:
                    nm_hit = _eval_section_state(fh / fps, sections, fixed_nm, has_drums)[0] if LOOK_DYNAMIC else fixed_nm
                    flash_peak = (8.0 + 10.0 * nm_hit) * vel_scale
                    f_pre = max(0, fh - 1)
                    f_mid = min(total_frames, fh + 4)
                    f_end = min(total_frames, fh + 9, nxt_hit - 2)
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
            create_smooth_pipe_polyline(f"{rid}_front", r["pipe_front"], r["pipe_radius"], rail_center_melody_mat, col_rails)
            create_smooth_pipe_polyline(f"{rid}_back", r["pipe_back"], r["pipe_radius"], rail_center_melody_mat, col_rails)
            for t_idx, tie_pts in enumerate(r.get("ties", [])):
                create_smooth_pipe_polyline(f"{rid}_tie_{t_idx:02d}", tie_pts, 0.0085, hardware_dark_mat, col_rails)

            front_pts = r["pipe_front"]
            back_pts = r["pipe_back"]
            n_bp = len(back_pts)
            stations = sorted(set([min(n_bp - 2, 4), n_bp // 2, max(2, n_bp - 5)]))
            n_rods = 0
            for k_i in stations:
                rear_is_front = (front_pts[k_i][1] > back_pts[k_i][1])
                px_, py_, pz_ = front_pts[k_i] if rear_is_front else back_pts[k_i]
                mode_r, pin_b_r, p_elb_r = find_reverse_wall_anchor(
                    (px_, py_, pz_), px_, pz_, ("M", b_idx), None, 0.0080, wall_y, support_obstacles,
                    self_rail_id=rid, self_rail_rear_is_front=rear_is_front,
                )
                rail_rod_audit[mode_r] = rail_rod_audit.get(mode_r, 0) + 1
                if p_elb_r is not None:
                    elb_v = Vector(p_elb_r)
                    base_scene.create_cylinder_between(
                        f"{rid}_riser_{n_rods:02d}", Vector((px_, py_, pz_)), elb_v, 0.0080, hardware_dark_mat, col_rails, vertices=12
                    )
                    base_scene.create_cylinder_between(
                        f"{rid}_support_{n_rods:02d}", elb_v, Vector(pin_b_r), 0.0080, hardware_dark_mat, col_rails, vertices=12
                    )
                else:
                    base_scene.create_cylinder_between(
                        f"{rid}_support_{n_rods:02d}", Vector((px_, py_, pz_)), Vector(pin_b_r), 0.0080, hardware_dark_mat, col_rails, vertices=12
                    )
                base_scene.create_wall_flange(
                    f"{rid}_support_flange_{n_rods:02d}",
                    (pin_b_r[0], wall_y + 0.003, pin_b_r[2]),
                    0.020,
                    brass_rod_mat,
                    col_rails,
                )
                n_rods += 1
            print(f"rail {rid}: {n_rods} wall support rods")

        melody_marble_objs[b_idx] = build_marble_assembly(
            f"{b_idx}",
            pal["name"],
            pal["color"],
            SWIRL_COLORS[b_idx % len(SWIRL_COLORS)],
            radius,
            ball["samples"],
            col_marbles,
            swirl_em,
        )

    print(f"[Reverse Support Audit] melody_boards={pin_audit} | melody_rail_rods={rail_rod_audit}")
    assert pin_audit["fail"] == 0 and rail_rod_audit["fail"] == 0, f"Reverse support search failed: {pin_audit}, {rail_rod_audit}"

    # Spawn flanking Left Kick & Right Snare/HiHat Drum Towers + silent bronze parking glides
    if drum_balls:
        drum_shell_walnut = create_wood_bar_material_from_colors(
            "DrumShellWalnutMat", (0.20, 0.09, 0.045, 1.0), (0.09, 0.038, 0.018, 1.0)
        )
        drum_head_leather = bpy.data.materials.new("DrumHeadLeatherMat")
        drum_head_leather.use_nodes = True
        d_nt = drum_head_leather.node_tree
        d_bsdf = next(nd for nd in d_nt.nodes if nd.type == "BSDF_PRINCIPLED")
        d_bsdf.inputs["Base Color"].default_value = (0.76, 0.65, 0.48, 1.0)
        d_bsdf.inputs["Roughness"].default_value = 0.42

        cymbal_bronze_mat = bpy.data.materials.new("DrumCymbalBronzeMat")
        cymbal_bronze_mat.use_nodes = True
        c_bsdf = next(nd for nd in cymbal_bronze_mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
        c_bsdf.inputs["Base Color"].default_value = (0.78, 0.56, 0.24, 1.0)
        c_bsdf.inputs["Metallic"].default_value = 0.92
        c_bsdf.inputs["Roughness"].default_value = 0.22

        drum_protos = build_drum_pad_mesh_protos(
            drum_shell_walnut, drum_head_leather, brass_rod_mat, cymbal_bronze_mat
        )
        DRUM_SWIRL = [
            (0.98, 0.72, 0.22),  # Warm gold ribbon inside Smoked Ruby Kick sphere
            (0.18, 0.96, 0.48),  # Emerald ribbon inside Champagne Silver Snare sphere
        ]
        DRUM_RING_COLORS = {
            "drum_kick": [1.00, 0.22, 0.06],
            "drum_snare": [0.16, 1.00, 0.44],
            "drum_hat": [1.00, 0.84, 0.34],
            "drum_catch": [1.00, 0.70, 0.22],
        }
        idle_ring = 0.95 if (LOOK_NIGHT or LOOK_DYNAMIC) else 0.65

        y_hub_mesh = bpy.data.meshes.new("DrumYHubMesh")
        y_hbm = bmesh.new()
        bmesh.ops.create_uvsphere(y_hbm, u_segments=14, v_segments=10, radius=0.0155)
        y_hbm.to_mesh(y_hub_mesh)
        y_hbm.free()
        for pl in y_hub_mesh.polygons:
            pl.use_smooth = True
        y_hub_mesh.materials.append(brass_rod_mat)

        y_clamp_mesh = bpy.data.meshes.new("DrumYClampMesh")
        y_cbm = bmesh.new()
        bmesh.ops.create_uvsphere(y_cbm, u_segments=14, v_segments=10, radius=0.0205)
        y_cbm.to_mesh(y_clamp_mesh)
        y_cbm.free()
        for pl in y_clamp_mesh.polygons:
            pl.use_smooth = True
        y_clamp_mesh.materials.append(brass_rod_mat)

        for d_ball in drum_balls:
            d_idx = int(d_ball["drum_idx"])
            r_d = float(d_ball["radius"])
            pal = d_ball["palette"]
            d_col = base_scene.get_or_create_collection(f"Drum_{d_idx}_{pal['name']}", col_boards)

            hl_d = bpy.data.lights.new(f"DrumFlashLight_{d_idx}", type="POINT")
            hl_d.energy = 0.0
            hl_d.shadow_soft_size = 0.22
            hl_d.use_shadow = False
            hl_d_obj = bpy.data.objects.new(f"DrumFlashLight_{d_idx}", hl_d)
            col_fx.objects.link(hl_d_obj)
            hl_d.keyframe_insert(data_path="energy", frame=0)

            for p_pos_i, bd in enumerate(d_ball["boards"]):
                idx = int(bd["index"])
                b_kind = str(bd.get("board_kind", "drum_kick"))
                m_pad_p, m_ring_p = drum_protos[b_kind]
                ring_rgb = DRUM_RING_COLORS.get(b_kind, [1.0, 0.5, 0.15])

                p_obj = bpy.data.objects.new(f"DrumPad_{d_idx}_{idx:03d}_{b_kind}", m_pad_p)
                d_col.objects.link(p_obj)
                u = Vector(bd["u"]).normalized()
                v = Vector((0.0, 1.0, 0.0))
                n = Vector(bd["n"]).normalized()
                rot_mat = Matrix(((u.x, v.x, n.x), (u.y, v.y, n.y), (u.z, v.z, n.z)))
                c_vec = Vector(bd["board_center"])
                p_obj.location = c_vec
                p_obj.rotation_mode = "QUATERNION"
                p_obj.rotation_quaternion = rot_mat.to_quaternion()

                # Per-pad ring mesh copy with dedicated emissive flash material
                r_mesh = m_ring_p.copy()
                r_mesh.name = f"DrumRingMesh_{d_idx}_{idx:03d}"
                r_mat = bpy.data.materials.new(f"DrumRingMat_{d_idx}_{idx:03d}")
                r_mat.use_nodes = True
                r_nt = r_mat.node_tree
                r_nt.nodes.clear()
                r_out = r_nt.nodes.new("ShaderNodeOutputMaterial")
                r_em = r_nt.nodes.new("ShaderNodeEmission")
                r_em.inputs["Color"].default_value = (ring_rgb[0], ring_rgb[1], ring_rgb[2], 1.0)
                r_em.inputs["Strength"].default_value = idle_ring
                r_nt.links.new(r_em.outputs["Emission"], r_out.inputs["Surface"])
                r_mesh.materials.append(r_mat)

                r_obj = bpy.data.objects.new(f"DrumRing_{d_idx}_{idx:03d}", r_mesh)
                d_col.objects.link(r_obj)
                r_obj.parent = p_obj

                # Structural rear mounting rod + brass wall flange
                pad_r_f = float(bd.get("pad_radius", 0.12))
                pin_a = c_vec + Vector((0.0, pad_r_f * 0.55, 0.0))
                pin_b = Vector((c_vec.x, wall_y + 0.025, c_vec.z))
                base_scene.create_cylinder_between(
                    f"DrumPin_{d_idx}_{idx:03d}",
                    pin_a,
                    pin_b,
                    0.012,
                    hardware_dark_mat,
                    d_col,
                    vertices=16,
                )
                base_scene.create_wall_flange(
                    f"DrumFlange_{d_idx}_{idx:03d}",
                    (c_vec.x, wall_y + 0.003, c_vec.z),
                    0.026,
                    brass_rod_mat,
                    d_col,
                )

                # Impact recoil + LED ring flash + FlashLight bounce
                fh = int(bd["frame_hit"])
                vel = float(bd.get("velocity", 95.0))
                vel_s = 0.55 + 0.65 * max(0.0, min(1.0, vel / 127.0))

                p_obj.location = c_vec
                p_obj.keyframe_insert(data_path="location", frame=max(0, fh - 1))
                p_obj.keyframe_insert(data_path="location", frame=fh)
                p_obj.location = c_vec - n * (0.020 * vel_s)
                p_obj.keyframe_insert(data_path="location", frame=min(total_frames, fh + 2))
                p_obj.location = c_vec
                p_obj.keyframe_insert(data_path="location", frame=min(total_frames, fh + 10))

                pk_r = 24.0 * vel_s
                r_sock = r_em.inputs["Strength"]
                r_sock.default_value = idle_ring
                r_sock.keyframe_insert(data_path="default_value", frame=max(0, fh - 1))
                r_sock.default_value = pk_r
                r_sock.keyframe_insert(data_path="default_value", frame=fh)
                if fh + 5 <= total_frames:
                    r_sock.default_value = 0.42 * pk_r
                    r_sock.keyframe_insert(data_path="default_value", frame=fh + 5)
                if fh + 13 <= total_frames:
                    r_sock.default_value = idle_ring
                    r_sock.keyframe_insert(data_path="default_value", frame=fh + 13)

                if not bd.get("is_terminal") or finale_spec is not None:
                    nm_h = _eval_section_state(fh / fps, sections, fixed_nm, True)[0] if LOOK_DYNAMIC else fixed_nm
                    fpk = (10.0 + 12.0 * nm_h) * vel_s
                    f_pre = max(0, fh - 1)
                    f_mid = min(total_frames, fh + 4)
                    f_end = min(total_frames, fh + 10)
                    if p_pos_i + 1 < len(d_ball["boards"]):
                        f_end = min(f_end, int(d_ball["boards"][p_pos_i + 1]["frame_hit"]) - 2)
                    f_mid = min(f_mid, max(fh, f_end - 1))

                    hl_d_obj.location = (c_vec.x, c_vec.y - 0.24, c_vec.z)
                    hl_d.color = tuple(ring_rgb)
                    hl_d.energy = 0.0
                    hl_d_obj.keyframe_insert(data_path="location", frame=f_pre)
                    hl_d.keyframe_insert(data_path="color", frame=f_pre)
                    hl_d.keyframe_insert(data_path="energy", frame=f_pre)
                    hl_d.energy = fpk
                    hl_d.keyframe_insert(data_path="energy", frame=fh)
                    if f_mid > fh:
                        hl_d.energy = 0.42 * fpk
                        hl_d.keyframe_insert(data_path="energy", frame=f_mid)
                    hl_d_obj.location = (c_vec.x, c_vec.y - 0.24, c_vec.z)
                    hl_d.color = tuple(ring_rgb)
                    hl_d.energy = 0.0
                    f_fin = max(fh + 1, f_end)
                    hl_d_obj.keyframe_insert(data_path="location", frame=f_fin)
                    hl_d.keyframe_insert(data_path="color", frame=f_fin)
                    hl_d.keyframe_insert(data_path="energy", frame=f_fin)

            # Guide pipes (burnished antique copper-bronze) & Y-shaped wall brackets
            for r in d_ball["rails"]:
                rid = r["id"]
                create_smooth_pipe_polyline(f"{rid}_left", r["pipe_front"], r["pipe_radius"], rail_side_drum_mat, col_rails)
                create_smooth_pipe_polyline(f"{rid}_right", r["pipe_back"], r["pipe_radius"], rail_side_drum_mat, col_rails)
                for bi, yb in enumerate(r.get("y_brackets", [])):
                    p_fork = Vector(yb["fork"])
                    p_left = Vector(yb["left"])
                    p_right = Vector(yb["right"])
                    p_wall = Vector((float(yb["wall_x"]), wall_y + 0.025, float(yb["wall_z"])))
                    base_scene.create_cylinder_between(
                        f"{rid}_ystem_{bi:02d}", p_fork, p_wall, 0.0095, hardware_dark_mat, col_rails, vertices=14
                    )
                    base_scene.create_cylinder_between(
                        f"{rid}_yarmL_{bi:02d}", p_fork, p_left, 0.0080, hardware_dark_mat, col_rails, vertices=12
                    )
                    base_scene.create_cylinder_between(
                        f"{rid}_yarmR_{bi:02d}", p_fork, p_right, 0.0080, hardware_dark_mat, col_rails, vertices=12
                    )
                    hub_obj = bpy.data.objects.new(f"{rid}_yhub_{bi:02d}", y_hub_mesh)
                    col_rails.objects.link(hub_obj)
                    hub_obj.location = p_fork
                    cl_obj = bpy.data.objects.new(f"{rid}_yclampL_{bi:02d}", y_clamp_mesh)
                    col_rails.objects.link(cl_obj)
                    cl_obj.location = p_left
                    cr_obj = bpy.data.objects.new(f"{rid}_yclampR_{bi:02d}", y_clamp_mesh)
                    col_rails.objects.link(cr_obj)
                    cr_obj.location = p_right
                    base_scene.create_wall_flange(
                        f"{rid}_yflange_{bi:02d}",
                        (float(yb["wall_x"]), wall_y + 0.003, float(yb["wall_z"])),
                        0.023,
                        brass_rod_mat,
                        col_rails,
                    )

            drum_marble_objs[d_idx] = build_marble_assembly(
                f"Drum_{d_idx}",
                pal["name"],
                pal["color"],
                DRUM_SWIRL[d_idx % len(DRUM_SWIRL)],
                r_d,
                d_ball["samples"],
                col_marbles,
                swirl_em,
            )

    if finale_spec is not None:
        create_animated_finale_title_plaque(
            wall_y, finale_spec, col_stage, col_fx, melody_marble_objs, drum_marble_objs
        )
        if finale_spec.get("bridge"):
            build_unified_finale_bridge(
                wall_y,
                finale_spec["bridge"],
                int(finale_spec.get("credit_fade_end_frame", total_frames - 10)),
                brass_rod_mat,
                hardware_dark_mat,
                col_stage,
                col_fx,
                total_frames,
            )

    # Soft indoor warm Christmas lighting + Three-Act dynamic blending
    key_data = bpy.data.lights.new("StudioKeyLight", type="SPOT")
    key_data.energy = 5500.0
    key_data.spot_size = math.radians(56.0)
    key_data.spot_blend = 0.90
    key_data.shadow_soft_size = 1.35
    key_data.use_shadow = True
    key_data.color = (1.0, 0.72, 0.40)
    key_obj = bpy.data.objects.new("StudioKeyLight", key_data)
    col_stage.objects.link(key_obj)

    fill_data = bpy.data.lights.new("StudioRimLight", type="AREA")
    fill_data.energy = 1020.0
    fill_data.size = 7.0
    fill_data.use_shadow = False
    fill_data.color = (1.0, 0.64, 0.38)
    fill_obj = bpy.data.objects.new("StudioRimLight", fill_data)
    col_stage.objects.link(fill_obj)

    cam_data = bpy.data.cameras.new("MainCamera16x9")
    cam_data.lens = 38.0 if has_drums else 42.0
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

    # Smooth centered Z & vertical span across active balls so all 5 stay cleanly inside 16:9 FOV
    all_render_balls = balls + drum_balls
    raw_z = []
    raw_span = []
    for f in range(total_frames + 1):
        zs_mel = [b["samples"][f]["position"][2] for b in balls]
        zs_all = [b["samples"][f]["position"][2] for b in all_render_balls]
        z_mel_avg = sum(zs_mel) / len(zs_mel)
        z_mid_all = 0.5 * (min(zs_all) + max(zs_all))
        raw_z.append(0.5 * (z_mel_avg + z_mid_all) if has_drums else z_mel_avg)
        raw_span.append(max(zs_all) - min(zs_all))
    hw = 24
    weights = [math.exp(-0.5 * (k / (hw * 0.45)) ** 2) for k in range(-hw, hw + 1)]
    cam_x = 0.16 if has_drums else 0.36
    for f in range(total_frames + 1):
        az = sp_z = ws = 0.0
        for idx_w, off in enumerate(range(-hw, hw + 1)):
            j = max(0, min(total_frames, f + off))
            w = weights[idx_w]
            az += w * raw_z[j]
            sp_z += w * raw_span[j]
            ws += w
        tz = az / ws
        span_s = sp_z / ws
        t_sec_f = f / fps
        nm, lens_mm, cam_y = _eval_section_state(
            t_sec_f, sections if LOOK_DYNAMIC else None, fixed_nm, has_drums
        )
        w_fin = 0.0
        if finale_spec is not None:
            t_c0 = float(finale_spec["t_cam_lock_start"])
            t_c1 = float(finale_spec["t_cam_lock_full"])
            if t_sec_f > t_c0:
                u_fin = min(1.0, (t_sec_f - t_c0) / max(1e-4, t_c1 - t_c0))
                w_fin = 0.5 - 0.5 * math.cos(math.pi * u_fin)

        span_eff = max(0.0, span_s - 2.2) * (1.0 - w_fin)
        if span_eff > 0.0:
            cam_y -= min(0.65, span_eff * 0.55)

        cam_x_f = cam_x * (1.0 - w_fin)
        base_cz = tz + 0.42
        base_tz = tz - 0.15
        if finale_spec is not None:
            fz_anchor = float(finale_spec["z_finale"])
            cam_z_f = base_cz + w_fin * ((fz_anchor + 0.14) - base_cz)
            tgt_y_f = 0.45 * w_fin
            tgt_z_f = base_tz + w_fin * ((fz_anchor - 0.12) - base_tz)
            tz_light = tz + w_fin * (fz_anchor - tz)
        else:
            cam_z_f = base_cz
            tgt_y_f = 0.0
            tgt_z_f = base_tz
            tz_light = tz

        cam.location = (cam_x_f, cam_y, cam_z_f)
        cam.keyframe_insert(data_path="location", frame=f)
        target_empty.location = (0.0, tgt_y_f, tgt_z_f)
        target_empty.keyframe_insert(data_path="location", frame=f)

        if f % 4 == 0 or f == total_frames:
            cam_data.lens = lens_mm
            cam_data.keyframe_insert(data_path="lens", frame=f)

        if f % 10 == 0 or f == total_frames:
            kx = -3.4 + nm * (-0.9 - (-3.4))
            ky = -4.5 + nm * (-4.8 - (-4.5))
            kz = tz_light + (3.4 + nm * (2.8 - 3.4))
            tx = 0.35 * (1.0 - nm) * (1.0 - w_fin)
            ty = 0.55 - 0.25 * nm
            tz_aim = tz_light - (0.45 - 0.30 * nm)
            key_obj.location = (kx, ky, kz)
            key_dir = Vector((tx, ty, tz_aim)) - Vector(key_obj.location)
            key_obj.rotation_euler = key_dir.to_track_quat("-Z", "Y").to_euler()
            key_data.energy = 5500.0 + nm * (3000.0 - 5500.0)
            key_data.spot_size = math.radians(56.0 + nm * (48.0 - 56.0))
            key_obj.keyframe_insert(data_path="location", frame=f)
            key_obj.keyframe_insert(data_path="rotation_euler", frame=f)
            key_data.keyframe_insert(data_path="energy", frame=f)
            key_data.keyframe_insert(data_path="spot_size", frame=f)

            fill_obj.location = (3.8, -1.8 + nm * (-2.4 - (-1.8)), tz_light + (1.4 + nm * 0.4))
            fill_obj.rotation_euler = (math.radians(74.0), 0.0, math.radians(54.0))
            fill_data.energy = 1020.0 + nm * (360.0 - 1020.0)
            fill_data.color = _lerp_tuple((1.0, 0.64, 0.38), (0.36, 0.54, 1.0), nm)
            fill_obj.keyframe_insert(data_path="location", frame=f)
            fill_data.keyframe_insert(data_path="energy", frame=f)
            fill_data.keyframe_insert(data_path="color", frame=f)

            bg_col = _lerp_tuple((1.0, 0.66, 0.38, 1.0), (0.10, 0.16, 0.38, 1.0), nm)
            bg_str = 0.28 + nm * (0.055 - 0.28)
            for nd in world_bg_nodes:
                if not nd.inputs["Color"].is_linked:
                    nd.inputs["Color"].default_value = bg_col
                    nd.inputs["Color"].keyframe_insert(data_path="default_value", frame=f)
                nd.inputs["Strength"].default_value = bg_str
                nd.inputs["Strength"].keyframe_insert(data_path="default_value", frame=f)

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
    print(f"Saved 16:9 ensemble scene: {out_blend} (frames 0..{total_frames}, {len(balls)} melody + {len(drum_balls)} drum balls)")

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
    look_arg = args[5] if len(args) >= 6 else "dynamic"
    title_arg = args[6] if len(args) >= 7 else "bulb"
    build_multiball_scene(physics_json, out_blend, audio_path=audio_path, frames_dir=frames_dir, check_frames=check_frames, look=look_arg, title_style=title_arg)
