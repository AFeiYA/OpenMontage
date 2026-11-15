#!/usr/bin/env python3
"""16:9 (1920x1080 @ 60fps) Multi-Template Widescreen Marble Concert Builder.

Completely decoupled from the 9:16 vertical pipeline (`blender_render_marble_run.py`).
Supports all three modular choreography templates in `landscape_templates.py`:
- `serpentine_descent`: Continuous Left <-> Right stepped downward cascade + chase cam.
- `diamond_loop`: Seamless 4-edge 3D Diamond Loop with spring piston climbs.
- `horological_wheel`: Rotating 360-deg Clockwork Music-Box Turntable + outer brass scoop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import bpy
from mathutils import Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from landscape_templates import (
    FPS,
    TEMPLATE_PLANNERS,
    build_concert_world,
    build_parallel_rail_geometry,
    clear_scene,
    create_hero_marble,
    create_lighting_rig,
    create_stage_backdrop_and_props,
    get_eevee_engine,
    get_or_create_collection,
    load_notes_slice,
    load_static_instrument_library,
    make_pbr_material,
    setup_bloom_compositor,
    setup_cinema_camera,
    spawn_station,
    verify_zero_intersections,
)

PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
RENDERS_DIR = os.path.join(PROJECT_DIR, "renders")


def _configure_render_settings(scene: bpy.types.Scene) -> None:
    scene.render.engine = get_eevee_engine()
    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 24
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = False
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.12


def _pick_auto_stills(plan: dict) -> list[int]:
    bounces = plan.get("bounces", [])
    total_f = int(plan["total_frames"])
    if not bounces:
        return [min(12, total_f), max(0, total_f // 2)]
    f_early = int(bounces[min(1, len(bounces) - 1)]["frame"])
    f_mid = int(bounces[len(bounces) // 2]["frame"])
    if plan["template"] == "diamond_loop":
        # Prefer showing an active climb or mid-diamond phrase switch
        f_mid = max(f_early + 30, int(round(total_f * 0.56)))
    elif plan["template"] == "horological_wheel":
        f_mid = max(f_early + 30, int(round(total_f * 0.42)))
    return [max(0, min(total_f, f_early)), max(0, min(total_f, f_mid))]


def build_template_scene(
    template_name: str,
    notes_json: str,
    physics_out: str,
    blend_out: str,
    render_frames_dir: str | None = None,
    still_spec: str = "",
    render_anim: bool = False,
    max_note_t: float = 10.0,
) -> dict:
    if template_name not in TEMPLATE_PLANNERS:
        raise ValueError(f"Unknown template '{template_name}'. Choices: {list(TEMPLATE_PLANNERS)}")

    clear_scene()
    scene = bpy.context.scene
    _configure_render_settings(scene)

    templates, mats, manifest = load_static_instrument_library()
    notes = load_notes_slice(notes_json, max_note_t=max_note_t)
    if not notes:
        raise RuntimeError(f"No notes found in {notes_json} within <= {max_note_t:.2f}s")

    plan = TEMPLATE_PLANNERS[template_name](notes, templates, manifest)

    col_stage = get_or_create_collection("Stage_16x9")
    col_inst = get_or_create_collection("Instruments_16x9")
    col_rails = get_or_create_collection("Rails_16x9")
    col_fx = get_or_create_collection("ImpactSparks_16x9")
    col_marble = get_or_create_collection("Marble_16x9")
    col_cam = get_or_create_collection("Camera_16x9")

    zs = [b["marble_pos"][2] for b in plan["bounces"]]
    z_mid = 0.5 * (min(zs) + max(zs)) if zs else 0.0

    build_concert_world(scene)
    setup_bloom_compositor(scene)
    create_lighting_rig(col_stage, z_center=z_mid)
    wheel_root = create_stage_backdrop_and_props(plan, col_stage, mats)

    rail_mat = make_pbr_material("BrassCoasterRail", (0.76, 0.56, 0.22), metallic=0.95, roughness=0.22, coat=0.35)
    tie_dark = make_pbr_material("RailTieDarkBronze", (0.14, 0.10, 0.08), metallic=0.75, roughness=0.32)
    brass_mat = mats.get("LibMat_Brass") or rail_mat

    for st in plan["stations"]:
        parent_root = wheel_root if st.get("wheel_local") else None
        spawn_station(st, templates, mats, manifest, col_inst, col_fx, wheel_root=parent_root)

    rails_geom = []
    for rp in plan["rails_plan"]:
        pts_v = [Vector(p) for p in rp["points"]]
        rails_geom.append(
            build_parallel_rail_geometry(rp["name"], pts_v, rail_mat, brass_mat, tie_dark, col_rails)
        )

    verify_zero_intersections(plan, templates, manifest, rails_geom)

    create_hero_marble(col_marble, plan["frame_samples"], plan["bounces"])
    setup_cinema_camera(scene, col_cam, plan)

    scene.frame_start = 0
    scene.frame_end = int(plan["total_frames"])

    os.makedirs(os.path.dirname(physics_out), exist_ok=True)
    with open(physics_out, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    os.makedirs(os.path.dirname(blend_out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend_out)
    print(f"[Saved 16:9 Scene] {blend_out} ({plan['total_frames']} frames @ {FPS}fps = {plan['duration_s']}s)")

    stills: list[int] = []
    if still_spec.strip().lower() == "auto":
        stills = _pick_auto_stills(plan)
    elif still_spec.strip():
        stills = [int(x.strip()) for x in still_spec.split(",") if x.strip()]

    if render_frames_dir and stills:
        os.makedirs(render_frames_dir, exist_ok=True)
        for sf in stills:
            f_clamped = max(0, min(scene.frame_end, int(sf)))
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
    ap = argparse.ArgumentParser(description="Standalone 16:9 Multi-Template Marble Concert Builder")
    ap.add_argument(
        "--template",
        default="all",
        choices=["all", *TEMPLATE_PLANNERS.keys()],
        help="Template to build (default: all)",
    )
    ap.add_argument("--notes", default=os.path.join(ARTIFACTS_DIR, "sample_14s_notes.json"))
    ap.add_argument("--max-note-t", type=float, default=10.0)
    ap.add_argument("--stills", default="", help="Comma-separated frame numbers or 'auto'")
    ap.add_argument("--render-anim", action="store_true", help="Render full 60fps PNG sequence")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    targets = list(TEMPLATE_PLANNERS.keys()) if args.template == "all" else [args.template]
    tag_sec = f"{int(round(args.max_note_t))}s"

    for tname in targets:
        stem = f"concert_16x9_{tname}_{tag_sec}"
        physics_out = os.path.join(ARTIFACTS_DIR, f"{stem}_physics.json")
        blend_out = os.path.join(RENDERS_DIR, f"{stem}.blend")
        frames_dir = os.path.join(RENDERS_DIR, f"frames_{tname}_{tag_sec}") if (args.stills or args.render_anim) else None
        build_template_scene(
            template_name=tname,
            notes_json=args.notes,
            physics_out=physics_out,
            blend_out=blend_out,
            render_frames_dir=frames_dir,
            still_spec=args.stills,
            render_anim=args.render_anim,
            max_note_t=args.max_note_t,
        )
