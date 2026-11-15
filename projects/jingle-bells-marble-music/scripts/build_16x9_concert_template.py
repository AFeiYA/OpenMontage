#!/usr/bin/env python3
"""16:9 (1920x1080 @ 60fps) Widescreen Polyphonic Concert Theatre Template Builder.

Designed from the ground up for long-form widescreen marble music productions:
- Multi-marble polyphonic choreography (2+ color-coded marbles: Emerald Lead + Gold Counterpoint,
  optional Sapphire Synth/Bass) with sweeping left-right call-and-response and climax convergence.
- Exact 60fps ballistic projectile arcs + Hermite S-curve twin-pipe guide chutes.
- 4 themed acoustic Act Chambers connected seamlessly vertically & diagonally across X in [-3.4, +3.4]:
    * Act I  : Conservatory Salon     (Smoked Walnut + Burnished Brass)
    * Act II : Strings & Brass Rotunda (Oxblood Velvet + Gold Filigree)
    * Act III: Folklore Pavilion       (Slate-Jade + Copper Ribs)
    * Act IV : Neon Synth Vault        (Carbon Obsidian + Cyan/Emerald Light Bars)
- Direct reuse of the 64-instrument v3.0 library (`instrument_library_v2.blend`) and per-part
  velocity-coupled kinetic + emissive animations from `blender_render_marble_run.py`.
- 3-mode 16:9 Cinema Camera Choreography:
    1) Establishing Wide Panorama (opening marquee + climax convergence)
    2) Active Polyphonic Rule-of-Thirds Focus Blend
    3) Subtle Velocity Impact Micro-Impulse
- Outputs structured physics plan JSON, HUD event sidecar JSON (`*_hud.json`), and ready-to-render
  `1920x1080` Blender scene (`.blend`).
"""

from __future__ import annotations

import json
import math
import os
import sys

import bpy
import bmesh
from mathutils import Matrix, Quaternion, Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from blender_render_marble_run import (
    DEFAULT_LIB_BLEND,
    DEFAULT_MANIFEST_JSON,
    build_studio_reflection_world,
    clear_scene,
    create_cylinder_between,
    create_pipe_polyline,
    create_shockwave_ring_mesh,
    create_wall_flange,
    get_eevee_engine,
    get_or_create_collection,
    load_instrument_library,
    set_socket,
    setup_bloom_compositor,
    spawn_instrument_board,
)

# ---------------------------------------------------------------------------
# Widescreen Theatre Constants & Act Themes
# ---------------------------------------------------------------------------

FPS = 60
GRAVITY = 20.0
MARBLE_RADIUS = 0.105
WALL_Y = 0.84
BOARD_THICKNESS = 0.072
BOARD_WIDTH_Y = 0.24
RAIL_HALF_GAP = 0.068
RAIL_PIPE_R = 0.015
BALLISTIC_MAX_DT = 0.62

ACT_THEMES = [
    {
        "act_idx": 0,
        "code": "ACT I",
        "title": "ACT I  ·  CONSERVATORY SALON",
        "z_top": 5.95,
        "z_bot": 2.95,
        "panel_color": (0.022, 0.018, 0.015),
        "accent_color": (0.86, 0.64, 0.22),
        "accent_emit": 0.65,
        "spot_color": (1.00, 0.90, 0.76),
    },
    {
        "act_idx": 1,
        "code": "ACT II",
        "title": "ACT II  ·  STRINGS & BRASS ROTUNDA",
        "z_top": 2.95,
        "z_bot": -0.25,
        "panel_color": (0.045, 0.010, 0.014),
        "accent_color": (0.94, 0.70, 0.26),
        "accent_emit": 0.75,
        "spot_color": (1.00, 0.86, 0.72),
    },
    {
        "act_idx": 2,
        "code": "ACT III",
        "title": "ACT III  ·  FOLKLORE PAVILION",
        "z_top": -0.25,
        "z_bot": -3.45,
        "panel_color": (0.010, 0.032, 0.030),
        "accent_color": (0.86, 0.44, 0.22),
        "accent_emit": 0.75,
        "spot_color": (0.96, 0.92, 0.84),
    },
    {
        "act_idx": 3,
        "code": "ACT IV",
        "title": "ACT IV  ·  NEON SYNTH VAULT",
        "z_top": -3.45,
        "z_bot": -7.10,
        "panel_color": (0.008, 0.010, 0.016),
        "accent_color": (0.14, 0.88, 0.98),
        "accent_emit": 2.40,
        "spot_color": (0.76, 0.92, 1.00),
    },
]

MARBLE_CONFIGS = {
    0: {"id": "emerald", "name": "EmeraldMarble", "mat_key": "LibMat_EmeraldMarble", "glow": (0.20, 1.00, 0.52)},
    1: {"id": "gold", "name": "GoldMarble", "mat_key": "LibMat_GoldMarble", "glow": (1.00, 0.68, 0.18)},
    2: {"id": "sapphire", "name": "SapphireMarble", "mat_key": "LibMat_SapphireMarble", "glow": (0.24, 0.68, 1.00)},
}


# ---------------------------------------------------------------------------
# Polyphonic 16:9 Showcase Score Generator (All 4 Acts, 6 Instrument Families)
# ---------------------------------------------------------------------------

def pitch_name_from_midi(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{(midi // 12) - 1}"


def build_default_16x9_showcase_score() -> dict:
    """Create a 12.2-second 2-marble polyphonic widescreen showcase across all 4 Acts & 6 families."""
    # Each tuple: (t_hit, marble_idx, act_idx, x, z, gm_program, midi, duration, velocity)
    events = [
        # --- ACT I: Conservatory Salon (Z in [3.1, 5.4]) ---
        (1.05, 0, 0, -1.75, 5.05, 0,   64, 0.38, 102),  # E4 Acoustic Grand Piano (Emerald Left)
        (1.42, 0, 0, -0.82, 4.45, 0,   67, 0.36, 108),  # G4 Acoustic Grand Piano (ballistic bounce)
        (1.65, 1, 0,  1.75, 4.85, 46,  72, 0.42, 98),   # C5 Orchestral Harp (Gold Right answer)
        (2.05, 1, 0,  0.85, 4.25, 10,  76, 0.40, 104),  # E5 Music Box (ballistic bounce)
        (2.42, 0, 0, -0.35, 3.55, 11,  71, 0.45, 110),  # B4 Vibraphone (center-left bridge)

        # --- ACT II: Strings & Brass Rotunda (Z in [-0.1, 2.3]) ---
        (3.40, 0, 1, -1.70, 2.05, 40,  69, 0.48, 112),  # A4 Violin (Emerald)
        (3.82, 0, 1, -0.80, 1.45, 21,  64, 0.45, 106),  # E4 Accordion (ballistic bounce)
        (4.15, 1, 1,  1.70, 1.90, 60,  60, 0.50, 114),  # C4 French Horn (Gold)
        (4.58, 1, 1,  0.78, 1.25, 57,  55, 0.48, 110),  # G3 Trombone (ballistic bounce)
        (5.05, 0, 1, -0.25, 0.45, 48,  72, 0.42, 108),  # C5 String Ensemble

        # --- ACT III: Folklore Pavilion (Z in [-3.1, -0.8]) ---
        (6.05, 0, 2, -1.75, -0.95, 104, 62, 0.42, 106), # D4 Sitar (Emerald)
        (6.48, 0, 2, -0.82, -1.55, 109, 67, 0.45, 112), # G4 Bagpipe (ballistic bounce)
        (6.80, 1, 2,  1.75, -1.10, 114, 74, 0.40, 110), # D5 Steel Drums (Gold)
        (7.24, 1, 2,  0.82, -1.75, 73,  79, 0.44, 104), # G5 Flute (ballistic bounce)
        (7.70, 0, 2, -0.30, -2.55, 12,  69, 0.38, 108), # A4 Marimba

        # --- ACT IV: Neon Synth Vault & Synchronized Finale (Z in [-6.5, -4.0]) ---
        (8.75, 0, 3, -1.65, -4.10, 80,  60, 0.42, 114), # C4 Square Lead (Emerald)
        (8.85, 1, 3,  1.65, -4.15, 81,  67, 0.42, 114), # G4 Sawtooth Lead (Gold)
        (9.22, 0, 3, -0.85, -4.85, 81,  64, 0.40, 116), # E4 Sawtooth Lead
        (9.32, 1, 3,  0.85, -4.90, 88,  72, 0.42, 115), # C5 New Age Pad
        (9.75, 0, 3, -0.68, -5.70, 55,  48, 0.85, 125), # C3 Orchestra Hit / Gong CLIMAX L
        (9.78, 1, 3,  0.68, -5.70, 55,  60, 0.85, 125), # C4 Orchestra Hit / Gong CLIMAX R
    ]
    notes = []
    for idx, (t_hit, m_idx, act_idx, x, z, gm, midi, dur, vel) in enumerate(events):
        notes.append({
            "index": idx,
            "t_hit": float(t_hit),
            "marble_idx": int(m_idx),
            "act_idx": int(act_idx),
            "target_x": float(x),
            "target_z": float(z),
            "gm_program": int(gm),
            "midi": int(midi),
            "duration": float(dur),
            "velocity": int(vel),
        })
    return {
        "title": "16:9 Polyphonic Concert Showcase",
        "pre_roll": 0.90,
        "post_roll": 1.60,
        "fps": FPS,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Physics Trajectory + Swept Hermite Chute Solver
# ---------------------------------------------------------------------------

def _hermite_eval(p0: Vector, m0: Vector, p1: Vector, m1: Vector, s: float) -> tuple[Vector, Vector]:
    s2 = s * s
    s3 = s2 * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    dh00 = 6 * s2 - 6 * s
    dh10 = 3 * s2 - 4 * s + 1
    dh01 = -6 * s2 + 6 * s
    dh11 = 3 * s2 - 2 * s
    tangent = dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1
    return pos, tangent


def _build_rail_geometry(
    rid: str,
    marble_idx: int,
    curve_pts: list[Vector],
    pipe_radius: float = RAIL_PIPE_R,
    half_gap: float = RAIL_HALF_GAP,
) -> dict:
    """Construct twin offset rail polylines & U-tie supports strictly below marble center."""
    drop = MARBLE_RADIUS * 0.72 + pipe_radius * 0.6
    front_pts: list[list[float]] = []
    back_pts: list[list[float]] = []
    tie_pts_list: list[list[list[float]]] = []

    for i, p in enumerate(curve_pts):
        base = Vector((p.x, p.y, p.z - drop))
        pf = [round(base.x, 5), round(base.y - half_gap, 5), round(base.z, 5)]
        pb = [round(base.x, 5), round(base.y + half_gap, 5), round(base.z, 5)]
        front_pts.append(pf)
        back_pts.append(pb)
        if i in (0, len(curve_pts) - 1) or (i % 7 == 0 and 2 < i < len(curve_pts) - 3):
            low_z = round(base.z - 0.036, 5)
            tie = [
                pf,
                [pf[0], round(base.y - half_gap * 0.45, 5), low_z],
                [pb[0], round(base.y + half_gap * 0.45, 5), low_z],
                pb,
            ]
            tie_pts_list.append(tie)

    return {
        "id": rid,
        "marble_idx": marble_idx,
        "pipe_radius": pipe_radius,
        "pipe_front": front_pts,
        "pipe_back": back_pts,
        "ties": tie_pts_list,
    }


def _solve_segment_ballistic(p0: Vector, p1: Vector, dt: float) -> tuple[Vector, Vector]:
    vx = (p1.x - p0.x) / dt
    vy = (p1.y - p0.y) / dt
    vz0 = (p1.z - p0.z + 0.5 * GRAVITY * dt * dt) / dt
    v0 = Vector((vx, vy, vz0))
    v1 = Vector((vx, vy, vz0 - GRAVITY * dt))
    return v0, v1


def _board_axes_from_normal(n: Vector) -> tuple[Vector, Vector, Vector]:
    n = n.normalized()
    v = Vector((0.0, 1.0, 0.0))
    u = v.cross(n)
    if u.length < 1e-5:
        u = Vector((1.0, 0.0, 0.0))
    else:
        u.normalize()
    v = n.cross(u).normalized()
    return u, v, n


def plan_16x9_polyphonic_show(score: dict, manifest: dict) -> dict:
    """Solve exact 60fps trajectories, twin-pipe guide chutes, and instrument placements for N marbles."""
    fps = int(score.get("fps", FPS))
    notes = sorted(score["notes"], key=lambda n: (float(n["t_hit"]), int(n["marble_idx"])))
    max_t = max(float(n["t_hit"]) + float(n.get("duration", 0.35)) for n in notes) + float(score.get("post_roll", 1.5))
    total_frames = int(math.ceil(max_t * fps))
    by_prog = manifest.get("by_program", {})

    # Partition notes per marble track
    notes_by_marble: dict[int, list[dict]] = {}
    for ev in notes:
        notes_by_marble.setdefault(int(ev["marble_idx"]), []).append(ev)

    boards: list[dict] = []
    rails: list[dict] = []
    marble_samples: dict[int, list[dict]] = {}

    for m_idx, m_notes in sorted(notes_by_marble.items()):
        # Contact & sphere center positions (board normal pushes sphere out by R + 0.5*T)
        # First pass: compute velocities between hits (ballistic for short hops, gentle ramp exit/entry for chutes)
        n_count = len(m_notes)
        seg_specs: list[dict] = []
        centers = [Vector((ev["target_x"], 0.0, ev["target_z"])) for ev in m_notes]

        for k in range(n_count - 1):
            dt = float(m_notes[k + 1]["t_hit"]) - float(m_notes[k]["t_hit"])
            p0, p1 = centers[k], centers[k + 1]
            if dt <= BALLISTIC_MAX_DT:
                v_out, v_in = _solve_segment_ballistic(p0, p1, dt)
                seg_specs.append({"kind": "ballistic", "dt": dt, "v0": v_out, "v1": v_in, "p0": p0, "p1": p1})
            else:
                dx = p1.x - p0.x
                dz = p1.z - p0.z
                chord_len = max(0.5, math.hypot(dx, dz))
                tx = dx / chord_len
                m0 = Vector((tx * chord_len * 0.72, 0.0, -0.32 * chord_len))
                m1 = Vector((tx * chord_len * 0.72, 0.0, -0.45 * chord_len))
                seg_specs.append({"kind": "chute", "dt": dt, "m0": m0, "m1": m1, "p0": p0, "p1": p1})

        # Determine realistic upward-facing board normals from incoming/outgoing velocities
        v_in_hits: list[Vector] = []
        v_out_hits: list[Vector] = []
        for k in range(n_count):
            if k == 0:
                p0 = centers[0]
                vin = Vector((-math.copysign(1.2, p0.x) if abs(p0.x) > 0.2 else 1.2, 0.0, -2.4))
            elif seg_specs[k - 1]["kind"] == "ballistic":
                vin = seg_specs[k - 1]["v1"]
            else:
                _, tang = _hermite_eval(seg_specs[k - 1]["p0"], seg_specs[k - 1]["m0"], seg_specs[k - 1]["p1"], seg_specs[k - 1]["m1"], 1.0)
                vin = tang / max(0.35, seg_specs[k - 1]["dt"])

            if k == n_count - 1:
                vout = Vector((math.copysign(1.4, -centers[k].x) if abs(centers[k].x) > 0.2 else 1.4, 0.0, -1.6))
            elif seg_specs[k]["kind"] == "ballistic":
                vout = seg_specs[k]["v0"]
            else:
                _, tang = _hermite_eval(seg_specs[k]["p0"], seg_specs[k]["m0"], seg_specs[k]["p1"], seg_specs[k]["m1"], 0.0)
                vout = tang / max(0.35, seg_specs[k]["dt"])

            v_in_hits.append(vin)
            v_out_hits.append(vout)

        # Build frame-by-frame samples `0..total_frames` for this marble
        t_first = float(m_notes[0]["t_hit"])
        entry_p0 = Vector((centers[0].x - (0.85 if centers[0].x >= 0 else -0.85), 0.0, centers[0].z + 0.62))
        entry_m0 = Vector((math.copysign(0.55, centers[0].x - entry_p0.x), 0.0, -0.25))
        entry_m1 = Vector((math.copysign(0.85, centers[0].x - entry_p0.x), 0.0, -0.55))

        t_last = float(m_notes[-1]["t_hit"])
        exit_p1 = Vector((centers[-1].x * 1.4, 0.0, centers[-1].z - 0.85))
        exit_m0 = Vector((v_out_hits[-1].x * 0.45, 0.0, -0.45))
        exit_m1 = Vector((v_out_hits[-1].x * 0.55, 0.0, -0.95))

        # Construct rails for entry, chutes, and exit
        entry_pts = [_hermite_eval(entry_p0, entry_m0, centers[0], entry_m1, i / 18.0)[0] for i in range(19)]
        rails.append(_build_rail_geometry(f"Rail_M{m_idx}_Entry", m_idx, entry_pts))

        for k, sp in enumerate(seg_specs):
            if sp["kind"] == "chute":
                pts = [_hermite_eval(sp["p0"], sp["m0"], sp["p1"], sp["m1"], i / 26.0)[0] for i in range(27)]
                rails.append(_build_rail_geometry(f"Rail_M{m_idx}_Chute_{k:02d}", m_idx, pts))

        exit_pts = [_hermite_eval(centers[-1], exit_m0, exit_p1, exit_m1, i / 18.0)[0] for i in range(19)]
        rails.append(_build_rail_geometry(f"Rail_M{m_idx}_Exit", m_idx, exit_pts))

        samples: list[dict] = []
        for f in range(total_frames + 1):
            t = f / float(fps)
            if t <= t_first:
                s = max(0.0, min(1.0, t / max(0.05, t_first)))
                # Smooth ease-in roll from rest at entry gate into note 0
                s_ease = s * s * (2.0 - s)
                pos, _ = _hermite_eval(entry_p0, entry_m0, centers[0], entry_m1, s_ease)
            elif t >= t_last:
                dt_exit = max(0.4, max_t - t_last)
                s = max(0.0, min(1.0, (t - t_last) / dt_exit))
                pos, _ = _hermite_eval(centers[-1], exit_m0, exit_p1, exit_m1, s)
            else:
                # Find segment containing `t`
                pos = centers[-1]
                for k in range(n_count - 1):
                    ta = float(m_notes[k]["t_hit"])
                    tb = float(m_notes[k + 1]["t_hit"])
                    if ta <= t <= tb:
                        sp = seg_specs[k]
                        tau = t - ta
                        if sp["kind"] == "ballistic":
                            p0, v0 = sp["p0"], sp["v0"]
                            pos = Vector((
                                p0.x + v0.x * tau,
                                p0.y + v0.y * tau,
                                p0.z + v0.z * tau - 0.5 * GRAVITY * tau * tau,
                            ))
                        else:
                            s = tau / max(1e-4, sp["dt"])
                            pos, _ = _hermite_eval(sp["p0"], sp["m0"], sp["p1"], sp["m1"], s)
                        break
            samples.append({"frame": f, "t": round(t, 5), "position": [round(pos.x, 5), round(pos.y, 5), round(pos.z, 5)]})
        marble_samples[m_idx] = samples

        # Build board descriptors cleanly underneath marble strike point
        for k, ev in enumerate(m_notes):
            vin_u = v_in_hits[k].normalized()
            vout_u = v_out_hits[k].normalized()
            bis = (vout_u - vin_u)
            if bis.length < 1e-4 or bis.z < 0.15:
                bis = Vector((0.18 * (-1.0 if centers[k].x > 0 else 1.0), 0.0, 1.0))
            bis.y = 0.0
            bis.normalize()
            if bis.z < 0.65:
                bis = Vector((bis.x * 0.65, 0.0, 0.92)).normalized()

            u, v, n = _board_axes_from_normal(bis)
            contact = centers[k] - n * MARBLE_RADIUS
            board_center = contact - n * (BOARD_THICKNESS * 0.5)
            gm = int(ev["gm_program"])
            m_info = by_prog.get(str(gm), {})
            inst_name = m_info.get("name", f"GM_{gm:03d}")
            length = round(max(0.84, min(1.16, 0.88 + 0.16 * float(ev["duration"]))), 4)
            pin_a = board_center + Vector((0.0, BOARD_WIDTH_Y * 0.5, 0.0))

            boards.append({
                "index": int(ev["index"]),
                "marble_idx": m_idx,
                "act_idx": int(ev["act_idx"]),
                "t_hit": float(ev["t_hit"]),
                "frame_hit": int(round(float(ev["t_hit"]) * fps)),
                "gm_program": gm,
                "track_idx": m_idx + 1,
                "instrument": inst_name,
                "note": int(ev["midi"]),
                "pitch_name": pitch_name_from_midi(int(ev["midi"])),
                "duration": float(ev["duration"]),
                "velocity": int(ev["velocity"]),
                "length": length,
                "width_y": BOARD_WIDTH_Y,
                "thickness": BOARD_THICKNESS,
                "center": [round(board_center.x, 5), round(board_center.y, 5), round(board_center.z, 5)],
                "contact": [round(contact.x, 5), round(contact.y, 5), round(contact.z, 5)],
                "axis_u": [round(u.x, 5), round(u.y, 5), round(u.z, 5)],
                "axis_v": [round(v.x, 5), round(v.y, 5), round(v.z, 5)],
                "axis_n": [round(n.x, 5), round(n.y, 5), round(n.z, 5)],
                "pins": [{"a": [round(pin_a.x, 5), round(pin_a.y, 5), round(pin_a.z, 5)], "radius": 0.015}],
            })

    boards.sort(key=lambda b: (b["frame_hit"], b["index"]))
    return {
        "schema": "marble_16x9_concert_v1",
        "fps": fps,
        "total_duration": round(max_t, 4),
        "total_frames": total_frames,
        "marble_radius": MARBLE_RADIUS,
        "back_wall_y": WALL_Y,
        "marbles": {str(k): v for k, v in marble_samples.items()},
        "boards": boards,
        "rails": rails,
        "validation": {"ok": True},
    }


# ---------------------------------------------------------------------------
# Widescreen Themed Acoustic Stage Architecture
# ---------------------------------------------------------------------------

def _make_chamber_material(name: str, color, roughness=0.92, metallic=0.0, emit_strength=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    set_socket(bsdf, "Base Color", (*color[:3], 1.0))
    set_socket(bsdf, "Metallic", metallic)
    set_socket(bsdf, "Roughness", roughness)
    if emit_strength > 0.0:
        set_socket(bsdf, "Emission Color", (*color[:3], 1.0))
        set_socket(bsdf, "Emission Strength", emit_strength)
    return mat


def _create_3d_text(name: str, body: str, loc, size: float, extrude: float, bevel: float, mat, col, align_x="CENTER"):
    curve = bpy.data.curves.new(name=f"{name}_Curve", type="FONT")
    curve.body = body
    curve.align_x = align_x
    curve.align_y = "CENTER"
    curve.size = size
    curve.extrude = extrude
    curve.bevel_depth = bevel
    curve.bevel_resolution = 3
    curve.space_character = 1.06
    obj = bpy.data.objects.new(name, curve)
    obj.location = loc
    obj.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
    if mat is not None:
        obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def _create_box_direct(name: str, dims, loc, mat, col, bevel=0.006):
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=dims, verts=bm.verts)
    if bevel > 0.0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=2, affect="EDGES", clamp_overlap=True)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = loc
    if mat is not None:
        obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def create_16x9_concert_stage(lib_mats: dict, col_stage: bpy.types.Collection, boards: list[dict]):
    """Build the 4 themed widescreen acoustic Act Chambers + grand channel pediment."""
    wood_mat = lib_mats["LibMat_WoodSlatDark"]
    brass_mat = lib_mats["LibMat_WallBrassTrim"]
    bronze_mat = lib_mats["LibMat_HardwareDark"]
    stage_w = 12.8

    for act in ACT_THEMES:
        idx = act["act_idx"]
        z_top = act["z_top"]
        z_bot = act["z_bot"]
        h = z_top - z_bot
        z_mid = 0.5 * (z_top + z_bot)

        p_mat = _make_chamber_material(f"ActPanelMat_{idx}", act["panel_color"], roughness=0.90)
        a_mat = _make_chamber_material(
            f"ActAccentMat_{idx}",
            act["accent_color"],
            roughness=0.22 if idx < 3 else 0.12,
            metallic=0.88 if idx < 3 else 0.15,
            emit_strength=act["accent_emit"],
        )

        # Back acoustic panel
        _create_box_direct(f"ActBackdrop_{idx}", (stage_w, 0.08, h - 0.04), (0.0, WALL_Y + 0.05, z_mid), p_mat, col_stage, bevel=0.0)

        # Sculpted acoustic architectural columns & central twin spine framing
        for sx in (-3.15, -2.55, 2.55, 3.15):
            _create_box_direct(f"ActPilaster_{idx}_{sx:.1f}", (0.20, 0.16, h), (sx, WALL_Y - 0.01, z_mid), wood_mat, col_stage, bevel=0.008)
            _create_box_direct(f"ActTrim_{idx}_{sx:.1f}", (0.024, 0.032, h - 0.14), (sx - 0.09 if sx > 0 else sx + 0.09, WALL_Y - 0.08, z_mid), a_mat, col_stage, bevel=0.003)

        # Subtle center acoustic slat flutes (adds architectural texture behind central space without glare)
        for cx in (-0.36, -0.12, 0.12, 0.36):
            _create_box_direct(f"ActCenterFlute_{idx}_{cx:.2f}", (0.055, 0.025, h - 0.42), (cx, WALL_Y + 0.005, z_mid - 0.08), wood_mat, col_stage, bevel=0.004)

        # Horizontal Act Cornice Divider + Centered Extruded 3D Medallion Header
        _create_box_direct(f"ActCornice_{idx}", (stage_w * 0.72, 0.06, 0.045), (0.0, WALL_Y - 0.03, z_top - 0.03), brass_mat, col_stage, bevel=0.004)
        _create_box_direct(f"ActGlowStrip_{idx}", (stage_w * 0.58, 0.025, 0.012), (0.0, WALL_Y - 0.06, z_top - 0.082), a_mat, col_stage, bevel=0.002)
        _create_box_direct(f"ActPlaqueBacking_{idx}", (2.35, 0.04, 0.24), (0.0, WALL_Y - 0.025, z_top - 0.22), wood_mat, col_stage, bevel=0.006)
        _create_3d_text(f"ActBanner3D_{idx}", act["title"], (0.0, WALL_Y - 0.055, z_top - 0.22), 0.102, 0.014, 0.002, brass_mat if idx < 3 else a_mat, col_stage, align_x="CENTER")

    # Grand Opening Widescreen Concert Pediment above Act I
    ped_z = ACT_THEMES[0]["z_top"] + 0.42
    _create_box_direct("GrandPedimentBacking", (4.80, 0.09, 0.68), (0.0, WALL_Y - 0.01, ped_z), wood_mat, col_stage, bevel=0.012)
    _create_box_direct("GrandPedimentRimTop", (4.86, 0.11, 0.035), (0.0, WALL_Y - 0.03, ped_z + 0.34), brass_mat, col_stage, bevel=0.005)
    _create_3d_text("GrandBrandTitle3D", "LUCA MARBLE MUSIC", (0.0, WALL_Y - 0.065, ped_z + 0.10), 0.22, 0.024, 0.004, brass_mat, col_stage)
    _create_3d_text("GrandBrandSub3D", "POLYPHONIC WIDESCREEN CONCERT THEATRE", (0.0, WALL_Y - 0.055, ped_z - 0.14), 0.082, 0.010, 0.002, bronze_mat, col_stage)

    # Rear spine mounting arms connecting into outer vertical rails or wall anchors
    for b in boards:
        idx_b = int(b["index"])
        p_a = b["pins"][0]["a"]
        wall_anchor = (p_a[0], WALL_Y - 0.02, p_a[2])
        create_cylinder_between(f"WidescreenPin_{idx_b:03d}", p_a, wall_anchor, 0.014, bronze_mat, col_stage, vertices=20)
        create_wall_flange(f"WidescreenFlange_{idx_b:03d}", (p_a[0], WALL_Y, p_a[2]), 0.034, brass_mat, col_stage)


# ---------------------------------------------------------------------------
# 3-Mode 16:9 Cinema Camera Choreography
# ---------------------------------------------------------------------------

def _smoothstep(a: float, b: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - a) / max(1e-5, b - a)))
    return t * t * (3.0 - 2.0 * t)


def plan_16x9_cinema_camera(plan: dict) -> tuple[list[Vector], list[Vector], list[float]]:
    """Compute smooth 16:9 camera locations, look-at targets, and focal lengths across Acts & Climax."""
    fps = int(plan["fps"])
    total_frames = int(plan["total_frames"])
    boards = plan["boards"]
    marbles = plan["marbles"]

    last_t = max(float(b["t_hit"]) for b in boards)
    climax_t = last_t - 0.85

    raw_cam: list[Vector] = []
    raw_tgt: list[Vector] = []
    raw_lens: list[float] = []

    for f in range(total_frames + 1):
        t = f / float(fps)

        # Find upcoming hit board for lead framing bias
        upcoming = next((b for b in boards if float(b["t_hit"]) >= t - 0.08), boards[-1])
        act_m_idx = int(upcoming["marble_idx"])

        # Centroid + active marble bias
        m_positions: dict[int, Vector] = {}
        for m_k, samps in marbles.items():
            m_positions[int(m_k)] = Vector(samps[min(f, len(samps) - 1)]["position"])

        active_pos = m_positions.get(act_m_idx, Vector((0.0, 0.0, 0.0)))
        mean_pos = sum(m_positions.values(), Vector((0.0, 0.0, 0.0))) / max(1, len(m_positions))
        blend_pos = 0.55 * active_pos + 0.45 * mean_pos

        # Blend wide opening & wide finale modes smoothly
        w_open = 1.0 - _smoothstep(0.35, 1.55, t)
        w_climax = _smoothstep(climax_t - 0.45, climax_t + 0.35, t)
        w_wide = max(w_open, w_climax)

        # Keep center stage composition strong so BOTH tracks & centered Act header stay readable
        tgt_x = (1.0 - w_wide) * (0.28 * blend_pos.x)
        tgt_z = blend_pos.z + 0.30 * w_open - 0.10 * (1.0 - w_open)
        raw_tgt.append(Vector((tgt_x, 0.12, tgt_z)))

        # Widescreen 3/4 parallax: subtle camera X drift + wide cinema distance
        cam_x = (1.0 - w_wide) * (0.22 * blend_pos.x) + 0.35 * math.sin(t * 0.42)
        cam_y = -4.95 - 1.15 * w_wide
        cam_z = tgt_z + 0.34 + 0.25 * w_open
        lens = 38.0 - 5.0 * w_wide

        # Micro-punch impulse on heavy impacts (vel >= 112)
        shake = Vector((0.0, 0.0, 0.0))
        for b in boards:
            dt_h = t - float(b["t_hit"])
            if 0.0 <= dt_h <= 0.18 and float(b.get("velocity", 100)) >= 112.0:
                amp = 0.018 * math.exp(-dt_h * 18.0) * math.sin(dt_h * 90.0)
                shake += Vector((0.4 * amp, 0.0, amp))

        raw_cam.append(Vector((cam_x, cam_y, cam_z)) + shake)
        raw_lens.append(lens)

    # Low-pass smooth (moving Gaussian-like window) so sweeps feel Steadicam-fluid (keep micro-impulse lightly)
    half_win = 14
    sm_cam: list[Vector] = []
    sm_tgt: list[Vector] = []
    sm_lens: list[float] = []
    for f in range(total_frames + 1):
        i0 = max(0, f - half_win)
        i1 = min(total_frames, f + half_win)
        cnt = float(i1 - i0 + 1)
        c_avg = sum(raw_cam[i0 : i1 + 1], Vector((0.0, 0.0, 0.0))) / cnt
        t_avg = sum(raw_tgt[i0 : i1 + 1], Vector((0.0, 0.0, 0.0))) / cnt
        l_avg = sum(raw_lens[i0 : i1 + 1]) / cnt
        sm_cam.append(c_avg)
        sm_tgt.append(t_avg)
        sm_lens.append(l_avg)

    return sm_cam, sm_tgt, sm_lens


# ---------------------------------------------------------------------------
# Blender 16:9 Scene Assembler
# ---------------------------------------------------------------------------

def build_16x9_scene(
    plan: dict,
    out_blend: str | None = None,
    out_hud_json: str | None = None,
    audio_path: str | None = None,
    lib_blend: str = DEFAULT_LIB_BLEND,
    manifest_json: str = DEFAULT_MANIFEST_JSON,
) -> None:
    clear_scene()
    scene = bpy.context.scene
    root = scene.collection
    templates, lib_mats, manifest = load_instrument_library(lib_blend, manifest_json)

    fps = int(plan["fps"])
    total_frames = int(plan["total_frames"])
    radius = float(plan["marble_radius"])
    boards = plan["boards"]
    rails = plan["rails"]
    marbles = plan["marbles"]

    scene.render.engine = get_eevee_engine()
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    scene.frame_start = 0
    scene.frame_end = total_frames
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 28
        if hasattr(scene.eevee, "use_shadows"):
            scene.eevee.use_shadows = True
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = True
        if hasattr(scene.eevee, "gi_cubemap_resolution"):
            scene.eevee.gi_cubemap_resolution = "2048"

    scene.view_settings.view_transform = "AgX" if "AgX" in [
        i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items
    ] else "Filmic"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0

    build_studio_reflection_world(scene)
    setup_bloom_compositor(scene)

    col_marbles = get_or_create_collection("01_Marbles", root)
    col_instruments = get_or_create_collection("02_Instruments", root)
    col_rails = get_or_create_collection("03_GuideRails", root)
    col_stage = get_or_create_collection("04_ConcertStage", root)
    col_fx = get_or_create_collection("05_ImpactFX", root)

    # 1. Concert Theatre Stage & Act Chambers
    create_16x9_concert_stage(lib_mats, col_stage, boards)

    # 2. Spawn all v3.0 Sculpted Instruments + Per-Part Hit Feedback
    ring_mesh = create_shockwave_ring_mesh()
    ring_mesh.materials.append(None)
    act_cols: dict[int, bpy.types.Collection] = {}
    for b in boards:
        act_i = int(b.get("act_idx", 0))
        if act_i not in act_cols:
            act_cols[act_i] = get_or_create_collection(f"Act_{act_i + 1:02d}_Instruments", col_instruments)
        spawn_instrument_board(b, templates, lib_mats, manifest, act_cols[act_i], col_fx, ring_mesh, [], fps)

    # 3. Twin-Pipe Swept Guide Chutes
    rail_mat = lib_mats.get("LibMat_ChromeRail", lib_mats["LibMat_WallBrassTrim"])
    dark_mat = lib_mats["LibMat_HardwareDark"]
    for r in rails:
        rid = r["id"]
        create_pipe_polyline(f"{rid}_front", r["pipe_front"], r["pipe_radius"], rail_mat, col_rails)
        create_pipe_polyline(f"{rid}_back", r["pipe_back"], r["pipe_radius"], rail_mat, col_rails)
        for ti, tie_pts in enumerate(r.get("ties", [])):
            create_pipe_polyline(f"{rid}_tie_{ti:02d}", tie_pts, 0.009, dark_mat, col_rails)
        pts_back = r["pipe_back"]
        for frac in (0.18, 0.50, 0.82):
            pk = pts_back[min(len(pts_back) - 1, max(0, int(round(frac * (len(pts_back) - 1)))))]
            create_cylinder_between(f"{rid}_pin_{int(frac*100)}", (pk[0], pk[1] + 0.01, pk[2]), (pk[0], WALL_Y, pk[2]), 0.011, dark_mat, col_rails, vertices=18)
            create_wall_flange(f"{rid}_flange_{int(frac*100)}", (pk[0], WALL_Y, pk[2]), 0.026, dark_mat, col_rails)

    # 4. Color-Coded Polyphonic Marbles + Velocity/Impact Glow Lights
    marble_mesh = bpy.data.meshes.new("WidescreenMarbleMesh")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=72, v_segments=44, radius=radius)
    bm.to_mesh(marble_mesh)
    bm.free()
    for poly in marble_mesh.polygons:
        poly.use_smooth = True

    marble_objs: list[bpy.types.Object] = []
    for m_k, samps in sorted(marbles.items(), key=lambda kv: int(kv[0])):
        m_idx = int(m_k)
        cfg = MARBLE_CONFIGS.get(m_idx, MARBLE_CONFIGS[0])
        m_mesh = marble_mesh.copy()
        m_mesh.name = f"{cfg['name']}Mesh"
        mobj = bpy.data.objects.new(cfg["name"], m_mesh)
        col_marbles.objects.link(mobj)
        mmat = lib_mats.get(cfg["mat_key"], lib_mats["LibMat_EmeraldMarble"])
        mobj.data.materials.clear()
        mobj.data.materials.append(mmat)
        mobj.rotation_mode = "QUATERNION"
        marble_objs.append(mobj)

        ml_data = bpy.data.lights.new(f"{cfg['name']}_Glow", type="POINT")
        ml_data.energy = 2.0
        ml_data.color = cfg["glow"]
        ml_data.shadow_soft_size = 0.09
        ml_obj = bpy.data.objects.new(f"{cfg['name']}_Glow", ml_data)
        col_marbles.objects.link(ml_obj)
        ml_obj.parent = mobj

        ml_data.keyframe_insert(data_path="energy", frame=0)
        for b in boards:
            if int(b["marble_idx"]) == m_idx:
                fh = int(b["frame_hit"])
                ml_data.energy = 2.0
                ml_data.keyframe_insert(data_path="energy", frame=max(0, fh - 1))
                ml_data.energy = 6.2
                ml_data.keyframe_insert(data_path="energy", frame=fh)
                ml_data.energy = 2.0
                ml_data.keyframe_insert(data_path="energy", frame=fh + 5)

        spin_q = Quaternion((1.0, 0.0, 0.0, 0.0))
        for i_s, s in enumerate(samps):
            f = int(s["frame"])
            pos = Vector(s["position"])
            mobj.location = pos
            mobj.keyframe_insert(data_path="location", frame=f)
            if i_s > 0:
                dp = pos - Vector(samps[i_s - 1]["position"])
                dist = dp.length
                if dist > 1e-6:
                    axis = Vector((0.0, 0.0, 1.0)).cross(dp)
                    axis = axis.normalized() if axis.length > 1e-5 else Vector((0.0, 1.0, 0.0))
                    spin_q = (Quaternion(axis, dist / radius) @ spin_q).normalized()
            mobj.rotation_quaternion = spin_q
            mobj.keyframe_insert(data_path="rotation_quaternion", frame=f)

    # 5. Tracking 16:9 Cinema Lighting Rig + 3-Mode Cinema Camera
    target_empty = bpy.data.objects.new("CinemaCamTarget", None)
    col_stage.objects.link(target_empty)

    key_spot = bpy.data.lights.new("StageKeySpot", type="SPOT")
    key_spot.energy = 1450.0
    key_spot.color = (1.0, 0.91, 0.80)
    key_spot.spot_size = math.radians(74.0)
    key_spot.spot_blend = 0.62
    key_spot.shadow_soft_size = 0.35
    key_obj = bpy.data.objects.new("StageKeySpot", key_spot)
    col_stage.objects.link(key_obj)
    key_obj.parent = target_empty
    key_obj.location = (-2.6, -4.4, 2.8)
    k_cons = key_obj.constraints.new(type="TRACK_TO")
    k_cons.target = target_empty
    k_cons.track_axis = "TRACK_NEGATIVE_Z"
    k_cons.up_axis = "UP_Y"

    rim_area = bpy.data.lights.new("StageCoolRim", type="AREA")
    rim_area.energy = 420.0
    rim_area.size = 1.6
    rim_area.color = (0.62, 0.85, 1.0)
    rim_obj = bpy.data.objects.new("StageCoolRim", rim_area)
    col_stage.objects.link(rim_obj)
    rim_obj.parent = target_empty
    rim_obj.location = (3.8, -1.6, 1.4)
    r_cons = rim_obj.constraints.new(type="TRACK_TO")
    r_cons.target = target_empty
    r_cons.track_axis = "TRACK_NEGATIVE_Z"
    r_cons.up_axis = "UP_Y"

    # Wide subtle fill area so left/right Act pilasters & instrument wood grain read cleanly
    fill_area = bpy.data.lights.new("WideSoftFill", type="AREA")
    fill_area.energy = 180.0
    fill_area.size = 4.2
    fill_area.color = (0.94, 0.95, 1.0)
    fill_obj = bpy.data.objects.new("WideSoftFill", fill_area)
    col_stage.objects.link(fill_obj)
    fill_obj.parent = target_empty
    fill_obj.location = (0.0, -5.2, 0.4)

    cam_data = bpy.data.cameras.new("CinemaCamera_16x9")
    cam_data.sensor_width = 36.0
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100.0
    if marble_objs:
        cam_data.dof.use_dof = True
        cam_data.dof.focus_object = marble_objs[0]
        cam_data.dof.aperture_fstop = 5.6
    cam_obj = bpy.data.objects.new("CinemaCamera_16x9", cam_data)
    col_stage.objects.link(cam_obj)
    scene.camera = cam_obj

    c_cons = cam_obj.constraints.new(type="TRACK_TO")
    c_cons.target = target_empty
    c_cons.track_axis = "TRACK_NEGATIVE_Z"
    c_cons.up_axis = "UP_Y"

    cam_locs, tgt_locs, cam_lens = plan_16x9_cinema_camera(plan)
    for f in range(total_frames + 1):
        cam_obj.location = cam_locs[f]
        cam_obj.keyframe_insert(data_path="location", frame=f)
        target_empty.location = tgt_locs[f]
        target_empty.keyframe_insert(data_path="location", frame=f)
        cam_data.lens = cam_lens[f]
        cam_data.keyframe_insert(data_path="lens", frame=f)

    if audio_path and os.path.exists(audio_path):
        scene.sequence_editor_create()
        seq = scene.sequence_editor
        strips = getattr(seq, "strips", None) or getattr(seq, "sequences", None)
        if strips is not None:
            strips.new_sound("ConcertAudio", os.path.abspath(audio_path), channel=1, frame_start=0)
        scene.use_audio = True
        scene.sync_mode = "AUDIO_SYNC"

    if out_hud_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_hud_json)), exist_ok=True)
        hud_payload = {
            "fps": fps,
            "total_frames": total_frames,
            "acts": [{"act_idx": a["act_idx"], "code": a["code"], "title": a["title"]} for a in ACT_THEMES],
            "events": [
                {
                    "frame_hit": b["frame_hit"],
                    "t_hit": b["t_hit"],
                    "marble_idx": b["marble_idx"],
                    "act_idx": b["act_idx"],
                    "instrument": b["instrument"],
                    "pitch_name": b["pitch_name"],
                    "velocity": b["velocity"],
                }
                for b in boards
            ],
        }
        with open(out_hud_json, "w", encoding="utf-8") as f:
            json.dump(hud_payload, f, indent=2, ensure_ascii=False)

    if out_blend:
        os.makedirs(os.path.dirname(os.path.abspath(out_blend)), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend))
        print(f"Saved 16:9 Concert Theatre scene: {out_blend} ({total_frames} frames, {len(boards)} instruments, {len(marbles)} marbles)")


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    score_json = None
    if "--score" in args:
        si = args.index("--score")
        score_json = args[si + 1]
        args = args[:si] + args[si + 2 :]

    out_plan = args[0] if len(args) >= 1 else "projects/jingle-bells-marble-music/artifacts/template_16x9_showcase_physics.json"
    out_blend = args[1] if len(args) >= 2 else "projects/jingle-bells-marble-music/renders/template_16x9_showcase.blend"
    out_hud = args[2] if len(args) >= 3 else "projects/jingle-bells-marble-music/artifacts/template_16x9_showcase_hud.json"
    audio_path = args[3] if len(args) >= 4 else None

    if score_json and os.path.exists(score_json):
        with open(score_json, "r", encoding="utf-8") as f:
            score = json.load(f)
    else:
        score = build_default_16x9_showcase_score()

    with open(DEFAULT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    plan = plan_16x9_polyphonic_show(score, manifest)
    os.makedirs(os.path.dirname(os.path.abspath(out_plan)), exist_ok=True)
    with open(out_plan, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    build_16x9_scene(plan, out_blend=out_blend, out_hud_json=out_hud, audio_path=audio_path)


if __name__ == "__main__":
    main()
