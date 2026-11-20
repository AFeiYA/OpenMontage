#!/usr/bin/env python3
"""16:9 Multi-Ball Parallel-Lane Physics Trajectory Solver.

Solves 3 independent, collision-free trajectories in Left / Center / Right lanes
with a shared Z(t) descent anchor so all 3 balls stay framed together in 16:9.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import plan_physics_trajectory as phys

FPS = phys.FPS
PRE_ROLL = phys.PRE_ROLL
POST_ROLL = phys.POST_ROLL
GRAVITY = phys.GRAVITY
MARBLE_RADIUS = phys.MARBLE_RADIUS

LANE_CENTERS_X = [0.00, 0.00, 0.00]
LANE_LAYERS_Y = [-0.65, 0.00, 0.65]
LANE_HALF_X = 1.08
BACK_WALL_Y_MB = 1.08
Z_REF_RATE = -1.85  # Crisp vertical drop rate (~0.44m per 8th note) keeping all 5 balls tightly synced & separated

BALL_PALETTES = [
    {"name": "EmeraldLead", "color": [0.12, 0.92, 0.48], "glow": [0.18, 1.00, 0.55]},
    {"name": "AmberAlt", "color": [1.00, 0.62, 0.14], "glow": [1.00, 0.68, 0.20]},
    {"name": "CyanHarmony", "color": [0.18, 0.78, 1.00], "glow": [0.24, 0.85, 1.00]},
]

DRUM_HALF_X = 0.25
DRUM_LANES = [
    {
        "drum_idx": 0,
        "key": "Drum_Left_Kick",
        "name": "SmokedRubyKick",
        "side_sign": -1.0,
        "lane_x": -1.76,
        "layer_y": -0.15,
        "radius": 0.105,
        "color": [0.95, 0.14, 0.08],
        "glow": [1.00, 0.28, 0.08],
    },
    {
        "drum_idx": 1,
        "key": "Drum_Right_SnareHat",
        "name": "ChampagneSilverSnare",
        "side_sign": 1.0,
        "lane_x": 1.76,
        "layer_y": -0.15,
        "radius": 0.092,
        "color": [0.96, 0.90, 0.72],
        "glow": [0.22, 1.00, 0.48],
    },
]


def z_ref(t_after_preroll: float) -> float:
    return Z_REF_RATE * max(0.0, t_after_preroll)


def solve_center_entry_ramp(hit0_pos, v_in0) -> dict[str, Any]:
    """Entry starter ramp constrained strictly inside center stage so it clears Bass (-1.36) & Left Drum (-1.78)."""
    dt_drop0 = 0.10
    dt_rail0 = PRE_ROLL - dt_drop0

    v_lip = (v_in0[0], 0.0, v_in0[2] + GRAVITY * dt_drop0)
    p_lip = (
        hit0_pos[0] - v_lip[0] * dt_drop0,
        0.0,
        hit0_pos[2] - (v_lip[2] * dt_drop0 - 0.5 * GRAVITY * dt_drop0 * dt_drop0),
    )
    T_lip = phys.vnorm(v_lip)
    target_v_lip = phys.vlen(v_lip)

    v0_start = 0.35
    best = None
    best_err = 1e9
    for x_start in (-1.18, -1.12, -1.06, -1.00):
        dx = p_lip[0] - x_start
        if dx < 0.35:
            continue
        T_start = phys.vnorm((dx, 0.0, -0.24 * dx))
        for dz_start in (0.18, 0.26, 0.34, 0.42, 0.52):
            p_start = (x_start, 0.0, p_lip[2] + dz_start)
            p_mid = (0.5 * (x_start + p_lip[0]), 0.0, p_lip[2] + 0.54 * dz_start)
            T_mid = phys.vnorm((dx, 0.0, -dz_start))
            pts = phys.hermite_piecewise([(p_start, T_start), (p_mid, T_mid), (p_lip, T_lip)], samples_per_seg=44)
            sim = phys.simulate_rolling_pts(pts, v0_start)
            if sim is None or sim["max_slope_deg"] > 42.0:
                continue
            err = abs(sim["T"] - dt_rail0) * 6.0 + abs(sim["v_exit"] - target_v_lip)
            if err < best_err:
                best_err = err
                best = (sim, p_start, T_start, dz_start)

    assert best is not None, "Center entry ramp solver found no candidate"
    sim0, _, _, _ = best
    pts = sim0["pts"]
    lo_v, hi_v = 0.12, 1.80
    for _ in range(24):
        mid_v = 0.5 * (lo_v + hi_v)
        sm = phys.simulate_rolling_pts(pts, mid_v)
        if sm is None or sm["T"] > dt_rail0:
            lo_v = mid_v
        else:
            hi_v = mid_v
    sim_final = phys.simulate_rolling_pts(pts, 0.5 * (lo_v + hi_v))
    v_lip_actual = phys.vscale(T_lip, sim_final["v_exit"])
    dt_ff_actual = PRE_ROLL - sim_final["T"]
    hit0_exact = (
        p_lip[0] + v_lip_actual[0] * dt_ff_actual,
        0.0,
        p_lip[2] + v_lip_actual[2] * dt_ff_actual - 0.5 * GRAVITY * dt_ff_actual * dt_ff_actual,
    )
    v_in0_exact = (v_lip_actual[0], 0.0, v_lip_actual[2] - GRAVITY * dt_ff_actual)
    return {
        "sim": sim_final,
        "t0": 0.0,
        "t1": sim_final["T"],
        "p_lip": p_lip,
        "v_lip": v_lip_actual,
        "hit0": hit0_exact,
        "v_in0": v_in0_exact,
    }


def solve_planar_chute(
    p_enter,
    v_enter,
    dt_rail: float,
    clear_fn=None,
    dt_drop: float = 0.16,
    drop_clear_fn=None,
) -> dict[str, Any] | None:
    """Strictly planar (y_bow=0.0) Hermite rolling chute so layers never cross in Y."""
    T0 = phys.vnorm(v_enter)
    v0_speed = phys.vlen(v_enter)
    in_dir = 1.0 if v_enter[0] >= 0 else -1.0

    configs = []
    # Family A: Cross-stage chute staying strictly within layer's own Y corridor (|y_bow| <= 0.10)
    for x_ex in (in_dir * 1.02, in_dir * 0.88, in_dir * 0.72, in_dir * 0.52, in_dir * 0.30, 0.0):
        if (x_ex - p_enter[0]) * in_dir >= 0.25:
            for y_bow in (0.0, -0.10):
                configs.append(("cross", x_ex, y_bow, in_dir))

    # Family B: Confined S/U-turn chute within layer's own Y corridor (|y_bow| <= 0.16)
    x_wing_clamped = max(-1.05, min(1.05, p_enter[0] + in_dir * 0.65))
    x_wings = sorted({round(x_wing_clamped, 3), round(in_dir * 0.98, 3)}, key=lambda v: -abs(v - p_enter[0]))
    for y_bow in (-0.12, -0.08):
        for x_wing in x_wings:
            if (x_wing - p_enter[0]) * in_dir < 0.25:
                continue
            for x_ex in (-in_dir * 0.65, -in_dir * 0.45, -in_dir * 0.85, 0.0, -in_dir * 0.25):
                configs.append(("uturn", x_wing, y_bow, x_ex))

    best_meta = None
    best_cost = 1e9

    def build_knots(cfg, dz_drop: float, T1):
        if cfg[0] == "cross":
            _, x_ex, y_bow, _ = cfg
            p_mid = (0.5 * (p_enter[0] + x_ex), y_bow, p_enter[2] - 0.42 * dz_drop)
            p_exit = (x_ex, 0.0, p_enter[2] - dz_drop)
            T_mid = phys.vnorm((x_ex - p_enter[0], 0.0, -0.50 * dz_drop))
            return [(p_enter, T0), (p_mid, T_mid), (p_exit, T1)], p_exit
        else:
            _, x_wing, y_bow, x_ex = cfg
            p2x = 0.5 * (x_wing + x_ex)
            p1y = 0.60 * y_bow
            p2y = y_bow
            d1 = math.hypot(x_wing - p_enter[0], p1y)
            d2 = math.hypot(p2x - x_wing, p2y - p1y)
            d3 = math.hypot(x_ex - p2x, -p2y)
            dtot = max(0.3, d1 + d2 + d3)
            f1 = max(0.24, min(0.42, (d1 + 0.15) / dtot))
            f2 = max(f1 + 0.24, min(0.80, (d1 + d2) / dtot))
            p1 = (x_wing, p1y, p_enter[2] - f1 * dz_drop)
            T_p1 = phys.vnorm((0.0, -0.45, -0.35))  # Gentle turn without Hermite +Y overshoot
            p2 = (p2x, p2y, p_enter[2] - f2 * dz_drop)
            T_p2 = phys.vnorm((x_ex - p2x, -0.45 * p2y, -(d3 / dtot) * dz_drop))
            p_exit = (x_ex, 0.0, p_enter[2] - dz_drop)
            return [(p_enter, T0), (p1, T_p1), (p2, T_p2), (p_exit, T1)], p_exit

    dz_grid = (0.45, 0.65, 0.85, 1.05, 1.25, 1.50, 1.80, 2.15, 2.55, 3.00, 3.55, 4.15, 4.80, 5.50)
    for cfg in configs:
        exit_dir = cfg[3] if cfg[0] == "cross" else (-in_dir)
        for exit_ang in (26.0, 22.0, 18.0, 32.0):
            T1 = phys.vnorm((exit_dir * math.cos(math.radians(exit_ang)), 0.0, -math.sin(math.radians(exit_ang))))
            prev_dz = None
            prev_r = None
            sol = None
            for dz_probe in dz_grid:
                knots_p, _ = build_knots(cfg, dz_probe, T1)
                rp = phys.simulate_rolling_pts(phys.hermite_piecewise(knots_p, samples_per_seg=14), v0_speed)
                if rp is not None and prev_r is not None:
                    if (prev_r["T"] - dt_rail) * (rp["T"] - dt_rail) <= 0:
                        a, b = prev_dz, dz_probe
                        ra = prev_r
                        for _ in range(18):
                            mid = 0.5 * (a + b)
                            knots_m, px_m = build_knots(cfg, mid, T1)
                            rm = phys.simulate_rolling_pts(phys.hermite_piecewise(knots_m, samples_per_seg=14), v0_speed)
                            if rm is None:
                                a = mid
                                continue
                            if (ra["T"] - dt_rail) * (rm["T"] - dt_rail) <= 0:
                                b = mid
                            else:
                                a, ra = mid, rm
                        knots_m, px_m = build_knots(cfg, 0.5 * (a + b), T1)
                        rm = phys.simulate_rolling_pts(phys.hermite_piecewise(knots_m, samples_per_seg=14), v0_speed)
                        if rm is not None and abs(rm["T"] - dt_rail) <= 0.025:
                            sol = (rm, 0.5 * (a + b), px_m)
                            break
                if rp is not None:
                    prev_dz, prev_r = dz_probe, rp
            if sol is None:
                continue
            rm, dz_val, px = sol
            vx_ex_signed = rm["v_exit"] * T1[0]
            x_land = px[0] + vx_ex_signed * dt_drop
            if abs(x_land) > LANE_HALF_X:
                continue
            if rm["max_slope_deg"] <= 48.0 and rm["min_slope_deg"] >= 2.5:
                if clear_fn is not None and not clear_fn(rm["pts"]):
                    continue
                if drop_clear_fn is not None and not drop_clear_fn(px, phys.vscale(T1, rm["v_exit"]), rm):
                    continue
                cost = (0.0 if cfg[0] == "cross" else 8.0) + 1.4 * rm["max_slope_deg"] + 4.0 * max(0.0, abs(vx_ex_signed) - 2.6)
                if cost < best_cost:
                    best_cost = cost
                    best_meta = (cfg, dz_val, T1)

    if best_meta is None:
        return None
    cfg, dz_val, T1 = best_meta
    knots_hi, p_exit = build_knots(cfg, dz_val, T1)
    pts_hi = phys.hermite_piecewise(knots_hi, samples_per_seg=48)
    sim_hi = phys.simulate_rolling_pts(pts_hi, v0_speed)
    if sim_hi is None:
        return None
    return {"kind": cfg[0], "T": sim_hi["T"], "dz": dz_val, "v_exit": sim_hi["v_exit"], "max_slope_deg": sim_hi["max_slope_deg"], "p_exit": p_exit, "T1": T1, "sim": sim_hi}


def solve_lane_trajectory(
    notes: list[dict[str, Any]],
    ball_idx: int,
    lane_x: float,
    layer_y: float = 0.0,
) -> dict[str, Any]:
    """Solve trajectory for one ball in its local lane centered at lane_x."""
    t_spawn = float(notes[0]["start_time"])
    # Shift notes relative to t_spawn so entry ramp always takes PRE_ROLL=0.90s,
    # then shift all timestamps back by +t_spawn at the end.
    rel_notes = []
    for n in notes:
        nc = dict(n)
        nc["start_time"] = float(n["start_time"]) - t_spawn
        rel_notes.append(nc)

    z0_hit = z_ref(t_spawn)
    entry = solve_center_entry_ramp((-0.28, 0.0, z0_hit), (1.15, 0.0, -2.90))
    p = entry["hit0"]
    v_in = entry["v_in0"]

    segments: list[dict[str, Any]] = [
        {
            "kind": "chute",
            "t0": t_spawn,
            "t1": t_spawn + entry["t1"],
            "sim": entry["sim"],
            "rail_id": f"b{ball_idx}_entry_rail",
        },
        {
            "kind": "ballistic",
            "t0": t_spawn + entry["t1"],
            "t1": t_spawn + PRE_ROLL,
            "p0": entry["p_lip"],
            "v0": entry["v_lip"],
            "p1": p,
            "v1": v_in,
        },
    ]
    rails: list[dict[str, Any]] = [
        {
            "id": f"b{ball_idx}_entry_rail",
            "from_bounce": -1,
            "to_bounce": 0,
            "t0": t_spawn,
            "t1": t_spawn + entry["t1"],
            "sim": entry["sim"],
        }
    ]
    hits: list[tuple[float, float, float]] = [p]
    hit_times: list[float] = [PRE_ROLL + t_spawn]

    recent_boards: list[dict[str, Any]] = []
    recent_rails_pts: list[tuple[str, float, float, list]] = [
        (f"b{ball_idx}_entry_rail", t_spawn, t_spawn + entry["t1"], entry["sim"]["pts"])
    ]
    boards_data: list[dict[str, Any]] = []

    def make_board_obb(idx: int, hit_p, u_ax, n_ax) -> dict[str, Any]:
        dt_prev = (rel_notes[idx]["start_time"] - rel_notes[idx - 1]["start_time"]) if idx > 0 else 0.30
        dt_next = (rel_notes[idx + 1]["start_time"] - rel_notes[idx]["start_time"]) if idx + 1 < len(rel_notes) else 0.30
        min_dt = min(dt_prev, dt_next)
        base_len = 0.28 if min_dt < 0.25 else 0.36
        pitch_mod = max(-0.03, min(0.03, ((72 - rel_notes[idx]["note"]) / 24.0) * 0.03))
        board_len = max(0.24, base_len + pitch_mod)
        side_b = -1.0 if hit_p[0] < 0 else 1.0
        half_len = board_len * 0.5
        shift_u = side_b * (half_len * 0.34)
        contact = phys.vsub(hit_p, phys.vscale(n_ax, MARBLE_RADIUS))
        center = phys.vadd(phys.vsub(contact, phys.vscale(n_ax, phys.BOARD_THICKNESS * 0.5)), phys.vscale(u_ax, shift_u))
        return {
            "index": idx,
            "t_hit": float(notes[idx]["start_time"]) + PRE_ROLL,
            "center": center,
            "u": u_ax,
            "v": (0.0, 1.0, 0.0),
            "n": n_ax,
            "half_extents": (half_len, phys.BOARD_DEPTH_Y * 0.5, phys.BOARD_THICKNESS * 0.5),
        }

    def arc_clear(p0, v0, dur: float, t0_abs: float, self_b: dict[str, Any] | None = None) -> bool:
        if dur <= 0.015:
            return True
        b_slice = recent_boards[-12:] + ([self_b] if self_b is not None else [])
        r_slice = recent_rails_pts[-4:]
        n_samp = 24
        for s_k in range(1, n_samp):
            tau = dur * (s_k / n_samp)
            t_abs = t0_abs + tau
            pos = (
                p0[0] + v0[0] * tau,
                0.0,
                p0[2] + v0[2] * tau - 0.5 * GRAVITY * tau * tau,
            )
            for b in b_slice:
                if abs(t_abs - b["t_hit"]) < 0.042:
                    continue
                if phys.point_to_obb_dist(pos, b["center"], b["u"], b["v"], b["n"], b["half_extents"]) - MARBLE_RADIUS < 0.010:
                    return False
            for _, rt0, rt1, rpts in r_slice:
                if abs(t_abs - rt0) < 0.035 or abs(t_abs - rt1) < 0.035:
                    continue
                for k in range(0, len(rpts) - 1, 2):
                    if phys.point_to_segment_dist(pos, rpts[k], rpts[min(len(rpts) - 1, k + 2)]) < 0.148:
                        return False
        return True

    def chute_pts_clear(pts_cand, extra_board: dict[str, Any] | None = None) -> bool:
        b_slice = recent_boards[-12:] + ([extra_board] if extra_board is not None else [])
        r_slice = recent_rails_pts[-4:]
        for idx_p in range(1, len(pts_cand) - 1, 2):
            pos = pts_cand[idx_p]
            for b in b_slice:
                if phys.point_to_obb_dist(pos, b["center"], b["u"], b["v"], b["n"], b["half_extents"]) - MARBLE_RADIUS < 0.025:
                    return False
            for _, _, _, rpts in r_slice:
                for k in range(0, len(rpts) - 1, 3):
                    if phys.point_to_segment_dist(pos, rpts[k], rpts[min(len(rpts) - 1, k + 3)]) < 0.175:
                        return False
        return True

    def board_clears_rails(cand_b: dict[str, Any]) -> bool:
        """Ensure no newly placed board physically touches/clips an earlier rail on the same layer."""
        for _, _, _, rpts in recent_rails_pts[-4:]:
            for k in range(0, len(rpts) - 1, 2):
                pos = rpts[k]
                if phys.point_to_obb_dist(pos, cand_b["center"], cand_b["u"], cand_b["v"], cand_b["n"], cand_b["half_extents"]) - MARBLE_RADIUS < 0.020:
                    return False
        return True

    for i in range(len(notes) - 1):
        t0 = float(notes[i]["start_time"]) + PRE_ROLL
        t1 = float(notes[i + 1]["start_time"]) + PRE_ROLL
        dt = t1 - t0
        t_music_next = float(notes[i + 1]["start_time"])
        z_targ_next = z_ref(t_music_next)
        side = -1.0 if p[0] < 0 else 1.0

        if dt <= 0.62:
            x_targ = -side * (0.42 + 0.14 * ((notes[i + 1]["note"] % 5) / 4.0))
            best_b = None
            best_score = 1e9
            e_cands = (0.72, 0.64, 0.80, 0.56, 0.48, 0.86, 0.38, 0.30, 0.24)
            for obb_min_req in (0.34, 0.24):
                if best_b is not None:
                    break
                for e in e_cands:
                    angs = [(-side) * a for a in range(4, 50, 2)] + [side * a for a in range(6, 38, 3)]
                    for a_signed in angs:
                        alpha = math.radians(a_signed)
                        res = phys.bounce_velocity(v_in, alpha, e)
                        if res is None:
                            continue
                        v_out, vn_in, vn_out, u_b, n_b = res
                        if vn_in >= -0.70 or vn_out < 0.70:
                            continue
                        p_next = (
                            p[0] + v_out[0] * dt,
                            0.0,
                            p[2] + v_out[2] * dt - 0.5 * GRAVITY * dt * dt,
                        )
                        v_next_in = (v_out[0], 0.0, v_out[2] - GRAVITY * dt)
                        dz = p[2] - p_next[2]
                        if v_next_in[2] >= -1.2:
                            continue
                        if abs(p_next[0]) > LANE_HALF_X:
                            continue
                        if dz < 0.12:
                            continue
                        if abs(p_next[2] - z_targ_next) > 3.20:
                            continue
                        cand_b_i = make_board_obb(i, p, u_b, n_b)
                        obb_bad = False
                        for rb in recent_boards[-8:]:
                            if phys.obb_overlap(cand_b_i, rb, margin=0.08):
                                obb_bad = True
                                break
                            if phys.point_to_obb_dist(p_next, rb["center"], rb["u"], rb["v"], rb["n"], rb["half_extents"]) < obb_min_req:
                                obb_bad = True
                                break
                        if obb_bad:
                            continue
                        if not board_clears_rails(cand_b_i):
                            continue
                        if not arc_clear(p, v_out, dt, t0, self_b=cand_b_i):
                            continue

                        sc = (
                            abs(p_next[0] - x_targ)
                            + 2.4 * abs(p_next[2] - z_targ_next)
                            + 0.15 * abs(abs(v_next_in[2]) - 4.5)
                        )
                        if sc < best_score:
                            best_score = sc
                            best_b = (a_signed, e, u_b, n_b, v_out, p_next, v_next_in, cand_b_i)

            if best_b is None:
                raise RuntimeError(f"Ball {ball_idx} bounce failed at note {i} (dt={dt}, p={p}, z_targ={z_targ_next})")

            a_signed, e, u_b, n_b, v_out, p_next, v_next_in, cand_b_i = best_b
            boards_data.append({"index": i, "hit": p, "u": u_b, "n": n_b, "obb": cand_b_i, "e": e, "tilt_deg": a_signed})
            recent_boards.append(cand_b_i)
            segments.append({"kind": "ballistic", "t0": t0, "t1": t1, "p0": p, "v0": v_out, "p1": p_next, "v1": v_next_in})
            p, v_in = p_next, v_next_in
        else:
            # Glide rail transition for dt > 0.62s (up to 1.20s) confined strictly to own Y layer
            best_g = None
            best_sc = 1e9
            dt_drop = 0.12 if dt < 0.80 else 0.15
            for dt_enter in (0.05, 0.11, 0.16):
                dt_rail = dt - dt_enter - dt_drop
                if dt_rail < 0.38:
                    continue
                for e_g in (0.28, 0.38, 0.18, 0.50):
                    for a_signed in [(-side) * a for a in range(4, 42, 4)] + [side * a for a in range(10, 34, 6)]:
                        res = phys.bounce_velocity(v_in, math.radians(a_signed), e_g)
                        if res is None:
                            continue
                        v_out, vn_in, vn_out, u_b, n_b = res
                        if vn_in >= -0.70 or vn_out < 0.25:
                            continue
                        p_enter = (
                            p[0] + v_out[0] * dt_enter,
                            0.0,
                            p[2] + v_out[2] * dt_enter - 0.5 * GRAVITY * dt_enter * dt_enter,
                        )
                        v_enter = (v_out[0], 0.0, v_out[2] - GRAVITY * dt_enter)
                        if v_enter[2] >= -0.15 or abs(p_enter[0]) > LANE_HALF_X:
                            continue
                        cand_b_i = make_board_obb(i, p, u_b, n_b)
                        obb_bad = False
                        for rb in recent_boards[-10:]:
                            if phys.obb_overlap(cand_b_i, rb, margin=0.08):
                                obb_bad = True
                                break
                        if obb_bad:
                            continue
                        if not board_clears_rails(cand_b_i):
                            continue
                        if not arc_clear(p, v_out, dt_enter, t0, self_b=cand_b_i):
                            continue
                        chute = solve_planar_chute(
                            p_enter,
                            v_enter,
                            dt_rail,
                            clear_fn=lambda pts_c: chute_pts_clear(pts_c, extra_board=cand_b_i),
                            dt_drop=dt_drop,
                            drop_clear_fn=lambda p_ex, v_ex, _rm: arc_clear(p_ex, v_ex, dt_drop, t1 - dt_drop, self_b=cand_b_i),
                        )
                        if chute is None:
                            continue
                        p_exit_bi = chute["p_exit"]
                        v_exit_vec = phys.vscale(chute["T1"], chute["v_exit"])
                        p_next = (
                            p_exit_bi[0] + v_exit_vec[0] * dt_drop,
                            0.0,
                            p_exit_bi[2] + v_exit_vec[2] * dt_drop - 0.5 * GRAVITY * dt_drop * dt_drop,
                        )
                        v_next_in = (v_exit_vec[0], 0.0, v_exit_vec[2] - GRAVITY * dt_drop)
                        if abs(p_next[0]) > LANE_HALF_X:
                            continue
                        sc = abs(p_next[2] - z_targ_next) + 0.1 * chute["max_slope_deg"]
                        if sc < best_sc:
                            best_sc = sc
                            best_g = (a_signed, e_g, u_b, n_b, v_out, p_enter, v_enter, chute["sim"], p_exit_bi, v_exit_vec, p_next, v_next_in, cand_b_i, dt_enter)
                        if best_sc < 0.70:
                            break
                    if best_sc < 0.70:
                        break
                if best_sc < 0.70:
                    break

            if best_g is None:
                raise RuntimeError(f"Ball {ball_idx} rail failed at note {i} (dt={dt}, p={p})")

            a_signed, e_g, u_b, n_b, v_out, p_enter, v_enter, sim_bi, p_exit_bi, v_exit_vec, p_next, v_next_in, cand_b_i, dt_enter_used = best_g
            rid = f"b{ball_idx}_rail_{i:03d}"
            t_enter = t0 + dt_enter_used
            t_exit = t_enter + sim_bi["T"]
            boards_data.append({"index": i, "hit": p, "u": u_b, "n": n_b, "obb": cand_b_i, "e": e_g, "tilt_deg": a_signed})
            recent_boards.append(cand_b_i)
            segments.append({"kind": "ballistic", "t0": t0, "t1": t_enter, "p0": p, "v0": v_out, "p1": p_enter, "v1": v_enter})
            segments.append({"kind": "chute", "t0": t_enter, "t1": t_exit, "sim": sim_bi, "rail_id": rid})
            segments.append({"kind": "ballistic", "t0": t_exit, "t1": t1, "p0": p_exit_bi, "v0": v_exit_vec, "p1": p_next, "v1": v_next_in})
            rails.append({"id": rid, "from_bounce": i, "to_bounce": i + 1, "t0": t_enter, "t1": t_exit, "sim": sim_bi})
            recent_rails_pts.append((rid, t_enter, t_exit, sim_bi["pts"]))
            p, v_in = p_next, v_next_in

        hits.append(p)
        hit_times.append(t1)

    # Save final state before last board so all 3 balls can coordinate a shared Z_END landing
    last_i = len(notes) - 1
    t_last = float(notes[last_i]["start_time"]) + PRE_ROLL
    return {
        "ball_idx": ball_idx,
        "lane_x": lane_x,
        "layer_y": layer_y,
        "palette": BALL_PALETTES[ball_idx],
        "t_spawn": t_spawn,
        "t_last": t_last,
        "p_last": p,
        "v_in_last": v_in,
        "last_note": notes[last_i],
        "notes": notes,
        "boards_data": boards_data,
        "recent_boards": recent_boards[-10:],
        "recent_rails": recent_rails_pts[-4:],
        "rails": rails,
        "segments": segments,
        "make_board_obb": make_board_obb,
    }


BULB_CHAR_WIDTHS = {ch: 5 for ch in "ABCDEGHILMNORSTWZ"}
BULB_CHAR_WIDTHS[":"] = 1
BULB_CHAR_WIDTHS["."] = 1


def _layout_bulb_row_words(segments: list[tuple[str | None, str]], pitch: float) -> tuple[dict[str, float], float]:
    """Compute exact world X centers of split words matching BULB_FONT kerning (5 cols + 1 gap, 4 cols/space)."""
    full_text = "".join(txt for _, txt in segments)
    char_cols: list[tuple[int, int]] = []
    cur = 0
    for ch in full_text:
        if ch == " ":
            char_cols.append((cur, cur + 3))
            cur += 4
        else:
            w = BULB_CHAR_WIDTHS.get(ch, 5)
            char_cols.append((cur, cur + w - 1))
            cur += w + 1
    total_cols = max(1, cur - 1)
    centers: dict[str, float] = {}
    ci = 0
    for seg_id, txt in segments:
        c0 = None
        c1 = None
        for ch in txt:
            a, b = char_cols[ci]
            ci += 1
            if ch != " ":
                if c0 is None:
                    c0 = a
                c1 = b
        if seg_id is not None and c0 is not None and c1 is not None:
            x0 = (c0 - 0.5 * (total_cols - 1)) * pitch
            x1 = (c1 - 0.5 * (total_cols - 1)) * pitch
            centers[seg_id] = round(0.5 * (x0 + x1), 5)
    return centers, total_cols * pitch


def compute_finale_word_layout(z_end_min: float) -> dict[str, Any]:
    """Compute centered 2-row GLASS MARBLES / ON WOOD + credit layout and unified bridge elevation."""
    z_finale = round(z_end_min - 1.95, 3)
    z_row1 = round(z_finale + 0.28, 3)
    z_row2 = round(z_finale - 0.16, 3)
    z_row3 = round(z_finale - 0.50, 3)
    z_ledge = round(z_finale - 0.94, 3)
    z_top_bridge = round(z_ledge - MARBLE_RADIUS, 5)
    y_morph = round(BACK_WALL_Y_MB - 0.14, 3)

    p1, r1_rad = 0.036, 0.0135
    p2, r2_rad = 0.032, 0.0120
    r1_map, _ = _layout_bulb_row_words([("GLASS", "GLASS"), (None, " "), ("MARBLE", "MARBLE"), ("S", "S")], p1)
    r2_map, _ = _layout_bulb_row_words([("ON", "ON"), (None, " "), ("WOOD", "WOOD")], p2)

    targets = {
        "GLASS":  {"text": "GLASS",  "row": 1, "pitch": p1, "radius": r1_rad, "x": r1_map["GLASS"],  "y": y_morph, "z": z_row1},
        "MARBLE": {"text": "MARBLE", "row": 1, "pitch": p1, "radius": r1_rad, "x": r1_map["MARBLE"], "y": y_morph, "z": z_row1},
        "S":      {"text": "S",      "row": 1, "pitch": p1, "radius": r1_rad, "x": r1_map["S"],      "y": y_morph, "z": z_row1},
        "ON":     {"text": "ON",     "row": 2, "pitch": p2, "radius": r2_rad, "x": r2_map["ON"],     "y": y_morph, "z": z_row2},
        "WOOD":   {"text": "WOOD",   "row": 2, "pitch": p2, "radius": r2_rad, "x": r2_map["WOOD"],   "y": y_morph, "z": z_row2},
    }
    return {
        "z_finale": z_finale,
        "z_row1": z_row1,
        "z_row2": z_row2,
        "z_row3": z_row3,
        "z_ledge": z_ledge,
        "z_top_bridge": z_top_bridge,
        "y_morph": y_morph,
        "row1_text": "GLASS MARBLES",
        "row1_pitch": p1,
        "row1_radius": r1_rad,
        "row2_text": "ON WOOD",
        "row2_pitch": p2,
        "row2_radius": r2_rad,
        "credit_text": "ARTIST : ZERO.SOE",
        "credit_pitch": 0.0185,
        "credit_radius": 0.0070,
        "targets": targets,
    }


def _parabolic_v0(p0: tuple[float, float, float], p1: tuple[float, float, float], dt: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    vx = (p1[0] - p0[0]) / dt
    vy = (p1[1] - p0[1]) / dt
    vz0 = (p1[2] - p0[2] + 0.5 * GRAVITY * dt * dt) / dt
    return (vx, vy, vz0), (vx, vy, vz0 - GRAVITY * dt)


def finalize_lane_with_catch_board(
    sol: dict[str, Any],
    x_end_world: float,
    z_end: float,
    num_balls: int,
) -> dict[str, Any]:
    """Solve the final bounce off the last musical board onto a level horizontal catch board."""
    ball_idx = sol["ball_idx"]
    lane_x = sol["lane_x"]
    layer_y = sol["layer_y"]
    t_last = sol["t_last"]
    p = sol["p_last"]
    v_in = sol["v_in_last"]
    boards_data = sol["boards_data"]
    segments = sol["segments"]
    rails = sol["rails"]
    make_board_obb = sol["make_board_obb"]
    last_i = len(boards_data)

    x_end_local = x_end_world - lane_x
    best_final = None
    best_err = 1e9
    for dt_end in [0.35 + 0.01 * k for k in range(75)]:  # 0.35s .. 1.09s
        for e_c in [0.25 + 0.05 * j for j in range(15)]:
            for a_signed in range(-55, 56, 1):
                res = phys.bounce_velocity(v_in, math.radians(a_signed), e_c)
                if res is None:
                    continue
                v_out, vn_in, vn_out, u_b, n_b = res
                if vn_in >= -0.60 or vn_out < 0.40:
                    continue
                p_land = (
                    p[0] + v_out[0] * dt_end,
                    0.0,
                    p[2] + v_out[2] * dt_end - 0.5 * GRAVITY * dt_end * dt_end,
                )
                v_land = (v_out[0], 0.0, v_out[2] - GRAVITY * dt_end)
                if v_land[2] >= -1.2:
                    continue
                cand_last_try = make_board_obb(last_i, p, u_b, n_b)
                # Ensure final flight clears last board and recent boards/rails
                arc_ok = True
                for s_k in range(1, 20):
                    tau = dt_end * (s_k / 20.0)
                    pos_t = (p[0] + v_out[0] * tau, 0.0, p[2] + v_out[2] * tau - 0.5 * GRAVITY * tau * tau)
                    if tau > 0.042:
                        if phys.point_to_obb_dist(pos_t, cand_last_try["center"], cand_last_try["u"], cand_last_try["v"], cand_last_try["n"], cand_last_try["half_extents"]) - MARBLE_RADIUS < 0.010:
                            arc_ok = False
                            break
                    for rb in sol.get("recent_boards", []):
                        if abs((t_last + tau) - rb["t_hit"]) >= 0.042:
                            if phys.point_to_obb_dist(pos_t, rb["center"], rb["u"], rb["v"], rb["n"], rb["half_extents"]) - MARBLE_RADIUS < 0.010:
                                arc_ok = False
                                break
                    if not arc_ok:
                        break
                if not arc_ok:
                    continue
                err = abs(p_land[0] - x_end_local) + 4.0 * abs(p_land[2] - z_end)
                if err < best_err:
                    best_err = err
                    best_final = (dt_end, a_signed, e_c, u_b, n_b, v_out, p_land, v_land)

    assert best_final is not None and best_err < 0.35, f"Ball {ball_idx} final catch landing failed (err={best_err})"
    dt_end, a_signed, e_c, u_b, n_b, v_out, p_land, v_land = best_final
    t_land = t_last + dt_end

    # Last musical board
    cand_last = make_board_obb(last_i, p, u_b, n_b)
    boards_data.append({"index": last_i, "hit": p, "u": u_b, "n": n_b, "obb": cand_last, "e": e_c, "tilt_deg": a_signed, "is_terminal": False})

    # Ballistic flight onto the horizontal catch board
    segments.append({"kind": "ballistic", "t0": t_last, "t1": t_land, "p0": p, "v0": v_out, "p1": p_land, "v1": v_land})
    # Rest cleanly on the catch board for POST_ROLL
    t_end_total = t_land + POST_ROLL
    segments.append({"kind": "rest", "t0": t_land, "t1": t_end_total, "pos": p_land})

    # Horizontal termination catch board centered at evenly-spaced x_end_local
    u_flat = (1.0, 0.0, 0.0)
    n_flat = (0.0, 0.0, 1.0)
    catch_len = 0.62
    catch_center = (
        x_end_local,
        0.0,
        p_land[2] - MARBLE_RADIUS - phys.BOARD_THICKNESS * 0.5,
    )
    catch_obb = {
        "index": last_i + 1,
        "t_hit": t_land,
        "center": catch_center,
        "u": u_flat,
        "v": (0.0, 1.0, 0.0),
        "n": n_flat,
        "half_extents": (catch_len * 0.5, phys.BOARD_DEPTH_Y * 0.5, phys.BOARD_THICKNESS * 0.5),
    }
    boards_data.append({"index": last_i + 1, "hit": p_land, "u": u_flat, "n": n_flat, "obb": catch_obb, "e": 0.0, "tilt_deg": 0.0, "is_terminal": True})

    # Shift all X by +lane_x and Y by +layer_y into world space
    def shift_xy(pt):
        return (round(pt[0] + lane_x, 5), round(pt[1] + layer_y, 5), round(pt[2], 5))

    boards_out = []
    last_note = sol["last_note"]
    notes_list = sol["notes"]
    for bd in boards_data:
        idx = bd["index"]
        obb = bd["obb"]
        is_term = bool(bd.get("is_terminal", False))
        t_hit_bd = obb["t_hit"]
        n_cur = last_note if is_term else notes_list[idx]
        src_stem = "terminal" if is_term else str(n_cur.get("source_stem", "synth"))
        inst_type = "terminal_catch" if is_term else str(
            n_cur.get("instrument_type", "keyboard_fill" if src_stem == "keyboard" else "square_lead")
        )
        boards_out.append({
            "index": idx,
            "ball_idx": ball_idx,
            "is_terminal": is_term,
            "t_hit": round(t_hit_bd, 5),
            "frame_hit": int(round(t_hit_bd * FPS)),
            "track_idx": ball_idx + 1,
            "source_stem": src_stem,
            "instrument": "Terminal Catch" if is_term else n_cur["instrument"],
            "instrument_type": inst_type,
            "note": int(n_cur["note"]),
            "pitch_name": "END" if is_term else n_cur["pitch_name"],
            "velocity": 110 if is_term else int(n_cur["velocity"]),
            "hit_center": shift_xy(bd["hit"]),
            "board_center": shift_xy(obb["center"]),
            "u": [round(x, 6) for x in obb["u"]],
            "n": [round(x, 6) for x in obb["n"]],
            "length": round(obb["half_extents"][0] * 2.0, 5),
            "width_y": round(obb["half_extents"][1] * 2.0, 5),
            "thickness": round(obb["half_extents"][2] * 2.0, 5),
        })

    # Shift rails into world space
    rails_shifted = []
    for r in rails:
        rc = dict(r)
        sim_c = dict(r["sim"])
        sim_c["pts"] = [(pt[0] + lane_x, pt[1] + layer_y, pt[2]) for pt in r["sim"]["pts"]]
        rc["sim"] = sim_c
        rails_shifted.append(rc)
    rails_out = phys.build_rail_pipe_polylines(rails_shifted)

    return {
        "ball_idx": ball_idx,
        "lane_x": lane_x,
        "layer_y": layer_y,
        "palette": sol["palette"],
        "t_spawn": sol["t_spawn"],
        "t_end": t_end_total,
        "land_pos": shift_xy(p_land),
        "boards": boards_out,
        "rails": rails_out,
        "segments": segments,
    }


def finalize_lane_with_finale_bounce(
    sol: dict[str, Any],
    pad_x_target: float,
    z_ledge: float,
    word_id: str,
    word_target_world: tuple[float, float, float],
    skip_hops_world: list[tuple[tuple[float, float, float], float]],
    dt_leap: float,
    unified_bridge: bool = True,
) -> dict[str, Any]:
    """Solve bounce off last musical board onto the Finale Bridge (converging in Y), optional skip-hops, then 3D upward leap to word center."""
    ball_idx = sol["ball_idx"]
    lane_x = sol["lane_x"]
    layer_y = sol["layer_y"]
    t_last = sol["t_last"]
    p = sol["p_last"]
    v_in = sol["v_in_last"]
    boards_data = sol["boards_data"]
    segments = sol["segments"]
    rails = sol["rails"]
    make_board_obb = sol["make_board_obb"]
    last_i = len(boards_data)

    x_end_local = pad_x_target - lane_x
    best_final = None
    best_err = 1e9
    for e_c in [0.12 + 0.04 * j for j in range(18)]:
        for a_signed in range(-58, 59, 1):
            res = phys.bounce_velocity(v_in, math.radians(a_signed), e_c)
            if res is None:
                continue
            v_out, vn_in, vn_out, u_b, n_b = res
            if vn_in >= -0.60 or vn_out < 0.35:
                continue
            disc = v_out[2] ** 2 - 2.0 * GRAVITY * (z_ledge - p[2])
            if disc <= 0.0:
                continue
            dt_end = (v_out[2] + math.sqrt(disc)) / GRAVITY
            if not (0.35 <= dt_end <= 1.25):
                continue
            vy_conv = (-layer_y / dt_end) if unified_bridge else 0.0
            p_land = (
                p[0] + v_out[0] * dt_end,
                -layer_y if unified_bridge else 0.0,
                z_ledge,
            )
            v_out_3d = (v_out[0], vy_conv, v_out[2])
            v_land = (v_out[0], vy_conv, v_out[2] - GRAVITY * dt_end)
            if v_land[2] >= -1.5 or phys.vlen(v_land) > 14.5:
                continue
            err = abs(p_land[0] - x_end_local)
            if err >= best_err:
                continue
            cand_last_try = make_board_obb(last_i, p, u_b, n_b)
            arc_ok = True
            for s_k in range(1, 20):
                tau = dt_end * (s_k / 20.0)
                pos_t = (p[0] + v_out_3d[0] * tau, vy_conv * tau, p[2] + v_out_3d[2] * tau - 0.5 * GRAVITY * tau * tau)
                if tau > 0.042:
                    if phys.point_to_obb_dist(pos_t, cand_last_try["center"], cand_last_try["u"], cand_last_try["v"], cand_last_try["n"], cand_last_try["half_extents"]) - MARBLE_RADIUS < 0.010:
                        arc_ok = False
                        break
                for rb in sol.get("recent_boards", []):
                    if abs((t_last + tau) - rb["t_hit"]) >= 0.042:
                        if phys.point_to_obb_dist(pos_t, rb["center"], rb["u"], rb["v"], rb["n"], rb["half_extents"]) - MARBLE_RADIUS < 0.010:
                            arc_ok = False
                            break
                if not arc_ok:
                    break
            if not arc_ok:
                continue
            best_err = err
            best_final = (dt_end, a_signed, e_c, u_b, n_b, v_out_3d, p_land, v_land)

    assert best_final is not None and best_err < 0.25, f"Ball {ball_idx} finale bounce failed (err={best_err})"
    dt_end, a_signed, e_c, u_b, n_b, v_out, p_land, v_land = best_final
    t_pad = t_last + dt_end

    cand_last = make_board_obb(last_i, p, u_b, n_b)
    boards_data.append({"index": last_i, "hit": p, "u": u_b, "n": n_b, "obb": cand_last, "e": e_c, "tilt_deg": a_signed, "is_terminal": False})

    # Drop from last musical board onto the Finale Bridge (converging cleanly onto y=0.0)
    segments.append({"kind": "ballistic", "t0": t_last, "t1": t_pad, "p0": p, "v0": v_out, "p1": p_land, "v1": v_land})

    bridge_hits = [
        {
            "ball_type": "melody",
            "ball_idx": ball_idx,
            "x": round(p_land[0] + lane_x, 5),
            "t_hit": round(t_pad, 5),
            "frame_hit": int(round(t_pad * FPS)),
            "rgb": sol["palette"]["color"],
        }
    ]
    pad_hit_times = [t_pad]
    pad_hit_xs_local = [p_land[0]]
    p_cur = p_land
    t_cur = t_pad

    # Optional playful skip-hops across the Finale Bridge
    for target_w, dt_hop in skip_hops_world:
        y_tgt = (0.0 - layer_y) if unified_bridge else (target_w[1] - layer_y)
        p_nxt = (target_w[0] - lane_x, y_tgt, z_ledge)
        v0_h, v1_h = _parabolic_v0(p_cur, p_nxt, dt_hop)
        segments.append({"kind": "ballistic", "t0": t_cur, "t1": t_cur + dt_hop, "p0": p_cur, "v0": v0_h, "p1": p_nxt, "v1": v1_h})
        p_cur = p_nxt
        t_cur += dt_hop
        pad_hit_times.append(t_cur)
        pad_hit_xs_local.append(p_cur[0])
        bridge_hits.append({
            "ball_type": "melody",
            "ball_idx": ball_idx,
            "x": round(p_cur[0] + lane_x, 5),
            "t_hit": round(t_cur, 5),
            "frame_hit": int(round(t_cur * FPS)),
            "rgb": sol["palette"]["color"],
        })

    # Heroic 3D upward parabolic leap straight into the word center on the title plank
    p_word_local = (word_target_world[0] - lane_x, word_target_world[1] - layer_y, word_target_world[2])
    v0_leap, v1_leap = _parabolic_v0(p_cur, p_word_local, dt_leap)
    t_morph = t_cur + dt_leap
    segments.append({"kind": "ballistic", "t0": t_cur, "t1": t_morph, "p0": p_cur, "v0": v0_leap, "p1": p_word_local, "v1": v1_leap})
    segments.append({"kind": "rest", "t0": t_morph, "t1": t_morph + 5.0, "pos": p_word_local})

    if not unified_bridge:
        min_px = min(pad_hit_xs_local) - 0.16
        max_px = max(pad_hit_xs_local) + 0.16
        plank_cx = 0.5 * (min_px + max_px)
        plank_len = max(0.46, max_px - min_px)
        u_flat = (1.0, 0.0, 0.0)
        n_flat = (0.0, 0.0, 1.0)
        catch_center = (
            plank_cx,
            0.0,
            z_ledge - MARBLE_RADIUS - phys.BOARD_THICKNESS * 0.5,
        )
        catch_obb = {
            "index": last_i + 1,
            "t_hit": t_pad,
            "t_hits": pad_hit_times,
            "center": catch_center,
            "u": u_flat,
            "v": (0.0, 1.0, 0.0),
            "n": n_flat,
            "half_extents": (plank_len * 0.5, phys.BOARD_DEPTH_Y * 0.5, phys.BOARD_THICKNESS * 0.5),
        }
        boards_data.append({"index": last_i + 1, "hit": p_land, "u": u_flat, "n": n_flat, "obb": catch_obb, "e": 0.9, "tilt_deg": 0.0, "is_terminal": True})

    def shift_xy(pt):
        return (round(pt[0] + lane_x, 5), round(pt[1] + layer_y, 5), round(pt[2], 5))

    boards_out = []
    last_note = sol["last_note"]
    notes_list = sol["notes"]
    for bd in boards_data:
        idx = bd["index"]
        obb = bd["obb"]
        is_term = bool(bd.get("is_terminal", False))
        t_hit_bd = obb["t_hit"]
        n_cur = last_note if is_term else notes_list[idx]
        src_stem = "terminal" if is_term else str(n_cur.get("source_stem", "synth"))
        inst_type = "finale_trampoline" if is_term else str(
            n_cur.get("instrument_type", "keyboard_fill" if src_stem == "keyboard" else "square_lead")
        )
        bd_dict = {
            "index": idx,
            "ball_idx": ball_idx,
            "is_terminal": is_term,
            "t_hit": round(t_hit_bd, 5),
            "frame_hit": int(round(t_hit_bd * FPS)),
            "track_idx": ball_idx + 1,
            "source_stem": src_stem,
            "instrument": "Finale Trampoline" if is_term else n_cur["instrument"],
            "instrument_type": inst_type,
            "note": int(n_cur["note"]),
            "pitch_name": f"FIN_{word_id}" if is_term else n_cur["pitch_name"],
            "velocity": 115 if is_term else int(n_cur["velocity"]),
            "hit_center": shift_xy(bd["hit"]),
            "board_center": shift_xy(obb["center"]),
            "u": [round(x, 6) for x in obb["u"]],
            "n": [round(x, 6) for x in obb["n"]],
            "length": round(obb["half_extents"][0] * 2.0, 5),
            "width_y": round(obb["half_extents"][1] * 2.0, 5),
            "thickness": round(obb["half_extents"][2] * 2.0, 5),
        }
        if is_term and obb.get("t_hits"):
            bd_dict["t_hits"] = [round(th, 5) for th in obb["t_hits"]]
            bd_dict["hit_frames"] = [int(round(th * FPS)) for th in obb["t_hits"]]
        boards_out.append(bd_dict)

    rails_shifted = []
    for r in rails:
        rc = dict(r)
        sim_c = dict(r["sim"])
        sim_c["pts"] = [(pt[0] + lane_x, pt[1] + layer_y, pt[2]) for pt in r["sim"]["pts"]]
        rc["sim"] = sim_c
        rails_shifted.append(rc)
    rails_out = phys.build_rail_pipe_polylines(rails_shifted)

    return {
        "ball_idx": ball_idx,
        "lane_x": lane_x,
        "layer_y": layer_y,
        "palette": sol["palette"],
        "t_spawn": sol["t_spawn"],
        "t_pad": round(t_pad, 5),
        "t_morph": round(t_morph, 5),
        "word_id": word_id,
        "t_end": t_morph + 3.50,
        "land_pos": shift_xy(p_land),
        "bridge_hits": bridge_hits,
        "boards": boards_out,
        "rails": rails_out,
        "segments": segments,
    }


def _sample_chute_sim(sim: dict, tau: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    import bisect
    tp = sim["t_prof"]
    pts = sim["pts"]
    vp = sim["v_prof"]
    if tau <= 0.0:
        d = phys.vnorm(phys.vsub(pts[1], pts[0]))
        return pts[0], phys.vscale(d, vp[0])
    if tau >= tp[-1]:
        d = phys.vnorm(phys.vsub(pts[-1], pts[-2]))
        return pts[-1], phys.vscale(d, vp[-1])
    k = max(0, min(len(tp) - 2, bisect.bisect_right(tp, tau) - 1))
    dt_k = max(1e-9, tp[k + 1] - tp[k])
    u = (tau - tp[k]) / dt_k
    pos = (
        pts[k][0] + u * (pts[k + 1][0] - pts[k][0]),
        pts[k][1] + u * (pts[k + 1][1] - pts[k][1]),
        pts[k][2] + u * (pts[k + 1][2] - pts[k][2]),
    )
    spd = vp[k] + u * (vp[k + 1] - vp[k])
    d = phys.vnorm(phys.vsub(pts[k + 1], pts[k]))
    return pos, phys.vscale(d, spd)


def sample_ball_at_t(ball_sol: dict, t: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lane_x = ball_sol["lane_x"]
    layer_y = ball_sol.get("layer_y", 0.0)
    segs = ball_sol["segments"]
    if t <= segs[0]["t0"]:
        pt = segs[0]["sim"]["pts"][0]
        return (pt[0] + lane_x, pt[1] + layer_y, pt[2]), (0.0, 0.0, 0.0)
    if t >= segs[-1]["t1"]:
        s = segs[-1]
        if s["kind"] == "rest":
            pt = s["pos"]
            return (pt[0] + lane_x, pt[1] + layer_y, pt[2]), (0.0, 0.0, 0.0)
        tau = s["t1"] - s["t0"]
        p0, v0 = s["p0"], s["v0"]
        pos = (p0[0] + v0[0] * tau + lane_x, p0[1] + v0[1] * tau + layer_y, p0[2] + v0[2] * tau - 0.5 * GRAVITY * tau * tau)
        return pos, (v0[0], v0[1], v0[2] - GRAVITY * tau)

    for s in segs:
        if s["t0"] - 1e-9 <= t <= s["t1"] + 1e-9:
            if s["kind"] == "rest":
                pt = s["pos"]
                return (pt[0] + lane_x, pt[1] + layer_y, pt[2]), (0.0, 0.0, 0.0)
            elif s["kind"] == "ballistic":
                tau = max(0.0, t - s["t0"])
                p0, v0 = s["p0"], s["v0"]
                pos = (p0[0] + v0[0] * tau + lane_x, p0[1] + v0[1] * tau + layer_y, p0[2] + v0[2] * tau - 0.5 * GRAVITY * tau * tau)
                vel = (v0[0], v0[1], v0[2] - GRAVITY * tau)
                return pos, vel
            else:
                pos_l, vel = phys.eval_rolling_table(s["sim"], max(0.0, t - s["t0"]))
                return (pos_l[0] + lane_x, pos_l[1] + layer_y, pos_l[2]), vel
    return (lane_x, layer_y, 0.0), (0.0, 0.0, 0.0)


def _build_serpentine_glide_sim(
    p_start,
    v_start,
    p_end,
    v_end,
    dt_rail: float,
    side_sign: float,
    straight_head_sec: float = 0.0,
) -> dict[str, Any]:
    """C1-continuous piecewise cubic Hermite trajectory strictly monotone downward (dz/dtau < 0)."""
    dt_rail = max(0.20, float(dt_rail))
    waypoints: list[tuple[float, tuple[float, float, float], float]] = [(0.0, p_start, float(v_start[0]))]
    if dt_rail > 1.35:
        n_wp = max(1, int(round(dt_rail / 1.15)))
        for k in range(1, n_wp + 1):
            frac = k / (n_wp + 1)
            tau_k = dt_rail * frac
            if tau_k < straight_head_sec or tau_k > dt_rail - 0.85:
                xk = 0.0
            else:
                xk = (0.11 if k % 2 == 1 else -0.11) * side_sign
            zk = p_start[2] + (p_end[2] - p_start[2]) * frac
            waypoints.append((tau_k, (xk, 0.0, zk), 0.0))
    waypoints.append((dt_rail, p_end, float(v_end[0])))

    n_seg = len(waypoints) - 1
    seg_avg_vz = []
    for k in range(n_seg):
        seg_dt = max(1e-6, waypoints[k + 1][0] - waypoints[k][0])
        seg_avg_vz.append((waypoints[k + 1][1][2] - waypoints[k][1][2]) / seg_dt)

    # Compute Fritsch-Carlson monotone vz at every waypoint so Hermite never overshoots upward
    vz_nodes: list[float] = []
    for k in range(len(waypoints)):
        if k == 0:
            vz_raw = float(v_start[2])
            lo_vz = min(-0.30, 2.4 * seg_avg_vz[0])
            hi_vz = min(-0.25, 0.25 * seg_avg_vz[0])
            vz_nodes.append(max(lo_vz, min(hi_vz, vz_raw)))
        elif k == len(waypoints) - 1:
            vz_raw = float(v_end[2])
            lo_vz = min(-0.30, 2.4 * seg_avg_vz[-1])
            hi_vz = min(-0.25, 0.25 * seg_avg_vz[-1])
            vz_nodes.append(max(lo_vz, min(hi_vz, vz_raw)))
        else:
            v_avg_adj = 0.5 * (seg_avg_vz[k - 1] + seg_avg_vz[k])
            lim_prev = min(-0.30, 2.4 * seg_avg_vz[k - 1])
            lim_next = min(-0.30, 2.4 * seg_avg_vz[k])
            vz_nodes.append(max(max(lim_prev, lim_next), min(-0.25, v_avg_adj)))

    pts: list[tuple[float, float, float]] = []
    t_cum: list[float] = []
    v_prof: list[float] = []
    for k in range(n_seg):
        tau_a, pa, vxa = waypoints[k]
        tau_b, pb, vxb = waypoints[k + 1]
        vza, vzb = vz_nodes[k], vz_nodes[k + 1]
        seg_dt = max(1e-6, tau_b - tau_a)
        n_samp = max(10 if n_seg > 1 else 24, int(math.ceil(seg_dt / 0.025)))
        ma_x, mb_x = vxa * seg_dt, vxb * seg_dt
        ma_z, mb_z = vza * seg_dt, vzb * seg_dt
        i_start = 0 if k == 0 else 1
        for i in range(i_start, n_samp + 1):
            u = i / n_samp
            u2 = u * u
            u3 = u2 * u
            h00 = 2.0 * u3 - 3.0 * u2 + 1.0
            h10 = u3 - 2.0 * u2 + u
            h01 = -2.0 * u3 + 3.0 * u2
            h11 = u3 - u2
            dh00 = 6.0 * u2 - 6.0 * u
            dh10 = 3.0 * u2 - 4.0 * u + 1.0
            dh01 = -6.0 * u2 + 6.0 * u
            dh11 = 3.0 * u2 - 2.0 * u
            px = h00 * pa[0] + h10 * ma_x + h01 * pb[0] + h11 * mb_x
            pz = h00 * pa[2] + h10 * ma_z + h01 * pb[2] + h11 * mb_z
            vx = (dh00 * pa[0] + dh10 * ma_x + dh01 * pb[0] + dh11 * mb_x) / seg_dt
            vz = (dh00 * pa[2] + dh10 * ma_z + dh01 * pb[2] + dh11 * mb_z) / seg_dt
            pts.append((px, 0.0, pz))
            t_cum.append(tau_a + u * seg_dt)
            v_prof.append(math.hypot(vx, vz))

    return {
        "T": t_cum[-1],
        "v_exit": v_prof[-1],
        "max_slope_deg": 88.0,
        "min_slope_deg": 45.0,
        "pts": pts,
        "t_cum": t_cum,
        "v_prof": v_prof,
    }


def build_drum_guide_pipes(rails: list[dict[str, Any]], ball_radius: float) -> list[dict[str, Any]]:
    """Build twist-free twin bronze guide pipes + Y-bracket mount metadata (one arm per pipe)."""
    pipe_r = 0.015
    lat = 0.89 * ball_radius + pipe_r
    back_y = 0.52 * ball_radius
    y_fork_off = ball_radius + 0.22
    out = []
    for r in rails:
        pts = r["sim"]["pts"]
        n_pts = len(pts)
        s_arr = [0.0]
        for k in range(1, n_pts):
            s_arr.append(s_arr[-1] + phys.vlen(phys.vsub(pts[k], pts[k - 1])))
        total_s = s_arr[-1]

        if n_pts >= 4:
            if total_s < 1.50:
                s_targs = [0.25 * total_s, 0.75 * total_s]
            else:
                s_targs = []
                sc = 0.45
                while sc <= total_s - 0.45:
                    s_targs.append(sc)
                    sc += 1.45
                if len(s_targs) < 2:
                    s_targs = [0.25 * total_s, 0.75 * total_s]
            brk_set = set()
            for st in s_targs:
                k_best = min(range(1, n_pts - 1), key=lambda kk: abs(s_arr[kk] - st))
                brk_set.add(k_best)
            if len(brk_set) < 2 and n_pts >= 4:
                brk_set.update((1, n_pts - 2))
        else:
            brk_set = set(range(n_pts))

        left_pts = []
        right_pts = []
        y_brackets = []
        for k in range(n_pts):
            c = pts[k]
            if k == 0:
                T = phys.vnorm(phys.vsub(pts[1], pts[0]))
            elif k == n_pts - 1:
                T = phys.vnorm(phys.vsub(pts[-1], pts[-2]))
            else:
                T = phys.vnorm(phys.vsub(pts[k + 1], pts[k - 1]))
            N_xz = phys.vnorm((-T[2], 0.0, T[0]), fallback=(1.0, 0.0, 0.0))
            if N_xz[0] < 0.0:
                N_xz = (-N_xz[0], 0.0, -N_xz[2])
            s_frac = k / max(1, n_pts - 1)
            flare = 0.0
            if s_frac > 0.94:
                u = (s_frac - 0.94) / 0.06
                flare = 0.016 * (u * u)
            elif r.get("is_entry") is False and s_frac < 0.06:
                u = (0.06 - s_frac) / 0.06
                flare = 0.016 * (u * u)
            lat_k = lat + flare
            pl = (c[0] - lat_k * N_xz[0], c[1] + back_y, c[2] - lat_k * N_xz[2])
            pr = (c[0] + lat_k * N_xz[0], c[1] + back_y, c[2] + lat_k * N_xz[2])
            left_pts.append(pl)
            right_pts.append(pr)
            if k in brk_set:
                p_fork = (c[0], c[1] + y_fork_off, c[2])
                y_brackets.append({
                    "left": phys.r5(pl),
                    "right": phys.r5(pr),
                    "fork": phys.r5(p_fork),
                    "wall_x": round(c[0], 5),
                    "wall_z": round(c[2], 5),
                })

        out.append({
            "id": r["id"],
            "is_drum_guide": True,
            "ball_radius": ball_radius,
            "from_bounce": r["from_bounce"],
            "to_bounce": r["to_bounce"],
            "t0": r["t0"],
            "t1": r["t1"],
            "pipe_radius": pipe_r,
            "pipe_front": [phys.r5(p) for p in left_pts],
            "pipe_back": [phys.r5(p) for p in right_pts],
            "ties": [],
            "y_brackets": y_brackets,
            "center_pts": [phys.r5(p) for p in pts],
        })
    return out


def solve_drum_tower_lane(
    notes: list[dict[str, Any]],
    cfg: dict[str, Any],
    z_end_drum: float,
    t_finale_land: float,
    t_end_total: float,
    finale_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve collision-free Left Kick or Right Snare/HiHat flanking drum tower with silent breakdown glides."""
    drum_idx = int(cfg["drum_idx"])
    side_sign = float(cfg["side_sign"])
    lane_x = float(cfg["lane_x"])
    layer_y = float(cfg["layer_y"])
    r_d = float(cfg["radius"])

    def pad_spec(note_item: dict[str, Any]) -> tuple[str, float, float]:
        if drum_idx == 0:
            return "drum_kick", 0.145, 0.052
        if int(note_item["note"]) == 38:
            return "drum_snare", 0.130, 0.044
        return "drum_hat", 0.110, 0.026

    def make_pad_obb(idx: int, hit_p, u_ax, n_ax, pad_r: float, pad_th: float, t_hit_abs: float) -> dict[str, Any]:
        contact = phys.vsub(hit_p, phys.vscale(n_ax, r_d))
        center = phys.vsub(contact, phys.vscale(n_ax, pad_th * 0.5))
        return {
            "index": idx,
            "t_hit": t_hit_abs,
            "center": center,
            "u": u_ax,
            "v": (0.0, 1.0, 0.0),
            "n": n_ax,
            "half_extents": (pad_r, pad_r, pad_th * 0.5),
            "pad_radius": pad_r,
            "pad_thick": pad_th,
        }

    # Pre-entry smooth guide chute (t=0 .. hit0)
    t_hit0 = float(notes[0]["start_time"]) + PRE_ROLL
    dt_drop0 = 0.24
    t_rel0 = t_hit0 - dt_drop0
    p = (-0.20 * side_sign, 0.0, z_ref(float(notes[0]["start_time"])))
    v_lip0 = (-0.80 * side_sign, 0.0, -1.65)
    p_lip0 = (
        p[0] - v_lip0[0] * dt_drop0,
        0.0,
        p[2] - (v_lip0[2] * dt_drop0 - 0.5 * GRAVITY * dt_drop0 * dt_drop0),
    )
    v_in = (v_lip0[0], 0.0, v_lip0[2] - GRAVITY * dt_drop0)
    sim_pre = _build_serpentine_glide_sim(
        (0.0, 0.0, 0.48), (0.0, 0.0, -0.45), p_lip0, v_lip0, t_rel0, side_sign, straight_head_sec=1.2
    )
    rid0 = f"d{drum_idx}_entry_guide"
    segments: list[dict[str, Any]] = [
        {"kind": "chute", "t0": 0.0, "t1": sim_pre["T"], "sim": sim_pre, "rail_id": rid0},
        {"kind": "ballistic", "t0": sim_pre["T"], "t1": t_hit0, "p0": p_lip0, "v0": v_lip0, "p1": p, "v1": v_in},
    ]
    rails: list[dict[str, Any]] = [
        {"id": rid0, "is_entry": True, "from_bounce": -1, "to_bounce": 0, "t0": 0.0, "t1": sim_pre["T"], "sim": sim_pre}
    ]
    recent_pads: list[dict[str, Any]] = []
    boards_data: list[dict[str, Any]] = []

    def arc_clear_drum(p0, v0, dur: float, t0_abs: float, self_pad: dict[str, Any] | None = None) -> bool:
        if dur <= 0.015:
            return True
        slice_p = recent_pads[-6:] + ([self_pad] if self_pad is not None else [])
        n_s = 20
        for sk in range(1, n_s):
            tau = dur * (sk / n_s)
            t_a = t0_abs + tau
            pos = (p0[0] + v0[0] * tau, 0.0, p0[2] + v0[2] * tau - 0.5 * GRAVITY * tau * tau)
            for rb in slice_p:
                if abs(t_a - rb["t_hit"]) < 0.036:
                    continue
                if phys.point_to_obb_dist(pos, rb["center"], rb["u"], rb["v"], rb["n"], rb["half_extents"]) - r_d < 0.008:
                    return False
        return True

    def find_hop_off_pad(idx: int, t0: float, pos_p, vel_in, dt_hop: float, e_cands, ang_cands, require_next: bool, dt_full: float, x_targ: float, z_targ: float, pad_r: float, pad_th: float):
        best = None
        best_sc = 1e9
        min_vn_out = 0.60 if require_next else 0.30
        for e in e_cands:
            for a_signed in ang_cands:
                res = phys.bounce_velocity(vel_in, math.radians(a_signed), e)
                if res is None:
                    continue
                v_out, vn_in, vn_out, u_b, n_b = res
                if vn_in >= -0.80 or vn_out < min_vn_out:
                    continue
                p_next = (
                    pos_p[0] + v_out[0] * dt_hop,
                    0.0,
                    pos_p[2] + v_out[2] * dt_hop - 0.5 * GRAVITY * dt_hop * dt_hop,
                )
                v_next = (v_out[0], 0.0, v_out[2] - GRAVITY * dt_hop)
                if require_next:
                    if v_next[2] >= -1.8 or abs(p_next[0]) > DRUM_HALF_X or (pos_p[2] - p_next[2]) < 0.14:
                        continue
                    if abs(p_next[2] - z_targ) > 2.00:
                        continue
                else:
                    if v_next[2] >= -0.10 or abs(p_next[0]) > DRUM_HALF_X:
                        continue
                cand_pad = make_pad_obb(idx, pos_p, u_b, n_b, pad_r, pad_th, t0)
                bad = False
                for rb in recent_pads[-6:]:
                    if phys.obb_overlap(cand_pad, rb, margin=0.025):
                        bad = True
                        break
                    if require_next and phys.point_to_obb_dist(p_next, rb["center"], rb["u"], rb["v"], rb["n"], rb["half_extents"]) < r_d + 0.018:
                        bad = True
                        break
                if bad:
                    continue
                if not arc_clear_drum(pos_p, v_out, dt_hop, t0, self_pad=cand_pad):
                    continue
                if require_next:
                    sc = 4.0 * abs(p_next[0] - x_targ) + 2.5 * abs(p_next[2] - z_targ) + 0.03 * abs(a_signed)
                else:
                    sc = 0.05 * abs(a_signed) + 1.5 * abs(p_next[0]) + 0.2 * abs(e - 0.28)
                if sc < best_sc:
                    best_sc = sc
                    best = (a_signed, e, u_b, n_b, v_out, p_next, v_next, cand_pad)
        return best

    fine_angs = [k * 0.5 for k in range(-76, 77)]

    def find_case_b_hop(idx: int, t0_h: float, pos_p, vel_in, dt_full: float, pad_r_h: float, pad_th_h: float):
        dt_enter_cands = (0.28, 0.32, 0.24, 0.36) if dt_full > 0.85 else (0.26, 0.24, 0.28, 0.22)
        best = None
        best_sc = 1e9
        xt = -0.20 if pos_p[0] >= 0.0 else 0.20
        for dt_e in dt_enter_cands:
            for e in (0.42, 0.34, 0.28, 0.22, 0.16, 0.50, 0.12):
                for a_signed in fine_angs:
                    res = phys.bounce_velocity(vel_in, math.radians(a_signed), e)
                    if res is None:
                        continue
                    v_out, vn_in, vn_out, u_b, n_b = res
                    if vn_in >= -0.80 or vn_out < 0.35:
                        continue
                    pe = (
                        pos_p[0] + v_out[0] * dt_e,
                        0.0,
                        pos_p[2] + v_out[2] * dt_e - 0.5 * GRAVITY * dt_e * dt_e,
                    )
                    ve = (v_out[0], 0.0, v_out[2] - GRAVITY * dt_e)
                    if ve[2] >= -0.80 or abs(pe[0]) > DRUM_HALF_X:
                        continue
                    cand_pad = make_pad_obb(idx, pos_p, u_b, n_b, pad_r_h, pad_th_h, t0_h)
                    if pe[2] > cand_pad["center"][2] - (r_d + pad_th_h * 0.5 + 0.08):
                        continue
                    clr_self = phys.point_to_obb_dist(
                        pe, cand_pad["center"], cand_pad["u"], cand_pad["v"], cand_pad["n"], cand_pad["half_extents"]
                    ) - r_d
                    if clr_self < 0.075:
                        continue
                    bad = False
                    for rb in recent_pads[-6:]:
                        if phys.obb_overlap(cand_pad, rb, margin=0.025):
                            bad = True
                            break
                        if phys.point_to_obb_dist(pe, rb["center"], rb["u"], rb["v"], rb["n"], rb["half_extents"]) - r_d < 0.060:
                            bad = True
                            break
                    if bad:
                        continue
                    if not arc_clear_drum(pos_p, v_out, dt_e, t0_h, self_pad=cand_pad):
                        continue
                    sc = 2.5 * abs(pe[0] - xt) + 0.03 * abs(a_signed) + 0.5 * abs(dt_e - 0.28)
                    if sc < best_sc:
                        best_sc = sc
                        best = (dt_e, a_signed, e, u_b, n_b, v_out, pe, ve, cand_pad)
            if best is not None:
                break
        return best

    for i in range(len(notes) - 1):
        t0 = float(notes[i]["start_time"]) + PRE_ROLL
        t1 = float(notes[i + 1]["start_time"]) + PRE_ROLL
        dt = t1 - t0
        z_targ_next = z_ref(float(notes[i + 1]["start_time"]))
        b_kind, pad_r, pad_th = pad_spec(notes[i])
        x_targ = -0.20 if p[0] >= 0.0 else 0.20

        if dt <= 0.65:
            best_b = find_hop_off_pad(
                i, t0, p, v_in, dt, (0.84, 0.74, 0.64, 0.56, 0.48, 0.40, 0.32, 0.24, 0.16),
                fine_angs, True, dt, x_targ, z_targ_next, pad_r, pad_th,
            )
            if best_b is None:
                raise RuntimeError(f"Drum {drum_idx} bounce failed at note {i} (dt={dt}, p={p}, z_targ={z_targ_next})")
            a_signed, e, u_b, n_b, v_out, p_next, v_next_in, cand_pad = best_b
            boards_data.append({
                "index": i, "hit": p, "u": u_b, "n": n_b, "obb": cand_pad,
                "e": e, "tilt_deg": a_signed, "is_terminal": False,
                "board_kind": b_kind, "pad_r": pad_r, "pad_th": pad_th, "note": notes[i],
            })
            recent_pads.append(cand_pad)
            segments.append({"kind": "ballistic", "t0": t0, "t1": t1, "p0": p, "v0": v_out, "p1": p_next, "v1": v_next_in})
            p, v_in = p_next, v_next_in
        else:
            best_h = find_case_b_hop(i, t0, p, v_in, dt, pad_r, pad_th)
            if best_h is None:
                raise RuntimeError(f"Drum {drum_idx} enter hop failed at note {i} (dt={dt}, p={p})")
            dt_enter, a_signed, e, u_b, n_b, v_out, p_enter, v_enter, cand_pad = best_h
            dt_drop = 0.16 if dt <= 0.90 else 0.24
            dt_rail = dt - dt_enter - dt_drop

            x_next = (-0.20 if p_enter[0] >= 0.0 else 0.20)
            v_exit_z = -1.20 if dt <= 0.90 else -1.65
            vx_exit = 0.78 * (1.0 if x_next >= 0.0 else -1.0)
            v_exit = (vx_exit, 0.0, v_exit_z)
            dz_drop = v_exit[2] * dt_drop - 0.5 * GRAVITY * dt_drop * dt_drop
            z_max_allowed = p_enter[2] + dz_drop - 0.40 * dt_rail
            z_next = min(z_targ_next, z_max_allowed)
            p_next = (x_next, 0.0, z_next)
            p_exit = (p_next[0] - v_exit[0] * dt_drop, 0.0, p_next[2] - dz_drop)
            v_next_in = (v_exit[0], 0.0, v_exit[2] - GRAVITY * dt_drop)

            sim_g = _build_serpentine_glide_sim(p_enter, v_enter, p_exit, v_exit, dt_rail, side_sign)
            rid = f"d{drum_idx}_guide_{i:03d}"
            t_enter = t0 + dt_enter
            t_exit = t_enter + sim_g["T"]
            boards_data.append({
                "index": i, "hit": p, "u": u_b, "n": n_b, "obb": cand_pad,
                "e": e, "tilt_deg": a_signed, "is_terminal": False,
                "board_kind": b_kind, "pad_r": pad_r, "pad_th": pad_th, "note": notes[i],
            })
            if dt > 1.2:
                recent_pads.clear()
            else:
                recent_pads.append(cand_pad)
            segments.append({"kind": "ballistic", "t0": t0, "t1": t_enter, "p0": p, "v0": v_out, "p1": p_enter, "v1": v_enter})
            segments.append({"kind": "chute", "t0": t_enter, "t1": t_exit, "sim": sim_g, "rail_id": rid})
            segments.append({"kind": "ballistic", "t0": t_exit, "t1": t1, "p0": p_exit, "v0": v_exit, "p1": p_next, "v1": v_next_in})
            rails.append({"id": rid, "is_entry": False, "from_bounce": i, "to_bounce": i + 1, "t0": t_enter, "t1": t_exit, "sim": sim_g})
            p, v_in = p_next, v_next_in

    # Final drum hit -> serpentine/smooth bronze guide chute down to Unified Finale Bridge & upward word leap
    last_i = len(notes) - 1
    b_kind_last, pad_r_last, pad_th_last = pad_spec(notes[last_i])
    t_last = float(notes[last_i]["start_time"]) + PRE_ROLL
    t_land = float(finale_target["t_pad"]) if finale_target else max(t_finale_land, t_last + 0.95)
    dt_end = t_land - t_last
    use_unified_bridge = bool(finale_target and finale_target.get("unified_bridge"))
    dt_drop = 0.34 if use_unified_bridge else 0.24
    best_hl = find_case_b_hop(last_i, t_last, p, v_in, dt_end, pad_r_last, pad_th_last)
    assert best_hl is not None, f"Drum {drum_idx} final hop failed"
    dt_enter, a_s, e_s, u_l, n_l, v_out_l, p_enter_l, v_enter_l, cand_pad_l = best_hl
    dt_rail = dt_end - dt_enter - dt_drop
    boards_data.append({
        "index": last_i, "hit": p, "u": u_l, "n": n_l, "obb": cand_pad_l,
        "e": e_s, "tilt_deg": a_s, "is_terminal": False,
        "board_kind": b_kind_last, "pad_r": pad_r_last, "pad_th": pad_th_last, "note": notes[last_i],
    })
    v_exit_z = -1.55
    dz_drop_f = v_exit_z * dt_drop - 0.5 * GRAVITY * dt_drop * dt_drop
    bridge_hits: list[dict[str, Any]] = []
    if use_unified_bridge:
        local_x_pad = float(finale_target["pad_x_world"]) - lane_x
        local_x_exit = float(finale_target.get("local_x_exit", 0.0))
        z_land_d = float(finale_target["z_top_bridge"]) + r_d
        p_land = (local_x_pad, 0.0 - layer_y, z_land_d)
        p_exit_f = (local_x_exit, 0.0, z_land_d - dz_drop_f)
        v_drop_3d = ((p_land[0] - p_exit_f[0]) / dt_drop, (p_land[1] - p_exit_f[1]) / dt_drop, v_exit_z)
        v_exit_f = (v_drop_3d[0], 0.0, v_exit_z)
        v_land = (v_drop_3d[0], v_drop_3d[1], v_exit_z - GRAVITY * dt_drop)
        bridge_hits.append({
            "ball_type": "drum",
            "ball_idx": drum_idx,
            "x": round(float(finale_target["pad_x_world"]), 5),
            "t_hit": round(t_land, 5),
            "frame_hit": int(round(t_land * FPS)),
            "rgb": cfg["color"],
        })
    else:
        if finale_target:
            local_x_pad = float(finale_target["pad_x_world"]) - lane_x
            p_land = (local_x_pad, 0.0, float(finale_target["z_ledge"]))
        else:
            p_land = (0.0, 0.0, min(z_end_drum, p_enter_l[2] - 0.55 * dt_end))
        v_exit_f = (0.0, 0.0, v_exit_z)
        p_exit_f = (p_land[0], 0.0, p_land[2] - dz_drop_f)
        v_drop_3d = v_exit_f
        v_land = (0.0, 0.0, v_exit_z - GRAVITY * dt_drop)

    sim_f = _build_serpentine_glide_sim(p_enter_l, v_enter_l, p_exit_f, v_exit_f, dt_rail, side_sign)
    rid_f = f"d{drum_idx}_finale_guide"
    t_ef = t_last + dt_enter
    t_xf = t_ef + sim_f["T"]
    segments.append({"kind": "ballistic", "t0": t_last, "t1": t_ef, "p0": p, "v0": v_out_l, "p1": p_enter_l, "v1": v_enter_l})
    segments.append({"kind": "chute", "t0": t_ef, "t1": t_xf, "sim": sim_f, "rail_id": rid_f})
    segments.append({"kind": "ballistic", "t0": t_xf, "t1": t_land, "p0": p_exit_f, "v0": v_drop_3d, "p1": p_land, "v1": v_land})
    if finale_target:
        dt_leap = float(finale_target["dt_leap"])
        w_world = finale_target["word_target_world"]
        p_word_local = (w_world[0] - lane_x, w_world[1] - layer_y, w_world[2])
        v0_leap, v1_leap = _parabolic_v0(p_land, p_word_local, dt_leap)
        t_morph = t_land + dt_leap
        segments.append({"kind": "ballistic", "t0": t_land, "t1": t_morph, "p0": p_land, "v0": v0_leap, "p1": p_word_local, "v1": v1_leap})
        segments.append({"kind": "rest", "t0": t_morph, "t1": t_end_total, "pos": p_word_local})
    else:
        t_morph = t_land
        segments.append({"kind": "rest", "t0": t_land, "t1": t_end_total, "pos": p_land})
    rails.append({"id": rid_f, "is_entry": False, "from_bounce": last_i, "to_bounce": last_i + 1, "t0": t_ef, "t1": t_xf, "sim": sim_f})

    if not use_unified_bridge:
        catch_r = 0.165 if finale_target else 0.210
        catch_th = 0.056 if finale_target else 0.062
        catch_obb = make_pad_obb(last_i + 1, p_land, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), catch_r, catch_th, t_land)
        boards_data.append({
            "index": last_i + 1, "hit": p_land, "u": (1.0, 0.0, 0.0), "n": (0.0, 0.0, 1.0), "obb": catch_obb,
            "e": 0.9 if finale_target else 0.0, "tilt_deg": 0.0, "is_terminal": True,
            "board_kind": "drum_catch", "pad_r": catch_r, "pad_th": catch_th, "note": notes[last_i],
        })

    def shift_xy(pt):
        return (round(pt[0] + lane_x, 5), round(pt[1] + layer_y, 5), round(pt[2], 5))

    boards_out = []
    for bd in boards_data:
        idx = bd["index"]
        obb = bd["obb"]
        is_term = bool(bd["is_terminal"])
        n_cur = bd["note"]
        t_hit_bd = obb["t_hit"]
        boards_out.append({
            "index": idx,
            "drum_idx": drum_idx,
            "is_terminal": is_term,
            "board_kind": bd["board_kind"],
            "pad_radius": bd["pad_r"],
            "t_hit": round(t_hit_bd, 5),
            "frame_hit": int(round(t_hit_bd * FPS)),
            "note": int(n_cur["note"]),
            "pitch_name": "END" if is_term else str(n_cur.get("pitch_name", bd["board_kind"])),
            "velocity": 110 if is_term else int(n_cur["velocity"]),
            "hit_center": shift_xy(bd["hit"]),
            "board_center": shift_xy(obb["center"]),
            "u": [round(x, 6) for x in obb["u"]],
            "n": [round(x, 6) for x in obb["n"]],
            "length": round(bd["pad_r"] * 2.0, 5),
            "width_y": round(bd["pad_r"] * 2.0, 5),
            "thickness": round(bd["pad_th"], 5),
        })

    rails_shifted = []
    for r in rails:
        rc = dict(r)
        sim_c = dict(r["sim"])
        sim_c["pts"] = [(pt[0] + lane_x, pt[1] + layer_y, pt[2]) for pt in r["sim"]["pts"]]
        rc["sim"] = sim_c
        rails_shifted.append(rc)
    rails_out = build_drum_guide_pipes(rails_shifted, r_d)

    return {
        "drum_idx": drum_idx,
        "key": cfg["key"],
        "name": cfg["name"],
        "side_sign": side_sign,
        "lane_x": lane_x,
        "layer_y": layer_y,
        "radius": r_d,
        "palette": {"name": cfg["name"], "color": cfg["color"], "glow": cfg["glow"]},
        "t_pad": round(t_land, 5),
        "t_morph": round(t_morph, 5),
        "word_id": finale_target["word_id"] if finale_target else None,
        "t_end": t_end_total,
        "land_pos": shift_xy(p_land),
        "bridge_hits": bridge_hits,
        "boards": boards_out,
        "rails": rails_out,
        "segments": segments,
    }


def _sample_ball_list(sols: list[dict], total_frames: int, is_drum: bool = False) -> list[dict]:
    out = []
    for sol in sols:
        samples = []
        for f_idx in range(total_frames + 1):
            t = f_idx / FPS
            pos, vel = sample_ball_at_t(sol, t)
            samples.append({
                "frame": f_idx,
                "t": round(t, 5),
                "position": [round(x, 5) for x in pos],
                "velocity": [round(x, 5) for x in vel],
            })
        entry = {
            "ball_idx": sol["drum_idx"] if is_drum else sol["ball_idx"],
            "lane_x": sol["lane_x"],
            "palette": sol["palette"],
            "board_count": len(sol["boards"]),
            "rail_count": len(sol["rails"]),
            "boards": sol["boards"],
            "rails": sol["rails"],
            "samples": samples,
        }
        if is_drum:
            entry["drum_idx"] = sol["drum_idx"]
            entry["key"] = sol["key"]
            entry["radius"] = sol["radius"]
            entry["side_sign"] = sol["side_sign"]
        out.append(entry)
    return out


def solve_multiball_plan(in_json: str, out_json: str) -> dict:
    with open(in_json, "r", encoding="utf-8") as f:
        src = json.load(f)

    is_ensemble = "melody_3ball" in src
    melody_balls_src = src["melody_3ball"]["balls"] if is_ensemble else src["balls"]

    raw_sols = []
    for b_obj in melody_balls_src:
        b_idx = int(b_obj["ball_idx"])
        raw_sols.append(solve_lane_trajectory(b_obj["notes"], b_idx, LANE_CENTERS_X[b_idx], LANE_LAYERS_Y[b_idx]))

    num_balls = len(raw_sols)
    z_end_min = min(s["p_last"][2] for s in raw_sols)
    z_end_base = z_end_min - 1.85
    finale_layout = compute_finale_word_layout(z_end_min) if is_ensemble else None
    finale_plan: dict[str, Any] | None = None

    ball_sols = []
    max_t = 0.0
    drum_sols = []

    if is_ensemble and finale_layout is not None:
        z_ledge = finale_layout["z_ledge"]
        z_top_bridge = finale_layout["z_top_bridge"]
        targs = finale_layout["targets"]
        melody_cfg = {
            0: {"word": "MARBLE", "pad_x": +0.05, "skips": [], "dt_leap": 0.44},
            1: {"word": "S",      "pad_x": +0.36, "skips": [((+0.50, 0.0, z_ledge), 0.52), ((+1.14, 0.0, z_ledge), 0.62)], "dt_leap": 0.46},
            2: {"word": "ON",     "pad_x": -0.48, "skips": [((-0.48, 0.0, z_ledge), 0.82)], "dt_leap": 0.42},
        }
        for s in raw_sols:
            b_idx = s["ball_idx"]
            mc = melody_cfg[b_idx]
            wt = targs[mc["word"]]
            fin = finalize_lane_with_finale_bounce(
                s,
                pad_x_target=mc["pad_x"],
                z_ledge=z_ledge,
                word_id=mc["word"],
                word_target_world=(wt["x"], wt["y"], wt["z"]),
                skip_hops_world=mc["skips"],
                dt_leap=mc["dt_leap"],
                unified_bridge=True,
            )
            ball_sols.append(fin)

        drum_cfg_fin = {
            0: {"word": "GLASS", "pad_x_world": -1.08, "local_x_exit": +0.14, "t_pad": 101.92, "dt_leap": 0.44},
            1: {"word": "WOOD",  "pad_x_world": +1.08, "local_x_exit": -0.14, "t_pad": 102.74, "dt_leap": 0.42},
        }
        t_morph_last = max(fin["t_morph"] for fin in ball_sols)
        rnb = src.get("rhythm_and_bass", {})
        for cfg in DRUM_LANES:
            df = drum_cfg_fin[int(cfg["drum_idx"])]
            t_morph_last = max(t_morph_last, df["t_pad"] + df["dt_leap"])

        max_t = t_morph_last + 3.50
        for fin in ball_sols:
            fin["t_end"] = max_t
            if fin["segments"][-1]["kind"] == "rest":
                fin["segments"][-1]["t1"] = max_t

        for cfg in DRUM_LANES:
            d_notes = rnb.get(cfg["key"], {}).get("notes", [])
            if not d_notes:
                continue
            d_i = int(cfg["drum_idx"])
            df = drum_cfg_fin[d_i]
            wt = targs[df["word"]]
            d_fin = {
                "unified_bridge": True,
                "word_id": df["word"],
                "pad_x_world": df["pad_x_world"],
                "local_x_exit": df["local_x_exit"],
                "z_ledge": z_ledge,
                "z_top_bridge": z_top_bridge,
                "t_pad": df["t_pad"],
                "dt_leap": df["dt_leap"],
                "word_target_world": (wt["x"], wt["y"], wt["z"]),
            }
            drum_sols.append(solve_drum_tower_lane(d_notes, cfg, z_end_base, df["t_pad"], max_t, finale_target=d_fin))

        all_bridge_hits = []
        for fin in ball_sols:
            all_bridge_hits.extend(fin.get("bridge_hits", []))
        for d_sol in drum_sols:
            all_bridge_hits.extend(d_sol.get("bridge_hits", []))
        all_bridge_hits.sort(key=lambda h: h["t_hit"])

        # Assemble sorted word ignition metadata
        words_meta = []
        for fin in ball_sols:
            wid = fin["word_id"]
            wt = targs[wid]
            words_meta.append({
                "id": wid,
                "text": wt["text"],
                "row": wt["row"],
                "pitch": wt["pitch"],
                "radius": wt["radius"],
                "x_center": wt["x"],
                "z_center": wt["z"],
                "ball_type": "melody",
                "ball_idx": fin["ball_idx"],
                "pad_t": fin["t_pad"],
                "morph_t": fin["t_morph"],
                "morph_frame": int(round(fin["t_morph"] * FPS)),
                "burst_rgb": fin["palette"]["color"],
            })
        for d_sol in drum_sols:
            wid = d_sol["word_id"]
            wt = targs[wid]
            words_meta.append({
                "id": wid,
                "text": wt["text"],
                "row": wt["row"],
                "pitch": wt["pitch"],
                "radius": wt["radius"],
                "x_center": wt["x"],
                "z_center": wt["z"],
                "ball_type": "drum",
                "ball_idx": d_sol["drum_idx"],
                "pad_t": d_sol["t_pad"],
                "morph_t": d_sol["t_morph"],
                "morph_frame": int(round(d_sol["t_morph"] * FPS)),
                "burst_rgb": d_sol["palette"]["color"],
            })
        words_meta.sort(key=lambda w: w["morph_t"])
        f_morph_s = next(w["morph_frame"] for w in words_meta if w["id"] == "S")
        finale_plan = {
            "z_finale": finale_layout["z_finale"],
            "z_row1": finale_layout["z_row1"],
            "z_row2": finale_layout["z_row2"],
            "z_row3": finale_layout["z_row3"],
            "z_ledge": finale_layout["z_ledge"],
            "y_morph": finale_layout["y_morph"],
            "row1_text": finale_layout["row1_text"],
            "row1_pitch": finale_layout["row1_pitch"],
            "row1_radius": finale_layout["row1_radius"],
            "row2_text": finale_layout["row2_text"],
            "row2_pitch": finale_layout["row2_pitch"],
            "row2_radius": finale_layout["row2_radius"],
            "t_cam_lock_start": 100.9,
            "t_cam_lock_full": 102.5,
            "credit_text": finale_layout["credit_text"],
            "credit_pitch": finale_layout["credit_pitch"],
            "credit_radius": finale_layout["credit_radius"],
            "credit_fade_start_frame": f_morph_s + 22,
            "credit_fade_end_frame": f_morph_s + 58,
            "celebration_frame": f_morph_s,
            "words": words_meta,
            "bridge": {
                "center": [0.0, 0.0, round(z_top_bridge - 0.044, 5)],
                "z_top": z_top_bridge,
                "length": 2.74,
                "width_y": 0.28,
                "thickness": 0.088,
                "hits": all_bridge_hits,
            },
        }
    else:
        # Stack termination boards vertically at center X=0.0 (back layer highest, front lowest so all 3 stay visible)
        z_step = 0.44
        for s in raw_sols:
            b_idx = s["ball_idx"]
            x_end_world = 0.0
            z_end_b = z_end_base + (b_idx - 0.5 * (num_balls - 1)) * z_step
            fin = finalize_lane_with_catch_board(s, x_end_world, z_end_b, num_balls)
            ball_sols.append(fin)
            max_t = max(max_t, fin["t_end"])

    total_frames = int(math.ceil(max_t * FPS))
    balls_out = _sample_ball_list(ball_sols, total_frames, is_drum=False)
    drum_balls_out = _sample_ball_list(drum_sols, total_frames, is_drum=True) if drum_sols else []

    # Measure max Z spread across melody balls and across all 5 balls
    all_sample_balls = [(b["samples"], MARBLE_RADIUS) for b in balls_out] + [
        (d["samples"], float(d["radius"])) for d in drum_balls_out
    ]
    max_z_spread = 0.0
    max_z_spread_all5 = 0.0
    for f_idx in range(int(PRE_ROLL * FPS), total_frames - int(POST_ROLL * FPS)):
        zs_mel = [b["samples"][f_idx]["position"][2] for b in balls_out]
        max_z_spread = max(max_z_spread, max(zs_mel) - min(zs_mel))
        zs_all = [smp[f_idx]["position"][2] for smp, _ in all_sample_balls]
        max_z_spread_all5 = max(max_z_spread_all5, max(zs_all) - min(zs_all))

    # Automated 5-ball clearance + jump continuity audit
    min_pair_clear = 1e9
    max_jump = 0.0
    for f_idx in range(total_frames + 1):
        for i_b in range(len(all_sample_balls)):
            smp_i, rad_i = all_sample_balls[i_b]
            pi = smp_i[f_idx]["position"]
            if f_idx > 0:
                pi_prev = smp_i[f_idx - 1]["position"]
                max_jump = max(
                    max_jump,
                    math.sqrt((pi[0] - pi_prev[0]) ** 2 + (pi[1] - pi_prev[1]) ** 2 + (pi[2] - pi_prev[2]) ** 2),
                )
            for j_b in range(i_b + 1, len(all_sample_balls)):
                smp_j, rad_j = all_sample_balls[j_b]
                pj = smp_j[f_idx]["position"]
                dist = math.sqrt((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2 + (pi[2] - pj[2]) ** 2)
                min_pair_clear = min(min_pair_clear, dist - rad_i - rad_j)

    assert min_pair_clear > 0.05, f"5-ball collision detected! min_pair_clear={min_pair_clear:.4f}m"
    assert max_jump < 0.25, f"Trajectory discontinuity detected! max_jump={max_jump:.4f}m/frame"

    # Exhaustive 3-way global audit: rail-vs-board, ball-vs-board, ball-vs-rail across ALL 5 balls
    all_audit_balls = [("M" + str(b["ball_idx"]), b, MARBLE_RADIUS) for b in balls_out] + [
        ("D" + str(d["drum_idx"]), d, float(d["radius"])) for d in drum_balls_out
    ]
    rb_clips = 0
    min_rb = 1e9
    for _tr, br, _ in all_audit_balls:
        for r in br["rails"]:
            pr = float(r["pipe_radius"])
            pts = r["pipe_front"][::2] + r["pipe_back"][::2]
            zlo = min(pt[2] for pt in pts) - 0.5
            zhi = max(pt[2] for pt in pts) + 0.5
            for _tb, bb, _ in all_audit_balls:
                for bd in bb["boards"]:
                    bc = bd["board_center"]
                    if not (zlo <= bc[2] <= zhi):
                        continue
                    u = tuple(bd["u"])
                    v = (0.0, 1.0, 0.0)
                    n = tuple(bd["n"])
                    he = (bd["length"] * 0.5, bd["width_y"] * 0.5, bd["thickness"] * 0.5)
                    md = min(phys.point_to_obb_dist(pt, tuple(bc), u, v, n, he) - pr for pt in pts)
                    if md < min_rb:
                        min_rb = md
                    if md < 0.015:
                        rb_clips += 1

    bb_clips = 0
    min_bb = 1e9
    for tag_m, bm, rad_m in all_audit_balls:
        samps = bm["samples"]
        for tag_b, bb, _ in all_audit_balls:
            for bd in bb["boards"]:
                bc = tuple(bd["board_center"])
                u = tuple(bd["u"])
                v = (0.0, 1.0, 0.0)
                n = tuple(bd["n"])
                he = (bd["length"] * 0.5, bd["width_y"] * 0.5, bd["thickness"] * 0.5)
                fh = int(bd["frame_hit"])
                hit_frames = [int(h) for h in (bd.get("hit_frames") or [fh])]
                is_term = bool(bd.get("is_terminal", False))
                x_lim = max(0.65, bd["length"] * 0.5 + 0.25)
                f_lo = max(0, min(hit_frames) - 150)
                f_hi = min(total_frames, max(hit_frames) + 150)
                for f in range(f_lo, f_hi + 1):
                    if tag_m == tag_b:
                        if any(abs(f - hf) <= 3 for hf in hit_frames):
                            continue
                        if is_term and not is_ensemble and f >= fh - 3:
                            continue
                    pos = samps[f]["position"]
                    if abs(pos[2] - bc[2]) > 0.65 or abs(pos[0] - bc[0]) > x_lim or abs(pos[1] - bc[1]) > 0.65:
                        continue
                    dd = phys.point_to_obb_dist(tuple(pos), bc, u, v, n, he) - rad_m
                    if dd < min_bb:
                        min_bb = dd
                    if dd < -0.002:
                        bb_clips += 1

    br_clips = 0
    min_br = 1e9
    for tag_m, bm, rad_m in all_audit_balls:
        samps = bm["samples"]
        for tag_r, br, _ in all_audit_balls:
            for r in br["rails"]:
                pr = float(r["pipe_radius"])
                rf0 = int(round(r["t0"] * FPS))
                rf1 = int(round(r["t1"] * FPS))
                pts = r["pipe_front"] + r["pipe_back"]
                zlo = min(pt[2] for pt in pts) - 0.35
                zhi = max(pt[2] for pt in pts) + 0.35
                f_lo = max(0, rf0 - 150)
                f_hi = min(total_frames, rf1 + 150)
                for f in range(f_lo, f_hi + 1):
                    if tag_m == tag_r and (f <= 2 or (rf0 - 4 <= f <= rf1 + 4)):
                        continue
                    pos = samps[f]["position"]
                    if not (zlo <= pos[2] <= zhi):
                        continue
                    dd = min(
                        math.sqrt((pos[0] - pt[0]) ** 2 + (pos[1] - pt[1]) ** 2 + (pos[2] - pt[2]) ** 2) - rad_m - pr
                        for pt in pts[::2]
                    )
                    if dd < min_br:
                        min_br = dd
                    if dd < -0.002:
                        br_clips += 1

    bridge_clips = 0
    min_bridge = 1e9
    if finale_plan is not None and finale_plan.get("bridge"):
        br_spec = finale_plan["bridge"]
        br_c = tuple(br_spec["center"])
        br_he = (br_spec["length"] * 0.5, br_spec["width_y"] * 0.5, br_spec["thickness"] * 0.5)
        u_id, v_id, n_id = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        f_br0 = max(0, min(h["frame_hit"] for h in br_spec["hits"]) - 150)
        for tag_m, bm, rad_m in all_audit_balls:
            is_d_m = tag_m.startswith("D")
            b_idx_m = int(tag_m[1:])
            my_hits = [
                h["frame_hit"] for h in br_spec["hits"]
                if (h["ball_type"] == ("drum" if is_d_m else "melody") and int(h["ball_idx"]) == b_idx_m)
            ]
            samps = bm["samples"]
            for f in range(f_br0, total_frames + 1):
                if any(abs(f - hf) <= 3 for hf in my_hits):
                    continue
                pos = samps[f]["position"]
                dd = phys.point_to_obb_dist(tuple(pos), br_c, u_id, v_id, n_id, br_he) - rad_m
                if dd < min_bridge:
                    min_bridge = dd
                if dd < -0.002:
                    bridge_clips += 1

    assert rb_clips == 0, f"Global rail-vs-board clip! rb_clips={rb_clips}, min_rb={min_rb:.4f}m"
    assert bb_clips == 0, f"Global ball-vs-board clip! bb_clips={bb_clips}, min_bb={min_bb:.4f}m"
    assert br_clips == 0, f"Global ball-vs-rail clip! br_clips={br_clips}, min_br={min_br:.4f}m"
    assert bridge_clips == 0, f"Global ball-vs-bridge clip! bridge_clips={bridge_clips}, min_bridge={min_bridge:.4f}m"
    print(f"  [Global Audit PASS] rb_clips={rb_clips} (min={min_rb:.4f}m), bb_clips={bb_clips} (min={min_bb:.4f}m), br_clips={br_clips} (min={min_br:.4f}m), bridge_clips={bridge_clips} (min={min_bridge:.4f}m)")

    plan: dict[str, Any] = {
        "schema": "marble_ensemble_16x9_v1" if is_ensemble else "marble_multiball_16x9_v1",
        "fps": FPS,
        "pre_roll": PRE_ROLL,
        "post_roll": POST_ROLL,
        "gravity": GRAVITY,
        "marble_radius": MARBLE_RADIUS,
        "back_wall_y": BACK_WALL_Y_MB,
        "total_duration": round(max_t, 4),
        "total_frames": total_frames,
        "num_balls": len(balls_out) + len(drum_balls_out),
        "num_melody_balls": len(balls_out),
        "num_drum_balls": len(drum_balls_out),
        "max_z_spread": round(max_z_spread, 4),
        "max_z_spread_all5": round(max_z_spread_all5, 4),
        "min_pair_clear_m": round(min_pair_clear, 4),
        "max_frame_jump_m": round(max_jump, 4),
        "balls": balls_out,
    }

    if is_ensemble:
        rnb = src.get("rhythm_and_bass", {})
        bass_events = []
        for idx_b, it in enumerate(rnb.get("Bass_Resonator", {}).get("notes", [])):
            ta = float(it["start_time"]) + PRE_ROLL
            bass_events.append({
                "index": idx_b,
                "t_anim": round(ta, 5),
                "frame_hit": int(round(ta * FPS)),
                "note": int(it["note"]),
                "pitch_name": str(it["pitch_name"]),
                "velocity": int(it["velocity"]),
            })
        perc_events = []
        for idx_p, it in enumerate(rnb.get("Percussion_Lights", {}).get("notes", [])):
            ta = float(it["start_time"]) + PRE_ROLL
            perc_events.append({
                "index": idx_p,
                "t_anim": round(ta, 5),
                "frame_hit": int(round(ta * FPS)),
                "note": int(it["note"]),
                "velocity": int(it["velocity"]),
            })
        sections = [
            {"name": "Intro", "t0": 0.0, "t1": 15.4, "night_mix": 0.85, "lens": 41.0, "cam_y": -5.45},
            {"name": "Drop 1", "t0": 15.4, "t1": 40.4, "night_mix": 0.0, "lens": 37.5, "cam_y": -5.65},
            {"name": "Breakdown 1", "t0": 40.4, "t1": 50.9, "night_mix": 0.80, "lens": 41.5, "cam_y": -5.42},
            {"name": "Drop 2", "t0": 50.9, "t1": 70.9, "night_mix": 0.0, "lens": 37.5, "cam_y": -5.65},
            {"name": "Bridge", "t0": 70.9, "t1": 81.4, "night_mix": 0.90, "lens": 42.0, "cam_y": -5.40},
            {"name": "Climax", "t0": 81.4, "t1": 102.2, "night_mix": 0.0, "lens": 37.0, "cam_y": -5.75},
            {"name": "Finale", "t0": 102.2, "t1": round(max_t, 4), "night_mix": 0.72, "lens": 39.0, "cam_y": -5.95},
        ]
        plan["drum_balls"] = drum_balls_out
        plan["bass_events"] = bass_events
        plan["percussion_events"] = perc_events
        plan["sections"] = sections
        if finale_plan is not None:
            plan["finale"] = finale_plan

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    src_counts = {}
    for b in balls_out:
        for bd in b["boards"]:
            st = bd.get("source_stem", "unknown")
            src_counts[st] = src_counts.get(st, 0) + 1

    print(
        f"[{os.path.basename(out_json)}] total_balls={plan['num_balls']} "
        f"(melody={len(balls_out)}, drum={len(drum_balls_out)}), frames={total_frames}, "
        f"dur={plan['total_duration']}s, melody_boards={[b['board_count'] for b in balls_out]}, "
        f"drum_boards={[d['board_count'] for d in drum_balls_out]}, "
        f"max_z_spread={max_z_spread:.3f}m, all5_spread={max_z_spread_all5:.3f}m, "
        f"min_pair_clear={min_pair_clear:.4f}m, max_jump={max_jump:.4f}m, src={src_counts}"
    )
    return plan


if __name__ == "__main__":
    in_j = sys.argv[1] if len(sys.argv) >= 2 else "projects/jingle-bells-marble-music/artifacts/glass_marbles_synth_3ball_notes.json"
    out_j = sys.argv[2] if len(sys.argv) >= 3 else "projects/jingle-bells-marble-music/artifacts/glass_marbles_synth_3ball_16x9_physics.json"
    solve_multiball_plan(in_j, out_j)
