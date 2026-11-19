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
Z_REF_RATE = -2.10  # Crisp vertical drop rate (~0.50m per 8th note) keeping boards & rails cleanly separated

BALL_PALETTES = [
    {"name": "EmeraldLead", "color": [0.12, 0.92, 0.48], "glow": [0.18, 1.00, 0.55]},
    {"name": "AmberAlt", "color": [1.00, 0.62, 0.14], "glow": [1.00, 0.68, 0.20]},
    {"name": "CyanHarmony", "color": [0.18, 0.78, 1.00], "glow": [0.24, 0.85, 1.00]},
]


def z_ref(t_after_preroll: float) -> float:
    return Z_REF_RATE * max(0.0, t_after_preroll)


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
    entry = phys.solve_entry_ramp((-0.42, 0.0, z0_hit), (1.30, 0.0, -2.90))
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
            e_cands = (0.72, 0.64, 0.80, 0.56, 0.48, 0.86, 0.38)
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
                            + 1.6 * abs(p_next[2] - z_targ_next)
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
        "boards_data": boards_data,
        "recent_boards": recent_boards[-10:],
        "recent_rails": recent_rails_pts[-4:],
        "rails": rails,
        "segments": segments,
        "make_board_obb": make_board_obb,
    }


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
    for bd in boards_data:
        idx = bd["index"]
        obb = bd["obb"]
        is_term = bool(bd.get("is_terminal", False))
        t_hit_bd = obb["t_hit"]
        boards_out.append({
            "index": idx,
            "ball_idx": ball_idx,
            "is_terminal": is_term,
            "t_hit": round(t_hit_bd, 5),
            "frame_hit": int(round(t_hit_bd * FPS)),
            "track_idx": ball_idx + 1,
            "instrument": "Terminal Catch" if is_term else last_note["instrument"],
            "instrument_type": "terminal_catch" if is_term else "clean_bar",
            "note": int(last_note["note"]),
            "pitch_name": "END" if is_term else last_note["pitch_name"],
            "velocity": 110 if is_term else int(last_note["velocity"]),
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


def solve_multiball_plan(in_json: str, out_json: str) -> dict:
    with open(in_json, "r", encoding="utf-8") as f:
        src = json.load(f)

    raw_sols = []
    for b_obj in src["balls"]:
        b_idx = int(b_obj["ball_idx"])
        raw_sols.append(solve_lane_trajectory(b_obj["notes"], b_idx, LANE_CENTERS_X[b_idx], LANE_LAYERS_Y[b_idx]))

    num_balls = len(raw_sols)
    z_end_base = min(s["p_last"][2] for s in raw_sols) - 1.85
    # Stack termination boards vertically at center X=0.0 (back layer highest, front lowest so all 3 stay visible)
    z_step = 0.44
    ball_sols = []
    max_t = 0.0
    for s in raw_sols:
        b_idx = s["ball_idx"]
        x_end_world = 0.0
        z_end_b = z_end_base + (b_idx - 0.5 * (num_balls - 1)) * z_step
        fin = finalize_lane_with_catch_board(s, x_end_world, z_end_b, num_balls)
        ball_sols.append(fin)
        max_t = max(max_t, fin["t_end"])

    total_frames = int(math.ceil(max_t * FPS))
    # Sample every frame for all 3 balls
    balls_out = []
    for sol in ball_sols:
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
        balls_out.append({
            "ball_idx": sol["ball_idx"],
            "lane_x": sol["lane_x"],
            "palette": sol["palette"],
            "board_count": len(sol["boards"]),
            "rail_count": len(sol["rails"]),
            "boards": sol["boards"],
            "rails": sol["rails"],
            "samples": samples,
        })

    # Measure max Z spread across the 3 balls during active play
    max_z_spread = 0.0
    for f_idx in range(int(PRE_ROLL * FPS), total_frames - int(POST_ROLL * FPS)):
        zs = [b["samples"][f_idx]["position"][2] for b in balls_out]
        max_z_spread = max(max_z_spread, max(zs) - min(zs))

    plan = {
        "schema": "marble_multiball_16x9_v1",
        "fps": FPS,
        "pre_roll": PRE_ROLL,
        "post_roll": POST_ROLL,
        "gravity": GRAVITY,
        "marble_radius": MARBLE_RADIUS,
        "back_wall_y": BACK_WALL_Y_MB,
        "total_duration": round(max_t, 4),
        "total_frames": total_frames,
        "num_balls": len(balls_out),
        "max_z_spread": round(max_z_spread, 4),
        "balls": balls_out,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(
        f"[{os.path.basename(out_json)}] balls={len(balls_out)}, frames={total_frames}, "
        f"dur={plan['total_duration']}s, boards={[b['board_count'] for b in balls_out]}, "
        f"rails={[b['rail_count'] for b in balls_out]}, max_z_spread={max_z_spread:.3f}m"
    )
    return plan


if __name__ == "__main__":
    in_j = sys.argv[1] if len(sys.argv) >= 2 else "projects/jingle-bells-marble-music/artifacts/glass_marbles_synth_3ball_notes.json"
    out_j = sys.argv[2] if len(sys.argv) >= 3 else "projects/jingle-bells-marble-music/artifacts/glass_marbles_synth_3ball_16x9_physics.json"
    solve_multiball_plan(in_j, out_j)
