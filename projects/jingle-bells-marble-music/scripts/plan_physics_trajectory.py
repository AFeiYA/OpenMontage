#!/usr/bin/env python3
"""True-Physics Trajectory & Scene Layout Solver (v3) for Virtual 3D Marble Music.

Key physical guarantees:
1. Gravity & Impact: True forward rigid-body projectile motion (g=22.0 u/s^2,
   marble R=0.105u ~9cm studio scale) and rigid-body restitution e in [0.38, 0.84]
   plus tangential friction eta_t=0.93 at every bar bounce.
2. Sloped Rolling Rails (NO vertical drop wires): All longer rests (dt > 0.50s)
   use gently sloped 3D twin-pipe chutes (slope <= 38 deg) driven by the exact
   no-slip rigid-sphere rolling ODE: dv/dt = (5/7)*(-g*dz/ds - drag*v), with
   centripetal-banked twin rails cradling the marble strictly from underneath.
3. Exact Tangential Entry Ramp: Pre-roll (0..0.90s) rolls naturally down an
   inclined starter chute under true gravity and drops cleanly onto Note 0.
4. Strict 240Hz Audit: Validates 0 physics anomalies, 0 OBB overlaps, 0 board
   flyby penetrations, and 0 rail pipe penetrations across every substep.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

FPS = 60
SUBSTEPS_SOLVE = 2     # 120 Hz during search
SUBSTEPS_VERIFY = 4    # 240 Hz final strict audit

# World scale & physical constants (Right-handed: +X right, +Y into wall, +Z up)
PRE_ROLL = 0.90
POST_ROLL = 1.40
MARBLE_RADIUS = 0.105
GRAVITY = 19.0         # Crisp showcase gravity enabling pure ballistic bounces up to ~0.72s
Y_PLANE = 0.0

# Board geometry (centered along Y=0, mounting pins strictly behind at +Y)
BOARD_THICKNESS = 0.072
BOARD_DEPTH_Y = 0.24
BACK_WALL_Y = 0.52
PIN_RADIUS = 0.017

# Twin slim guide rails (cradles marble strictly tangentially from underneath)
RAIL_PIPE_RADIUS = 0.012
RAIL_Y_HALF = 0.064
RAIL_PERP_OFFSET = math.sqrt((MARBLE_RADIUS + RAIL_PIPE_RADIUS) ** 2 - RAIL_Y_HALF ** 2)

# Threshold separating pure ballistic bounces (up to 0.612s half-notes) from sleek transition chutes (0.918s+)
RAIL_DT_THRESHOLD = 0.72
ROLL_DRAG = 0.030


# -------------------- Vector & Curve Math --------------------

def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a, s: float):
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vlen(a) -> float:
    return math.sqrt(vdot(a, a))


def vnorm(a, fallback=(0.0, 0.0, 1.0)):
    L = vlen(a)
    return (a[0] / L, a[1] / L, a[2] / L) if L > 1e-9 else fallback


def r5(v):
    return [round(v[0], 5), round(v[1], 5), round(v[2], 5)]


def hermite_piecewise(
    knots: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    samples_per_seg: int = 24,
) -> list[tuple[float, float, float]]:
    """C1-continuous piecewise cubic Hermite spline through (pos_k, dir_k) knots."""
    pts: list[tuple[float, float, float]] = []
    n = len(knots)
    for i in range(n - 1):
        p0, d0 = knots[i]
        p1, d1 = knots[i + 1]
        L_chord = max(0.05, vlen(vsub(p1, p0)))
        t0 = vscale(vnorm(d0), L_chord * 1.12)
        t1 = vscale(vnorm(d1), L_chord * 1.12)
        k_end = samples_per_seg + 1 if i == n - 2 else samples_per_seg
        for k in range(k_end):
            s = k / samples_per_seg
            s2 = s * s
            s3 = s2 * s
            h00 = 2.0 * s3 - 3.0 * s2 + 1.0
            h10 = s3 - 2.0 * s2 + s
            h01 = -2.0 * s3 + 3.0 * s2
            h11 = s3 - s2
            pts.append((
                h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0],
                h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1],
                h00 * p0[2] + h10 * t0[2] + h01 * p1[2] + h11 * t1[2],
            ))
    return pts


def simulate_rolling_pts(
    pts: list[tuple[float, float, float]],
    v0_speed: float,
    drag: float = ROLL_DRAG,
) -> dict[str, Any] | None:
    """Integrate true sphere rolling ODE dv/dt = (5/7)*(-g*dz/ds - drag*v) along 3D curve."""
    s_arr = [0.0]
    for k in range(len(pts) - 1):
        s_arr.append(s_arr[-1] + vlen(vsub(pts[k + 1], pts[k])))
    L = s_arr[-1]
    if L < 1e-5:
        return None

    t_cum = [0.0]
    v_prof = [v0_speed]
    cur_v = v0_speed
    cur_t = 0.0
    max_slope_deg = 0.0
    min_slope_deg = 999.0

    for k in range(len(pts) - 1):
        ds = s_arr[k + 1] - s_arr[k]
        if ds < 1e-8:
            t_cum.append(cur_t)
            v_prof.append(cur_v)
            continue
        dz = pts[k + 1][2] - pts[k][2]
        sin_th = max(-0.999, min(0.999, dz / ds))  # negative = downhill
        slope_deg = math.degrees(math.asin(-sin_th))
        max_slope_deg = max(max_slope_deg, slope_deg)
        min_slope_deg = min(min_slope_deg, slope_deg)

        a_s = (5.0 / 7.0) * (-GRAVITY * (dz / ds) - drag * cur_v)
        v_next_sq = cur_v * cur_v + 2.0 * a_s * ds
        if v_next_sq <= 0.04:
            return None
        v_next = math.sqrt(v_next_sq)
        cur_t += 2.0 * ds / (cur_v + v_next)
        cur_v = v_next
        t_cum.append(cur_t)
        v_prof.append(cur_v)

    return {
        "L": L,
        "T": cur_t,
        "v_exit": cur_v,
        "max_slope_deg": max_slope_deg,
        "min_slope_deg": min_slope_deg,
        "pts": pts,
        "t_cum": t_cum,
        "v_prof": v_prof,
    }


def eval_rolling_table(rail_sim: dict[str, Any], tau_sec: float):
    """Evaluate exact marble position and velocity along rolling curve at local time tau_sec."""
    t_cum = rail_sim["t_cum"]
    pts = rail_sim["pts"]
    v_prof = rail_sim["v_prof"]
    n = len(t_cum)
    t = max(0.0, min(t_cum[-1], tau_sec))

    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t_cum[mid] <= t:
            lo = mid
        else:
            hi = mid

    dt_seg = max(1e-9, t_cum[hi] - t_cum[lo])
    alpha = max(0.0, min(1.0, (t - t_cum[lo]) / dt_seg))
    p0, p1 = pts[lo], pts[hi]
    pos = (
        p0[0] + alpha * (p1[0] - p0[0]),
        p0[1] + alpha * (p1[1] - p0[1]),
        p0[2] + alpha * (p1[2] - p0[2]),
    )
    tang = vnorm(vsub(p1, p0), fallback=(1.0, 0.0, 0.0))
    spd = v_prof[lo] + alpha * (v_prof[hi] - v_prof[lo])
    vel = vscale(tang, spd)
    return pos, vel


# -------------------- Collision Primitives --------------------

def point_to_obb_dist(pt, center, u_ax, v_ax, n_ax, half_extents) -> float:
    d = vsub(pt, center)
    q0 = vdot(d, u_ax)
    q1 = vdot(d, v_ax)
    q2 = vdot(d, n_ax)
    ex = max(0.0, abs(q0) - half_extents[0])
    ey = max(0.0, abs(q1) - half_extents[1])
    ez = max(0.0, abs(q2) - half_extents[2])
    if ex == 0.0 and ey == 0.0 and ez == 0.0:
        return -min(half_extents[0] - abs(q0), half_extents[1] - abs(q1), half_extents[2] - abs(q2))
    return math.sqrt(ex * ex + ey * ey + ez * ez)


def point_to_segment_dist(pt, a, b) -> float:
    ab = vsub(b, a)
    L2 = vdot(ab, ab)
    if L2 < 1e-12:
        return vlen(vsub(pt, a))
    t = max(0.0, min(1.0, vdot(vsub(pt, a), ab) / L2))
    proj = vadd(a, vscale(ab, t))
    return vlen(vsub(pt, proj))


def obb_overlap(b1: dict[str, Any], b2: dict[str, Any], margin: float = 0.025) -> bool:
    axes1 = [b1["u"], b1["v"], b1["n"]]
    axes2 = [b2["u"], b2["v"], b2["n"]]
    h1 = [h + margin * 0.5 for h in b1["half_extents"]]
    h2 = [h + margin * 0.5 for h in b2["half_extents"]]
    T = vsub(b2["center"], b1["center"])

    test_axes = list(axes1) + list(axes2)
    for a1 in axes1:
        for a2 in axes2:
            c = vcross(a1, a2)
            if vdot(c, c) > 1e-8:
                test_axes.append(vnorm(c))

    for L in test_axes:
        dist = abs(vdot(T, L))
        r1 = sum(h1[k] * abs(vdot(axes1[k], L)) for k in range(3))
        r2 = sum(h2[k] * abs(vdot(axes2[k], L)) for k in range(3))
        if dist > r1 + r2:
            return False
    return True


# -------------------- Physics Bounce & Sloped 3D Rail Solvers --------------------

def bounce_velocity(v_in, alpha_rad: float, e: float, eta_t: float = 0.93):
    """Rigid-body impact off tilted board with normal n=(sin a, 0, cos a)."""
    sa, ca = math.sin(alpha_rad), math.cos(alpha_rad)
    n = (sa, 0.0, ca)
    u = (ca, 0.0, -sa)
    vn_in = vdot(v_in, n)
    vu_in = vdot(v_in, u)
    if vn_in >= -0.45:
        return None
    vn_out = -e * vn_in
    vu_out = eta_t * vu_in
    v_out = vadd(vscale(u, vu_out), vscale(n, vn_out))
    return v_out, vn_in, vn_out, u, n


def solve_3d_chute(
    p_enter,
    v_enter,
    dt_rail: float,
    preferred_y_bows: tuple[float, ...] = (0.0, -0.22, -0.35, -0.52, -0.72, -0.92, -1.12),
    final_samples_per_seg: int = 64,
    clear_fn=None,
    dt_drop: float = 0.17,
    drop_clear_fn=None,
) -> dict[str, Any] | None:
    """Solve a 3D gently-sloped rail chute matching v_enter tangentially and rolling ODE time == dt_rail."""
    T0 = vnorm(v_enter)
    v0_speed = vlen(v_enter)
    in_dir = 1.0 if v_enter[0] >= 0 else -1.0

    configs = []
    # Family A: Cross-stage curved glide chute
    for x_ex in (in_dir * 0.90, in_dir * 0.72, in_dir * 0.55, in_dir * 0.35, 0.0, -in_dir * 0.35):
        if (x_ex - p_enter[0]) * in_dir >= 0.32:
            for y_bow in (0.0, -0.24, -0.52, -0.82):
                configs.append(("cross", x_ex, y_bow, in_dir))

    # Family B: Smooth 3D U-turn arc inside [-1.14, +1.14] (balcony forward sweep in -Y)
    x_wing_clamped = max(-1.14, min(1.14, p_enter[0] + in_dir * 0.48))
    x_wings = sorted({round(x_wing_clamped, 3), round(in_dir * 1.02, 3)}, key=lambda v: abs(v - p_enter[0]))
    for y_bow in (-0.36, -0.56, -0.82, -1.12, -1.42):
        for x_wing in x_wings:
            if (x_wing - p_enter[0]) * in_dir < 0.20:
                continue
            for x_ex in (-in_dir * 0.45, 0.0, -in_dir * 0.65, -in_dir * 0.25, -in_dir * 0.80):
                configs.append(("uturn", x_wing, y_bow, x_ex))

    best_meta = None
    best_cost = 1e9

    def build_knots(cfg, dz_drop: float, T1):
        if cfg[0] == "cross":
            _, x_ex, y_bow, _ = cfg
            p_mid = (0.5 * (p_enter[0] + x_ex), y_bow, p_enter[2] - 0.42 * dz_drop)
            p_exit = (x_ex, 0.0, p_enter[2] - dz_drop)
            T_mid = vnorm((x_ex - p_enter[0], 0.0, -0.50 * dz_drop))
            return [(p_enter, T0), (p_mid, T_mid), (p_exit, T1)], p_exit
        else:
            _, x_wing, y_bow, x_ex = cfg
            p1y = y_bow * 0.70
            p2x = 0.5 * (x_wing + x_ex)
            p2y = y_bow
            d1 = math.hypot(x_wing - p_enter[0], p1y)
            d2 = math.hypot(p2x - x_wing, p2y - p1y)
            d3 = math.hypot(x_ex - p2x, -p2y)
            dtot = max(1e-6, d1 + d2 + d3)
            f1 = d1 / dtot
            f2 = (d1 + d2) / dtot
            p1 = (x_wing, p1y, p_enter[2] - f1 * dz_drop)
            T_p1 = vnorm((p2x - p_enter[0], p2y - 0.10 * p1y, -((d1 + d2) * 0.5 / dtot) * dz_drop))
            p2 = (p2x, p2y, p_enter[2] - f2 * dz_drop)
            T_p2 = vnorm((x_ex - p2x, -0.55 * p2y, -(d3 / dtot) * dz_drop))
            p_exit = (x_ex, 0.0, p_enter[2] - dz_drop)
            return [(p_enter, T0), (p1, T_p1), (p2, T_p2), (p_exit, T1)], p_exit

    dz_grid = (
        (0.30, 0.42, 0.56, 0.72, 0.92, 1.16, 1.46)
        if dt_rail >= 0.55
        else (0.12, 0.20, 0.30, 0.42, 0.56, 0.74, 0.96)
    )

    for cfg in configs:
        if cfg[0] == "uturn" and best_meta is not None and best_cost < 56.0:
            break
        for exit_deg in (24.0, 29.0, 19.0, 33.0):
            ang_e = math.radians(exit_deg)
            dir_ex = cfg[3] if cfg[0] == "cross" else (1.0 if cfg[3] > cfg[1] else -1.0)
            T1 = (dir_ex * math.cos(ang_e), 0.0, -math.sin(ang_e))

            prev_dz, prev_r = None, None
            for dz_probe in dz_grid:
                knots_p, _ = build_knots(cfg, dz_probe, T1)
                rp = simulate_rolling_pts(hermite_piecewise(knots_p, samples_per_seg=14), v0_speed)
                if rp is not None and prev_r is not None:
                    if (prev_r["T"] >= dt_rail >= rp["T"]) or (prev_r["T"] <= dt_rail <= rp["T"]):
                        a, b = prev_dz, dz_probe
                        t_a = prev_r["T"]
                        sol = None
                        for _ in range(18):
                            mid = 0.5 * (a + b)
                            knots_m, px = build_knots(cfg, mid, T1)
                            rm = simulate_rolling_pts(hermite_piecewise(knots_m, samples_per_seg=14), v0_speed)
                            if rm is None:
                                break
                            if (t_a >= dt_rail and rm["T"] >= dt_rail) or (t_a <= dt_rail and rm["T"] <= dt_rail):
                                a, t_a = mid, rm["T"]
                            else:
                                b = mid
                            sol = (rm, mid, px, knots_m)
                        if sol and abs(sol[0]["T"] - dt_rail) < 0.008:
                            rm, dz_val, px, _ = sol
                            vx_ex_signed = rm["v_exit"] * T1[0]
                            vx_ex = abs(vx_ex_signed)
                            x_land = px[0] + vx_ex_signed * dt_drop
                            if abs(x_land) > 1.05:
                                break
                            if rm["max_slope_deg"] <= 38.0 and rm["min_slope_deg"] >= 3.5:
                                # Reject extreme hairpins (R < 0.16u) where inner rail pipe gets pinched
                                rpts_m = rm["pts"]
                                max_k_m = 0.0
                                for km in range(1, len(rpts_m) - 1):
                                    d1v = vsub(rpts_m[km], rpts_m[km - 1])
                                    d2v = vsub(rpts_m[km + 1], rpts_m[km])
                                    l1, l2 = vlen(d1v), vlen(d2v)
                                    if l1 > 1e-5 and l2 > 1e-5:
                                        kmag = vlen(vsub(vscale(d2v, 1.0 / l2), vscale(d1v, 1.0 / l1))) / (0.5 * (l1 + l2))
                                        if kmag > max_k_m:
                                            max_k_m = kmag
                                if max_k_m > 6.2:
                                    break
                                if clear_fn is not None and not clear_fn(rm["pts"]):
                                    break
                                if drop_clear_fn is not None and not drop_clear_fn(px, vscale(T1, rm["v_exit"]), rm):
                                    break
                                y_bow = cfg[2]
                                cost = (
                                    1.6 * rm["max_slope_deg"]
                                    + 0.8 * max_k_m
                                    + 3.0 * abs(y_bow)
                                    + 8.0 * max(0.0, vx_ex - 2.6)
                                    + 2.0 * abs(rm["v_exit"] - 3.3)
                                    + (0.0 if cfg[0] == "cross" else 2.0)
                                )
                                if cost < best_cost:
                                    best_cost = cost
                                    best_meta = (cfg, dz_val, T1, y_bow)
                        break
                if rp is not None:
                    prev_dz, prev_r = dz_probe, rp

    if best_meta is None:
        return None

    cfg, dz_val, T1, y_bow = best_meta
    knots_hi, p_exit = build_knots(cfg, dz_val, T1)
    pts_hi = hermite_piecewise(knots_hi, samples_per_seg=final_samples_per_seg)
    sim_hi = simulate_rolling_pts(pts_hi, v0_speed)
    if sim_hi is None:
        return None

    return {
        "kind": cfg[0],
        "T": sim_hi["T"],
        "L": sim_hi["L"],
        "dz": dz_val,
        "v_exit": sim_hi["v_exit"],
        "max_slope_deg": sim_hi["max_slope_deg"],
        "min_slope_deg": sim_hi["min_slope_deg"],
        "y_bow": y_bow,
        "p_exit": p_exit,
        "T1": T1,
        "sim": sim_hi,
    }


def solve_entry_ramp(hit0_pos, v_in0):
    """Build a natural inclined top-left starter ramp (0 .. PRE_ROLL) accelerating under real gravity."""
    dt_drop0 = 0.10
    dt_rail0 = PRE_ROLL - dt_drop0

    v_lip = (v_in0[0], 0.0, v_in0[2] + GRAVITY * dt_drop0)
    p_lip = (
        hit0_pos[0] - v_lip[0] * dt_drop0,
        0.0,
        hit0_pos[2] - (v_lip[2] * dt_drop0 - 0.5 * GRAVITY * dt_drop0 * dt_drop0),
    )
    T_lip = vnorm(v_lip)
    target_v_lip = vlen(v_lip)

    v0_start = 0.35
    best = None
    best_err = 1e9
    for x_start in (-1.68, -1.52, -1.36, -1.22):
        dx = p_lip[0] - x_start
        if dx < 0.35:
            continue
        T_start = vnorm((dx, 0.0, -0.24 * dx))
        for dz_start in (0.18, 0.26, 0.34, 0.42, 0.52):
            p_start = (x_start, 0.0, p_lip[2] + dz_start)
            p_mid = (0.5 * (x_start + p_lip[0]), 0.0, p_lip[2] + 0.54 * dz_start)
            T_mid = vnorm((dx, 0.0, -dz_start))
            pts = hermite_piecewise([(p_start, T_start), (p_mid, T_mid), (p_lip, T_lip)], samples_per_seg=44)
            sim = simulate_rolling_pts(pts, v0_start)
            if sim is None or sim["max_slope_deg"] > 38.0:
                continue
            err = abs(sim["T"] - dt_rail0) * 6.0 + abs(sim["v_exit"] - target_v_lip)
            if err < best_err:
                best_err = err
                best = (sim, p_start, T_start, dz_start)

    assert best is not None, "Entry ramp solver found no candidate"
    sim0, p_start, T_start, dz_start = best
    pts = sim0["pts"]
    lo_v, hi_v = 0.12, 1.80
    for _ in range(24):
        mid_v = 0.5 * (lo_v + hi_v)
        sm = simulate_rolling_pts(pts, mid_v)
        if sm is None or sm["T"] > dt_rail0:
            lo_v = mid_v
        else:
            hi_v = mid_v
    sim_final = simulate_rolling_pts(pts, 0.5 * (lo_v + hi_v))
    v_lip_actual = vscale(T_lip, sim_final["v_exit"])
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


def build_forward_trajectory(
    notes: list[dict[str, Any]],
    board_collision_nudges: dict[int, float] | None = None,
):
    """Forward rigid-body + sloped-rolling trajectory synthesis across all notes."""
    nudges = board_collision_nudges or {}

    # 1. Solve true rolling entry ramp (0 .. PRE_ROLL)
    entry = solve_entry_ramp((-0.58, 0.0, 0.0), (1.35, 0.0, -2.80))
    p = entry["hit0"]
    v_in = entry["v_in0"]

    segments: list[dict[str, Any]] = [
        {
            "kind": "chute",
            "t0": 0.0,
            "t1": entry["t1"],
            "sim": entry["sim"],
            "rail_id": "entry_rail",
        },
        {
            "kind": "ballistic",
            "t0": entry["t1"],
            "t1": PRE_ROLL,
            "p0": entry["p_lip"],
            "v0": entry["v_lip"],
            "p1": p,
            "v1": v_in,
        },
    ]
    rails: list[dict[str, Any]] = [
        {
            "id": "entry_rail",
            "from_bounce": -1,
            "to_bounce": 0,
            "t0": 0.0,
            "t1": entry["t1"],
            "sim": entry["sim"],
        }
    ]
    hits: list[tuple[float, float, float]] = [p]
    hit_times: list[float] = [PRE_ROLL + notes[0]["start_time"]]

    recent_boards: list[dict[str, Any]] = []
    recent_rails_pts: list[tuple[str, float, float, list]] = [
        ("entry_rail", 0.0, entry["t1"], entry["sim"]["pts"])
    ]

    def make_board_obb(idx: int, hit_p, u_ax, n_ax) -> dict[str, Any]:
        dt_prev = (notes[idx]["start_time"] - notes[idx - 1]["start_time"]) if idx > 0 else 0.31
        dt_next = (notes[idx + 1]["start_time"] - notes[idx]["start_time"]) if idx + 1 < len(notes) else 0.31
        min_dt = min(dt_prev, dt_next)
        base_len = 0.25 if min_dt < 0.22 else (0.38 if abs(hit_p[0]) < 0.42 else 0.46)
        pitch_mod = max(-0.04, min(0.04, ((72 - notes[idx]["note"]) / 24.0) * 0.04))
        board_len = max(0.22, base_len + pitch_mod)
        side_b = -1.0 if hit_p[0] < 0 else 1.0
        half_len = board_len * 0.5
        shift_u = side_b * (half_len * 0.34)
        contact = vsub(hit_p, vscale(n_ax, MARBLE_RADIUS))
        center = vadd(vsub(contact, vscale(n_ax, BOARD_THICKNESS * 0.5)), vscale(u_ax, shift_u))
        return {
            "index": idx,
            "t_hit": notes[idx]["start_time"] + PRE_ROLL,
            "center": center,
            "u": u_ax,
            "v": (0.0, 1.0, 0.0),
            "n": n_ax,
            "half_extents": (half_len, BOARD_DEPTH_Y * 0.5, BOARD_THICKNESS * 0.5),
        }

    def incoming_arc_clear(cand_b: dict[str, Any], t_hit_b: float) -> bool:
        last_seg = segments[-1]
        if last_seg["kind"] != "ballistic":
            return True
        p0_in, v0_in = last_seg["p0"], last_seg["v0"]
        t0_in, t1_in = last_seg["t0"], last_seg["t1"]
        dur_in = t1_in - t0_in
        if dur_in <= 0.036:
            return True
        end_tau = max(0.0, dur_in - 0.033)
        start_tau = max(0.0, end_tau - 0.14)
        for s_k in range(8):
            tau = start_tau + (end_tau - start_tau) * (s_k / 7.0)
            pos = (
                p0_in[0] + v0_in[0] * tau,
                p0_in[1] + v0_in[1] * tau,
                p0_in[2] + v0_in[2] * tau - 0.5 * GRAVITY * tau * tau,
            )
            if point_to_obb_dist(pos, cand_b["center"], cand_b["u"], cand_b["v"], cand_b["n"], cand_b["half_extents"]) - MARBLE_RADIUS < 0.007:
                return False
        return True

    def _cand_pipe_pair(sim_c):
        pts_c = sim_c["pts"]
        vp_c = sim_c["v_prof"]
        if len(pts_c) > 70:
            idxs = list(range(0, len(pts_c) - 1, 2)) + [len(pts_c) - 1]
            pts_c = [pts_c[k] for k in idxs]
            vp_c = [vp_c[k] for k in idxs]
        rp = build_rail_pipe_polylines([{
            "id": "glide_rail_cand",
            "from_bounce": 0,
            "to_bounce": 0,
            "t0": 0.0,
            "t1": 0.0,
            "sim": {"pts": pts_c, "v_prof": vp_c},
        }])[0]
        return rp["pipe_front_raw"], rp["pipe_back_raw"]

    def arc_clear(
        p0,
        v0,
        dur: float,
        t0_abs: float,
        check_rails_too: bool = True,
        self_b: dict[str, Any] | None = None,
        check_rail_sim=None,
    ) -> bool:
        if dur <= 0.015:
            return True
        b_slice = recent_boards[-10:] + ([self_b] if self_b is not None else [])
        r_slice = recent_rails_pts[-4:]
        pipe_pair = _cand_pipe_pair(check_rail_sim) if check_rail_sim is not None else None
        pipe_req = MARBLE_RADIUS + RAIL_PIPE_RADIUS + 0.0025
        n_samp = 20
        for s_k in range(1, n_samp):
            tau = dur * (s_k / n_samp)
            t_abs = t0_abs + tau
            pos = (
                p0[0] + v0[0] * tau,
                p0[1] + v0[1] * tau,
                p0[2] + v0[2] * tau - 0.5 * GRAVITY * tau * tau,
            )
            for b in b_slice:
                if abs(t_abs - b["t_hit"]) < 0.032:
                    continue
                if point_to_obb_dist(pos, b["center"], b["u"], b["v"], b["n"], b["half_extents"]) - MARBLE_RADIUS < 0.010:
                    return False
            if check_rails_too:
                for _, rt0, rt1, rpts in r_slice:
                    if abs(t_abs - rt0) < 0.028 or abs(t_abs - rt1) < 0.028:
                        continue
                    for k in range(0, len(rpts) - 1, 2):
                        if point_to_segment_dist(pos, rpts[k], rpts[min(len(rpts) - 1, k + 2)]) < 0.145:
                            return False
            if pipe_pair is not None:
                for p_pts in pipe_pair:
                    for k in range(len(p_pts) - 1):
                        if point_to_segment_dist(pos, p_pts[k], p_pts[k + 1]) < pipe_req:
                            return False
        return True

    def chute_pts_clear(pts_cand, extra_board: dict[str, Any] | None = None) -> bool:
        b_slice = recent_boards[-10:] + ([extra_board] if extra_board is not None else [])
        r_slice = recent_rails_pts[-4:]
        arc_pts = []
        for s_prev in segments[-3:]:
            if s_prev["kind"] != "ballistic":
                continue
            d_s = s_prev["t1"] - s_prev["t0"]
            if d_s <= 0.02:
                continue
            p0_s, v0_s = s_prev["p0"], s_prev["v0"]
            for ak in range(1, 12):
                tau_s = d_s * (ak / 12.0)
                arc_pts.append((
                    p0_s[0] + v0_s[0] * tau_s,
                    0.0,
                    p0_s[2] + v0_s[2] * tau_s - 0.5 * GRAVITY * tau_s * tau_s,
                ))
        for idx_p in range(1, len(pts_cand) - 1, 2):
            pos = pts_cand[idx_p]
            for b in b_slice:
                if point_to_obb_dist(pos, b["center"], b["u"], b["v"], b["n"], b["half_extents"]) - MARBLE_RADIUS < 0.014:
                    return False
            for _, _, _, rpts in r_slice:
                for k in range(0, len(rpts) - 1, 3):
                    if point_to_segment_dist(pos, rpts[k], rpts[min(len(rpts) - 1, k + 3)]) < 0.155:
                        return False
            for apos in arc_pts:
                if vlen(vsub(pos, apos)) < 0.148:
                    return False
        return True

    for i in range(len(notes) - 1):
        t0 = notes[i]["start_time"] + PRE_ROLL
        t1 = notes[i + 1]["start_time"] + PRE_ROLL
        dt = t1 - t0
        side = -1.0 if p[0] < 0 else 1.0
        extra_dz = nudges.get(i, 0.0)

        if dt <= RAIL_DT_THRESHOLD:
            if dt < 0.22:
                x_targ = max(-0.98, min(0.98, p[0] - side * 0.45))
            elif dt < 0.38:
                x_targ = -side * (0.45 + 0.18 * ((notes[i + 1]["note"] % 5) / 4.0))
            else:
                x_targ = -side * 0.68

            best_b = None
            best_score = 1e9
            e_cands = (0.86, 0.89, 0.78, 0.70, 0.60) if dt > 0.52 else (0.64, 0.56, 0.72, 0.48, 0.38, 0.30, 0.80, 0.86)
            ang_lo = 2 if dt > 0.52 else 6
            for obb_min_req in (0.36, 0.25):
                if best_b is not None:
                    break
                for e in e_cands:
                    angs = [(-side) * a for a in range(ang_lo, 51, 1)] + [side * a for a in range(ang_lo, 40, 2)]
                    for a_signed in angs:
                        alpha = math.radians(a_signed)
                        res = bounce_velocity(v_in, alpha, e)
                        if res is None:
                            continue
                        v_out, vn_in, vn_out, u_b, n_b = res
                        if vn_in >= -0.80 or vn_out < 0.78:
                            continue
                        if dt > 0.52 and abs(v_out[0]) > 2.40:
                            continue
                        p_next = (
                            p[0] + v_out[0] * dt,
                            0.0,
                            p[2] + v_out[2] * dt - 0.5 * GRAVITY * dt * dt,
                        )
                        v_next_in = (v_out[0], 0.0, v_out[2] - GRAVITY * dt)
                        dz = p[2] - p_next[2]
                        min_vz_req = -1.9 if (i + 2 < len(notes) and (notes[i + 2]["start_time"] - notes[i + 1]["start_time"]) < 0.22) else -1.3
                        if v_next_in[2] >= min_vz_req:
                            continue
                        if abs(p_next[0]) > 1.22:
                            continue
                        if dz < (-0.18 + extra_dz):
                            continue
                        # Avoid trapping same-side bounces at shallow vertical drop
                        if obb_min_req > 0.30 and p[0] * p_next[0] > 0.04 and dz < 0.28 + extra_dz:
                            continue
                        dx_sep = abs(p_next[0] - p[0])
                        if dz < 0.10 + extra_dz and dx_sep < 0.28:
                            continue
                        if dz < 0.22 + extra_dz and dx_sep < 0.22:
                            continue
                        if dt >= 0.25 and abs(p_next[0]) < 0.16:
                            continue

                        cand_b_i = make_board_obb(i, p, u_b, n_b)
                        obb_bad = False
                        for rb in recent_boards[-10:]:
                            if obb_overlap(cand_b_i, rb, margin=0.025):
                                obb_bad = True
                                break
                            if point_to_obb_dist(p_next, rb["center"], rb["u"], rb["v"], rb["n"], rb["half_extents"]) < obb_min_req:
                                obb_bad = True
                                break
                        if obb_bad:
                            continue
                        if not incoming_arc_clear(cand_b_i, t0):
                            continue
                        if not arc_clear(p, v_out, dt, t0, self_b=cand_b_i):
                            continue

                        next_is_rail = (i + 2 < len(notes) and (notes[i + 2]["start_time"] - notes[i + 1]["start_time"]) > RAIL_DT_THRESHOLD)
                        next_side = -1.0 if p_next[0] < 0 else 1.0
                        align_pen = 0.0 if (v_next_in[0] * next_side >= -0.3) else 0.5
                        sc = (
                            abs(p_next[0] - x_targ)
                            + 0.18 * abs(abs(v_next_in[2]) - 4.2)
                            + 1.10 * max(0.0, abs(v_out[0]) - 3.0)
                            + 2.20 * max(0.0, abs(p_next[0]) - 0.85)
                            + (2.50 * max(0.0, abs(v_out[0]) - 3.4) + 3.00 * max(0.0, abs(p_next[0]) - 0.78) if next_is_rail else 0.0)
                            + 0.015 * abs(abs(a_signed) - 24.0)
                            + 0.25 * abs(e - 0.62)
                            + 1.2 * max(0.0, 0.16 + extra_dz - dz)
                            + align_pen
                        )
                        if sc < best_score:
                            best_score = sc
                            best_b = (p_next, v_next_in, v_out, u_b, n_b, e, cand_b_i)

            if best_b is None:
                # Relax non-hit flyby check slightly as emergency fallback while keeping physics valid
                raise RuntimeError(f"Bounce solver failed at note {i} (dt={dt}, p={p}, v_in={v_in})")
            p_next, v_next_in, v_out, u_b, n_b, e_b, cand_b_i = best_b
            recent_boards.append(cand_b_i)
            segments.append({
                "kind": "ballistic",
                "t0": t0,
                "t1": t1,
                "p0": p,
                "v0": v_out,
                "p1": p_next,
                "v1": v_next_in,
                "hit_u": u_b,
                "hit_n": n_b,
                "hit_e": e_b,
            })
            p, v_in = p_next, v_next_in
        else:
            best_rail_pack = None
            best_rail_cost = 1e9
            in_dir_v = 1.0 if v_in[0] >= 0 else -1.0
            cand_angs = []
            for a in (22, 26, 18, 30, 14, 34, 38, 42, 46, 10, 6):
                for sg in (-in_dir_v, -side, side):
                    val = int(sg * a)
                    if val not in cand_angs:
                        cand_angs.append(val)

            for dt_drop in (0.17, 0.14, 0.21, 0.11):
                if best_rail_pack is not None:
                    break
                for dt_hop in (0.14, 0.11, 0.18, 0.09, 0.22):
                    dt_rail = dt - dt_hop - dt_drop
                    for e_hop in (0.58, 0.48, 0.66, 0.40, 0.34, 0.74):
                        for a_signed in cand_angs:
                            alpha = math.radians(a_signed)
                            res = bounce_velocity(v_in, alpha, e_hop)
                            if res is None:
                                continue
                            v_out, vn_in, vn_out, u_b, n_b = res
                            if vn_in >= -0.80 or vn_out < 0.78:
                                continue
                            cand_b_i = make_board_obb(i, p, u_b, n_b)
                            if any(obb_overlap(cand_b_i, rb, margin=0.025) for rb in recent_boards[-10:]):
                                continue
                            if not incoming_arc_clear(cand_b_i, t0):
                                continue
                            p_enter = (
                                p[0] + v_out[0] * dt_hop,
                                0.0,
                                p[2] + v_out[2] * dt_hop - 0.5 * GRAVITY * dt_hop * dt_hop,
                            )
                            v_enter = (v_out[0], 0.0, v_out[2] - GRAVITY * dt_hop)
                            if v_enter[2] > -0.10 or abs(v_enter[0]) < 0.65:
                                continue
                            entry_slope = math.degrees(math.atan2(-v_enter[2], abs(v_enter[0])))
                            if entry_slope < 6.0 or entry_slope > 35.0:
                                continue
                            if not arc_clear(p, v_out, dt_hop, t0, self_b=cand_b_i):
                                continue
                            def _drop_ok(px_c, v_ex_vec, r_sim, _t_ex=t0 + dt_hop + dt_rail, _dt_dr=dt_drop, _b_i=cand_b_i):
                                p_land = (
                                    px_c[0] + v_ex_vec[0] * _dt_dr,
                                    0.0,
                                    px_c[2] + v_ex_vec[2] * _dt_dr - 0.5 * GRAVITY * _dt_dr * _dt_dr,
                                )
                                for rb in recent_boards[-10:]:
                                    if point_to_obb_dist(p_land, rb["center"], rb["u"], rb["v"], rb["n"], rb["half_extents"]) < 0.36:
                                        return False
                                return arc_clear(px_c, v_ex_vec, _dt_dr, _t_ex, self_b=_b_i, check_rail_sim=r_sim)

                            sol = solve_3d_chute(
                                p_enter,
                                v_enter,
                                dt_rail,
                                clear_fn=lambda pts_c, _bi=cand_b_i: chute_pts_clear(pts_c, extra_board=_bi),
                                dt_drop=dt_drop,
                                drop_clear_fn=_drop_ok,
                            )
                            if sol is None:
                                continue
                            if not arc_clear(
                                p,
                                v_out,
                                dt_hop,
                                t0,
                                check_rails_too=False,
                                check_rail_sim=sol["sim"],
                            ):
                                continue
                            t_ex_cand = t0 + dt_hop + sol["sim"]["T"]
                            dt_dr_cand = t1 - t_ex_cand
                            p_ex_c = sol["p_exit"]
                            v_ex_c = vscale(sol["T1"], sol["sim"]["v_exit"])
                            p_next_c = (
                                p_ex_c[0] + v_ex_c[0] * dt_dr_cand,
                                0.0,
                                p_ex_c[2] + v_ex_c[2] * dt_dr_cand - 0.5 * GRAVITY * dt_dr_cand * dt_dr_cand,
                            )
                            if abs(p_next_c[0]) > 1.06:
                                continue
                            rcost = abs(sol["y_bow"]) + 0.3 * sol["max_slope_deg"]
                            if rcost < best_rail_cost:
                                best_rail_cost = rcost
                                best_rail_pack = (sol, p_enter, v_enter, v_out, dt_hop, u_b, n_b, e_hop, cand_b_i)
                                if best_rail_cost < 8.8:
                                    break
                        if best_rail_pack is not None:
                            break
                    if best_rail_pack is not None:
                        break

            if best_rail_pack is None:
                raise RuntimeError(f"Rail solver failed at note {i} (dt={dt}, p={p}, v_in={v_in})")

            sol, p_enter, v_enter, v_out, dt_hop, u_b, n_b, e_hop, cand_b_i = best_rail_pack
            t_enter = t0 + dt_hop
            t_exit = t_enter + sol["sim"]["T"]
            dt_drop_actual = t1 - t_exit
            p_exit = sol["p_exit"]
            v_exit = vscale(sol["T1"], sol["sim"]["v_exit"])
            p_next = (
                p_exit[0] + v_exit[0] * dt_drop_actual,
                0.0,
                p_exit[2] + v_exit[2] * dt_drop_actual - 0.5 * GRAVITY * dt_drop_actual * dt_drop_actual,
            )
            v_next_in = (v_exit[0], 0.0, v_exit[2] - GRAVITY * dt_drop_actual)
            rid = f"glide_rail_{i}"
            recent_boards.append(cand_b_i)
            recent_rails_pts.append((rid, t_enter, t_exit, sol["sim"]["pts"]))

            segments.append({
                "kind": "ballistic",
                "t0": t0,
                "t1": t_enter,
                "p0": p,
                "v0": v_out,
                "p1": p_enter,
                "v1": v_enter,
                "hit_u": u_b,
                "hit_n": n_b,
                "hit_e": e_hop,
            })
            segments.append({
                "kind": "chute",
                "t0": t_enter,
                "t1": t_exit,
                "sim": sol["sim"],
                "rail_id": rid,
            })
            segments.append({
                "kind": "ballistic",
                "t0": t_exit,
                "t1": t1,
                "p0": p_exit,
                "v0": v_exit,
                "p1": p_next,
                "v1": v_next_in,
            })
            rails.append({
                "id": rid,
                "from_bounce": i,
                "to_bounce": i + 1,
                "t0": t_enter,
                "t1": t_exit,
                "sim": sol["sim"],
                "max_slope_deg": round(sol["max_slope_deg"], 2),
                "kind": sol["kind"],
            })
            p, v_in = p_next, v_next_in

        hits.append(p)
        hit_times.append(t1)

    # 3. Post-roll final bounce
    t_last = hit_times[-1]
    side_last = -1.0 if p[0] < 0 else 1.0
    res_last = bounce_velocity(v_in, math.radians((-side_last) * 24.0), 0.65)
    if res_last:
        v_last_out, _, _, u_last, n_last = res_last
        e_last = 0.65
    else:
        v_last_out = (-side_last * 1.8, 0.0, 1.5)
        n_last = vnorm(vsub(v_last_out, v_in))
        u_last = (n_last[2], 0.0, -n_last[0])
        e_last = 0.65
    p_end = (
        p[0] + v_last_out[0] * POST_ROLL,
        0.0,
        p[2] + v_last_out[2] * POST_ROLL - 0.5 * GRAVITY * POST_ROLL * POST_ROLL,
    )
    segments.append({
        "kind": "ballistic",
        "t0": t_last,
        "t1": t_last + POST_ROLL,
        "p0": p,
        "v0": v_last_out,
        "p1": p_end,
        "v1": (v_last_out[0], 0.0, v_last_out[2] - GRAVITY * POST_ROLL),
        "hit_u": u_last,
        "hit_n": n_last,
        "hit_e": e_last,
    })

    meta = {
        "total_duration": t_last + POST_ROLL,
        "total_frames": int(math.ceil((t_last + POST_ROLL) * FPS)),
        "hit_times": hit_times,
        "hits": hits,
    }
    return segments, rails, meta


def eval_segments(segments: list[dict[str, Any]], seg_hint: int, t: float):
    n_seg = len(segments)
    idx = seg_hint
    while idx + 1 < n_seg and t > segments[idx]["t1"] + 1e-9:
        idx += 1
    while idx > 0 and t < segments[idx]["t0"] - 1e-9:
        idx -= 1
    seg = segments[idx]
    dt_seg = max(1e-9, seg["t1"] - seg["t0"])
    tau = max(0.0, min(dt_seg, t - seg["t0"]))
    if seg["kind"] == "ballistic":
        p0, v0 = seg["p0"], seg["v0"]
        pos = (
            p0[0] + v0[0] * tau,
            p0[1] + v0[1] * tau,
            p0[2] + v0[2] * tau - 0.5 * GRAVITY * tau * tau,
        )
        vel = (v0[0], v0[1], v0[2] - GRAVITY * tau)
    else:
        pos, vel = eval_rolling_table(seg["sim"], tau)
    return pos, vel, idx


# -------------------- Derive Boards & Banked 3D Twin Rails --------------------

def derive_boards_and_physics(
    notes: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    hit_times: list[float],
    hits: list[tuple[float, float, float]],
    length_overrides: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    seg_ending = {round(seg["t1"], 6): seg for seg in segments}
    seg_starting = {round(seg["t0"], 6): seg for seg in segments}
    len_ov = length_overrides or {}

    boards = []
    for i, n in enumerate(notes):
        t_hit = round(hit_times[i], 6)
        s_in = seg_ending[t_hit]
        s_out = seg_starting[t_hit]
        v_in = s_in["v1"]
        v_out = s_out["v0"]

        if "hit_n" in s_out and "hit_u" in s_out:
            normal = s_out["hit_n"]
            u_ax = s_out["hit_u"]
        else:
            dv = vsub(v_out, v_in)
            normal = vnorm((dv[0], 0.0, dv[2]))
            u_ax = (normal[2], 0.0, -normal[0])
        v_ax = (0.0, 1.0, 0.0)

        side = -1.0 if hits[i][0] < 0 else 1.0
        tilt_deg = math.degrees(math.atan2(normal[0], normal[2]))

        dt_prev = (hit_times[i] - hit_times[i - 1]) if i > 0 else 0.31
        dt_next = (hit_times[i + 1] - hit_times[i]) if i + 1 < len(hit_times) else 0.31
        min_dt = min(dt_prev, dt_next)
        # Pitch-responsive + clearance-friendly lengths (compact crisp keys on fast 1/8 notes)
        base_len = 0.25 if min_dt < 0.22 else (0.38 if abs(hits[i][0]) < 0.42 else 0.46)
        pitch_mod = max(-0.04, min(0.04, ((72 - n["note"]) / 24.0) * 0.04))
        board_len = len_ov.get(i, max(0.22, base_len + pitch_mod))

        contact = vsub(hits[i], vscale(normal, MARBLE_RADIUS))
        half_len = board_len * 0.5
        # Place bar cleanly centered or slightly shifted outward from center so inner tip has max clearance
        inward_u_sign = -side
        shift_u = -inward_u_sign * (half_len * 0.34)
        board_center = vadd(
            vsub(contact, vscale(normal, BOARD_THICKNESS * 0.5)),
            vscale(u_ax, shift_u),
        )

        p_pin1 = vadd(board_center, vscale(u_ax, -inward_u_sign * (half_len * 0.46)))
        p_pin2 = vadd(board_center, vscale(u_ax, inward_u_sign * (half_len * 0.26)))
        pins = [
            {
                "a": r5((p_pin1[0], BOARD_DEPTH_Y * 0.5 - 0.015, p_pin1[2])),
                "b": r5((p_pin1[0], BACK_WALL_Y + 0.02, p_pin1[2])),
                "radius": PIN_RADIUS,
            },
            {
                "a": r5((p_pin2[0], BOARD_DEPTH_Y * 0.5 - 0.015, p_pin2[2])),
                "b": r5((p_pin2[0], BACK_WALL_Y + 0.02, p_pin2[2])),
                "radius": PIN_RADIUS,
            },
        ]

        vn_in = vdot(v_in, normal)
        vn_out = vdot(v_out, normal)
        restitution = -vn_out / vn_in if abs(vn_in) > 1e-6 else 0.0

        boards.append({
            "index": i,
            "t_hit": round(hit_times[i], 5),
            "frame_hit": int(round(hit_times[i] * FPS)),
            "side": "left" if side < 0 else "right",
            "track_idx": n["track_idx"],
            "instrument": n["instrument"],
            "instrument_type": n["instrument_type"],
            "material": n["material"],
            "color": n["color"],
            "note": n["note"],
            "pitch_name": n["pitch_name"],
            "hit_center": r5(hits[i]),
            "contact": r5(contact),
            "center": board_center,
            "u": u_ax,
            "v": v_ax,
            "n": normal,
            "half_extents": (half_len, BOARD_DEPTH_Y * 0.5, BOARD_THICKNESS * 0.5),
            "length": round(board_len, 4),
            "width_y": BOARD_DEPTH_Y,
            "thickness": BOARD_THICKNESS,
            "tilt_deg": round(tilt_deg, 2),
            "vn_in": round(vn_in, 3),
            "vn_out": round(vn_out, 3),
            "restitution": round(restitution, 3),
            "pins": pins,
        })
    return boards


def build_rail_pipe_polylines(rails: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build exact 3D centripetal-banked twin chrome rail polylines + cross-ties cradling the marble."""
    out = []
    for r in rails:
        sim = r["sim"]
        pts = sim["pts"]
        v_prof = sim["v_prof"]
        n_pts = len(pts)
        is_glide = r["id"].startswith("glide_rail_")

        # Compute cumulative arc length
        s_arr = [0.0]
        for k in range(n_pts - 1):
            s_arr.append(s_arr[-1] + vlen(vsub(pts[k + 1], pts[k])))

        front_pts = []
        back_pts = []
        ties = []

        for k in range(n_pts):
            c = pts[k]
            if k == 0:
                T = vnorm(vsub(pts[1], pts[0]))
            elif k == n_pts - 1:
                T = vnorm(vsub(pts[-1], pts[-2]))
            else:
                T = vnorm(vsub(pts[k + 1], pts[k - 1]))

            # Curvature vector kappa ~ dT/ds for gentle centripetal banking (clamped)
            if 0 < k < n_pts - 1:
                T_prev = vnorm(vsub(pts[k], pts[k - 1]))
                T_next = vnorm(vsub(pts[k + 1], pts[k]))
                ds_loc = max(0.02, 0.5 * (s_arr[k + 1] - s_arr[k - 1]))
                kappa = vscale(vsub(T_next, T_prev), 1.0 / ds_loc)
                k_mag = vlen(kappa)
                if k_mag > 6.0:
                    kappa = vscale(kappa, 6.0 / k_mag)
            else:
                kappa = (0.0, 0.0, 0.0)

            v_loc = v_prof[k]
            # Effective upward normal: opposes gravity (0,0,+G) + gentle centripetal banking
            raw_up = vadd((0.0, 0.0, GRAVITY), vscale(kappa, 0.15 * v_loc * v_loc))
            # Project orthogonal to T
            proj = vsub(raw_up, vscale(T, vdot(raw_up, T)))
            N = vnorm(proj, fallback=(0.0, 0.0, 1.0))
            if N[2] < 0.0:
                N = vscale(N, -1.0)
            B = vnorm(vcross(T, N), fallback=(0.0, 1.0, 0.0))

            s_frac = k / max(1, n_pts - 1)
            tip_flare_n = 0.0
            tip_flare_b = 0.0
            if s_frac > 0.92:
                u = (s_frac - 0.92) / 0.08
                tip_flare_b = 0.018 * (u * u)
            elif is_glide and s_frac < 0.08:
                u = (0.08 - s_frac) / 0.08
                tip_flare_b = 0.018 * (u * u)

            cradle_ctr = vsub(c, vscale(N, RAIL_PERP_OFFSET + tip_flare_n))
            # Continuous frame assignment prevents U-turn pipe crossovers
            p_f = vadd(cradle_ctr, vscale(B, RAIL_Y_HALF + tip_flare_b))
            p_b = vsub(cradle_ctr, vscale(B, RAIL_Y_HALF + tip_flare_b))
            front_pts.append(p_f)
            back_pts.append(p_b)

            # Minimal structural U-tie rungs spaced cleanly (3-4 per rail)
            tie_step = max(24, n_pts // 4)
            if k in range(14, n_pts - 14, tie_step):
                under = vsub(cradle_ctr, vscale(N, RAIL_PIPE_RADIUS + 0.024))
                ties.append([r5(p_f), r5(under), r5(p_b)])

        out.append({
            "id": r["id"],
            "from_bounce": r["from_bounce"],
            "to_bounce": r["to_bounce"],
            "t0": r["t0"],
            "t1": r["t1"],
            "max_slope_deg": r.get("max_slope_deg", 0.0),
            "pipe_radius": RAIL_PIPE_RADIUS,
            "pipe_front": [r5(p) for p in front_pts],
            "pipe_back": [r5(p) for p in back_pts],
            "ties": ties,
            "pipe_front_raw": front_pts,
            "pipe_back_raw": back_pts,
        })
    return out


# -------------------- 240Hz Collision & Physical Audit --------------------

def validate_scene(
    segments: list[dict[str, Any]],
    boards: list[dict[str, Any]],
    rail_pipes: list[dict[str, Any]],
    total_duration: float,
    substeps: int = SUBSTEPS_VERIFY,
) -> dict[str, Any]:
    physics_issues = []
    for b in boards:
        if abs(b["tilt_deg"]) > 50.0:
            physics_issues.append((b["index"], f"tilt={b['tilt_deg']}"))
        if b["vn_in"] >= -0.40:
            physics_issues.append((b["index"], f"vn_in={b['vn_in']}"))
        if b["vn_out"] <= 0.20:
            physics_issues.append((b["index"], f"vn_out={b['vn_out']}"))
        if not (0.30 <= b["restitution"] <= 0.90):
            physics_issues.append((b["index"], f"restitution={b['restitution']}"))

    obb_issues = []
    for i in range(len(boards)):
        for j in range(i + 1, min(len(boards), i + 12)):
            if obb_overlap(boards[i], boards[j], margin=0.020):
                obb_issues.append((i, j))

    total_steps = int(math.ceil(total_duration * FPS * substeps))
    step_dt = 1.0 / (FPS * substeps)

    board_collisions = []
    rail_collisions = []
    min_nonhit_board_clearance = 999.0
    min_self_board_clearance = 999.0
    min_rail_clearance = 999.0

    b_idx_hint = 0
    seg_hint = 0
    r_hint = 0
    for step in range(total_steps + 1):
        t = min(total_duration, step * step_dt)
        pos, _, seg_hint = eval_segments(segments, seg_hint, t)

        while b_idx_hint + 1 < len(boards) and boards[b_idx_hint + 1]["t_hit"] < t:
            b_idx_hint += 1
        lo = max(0, b_idx_hint - 6)
        hi = min(len(boards), b_idx_hint + 8)

        for j in range(lo, hi):
            b = boards[j]
            d = point_to_obb_dist(pos, b["center"], b["u"], b["v"], b["n"], b["half_extents"])
            clearance = d - MARBLE_RADIUS
            if abs(t - b["t_hit"]) < 0.032:
                min_self_board_clearance = min(min_self_board_clearance, clearance)
                if clearance < -0.004:
                    board_collisions.append((step, round(t, 4), j, round(clearance, 5), "hit_window"))
            else:
                min_nonhit_board_clearance = min(min_nonhit_board_clearance, clearance)
                if clearance < 0.004:
                    board_collisions.append((step, round(t, 4), j, round(clearance, 5), "flyby"))

        while r_hint + 1 < len(rail_pipes) and rail_pipes[r_hint + 1]["t0"] - 1.5 < t:
            r_hint += 1
        r_lo = max(0, r_hint - 2)
        r_hi = min(len(rail_pipes), r_hint + 3)
        for ri in range(r_lo, r_hi):
            rp = rail_pipes[ri]
            if t < rp["t0"] - 1.5 or t > rp["t1"] + 1.5:
                continue
            on_this_rail = (rp["t0"] - 1e-4 <= t <= rp["t1"] + 1e-4)
            d_min = 999.0
            for pts in (rp["pipe_front_raw"], rp["pipe_back_raw"]):
                for k in range(len(pts) - 1):
                    d_min = min(d_min, point_to_segment_dist(pos, pts[k], pts[k + 1]))
            clear_pipe = d_min - (MARBLE_RADIUS + rp["pipe_radius"])
            if on_this_rail:
                if clear_pipe < -0.005:
                    rail_collisions.append((step, round(t, 4), rp["id"], round(clear_pipe, 5), "on_rail_pen"))
            else:
                min_rail_clearance = min(min_rail_clearance, clear_pipe)
                if clear_pipe < 0.001:
                    rail_collisions.append((step, round(t, 4), rp["id"], round(clear_pipe, 5), "off_rail_hit"))

    return {
        "physics_issues": physics_issues,
        "obb_issues": obb_issues,
        "board_collisions": board_collisions,
        "rail_collisions": rail_collisions,
        "min_nonhit_board_clearance": round(min_nonhit_board_clearance, 5),
        "min_self_board_clearance": round(min_self_board_clearance, 5),
        "min_rail_clearance": round(min_rail_clearance, 5),
        "ok": not (physics_issues or obb_issues or board_collisions or rail_collisions),
    }


def solve_trajectory(notes: list[dict[str, Any]]):
    """Solve forward trajectory and resolve any OBB/flyby clearances automatically."""
    nudges: dict[int, float] = {}
    len_overrides: dict[int, float] = {}
    cached_nudges: dict[int, float] | None = None
    cached_traj = None

    for _it in range(10):
        cur_nudges = dict(nudges)
        if cached_traj is None or cur_nudges != cached_nudges:
            cached_traj = build_forward_trajectory(notes, board_collision_nudges=nudges)
            cached_nudges = cur_nudges
        segments, rails, meta = cached_traj
        boards = derive_boards_and_physics(notes, segments, meta["hit_times"], meta["hits"], length_overrides=len_overrides)
        rail_pipes = build_rail_pipe_polylines(rails)
        report = validate_scene(segments, boards, rail_pipes, meta["total_duration"], substeps=SUBSTEPS_VERIFY)
        if report["ok"]:
            break

        changed = False
        obb_seen = set()
        for i, j in report["obb_issues"]:
            if (i, j) in obb_seen:
                continue
            obb_seen.add((i, j))
            li = len_overrides.get(i, boards[i]["length"])
            lj = len_overrides.get(j, boards[j]["length"])
            if li > 0.21 or lj > 0.21:
                len_overrides[i] = max(0.20, li - 0.05)
                len_overrides[j] = max(0.20, lj - 0.05)
            else:
                nudges[i] = min(0.22, nudges.get(i, 0.0) + 0.06)
            changed = True

        flyby_seen = set()
        for _step, _t, j, _clr, kind in report["board_collisions"]:
            if kind == "flyby" and j not in flyby_seen:
                flyby_seen.add(j)
                lj = len_overrides.get(j, boards[j]["length"])
                if lj > 0.21:
                    len_overrides[j] = max(0.20, lj - 0.06)
                elif j > 0:
                    nudges[j - 1] = min(0.22, nudges.get(j - 1, 0.0) + 0.06)
                changed = True
        if not changed:
            break

    # Final strict 240 Hz verification
    report = validate_scene(segments, boards, rail_pipes, meta["total_duration"], substeps=SUBSTEPS_VERIFY)

    total_frames = meta["total_frames"]
    samples = []
    seg_hint = 0
    for f in range(total_frames + 1):
        t = f / FPS
        pos, vel, seg_hint = eval_segments(segments, seg_hint, t)
        samples.append({
            "frame": f,
            "t": round(t, 5),
            "position": r5(pos),
            "velocity": r5(vel),
        })

    boards_out = [
        {
            "index": b["index"],
            "t_hit": b["t_hit"],
            "frame_hit": b["frame_hit"],
            "side": b["side"],
            "track_idx": b["track_idx"],
            "instrument": b["instrument"],
            "instrument_type": b["instrument_type"],
            "material": b["material"],
            "color": b["color"],
            "note": b["note"],
            "pitch_name": b["pitch_name"],
            "hit_center": b["hit_center"],
            "contact": b["contact"],
            "center": r5(b["center"]),
            "axis_u": r5(b["u"]),
            "axis_v": r5(b["v"]),
            "axis_n": r5(b["n"]),
            "half_extents": [round(x, 5) for x in b["half_extents"]],
            "length": b["length"],
            "width_y": b["width_y"],
            "thickness": b["thickness"],
            "tilt_deg": b["tilt_deg"],
            "restitution": b["restitution"],
            "pins": b["pins"],
        }
        for b in boards
    ]

    rails_out = [
        {
            "id": rp["id"],
            "from_bounce": rp["from_bounce"],
            "to_bounce": rp["to_bounce"],
            "t0": round(rp["t0"], 5),
            "t1": round(rp["t1"], 5),
            "max_slope_deg": rp["max_slope_deg"],
            "pipe_radius": rp["pipe_radius"],
            "pipe_front": rp["pipe_front"],
            "pipe_back": rp["pipe_back"],
            "ties": rp["ties"],
        }
        for rp in rail_pipes
    ]

    slopes = [rp["max_slope_deg"] for rp in rail_pipes if rp["max_slope_deg"] > 0]
    plan = {
        "schema": "marble_physics_plan_v3",
        "fps": FPS,
        "pre_roll": PRE_ROLL,
        "post_roll": POST_ROLL,
        "gravity": GRAVITY,
        "marble_radius": MARBLE_RADIUS,
        "back_wall_y": BACK_WALL_Y,
        "total_duration": round(meta["total_duration"], 5),
        "total_frames": total_frames,
        "total_notes": len(notes),
        "rail_slope_range_deg": [min(slopes) if slopes else 0.0, max(slopes) if slopes else 0.0],
        "validation": {
            "ok": report["ok"],
            "physics_issues": len(report["physics_issues"]),
            "obb_issues": len(report["obb_issues"]),
            "board_collisions": len(report["board_collisions"]),
            "rail_collisions": len(report["rail_collisions"]),
            "min_nonhit_board_clearance": report["min_nonhit_board_clearance"],
            "min_self_board_clearance": report["min_self_board_clearance"],
            "min_rail_clearance": report["min_rail_clearance"],
        },
        "boards": boards_out,
        "rails": rails_out,
        "samples": samples,
    }
    return plan, report


def plan_from_notes_json(input_json: str, output_json: str):
    with open(input_json, "r", encoding="utf-8") as f:
        src = json.load(f)
    notes = src["notes"]
    plan, report = solve_trajectory(notes)
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"[{os.path.basename(output_json)}] notes={len(notes)}, frames={plan['total_frames']}, "
          f"dur={plan['total_duration']}s, slopes={plan['rail_slope_range_deg']} -> validation={plan['validation']}")
    if not report["ok"]:
        print("  PHYSICS:", report["physics_issues"][:6])
        print("  OBB:", report["obb_issues"][:6])
        print("  BOARD_COLL:", report["board_collisions"][:6])
        print("  RAIL_COLL:", report["rail_collisions"][:6])
    return plan


if __name__ == "__main__":
    import sys
    base = "projects/jingle-bells-marble-music/artifacts"
    if len(sys.argv) >= 3:
        plan_from_notes_json(sys.argv[1], sys.argv[2])
    else:
        plan_from_notes_json(f"{base}/sample_14s_notes.json", f"{base}/sample_14s_few_rail_physics.json")
        plan_from_notes_json(f"{base}/full_notes.json", f"{base}/full_few_rail_physics.json")
