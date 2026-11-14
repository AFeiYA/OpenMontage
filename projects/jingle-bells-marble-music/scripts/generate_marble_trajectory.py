#!/usr/bin/env python3
"""MIDI -> marble trajectory JSON (samples + bounce normals).

Trajectory is the authority. Blender only consumes this file to place boards
from bounce normals and keyframe the marble from samples.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import mido

FPS = 60
DESCENT_SPEED = 2.2
X_WIDTH = 2.4
HIT_Y = 0.0
MARBLE_RADIUS = 0.12
LAUNCH_PRE_ROLL = 0.85  # seconds before first hit
LAUNCH_Z_ABOVE = 1.8

# Arc height for bounce segments (world +Z lift above chord)
H_FAST = 0.22
H_HIGH = 0.38
H_RAIL = 0.06
H_DT_GAIN = 0.35
H_MIN = 0.10
H_MAX = 0.85

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

INSTRUMENT_CONFIGS = {
    1: {"name": "Castanets", "color": [0.65, 0.35, 0.15], "material": "wood_dark", "type": "castanet"},
    2: {"name": "Piano", "color": [0.95, 0.95, 0.92], "material": "ivory", "type": "piano_key"},
    3: {"name": "Vibraphone", "color": [0.75, 0.82, 0.95], "material": "aluminum", "type": "metal_bar"},
    4: {"name": "Tubular Bells", "color": [0.85, 0.75, 0.35], "material": "brass", "type": "brass_tube"},
    5: {"name": "Marimba", "color": [0.55, 0.25, 0.10], "material": "rosewood", "type": "marimba_bar"},
    6: {"name": "Kalimba", "color": [0.45, 0.30, 0.20], "material": "steel_tine", "type": "tine"},
    7: {"name": "Flute", "color": [0.88, 0.90, 0.92], "material": "silver", "type": "flute_pipe"},
    8: {"name": "Xylophone", "color": [0.95, 0.40, 0.20], "material": "rainbow_wood", "type": "rainbow_bar"},
    9: {"name": "Ocarina", "color": [0.90, 0.85, 0.80], "material": "ceramic", "type": "ceramic_tile"},
    10: {"name": "Glockenspiel", "color": [0.90, 0.92, 0.95], "material": "chrome", "type": "chrome_bell"},
    11: {"name": "Steel Drums", "color": [0.35, 0.45, 0.55], "material": "hammered_steel", "type": "steel_pan"},
    12: {"name": "Guitar", "color": [0.80, 0.50, 0.20], "material": "spruce_acoustic", "type": "guitar_block"},
}


def note_name(n: int) -> str:
    return f"{NOTE_NAMES[n % 12]}{(n // 12) - 1}"


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vec_len(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vec_normalize(a, fallback=(0.0, 0.0, 1.0)):
    L = vec_len(a)
    if L < 1e-8:
        return fallback
    return (a[0] / L, a[1] / L, a[2] / L)


def lerp3(a, b, s):
    return (
        a[0] + (b[0] - a[0]) * s,
        a[1] + (b[1] - a[1]) * s,
        a[2] + (b[2] - a[2]) * s,
    )


def round3(v):
    return [round(v[0], 5), round(v[1], 5), round(v[2], 5)]


def parse_midi_notes(midi_path: str, max_time: float | None = None) -> tuple[float, list[dict[str, Any]]]:
    mid = mido.MidiFile(midi_path)
    tempo = 500000
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo = msg.tempo
            break
    bpm = mido.tempo2bpm(tempo)
    print(f"BPM: {bpm}, Ticks per beat: {mid.ticks_per_beat}")

    all_notes: list[dict[str, Any]] = []
    for track_idx in range(1, 13):
        if track_idx >= len(mid.tracks):
            break
        track = mid.tracks[track_idx]
        current_tick = 0
        active_notes: dict[int, tuple[float, int]] = {}
        for msg in track:
            current_tick += msg.time
            sec = mido.tick2second(current_tick, mid.ticks_per_beat, tempo)
            if msg.type == "note_on" and msg.velocity > 0:
                active_notes[msg.note] = (sec, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active_notes:
                    start_sec, vel = active_notes.pop(msg.note)
                    if max_time is not None and start_sec > max_time:
                        continue
                    cfg = INSTRUMENT_CONFIGS.get(
                        track_idx,
                        {"name": track.name, "color": [0.8, 0.8, 0.8], "material": "default", "type": "standard"},
                    )
                    all_notes.append(
                        {
                            "track_idx": track_idx,
                            "instrument": cfg["name"],
                            "instrument_type": cfg["type"],
                            "material": cfg["material"],
                            "color": cfg["color"],
                            "note": msg.note,
                            "pitch_name": note_name(msg.note),
                            "start_time": round(start_sec, 4),
                            "duration": round(sec - start_sec, 4),
                            "velocity": vel,
                        }
                    )

    all_notes.sort(key=lambda x: x["start_time"])
    return bpm, all_notes


def assign_anchors(notes: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    """Pitch -> X, time -> Z. Y fixed at HIT_Y for stable wall-mounted boards."""
    if not notes:
        return []
    min_pitch = min(n["note"] for n in notes)
    max_pitch = max(n["note"] for n in notes)
    pitch_span = max(1, max_pitch - min_pitch)
    anchors = []
    for note in notes:
        norm_p = (note["note"] - min_pitch) / pitch_span
        x = (norm_p - 0.5) * X_WIDTH
        z = -note["start_time"] * DESCENT_SPEED
        anchors.append((x, HIT_Y, z))
        note["pos"] = [round(x, 3), round(HIT_Y, 3), round(z, 3)]
        # keep legacy fields for debugging
    for i, note in enumerate(notes):
        if i + 1 < len(notes):
            dt = notes[i + 1]["start_time"] - note["start_time"]
            note["dt_next"] = round(dt, 4)
            if dt > 0.8:
                note["transition_to_next"] = "rail_glide"
            elif dt > 0.4:
                note["transition_to_next"] = "high_bounce"
            else:
                note["transition_to_next"] = "fast_bounce"
        else:
            note["dt_next"] = 0
            note["transition_to_next"] = "finish"
    return anchors


def arc_height(transition: str, dt: float, dx: float) -> float:
    if transition == "rail_glide":
        base = H_RAIL
    elif transition == "high_bounce":
        base = H_HIGH
    else:
        base = H_FAST
    h = base + H_DT_GAIN * dt
    # Near-vertical drops (same pitch) need a bit more bounce visibility
    if abs(dx) < 0.15:
        h += 0.08
    return max(H_MIN, min(H_MAX, h))


def sample_segment(p0, p1, t0: float, t1: float, transition: str) -> list[tuple[float, tuple[float, float, float]]]:
    """Dense time samples along one segment. Includes t0, excludes t1 (next segment owns it)."""
    dt = max(1e-6, t1 - t0)
    dx = p1[0] - p0[0]
    h = arc_height(transition, dt, dx)
    n_frames = max(1, int(round(dt * FPS)))
    out = []
    for i in range(n_frames):
        s = i / n_frames
        t = t0 + s * dt
        chord = lerp3(p0, p1, s)
        # Parabola peak at s=0.5; rail_glide stays nearly flat
        lift = 4.0 * h * s * (1.0 - s)
        pos = (chord[0], chord[1], chord[2] + lift)
        out.append((t, pos))
    return out


def build_path_samples(
    anchors: list[tuple[float, float, float]], notes: list[dict[str, Any]]
) -> tuple[list[tuple[float, tuple[float, float, float]]], tuple[float, float, float]]:
    """Full path including launch pre-roll. Returns (samples_with_t, launch_start_pos)."""
    if not anchors:
        return [], (0.0, HIT_Y, LAUNCH_Z_ABOVE)

    p0 = anchors[0]
    t0 = notes[0]["start_time"]
    launch_start = (p0[0] * 0.15, HIT_Y, p0[2] + LAUNCH_Z_ABOVE)
    launch_t0 = max(0.0, t0 - LAUNCH_PRE_ROLL)

    samples: list[tuple[float, tuple[float, float, float]]] = []

    # Launch: ease into first hit (quadratic drop in Z, linear XY)
    launch_frames = max(1, int(round((t0 - launch_t0) * FPS)))
    for i in range(launch_frames):
        s = i / launch_frames
        t = launch_t0 + s * (t0 - launch_t0)
        ease = s * s
        pos = lerp3(launch_start, p0, ease)
        samples.append((t, pos))

    for i in range(len(anchors) - 1):
        transition = notes[i]["transition_to_next"]
        seg = sample_segment(
            anchors[i],
            anchors[i + 1],
            notes[i]["start_time"],
            notes[i + 1]["start_time"],
            transition,
        )
        samples.extend(seg)

    # Exact final hit + hold
    t_last = notes[-1]["start_time"]
    samples.append((t_last, anchors[-1]))
    hold_end = t_last + 1.5
    hold_frames = max(1, int(round(1.5 * FPS)))
    for i in range(1, hold_frames + 1):
        t = t_last + i / FPS
        if t > hold_end:
            break
        samples.append((t, anchors[-1]))

    return samples, launch_start


def velocity_at_hit(
    samples: list[tuple[float, tuple[float, float, float]]],
    hit_t: float,
    hit_pos,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Estimate v_in and v_out near a hit by looking at nearby samples."""
    # Find closest sample index to hit_t
    best_i = 0
    best_d = abs(samples[0][0] - hit_t)
    for i, (t, _) in enumerate(samples):
        d = abs(t - hit_t)
        if d < best_d:
            best_d = d
            best_i = i

    # Look back / forward a few frames
    back = max(0, best_i - 3)
    fwd = min(len(samples) - 1, best_i + 3)
    p_back = samples[back][1]
    p_fwd = samples[fwd][1]
    v_in = vec_normalize(vec_sub(hit_pos, p_back), fallback=(0.0, 0.0, -1.0))
    v_out = vec_normalize(vec_sub(p_fwd, hit_pos), fallback=(0.0, 0.0, -1.0))
    return v_in, v_out


def bounce_normal(v_in, v_out) -> tuple[float, float, float]:
    """Specular surface normal that reflects v_in into v_out: n ∝ (-v_in + v_out)."""
    n = vec_add(vec_scale(v_in, -1.0), v_out)
    n = vec_normalize(n, fallback=(0.0, 0.0, 1.0))
    # Prefer upward-ish normals so boards face camera-up
    if n[2] < 0.15:
        n = vec_normalize(vec_add(n, (0.0, 0.0, 0.55)), fallback=(0.0, 0.0, 1.0))
    return n


def build_bounces(
    notes: list[dict[str, Any]],
    anchors: list[tuple[float, float, float]],
    samples: list[tuple[float, tuple[float, float, float]]],
) -> list[dict[str, Any]]:
    bounces = []
    for i, note in enumerate(notes):
        pos = anchors[i]
        t = note["start_time"]
        v_in, v_out = velocity_at_hit(samples, t, pos)
        n = bounce_normal(v_in, v_out)
        tangent_out = v_out
        bounces.append(
            {
                "note_index": i,
                "t": t,
                "frame": int(round(t * FPS)),
                "position": round3(pos),
                "normal": round3(n),
                "tangent_out": round3(tangent_out),
                "v_in": round3(v_in),
                "transition_to_next": note["transition_to_next"],
                "dt_next": note["dt_next"],
                "track_idx": note["track_idx"],
                "instrument": note["instrument"],
                "instrument_type": note["instrument_type"],
                "material": note["material"],
                "color": note["color"],
                "pitch_name": note["pitch_name"],
                "note": note["note"],
                "velocity": note["velocity"],
            }
        )
    return bounces


def build_rails(
    notes: list[dict[str, Any]], anchors: list[tuple[float, float, float]]
) -> list[dict[str, Any]]:
    rails = []
    for i, note in enumerate(notes[:-1]):
        if note["transition_to_next"] != "rail_glide":
            continue
        p0, p1 = anchors[i], anchors[i + 1]
        # Mid control point slightly toward camera / down the chord
        mid = lerp3(p0, p1, 0.5)
        mid = (mid[0], mid[1] - 0.02, mid[2] + H_RAIL)
        points = [round3(p0), round3(mid), round3(p1)]
        rails.append(
            {
                "from_bounce": i,
                "to_bounce": i + 1,
                "t0": note["start_time"],
                "t1": notes[i + 1]["start_time"],
                "points": points,
            }
        )
    return rails


def densify_frame_samples(
    samples: list[tuple[float, tuple[float, float, float]]], end_t: float
) -> list[dict[str, Any]]:
    """One sample per frame from 0..end_frame via linear interp in time."""
    if not samples:
        return []
    # Sort and dedupe by time
    samples = sorted(samples, key=lambda x: x[0])
    end_frame = int(round(end_t * FPS))
    out = []
    j = 0
    for f in range(0, end_frame + 1):
        t = f / FPS
        while j + 1 < len(samples) and samples[j + 1][0] < t:
            j += 1
        if j + 1 < len(samples):
            t0, p0 = samples[j]
            t1, p1 = samples[j + 1]
            if t1 <= t0:
                pos = p0
            else:
                s = (t - t0) / (t1 - t0)
                s = max(0.0, min(1.0, s))
                pos = lerp3(p0, p1, s)
        else:
            pos = samples[-1][1]
        out.append({"t": round(t, 5), "frame": f, "position": round3(pos)})
    return out


def assemble_trajectory(bpm: float, notes: list[dict[str, Any]]) -> dict[str, Any]:
    anchors = assign_anchors(notes)
    raw_samples, launch_start = build_path_samples(anchors, notes)
    bounces = build_bounces(notes, anchors, raw_samples)
    rails = build_rails(notes, anchors)

    end_t = (notes[-1]["start_time"] + 1.5) if notes else 1.0
    frame_samples = densify_frame_samples(raw_samples, end_t)

    for b in bounces:
        f = b["frame"]
        if 0 <= f < len(frame_samples):
            frame_samples[f]["position"] = b["position"]

    return {
        "schema": "marble_trajectory_v1",
        "bpm": bpm,
        "fps": FPS,
        "descent_speed": DESCENT_SPEED,
        "marble_radius": MARBLE_RADIUS,
        "x_width": X_WIDTH,
        "hit_y": HIT_Y,
        "launch_start": round3(launch_start),
        "total_notes": len(notes),
        "notes": notes,
        "bounces": bounces,
        "rails": rails,
        "samples": frame_samples,
    }


def save_trajectory(data: dict[str, Any], output_json: str) -> dict[str, Any]:
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(
        f"Saved trajectory: {len(data['bounces'])} bounces, "
        f"{len(data['samples'])} samples, {len(data['rails'])} rails -> {output_json}"
    )
    return data


def build_trajectory(midi_path: str, output_json: str, max_time: float | None = None) -> dict[str, Any]:
    bpm, notes = parse_midi_notes(midi_path, max_time=max_time)
    return save_trajectory(assemble_trajectory(bpm, notes), output_json)


def upgrade_notes_json(input_json: str, output_json: str | None = None) -> dict[str, Any]:
    """Rebuild trajectory fields from an existing notes JSON (no MIDI required)."""
    with open(input_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("schema") == "marble_trajectory_v1" and raw.get("bounces") and raw.get("samples"):
        print(f"Already trajectory schema: {input_json}")
        return raw
    notes = raw["notes"]
    # Drop stale pos/transition; assign_anchors recomputes
    for n in notes:
        n.pop("pos", None)
        n.pop("dt_next", None)
        n.pop("transition_to_next", None)
    bpm = float(raw.get("bpm", 120.0))
    out = output_json or input_json
    return save_trajectory(assemble_trajectory(bpm, notes), out)


if __name__ == "__main__":
    import sys

    root = "projects/jingle-bells-marble-music/artifacts"
    # Default: upgrade existing note dumps (works without Downloads MIDI access)
    if len(sys.argv) > 1 and sys.argv[1] == "--from-midi":
        midi = sys.argv[2] if len(sys.argv) > 2 else (
            "/Users/luca/Downloads/jingle_bells_12_marble_instruments_midi/"
            "02_Jingle_Bells_12_Instruments_12Tracks_Split.mid"
        )
        build_trajectory(midi, f"{root}/sample_14s_notes.json", max_time=14.5)
        build_trajectory(midi, f"{root}/full_notes.json")
    else:
        upgrade_notes_json(f"{root}/sample_14s_notes.json")
        upgrade_notes_json(f"{root}/full_notes.json")
