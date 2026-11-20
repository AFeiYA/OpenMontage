#!/usr/bin/env python3
"""Split full-length Synth + Keyboard stems into 3 alternating/polyphonic marble tracks at 125 BPM,
and export the 7-track Ensemble MIDI & JSON for the full 101.76s Christmas multi-instrument MV.

Phase alignment:
  - Estimates circular phase offset mod 16th-note grid (0.12s @ 125 BPM) from Drums Kick + Synth
    so quantized hits stay cleanly synced (< 35ms mean error) with the mixed WAV audio.

Exports:
  1. `assets/audio/glass_marbles_synth_3ball_125bpm.mid` + `artifacts/glass_marbles_synth_3ball_notes.json`
  2. `assets/audio/glass_marbles_ensemble_125bpm.mid` + `artifacts/glass_marbles_ensemble_notes.json`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import pretty_midi


def _find_track(pm: pretty_midi.PrettyMIDI, keyword: str) -> pretty_midi.Instrument | None:
    for inst in pm.instruments:
        if keyword.lower() in inst.name.lower():
            return inst
    return None


def _estimate_grid_phase(times: list[float], grid: float) -> float:
    """Circular mean phase modulo `grid` so 16th-note snapping aligns with true beat phase."""
    if not times:
        return 0.0
    cs = sum(math.cos(2.0 * math.pi * (t % grid) / grid) for t in times)
    sn = sum(math.sin(2.0 * math.pi * (t % grid) / grid) for t in times)
    ang = math.atan2(sn, cs)
    if ang < 0.0:
        ang += 2.0 * math.pi
    return round((ang / (2.0 * math.pi)) * grid, 4)


def _snap_to_grid(t: float, grid: float, phase: float) -> float:
    q = round(phase + round((t - phase) / grid) * grid, 4)
    if q < 0.0:
        q = round(q + grid, 4)
    return q


def _quantize_rhythm_track(
    inst: pretty_midi.Instrument | None,
    grid: float,
    phase: float,
    min_dt: float,
    pitch_priority: dict[int, int] | None = None,
    default_dur: float = 0.12,
) -> tuple[list[dict], list[float]]:
    """Quantize a drum/percussion track onto the phase grid while enforcing strict `min_dt`."""
    if inst is None or not inst.notes:
        return [], []
    by_slot: dict[float, list[pretty_midi.Note]] = {}
    for n in inst.notes:
        q = _snap_to_grid(float(n.start), grid, phase)
        by_slot.setdefault(q, []).append(n)

    out: list[dict] = []
    errs: list[float] = []
    last_t = -999.0
    for q in sorted(by_slot.keys()):
        if q - last_t < min_dt - 1e-6:
            continue
        cands = by_slot[q]
        best = max(
            cands,
            key=lambda x: ((pitch_priority.get(int(x.pitch), 0) if pitch_priority else 0), int(x.velocity)),
        )
        errs.append(abs(q - float(best.start)))
        out.append({
            "start_time": q,
            "duration": round(max(default_dur, float(best.end - best.start)), 4),
            "note": int(best.pitch),
            "pitch_name": "Kick" if int(best.pitch) == 36 else (
                "Snare" if int(best.pitch) == 38 else (
                    "ClosedHat" if int(best.pitch) == 42 else "WoodPerc"
                )
            ),
            "velocity": int(best.velocity),
        })
        last_t = q
    return out, errs


def _quantize_bass_track(inst: pretty_midi.Instrument | None, grid: float, phase: float, min_dt: float = 0.20) -> list[dict]:
    if inst is None or not inst.notes:
        return []
    by_slot: dict[float, list[pretty_midi.Note]] = {}
    for n in inst.notes:
        q = _snap_to_grid(float(n.start), grid, phase)
        by_slot.setdefault(q, []).append(n)

    out: list[dict] = []
    last_t = -999.0
    for q in sorted(by_slot.keys()):
        if q - last_t < min_dt - 1e-6:
            continue
        best = max(by_slot[q], key=lambda x: (int(x.velocity), -int(x.pitch)))
        out.append({
            "start_time": q,
            "duration": round(max(0.12, float(best.end - best.start)), 4),
            "note": int(best.pitch),
            "pitch_name": pretty_midi.note_number_to_name(int(best.pitch)),
            "velocity": int(best.velocity),
        })
        last_t = q
    return out


def split_synth_to_multiball(
    in_mid: str,
    out_mid: str,
    out_json: str,
    out_ens_mid: str,
    out_ens_json: str,
    num_balls: int = 3,
    bpm: float = 125.0,
    min_ball_dt: float = 0.22,
) -> dict:
    pm_in = pretty_midi.PrettyMIDI(in_mid)
    synth = _find_track(pm_in, "Synth")
    if synth is None:
        raise ValueError(f"No Synth track found in {in_mid}")
    keyboard = _find_track(pm_in, "Keyboard")
    bass = _find_track(pm_in, "Bass")
    kick = _find_track(pm_in, "Kick")
    snare = _find_track(pm_in, "Snare")
    perc = _find_track(pm_in, "Percussion")

    grid = 60.0 / bpm / 4.0  # 16th note = 0.12s at 125 BPM
    phase_samples = [float(n.start) for n in (kick.notes if kick else [])]
    if not phase_samples:
        phase_samples = [float(n.start) for n in synth.notes]
    phase = _estimate_grid_phase(phase_samples, grid)

    # Fuse Synth (primary priority) + Keyboard (secondary harmonic fill) onto quantized 16th-note slots
    synth_slots: dict[float, dict[int, tuple[pretty_midi.Note, str]]] = {}
    for n in synth.notes:
        q = _snap_to_grid(float(n.start), grid, phase)
        smap = synth_slots.setdefault(q, {})
        if n.pitch not in smap or n.velocity > smap[n.pitch][0].velocity:
            smap[n.pitch] = (n, "synth")

    if keyboard is not None:
        for n in keyboard.notes:
            q = _snap_to_grid(float(n.start), grid, phase)
            smap = synth_slots.setdefault(q, {})
            if n.pitch not in smap:
                smap[n.pitch] = (n, "keyboard")
            elif smap[n.pitch][1] == "keyboard" and n.velocity > smap[n.pitch][0].velocity:
                smap[n.pitch] = (n, "keyboard")

    slots = sorted(synth_slots.keys())
    last_t = [-999.0] * num_balls
    tracks: list[list[dict]] = [[] for _ in range(num_balls)]
    covered_slots = 0
    melody_errs: list[float] = []

    for idx, q in enumerate(slots):
        raw_items = list(synth_slots[q].values())
        # Synth primary (0) before Keyboard secondary (1), then stronger velocity / higher melody pitch
        cands = sorted(
            raw_items,
            key=lambda item: (0 if item[1] == "synth" else 1, -int(item[0].velocity), -int(item[0].pitch)),
        )
        avail = [b for b in range(num_balls) if q - last_t[b] >= min_ball_dt]
        avail.sort(key=lambda b: last_t[b])

        max_take = len(avail)
        if idx + 1 < len(slots) and slots[idx + 1] - q < min_ball_dt:
            q_nxt = slots[idx + 1]
            already_free_nxt = sum(
                1 for b in range(num_balls) if b not in avail and q_nxt - last_t[b] >= min_ball_dt
            )
            if already_free_nxt == 0 and len(avail) > 1:
                max_take = len(avail) - 1

        if avail and cands:
            covered_slots += 1
        for b, (note, src_stem) in zip(avail[:max_take], cands):
            dur = max(0.10, round(float(note.end - note.start), 4))
            melody_errs.append(abs(q - float(note.start)))
            tracks[b].append(
                {
                    "start_time": q,
                    "duration": dur,
                    "note": int(note.pitch),
                    "pitch_name": pretty_midi.note_number_to_name(int(note.pitch)),
                    "velocity": int(note.velocity),
                    "ball_idx": b,
                    "track_idx": b + 1,
                    "source_stem": src_stem,
                    "instrument": f"{'Square Lead' if src_stem == 'synth' else 'Grand Piano'} (Ball {b + 1})",
                    "instrument_type": "square_lead" if src_stem == "synth" else "keyboard_fill",
                }
            )
            last_t[b] = q

    # Export updated 3-ball melody MIDI & JSON (full 101.76s)
    pm_3ball = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for b, t_notes in enumerate(tracks):
        inst = pretty_midi.Instrument(program=80, name=f"Marble_{b + 1}_Melody")
        for item in t_notes:
            inst.notes.append(
                pretty_midi.Note(
                    velocity=item["velocity"],
                    pitch=item["note"],
                    start=item["start_time"],
                    end=item["start_time"] + item["duration"],
                )
            )
        pm_3ball.instruments.append(inst)

    os.makedirs(os.path.dirname(os.path.abspath(out_mid)), exist_ok=True)
    pm_3ball.write(out_mid)

    melody_dts = [
        round(min(t[i + 1]["start_time"] - t[i]["start_time"] for i in range(len(t) - 1)), 4) if len(t) > 1 else 0.0
        for t in tracks
    ]
    summary = {
        "bpm": bpm,
        "phase_offset_sec": phase,
        "grid_16th_sec": round(grid, 4),
        "num_balls": num_balls,
        "total_slots": len(slots),
        "covered_slots": covered_slots,
        "slot_coverage_pct": round(100.0 * covered_slots / max(1, len(slots)), 2),
        "total_notes_assigned": sum(len(t) for t in tracks),
        "per_ball_counts": [len(t) for t in tracks],
        "per_ball_min_dt": melody_dts,
        "synth_vs_keyboard_counts": {
            "synth": sum(1 for t in tracks for item in t if item["source_stem"] == "synth"),
            "keyboard": sum(1 for t in tracks for item in t if item["source_stem"] == "keyboard"),
        },
        "mean_quant_err_sec": round(statistics.mean(melody_errs), 4) if melody_errs else 0.0,
        "balls": [{"ball_idx": b, "notes": t} for b, t in enumerate(tracks)],
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Quantize rhythmic + bass companion tracks onto the same 125 BPM phase grid
    kick_notes, kick_errs = _quantize_rhythm_track(kick, grid, phase, min_dt=0.24, default_dur=0.14)
    snare_notes, snare_errs = _quantize_rhythm_track(
        snare, grid, phase, min_dt=0.24, pitch_priority={38: 2, 42: 1}, default_dur=0.12
    )
    bass_notes = _quantize_bass_track(bass, grid, phase, min_dt=0.20)
    perc_notes, perc_errs = _quantize_rhythm_track(perc, grid, phase, min_dt=0.11, default_dur=0.10)
    melody_max_t = max((it["start_time"] for tr in tracks for it in tr), default=100.8)
    perc_filtered = [(n, e) for n, e in zip(perc_notes, perc_errs) if n["start_time"] <= melody_max_t]
    perc_notes = [n for n, _ in perc_filtered]
    perc_errs = [e for _, e in perc_filtered]

    # Export 7-track Ensemble MIDI
    pm_ens = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for inst in pm_3ball.instruments:
        pm_ens.instruments.append(inst)

    def _add_ens_inst(name: str, program: int, is_drum: bool, items: list[dict]):
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
        for it in items:
            inst.notes.append(
                pretty_midi.Note(
                    velocity=int(it["velocity"]),
                    pitch=int(it["note"]),
                    start=float(it["start_time"]),
                    end=float(it["start_time"]) + float(it["duration"]),
                )
            )
        pm_ens.instruments.append(inst)

    _add_ens_inst("Drum_Left_Kick", 0, True, kick_notes)
    _add_ens_inst("Drum_Right_SnareHat", 0, True, snare_notes)
    _add_ens_inst("Bass_Resonator", 32, False, bass_notes)
    _add_ens_inst("Percussion_Lights", 0, True, perc_notes)

    os.makedirs(os.path.dirname(os.path.abspath(out_ens_mid)), exist_ok=True)
    pm_ens.write(out_ens_mid)

    ens_json = {
        "bpm": bpm,
        "phase_offset_sec": phase,
        "grid_16th_sec": round(grid, 4),
        "melody_3ball": {
            "total_slots": len(slots),
            "covered_slots": covered_slots,
            "slot_coverage_pct": summary["slot_coverage_pct"],
            "per_ball_counts": summary["per_ball_counts"],
            "per_ball_min_dt": melody_dts,
            "source_counts": summary["synth_vs_keyboard_counts"],
            "mean_quant_err_sec": summary["mean_quant_err_sec"],
            "balls": summary["balls"],
        },
        "rhythm_and_bass": {
            "Drum_Left_Kick": {"count": len(kick_notes), "mean_quant_err_sec": round(statistics.mean(kick_errs), 4) if kick_errs else 0.0, "notes": kick_notes},
            "Drum_Right_SnareHat": {"count": len(snare_notes), "mean_quant_err_sec": round(statistics.mean(snare_errs), 4) if snare_errs else 0.0, "notes": snare_notes},
            "Bass_Resonator": {"count": len(bass_notes), "notes": bass_notes},
            "Percussion_Lights": {"count": len(perc_notes), "mean_quant_err_sec": round(statistics.mean(perc_errs), 4) if perc_errs else 0.0, "notes": perc_notes},
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_ens_json)), exist_ok=True)
    with open(out_ens_json, "w", encoding="utf-8") as f:
        json.dump(ens_json, f, indent=2, ensure_ascii=False)

    print(
        f"[{os.path.basename(out_mid)}] bpm={bpm}, phase={phase:.4f}s, "
        f"slots={covered_slots}/{len(slots)} ({summary['slot_coverage_pct']}%), "
        f"melody={summary['per_ball_counts']} (min_dt={melody_dts}, src={summary['synth_vs_keyboard_counts']})"
    )
    print(
        f"[{os.path.basename(out_ens_mid)}] Kick={len(kick_notes)}, SnareHat={len(snare_notes)}, "
        f"Bass={len(bass_notes)}, PercLights={len(perc_notes)}"
    )
    return ens_json


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in-mid",
        default="projects/jingle-bells-marble-music/assets/audio/glass_marbles_on_wood_multitrack.mid",
    )
    p.add_argument(
        "--out-mid",
        default="projects/jingle-bells-marble-music/assets/audio/glass_marbles_synth_3ball_125bpm.mid",
    )
    p.add_argument(
        "--out-json",
        default="projects/jingle-bells-marble-music/artifacts/glass_marbles_synth_3ball_notes.json",
    )
    p.add_argument(
        "--out-ens-mid",
        default="projects/jingle-bells-marble-music/assets/audio/glass_marbles_ensemble_125bpm.mid",
    )
    p.add_argument(
        "--out-ens-json",
        default="projects/jingle-bells-marble-music/artifacts/glass_marbles_ensemble_notes.json",
    )
    p.add_argument("--balls", type=int, default=3)
    p.add_argument("--bpm", type=float, default=125.0)
    args = p.parse_args()
    split_synth_to_multiball(
        args.in_mid,
        args.out_mid,
        args.out_json,
        args.out_ens_mid,
        args.out_ens_json,
        num_balls=args.balls,
        bpm=args.bpm,
    )
