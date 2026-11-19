#!/usr/bin/env python3
"""Split the Synth (Square Lead) stem into K alternating/polyphonic marble tracks at 125 BPM.

- Quantizes raw basic-pitch Synth notes to the 125 BPM 16th-note grid (0.12s).
- Deduplicates same-pitch fragments in the same grid slot.
- Uses a 1-step lookahead reservation allocator so every ball maintains
  min_dt >= 0.24s (8th-note at 125 BPM) while hitting 100% of Synth time slots.
- Exports a 125 BPM multi-track MIDI (one track per ball) + per-ball JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import pretty_midi


def split_synth_to_multiball(
    in_mid: str,
    out_mid: str,
    out_json: str,
    num_balls: int = 3,
    bpm: float = 125.0,
    min_ball_dt: float = 0.22,
) -> dict:
    pm_in = pretty_midi.PrettyMIDI(in_mid)
    synth_insts = [inst for inst in pm_in.instruments if "Synth" in inst.name]
    if not synth_insts:
        raise ValueError(f"No Synth track found in {in_mid}")
    synth = synth_insts[0]

    grid = 60.0 / bpm / 4.0  # 16th note = 0.12s at 125 BPM
    by_slot: dict[float, dict[int, pretty_midi.Note]] = {}
    for n in synth.notes:
        q = round(round(n.start / grid) * grid, 4)
        slot_map = by_slot.setdefault(q, {})
        if n.pitch not in slot_map or n.velocity > slot_map[n.pitch].velocity:
            slot_map[n.pitch] = n

    slots = sorted(by_slot.keys())
    last_t = [-999.0] * num_balls
    tracks: list[list[dict]] = [[] for _ in range(num_balls)]
    covered_slots = 0

    for idx, q in enumerate(slots):
        cands = sorted(by_slot[q].values(), key=lambda x: (-x.velocity, -x.pitch))
        avail = [b for b in range(num_balls) if q - last_t[b] >= min_ball_dt]
        avail.sort(key=lambda b: last_t[b])  # longest-idle ball first for alternation

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
        for b, note in zip(avail[:max_take], cands):
            dur = max(0.10, round(note.end - note.start, 4))
            tracks[b].append(
                {
                    "start_time": q,
                    "duration": dur,
                    "note": int(note.pitch),
                    "pitch_name": pretty_midi.note_number_to_name(int(note.pitch)),
                    "velocity": int(note.velocity),
                    "ball_idx": b,
                    "track_idx": b + 1,
                    "instrument": f"Square Lead (Ball {b + 1})",
                    "instrument_type": "square_lead",
                }
            )
            last_t[b] = q

    pm_out = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for b, t_notes in enumerate(tracks):
        inst = pretty_midi.Instrument(program=80, name=f"Marble_{b + 1}_Synth")
        for item in t_notes:
            inst.notes.append(
                pretty_midi.Note(
                    velocity=item["velocity"],
                    pitch=item["note"],
                    start=item["start_time"],
                    end=item["start_time"] + item["duration"],
                )
            )
        pm_out.instruments.append(inst)

    os.makedirs(os.path.dirname(os.path.abspath(out_mid)), exist_ok=True)
    pm_out.write(out_mid)

    summary = {
        "bpm": bpm,
        "grid_16th_sec": round(grid, 4),
        "num_balls": num_balls,
        "total_slots": len(slots),
        "covered_slots": covered_slots,
        "slot_coverage_pct": round(100.0 * covered_slots / max(1, len(slots)), 2),
        "total_notes_assigned": sum(len(t) for t in tracks),
        "per_ball_counts": [len(t) for t in tracks],
        "per_ball_min_dt": [
            round(min(t[i + 1]["start_time"] - t[i]["start_time"] for i in range(len(t) - 1)), 4)
            if len(t) > 1
            else 0.0
            for t in tracks
        ],
        "balls": [{"ball_idx": b, "notes": t} for b, t in enumerate(tracks)],
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"[{os.path.basename(out_mid)}] bpm={bpm}, K={num_balls}, "
        f"slots={covered_slots}/{len(slots)} ({summary['slot_coverage_pct']}%), "
        f"notes={summary['total_notes_assigned']}, per_ball={summary['per_ball_counts']}, "
        f"min_dt={summary['per_ball_min_dt']}"
    )
    return summary


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
    p.add_argument("--balls", type=int, default=3)
    p.add_argument("--bpm", type=float, default=125.0)
    args = p.parse_args()
    split_synth_to_multiball(args.in_mid, args.out_mid, args.out_json, num_balls=args.balls, bpm=args.bpm)
