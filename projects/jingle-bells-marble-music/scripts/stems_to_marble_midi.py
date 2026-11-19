#!/usr/bin/env python3
"""
Convert separated Stems of 'Track 02：木轨上的玻璃珠 (Glass Marbles on Wood) Stems'
into:
  1. Mixed full stereo WAV audio (`assets/audio/glass_marbles_on_wood_full.wav`)
  2. Multi-track GM MIDI (`assets/audio/glass_marbles_on_wood_multitrack.mid`)
  3. Cleaned single-marble playable JSON (`artifacts/glass_marbles_on_wood_notes.json`)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pretty_midi
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

STEMS_DIR = "/Users/luca/Downloads/Track 02：木轨上的玻璃珠 (Glass Marbles on Wood) Stems"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MANIFEST_PATH = os.path.join(PROJECT_DIR, "assets", "instrument_manifest_v2.json")
OUT_WAV = os.path.join(PROJECT_DIR, "assets", "audio", "glass_marbles_on_wood_full.wav")
OUT_MID = os.path.join(PROJECT_DIR, "assets", "audio", "glass_marbles_on_wood_multitrack.mid")
OUT_JSON = os.path.join(PROJECT_DIR, "artifacts", "glass_marbles_on_wood_notes.json")

# (filename, track_idx, gm_program, priority_weight, min_pitch, max_pitch)
STEM_SPECS = [
    ("3 Keyboard.mp3", 1, 0, 1.35, 48, 96),
    ("4 Percussion.mp3", 2, 12, 1.25, 48, 96),
    ("2 Guitar.mp3", 3, 24, 1.20, 45, 92),
    ("5 Synth.mp3", 4, 80, 1.15, 48, 96),
    ("6 Other.mp3", 5, 11, 1.05, 48, 96),
    ("1 Bass.mp3", 6, 32, 0.70, 28, 67),
]

ALL_AUDIO_STEMS = [
    "0 Drums.mp3",
    "1 Bass.mp3",
    "2 Guitar.mp3",
    "3 Keyboard.mp3",
    "4 Percussion.mp3",
    "5 Synth.mp3",
    "6 Other.mp3",
]


def mix_stems_to_wav(stems_dir: str, out_wav: str) -> None:
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    for fn in ALL_AUDIO_STEMS:
        p = os.path.join(stems_dir, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing stem: {p}")
        cmd.extend(["-i", p])
    cmd.extend([
        "-filter_complex", f"amix=inputs={len(ALL_AUDIO_STEMS)}:duration=longest:normalize=0",
        "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", out_wav,
    ])
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg mix failed:\n{res.stderr}")
    print(f"[Mixed Audio] {out_wav}")


def main() -> None:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    by_prog = manifest.get("by_program", {})

    mix_stems_to_wav(STEMS_DIR, OUT_WAV)

    pm_out = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    raw_events: list[dict] = []

    for fn, track_idx, gm_prog, prio, p_min, p_max in STEM_SPECS:
        stem_path = os.path.join(STEMS_DIR, fn)
        inst_name = str(by_prog.get(str(gm_prog), {}).get("name", f"GM_{gm_prog}"))
        print(f"[Transcribing] {fn} -> Track {track_idx} GM {gm_prog} ({inst_name})...")
        _model_out, midi_data, note_events = predict(
            stem_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=0.52,
            frame_threshold=0.35,
            minimum_note_length=90.0,
            minimum_frequency=float(pretty_midi.note_number_to_hz(p_min)),
            maximum_frequency=float(pretty_midi.note_number_to_hz(p_max)),
        )

        pm_inst = pretty_midi.Instrument(program=gm_prog, is_drum=False, name=f"{fn} ({inst_name})")
        kept = 0
        for start_s, end_s, pitch, amp, _bends in note_events:
            dur_s = float(end_s - start_s)
            p_int = int(pitch)
            if dur_s < 0.08 or float(amp) < 0.25 or not (p_min <= p_int <= p_max):
                continue
            vel = max(45, min(127, int(round(float(amp) * 127.0))))
            pm_inst.notes.append(pretty_midi.Note(velocity=vel, pitch=p_int, start=float(start_s), end=float(end_s)))
            raw_events.append({
                "start_time": float(start_s),
                "duration": dur_s,
                "track_idx": track_idx,
                "gm_program": gm_prog,
                "instrument": inst_name,
                "note": p_int,
                "velocity": vel,
                "score": float(amp) * prio + 0.004 * p_int,
            })
            kept += 1
        pm_out.instruments.append(pm_inst)
        print(f"  -> kept {kept} notes from {fn}")

    os.makedirs(os.path.dirname(OUT_MID), exist_ok=True)
    pm_out.write(OUT_MID)
    print(f"[Saved MultiTrack MIDI] {OUT_MID} ({len(raw_events)} total raw notes)")

    # Filter into single-marble sequential melody events with strict min_dt >= 0.15s
    raw_events.sort(key=lambda e: (e["start_time"], -e["score"]))
    clean_notes: list[dict] = []
    idx = 0
    while idx < len(raw_events):
        t_anchor = raw_events[idx]["start_time"]
        if clean_notes and (t_anchor - clean_notes[-1]["start_time"]) < 0.15:
            idx += 1
            continue
        win = [raw_events[idx]]
        j = idx + 1
        while j < len(raw_events) and (raw_events[j]["start_time"] - t_anchor) < 0.15:
            win.append(raw_events[j])
            j += 1
        best = max(win, key=lambda e: e["score"])
        clean_notes.append({
            "start_time": round(best["start_time"], 4),
            "duration": round(best["duration"], 4),
            "track_idx": best["track_idx"],
            "gm_program": best["gm_program"],
            "instrument": best["instrument"],
            "note": best["note"],
            "velocity": best["velocity"],
        })
        idx = j

    dts = [clean_notes[i + 1]["start_time"] - clean_notes[i]["start_time"] for i in range(len(clean_notes) - 1)]
    counts: dict[str, int] = {}
    for n in clean_notes:
        counts[n["instrument"]] = counts.get(n["instrument"], 0) + 1

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_stems": STEMS_DIR,
                "midi_path": OUT_MID,
                "audio_wav": OUT_WAV,
                "total_notes": len(clean_notes),
                "duration_s": clean_notes[-1]["start_time"] if clean_notes else 0.0,
                "notes": clean_notes,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[Saved Marble Notes JSON] {OUT_JSON}: {len(clean_notes)} notes, "
        f"min_dt={min(dts):.4f}s, max_dt={max(dts):.4f}s, span={clean_notes[-1]['start_time']:.2f}s"
    )
    print(f"Instrument distribution: {counts}")


if __name__ == "__main__":
    main()
