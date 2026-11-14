#!/usr/bin/env python3
"""Prepare 64-Instrument Encyclopedia notes JSON, timeline summary, and FluidSynth WAV audio."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mido

PROJ = Path(__file__).resolve().parents[1]
MIDI_RELAY = Path("/Users/luca/Downloads/jingle_bells_64_instruments_encyclopedia/01_Jingle_Bells_64_Instruments_SingleStaff_Relay.mid")
MIDI_AUDIO = Path("/Users/luca/Downloads/jingle_bells_64_instruments_encyclopedia/02_Jingle_Bells_64_Instruments_Family_MultiTrack.mid")
SF2_PATH = Path("/Users/luca/OpenMontage/assets/audio/soundfonts/GeneralUser-GS.sf2")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

GM_INFO: dict[int, tuple[str, str, int, str]] = {
    # prog: (Display Name, archetype_for_blender, family_idx, family_name)
    0: ("Acoustic Grand Piano", "piano", 1, "Pianos & Mallets"),
    2: ("Electric Grand Piano", "piano", 1, "Pianos & Mallets"),
    4: ("Electric Piano", "piano", 1, "Pianos & Mallets"),
    6: ("Harpsichord", "piano", 1, "Pianos & Mallets"),
    8: ("Celesta", "glockenspiel", 1, "Pianos & Mallets"),
    9: ("Glockenspiel", "glockenspiel", 1, "Pianos & Mallets"),
    10: ("Music Box", "glockenspiel", 1, "Pianos & Mallets"),
    11: ("Vibraphone", "xylophone", 1, "Pianos & Mallets"),
    12: ("Marimba", "xylophone", 1, "Pianos & Mallets"),
    13: ("Xylophone", "xylophone", 1, "Pianos & Mallets"),
    14: ("Tubular Bells", "tubular_bells", 1, "Pianos & Mallets"),
    15: ("Dulcimer", "guitar", 1, "Pianos & Mallets"),

    16: ("Drawbar Organ", "organ", 2, "Organs, Guitars & Harp"),
    19: ("Church Organ", "organ", 2, "Organs, Guitars & Harp"),
    21: ("Accordion", "accordion", 2, "Organs, Guitars & Harp"),
    22: ("Harmonica", "accordion", 2, "Organs, Guitars & Harp"),
    24: ("Nylon Guitar", "guitar", 2, "Organs, Guitars & Harp"),
    25: ("Steel Guitar", "guitar", 2, "Organs, Guitars & Harp"),
    27: ("Clean Electric Guitar", "guitar", 2, "Organs, Guitars & Harp"),
    30: ("Distortion Guitar", "guitar", 2, "Organs, Guitars & Harp"),
    32: ("Acoustic Bass", "guitar", 2, "Organs, Guitars & Harp"),
    35: ("Fretless Bass", "guitar", 2, "Organs, Guitars & Harp"),
    45: ("Pizzicato Strings", "strings", 2, "Organs, Guitars & Harp"),
    46: ("Orchestral Harp", "guitar", 2, "Organs, Guitars & Harp"),

    40: ("Violin", "strings", 3, "Strings & Choir"),
    41: ("Viola", "strings", 3, "Strings & Choir"),
    42: ("Cello", "strings", 3, "Strings & Choir"),
    43: ("Contrabass", "strings", 3, "Strings & Choir"),
    48: ("String Ensemble", "strings", 3, "Strings & Choir"),
    52: ("Choir Aahs", "choir", 3, "Strings & Choir"),
    53: ("Voice Oohs", "choir", 3, "Strings & Choir"),
    55: ("Orchestra Hit", "gong", 3, "Strings & Choir"),

    56: ("Trumpet", "brass", 4, "Brass & Saxophones"),
    57: ("Trombone", "brass", 4, "Brass & Saxophones"),
    58: ("Tuba", "brass", 4, "Brass & Saxophones"),
    59: ("Muted Trumpet", "brass", 4, "Brass & Saxophones"),
    60: ("French Horn", "brass", 4, "Brass & Saxophones"),
    61: ("Brass Section", "brass", 4, "Brass & Saxophones"),
    64: ("Soprano Sax", "brass", 4, "Brass & Saxophones"),
    65: ("Alto Sax", "brass", 4, "Brass & Saxophones"),
    66: ("Tenor Sax", "brass", 4, "Brass & Saxophones"),

    68: ("Oboe", "flute", 5, "Woodwinds & Flutes"),
    69: ("English Horn", "flute", 5, "Woodwinds & Flutes"),
    70: ("Bassoon", "flute", 5, "Woodwinds & Flutes"),
    71: ("Clarinet", "flute", 5, "Woodwinds & Flutes"),
    72: ("Piccolo", "flute", 5, "Woodwinds & Flutes"),
    73: ("Flute", "flute", 5, "Woodwinds & Flutes"),
    74: ("Recorder", "flute", 5, "Woodwinds & Flutes"),
    75: ("Pan Flute", "flute", 5, "Woodwinds & Flutes"),
    77: ("Shakuhachi", "flute", 5, "Woodwinds & Flutes"),
    78: ("Whistle", "flute", 5, "Woodwinds & Flutes"),
    79: ("Ocarina", "flute", 5, "Woodwinds & Flutes"),

    104: ("Sitar", "guitar", 6, "World Ethnic & Synth"),
    105: ("Banjo", "guitar", 6, "World Ethnic & Synth"),
    106: ("Shamisen", "guitar", 6, "World Ethnic & Synth"),
    107: ("Koto", "guitar", 6, "World Ethnic & Synth"),
    108: ("Kalimba", "xylophone", 6, "World Ethnic & Synth"),
    109: ("Bagpipe", "flute", 6, "World Ethnic & Synth"),
    111: ("Shanai", "flute", 6, "World Ethnic & Synth"),
    114: ("Steel Drums", "steel_drum", 6, "World Ethnic & Synth"),
    115: ("Woodblock", "castanet", 6, "World Ethnic & Synth"),
    80: ("Square Lead", "synth", 6, "World Ethnic & Synth"),
    81: ("Sawtooth Lead", "synth", 6, "World Ethnic & Synth"),
    88: ("New Age Pad", "synth", 6, "World Ethnic & Synth"),
}

ARCHETYPE_TO_TEMPLATE_ID = {
    "castanet": 1,
    "piano": 2,
    "organ": 2,
    "accordion": 2,
    "synth": 3,
    "tubular_bells": 4,
    "brass": 4,
    "choir": 4,
    "gong": 4,
    "marimba": 5,
    "kalimba": 6,
    "flute": 7,
    "xylophone": 8,
    "ocarina": 9,
    "glockenspiel": 10,
    "steel_drum": 11,
    "guitar": 12,
    "strings": 12,
}

ARCHETYPE_STYLE = {
    "piano": ("ivory_ebony", [0.92, 0.90, 0.84]),
    "glockenspiel": ("silver_chrome", [0.85, 0.88, 0.92]),
    "xylophone": ("wood_warm", [0.76, 0.48, 0.24]),
    "tubular_bells": ("brass_gold", [0.86, 0.71, 0.34]),
    "guitar": ("wood_spruce", [0.78, 0.56, 0.32]),
    "organ": ("brass_bronze", [0.72, 0.54, 0.30]),
    "accordion": ("crimson_bellows", [0.68, 0.18, 0.18]),
    "strings": ("wood_maple", [0.72, 0.36, 0.16]),
    "choir": ("ivory_gold", [0.90, 0.84, 0.72]),
    "gong": ("brass_gold", [0.88, 0.70, 0.30]),
    "brass": ("brass_gold", [0.90, 0.74, 0.32]),
    "flute": ("silver_bamboo", [0.80, 0.76, 0.64]),
    "steel_drum": ("steel_blue", [0.48, 0.58, 0.68]),
    "castanet": ("wood_dark", [0.58, 0.32, 0.16]),
    "synth": ("emerald_synth", [0.20, 0.82, 0.58]),
}


def midi_note_to_name(note: int) -> str:
    return f"{NOTE_NAMES[note % 12]}{(note // 12) - 1}"


def main() -> None:
    (PROJ / "artifacts").mkdir(parents=True, exist_ok=True)
    (PROJ / "assets" / "audio").mkdir(parents=True, exist_ok=True)

    mid = mido.MidiFile(str(MIDI_RELAY))
    tempo_events = []
    t_tick = 0
    for msg in mid.tracks[0]:
        t_tick += msg.time
        if msg.type == "set_tempo":
            tempo_events.append((t_tick, msg.tempo))

    bpm = mido.tempo2bpm(tempo_events[0][1]) if tempo_events else 120.0

    def tick_to_sec(tick: int) -> float:
        sec = 0.0
        last_t = 0
        cur_tempo = 500000
        for tt, tp in tempo_events:
            if tick <= tt:
                break
            sec += mido.tick2second(tt - last_t, mid.ticks_per_beat, cur_tempo)
            last_t = tt
            cur_tempo = tp
        sec += mido.tick2second(tick - last_t, mid.ticks_per_beat, cur_tempo)
        return sec

    # Parse lead track (track 1)
    tr = mid.tracks[1]
    t_tick = 0
    cur_prog = 0
    active_notes: dict[int, tuple[float, int, int]] = {}
    events = []
    seen_progs: list[int] = []

    for msg in tr:
        t_tick += msg.time
        sec = tick_to_sec(t_tick)
        if msg.type == "program_change":
            cur_prog = msg.program
        elif msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note] = (sec, msg.velocity, cur_prog)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active_notes:
                t0, vel, prog = active_notes.pop(msg.note)
                if prog not in seen_progs:
                    seen_progs.append(prog)
                events.append((t0, max(0.05, sec - t0), msg.note, vel, prog))

    events.sort(key=lambda x: x[0])
    prog_to_enc_idx = {p: i + 1 for i, p in enumerate(seen_progs)}

    notes_out = []
    inst_timeline = []
    last_prog = -1
    for idx, (t0, dur, note, vel, prog) in enumerate(events):
        name, archetype, fam_idx, fam_name = GM_INFO.get(prog, (f"GM#{prog}", "piano", 1, "Pianos & Mallets"))
        mat_name, col = ARCHETYPE_STYLE.get(archetype, ("ivory_ebony", [0.85, 0.85, 0.85]))
        dt_next = round(events[idx + 1][0] - t0, 4) if idx + 1 < len(events) else 1.5
        enc_idx = prog_to_enc_idx[prog]

        if prog != last_prog:
            inst_timeline.append({
                "encyclopedia_idx": enc_idx,
                "program": prog,
                "instrument": name,
                "archetype": archetype,
                "family_idx": fam_idx,
                "family_name": fam_name,
                "start_time": round(t0, 4),
                "note_index": idx,
            })
            last_prog = prog

        notes_out.append({
            "track_idx": ARCHETYPE_TO_TEMPLATE_ID.get(archetype, 2),
            "instrument": name,
            "instrument_type": archetype,
            "material": mat_name,
            "color": col,
            "note": note,
            "pitch_name": midi_note_to_name(note),
            "start_time": round(t0, 4),
            "duration": round(dur, 4),
            "velocity": vel,
            "dt_next": dt_next,
            "encyclopedia_idx": enc_idx,
            "family_idx": fam_idx,
            "family_name": fam_name,
        })

    payload = {
        "schema": "marble_trajectory_v1",
        "bpm": bpm,
        "fps": 60,
        "descent_speed": 2.2,
        "marble_radius": 0.12,
        "x_width": 2.4,
        "hit_y": 0.0,
        "launch_start": [0.09, 0.0, 1.8],
        "total_notes": len(notes_out),
        "notes": notes_out,
    }
    notes_json = PROJ / "artifacts" / "full_64_notes.json"
    notes_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    timeline_json = PROJ / "artifacts" / "encyclopedia_64_timeline.json"
    timeline_json.write_text(json.dumps({
        "total_notes": len(notes_out),
        "total_instruments": len(seen_progs),
        "duration_sec": round(mid.length, 3),
        "switches": inst_timeline,
    }, indent=2), encoding="utf-8")

    print(f"Wrote {notes_json} ({len(notes_out)} notes, {len(seen_progs)} instruments)")
    print(f"Wrote {timeline_json} ({len(inst_timeline)} instrument switches)")

    # Synthesize WAV via FluidSynth
    raw_wav = PROJ / "assets" / "audio" / "full_64.wav"
    preroll_wav = PROJ / "assets" / "audio" / "full_64_preroll.wav"
    if not raw_wav.exists():
        print("Synthesizing full_64.wav via FluidSynth...")
        subprocess.run(
            ["fluidsynth", "-ni", "-F", str(raw_wav), "-r", "44100", str(SF2_PATH), str(MIDI_AUDIO)],
            check=True,
        )
    if not preroll_wav.exists():
        print("Creating full_64_preroll.wav (+0.90s leading silence)...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_wav), "-af", "adelay=900|900", str(preroll_wav)],
            check=True,
        )
    print("Audio ready:", preroll_wav)


if __name__ == "__main__":
    main()
