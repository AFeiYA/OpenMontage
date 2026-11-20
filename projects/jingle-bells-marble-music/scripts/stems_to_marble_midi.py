#!/usr/bin/env python3
"""
Convert separated Stems of 'Track 02：木轨上的玻璃珠 (Glass Marbles on Wood) Stems'
into:
  1. Mixed full stereo WAV audio (`assets/audio/glass_marbles_on_wood_full.wav`)
  2. Multi-track GM MIDI (`assets/audio/glass_marbles_on_wood_multitrack.mid`, full 101.76s)
  3. Cleaned single-marble playable JSON (`artifacts/glass_marbles_on_wood_notes.json`)

Note: The source MP3 stems are VBR MP3s without accurate Xing length headers; loading
them directly through librosa/soundfile silently truncates tracks near ~59s. All stems
are decoded via ffmpeg to 44.1 kHz mono PCM WAV first so transcription spans 100% of 101.76s.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import wave

import librosa
import numpy as np
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from scipy.signal import butter, sosfiltfilt

DEFAULT_STEMS_DIR = os.path.expanduser(
    "~/Downloads/Track 02：木轨上的玻璃珠 (Glass Marbles on Wood) Stems"
)
STEMS_DIR = os.environ.get("MARBLE_STEMS_DIR", DEFAULT_STEMS_DIR)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MANIFEST_PATH = os.path.join(PROJECT_DIR, "assets", "instrument_manifest_v2.json")
DECODED_WAV_DIR = os.path.join(PROJECT_DIR, "artifacts", "decoded_stems_wav")
OUT_WAV = os.path.join(PROJECT_DIR, "assets", "audio", "glass_marbles_on_wood_full.wav")
OUT_MID = os.path.join(PROJECT_DIR, "assets", "audio", "glass_marbles_on_wood_multitrack.mid")
OUT_JSON = os.path.join(PROJECT_DIR, "artifacts", "glass_marbles_on_wood_notes.json")

# Pitched stems transcribed with basic_pitch:
# (filename, track_idx, gm_program, track_name, priority_weight, min_pitch, max_pitch)
STEM_SPECS = [
    ("5 Synth.mp3", 1, 80, "Synth Lead (Square Lead)", 1.35, 48, 96),
    ("3 Keyboard.mp3", 2, 0, "Keyboard (Acoustic Grand Piano)", 1.25, 48, 96),
    ("1 Bass.mp3", 3, 32, "Bass (Acoustic Bass)", 0.80, 28, 67),
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


def decode_stem_to_wav(mp3_path: str, out_wav: str, sr: int = 44100) -> str:
    """Decode VBR MP3 completely via ffmpeg into uncompressed 44.1kHz mono PCM WAV."""
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"Missing stem: {mp3_path}")
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", mp3_path,
        "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", out_wav,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {mp3_path}:\n{res.stderr}")
    return out_wav


def read_mono_wav_float32(wav_path: str) -> tuple[np.ndarray, int]:
    """Read uncompressed 16-bit mono PCM WAV to normalized float32 ndarray."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return y, sr


def _local_rms(y: np.ndarray, sr: int, t: float, pre: float = 0.01, post: float = 0.06) -> float:
    i0 = max(0, int((t - pre) * sr))
    i1 = min(len(y), int((t + post) * sr))
    if i1 <= i0:
        return 0.0
    return float(np.sqrt(np.mean(y[i0:i1] ** 2)))


def _detect_peak_times(signal: np.ndarray, sr: int, hop: int, delta: float, wait_s: float) -> np.ndarray:
    oenv = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop)
    peak_val = float(np.max(oenv))
    if peak_val <= 1e-8:
        return np.array([], dtype=np.float64)
    wait_f = max(1, int(round(wait_s * sr / hop)))
    peaks = librosa.util.peak_pick(
        oenv,
        pre_max=3,
        post_max=3,
        pre_avg=5,
        post_avg=5,
        delta=delta * peak_val,
        wait=wait_f,
    )
    return librosa.frames_to_time(peaks, sr=sr, hop_length=hop)


def extract_drum_tracks(drums_wav: str, perc_wav: str) -> tuple[pretty_midi.Instrument, pretty_midi.Instrument, pretty_midi.Instrument]:
    """Extract Kick, Snare/HiHat, and Percussion GM drum tracks across full stem duration."""
    y_d, sr = read_mono_wav_float32(drums_wav)
    y_lo = sosfiltfilt(butter(4, 130.0, btype="lowpass", fs=sr, output="sos"), y_d)
    y_mid = sosfiltfilt(butter(4, [300.0, 2800.0], btype="bandpass", fs=sr, output="sos"), y_d)
    y_hi = sosfiltfilt(butter(4, 4000.0, btype="highpass", fs=sr, output="sos"), y_d)

    # 1. Drums Kick (< 130 Hz, pitch 36)
    lo_all = max(1e-6, float(np.sqrt(np.mean(y_lo ** 2))))
    kick_times_raw = _detect_peak_times(y_lo, sr, hop=512, delta=0.16, wait_s=0.22)
    kick_events: list[tuple[float, float]] = []
    for t in kick_times_raw:
        r = _local_rms(y_lo, sr, float(t))
        if r >= 0.35 * lo_all:
            kick_events.append((float(t), r))

    lo_max = max([r for _, r in kick_events], default=lo_all)
    inst_kick = pretty_midi.Instrument(program=0, is_drum=True, name="Drums Kick")
    for t, r in kick_events:
        vel = int(round(65.0 + 62.0 * min(1.0, r / max(lo_max, 1e-6))))
        inst_kick.notes.append(pretty_midi.Note(velocity=max(65, min(127, vel)), pitch=36, start=t, end=t + 0.15))

    # 2. Drums Snare/HiHat (300..2800 Hz + >4000 Hz, pitch 38 Snare / 42 Closed HiHat)
    comb_sn = y_mid + 1.4 * y_hi
    comb_times = _detect_peak_times(comb_sn, sr, hop=512, delta=0.14, wait_s=0.20)
    snare_raw: list[tuple[float, int, float]] = []
    kick_t_arr = np.array([kt for kt, _ in kick_events], dtype=np.float64)
    for t_val in comb_times:
        t = float(t_val)
        lo_r = _local_rms(y_lo, sr, t, pre=0.02, post=0.05)
        mid_r = _local_rms(y_mid, sr, t, pre=0.02, post=0.05)
        hi_r = _local_rms(y_hi, sr, t, pre=0.02, post=0.05)
        if lo_r > 1.8 * hi_r and mid_r < 1.15 * lo_r:
            continue
        near_kick = kick_t_arr.size > 0 and float(np.min(np.abs(kick_t_arr - t))) < 0.07
        if near_kick and not (mid_r > 0.045 and hi_r > 0.040):
            continue
        if max(mid_r, hi_r) < 0.020:
            continue
        pitch = 38 if mid_r >= 0.032 else 42
        strength = max(mid_r, hi_r)
        snare_raw.append((t, pitch, strength))

    sn_max = max([s for _, _, s in snare_raw], default=0.08)
    inst_snare = pretty_midi.Instrument(program=0, is_drum=True, name="Drums Snare/HiHat")
    for t, pitch, strength in snare_raw:
        vel = int(round(65.0 + 62.0 * min(1.0, strength / max(sn_max, 1e-6))))
        inst_snare.notes.append(pretty_midi.Note(velocity=max(65, min(127, vel)), pitch=pitch, start=t, end=t + 0.12))

    # 3. Percussion (full-band onsets, pitch 76 Hi Wood Block)
    y_p, sr_p = read_mono_wav_float32(perc_wav)
    perc_all = max(1e-6, float(np.sqrt(np.mean(y_p ** 2))))
    perc_times = _detect_peak_times(y_p, sr_p, hop=512, delta=0.14, wait_s=0.11)
    perc_raw: list[tuple[float, float]] = []
    dur_p = len(y_p) / float(sr_p)
    for t_val in perc_times:
        t = float(t_val)
        if t > dur_p - 0.6:
            continue
        r = _local_rms(y_p, sr_p, t)
        if r >= 0.35 * perc_all:
            perc_raw.append((t, r))

    perc_max = max([r for _, r in perc_raw], default=perc_all)
    inst_perc = pretty_midi.Instrument(program=0, is_drum=True, name="Percussion")
    for t, r in perc_raw:
        vel = int(round(55.0 + 72.0 * min(1.0, r / max(perc_max, 1e-6))))
        inst_perc.notes.append(pretty_midi.Note(velocity=max(55, min(127, vel)), pitch=76, start=t, end=t + 0.10))

    print(
        f"[Drum/Perc Onsets] Kick={len(inst_kick.notes)}, "
        f"Snare/HiHat={len(inst_snare.notes)} "
        f"(snare={sum(1 for n in inst_snare.notes if n.pitch == 38)}, hat={sum(1 for n in inst_snare.notes if n.pitch == 42)}), "
        f"Percussion={len(inst_perc.notes)}"
    )
    return inst_kick, inst_snare, inst_perc


def main() -> None:
    stems_dir = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else STEMS_DIR
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    by_prog = manifest.get("by_program", {})

    mix_stems_to_wav(stems_dir, OUT_WAV)

    # Decode all active stems to 44.1kHz mono PCM WAV first to prevent VBR MP3 truncation
    decoded_wavs: dict[str, str] = {}
    for fn in ALL_AUDIO_STEMS:
        stem_base = os.path.splitext(fn)[0]
        wav_path = os.path.join(DECODED_WAV_DIR, f"{stem_base}.wav")
        decoded_wavs[fn] = decode_stem_to_wav(os.path.join(stems_dir, fn), wav_path, sr=44100)
    print(f"[Decoded Stems WAV] {DECODED_WAV_DIR} ({len(decoded_wavs)} stems)")

    pm_out = pretty_midi.PrettyMIDI(initial_tempo=125.0)
    raw_events: list[dict] = []

    for fn, track_idx, gm_prog, track_name, prio, p_min, p_max in STEM_SPECS:
        wav_path = decoded_wavs[fn]
        inst_label = str(by_prog.get(str(gm_prog), {}).get("name", f"GM_{gm_prog}"))
        print(f"[Transcribing] {os.path.basename(wav_path)} -> Track {track_idx} GM {gm_prog} ({track_name})...")
        _model_out, _midi_data, note_events = predict(
            wav_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=0.52,
            frame_threshold=0.35,
            minimum_note_length=90.0,
            minimum_frequency=float(pretty_midi.note_number_to_hz(p_min)),
            maximum_frequency=float(pretty_midi.note_number_to_hz(p_max)),
        )

        pm_inst = pretty_midi.Instrument(program=gm_prog, is_drum=False, name=track_name)
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
                "instrument": inst_label,
                "note": p_int,
                "velocity": vel,
                "score": float(amp) * prio + 0.004 * p_int,
            })
            kept += 1
        pm_out.instruments.append(pm_inst)
        span_end = max((n.start for n in pm_inst.notes), default=0.0)
        print(f"  -> kept {kept} notes from {fn} (end={span_end:.2f}s)")

    # Tracks 4, 5, 6: Kick, Snare/HiHat, Percussion
    inst_kick, inst_snare, inst_perc = extract_drum_tracks(
        decoded_wavs["0 Drums.mp3"],
        decoded_wavs["4 Percussion.mp3"],
    )
    pm_out.instruments.extend([inst_kick, inst_snare, inst_perc])

    os.makedirs(os.path.dirname(OUT_MID), exist_ok=True)
    pm_out.write(OUT_MID)
    print(f"[Saved MultiTrack MIDI] {OUT_MID} ({len(pm_out.instruments)} tracks)")

    # Backward-compatible single-marble sequential melody summary from pitched tracks
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
                "source_stems": stems_dir,
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
