#!/usr/bin/env python3
"""Synthesize seamless G-major walnut tap foley + glass-harp / celesta finale coda and mix into the full MV WAV."""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt


G_PENT = (196.00, 246.94, 293.66, 392.00, 440.00)  # G3, B3, D4, G4, A4


def _pan_gains(pan: float) -> tuple[float, float]:
    p = max(-1.0, min(1.0, float(pan)))
    ang = (p + 1.0) * 0.25 * math.pi
    return math.cos(ang), math.sin(ang)


def _add_stereo(buf: np.ndarray, sig: np.ndarray, t_sec: float, sr: int, pan: float, gain: float = 1.0) -> None:
    i0 = int(round(t_sec * sr))
    if i0 >= len(buf) or len(sig) == 0:
        return
    src_start = max(0, -i0)
    dst_start = max(0, i0)
    n_copy = min(len(sig) - src_start, len(buf) - dst_start)
    if n_copy <= 0:
        return
    gl, gr = _pan_gains(pan)
    seg = sig[src_start:src_start + n_copy] * gain
    buf[dst_start:dst_start + n_copy, 0] += seg * gl
    buf[dst_start:dst_start + n_copy, 1] += seg * gr


def _bandpass(sig: np.ndarray, lo_hz: float, hi_hz: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = 0.5 * sr
    lo = max(20.0, lo_hz) / nyq
    hi = min(nyq * 0.96, hi_hz) / nyq
    sos = butter(order, [lo, hi], btype="bandpass", output="sos")
    return sosfilt(sos, sig).astype(np.float64)


def synth_walnut_tap(f0: float, sr: int, rng: np.random.Generator, is_drum: bool = False) -> np.ndarray:
    """Modal walnut bar tap + gentle glass-marble click (~0.45s)."""
    dur = 0.50
    t = np.arange(int(round(dur * sr)), dtype=np.float64) / sr
    # Walnut fundamental + marimba 1st/2nd modal overtones
    env0 = (1.0 - np.exp(-t / 0.0025)) * np.exp(-t / (0.17 if is_drum else 0.13))
    env1 = (1.0 - np.exp(-t / 0.0018)) * np.exp(-t / 0.038)
    env2 = (1.0 - np.exp(-t / 0.0010)) * np.exp(-t / 0.018)
    wood = (
        0.72 * np.sin(2.0 * math.pi * f0 * t) * env0
        + 0.24 * np.sin(2.0 * math.pi * (3.98 * f0) * t) * env1
        + 0.10 * np.sin(2.0 * math.pi * (9.20 * f0) * t) * env2
    )
    # Short hardwood impact transient (1.2-4.5 kHz)
    noise = rng.standard_normal(len(t))
    strike = _bandpass(noise, 1200.0, 4500.0, sr, order=2) * ((1.0 - np.exp(-t / 0.001)) * np.exp(-t / 0.006))
    # Subtle glass bead clink (3.15 kHz + 5.4 kHz)
    glass_env = (1.0 - np.exp(-t / 0.0008)) * np.exp(-t / 0.042)
    glass = (0.55 * np.sin(2.0 * math.pi * 3150.0 * t) + 0.35 * np.sin(2.0 * math.pi * 5400.0 * t)) * glass_env
    return wood * 0.55 + strike * 0.42 + glass * 0.14


def synth_glass_chime(freq: float, dur: float, decay_tau: float, sr: int, atk_sec: float = 0.016) -> np.ndarray:
    """Warm crystal glass-harp / celesta note with slow ±1.1 Hz harmonic shimmer."""
    t = np.arange(int(round(dur * sr)), dtype=np.float64) / sr
    atk = 1.0 - np.exp(-t / max(0.002, atk_sec))
    partials = (
        (1.0, 1.00, 1.00),
        (2.0, 0.32, 0.72),
        (2.756, 0.10, 0.36),
        (4.02, 0.045, 0.22),
    )
    sig = np.zeros_like(t)
    for ratio, amp, d_fac in partials:
        f_p = freq * ratio
        if f_p >= 0.45 * sr:
            continue
        env = atk * np.exp(-t / max(0.05, decay_tau * d_fac))
        shimmer = 0.5 * (
            np.sin(2.0 * math.pi * (f_p - 0.55) * t)
            + np.sin(2.0 * math.pi * (f_p + 0.55) * t + 0.4)
        )
        sig += amp * shimmer * env
    return sig * 0.55


def synth_marimba_note(freq: float, dur: float, decay_tau: float, sr: int) -> np.ndarray:
    """Warm resonant marimba tone for Gmaj9 finale chord sustain."""
    t = np.arange(int(round(dur * sr)), dtype=np.float64) / sr
    atk = 1.0 - np.exp(-t / 0.004)
    sig = (
        0.78 * np.sin(2.0 * math.pi * freq * t) * atk * np.exp(-t / decay_tau)
        + 0.18 * np.sin(2.0 * math.pi * (4.0 * freq) * t) * atk * np.exp(-t / (0.28 * decay_tau))
    )
    return sig * 0.45


def synth_sleigh_bell_shimmer(dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Delicate festive high bell dust shimmer decaying smoothly."""
    t = np.arange(int(round(dur * sr)), dtype=np.float64) / sr
    n = rng.standard_normal(len(t))
    hp = _bandpass(n, 5800.0, 11500.0, sr, order=3)
    # Delicate shimmering partials + modulated metallic sparkle
    bell_modes = np.zeros_like(t)
    for bf in (6270.0, 7420.0, 8850.0, 10340.0):
        bell_modes += np.sin(2.0 * math.pi * bf * t + rng.uniform(0, 6.28))
    trem = 0.65 + 0.35 * np.sin(2.0 * math.pi * 11.5 * t + 0.8)
    env = (1.0 - np.exp(-t / 0.045)) * np.exp(-t / 0.95)
    return (0.45 * hp + 0.12 * bell_modes) * trem * env


def apply_concert_hall_reverb(dry: np.ndarray, sr: int, rng: np.random.Generator, wet: float = 0.34, rt60: float = 2.1) -> np.ndarray:
    """Decorrelated stereo concert-hall impulse response via fftconvolve."""
    ir_len = int(round(rt60 * 1.35 * sr))
    t = np.arange(ir_len, dtype=np.float64) / sr
    decay = np.exp(-6.9078 * t / rt60)
    ir = np.zeros((ir_len, 2), dtype=np.float64)
    for ch in range(2):
        noise = rng.standard_normal(ir_len)
        # Gentle warmth: roll off harsh extremes cleanly
        tail = _bandpass(noise, 180.0, 6800.0, sr, order=2) * decay
        # Early reflections (8–48ms)
        ir[:, ch] = tail
        for ref_ms, ref_amp in ((11.0, 0.25), (19.5, 0.18), (31.0, 0.12), (44.0, 0.08)):
            idx = int(round((ref_ms + ch * 2.3) * 0.001 * sr))
            if idx < ir_len:
                ir[idx, ch] += ref_amp
        ir[:, ch] /= max(1e-9, math.sqrt(np.sum(ir[:, ch] ** 2)))

    wet_l = fftconvolve(dry[:, 0], ir[:, 0], mode="full")[:len(dry)]
    wet_r = fftconvolve(dry[:, 1], ir[:, 1], mode="full")[:len(dry)]
    return (1.0 - wet) * dry + wet * np.stack([wet_l, wet_r], axis=1)


def synthesize_finale_audio(physics_json: str, in_wav: str, out_wav: str) -> None:
    with open(physics_json, "r", encoding="utf-8") as f:
        plan = json.load(f)

    pre_roll = float(plan.get("pre_roll", 0.90))
    total_dur = float(plan["total_duration"])
    wav_dur = max(1.0, total_dur - pre_roll)

    orig_data, sr = sf.read(in_wav, dtype="float32", always_2d=True)
    if orig_data.shape[1] == 1:
        orig_data = np.repeat(orig_data, 2, axis=1)
    n_target = int(round(wav_dur * sr))
    out_buf = np.zeros((n_target, 2), dtype=np.float64)
    n_copy = min(len(orig_data), n_target)
    out_buf[:n_copy] = orig_data[:n_copy].astype(np.float64)

    finale = plan.get("finale") or {}
    bridge_hits = (finale.get("bridge") or {}).get("hits", [])
    words = finale.get("words", [])
    rng = np.random.default_rng(20261119)
    coda = np.zeros((n_target, 2), dtype=np.float64)

    # Layer 1: Walnut-wood tap foley for all Unified Bridge bounces
    for h in bridge_hits:
        t_w = float(h["t_hit"]) - pre_roll
        hx = float(h["x"])
        note_i = int(round((hx + 1.2) / 2.4 * (len(G_PENT) - 1)))
        note_i = max(0, min(len(G_PENT) - 1, note_i))
        f0 = G_PENT[note_i]
        tap = synth_walnut_tap(f0, sr, rng, is_drum=(h.get("ball_type") == "drum"))
        _add_stereo(coda, tap, t_w, sr, pan=hx / 1.5, gain=0.62)

    # Layer 2: G-Major Glass-Harp & Celesta word-ignition chimes + final Gmaj9 resolve
    for w in words:
        wid = str(w["id"])
        t_w = float(w["morph_t"]) - pre_roll
        if wid == "GLASS":
            _add_stereo(coda, synth_glass_chime(783.99, 2.6, 1.35, sr), t_w, sr, pan=-0.38, gain=0.52)
            _add_stereo(coda, synth_glass_chime(392.00, 2.6, 1.45, sr), t_w, sr, pan=-0.38, gain=0.26)
        elif wid == "MARBLE":
            _add_stereo(coda, synth_glass_chime(987.77, 2.6, 1.35, sr), t_w, sr, pan=+0.20, gain=0.50)
            _add_stereo(coda, synth_glass_chime(783.99, 2.4, 1.20, sr), t_w, sr, pan=+0.20, gain=0.18)
        elif wid == "ON":
            _add_stereo(coda, synth_glass_chime(1174.66, 2.5, 1.35, sr), t_w, sr, pan=-0.20, gain=0.48)
            _add_stereo(coda, synth_glass_chime(987.77, 2.2, 1.15, sr), t_w, sr, pan=-0.20, gain=0.18)
        elif wid == "WOOD":
            _add_stereo(coda, synth_glass_chime(1567.98, 2.6, 1.30, sr), t_w, sr, pan=+0.12, gain=0.44)
            for cf, cp, cg in ((392.00, -0.35, 0.22), (493.88, -0.10, 0.20), (587.33, +0.15, 0.18), (739.99, +0.35, 0.16)):
                _add_stereo(coda, synth_glass_chime(cf, 2.6, 1.40, sr, atk_sec=0.040), t_w, sr, pan=cp, gain=cg)
        elif wid == "S":
            # Sparkling 42ms staggered flourish F#6 -> G6 -> A6 -> B6
            for k_f, (ff, pp) in enumerate(((1479.98, +0.25), (1567.98, +0.35), (1760.00, +0.45), (1975.53, +0.52))):
                _add_stereo(coda, synth_glass_chime(ff, 2.4, 1.10, sr, atk_sec=0.008), t_w + k_f * 0.042, sr, pan=pp, gain=0.28)
            # Warm Gmaj9 marimba + glass-harp finale resolution chord
            chord = (
                (196.00, -0.25, 0.34),  # G3
                (293.66, -0.15, 0.30),  # D4
                (392.00, +0.10, 0.32),  # G4
                (493.88, -0.20, 0.28),  # B4
                (587.33, +0.20, 0.28),  # D5
                (880.00, +0.30, 0.24),  # A5
                (1567.98, +0.05, 0.22), # G6
            )
            for cf, cp, cg in chord:
                _add_stereo(coda, synth_marimba_note(cf, 3.8, 1.85, sr), t_w + 0.02, sr, pan=cp, gain=cg)
                _add_stereo(coda, synth_glass_chime(cf, 3.8, 1.95, sr, atk_sec=0.035), t_w + 0.02, sr, pan=-cp * 0.6, gain=cg * 0.72)
            # Festive sleigh-bell crystal shimmer tail
            bells = synth_sleigh_bell_shimmer(3.0, sr, rng)
            _add_stereo(coda, bells, t_w + 0.03, sr, pan=0.0, gain=0.14)

    # Layer 3: Concert-hall stereo reverb + calibrated dynamic leveling
    coda_wet = apply_concert_hall_reverb(coda, sr, rng, wet=0.34, rt60=2.1)

    # Normalize Finale peak so coda sits naturally at -14 dBFS RMS / ~0.55 peak headroom
    t_coda_check0 = max(0, int(round(100.0 * sr)))
    coda_peak = float(np.max(np.abs(coda_wet[t_coda_check0:]))) if len(coda_wet) > t_coda_check0 else 1.0
    if coda_peak > 1e-6:
        target_peak = 0.52  # ~ -5.7 dBFS peak, harmonious with song climax (-12 dBFS RMS) & gentle decay
        coda_wet *= target_peak / coda_peak

    mixed = out_buf + coda_wet

    # Smooth 0.65s raised-cosine fade-out at the very end
    n_fade = min(len(mixed), int(round(0.65 * sr)))
    if n_fade > 0:
        u = np.linspace(0.0, 1.0, n_fade, dtype=np.float64)
        fade = 0.5 * (1.0 + np.cos(math.pi * u))
        mixed[-n_fade:] *= fade[:, None]

    # Soft-clip safety ceiling (-0.5 dBFS ≈ 0.944)
    ceiling = 10.0 ** (-0.5 / 20.0)
    peak_all = float(np.max(np.abs(mixed)))
    if peak_all > ceiling:
        mixed = ceiling * np.tanh(mixed / ceiling)

    sf.write(out_wav, mixed.astype(np.float32), sr, subtype="PCM_16")
    rms_tail = float(np.sqrt(np.mean(mixed[t_coda_check0:] ** 2)))
    print(f"[Finale Audio] Saved {out_wav} (sr={sr}, dur={len(mixed)/sr:.3f}s, peak={20*math.log10(max(1e-6, float(np.max(np.abs(mixed))))):.2f} dBFS, tail_rms={20*math.log10(max(1e-6, rms_tail)):.2f} dBFS)")


if __name__ == "__main__":
    p_json = sys.argv[1] if len(sys.argv) >= 2 else "projects/jingle-bells-marble-music/artifacts/glass_marbles_ensemble_16x9_physics.json"
    in_w = sys.argv[2] if len(sys.argv) >= 3 else "projects/jingle-bells-marble-music/assets/audio/glass_marbles_on_wood_full.wav"
    out_w = sys.argv[3] if len(sys.argv) >= 4 else "projects/jingle-bells-marble-music/assets/audio/glass_marbles_on_wood_full_v22.wav"
    synthesize_finale_audio(p_json, in_w, out_w)
