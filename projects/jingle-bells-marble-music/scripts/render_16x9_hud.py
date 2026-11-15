#!/usr/bin/env python3
"""Lightweight 16:9 Broadcast Telemetry HUD Generator (ASS / FFmpeg overlay).

Reads `*_hud.json` emitted by `build_16x9_concert_template.py` and writes a crisp,
non-intrusive `.ass` subtitle/HUD track anchored to the bottom-left & top-right
safe margins of 1920x1080 renders — zero Pillow/external Python deps required.

Usage:
  python3 render_16x9_hud.py <hud_json> <out_ass> [input_mp4] [output_mp4]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    cs_total = int(round(sec * 100.0))
    cs = cs_total % 100
    s_total = cs_total // 100
    s = s_total % 60
    m = (s_total // 60) % 60
    h = s_total // 3600
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass_hud(hud_data: dict, out_ass: str) -> None:
    fps = float(hud_data.get("fps", 60))
    total_dur = float(hud_data.get("total_frames", 720)) / fps
    acts = {int(a["act_idx"]): a for a in hud_data.get("acts", [])}
    events = sorted(hud_data.get("events", []), key=lambda e: float(e["t_hit"]))

    header = "\n".join([
        "[Script Info]",
        "Title: Luca Marble Music 16:9 Concert HUD",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: NowPlaying,Helvetica Neue,30,&H00F4F6F8,&H000000FF,&H4010141A,&H900B0D12,-1,0,0,0,100,100,0.8,0,3,12,0,1,64,64,44,1",
        "Style: ActBadge,Helvetica Neue,26,&H005CD2EE,&H000000FF,&H4010141A,&H900B0D12,-1,0,0,0,100,100,1.2,0,3,10,0,7,64,64,38,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])

    lines = [header]
    for i, ev in enumerate(events):
        t0 = float(ev["t_hit"])
        t1 = float(events[i + 1]["t_hit"]) if i + 1 < len(events) else total_dur
        t1 = max(t0 + 0.15, min(total_dur, t1))
        voice = "EMERALD LEAD" if int(ev.get("marble_idx", 0)) == 0 else "GOLD HARMONY"
        inst = str(ev.get("instrument", "Instrument"))
        pitch = str(ev.get("pitch_name", ""))
        lines.append(
            f"Dialogue: 0,{_ass_time(t0)},{_ass_time(t1)},NowPlaying,,0,0,0,,{{\\fad(60,120)}}{voice}  ·  {inst}  [{pitch}]"
        )

    # Top-left Act span indicators
    act_spans: dict[int, tuple[float, float]] = {}
    for ev in events:
        ai = int(ev.get("act_idx", 0))
        t = float(ev["t_hit"])
        if ai not in act_spans:
            act_spans[ai] = (t, t + 0.8)
        else:
            act_spans[ai] = (act_spans[ai][0], t + 0.8)
    sorted_acts = sorted(act_spans.items())
    for j, (ai, (ts, _)) in enumerate(sorted_acts):
        t_start = 0.0 if j == 0 else ts - 0.35
        t_end = (sorted_acts[j + 1][1][0] - 0.35) if j + 1 < len(sorted_acts) else total_dur
        act_title = acts.get(ai, {}).get("title", f"ACT {ai + 1}")
        lines.append(
            f"Dialogue: 1,{_ass_time(t_start)},{_ass_time(t_end)},ActBadge,,0,0,0,,{{\\fad(200,200)}}{act_title}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_ass)), exist_ok=True)
    with open(out_ass, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: render_16x9_hud.py <hud_json> <out_ass> [input_mp4] [output_mp4]")
        sys.exit(1)

    hud_json = sys.argv[1]
    out_ass = sys.argv[2]
    with open(hud_json, "r", encoding="utf-8") as f:
        hud_data = json.load(f)

    build_ass_hud(hud_data, out_ass)
    print(f"Wrote 16:9 Concert HUD overlay: {out_ass}")

    if len(sys.argv) >= 5:
        in_mp4 = sys.argv[3]
        out_mp4 = sys.argv[4]
        ass_escaped = os.path.abspath(out_ass).replace("\\", "\\\\").replace(":", "\\:")
        cmd = [
            "ffmpeg", "-y", "-i", in_mp4,
            "-vf", f"ass={ass_escaped}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-c:a", "copy",
            out_mp4,
        ]
        subprocess.run(cmd, check=True)
        print(f"Burned 16:9 HUD into: {out_mp4}")


if __name__ == "__main__":
    main()
