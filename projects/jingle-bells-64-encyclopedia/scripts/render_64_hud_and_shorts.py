#!/usr/bin/env python3
"""Render 64-Instrument Encyclopedia HUD overlays and Shorts/Full matrix outputs."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
TIMELINE_JSON = PROJ / "artifacts" / "encyclopedia_64_timeline.json"
PRE_ROLL = 0.90

SWIFT_HUD = r'''
import AppKit
import CoreGraphics
import Foundation

func renderPlaque(outPath: String, eyebrow: String, title: String, isCTA: Bool) {
    let canvasW: CGFloat = 1080
    let canvasH: CGFloat = 1920
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil, width: Int(canvasW), height: Int(canvasH),
        bitsPerComponent: 8, bytesPerRow: 0, space: cs,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { fatalError("CGContext failed") }

    ctx.clear(CGRect(x: 0, y: 0, width: canvasW, height: canvasH))

    let w: CGFloat = isCTA ? 760 : 700
    let h: CGFloat = isCTA ? 128 : 114
    let topY: CGFloat = 210
    let boxX = (canvasW - w) / 2.0
    let boxY = canvasH - topY - h
    let rect = CGRect(x: boxX, y: boxY, width: w, height: h)
    let path = CGPath(roundedRect: rect, cornerWidth: 22, cornerHeight: 22, transform: nil)

    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 0, height: -8), blur: 24, color: NSColor(calibratedWhite: 0, alpha: 0.55).cgColor)
    ctx.setFillColor(NSColor(calibratedRed: 0.06, green: 0.07, blue: 0.09, alpha: isCTA ? 0.85 : 0.74).cgColor)
    ctx.addPath(path)
    ctx.fillPath()
    ctx.restoreGState()

    ctx.saveGState()
    ctx.setStrokeColor(NSColor(calibratedRed: 0.86, green: 0.73, blue: 0.46, alpha: isCTA ? 0.94 : 0.75).cgColor)
    ctx.setLineWidth(isCTA ? 2.2 : 1.5)
    ctx.addPath(path)
    ctx.strokePath()
    ctx.restoreGState()

    let dotR: CGFloat = isCTA ? 7.0 : 6.0
    let dotX = boxX + 34
    let dotY = boxY + h / 2.0 - dotR
    ctx.saveGState()
    ctx.setShadow(offset: .zero, blur: 10, color: NSColor(calibratedRed: 0.16, green: 0.85, blue: 0.52, alpha: 0.75).cgColor)
    ctx.setFillColor(NSColor(calibratedRed: 0.18, green: 0.86, blue: 0.54, alpha: 1.0).cgColor)
    ctx.fillEllipse(in: CGRect(x: dotX, y: dotY, width: dotR * 2, height: dotR * 2))
    ctx.restoreGState()

    let nsCtx = NSGraphicsContext(cgContext: ctx, flipped: false)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = nsCtx

    let textLeft = boxX + 66
    let textW = w - 94
    let brass = NSColor(calibratedRed: 0.87, green: 0.75, blue: 0.49, alpha: 0.96)
    let ivory = NSColor(calibratedRed: 0.96, green: 0.94, blue: 0.90, alpha: 1.0)

    let eyeFont = NSFont(name: "Avenir-Medium", size: isCTA ? 22 : 19) ?? NSFont.systemFont(ofSize: 20)
    let titleFont = NSFont(name: "Georgia-Bold", size: isCTA ? 33 : 32) ?? NSFont.boldSystemFont(ofSize: 32)

    let eyeAttr: [NSAttributedString.Key: Any] = [.font: eyeFont, .foregroundColor: brass, .kern: isCTA ? 0.6 : 1.6]
    let titleAttr: [NSAttributedString.Key: Any] = [.font: titleFont, .foregroundColor: ivory, .kern: 0.4]

    if isCTA {
        NSAttributedString(string: title, attributes: titleAttr).draw(in: CGRect(x: textLeft, y: boxY + h * 0.47, width: textW, height: 44))
        NSAttributedString(string: eyebrow, attributes: eyeAttr).draw(in: CGRect(x: textLeft, y: boxY + h * 0.15, width: textW, height: 32))
    } else {
        NSAttributedString(string: eyebrow, attributes: eyeAttr).draw(in: CGRect(x: textLeft, y: boxY + h * 0.56, width: textW, height: 28))
        NSAttributedString(string: title, attributes: titleAttr).draw(in: CGRect(x: textLeft, y: boxY + h * 0.14, width: textW, height: 44))
    }

    NSGraphicsContext.restoreGraphicsState()

    guard let img = ctx.makeImage() else { fatalError("makeImage failed") }
    let url = URL(fileURLWithPath: outPath)
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
        fatalError("CGImageDestinationCreateWithURL failed")
    }
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}

let args = CommandLine.arguments
let manifestPath = args[1]
let data = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
if let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
    for item in items {
        let out = item["out"] as! String
        let eye = item["eyebrow"] as! String
        let title = item["title"] as! String
        let isCTA = (item["is_cta"] as? Bool) ?? false
        renderPlaque(outPath: out, eyebrow: eye, title: title, isCTA: isCTA)
    }
}
'''


def apply_hud_to_clip(
    clip_mp4: Path,
    out_mp4: Path,
    t_offset_in_full: float,
    clip_dur: float,
    add_shorts_cta: bool = True,
) -> None:
    """Overlay matching 64-instrument encyclopedia plaques onto a rendered MP4 clip."""
    tl = json.loads(TIMELINE_JSON.read_text(encoding="utf-8"))
    switches = tl["switches"]

    # Build active plaque windows in clip-local time
    windows = []
    for i, sw in enumerate(switches):
        t_start = sw["start_time"] + PRE_ROLL - t_offset_in_full
        t_end = (switches[i + 1]["start_time"] + PRE_ROLL - t_offset_in_full) if i + 1 < len(switches) else clip_dur
        if t_end <= 0.1 or t_start >= clip_dur - (2.6 if add_shorts_cta else 0.1):
            continue
        t_s = max(0.25 if i == 0 and t_offset_in_full == 0 else 0.05, t_start)
        t_e = min(clip_dur - (2.7 if add_shorts_cta else 0.2), t_end - 0.12)
        if t_e - t_s < 0.8:
            continue
        windows.append({
            "t_s": round(t_s, 3),
            "t_e": round(t_e, 3),
            "eyebrow": f"FAMILY {sw['family_idx']} · {sw['family_name'].upper()}",
            "title": f"{sw['encyclopedia_idx']:02d}/64 · {sw['instrument']}",
            "is_cta": False,
        })

    if add_shorts_cta:
        windows.append({
            "t_s": round(max(0.5, clip_dur - 2.55), 3),
            "t_e": round(clip_dur, 3),
            "eyebrow": "Tap Related Video  ·  Subscribe @LucaMarbleMusic",
            "title": "Full 64-Instrument Encyclopedia  ↓",
            "is_cta": True,
        })

    if not windows:
        subprocess.run(["cp", str(clip_mp4), str(out_mp4)], check=True)
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        manifest = []
        for idx, w in enumerate(windows):
            png_path = tmp_dir / f"hud_{idx:03d}.png"
            manifest.append({
                "out": str(png_path),
                "eyebrow": w["eyebrow"],
                "title": w["title"],
                "is_cta": w["is_cta"],
            })
            w["png"] = png_path

        manifest_file = tmp_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        swift_file = tmp_dir / "render_hud.swift"
        swift_file.write_text(SWIFT_HUD, encoding="utf-8")
        subprocess.run(["swift", str(swift_file), str(manifest_file)], check=True)

        cmd = ["ffmpeg", "-y", "-i", str(clip_mp4)]
        for w in windows:
            cmd.extend(["-loop", "1", "-t", f"{clip_dur:.3f}", "-i", str(w["png"])])

        fc_parts = []
        prev_v = "0:v"
        for idx, w in enumerate(windows):
            inp = idx + 1
            dur = w["t_e"] - w["t_s"]
            fade_out_st = max(w["t_s"], w["t_e"] - 0.22)
            h_lbl = f"h{idx}"
            v_lbl = f"v{idx}"
            if w["is_cta"]:
                fc_parts.append(f"[{inp}:v]format=rgba,fade=t=in:st={w['t_s']:.3f}:d=0.25:alpha=1[{h_lbl}]")
                fc_parts.append(f"[{prev_v}][{h_lbl}]overlay=0:0:enable='gte(t,{w['t_s']:.3f})'[{v_lbl}]")
            else:
                fc_parts.append(
                    f"[{inp}:v]format=rgba,fade=t=in:st={w['t_s']:.3f}:d=0.20:alpha=1,"
                    f"fade=t=out:st={fade_out_st:.3f}:d=0.20:alpha=1[{h_lbl}]"
                )
                fc_parts.append(
                    f"[{prev_v}][{h_lbl}]overlay=0:0:enable='between(t,{w['t_s']:.3f},{w['t_e']:.3f})'[{v_lbl}]"
                )
            prev_v = v_lbl

        cmd.extend([
            "-filter_complex", ";".join(fc_parts),
            "-map", f"[{prev_v}]",
            "-map", "0:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
            "-c:a", "copy",
            "-shortest",
            str(out_mp4),
        ])
        subprocess.run(cmd, check=True)
        print(f"Saved HUD video ({len(windows)} plaques): {out_mp4}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--offset", type=float, default=0.0, help="Start time in full_64 timeline (sec)")
    p.add_argument("--duration", type=float, required=True, help="Clip duration (sec)")
    p.add_argument("--no-cta", action="store_true")
    a = p.parse_args()
    apply_hud_to_clip(Path(a.input), Path(a.output), a.offset, a.duration, add_shorts_cta=not a.no_cta)
