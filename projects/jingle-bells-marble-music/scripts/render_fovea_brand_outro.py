#!/usr/bin/env python3
"""Reusable Fovea Studio (@foveaSI) branded 8s outro generator & appender.

Appends an 8-second Mechanical Music Box branded outro (pushing a 2m56s full
video past 3m00s into YouTube standard long-form) with brass-framed placeholders
aligned for YouTube End Screen (Subscribe circle + Next Video card).
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

SWIFT_OUTRO_RENDERER = r'''
import AppKit
import CoreGraphics
import Foundation

func renderOutroCard(outPath: String, brandName: String, songTitle: String, seriesTag: String, handle: String) {
    let w: CGFloat = 1080
    let h: CGFloat = 1920
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil,
        width: Int(w),
        height: Int(h),
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        fatalError("Failed to create CGContext")
    }

    // 1. Matte dark charcoal backdrop matching the Mechanical Music Box wall
    ctx.setFillColor(NSColor(calibratedRed: 0.055, green: 0.063, blue: 0.078, alpha: 1.0).cgColor)
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))

    // Subtle warm radial spotlight in upper-center
    let gradColors = [
        NSColor(calibratedRed: 0.14, green: 0.15, blue: 0.18, alpha: 0.55).cgColor,
        NSColor(calibratedRed: 0.055, green: 0.063, blue: 0.078, alpha: 0.0).cgColor
    ] as CFArray
    if let grad = CGGradient(colorsSpace: colorSpace, colors: gradColors, locations: [0.0, 1.0]) {
        ctx.drawRadialGradient(
            grad,
            startCenter: CGPoint(x: w * 0.5, y: h * 0.62),
            startRadius: 20,
            endCenter: CGPoint(x: w * 0.5, y: h * 0.62),
            endRadius: 720,
            options: []
        )
    }

    // Outer subtle brass hairline frame
    let frameRect = CGRect(x: 54, y: 84, width: w - 108, height: h - 168)
    let framePath = CGPath(roundedRect: frameRect, cornerWidth: 28, cornerHeight: 28, transform: nil)
    ctx.setStrokeColor(NSColor(calibratedRed: 0.84, green: 0.71, blue: 0.45, alpha: 0.32).cgColor)
    ctx.setLineWidth(1.5)
    ctx.addPath(framePath)
    ctx.strokePath()

    // 2. Brand Emblem: Emerald Marble inside Brass Ring (top-center)
    let centerX = w * 0.5
    let emblemY = h - 300
    let ringR: CGFloat = 46
    ctx.saveGState()
    ctx.setStrokeColor(NSColor(calibratedRed: 0.86, green: 0.73, blue: 0.46, alpha: 0.90).cgColor)
    ctx.setLineWidth(2.2)
    ctx.strokeEllipse(in: CGRect(x: centerX - ringR, y: emblemY - ringR, width: ringR * 2, height: ringR * 2))
    ctx.restoreGState()

    let marbleR: CGFloat = 22
    ctx.saveGState()
    ctx.setShadow(offset: .zero, blur: 22, color: NSColor(calibratedRed: 0.16, green: 0.86, blue: 0.52, alpha: 0.80).cgColor)
    ctx.setFillColor(NSColor(calibratedRed: 0.18, green: 0.85, blue: 0.54, alpha: 1.0).cgColor)
    ctx.fillEllipse(in: CGRect(x: centerX - marbleR, y: emblemY - marbleR, width: marbleR * 2, height: marbleR * 2))
    ctx.restoreGState()

    // Highlight glint on marble emblem
    ctx.setFillColor(NSColor(calibratedWhite: 1.0, alpha: 0.75).cgColor)
    ctx.fillEllipse(in: CGRect(x: centerX - 10, y: emblemY + 5, width: 8, height: 8))

    // Helper for centered text
    let nsCtx = NSGraphicsContext(cgContext: ctx, flipped: false)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = nsCtx

    let pStyle = NSMutableParagraphStyle()
    pStyle.alignment = .center

    let brass = NSColor(calibratedRed: 0.87, green: 0.75, blue: 0.49, alpha: 0.96)
    let ivory = NSColor(calibratedRed: 0.96, green: 0.94, blue: 0.90, alpha: 1.0)
    let muted = NSColor(calibratedRed: 0.72, green: 0.71, blue: 0.69, alpha: 0.88)

    let brandFont = NSFont(name: "Avenir-Heavy", size: 40) ?? NSFont.boldSystemFont(ofSize: 40)
    let handleFont = NSFont(name: "Avenir-Medium", size: 24) ?? NSFont.systemFont(ofSize: 24)
    let songFont = NSFont(name: "Georgia-Bold", size: 34) ?? NSFont.boldSystemFont(ofSize: 34)
    let subFont = NSFont(name: "Avenir-Medium", size: 22) ?? NSFont.systemFont(ofSize: 22)

    NSAttributedString(string: brandName, attributes: [
        .font: brandFont, .foregroundColor: ivory, .kern: 6.0, .paragraphStyle: pStyle
    ]).draw(in: CGRect(x: 80, y: h - 425, width: w - 160, height: 54))

    NSAttributedString(string: handle, attributes: [
        .font: handleFont, .foregroundColor: brass, .kern: 2.2, .paragraphStyle: pStyle
    ]).draw(in: CGRect(x: 80, y: h - 468, width: w - 160, height: 34))

    // Divider hairline
    ctx.setStrokeColor(NSColor(calibratedRed: 0.86, green: 0.73, blue: 0.46, alpha: 0.45).cgColor)
    ctx.setLineWidth(1.2)
    ctx.move(to: CGPoint(x: centerX - 180, y: h - 512))
    ctx.addLine(to: CGPoint(x: centerX + 180, y: h - 512))
    ctx.strokePath()

    // Current piece credit
    NSAttributedString(string: songTitle, attributes: [
        .font: songFont, .foregroundColor: ivory, .kern: 0.6, .paragraphStyle: pStyle
    ]).draw(in: CGRect(x: 90, y: h - 590, width: w - 180, height: 48))

    NSAttributedString(string: seriesTag, attributes: [
        .font: subFont, .foregroundColor: muted, .kern: 1.8, .paragraphStyle: pStyle
    ]).draw(in: CGRect(x: 90, y: h - 635, width: w - 180, height: 32))

    // 3. YouTube End Screen Alignment Slots (safe zone y = 760..1460 from top)
    // Slot A (Upper-middle): Next Video Card (16:9 ratio: 640 x 360)
    let cardW: CGFloat = 640
    let cardH: CGFloat = 360
    let cardX = (w - cardW) / 2.0
    let cardY = h - 730 - cardH
    let cardRect = CGRect(x: cardX, y: cardY, width: cardW, height: cardH)
    let cardPath = CGPath(roundedRect: cardRect, cornerWidth: 20, cornerHeight: 20, transform: nil)

    ctx.setFillColor(NSColor(calibratedRed: 0.085, green: 0.095, blue: 0.115, alpha: 0.85).cgColor)
    ctx.addPath(cardPath)
    ctx.fillPath()
    ctx.setStrokeColor(NSColor(calibratedRed: 0.86, green: 0.73, blue: 0.46, alpha: 0.65).cgColor)
    ctx.setLineWidth(1.8)
    ctx.addPath(cardPath)
    ctx.strokePath()

    NSAttributedString(string: "NEXT MARBLE MACHINE", attributes: [
        .font: subFont, .foregroundColor: brass, .kern: 2.5, .paragraphStyle: pStyle
    ]).draw(in: CGRect(x: cardX, y: cardY + cardH * 0.46, width: cardW, height: 34))

    // Slot B (Lower-middle): Subscribe Circle Slot (diameter 210)
    let subR: CGFloat = 105
    let subCy = h - 1310
    ctx.setFillColor(NSColor(calibratedRed: 0.085, green: 0.095, blue: 0.115, alpha: 0.85).cgColor)
    ctx.fillEllipse(in: CGRect(x: centerX - subR, y: subCy - subR, width: subR * 2, height: subR * 2))
    ctx.setStrokeColor(NSColor(calibratedRed: 0.86, green: 0.73, blue: 0.46, alpha: 0.75).cgColor)
    ctx.setLineWidth(2.0)
    ctx.strokeEllipse(in: CGRect(x: centerX - subR, y: subCy - subR, width: subR * 2, height: subR * 2))

    NSAttributedString(string: "SUBSCRIBE", attributes: [
        .font: subFont, .foregroundColor: ivory, .kern: 2.8, .paragraphStyle: pStyle
    ]).draw(in: CGRect(x: centerX - 140, y: subCy - subR - 46, width: 280, height: 32))

    // Footer prompt
    NSAttributedString(string: "Comment the next song & instrument you want to hear", attributes: [
        .font: subFont, .foregroundColor: muted, .kern: 0.5, .paragraphStyle: pStyle
    ]).draw(in: CGRect(x: 80, y: 150, width: w - 160, height: 34))

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
renderOutroCard(outPath: args[1], brandName: args[2], songTitle: args[3], seriesTag: args[4], handle: args[5])
'''


def probe_duration(video_path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        text=True,
    ).strip()
    return float(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append Fovea Studio branded outro to a vertical 1080x1920 video.")
    parser.add_argument("--input", required=True, help="Input MP4 path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--brand", default="LUCA MARBLE MUSIC", help="Brand title on end card")
    parser.add_argument("--title", default="Jingle Bells", help="Song title on end card")
    parser.add_argument("--series", default="ACOUSTIC INSTRUMENTS · MARBLE MUSIC BOX", help="Subtitle / series tag")
    parser.add_argument("--handle", default="@LucaMarbleMusic  ·  YouTube", help="Channel handle line")
    parser.add_argument("--outro-sec", type=float, default=8.0, help="Outro duration in seconds")
    parser.add_argument("--fade-sec", type=float, default=0.8, help="Crossfade duration in seconds")
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    in_dur = probe_duration(in_path)
    xfade_offset = max(0.1, in_dur - args.fade_sec)
    total_dur = xfade_offset + args.outro_sec

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        swift_file = tmp_dir / "render_outro.swift"
        outro_png = tmp_dir / "outro_card.png"
        swift_file.write_text(SWIFT_OUTRO_RENDERER, encoding="utf-8")

        subprocess.run(
            ["swift", str(swift_file), str(outro_png), args.brand, args.title, args.series, args.handle],
            check=True,
        )

        # Smooth crossfade video + pad audio silently with natural tail fade
        fade_out_start = max(0.1, in_dur - 1.2)
        filter_complex = (
            f"[0:v]fps=60,settb=AVTB,format=rgba[v0];"
            f"[1:v]fps=60,settb=AVTB,format=rgba[v1];"
            f"[v0][v1]xfade=transition=fade:duration={args.fade_sec:.3f}:offset={xfade_offset:.3f},format=yuv420p[vout];"
            f"[0:a]afade=t=out:st={fade_out_start:.3f}:d=1.2,apad=whole_dur={total_dur:.3f}[aout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(in_path),
            "-loop", "1", "-t", f"{args.outro_sec:.3f}", "-r", "60", "-i", str(outro_png),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
            "-c:a", "aac", "-b:a", "192k",
            "-t", f"{total_dur:.3f}",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        print(f"Saved Fovea Studio branded video ({total_dur:.2f}s): {out_path}")


if __name__ == "__main__":
    main()
