#!/usr/bin/env python3
"""Render YouTube Shorts safe-zone brass/ivory HUD plaques and composite onto the 11s smooth video."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_MP4 = ROOT / "renders" / "sample_11s_multi_rail_smooth.mp4"
OUTPUT_MP4 = ROOT / "renders" / "sample_11s_multi_rail_shorts.mp4"

SWIFT_RENDERER = r'''
import AppKit
import CoreGraphics
import Foundation

struct PlaqueSpec {
    let outPath: String
    let width: CGFloat
    let height: CGFloat
    let topY: CGFloat
    let eyebrow: String
    let title: String
    let eyebrowSize: CGFloat
    let titleSize: CGFloat
    let isCTA: Bool
}

func renderPlaque(_ spec: PlaqueSpec) {
    let canvasW: CGFloat = 1080
    let canvasH: CGFloat = 1920
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil,
        width: Int(canvasW),
        height: Int(canvasH),
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        fatalError("Failed to create CGContext")
    }

    ctx.clear(CGRect(x: 0, y: 0, width: canvasW, height: canvasH))

    let boxX = (canvasW - spec.width) / 2.0
    // CoreGraphics origin is bottom-left
    let boxY = canvasH - spec.topY - spec.height
    let rect = CGRect(x: boxX, y: boxY, width: spec.width, height: spec.height)
    let path = CGPath(roundedRect: rect, cornerWidth: 22, cornerHeight: 22, transform: nil)

    // Soft drop shadow
    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 0, height: -8), blur: 24, color: NSColor(calibratedWhite: 0.0, alpha: 0.55).cgColor)
    ctx.setFillColor(NSColor(calibratedRed: 0.06, green: 0.07, blue: 0.09, alpha: spec.isCTA ? 0.84 : 0.74).cgColor)
    ctx.addPath(path)
    ctx.fillPath()
    ctx.restoreGState()

    // Brushed brass border
    ctx.saveGState()
    ctx.setStrokeColor(NSColor(calibratedRed: 0.86, green: 0.73, blue: 0.46, alpha: spec.isCTA ? 0.92 : 0.72).cgColor)
    ctx.setLineWidth(spec.isCTA ? 2.2 : 1.5)
    ctx.addPath(path)
    ctx.strokePath()
    ctx.restoreGState()

    // Emerald marble accent dot on left
    let dotR: CGFloat = spec.isCTA ? 7.0 : 6.0
    let dotX = boxX + 34
    let dotY = boxY + spec.height / 2.0 - dotR
    ctx.saveGState()
    ctx.setShadow(offset: .zero, blur: 10, color: NSColor(calibratedRed: 0.16, green: 0.85, blue: 0.52, alpha: 0.75).cgColor)
    ctx.setFillColor(NSColor(calibratedRed: 0.18, green: 0.86, blue: 0.54, alpha: 1.0).cgColor)
    ctx.fillEllipse(in: CGRect(x: dotX, y: dotY, width: dotR * 2, height: dotR * 2))
    ctx.restoreGState()

    // Draw text using NSGraphicsContext
    let nsCtx = NSGraphicsContext(cgContext: ctx, flipped: false)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = nsCtx

    let textLeft = boxX + 66
    let textW = spec.width - 96

    let eyebrowFont = NSFont(name: "Avenir-Medium", size: spec.eyebrowSize) ?? NSFont.systemFont(ofSize: spec.eyebrowSize, weight: .medium)
    let titleFont = NSFont(name: "Georgia-Bold", size: spec.titleSize) ?? NSFont.boldSystemFont(ofSize: spec.titleSize)

    let eyebrowColor = NSColor(calibratedRed: 0.87, green: 0.75, blue: 0.49, alpha: 0.96)
    let titleColor = NSColor(calibratedRed: 0.96, green: 0.94, blue: 0.90, alpha: 1.0)

    let eyebrowAttr: [NSAttributedString.Key: Any] = [
        .font: eyebrowFont,
        .foregroundColor: eyebrowColor,
        .kern: spec.isCTA ? 0.6 : 1.8
    ]
    let titleAttr: [NSAttributedString.Key: Any] = [
        .font: titleFont,
        .foregroundColor: titleColor,
        .kern: 0.4
    ]

    if spec.isCTA {
        // Title on top, eyebrow/subline below
        let titleStr = NSAttributedString(string: spec.title, attributes: titleAttr)
        let subStr = NSAttributedString(string: spec.eyebrow, attributes: eyebrowAttr)
        titleStr.draw(in: CGRect(x: textLeft, y: boxY + spec.height * 0.46, width: textW, height: 46))
        subStr.draw(in: CGRect(x: textLeft, y: boxY + spec.height * 0.16, width: textW, height: 34))
    } else {
        let eyeStr = NSAttributedString(string: spec.eyebrow, attributes: eyebrowAttr)
        let titleStr = NSAttributedString(string: spec.title, attributes: titleAttr)
        eyeStr.draw(in: CGRect(x: textLeft, y: boxY + spec.height * 0.56, width: textW, height: 30))
        titleStr.draw(in: CGRect(x: textLeft, y: boxY + spec.height * 0.14, width: textW, height: 46))
    }

    NSGraphicsContext.restoreGraphicsState()

    guard let image = ctx.makeImage() else { fatalError("makeImage failed") }
    let url = URL(fileURLWithPath: spec.outPath)
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
        fatalError("CGImageDestinationCreateWithURL failed")
    }
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

let args = CommandLine.arguments
let outDir = args[1]

renderPlaque(PlaqueSpec(
    outPath: "\(outDir)/hud_01.png",
    width: 650,
    height: 112,
    topY: 215,
    eyebrow: "JINGLE BELLS · MARBLE MACHINE",
    title: "01 · Castanets",
    eyebrowSize: 20,
    titleSize: 36,
    isCTA: false
))

renderPlaque(PlaqueSpec(
    outPath: "\(outDir)/hud_02.png",
    width: 650,
    height: 112,
    topY: 215,
    eyebrow: "JINGLE BELLS · MARBLE MACHINE",
    title: "02 · Piano",
    eyebrowSize: 20,
    titleSize: 36,
    isCTA: false
))

renderPlaque(PlaqueSpec(
    outPath: "\(outDir)/hud_03.png",
    width: 740,
    height: 126,
    topY: 210,
    eyebrow: "Tap Related Video  ·  Subscribe for Part 2",
    title: "Full 3-Min Machine  ↓",
    eyebrowSize: 23,
    titleSize: 35,
    isCTA: true
))
'''


def main() -> None:
    if not INPUT_MP4.exists():
        raise FileNotFoundError(f"Missing input video: {INPUT_MP4}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        swift_file = tmp_dir / "render_hud.swift"
        swift_file.write_text(SWIFT_RENDERER, encoding="utf-8")

        subprocess.run(["swift", str(swift_file), str(tmp_dir)], check=True)

        hud1 = tmp_dir / "hud_01.png"
        hud2 = tmp_dir / "hud_02.png"
        hud3 = tmp_dir / "hud_03.png"

        # Composite 3 safe-zone plaques with smooth fades, keeping audio untouched.
        filter_complex = (
            "[1:v]format=rgba,fade=t=in:st=0.40:d=0.25:alpha=1,fade=t=out:st=4.85:d=0.25:alpha=1[h1];"
            "[2:v]format=rgba,fade=t=in:st=5.40:d=0.25:alpha=1,fade=t=out:st=7.95:d=0.25:alpha=1[h2];"
            "[3:v]format=rgba,fade=t=in:st=8.45:d=0.30:alpha=1[h3];"
            "[0:v][h1]overlay=0:0:enable='between(t,0.40,5.15)'[v1];"
            "[v1][h2]overlay=0:0:enable='between(t,5.40,8.25)'[v2];"
            "[v2][h3]overlay=0:0:enable='gte(t,8.45)'[vout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(INPUT_MP4),
            "-loop", "1", "-t", "11.02", "-i", str(hud1),
            "-loop", "1", "-t", "11.02", "-i", str(hud2),
            "-loop", "1", "-t", "11.02", "-i", str(hud3),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "0:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
            "-c:a", "copy",
            "-shortest",
            str(OUTPUT_MP4),
        ]
        subprocess.run(cmd, check=True)
        print(f"Saved Shorts HUD video: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
