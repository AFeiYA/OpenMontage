#!/usr/bin/env python3
"""Generate YouTube channel branding assets for @LucaMarbleMusic."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT_DIR = ROOT / "channel_kit"

SWIFT_KIT = r'''
import AppKit
import CoreGraphics
import Foundation

func savePNG(_ ctx: CGContext, to path: String) {
    guard let img = ctx.makeImage() else { fatalError("makeImage failed") }
    let url = URL(fileURLWithPath: path)
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
        fatalError("CGImageDestinationCreateWithURL failed")
    }
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}

func makeContext(w: Int, h: Int) -> CGContext {
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
        space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { fatalError("CGContext failed") }
    return ctx
}

func drawMarbleEmblem(ctx: CGContext, cx: CGFloat, cy: CGFloat, ringR: CGFloat, marbleR: CGFloat, ringWidth: CGFloat) {
    // Brass outer ring
    ctx.saveGState()
    ctx.setStrokeColor(NSColor(calibratedRed: 0.87, green: 0.74, blue: 0.47, alpha: 0.95).cgColor)
    ctx.setLineWidth(ringWidth)
    ctx.strokeEllipse(in: CGRect(x: cx - ringR, y: cy - ringR, width: ringR * 2, height: ringR * 2))
    ctx.restoreGState()

    // Inner subtle brass ring
    let innerR = ringR * 0.84
    ctx.saveGState()
    ctx.setStrokeColor(NSColor(calibratedRed: 0.87, green: 0.74, blue: 0.47, alpha: 0.35).cgColor)
    ctx.setLineWidth(max(1.0, ringWidth * 0.4))
    ctx.strokeEllipse(in: CGRect(x: cx - innerR, y: cy - innerR, width: innerR * 2, height: innerR * 2))
    ctx.restoreGState()

    // Emerald marble with glow
    ctx.saveGState()
    ctx.setShadow(offset: .zero, blur: marbleR * 0.9, color: NSColor(calibratedRed: 0.14, green: 0.88, blue: 0.54, alpha: 0.85).cgColor)
    ctx.setFillColor(NSColor(calibratedRed: 0.16, green: 0.84, blue: 0.52, alpha: 1.0).cgColor)
    ctx.fillEllipse(in: CGRect(x: cx - marbleR, y: cy - marbleR, width: marbleR * 2, height: marbleR * 2))
    ctx.restoreGState()

    // Specular highlight on marble
    let glintR = marbleR * 0.22
    ctx.setFillColor(NSColor(calibratedWhite: 1.0, alpha: 0.82).cgColor)
    ctx.fillEllipse(in: CGRect(x: cx - marbleR * 0.48, y: cy + marbleR * 0.22, width: glintR * 2, height: glintR * 2))
}

func renderAvatar(outPath: String) {
    let w = 800, h = 800
    let ctx = makeContext(w: w, h: h)
    let cs = CGColorSpaceCreateDeviceRGB()

    ctx.setFillColor(NSColor(calibratedRed: 0.055, green: 0.063, blue: 0.078, alpha: 1.0).cgColor)
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))

    let gradColors = [
        NSColor(calibratedRed: 0.16, green: 0.17, blue: 0.21, alpha: 0.7).cgColor,
        NSColor(calibratedRed: 0.055, green: 0.063, blue: 0.078, alpha: 0.0).cgColor
    ] as CFArray
    if let g = CGGradient(colorsSpace: cs, colors: gradColors, locations: [0, 1]) {
        ctx.drawRadialGradient(g, startCenter: CGPoint(x: 400, y: 400), startRadius: 20, endCenter: CGPoint(x: 400, y: 400), endRadius: 420, options: [])
    }

    // Main emblem sized to read crisply even at 32px thumbnail
    drawMarbleEmblem(ctx: ctx, cx: 400, cy: 430, ringR: 220, marbleR: 112, ringWidth: 10)

    // Minimalist brand mark below emblem (inside circle safe area)
    let nsCtx = NSGraphicsContext(cgContext: ctx, flipped: false)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = nsCtx
    let pStyle = NSMutableParagraphStyle()
    pStyle.alignment = .center
    let font = NSFont(name: "Avenir-Heavy", size: 44) ?? NSFont.boldSystemFont(ofSize: 44)
    let ivory = NSColor(calibratedRed: 0.96, green: 0.94, blue: 0.90, alpha: 0.96)
    NSAttributedString(string: "LUCA MARBLE", attributes: [
        .font: font, .foregroundColor: ivory, .kern: 7.0, .paragraphStyle: pStyle
    ]).draw(in: CGRect(x: 80, y: 120, width: 640, height: 60))
    NSGraphicsContext.restoreGraphicsState()

    savePNG(ctx, to: outPath)
}

func renderBanner(outPath: String) {
    let w = 2560, h = 1440
    let ctx = makeContext(w: w, h: h)
    let cs = CGColorSpaceCreateDeviceRGB()

    ctx.setFillColor(NSColor(calibratedRed: 0.055, green: 0.063, blue: 0.078, alpha: 1.0).cgColor)
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))

    let gradColors = [
        NSColor(calibratedRed: 0.15, green: 0.16, blue: 0.20, alpha: 0.65).cgColor,
        NSColor(calibratedRed: 0.055, green: 0.063, blue: 0.078, alpha: 0.0).cgColor
    ] as CFArray
    if let g = CGGradient(colorsSpace: cs, colors: gradColors, locations: [0, 1]) {
        ctx.drawRadialGradient(g, startCenter: CGPoint(x: 1280, y: 720), startRadius: 40, endCenter: CGPoint(x: 1280, y: 720), endRadius: 900, options: [])
    }

    // Subtle decorative brass rail lines across full desktop width (1546 safe zone stays clean)
    ctx.setStrokeColor(NSColor(calibratedRed: 0.86, green: 0.73, blue: 0.46, alpha: 0.20).cgColor)
    ctx.setLineWidth(2.0)
    ctx.move(to: CGPoint(x: 0, y: 720 + 211))
    ctx.addLine(to: CGPoint(x: 2560, y: 720 + 211))
    ctx.move(to: CGPoint(x: 0, y: 720 - 211))
    ctx.addLine(to: CGPoint(x: 2560, y: 720 - 211))
    ctx.strokePath()

    // Safe Zone: 1546 x 423 centered at (1280, 720) -> x: 507..2053, y: 508.5..931.5
    drawMarbleEmblem(ctx: ctx, cx: 740, cy: 720, ringR: 96, marbleR: 48, ringWidth: 5.5)

    let nsCtx = NSGraphicsContext(cgContext: ctx, flipped: false)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = nsCtx

    let ivory = NSColor(calibratedRed: 0.96, green: 0.94, blue: 0.90, alpha: 1.0)
    let brass = NSColor(calibratedRed: 0.87, green: 0.75, blue: 0.49, alpha: 0.96)
    let muted = NSColor(calibratedRed: 0.74, green: 0.73, blue: 0.71, alpha: 0.92)

    let titleFont = NSFont(name: "Avenir-Heavy", size: 78) ?? NSFont.boldSystemFont(ofSize: 78)
    let subFont = NSFont(name: "Georgia-Bold", size: 36) ?? NSFont.boldSystemFont(ofSize: 36)
    let tagFont = NSFont(name: "Avenir-Medium", size: 28) ?? NSFont.systemFont(ofSize: 28)

    let textX: CGFloat = 900
    let textW: CGFloat = 1080
    NSAttributedString(string: "LUCA MARBLE MUSIC", attributes: [
        .font: titleFont, .foregroundColor: ivory, .kern: 6.5
    ]).draw(in: CGRect(x: textX, y: 735, width: textW, height: 96))

    NSAttributedString(string: "Mechanical 3D Music Box  ·  Physics-Driven Melodies", attributes: [
        .font: subFont, .foregroundColor: brass, .kern: 0.8
    ]).draw(in: CGRect(x: textX, y: 670, width: textW, height: 48))

    NSAttributedString(string: "@LucaMarbleMusic  ·  Sculpted Acoustic Instruments", attributes: [
        .font: tagFont, .foregroundColor: muted, .kern: 2.2
    ]).draw(in: CGRect(x: textX, y: 615, width: textW, height: 38))

    NSGraphicsContext.restoreGraphicsState()
    savePNG(ctx, to: outPath)
}

func renderWatermark(outPath: String) {
    let w = 150, h = 150
    let ctx = makeContext(w: w, h: h)
    ctx.clear(CGRect(x: 0, y: 0, width: w, height: h))

    // Subtle dark backing circle so it stays legible on any scene
    ctx.setFillColor(NSColor(calibratedRed: 0.055, green: 0.063, blue: 0.078, alpha: 0.82).cgColor)
    ctx.fillEllipse(in: CGRect(x: 8, y: 8, width: 134, height: 134))

    drawMarbleEmblem(ctx: ctx, cx: 75, cy: 75, ringR: 58, marbleR: 28, ringWidth: 4.0)
    savePNG(ctx, to: outPath)
}

let outDir = CommandLine.arguments[1]
renderAvatar(outPath: "\(outDir)/avatar_800x800.png")
renderBanner(outPath: "\(outDir)/banner_2560x1440.png")
renderWatermark(outPath: "\(outDir)/watermark_150x150.png")
'''


def main() -> None:
    KIT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        swift_file = Path(tmp) / "gen_kit.swift"
        swift_file.write_text(SWIFT_KIT, encoding="utf-8")
        subprocess.run(["swift", str(swift_file), str(KIT_DIR)], check=True)
    print(f"Generated channel assets in {KIT_DIR}")


if __name__ == "__main__":
    main()
