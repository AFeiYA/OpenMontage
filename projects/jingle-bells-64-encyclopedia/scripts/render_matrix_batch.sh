#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/luca/OpenMontage"
PROJ="$ROOT/projects/jingle-bells-64-encyclopedia"
BLEND="$PROJ/renders/full_64.blend"
FRAMES_DIR="$PROJ/renders/frames_full_64"
AUDIO="$PROJ/assets/audio/full_64_preroll.wav"
OUT_DIR="$PROJ/renders/matrix"
mkdir -p "$OUT_DIR" "$FRAMES_DIR"

render_short() {
  local name="$1"
  local f_start="$2"
  local f_end="$3"
  local t_off="$4"
  local dur="$5"

  echo "=== Rendering $name (frames $f_start..$f_end, offset=${t_off}s, dur=${dur}s) ==="
  /opt/homebrew/bin/blender -b "$BLEND" -s "$f_start" -e "$f_end" -a

  local raw_mp4="$OUT_DIR/${name}_raw.mp4"
  local final_mp4="$OUT_DIR/${name}.mp4"

  ffmpeg -y -framerate 60 -start_number "$f_start" -i "$FRAMES_DIR/frame_%04d.png" \
    -ss "$t_off" -t "$dur" -i "$AUDIO" \
    -t "$dur" -c:v libx264 -pix_fmt yuv420p -crf 18 -c:a aac -b:a 192k -shortest "$raw_mp4"

  /Users/luca/OpenMontage/.venv/bin/python3 "$PROJ/scripts/render_64_hud_and_shorts.py" \
    --input "$raw_mp4" --output "$final_mp4" --offset "$t_off" --duration "$dur"

  rm -f "$raw_mp4"
  echo "Done: $final_mp4"
}

MODE="${1:-short1}"

case "$MODE" in
  short1)
    render_short "short_01_opening_pianos" 0 1500 0.0 25.0
    ;;
  short2)
    render_short "short_02_mallets_bells" 2040 3540 34.0 25.0
    ;;
  short3)
    render_short "short_03_brass_climax" 9360 10860 156.0 25.0
    ;;
  short4)
    render_short "short_04_world_synth_finale" 17040 18840 284.0 30.0
    ;;
  all_shorts)
    render_short "short_01_opening_pianos" 0 1500 0.0 25.0
    render_short "short_02_mallets_bells" 2040 3540 34.0 25.0
    render_short "short_03_brass_climax" 9360 10860 156.0 25.0
    render_short "short_04_world_synth_finale" 17040 18840 284.0 30.0
    ;;
  full)
    /opt/homebrew/bin/blender -b "$BLEND" -a
    ffmpeg -y -framerate 60 -start_number 0 -i "$FRAMES_DIR/frame_%04d.png" \
      -i "$AUDIO" -t 319.4424 -c:v libx264 -pix_fmt yuv420p -crf 18 -c:a aac -b:a 192k -shortest \
      "$OUT_DIR/full_64_raw.mp4"
    /Users/luca/OpenMontage/.venv/bin/python3 "$PROJ/scripts/render_64_hud_and_shorts.py" \
      --input "$OUT_DIR/full_64_raw.mp4" --output "$OUT_DIR/full_64_hud.mp4" \
      --offset 0.0 --duration 319.4424 --no-cta
    /Users/luca/OpenMontage/.venv/bin/python3 "$ROOT/projects/jingle-bells-marble-music/scripts/render_fovea_brand_outro.py" \
      --input "$OUT_DIR/full_64_hud.mp4" \
      --output "$OUT_DIR/full_64_encyclopedia_luca.mp4" \
      --title "Jingle Bells (64 Instruments)" \
      --series "ACOUSTIC & SYNTH ENCYCLOPEDIA · @LucaMarbleMusic"
    rm -f "$OUT_DIR/full_64_raw.mp4" "$OUT_DIR/full_64_hud.mp4"
    ;;
  *)
    echo "Usage: $0 {short1|short2|short3|short4|all_shorts|full}"
    exit 1
    ;;
esac
