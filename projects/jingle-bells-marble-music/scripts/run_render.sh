#!/bin/bash
set -e

BLENDER_BIN=""
if command -v blender &> /dev/null; then
    BLENDER_BIN="blender"
elif [ -f "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
    BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
else
    echo "Blender not found! Please ensure Blender is installed."
    exit 1
fi

echo "Using Blender: $BLENDER_BIN"

PROJECT_DIR="projects/jingle-bells-marble-music"
JSON_FILE="${1:-$PROJECT_DIR/artifacts/sample_14s_notes.json}"
AUDIO_FILE="${2:-$PROJECT_DIR/assets/audio/sample_14s.wav}"
OUTPUT_MP4="${3:-$PROJECT_DIR/renders/sample_14s_traj.mp4}"
FRAMES_DIR="$PROJECT_DIR/renders/frames_traj"

START_FRAME="${4:-0}"
END_FRAME="${5:-}"

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$FRAMES_DIR"

# Ensure trajectory schema (bounces + samples + rails) before Blender consumes JSON
if command -v python3 >/dev/null 2>&1; then
  PY="${ROOT_DIR}/.venv/bin/python"
  [ -x "$PY" ] || PY="python3"
  echo "Upgrading / verifying marble trajectory JSON..."
  "$PY" "$PROJECT_DIR/scripts/generate_marble_trajectory.py" || true
fi

echo "Building 3D Scene in Blender..."
"$BLENDER_BIN" -b -P "$PROJECT_DIR/scripts/blender_render_marble_run.py" -- "$JSON_FILE" "$AUDIO_FILE" "$OUTPUT_MP4"

BLEND_FILE="${OUTPUT_MP4%.mp4}.blend"

if [ -n "$END_FRAME" ]; then
    echo "Rendering frame range $START_FRAME to $END_FRAME in Blender..."
    "$BLENDER_BIN" -b "$BLEND_FILE" -s "$START_FRAME" -e "$END_FRAME" -a
else
    echo "Rendering full sequence in Blender..."
    "$BLENDER_BIN" -b "$BLEND_FILE" -a
fi

echo "Compiling video with FFmpeg..."
ffmpeg -y -framerate 60 -i "$FRAMES_DIR/frame_%04d.png" -i "$AUDIO_FILE" -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 192k -shortest "$OUTPUT_MP4"

echo "Render complete! Deliverable at: $OUTPUT_MP4"
