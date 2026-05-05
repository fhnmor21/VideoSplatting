#!/usr/bin/env bash
# =============================================================================
# extract_frames.sh — Smart frame extraction for photogrammetry / Gaussian Splatting
#
# Strategy: Uses ffmpeg's scene-change detection score to extract frames only
# when visual content has changed enough from the previous frame. This naturally
# adapts to variable camera speed — slow movement = fewer frames extracted,
# fast movement = more frames extracted — ensuring good overlap without redundancy.
#
# Usage:
#   ./extract_frames.sh -i <video> [OPTIONS]
#
# Options:
#   -i  Input video file (required)
#   -o  Output directory for frames (default: ./frames)
#   -s  Scene change threshold, 0.0–1.0 (default: 0.35)
#         Lower  → more frames (more sensitive to small motion)
#         Higher → fewer frames (only large scene changes)
#   -m  Minimum gap between frames in seconds (default: 0.5)
#         Prevents burst extraction during fast motion
#   -x  Maximum gap between frames in seconds (default: 3.0)
#         Forces a frame even if scene score is low (avoids gaps in slow motion)
#   -r  Output resolution, e.g. 1920x1080 (default: original resolution)
#   -q  JPEG quality, 1–31 lower=better (default: 2)
#   -h  Show this help message
#
# Requirements: ffmpeg, ffprobe (both in PATH)
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
INPUT=""
OUTPUT_DIR="./frames"
SCENE_THRESHOLD="0.35"   # ffmpeg scene score: 0 = identical, 1 = completely different
MIN_GAP="0.5"            # seconds — minimum time between any two extracted frames
MAX_GAP="3.0"            # seconds — force a frame if this long since the last one
SCALE=""                 # empty = keep original resolution
QUALITY="2"              # ffmpeg jpeg qscale: 1 (best) – 31 (worst)

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
usage() {
    sed -n '/^# Usage/,/^# Requirements/p' "$0" | sed 's/^# \{0,3\}//'
    exit 0
}

while getopts ":i:o:s:m:x:r:q:h" opt; do
    case $opt in
        i) INPUT="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        s) SCENE_THRESHOLD="$OPTARG" ;;
        m) MIN_GAP="$OPTARG" ;;
        x) MAX_GAP="$OPTARG" ;;
        r) SCALE="$OPTARG" ;;
        q) QUALITY="$OPTARG" ;;
        h) usage ;;
        :) echo "ERROR: Option -$OPTARG requires an argument." >&2; exit 1 ;;
        \?) echo "ERROR: Unknown option -$OPTARG." >&2; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
if [[ -z "$INPUT" ]]; then
    echo "ERROR: No input file specified. Use -i <video>." >&2
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: Input file not found: $INPUT" >&2
    exit 1
fi

for cmd in ffmpeg ffprobe; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' not found in PATH. Please install ffmpeg." >&2
        exit 1
    fi
done

# Validate numeric options
for var_name in SCENE_THRESHOLD MIN_GAP MAX_GAP QUALITY; do
    val="${!var_name}"
    if ! [[ "$val" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        echo "ERROR: $var_name must be a positive number, got: $val" >&2
        exit 1
    fi
done

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " Smart Frame Extractor for Photogrammetry / Gaussian Splatting"
echo "============================================================"
echo " Input       : $INPUT"
echo " Output dir  : $OUTPUT_DIR"
echo " Scene thresh: $SCENE_THRESHOLD  (0=identical … 1=totally different)"
echo " Min gap     : ${MIN_GAP}s"
echo " Max gap     : ${MAX_GAP}s"
echo " JPEG quality: $QUALITY (1=best, 31=worst)"
[[ -n "$SCALE" ]] && echo " Scale       : $SCALE" || echo " Scale       : original"
echo "============================================================"

# Probe video duration and frame rate for progress reporting
DURATION=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null || echo "unknown")
FPS=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=r_frame_rate \
    -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1 || echo "unknown")

echo " Duration    : ${DURATION}s"
echo " Source FPS  : $FPS"
echo "============================================================"
echo ""
echo "Extracting frames... (this may take a while for long videos)"
echo ""

# --------------------------------------------------------------------------- #
# Build the ffmpeg filter chain
#
# How it works:
#
#   select='...'   — keeps a frame only when the expression is true
#
#   The expression:
#     (A) gt(scene, THRESHOLD)          scene score exceeds our sensitivity
#         AND
#         gte(t - prev_selected_t, MIN) at least MIN_GAP seconds since last kept frame
#   OR
#     (B) gte(t - prev_selected_t, MAX) it's been MAX_GAP seconds — force a frame
#         AND not(eq(prev_selected_t,0) and eq(t,0))  (skip the very first frame
#                                                       from the forced path to
#                                                       avoid double-export at t=0)
#
#   The first frame (prev_selected_t = -MAX_GAP so t=0 passes condition B) is
#   always captured.
#
#   vsync=vfr    — output only the selected frames (variable frame rate output)
#   qscale:v     — JPEG quality
# --------------------------------------------------------------------------- #

# Optionally prepend a scale filter
if [[ -n "$SCALE" ]]; then
    SCALE_FILTER="scale=${SCALE},"
else
    SCALE_FILTER=""
fi

SELECT_EXPR="
gt(scene\,${SCENE_THRESHOLD})*gte(t-prev_selected_t\,${MIN_GAP})+
gte(t-prev_selected_t\,${MAX_GAP})
"
# Remove whitespace so ffmpeg sees a single expression string
SELECT_EXPR=$(echo "$SELECT_EXPR" | tr -d ' \n')

ffmpeg \
    -i "$INPUT" \
    -vf "${SCALE_FILTER}select='${SELECT_EXPR}',metadata=print:file=-" \
    -vsync vfr \
    -qscale:v "$QUALITY" \
    -frame_pts true \
    "${OUTPUT_DIR}/frame_%06d.jpg" \
    2>&1 | grep -v "^$" | grep -E "(frame|fps|time=|select|pts|Error|error)" || true

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
FRAME_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "frame_*.jpg" | wc -l | tr -d ' ')
echo ""
echo "============================================================"
echo " Done!"
echo " Extracted frames : $FRAME_COUNT"
echo " Output directory : $OUTPUT_DIR"
echo "============================================================"

if [[ "$FRAME_COUNT" -eq 0 ]]; then
    echo ""
    echo "WARNING: No frames were extracted!"
    echo "  Try lowering the scene threshold (-s), e.g.: -s 0.15"
    echo "  Or lower the max gap (-x) to force more frames: -x 1.0"
    exit 1
fi

# Rough coverage check
if [[ "$DURATION" != "unknown" && "$FRAME_COUNT" -gt 0 ]]; then
    AVG_GAP=$(echo "scale=2; $DURATION / $FRAME_COUNT" | bc 2>/dev/null || echo "?")
    echo " Avg gap between frames: ~${AVG_GAP}s"
    echo ""
    echo "TIP: For Gaussian Splatting, aim for 60–80% overlap between frames."
    echo "     If coverage looks sparse, try: -s $(echo "$SCENE_THRESHOLD * 0.7" | bc) -x $(echo "$MAX_GAP * 0.7" | bc)"
fi
