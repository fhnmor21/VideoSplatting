#!/usr/bin/env bash
# =============================================================================
# run_colmap.sh — COLMAP sparse reconstruction for Gaussian Splatting
#
# Optimised for:
#   * Google Pixel phone (wide-angle primary lens, fixed focal length)
#   * Blackmagic Camera App (consistent colour science, no auto-crop)
#   * Interior building walkthrough (contiguous, overlapping frames)
#
# Pipeline stages:
#   1. feature_extractor   — SIFT keypoint detection per image
#   2. sequential_matcher  — matches neighbouring frames (exploits video order)
#   3. mapper              — incremental SfM -> sparse point cloud + cameras
#   4. image_undistorter   — rectify images for downstream (3DGS / NeRF)
#
# Output layout:
#   <workspace>/
#     database.db          — COLMAP feature/match database
#     sparse/0/            — cameras.bin, images.bin, points3D.bin
#     sparse/0/txt/        — human-readable copies of the above
#     dense/               — undistorted images + sparse model copy
#
# Usage:
#   ./run_colmap.sh -i <frames_dir> [OPTIONS]
#
# Options:
#   -i  Input frames directory (required, output of extract_frames.sh)
#   -w  Workspace directory    (default: ./colmap_workspace)
#   -c  Camera model           (default: OPENCV)
#   -f  Known focal length in pixels (optional but strongly recommended)
#       Compute: focal_px = (image_width_px / sensor_width_mm) * focal_length_mm
#       Pixel 8 example (24mm equiv, 6.9mm sensor, 4032px wide):
#         focal_px = 4032 / 6.9 * 3.65 = ~2133  -> pass -f 2133
#   -F  Force single camera model for all images: 1=yes 0=no (default: 1)
#   -g  GPU index for feature extraction/matching, -1=CPU (default: 0)
#   -t  Number of CPU threads (default: all available)
#   -v  Vocab tree path for loop-closure detection (optional)
#       Download: https://demuc.de/colmap/#dataset
#   -s  Skip feature extraction/matching (1=yes), go straight to mapper
#   -u  Run image_undistorter for 3DGS/NeRF output: 1=yes 0=no (default: 1)
#   -h  Show this help
#
# Requirements:
#   colmap >= 3.8  (https://colmap.github.io/install.html)
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
INPUT_DIR=""
WORKSPACE="./colmap_workspace"
CAMERA_MODEL="OPENCV"        # handles radial + tangential distortion (k1,k2,p1,p2)
FOCAL_PX=""                  # empty = COLMAP estimates from scratch
SINGLE_CAMERA="1"            # all frames share one camera (same phone, same lens)
GPU_INDEX="0"                # set -1 to force CPU
THREADS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
VOCAB_TREE=""
SKIP_TO_MAPPER="0"
RUN_UNDISTORT="1"

# --------------------------------------------------------------------------- #
# Google Pixel / Blackmagic Camera App focal-length priors
# --------------------------------------------------------------------------- #
# Blackmagic Camera always locks to the main (wide) lens: ~24mm equiv.
# Sensor geometry is consistent across Pixel 6/7/8/9:
#   Sensor width:    ~6.9 mm
#   Pixel pitch:     ~0.8 um   (IMX787 / IMX858)
#   24mm-equiv EFL:  ~3.65 mm actual
#
# focal_px = image_width_px * (actual_EFL / sensor_width_mm)
#          = image_width_px * (3.65 / 6.9)
#          = image_width_px * 0.5290
#
# Common resolutions:
#   4032 x 3024 (12 MP default)  -> ~2133 px
#   4080 x 3072 (Pixel 9)        -> ~2158 px
#   3840 x 2160 (4K video crop)  -> ~2031 px  (slight crop from 4:3 sensor)
#   1920 x 1080 (FHD)            -> ~1016 px
# --------------------------------------------------------------------------- #
PIXEL_FOCAL_RATIO="0.5290"   # actual_EFL / sensor_width = 3.65 / 6.9

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
usage() {
    sed -n '/^# Usage/,/^# Requirements/p' "$0" | sed 's/^# \{0,3\}//'
    exit 0
}

while getopts ":i:w:c:f:F:g:t:v:s:u:h" opt; do
    case $opt in
        i) INPUT_DIR="$OPTARG" ;;
        w) WORKSPACE="$OPTARG" ;;
        c) CAMERA_MODEL="$OPTARG" ;;
        f) FOCAL_PX="$OPTARG" ;;
        F) SINGLE_CAMERA="$OPTARG" ;;
        g) GPU_INDEX="$OPTARG" ;;
        t) THREADS="$OPTARG" ;;
        v) VOCAB_TREE="$OPTARG" ;;
        s) SKIP_TO_MAPPER="$OPTARG" ;;
        u) RUN_UNDISTORT="$OPTARG" ;;
        h) usage ;;
        :) echo "ERROR: Option -$OPTARG requires an argument." >&2; exit 1 ;;
        \?) echo "ERROR: Unknown option -$OPTARG." >&2; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
if [[ -z "$INPUT_DIR" ]]; then
    echo "ERROR: No input directory specified. Use -i <frames_dir>." >&2
    exit 1
fi
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory not found: $INPUT_DIR" >&2
    exit 1
fi
if ! command -v colmap &>/dev/null; then
    echo "ERROR: 'colmap' not found in PATH." >&2
    echo "       Install: https://colmap.github.io/install.html" >&2
    exit 1
fi

FRAME_COUNT=$(find "$INPUT_DIR" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l | tr -d ' ')
if [[ "$FRAME_COUNT" -lt 3 ]]; then
    echo "ERROR: Need at least 3 frames, found $FRAME_COUNT in $INPUT_DIR" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Setup workspace
# --------------------------------------------------------------------------- #
DATABASE="${WORKSPACE}/database.db"
SPARSE_DIR="${WORKSPACE}/sparse"
DENSE_DIR="${WORKSPACE}/dense"

mkdir -p "$WORKSPACE" "$SPARSE_DIR" "$DENSE_DIR"

# Detect image dimensions from first frame
FIRST_FRAME=$(find "$INPUT_DIR" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort | head -1)

IMG_W=""; IMG_H=""
if command -v ffprobe &>/dev/null; then
    IMG_W=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=width  -of default=noprint_wrappers=1:nokey=1 "$FIRST_FRAME" 2>/dev/null || echo "")
    IMG_H=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=height -of default=noprint_wrappers=1:nokey=1 "$FIRST_FRAME" 2>/dev/null || echo "")
elif command -v identify &>/dev/null; then
    DIMS=$(identify -format "%wx%h" "$FIRST_FRAME" 2>/dev/null || echo "")
    IMG_W="${DIMS%%x*}"; IMG_H="${DIMS##*x}"
fi

# Auto-estimate focal length from Pixel sensor geometry if not supplied
if [[ -z "$FOCAL_PX" && -n "$IMG_W" ]]; then
    FOCAL_PX=$(echo "scale=0; $IMG_W * $PIXEL_FOCAL_RATIO / 1" | bc 2>/dev/null || echo "")
    [[ -n "$FOCAL_PX" ]] && echo "INFO: Auto-estimated focal length = ${FOCAL_PX} px  (${IMG_W}px width, Pixel main cam geometry)"
    echo "      Override with -f <focal_px> if you know the exact value."
fi

# Build OPENCV camera params string: fx,fy,cx,cy,k1,k2,p1,p2
# cx/cy at image centre; distortion coefficients start at 0 (COLMAP refines).
CAMERA_PARAMS=""
if [[ -n "$FOCAL_PX" && -n "$IMG_W" && -n "$IMG_H" ]]; then
    CX=$(echo "scale=1; $IMG_W / 2" | bc)
    CY=$(echo "scale=1; $IMG_H / 2" | bc)
    CAMERA_PARAMS="${FOCAL_PX},${FOCAL_PX},${CX},${CY},0,0,0,0"
fi

# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #
echo "=================================================================="
echo " COLMAP Sparse Reconstruction — Interior Building"
echo "=================================================================="
echo " Camera        : Google Pixel + Blackmagic Camera App"
echo " Input frames  : $INPUT_DIR  ($FRAME_COUNT images)"
[[ -n "$IMG_W" ]] && echo " Image size    : ${IMG_W} x ${IMG_H}"
echo " Workspace     : $WORKSPACE"
echo " Camera model  : $CAMERA_MODEL  (radial + tangential distortion)"
echo " Focal prior   : ${FOCAL_PX:-'none — COLMAP will estimate'} px"
echo " Camera params : ${CAMERA_PARAMS:-'none'}"
echo " Single camera : $SINGLE_CAMERA"
echo " GPU index     : $GPU_INDEX"
echo " Threads       : $THREADS"
[[ -n "$VOCAB_TREE" ]] && echo " Vocab tree    : $VOCAB_TREE" || echo " Vocab tree    : none (loop closure limited)"
echo "=================================================================="

log_step() {
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo " STEP $1: $2"
    echo "──────────────────────────────────────────────────────────"
}

# --------------------------------------------------------------------------- #
# STAGE 1 — Feature Extraction
# --------------------------------------------------------------------------- #
if [[ "$SKIP_TO_MAPPER" != "1" ]]; then

log_step "1" "Feature Extraction (SIFT)"
echo "Detecting SIFT keypoints in all $FRAME_COUNT frames..."
echo ""
echo "Tuning notes (interior buildings):"
echo "  peak_threshold 0.003  — lower than default (0.0067) to find features"
echo "                          on low-contrast surfaces (white walls, ceilings)"
echo "  first_octave -1       — analyse finer scale; catches small surface detail"
echo "  domain_size_pooling   — better descriptors on textureless regions"
echo "  max_num_features 8192 — raise to 16384 if registration is poor"
echo ""

EXTRACT_ARGS=(
    --database_path  "$DATABASE"
    --image_path     "$INPUT_DIR"

    # ---- Camera model ----
    --ImageReader.camera_model  "$CAMERA_MODEL"
    --ImageReader.single_camera "$SINGLE_CAMERA"

    # ---- SIFT settings tuned for dark/low-texture interiors ----
    --SiftExtraction.use_gpu             1
    --SiftExtraction.gpu_index           "$GPU_INDEX"
    --SiftExtraction.num_threads         "$THREADS"
    --SiftExtraction.max_num_features    8192
    --SiftExtraction.first_octave        -1      # finer scale detection
    --SiftExtraction.peak_threshold      0.003   # more keypoints in low contrast
    --SiftExtraction.edge_threshold      10      # suppress edge-only responses
    --SiftExtraction.domain_size_pooling 1       # descriptor quality on flat surfaces
)

[[ -n "$CAMERA_PARAMS" ]] && EXTRACT_ARGS+=(--ImageReader.camera_params "$CAMERA_PARAMS")

colmap feature_extractor "${EXTRACT_ARGS[@]}"
echo "✓ Feature extraction complete."

# --------------------------------------------------------------------------- #
# STAGE 2 — Sequential Matching
# --------------------------------------------------------------------------- #
log_step "2" "Feature Matching (Sequential)"
echo "Matching frames sequentially (exploits capture order from extract_frames.sh)..."
echo ""
echo "  overlap 20       — each frame matched against 20 neighbours each side"
echo "  quadratic_overlap — also matches at 2x, 4x steps to bridge skipped frames"
echo "  loop_detection    — detects when walkthrough revisits a room"
echo ""

MATCH_ARGS=(
    --database_path "$DATABASE"

    # ---- SIFT matching ----
    --SiftMatching.use_gpu      1
    --SiftMatching.gpu_index    "$GPU_INDEX"
    --SiftMatching.num_threads  "$THREADS"
    --SiftMatching.max_ratio    0.80    # Lowe ratio test (default 0.8)
    --SiftMatching.max_distance 0.7
    --SiftMatching.cross_check  1       # mutual nearest-neighbour check

    # ---- Geometric verification ----
    # Interior scenes: short baseline, many repeated elements (tiles, windows).
    # Require a healthy inlier count to avoid spurious model connections.
    --TwoViewGeometry.min_num_inliers 15

    # ---- Sequential matcher ----
    --SequentialMatching.overlap           20    # neighbour window each side
    --SequentialMatching.quadratic_overlap  1    # exponential step sizes
)

if [[ -n "$VOCAB_TREE" && -f "$VOCAB_TREE" ]]; then
    echo "Vocab tree found — enabling loop-closure detection."
    MATCH_ARGS+=(
        --SequentialMatching.loop_detection              1
        --SequentialMatching.vocab_tree_path             "$VOCAB_TREE"
        --SequentialMatching.loop_detection_num_images  50
    )
else
    MATCH_ARGS+=(--SequentialMatching.loop_detection 0)
    echo "No vocab tree — loop detection disabled."
    echo "  TIP: Download vocab.bin from https://demuc.de/colmap/#dataset"
    echo "       and pass -v <path> to enable loop closure for revisited rooms."
fi

colmap sequential_matcher "${MATCH_ARGS[@]}"
echo "✓ Sequential matching complete."

# --------------------------------------------------------------------------- #
# STAGE 2b — Vocab Tree Matching (optional second pass for loop closure)
# --------------------------------------------------------------------------- #
if [[ -n "$VOCAB_TREE" && -f "$VOCAB_TREE" ]]; then
    log_step "2b" "Vocab Tree Matching (loop closure reinforcement)"
    echo "Running global vocab-tree pass to link revisited areas..."
    colmap vocab_tree_matcher \
        --database_path  "$DATABASE" \
        --SiftMatching.use_gpu     1 \
        --SiftMatching.gpu_index   "$GPU_INDEX" \
        --SiftMatching.num_threads "$THREADS" \
        --SiftMatching.max_ratio   0.80 \
        --VocabTreeMatching.vocab_tree_path  "$VOCAB_TREE" \
        --VocabTreeMatching.num_images       50
    echo "✓ Vocab tree matching complete."
fi

fi  # end SKIP_TO_MAPPER block

# --------------------------------------------------------------------------- #
# STAGE 3 — Incremental Mapper (Sparse SfM)
# --------------------------------------------------------------------------- #
log_step "3" "Incremental Mapper (Structure-from-Motion)"
echo "Running SfM reconstruction. This is the longest step."
echo "COLMAP will print progress as it registers each frame..."
echo ""

# Key tuning decisions for phone interior footage:
#
# init_min_num_inliers 100
#   The pair chosen to seed the reconstruction must share 100+ inliers.
#   This avoids initialising on a degenerate or nearly-planar pair (common
#   when panning across a flat wall).
#
# abs_pose_min_num_inliers 30
# abs_pose_min_inlier_ratio 0.25
#   Looser registration thresholds (vs default 30 / 0.25) so that frames with
#   partial overlap (doorways, corners) are still registered.
#
# ba_local_num_images 6
#   Local bundle adjustment window: small enough to be fast, large enough to
#   smooth out drift in a long corridor sequence.
#
# min/max_focal_length_ratio
#   Wide priors accepted; Pixel wide-angle sits outside COLMAP's default range.

colmap mapper \
    --database_path  "$DATABASE" \
    --image_path     "$INPUT_DIR" \
    --output_path    "$SPARSE_DIR" \
    \
    --Mapper.init_min_num_inliers       100   \
    --Mapper.init_max_forward_motion    0.95  \
    \
    --Mapper.abs_pose_min_num_inliers   30    \
    --Mapper.abs_pose_min_inlier_ratio  0.25  \
    \
    --Mapper.ba_local_num_images        6     \
    --Mapper.ba_global_images_ratio     1.1   \
    --Mapper.ba_global_points_ratio     1.1   \
    \
    --Mapper.min_focal_length_ratio     0.1   \
    --Mapper.max_focal_length_ratio    10.0   \
    --Mapper.max_extra_param            1.0   \
    \
    --Mapper.num_threads "$THREADS"

# Find the largest/best reconstruction sub-model
BEST_MODEL=$(find "$SPARSE_DIR" -maxdepth 1 -mindepth 1 -type d | sort -V | head -1)

if [[ -z "$BEST_MODEL" ]]; then
    echo "" >&2
    echo "ERROR: Mapper produced no output. Possible causes:" >&2
    echo "  * Too few matching pairs — increase overlap in extract_frames.sh (-x 1.5)" >&2
    echo "  * Featureless surfaces   — try --SiftExtraction.peak_threshold 0.001" >&2
    echo "  * No qualifying init pair — lower init_min_num_inliers to 50" >&2
    echo "  * Provide a vocab tree (-v) for better connectivity" >&2
    exit 1
fi

echo ""
echo "✓ Sparse reconstruction complete: $BEST_MODEL"

# Count registered images (works for both binary and text formats)
REG_IMAGES=$(python3 -c "
import struct, sys
try:
    with open('${BEST_MODEL}/images.bin', 'rb') as f:
        print(struct.unpack('<Q', f.read(8))[0])
except Exception:
    import os, glob
    lines = [l for l in open('${BEST_MODEL}/images.txt') if l.strip() and not l.startswith('#')]
    # images.txt has 2 lines per image
    print(len(lines) // 2)
" 2>/dev/null || echo "?")

echo "   Registered: $REG_IMAGES / $FRAME_COUNT frames"

# --------------------------------------------------------------------------- #
# STAGE 3b — Export human-readable text model
# --------------------------------------------------------------------------- #
log_step "3b" "Exporting text model"
TXT_DIR="${BEST_MODEL}/txt"
mkdir -p "$TXT_DIR"
colmap model_converter \
    --input_path  "$BEST_MODEL" \
    --output_path "$TXT_DIR" \
    --output_type TXT
echo "✓ Text model saved to $TXT_DIR"
echo "   cameras.txt / images.txt / points3D.txt"

# --------------------------------------------------------------------------- #
# STAGE 4 — Image Undistortion
# --------------------------------------------------------------------------- #
# Produces a COLMAP-format dense/ directory that 3D Gaussian Splatting
# (gaussian-splatting repo) expects as its -s / --source_path argument.
# --------------------------------------------------------------------------- #
if [[ "$RUN_UNDISTORT" == "1" ]]; then
    log_step "4" "Image Undistortion (for 3DGS / NeRF)"
    echo "Undistorting images — removes lens distortion using recovered camera params..."

    colmap image_undistorter \
        --image_path   "$INPUT_DIR" \
        --input_path   "$BEST_MODEL" \
        --output_path  "$DENSE_DIR" \
        --output_type  COLMAP         # outputs sparse/ + images/ compatible with 3DGS
        # Change to --output_type PMVS for classic MVS / OpenMVS pipelines

    echo "✓ Undistorted output ready at $DENSE_DIR"
    echo "   Structure: $DENSE_DIR/images/  (rectified frames)"
    echo "              $DENSE_DIR/sparse/  (copy of camera model)"
fi

# --------------------------------------------------------------------------- #
# Final summary
# --------------------------------------------------------------------------- #
echo ""
echo "=================================================================="
echo " Reconstruction Complete"
echo "=================================================================="
echo ""
echo " Sparse model       : $BEST_MODEL"
echo "   cameras.bin / images.bin / points3D.bin"
echo "   txt/cameras.txt  txt/images.txt  txt/points3D.txt"
[[ "$RUN_UNDISTORT" == "1" ]] && echo " 3DGS source path   : $DENSE_DIR"
echo ""
echo " Frames registered  : $REG_IMAGES / $FRAME_COUNT"
echo ""
echo "──────────────────────────────────────────────────────────"
echo " Next steps"
echo "──────────────────────────────────────────────────────────"
echo ""
echo " Launch 3D Gaussian Splatting training:"
echo "   python train.py -s $DENSE_DIR"
echo ""
echo " Inspect model interactively:"
echo "   colmap gui --database_path $DATABASE --import_path $BEST_MODEL"
echo ""
echo " Convert to NeRF (nerfstudio):"
echo "   ns-process-data images --data $INPUT_DIR \\"
echo "     --colmap-model-path $BEST_MODEL --output-dir ./nerfstudio_data"
echo ""

# Warn if registration rate is low
if [[ "$REG_IMAGES" =~ ^[0-9]+$ && "$FRAME_COUNT" =~ ^[0-9]+$ ]]; then
    THRESHOLD=$(( FRAME_COUNT * 80 / 100 ))
    if (( REG_IMAGES < THRESHOLD )); then
        echo "──────────────────────────────────────────────────────────"
        echo " WARNING: Only $REG_IMAGES / $FRAME_COUNT frames registered"
        echo " (below 80% threshold). Suggestions:"
        echo ""
        echo "  1. Add a vocab tree for loop closure:"
        echo "       -v /path/to/vocab_tree_flickr100K_words32K.bin"
        echo "  2. Re-extract with lower scene threshold (more frames):"
        echo "       ./extract_frames.sh -i <video> -s 0.20 -x 1.5"
        echo "  3. Lower SIFT peak threshold for textureless surfaces:"
        echo "       (edit script: SiftExtraction.peak_threshold 0.001)"
        echo "  4. Check lighting — very dark frames register poorly."
        echo "──────────────────────────────────────────────────────────"
    fi
fi
echo "=================================================================="
