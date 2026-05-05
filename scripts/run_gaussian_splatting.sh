#!/usr/bin/env bash
# =============================================================================
# run_gaussian_splatting.sh — 3D Gaussian Splatting training
#
# Consumes the output of run_colmap.sh and trains a 3DGS scene using the
# official Inria gaussian-splatting repository.
#
# Optimised for:
#   * Interior building scenes (bounded, ~10–100m scale)
#   * Google Pixel + Blackmagic Camera App source footage
#   * Single contiguous walkthrough (no turntable / multi-session data)
#
# Prerequisites (install once):
#   git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
#   cd gaussian-splatting
#   conda env create -f environment.yml
#   conda activate gaussian_splatting
#
# Usage:
#   ./run_gaussian_splatting.sh -s <colmap_dense_dir> [OPTIONS]
#
# Options:
#   -s  Source path: COLMAP dense/ dir from run_colmap.sh (required)
#       Must contain: images/ and sparse/ subdirectories
#   -o  Output directory for trained model (default: ./gs_output)
#   -r  Gaussian Splatting repo path (default: ./gaussian-splatting)
#   -e  Conda env name (default: gaussian_splatting)
#   -i  Training iterations (default: 30000)
#   -d  Densification iterations: start,end (default: 500,15000)
#   -R  Resolutions to train at, comma-separated (default: 1,2,4,8)
#       1=full res, 2=half, 4=quarter etc. Smaller res trains faster.
#   -W  Override image resolution cap in pixels (default: 1600)
#   -p  Port for the training monitor viewer (default: 6009)
#   -t  Test/hold-out every Nth image for eval (default: 8)
#   -c  Checkpoint interval in iterations, 0=disabled (default: 7000)
#   -C  Resume training from checkpoint path (optional)
#   -n  Dry run: print commands without executing (0/1, default: 0)
#   -h  Show this help
#
# Outputs (in <output_dir>/):
#   point_cloud/iteration_XXXXX/point_cloud.ply   — Gaussian splats (PLY)
#   cameras.json                                   — camera extrinsics
#   input.ply                                      — initial sparse point cloud
#   cfg_args                                        — training config record
#
# Viewing the result:
#   Use the SIBR viewer bundled with gaussian-splatting:
#     ./gaussian-splatting/SIBR_viewers/bin/SIBR_gaussianViewer_app \
#       -m <output_dir>
#   Or drag point_cloud.ply into https://antimatter15.com/splat/
#
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
SOURCE_PATH=""
OUTPUT_DIR="./gs_output"
REPO_PATH="./gaussian-splatting"
CONDA_ENV="gaussian_splatting"
ITERATIONS=30000
DENSIFY_START=500
DENSIFY_END=15000
RESOLUTIONS="1,2,4,8"
RESOLUTION_CAP=1600     # images larger than this are downscaled (VRAM guard)
VIEWER_PORT=6009
TEST_EVERY=8            # hold out every Nth frame for PSNR/SSIM eval
CHECKPOINT_INTERVAL=7000
RESUME_CHECKPOINT=""
DRY_RUN=0

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
usage() {
    sed -n '/^# Usage/,/^# Outputs/p' "$0" | sed 's/^# \{0,3\}//'
    exit 0
}

while getopts ":s:o:r:e:i:d:R:W:p:t:c:C:n:h" opt; do
    case $opt in
        s) SOURCE_PATH="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        r) REPO_PATH="$OPTARG" ;;
        e) CONDA_ENV="$OPTARG" ;;
        i) ITERATIONS="$OPTARG" ;;
        d) DENSIFY_START="${OPTARG%%,*}"; DENSIFY_END="${OPTARG##*,}" ;;
        R) RESOLUTIONS="$OPTARG" ;;
        W) RESOLUTION_CAP="$OPTARG" ;;
        p) VIEWER_PORT="$OPTARG" ;;
        t) TEST_EVERY="$OPTARG" ;;
        c) CHECKPOINT_INTERVAL="$OPTARG" ;;
        C) RESUME_CHECKPOINT="$OPTARG" ;;
        n) DRY_RUN="$OPTARG" ;;
        h) usage ;;
        :) echo "ERROR: Option -$OPTARG requires an argument." >&2; exit 1 ;;
        \?) echo "ERROR: Unknown option -$OPTARG." >&2; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Helper: run or print a command depending on DRY_RUN
# --------------------------------------------------------------------------- #
run() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

log_step() {
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo " STEP $1: $2"
    echo "──────────────────────────────────────────────────────────"
}

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
if [[ -z "$SOURCE_PATH" ]]; then
    echo "ERROR: No source path specified. Use -s <colmap_dense_dir>." >&2
    echo "       This should be the dense/ directory produced by run_colmap.sh" >&2
    exit 1
fi

if [[ ! -d "$SOURCE_PATH" ]]; then
    echo "ERROR: Source path not found: $SOURCE_PATH" >&2
    exit 1
fi

# Verify expected COLMAP structure
if [[ ! -d "${SOURCE_PATH}/images" ]]; then
    echo "ERROR: ${SOURCE_PATH}/images/ not found." >&2
    echo "       The source path must be the COLMAP dense/ directory containing" >&2
    echo "       images/ (undistorted frames) and sparse/ (camera model)." >&2
    exit 1
fi

if [[ ! -d "${SOURCE_PATH}/sparse" ]]; then
    echo "ERROR: ${SOURCE_PATH}/sparse/ not found." >&2
    echo "       Run run_colmap.sh with -u 1 (default) to generate undistorted output." >&2
    exit 1
fi

# Verify Gaussian Splatting repo
TRAIN_SCRIPT="${REPO_PATH}/train.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Gaussian Splatting train.py not found at: $TRAIN_SCRIPT" >&2
    echo "" >&2
    echo "  Install the repo first:" >&2
    echo "    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting" >&2
    echo "    cd gaussian-splatting" >&2
    echo "    conda env create -f environment.yml" >&2
    echo "    conda activate gaussian_splatting" >&2
    echo "" >&2
    echo "  Then re-run this script with -r <path/to/gaussian-splatting>" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# GPU detection
# --------------------------------------------------------------------------- #
HAS_GPU=0
GPU_INFO=""
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "")
    [[ -n "$GPU_INFO" ]] && HAS_GPU=1
fi

if [[ "$HAS_GPU" == "0" && "$DRY_RUN" != "1" ]]; then
    echo "WARNING: No NVIDIA GPU detected. 3DGS training requires a CUDA GPU." >&2
    echo "         Training on CPU is not supported by the official repo." >&2
    echo "         Continuing anyway (use -n 1 for a dry run)." >&2
fi

# Count input images
IMAGE_COUNT=$(find "${SOURCE_PATH}/images" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l | tr -d ' ')

# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #
echo "=================================================================="
echo " 3D Gaussian Splatting — Interior Building"
echo "=================================================================="
echo " Source (COLMAP dense) : $SOURCE_PATH"
echo " Input images          : $IMAGE_COUNT"
echo " Output directory      : $OUTPUT_DIR"
echo " GS repo               : $REPO_PATH"
echo " Conda env             : $CONDA_ENV"
echo " Iterations            : $ITERATIONS"
echo " Densification         : ${DENSIFY_START} → ${DENSIFY_END}"
echo " Resolutions           : $RESOLUTIONS"
echo " Resolution cap        : ${RESOLUTION_CAP}px"
echo " Test holdout          : every ${TEST_EVERY}th image"
echo " Checkpoint interval   : ${CHECKPOINT_INTERVAL} iters"
[[ -n "$RESUME_CHECKPOINT" ]] && echo " Resume from           : $RESUME_CHECKPOINT"
[[ -n "$GPU_INFO" ]]          && echo " GPU                   : $GPU_INFO"
[[ "$DRY_RUN" == "1" ]]       && echo " Mode                  : DRY RUN (commands printed, not executed)"
echo "=================================================================="

# --------------------------------------------------------------------------- #
# Activate conda environment
# --------------------------------------------------------------------------- #
log_step "1" "Activating Conda Environment"

# Locate conda init script — location varies by installation
CONDA_SH=""
for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"  \
    "/opt/conda/etc/profile.d/conda.sh"        \
    "/usr/local/conda/etc/profile.d/conda.sh"
do
    if [[ -f "$candidate" ]]; then
        CONDA_SH="$candidate"
        break
    fi
done

# Also try conda in PATH as a fallback
if [[ -z "$CONDA_SH" ]] && command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
    [[ -n "$CONDA_BASE" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] && \
        CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
fi

if [[ -z "$CONDA_SH" ]]; then
    echo "ERROR: Could not find conda.sh. Is Conda installed?" >&2
    echo "  Install Miniconda: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
    # shellcheck source=/dev/null
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    echo "✓ Conda env '$CONDA_ENV' activated."
    python --version
else
    echo "[DRY RUN] source $CONDA_SH && conda activate $CONDA_ENV"
fi

# --------------------------------------------------------------------------- #
# STAGE 2 — Pre-training checks
# --------------------------------------------------------------------------- #
log_step "2" "Pre-training Checks"

# Verify CUDA is accessible from within the env
if [[ "$DRY_RUN" != "1" ]]; then
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in this env!'; \
        print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}')" \
        || { echo "WARNING: CUDA check failed — training will likely crash."; }

    # Verify gaussian_splatting submodules are compiled
    python -c "
from diff_gaussian_rasterization import GaussianRasterizationSettings
from simple_knn._C import distCUDA2
print('  diff_gaussian_rasterization: OK')
print('  simple_knn: OK')
" 2>/dev/null || {
        echo ""
        echo "ERROR: CUDA extensions not compiled. Run inside the repo:" >&2
        echo "  cd $REPO_PATH" >&2
        echo "  pip install submodules/diff-gaussian-rasterization" >&2
        echo "  pip install submodules/simple-knn" >&2
        exit 1
    }
fi

echo "✓ Pre-training checks passed."

# --------------------------------------------------------------------------- #
# STAGE 3 — Build training command
# --------------------------------------------------------------------------- #
log_step "3" "Building Training Command"

mkdir -p "$OUTPUT_DIR"

# Core training arguments
TRAIN_ARGS=(
    python "${TRAIN_SCRIPT}"

    # I/O
    --source_path  "$SOURCE_PATH"
    --model_path   "$OUTPUT_DIR"

    # ---- Iterations ----
    --iterations "$ITERATIONS"

    # ---- Densification schedule ----
    # Interior buildings need more densification time because:
    #   * Large uniform surfaces (walls, floors) start sparse and need many Gaussians
    #   * The scene is bounded — densification converges differently than outdoors
    --densify_from_iter  "$DENSIFY_START"
    --densify_until_iter "$DENSIFY_END"
    --densify_grad_threshold 0.0002    # default 0.0002; lower=more Gaussians, higher=fewer

    # ---- Resolution handling ----
    # resolution -1 lets 3DGS pick automatically based on image size.
    # We use the resolution cap instead to guard VRAM on large phone images.
    --resolution -1
    --resolution_scale 1.0

    # Maximum image side in pixels — Pixel 4K frames (3840px) would otherwise
    # use ~12GB VRAM. 1600px is a good balance for quality vs VRAM.
    # Increase to 1920 or 2560 if you have 24GB+ VRAM.
    --image_resolution_cap "$RESOLUTION_CAP"

    # ---- Evaluation split ----
    # Hold out every Nth image for PSNR/SSIM/LPIPS evaluation.
    # Use --eval to enable; --llffhold sets the stride.
    --eval
    --llffhold "$TEST_EVERY"

    # ---- Opacity / pruning ----
    # Interior scenes accumulate many low-opacity floaters (common near walls).
    # The default opacity reset at iteration 3000 helps prune these.
    --opacity_reset_interval 3000

    # ---- Viewer (background monitor) ----
    # A network viewer runs on this port so you can inspect training live
    # using the SIBR viewer or a web viewer. Set to 0 to disable.
    --ip 127.0.0.1
    --port "$VIEWER_PORT"

    # ---- Checkpointing ----
    # Save intermediate .ply + model state so you can resume or inspect progress
)

# Conditional args
if [[ "$CHECKPOINT_INTERVAL" -gt 0 ]]; then
    TRAIN_ARGS+=(
        --save_iterations "$CHECKPOINT_INTERVAL" "$ITERATIONS"
        --checkpoint_iterations "$CHECKPOINT_INTERVAL" "$ITERATIONS"
    )
else
    TRAIN_ARGS+=(--save_iterations "$ITERATIONS")
fi

# Resume from checkpoint
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    if [[ ! -d "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
        echo "ERROR: Checkpoint not found: $RESUME_CHECKPOINT" >&2
        exit 1
    fi
    TRAIN_ARGS+=(--start_checkpoint "$RESUME_CHECKPOINT")
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

echo "Training command:"
echo ""
printf '  %s \\\n' "${TRAIN_ARGS[@]}"
echo ""

# --------------------------------------------------------------------------- #
# STAGE 4 — Training
# --------------------------------------------------------------------------- #
log_step "4" "Training (${ITERATIONS} iterations)"
echo "This will take 20–90 minutes depending on GPU and image count."
echo "Live progress is printed every 100 iterations."
echo "Connect a viewer at:  http://127.0.0.1:${VIEWER_PORT}  during training."
echo ""

START_TIME=$(date +%s)

run "${TRAIN_ARGS[@]}"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo ""
echo "✓ Training complete in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"

# --------------------------------------------------------------------------- #
# STAGE 5 — Render evaluation images
# --------------------------------------------------------------------------- #
log_step "5" "Rendering Evaluation Images"

RENDER_SCRIPT="${REPO_PATH}/render.py"
METRICS_SCRIPT="${REPO_PATH}/metrics.py"

if [[ -f "$RENDER_SCRIPT" ]]; then
    echo "Rendering held-out test views..."
    run python "$RENDER_SCRIPT" \
        --model_path "$OUTPUT_DIR" \
        --source_path "$SOURCE_PATH" \
        --iteration "$ITERATIONS" \
        --skip_train                   # only render test split
    echo "✓ Renders saved to ${OUTPUT_DIR}/test/"
else
    echo "WARNING: render.py not found at $RENDER_SCRIPT — skipping render step."
fi

# --------------------------------------------------------------------------- #
# STAGE 6 — Compute metrics
# --------------------------------------------------------------------------- #
log_step "6" "Computing Quality Metrics (PSNR / SSIM / LPIPS)"

if [[ -f "$METRICS_SCRIPT" ]]; then
    run python "$METRICS_SCRIPT" \
        --model_path "$OUTPUT_DIR" \
        --iteration "$ITERATIONS"

    METRICS_FILE="${OUTPUT_DIR}/results.json"
    if [[ -f "$METRICS_FILE" && "$DRY_RUN" != "1" ]]; then
        echo ""
        echo "Metrics (test split):"
        python -c "
import json
with open('${METRICS_FILE}') as f:
    data = json.load(f)
for split, vals in data.items():
    print(f'  [{split}]')
    for k, v in vals.items():
        if isinstance(v, float):
            print(f'    {k:<8}: {v:.4f}')
        else:
            print(f'    {k:<8}: {v}')
"
    fi
else
    echo "WARNING: metrics.py not found at $METRICS_SCRIPT — skipping metrics."
fi

# --------------------------------------------------------------------------- #
# Final summary
# --------------------------------------------------------------------------- #
FINAL_PLY="${OUTPUT_DIR}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"

echo ""
echo "=================================================================="
echo " Gaussian Splatting Complete"
echo "=================================================================="
echo ""
echo " Output model    : $OUTPUT_DIR"

if [[ -f "$FINAL_PLY" ]]; then
    PLY_SIZE=$(du -sh "$FINAL_PLY" 2>/dev/null | cut -f1 || echo "?")
    SPLAT_COUNT=$(python3 -c "
# Count Gaussians from PLY header
with open('${FINAL_PLY}', 'rb') as f:
    for line in f:
        line = line.decode('ascii', errors='ignore').strip()
        if line.startswith('element vertex'):
            print(line.split()[-1])
            break
" 2>/dev/null || echo "?")
    echo " Final PLY       : $FINAL_PLY"
    echo " PLY size        : $PLY_SIZE"
    echo " Gaussian count  : $SPLAT_COUNT"
else
    echo " Final PLY       : (not found — check training logs for errors)"
fi

echo ""
echo "──────────────────────────────────────────────────────────"
echo " Viewing the result"
echo "──────────────────────────────────────────────────────────"
echo ""
echo " SIBR real-time viewer (bundled with the repo):"
echo "   ${REPO_PATH}/SIBR_viewers/bin/SIBR_gaussianViewer_app \\"
echo "     -m $OUTPUT_DIR"
echo ""
echo " Web viewer (drag & drop PLY):"
echo "   https://antimatter15.com/splat/"
echo "   https://playcanvas.com/viewer"
echo ""
echo " Convert to .splat format (smaller, web-friendly):"
echo "   npx gsplat-tools convert $FINAL_PLY ${OUTPUT_DIR}/scene.splat"
echo ""
echo "──────────────────────────────────────────────────────────"
echo " Intermediate checkpoints"
echo "──────────────────────────────────────────────────────────"
find "$OUTPUT_DIR/point_cloud" -name "point_cloud.ply" 2>/dev/null | sort || true
echo ""
echo "──────────────────────────────────────────────────────────"
echo " Troubleshooting"
echo "──────────────────────────────────────────────────────────"
echo ""
echo " Floaters / noisy Gaussians:"
echo "   Lower --densify_grad_threshold to 0.00015 for more pruning"
echo "   or raise to 0.0003 to reduce over-densification"
echo ""
echo " Running out of VRAM:"
echo "   Lower -W (resolution cap), e.g. -W 1024"
echo "   Or reduce -i (iterations) to 15000 for a faster draft"
echo ""
echo " Poor quality / blurry:"
echo "   Increase -i to 50000 for more training time"
echo "   Ensure COLMAP registered >80% of frames (check run_colmap.sh output)"
echo "   Consider re-extracting frames with more overlap"
echo "=================================================================="
