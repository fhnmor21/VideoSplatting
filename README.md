# Video to 3D Gaussian Splatting Pipeline

**End-to-end workflow:** Video → Frame Extraction → COLMAP Reconstruction → 3D Gaussian Splatting

Optimised for Google Pixel cameras (wide lens) + Blackmagic Camera App, filming **contiguous interior building walkthroughs** in variable lighting.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Usage Guide](#usage-guide)
   - [Basic Usage](#basic-usage)
   - [All Command-Line Options](#all-command-line-options)
   - [Stage-by-Stage Tuning](#stage-by-stage-tuning)
7. [Output Structure](#output-structure)
8. [Resuming & Skipping Stages](#resuming--skipping-stages)
9. [Viewing Results](#viewing-results)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Topics](#advanced-topics)
    - [Camera Geometry (Pixel Sensor)](#camera-geometry--pixel-sensor)
    - [Interior Building Tuning](#interior-building-tuning)
    - [Project Structure](#project-structure)
12. [Individual Stage Scripts](#individual-stage-scripts)

---

## Overview

This project automates the complete video-to-3D-splats workflow in a single Python CLI. Instead of running separate bash scripts for frame extraction, COLMAP reconstruction, and Gaussian Splatting training, you call one command:

```bash
python main.py building.mp4
```

And it orchestrates:

1. **Frame Extraction** — Smart scene-change detection with ffmpeg
2. **COLMAP Reconstruction** — Sparse SfM with camera poses + 3D points
3. **Gaussian Splatting** — Neural radiance training on extracted points

Each stage outputs intermediate results so you can inspect geometry, check registration, and resume interrupted runs.

### Why This Approach?

- **Single entry point** — no need to manually run three separate scripts
- **Structured logging** — timestamped, colour-coded output shows progress at each stage
- **Resume capability** — crash during hour 2 of training? Call `--resume` to pick up where you left off
- **Dry-run mode** — print all commands without executing; audit parameters before GPU time
- **Modular design** — each stage is a self-contained Python class; easy to extend or test individually

---

## Pipeline Architecture

### Data Flow

```
building.mp4 (input video)
    │
    ├─ [Stage 1: Frame Extraction]
    │  Uses ffmpeg's scene-change detection score to extract frames
    │  at variable intervals (not every frame).
    │
    ├─→ frames/frame_000001.jpg … frame_000312.jpg  (~5–30 sec spacing)
    │
    ├─ [Stage 2: COLMAP Reconstruction]
    │  Detects SIFT keypoints, matches via sequential + vocab tree,
    │  runs incremental SfM to recover camera poses + sparse points.
    │
    ├─→ colmap/sparse/0/
    │   ├─ cameras.bin (intrinsics + extrinsics)
    │   ├─ images.bin  (per-image pose)
    │   ├─ points3D.bin (sparse point cloud)
    │   └─ txt/        (human-readable text copies)
    │
    ├─→ colmap/dense/
    │   ├─ images/     (undistorted frames)
    │   └─ sparse/     (camera model copy)
    │
    ├─ [Stage 3: 3D Gaussian Splatting]
    │  Initializes Gaussians from sparse points, trains via
    │  differentiable rasterization, densifies/prunes over 30K iterations.
    │
    ├─→ gaussian/point_cloud/iteration_30000/point_cloud.ply
    │   └─ (final 3D splat model — YOUR OUTPUT)
    │
    └─→ gaussian/test/
        └─ (rendered evaluation images)
```

### Stage Details

| Stage | Role | Key Tools | Tunable Params |
|-------|------|-----------|---|
| **1. Extract** | Scene-aware frame sampling | ffmpeg + select filter | `--scene-threshold`, `--min/max-gap` |
| **2. COLMAP** | Sparse SfM reconstruction | COLMAP feature_extractor, sequential_matcher, vocab_tree_matcher, mapper, image_undistorter | `--focal-px`, `--vocab-tree`, `--colmap-gpu` |
| **3. Gaussian** | Neural radiance training | gaussian-splatting (conda env) | `--iterations`, `--densify-*`, `--resolution-cap` |

---

## Prerequisites

### System Requirements

| Component | Requirement | Install |
|-----------|-------------|---------|
| **OS** | Linux or macOS | — |
| **GPU** | NVIDIA CUDA (RTX 3060 or better) | https://developer.nvidia.com/cuda-downloads |
| **Python** | 3.8+ | https://www.python.org |
| **ffmpeg** | ≥ 5.0 | https://ffmpeg.org/download.html |
| **COLMAP** | ≥ 3.8 | https://colmap.github.io/install.html |
| **Miniconda** | (for conda env) | https://docs.conda.io/en/latest/miniconda.html |

### Gaussian Splatting Repository

The pipeline calls the official Inria [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) repository, which must be cloned and its dependencies installed:

```bash
# Clone the repo (with CUDA submodules)
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
cd gaussian-splatting

# Create the conda environment
conda env create -f environment.yml
conda activate gaussian_splatting

# Verify CUDA extensions compiled
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('✓ OK')"
```

If the last line fails, install the extensions manually:

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### Optional: Vocabulary Tree for Loop Closure

For better reconstruction in rooms that are revisited during the walkthrough, download a vocab tree:

```bash
# Download the 256K-word tree (~800 MB)
wget https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin

# Pass to the pipeline
python main.py building.mp4 --vocab-tree vocab_tree_flickr100K_words256K.bin
```

---

## Installation

### 1. Clone or Download This Project

```bash
git clone <this-repo>  # or download as zip
cd gs_pipeline
```

### 2. Install Python Dependencies (Optional)

This project uses only the Python standard library for orchestration. However, for better output or future extensions, you can optionally install:

```bash
pip install -r requirements.txt
```

Currently empty, but placeholder for future nice-to-haves like `tqdm`, `rich`, etc.

### 3. Verify All Tools

```bash
python main.py --help
```

This should print the full CLI options. The script will also check for ffmpeg, COLMAP, and the GS repo during `--preflight` checks when you run an actual pipeline.

---

## Quick Start

### Minimal Example

```bash
# Assumes gaussian-splatting is at ./gaussian-splatting
python main.py building.mp4
```

**What happens:**
1. Extracts frames from `building.mp4`
2. Runs COLMAP reconstruction
3. Trains a Gaussian Splatting model
4. Saves final `point_cloud.ply` to `gs_pipeline_output/gaussian/`

### With Custom Paths

```bash
python main.py building.mp4 \
    -o ~/my_splat_output \
    --gs-repo ~/src/gaussian-splatting \
    --vocab-tree ~/colmap_vocab.bin
```

### Dry Run (Audit Parameters)

```bash
python main.py building.mp4 --dry-run
```

Prints every command without executing. Use this to:
- Check ffmpeg / COLMAP / train.py argument lists
- Verify scene-change thresholds, focal lengths, iteration counts
- Spot mistakes before committing hours of GPU time

### Resume an Interrupted Run

```bash
python main.py building.mp4 --resume
```

Skips any stage whose outputs already exist. For example, if training crashed on iteration 28,000, resuming will skip extract + COLMAP and jump straight to training (or run metrics if the final PLY exists).

---

## Usage Guide

### Basic Usage

```bash
python main.py <video_file> [options]
```

### All Command-Line Options

```
positional arguments:
  video                         Input video file

general options:
  -o, --output PATH             Output root directory
                                (default: ./gs_pipeline_output)
  --gs-repo PATH                Path to cloned gaussian-splatting repo
                                (default: ./gaussian-splatting)
  --conda-env NAME              Conda env with GS installed
                                (default: gaussian_splatting)
  --dry-run                     Print commands, don't execute
  --resume                      Skip stages with existing outputs

stage selection:
  --skip-extract                Skip frame extraction
  --skip-colmap                 Skip COLMAP reconstruction
  --skip-training               Skip Gaussian Splatting training
  --skip-render                 Skip render + metrics evaluation
```

#### Frame Extraction Options

```
--scene-threshold T             Scene-change score threshold (0–1)
                                Default: 0.35
                                Lower → more frames extracted
                                Higher → fewer, sparser frames

--min-gap S                     Minimum seconds between extracted frames
                                Default: 0.5
                                Prevents burst extraction during fast pans

--max-gap S                     Force a frame if this long since last
                                Default: 3.0
                                Prevents gaps during slow crawls

--frame-quality Q               JPEG quality (1=best, 31=worst)
                                Default: 2 (high quality)
```

#### COLMAP Options

```
--camera-model MODEL            Camera distortion model
                                Choices: SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV
                                Default: OPENCV  (k1, k2, p1, p2 — best for phones)

--focal-px F                    Known focal length in pixels
                                Default: auto-estimated from Pixel geometry
                                Override if you know the exact value (e.g., 2133 for 4032px width)

--vocab-tree PATH               Path to COLMAP vocab tree binary
                                Default: none (loop closure disabled)
                                Download: https://demuc.de/colmap/#dataset

--colmap-gpu IDX                GPU index for COLMAP
                                Default: 0
                                Use -1 to force CPU (slow)

--colmap-threads N              CPU threads for COLMAP
                                Default: all available
```

#### Gaussian Splatting Options

```
--iterations N                  Total training iterations
                                Default: 30000
                                Increase to 50000 for higher quality

--densify-start N               Start Gaussian densification iteration
                                Default: 500

--densify-end N                 Stop Gaussian densification iteration
                                Default: 15000
                                Set to ~half of --iterations for interiors
                                (flat surfaces over-densify if run too long)

--densify-grad-threshold T      Gradient threshold for splitting Gaussians
                                Default: 0.0002
                                Lower → more Gaussians (slower, higher quality)
                                Raise to 0.0003 if too noisy

--resolution-cap PX             Cap longest image side (pixels)
                                Default: 1600
                                Guards VRAM — Pixel 4K frames (~3840px) use ~16GB without this
                                Increase to 1920 or 2560 if you have 24GB+ VRAM

--test-holdout N                Hold out every Nth frame for evaluation
                                Default: 8
                                Uses these frames to compute PSNR/SSIM/LPIPS

--checkpoint-interval N         Save PLY checkpoint every N iterations
                                Default: 7000
                                Set to 0 to only save at the end

--viewer-port PORT              Port for live SIBR training monitor
                                Default: 6009
                                Set to 0 to disable the viewer
```

### Stage-by-Stage Tuning

#### Frame Extraction Tuning

**Goal:** Extract 50–300 frames with good overlap but no near-duplicates.

| Symptom | Cause | Fix |
|---------|-------|-----|
| Only 10 frames extracted | Threshold too high | Lower `--scene-threshold` to 0.20 |
| 1000+ frames (slow COLMAP) | Threshold too low | Raise `--scene-threshold` to 0.50 |
| Big gaps in output | Max gap too large | Lower `--max-gap` to 1.5 |
| Burst of 5 identical frames | Min gap too low | Raise `--min-gap` to 1.0 |

**Example:** For a 3-minute interior walkthrough:
```bash
python main.py building.mp4 \
    --scene-threshold 0.30 \
    --min-gap 0.4 \
    --max-gap 2.0
```

#### COLMAP Tuning

**Goal:** Register >80% of extracted frames.

| Symptom | Cause | Fix |
|---------|-------|-----|
| 40% registration | Weak matches | Add `--vocab-tree` for loop closure |
| | Poor lighting | Check video for dark passages; may need better lighting |
| | Sparse features | Increase SIFT peak threshold (edit `stage_colmap.py` line 86: `0.001` instead of `0.003`) |
| Reconstruction looks planar | Bad seed pair | Lower `--init-min-num-inliers` in mapper (edit `stage_colmap.py` line 179) |

**Example:** For a room revisited during walkthrough:
```bash
python main.py building.mp4 \
    --vocab-tree vocab_tree_flickr100K_words256K.bin \
    --colmap-threads 8
```

#### Gaussian Splatting Tuning

**Goal:** High-quality splats without floaters, in reasonable time.

| Symptom | Cause | Fix |
|---------|-------|-----|
| Floaters/noise near walls | Over-densification | Lower `--densify-grad-threshold` to 0.00015 |
| CUDA OOM during training | Images too large | Lower `--resolution-cap` to 1024 |
| Training too slow | GPU bottleneck or too many Gaussians | Reduce `--iterations` to 15000 for draft |
| Blurry result | Insufficient training | Increase `--iterations` to 50000 |
| Weird "popping" Gaussians | Bad COLMAP poses | Re-run COLMAP with more verbose logging |

**Example:** For a dark interior with lots of floaters:
```bash
python main.py building.mp4 \
    --iterations 50000 \
    --densify-end 20000 \
    --densify-grad-threshold 0.00015 \
    --resolution-cap 1024
```

---

## Output Structure

```
gs_pipeline_output/                       ← default, use -o to change
├── frames/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── frame_000NNN.jpg
│   
├── colmap/
│   ├── database.db                       ← COLMAP feature/match store
│   ├── sparse/
│   │   └── 0/                            ← best sub-model
│   │       ├── cameras.bin               ← camera intrinsics + extrinsics
│   │       ├── images.bin                ← per-image poses
│   │       ├── points3D.bin              ← sparse point cloud
│   │       └── txt/                      ← human-readable versions
│   │           ├── cameras.txt
│   │           ├── images.txt
│   │           └── points3D.txt
│   └── dense/                            ← 3DGS source path
│       ├── images/                       ← undistorted frames
│       │   ├── 0000.png
│       │   └── NNNN.png
│       ├── sparse/
│       │   ├── cameras.bin
│       │   ├── images.bin
│       │   └── points3D.bin
│       └── (calibration files)
│
└── gaussian/                             ← 3DGS output
    ├── point_cloud/
    │   ├── iteration_7000/
    │   │   └── point_cloud.ply           ← checkpoint splat
    │   ├── iteration_14000/
    │   │   └── point_cloud.ply
    │   └── iteration_30000/
    │       └── point_cloud.ply           ← FINAL SPLAT (your output!)
    ├── test/                             ← rendered test views
    │   ├── 0000.png
    │   └── NNNN.png
    ├── train/                            ← rendered training views (optional)
    ├── results.json                      ← PSNR / SSIM / LPIPS metrics
    ├── cameras.json
    ├── input.ply
    └── cfg_args
```

**The file you care about:** `gaussian/point_cloud/iteration_30000/point_cloud.ply`

---

## Resuming & Skipping Stages

### Resume from Interruption

If the pipeline crashes partway through, resume from where it stopped:

```bash
python main.py building.mp4 --resume
```

The pipeline skips any stage whose outputs already exist:
- ✅ Extract skipped if `frames/` has images
- ✅ COLMAP skipped if `colmap/sparse/` has a model
- ✅ Training skipped if `gaussian/point_cloud/iteration_30000/point_cloud.ply` exists

### Run Specific Stages Only

Extract only:
```bash
python main.py building.mp4 --skip-colmap --skip-training
```

COLMAP only (frames must already exist):
```bash
python main.py building.mp4 --skip-extract --skip-training
```

Training only (COLMAP must already be done):
```bash
python main.py building.mp4 --skip-extract --skip-colmap
```

Skip evaluation render + metrics:
```bash
python main.py building.mp4 --skip-render
```

---

## Viewing Results

### SIBR Real-Time Viewer

The official Gaussian Splatting viewer (bundled with the repo):

```bash
./gaussian-splatting/SIBR_viewers/bin/SIBR_gaussianViewer_app \
    -m gs_pipeline_output/gaussian
```

Controls:
- **Mouse:** Rotate view
- **Scroll:** Zoom
- **Space:** Pause/resume
- **S:** Save screenshot

### Web Viewers (Drag & Drop)

- **https://antimatter15.com/splat/** — Simple, fast splat viewer
- **https://playcanvas.com/viewer** — Alternative viewer

Drag the `.ply` file onto the web page. No upload needed; runs entirely in your browser.

### Convert to `.splat` Format (Web-Friendly)

For embedding in web apps, convert to the optimised binary `.splat` format:

```bash
npm install -g gsplat-tools
gsplat-tools convert \
    gs_pipeline_output/gaussian/point_cloud/iteration_30000/point_cloud.ply \
    gs_pipeline_output/scene.splat
```

The `.splat` file is 10–20× smaller and loads much faster in browsers.

---

## Troubleshooting

### Pre-Flight Checks Fail

**Problem:** `ffmpeg not found in PATH`

**Solution:**
```bash
# macOS (brew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Or download: https://ffmpeg.org/download.html
```

**Problem:** `COLMAP not found in PATH`

**Solution:**
```bash
# macOS (brew)
brew install colmap

# Ubuntu/Debian (from repos)
sudo apt-get install colmap

# Or build from source: https://colmap.github.io/install.html
```

**Problem:** `No NVIDIA GPU detected`

**Solution:**
- Verify GPU with `nvidia-smi`
- If missing, check NVIDIA drivers: `nvidia-smi`
- Install CUDA: https://developer.nvidia.com/cuda-downloads
- Verify CUDA in the conda env: 
  ```bash
  conda activate gaussian_splatting
  python -c "import torch; print(torch.cuda.is_available())"
  ```

### Frame Extraction Problems

**Problem:** Only 5 frames extracted

**Cause:** Scene-change threshold is too high; the video content isn't changing enough to trigger extraction.

**Solution:**
```bash
python main.py building.mp4 --scene-threshold 0.20
```

Gradually lower the threshold until you get 50–200 frames.

**Problem:** 2000+ frames extracted (COLMAP will be very slow)

**Cause:** Scene-change threshold is too low.

**Solution:**
```bash
python main.py building.mp4 --scene-threshold 0.50
```

Raise the threshold. Aim for 50–300 frames depending on video length.

**Problem:** Frames extracted, but all near the start of the video

**Cause:** Camera not moving much, OR video has static intro/outro.

**Solution:** Trim the video before running the pipeline, or edit the frame list manually.

### COLMAP Registration Problems

**Problem:** Only 30% of frames registered

**Cause:** Weak feature matching. Interior scenes have:
- Large flat surfaces (walls, ceilings)
- Repetitive patterns (tiles, windows)
- Variable lighting

**Solutions:**
1. Add a vocab tree for loop closure:
   ```bash
   python main.py building.mp4 --vocab-tree vocab_tree_flickr100K_words256K.bin
   ```

2. Extract more frames (lower `--scene-threshold` in Stage 1):
   ```bash
   python main.py building.mp4 --scene-threshold 0.20
   ```

3. Check the COLMAP text output — look for errors in the mapper logs.

4. Inspect intermediate results:
   ```bash
   colmap gui --database_path gs_pipeline_output/colmap/database.db \
     --import_path gs_pipeline_output/colmap/sparse/0
   ```

**Problem:** Reconstruction looks completely wrong (twisted, inverted, etc.)

**Cause:** Bad camera poses from incorrect focal length or distortion model.

**Solution:**
1. Verify your focal length:
   ```bash
   # For a 4032px-wide Pixel image:
   python -c "print(4032 * 0.529)"  # → ~2133
   python main.py building.mp4 --focal-px 2133
   ```

2. Try a different camera model (though OPENCV is almost always best for phones):
   ```bash
   python main.py building.mp4 --camera-model RADIAL
   ```

3. Check the input video — is it from a Pixel camera? If not, geometry estimates will be wrong.

### Gaussian Splatting Training Problems

**Problem:** `CUDA out of memory` during training

**Cause:** Image resolution is too high for your GPU.

**Solutions:**
- Lower the resolution cap:
  ```bash
  python main.py building.mp4 --resolution-cap 1024
  ```
- Use fewer iterations (trains faster, less VRAM):
  ```bash
  python main.py building.mp4 --iterations 15000
  ```
- Close other CUDA applications (Chrome, other ML processes, etc.)

**Problem:** Training is very slow (1 iteration = 1+ second)

**Cause:** GPU is under-utilised, OR too many Gaussians.

**Solutions:**
- Check GPU usage: `nvidia-smi -l 1` (update every second)
- If GPU memory isn't maxed out, densification might be too aggressive:
  ```bash
  python main.py building.mp4 --densify-grad-threshold 0.0003
  ```
- Check COLMAP registration — if <50% of frames registered, poses may be noisy, slowing convergence

**Problem:** Floaters / noise / "sparkly" Gaussians everywhere

**Cause:** Over-densification, especially near walls where no surface is truly visible at 360°.

**Solutions:**
- Reduce densification aggressiveness:
  ```bash
  python main.py building.mp4 --densify-grad-threshold 0.00015
  ```
- Stop densification earlier:
  ```bash
  python main.py building.mp4 --densify-end 10000
  ```
- Increase iterations to let the pruner clean up:
  ```bash
  python main.py building.mp4 --iterations 50000
  ```

**Problem:** Renders are blurry or poor quality

**Cause:** Insufficient iterations, OR bad COLMAP poses.

**Solutions:**
1. Check COLMAP registration — is it >80%?
   ```bash
   # Look at the stage_colmap output; it reports "X / Y frames registered"
   ```

2. Train longer:
   ```bash
   python main.py building.mp4 --iterations 50000
   ```

3. Ensure test split is reasonable:
   ```bash
   python main.py building.mp4 --test-holdout 10
   # This holds out every 10th frame instead of every 8th, giving more training data
   ```

---

## Advanced Topics

### Camera Geometry — Pixel Sensor

The Blackmagic Camera App **always locks to the Pixel's primary wide lens**. This lens has fixed geometry:

| Parameter | Value |
|-----------|-------|
| Equivalent focal length | ~24 mm |
| Actual focal length | ~3.65 mm |
| Sensor width | ~6.9 mm (1/1.31" on Pixel 8) |
| Focal ratio | 3.65 / 6.9 ≈ **0.529** |

**Auto-estimated focal length (pixels):**
```
focal_px = image_width_px × 0.529
```

Examples:
- 4032 px wide (12 MP, default) → 4032 × 0.529 ≈ **2133 px**
- 4080 px wide (Pixel 9) → 4080 × 0.529 ≈ **2158 px**
- 3840 px wide (4K video crop) → 3840 × 0.529 ≈ **2031 px**
- 1920 px wide (FHD) → 1920 × 0.529 ≈ **1016 px**

**Override with explicit value:**
```bash
# If you know your camera's exact focal length:
python main.py building.mp4 --focal-px 2133
```

COLMAP's bundle adjustment will refine this estimate during reconstruction, so an approximate value is fine.

### Interior Building Tuning

Interior reconstruction is harder than outdoor SfM due to:

1. **Repetitive patterns** — tiles, bricks, windows recur (confusing feature matching)
2. **Low texture** — white walls, ceilings, smooth floors
3. **Variable lighting** — shadows, reflections, flickering light
4. **Short baselines** — camera held close to surfaces
5. **Limited field of view** — can't see 360° around any point

**Best practices:**

1. **Good lighting:**
   - Record in daylight or with consistent artificial light
   - Avoid shadows, backlighting, and reflections
   - Darker rooms (night, basements) are harder

2. **Steady camera motion:**
   - Slow, smooth pans (not jerky or handheld blur)
   - Avoid fast rotations or rapid zoom
   - Try to keep some surfaces in view between frames

3. **Cover the space:**
   - Walk around corners, not just in straight lines
   - Revisit rooms from different angles
   - Look up and down to capture ceiling and floor

4. **Adjust extraction parameters:**
   - Lower `--scene-threshold` (0.20–0.30) to extract more frames in low-texture passages
   - Raise `--min-gap` (0.7–1.0) to avoid near-duplicate bursts during fast pans

5. **Add vocab tree:**
   ```bash
   python main.py building.mp4 --vocab-tree vocab_tree_flickr100K_words256K.bin
   ```
   Significantly improves reconstruction in looping paths (e.g., circular hallway).

6. **Densification tuning:**
   - Stop earlier: `--densify-end 10000`
   - Higher threshold: `--densify-grad-threshold 0.0003`
   - Prevents wall "fuzz" from accumulating

7. **Post-process if needed:**
   - Once you have a PLY, you can clean it up with Meshlab or CloudCompare
   - Remove stray points outside the building boundary
   - Decimate to reduce file size

---

### Project Structure

```
gs_pipeline/
├── main.py                              ← CLI entry point
│   └── parse_args() → PipelineRunner
│
├── config/
│   ├── __init__.py
│   └── settings.py                      ← PipelineConfig dataclass
│       └── All parameters, derived paths, camera geometry
│
├── pipeline/
│   ├── __init__.py
│   │
│   ├── runner.py                        ← PipelineRunner
│   │   ├── preflight() — tool checks, GPU detection
│   │   ├── run() — orchestrate stages in sequence
│   │   └── print_stage_summary()
│   │
│   ├── utils.py                         ← Shared utilities
│   │   ├── log_*() — timestamped, coloured logging
│   │   ├── run() — subprocess runner with dry-run support
│   │   ├── run_in_conda() — activate conda env
│   │   ├── find_conda_sh()
│   │   ├── probe_video() / probe_image() — ffprobe wrappers
│   │   └── COLMAP helpers (count_registered_images, count_ply_points)
│   │
│   ├── stage_extract.py                 ← FrameExtractor
│   │   └── Uses ffmpeg select filter for scene-change detection
│   │
│   ├── stage_colmap.py                  ← ColmapReconstructor
│   │   ├── _feature_extract() — SIFT keypoints
│   │   ├── _sequential_match() — video-order matching
│   │   ├── _vocab_tree_match() — loop closure (optional)
│   │   ├── _mapper() — incremental SfM
│   │   ├── _export_txt() — text model copies
│   │   └── _undistort() — rectify images
│   │
│   └── stage_gaussian.py                ← GaussianTrainer
│       ├── _validate_repo()
│       ├── _train() — invoke train.py in conda env
│       ├── _render() — optional render step
│       ├── _metrics() — optional metrics computation
│       └── _print_summary()
│
├── requirements.txt                     ← Python dependencies (minimal)
└── README.md                            ← This file
```

**Design principles:**

- **Single config object:** `PipelineConfig` holds all parameters; stages never have magic numbers
- **Functional stages:** Each stage is a self-contained class; easy to test or reorder
- **Uniform tool calls:** `run()` utility respects `--dry-run` everywhere
- **Clear logging:** Every print statement includes a timestamp and stage name

---

## Individual Stage Scripts

For reference, standalone bash scripts for each stage are also provided in the outputs. You can use these to understand the underlying commands or run stages manually:

### 1. extract_frames.sh

Standalone ffmpeg frame extraction script.

**Usage:**
```bash
./extract_frames.sh -i building.mp4 [OPTIONS]
```

**Key options:**
```
-i  Input video file
-o  Output directory (default: ./frames)
-s  Scene threshold (default: 0.35)
-m  Min gap seconds (default: 0.5)
-x  Max gap seconds (default: 3.0)
```

**Output:** `frames/frame_000001.jpg` … `frame_NNNNNN.jpg`

---

### 2. run_colmap.sh

Standalone COLMAP SfM reconstruction script.

**Usage:**
```bash
./run_colmap.sh -i frames/ [OPTIONS]
```

**Key options:**
```
-i  Input frames directory
-w  Workspace (default: ./colmap_workspace)
-f  Focal length in pixels (auto-estimated if omitted)
-v  Vocab tree path (optional)
-g  GPU index (default: 0)
```

**Output:**
```
colmap_workspace/sparse/0/          ← cameras.bin, images.bin, points3D.bin
colmap_workspace/dense/             ← undistorted images + camera model
```

---

### 3. run_gaussian_splatting.sh

Standalone 3D Gaussian Splatting training script.

**Usage:**
```bash
./run_gaussian_splatting.sh -s colmap_dense/ [OPTIONS]
```

**Key options:**
```
-s  Source path (COLMAP dense/)
-o  Output directory
-i  Iterations (default: 30000)
-R  Resolution list (default: 1,2,4,8)
-W  Resolution cap pixels (default: 1600)
```

**Output:**
```
gs_output/point_cloud/iteration_30000/point_cloud.ply   ← FINAL SPLAT
```

---

## Examples

### Example 1: Minimal Pipeline

```bash
python main.py my_building.mp4
```

Uses all defaults. Outputs to `gs_pipeline_output/`.

---

### Example 2: Dark Interior with Custom Paths

```bash
python main.py basement.mp4 \
    -o ./basement_splat \
    --scene-threshold 0.25 \
    --densify-grad-threshold 0.00015 \
    --resolution-cap 1024
```

- Lower scene threshold → more frames in dark passages
- More aggressive pruning → fewer floaters
- Smaller resolution → fits in 12GB VRAM

---

### Example 3: Large Open Atrium with Loop Closure

```bash
python main.py big_atrium.mp4 \
    -o ./atrium_splat \
    --gs-repo ~/src/gaussian-splatting \
    --vocab-tree vocab_tree_flickr100K_words256K.bin \
    --colmap-threads 16 \
    --iterations 50000
```

- Vocab tree → catches the loop in the circular balcony
- More threads → faster COLMAP
- More iterations → higher quality final splat

---

### Example 4: Resume from Training Crash

```bash
# Crashed at iteration 28,000? Resume:
python main.py my_building.mp4 --resume

# Or skip to metrics:
python main.py my_building.mp4 --resume --skip-extract --skip-colmap --skip-training
```

---

### Example 5: Dry-Run to Audit Parameters

```bash
python main.py my_building.mp4 \
    --scene-threshold 0.25 \
    --iterations 40000 \
    --dry-run
```

Prints all ffmpeg, COLMAP, and train.py commands without executing. Review, then re-run without `--dry-run`.

---

## Performance Expectations

| Stage | GPU | Time | Notes |
|-------|-----|------|-------|
| **Extract** | CPU | 1–5 min | Depends on video length; fully parallelizable |
| **COLMAP feature** | GPU | 5–30 min | Faster with more features (8192 default) |
| **COLMAP match** | GPU | 5–15 min | Sequential is much faster than exhaustive |
| **COLMAP mapper** | CPU | 30–120 min | Depends on frame count, feature density, baseline |
| **GS training** | GPU | 30–90 min | 30K iterations on RTX 3060; more iters = more time |
| **Render + metrics** | GPU | 5–10 min | Optional; renders test views and computes PSNR/SSIM |
| **Total** | — | 1.5–4 hours | Typical for a 5–10 min video of interior space |

**Speedups:**
- Use a faster GPU (RTX 4090 ~2–3× faster than RTX 3060)
- Lower resolution cap (1024 px vs 1600 px)
- Reduce iterations (15000 for draft, 50000 for final)
- More COLMAP threads (if many CPU cores available)

---

## Citation & References

This pipeline orchestrates:

1. **ffmpeg** — Video processing
   - https://ffmpeg.org

2. **COLMAP** — Structure-from-Motion
   - https://colmap.github.io
   - Paper: Schönberger & Frahm, "Structure-from-Motion Revisited" (CVPR 2016)

3. **3D Gaussian Splatting** — Neural radiance training
   - https://github.com/graphdeco-inria/gaussian-splatting
   - Paper: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)

If you publish results using this pipeline, please cite the original authors.

---

## License & Support

This orchestration pipeline is provided as-is. The underlying tools (ffmpeg, COLMAP, gaussian-splatting) have their own licenses.

**For issues:**
- Check the [Troubleshooting](#troubleshooting) section
- Run with `--dry-run` to audit parameters
- Review tool-specific docs (COLMAP GUI, GS training logs)

---

## Changelog

### v1.0 (Initial Release)

- Full pipeline: extract → COLMAP → Gaussian Splatting
- Resume capability
- Per-stage dry-run and skip flags
- Optimised for Google Pixel + Blackmagic Camera App
- Comprehensive README and inline documentation

---

**Happy splatting! 🎯✨**
