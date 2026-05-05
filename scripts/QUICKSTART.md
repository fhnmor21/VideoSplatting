# Bash Scripts Quick Start Guide

Three standalone shell scripts for video → 3D Gaussian Splatting pipeline. Use these if you prefer bash over Python, or want to understand the underlying commands.

---

## Scripts Overview

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `extract_frames.sh` | Extract frames from video using ffmpeg scene-change detection | MP4/MOV video | `frames/frame_*.jpg` |
| `run_colmap.sh` | COLMAP sparse reconstruction (SfM + camera poses) | Frame directory | `colmap/sparse/`, `colmap/dense/` |
| `run_gaussian_splatting.sh` | 3D Gaussian Splatting training | COLMAP dense output | `gs_output/point_cloud/iteration_30000/point_cloud.ply` |

---

## Quick Start (Full Pipeline)

```bash
# 1. Extract frames
./extract_frames.sh -i building.mp4

# 2. Run COLMAP
./run_colmap.sh -i ./frames

# 3. Train 3DGS
./run_gaussian_splatting.sh -s ./colmap_workspace/dense
```

**Done!** Your splat is at `./gs_output/point_cloud/iteration_30000/point_cloud.ply`

---

## Script 1: extract_frames.sh

Smart frame extraction using ffmpeg's scene-change detection.

### Syntax

```bash
./extract_frames.sh -i <video> [OPTIONS]
```

### Options

```
-i  Input video file (REQUIRED)
-o  Output directory (default: ./frames)
-s  Scene threshold, 0.0–1.0 (default: 0.35)
    Lower  → more frames
    Higher → fewer frames
-m  Minimum gap between frames in seconds (default: 0.5)
-x  Maximum gap between frames in seconds (default: 3.0)
-r  Resize output, e.g. 1920x1080 (default: original)
-q  JPEG quality, 1–31 (default: 2, best quality)
-h  Show help
```

### Examples

```bash
# Minimal (uses defaults)
./extract_frames.sh -i building.mp4

# Extract more frames (lower threshold)
./extract_frames.sh -i building.mp4 -s 0.25 -x 1.5

# Extract fewer frames (higher threshold)
./extract_frames.sh -i building.mp4 -s 0.45

# Custom output directory
./extract_frames.sh -i building.mp4 -o ./my_frames

# Downscale to smaller resolution (faster COLMAP)
./extract_frames.sh -i building.mp4 -r 1920x1080

# Combine options
./extract_frames.sh -i building.mp4 \
    -o ./frames \
    -s 0.30 \
    -m 0.4 \
    -x 2.5 \
    -q 2
```

### Output

```
frames/
├── frame_000001.jpg
├── frame_000002.jpg
└── frame_NNNNNN.jpg
```

Number of frames depends on:
- Video length
- Scene threshold (lower = more frames)
- Max gap (lower = more frames in slow passages)

### Tuning Tips

| Goal | Adjustment |
|------|------------|
| Extract more frames | Lower `-s` (0.20) or lower `-x` (1.5) |
| Extract fewer frames | Raise `-s` (0.50) or raise `-x` (4.0) |
| Faster next stages | Raise `-s` or `-r` to smaller resolution |
| Higher quality | Keep `-q 2` or `-q 1` |

---

## Script 2: run_colmap.sh

COLMAP sparse reconstruction: camera poses + 3D points.

### Syntax

```bash
./run_colmap.sh -i <frames_dir> [OPTIONS]
```

### Options

```
-i  Input frames directory (REQUIRED)
-w  Workspace directory (default: ./colmap_workspace)
-c  Camera model (default: OPENCV)
-f  Focal length in pixels (auto-estimated if omitted)
-F  Single camera model (1=yes, 0=no, default: 1)
-g  GPU index for feature extraction/matching, -1=CPU (default: 0)
-t  CPU threads (default: all available)
-v  Vocab tree path for loop-closure detection (optional)
-s  Skip to mapper if database exists (default: 0)
-u  Run image_undistorter (default: 1)
-h  Show help
```

### Examples

```bash
# Minimal (uses defaults, auto-estimates focal length)
./run_colmap.sh -i ./frames

# Specify focal length (for Pixel 8 @ 4032px)
./run_colmap.sh -i ./frames -f 2133

# With vocab tree (better loop closure)
./run_colmap.sh -i ./frames -v ~/vocab_tree_flickr100K_words256K.bin

# More CPU threads (faster)
./run_colmap.sh -i ./frames -t 16

# Use CPU instead of GPU (slow, but works)
./run_colmap.sh -i ./frames -g -1

# Custom workspace
./run_colmap.sh -i ./frames -w ./my_colmap
```

### Output

```
colmap_workspace/
├── database.db                  ← feature/match database
├── sparse/0/
│   ├── cameras.bin              ← camera intrinsics
│   ├── images.bin               ← camera poses
│   ├── points3D.bin             ← sparse point cloud
│   └── txt/                     ← human-readable copies
└── dense/
    ├── images/                  ← undistorted frames (for 3DGS)
    └── sparse/
```

**For 3DGS training, use the `dense/` directory.**

### Tuning Tips

| Problem | Fix |
|---------|-----|
| <50% frames registered | Add vocab tree `-v`, check lighting, lower scene threshold in Stage 1 |
| Wrong reconstruction | Verify focal length with `-f`, try different camera model `-c RADIAL` |
| COLMAP crashes | Lower CPU threads `-t 4`, check disk space |

### Focal Length Reference (Google Pixel)

Pixel uses a 24mm-equivalent main lens with ~0.529 focal ratio:

```
focal_px = image_width_px × 0.529
```

Common values:
- 4032 px wide → **2133 px**
- 4080 px wide → **2158 px**
- 1920 px wide → **1016 px**

---

## Script 3: run_gaussian_splatting.sh

3D Gaussian Splatting training on COLMAP output.

### Syntax

```bash
./run_gaussian_splatting.sh -s <colmap_dense> [OPTIONS]
```

### Options

```
-s  Source path: COLMAP dense/ directory (REQUIRED)
-o  Output directory (default: ./gs_output)
-r  gaussian-splatting repo path (default: ./gaussian-splatting)
-e  Conda env name (default: gaussian_splatting)
-i  Training iterations (default: 30000)
-d  Densification iterations: start,end (default: 500,15000)
-R  Resolutions to train at (default: 1,2,4,8)
-W  Resolution cap in pixels (default: 1600)
-p  Port for training viewer (default: 6009)
-t  Test holdout every Nth image (default: 8)
-c  Checkpoint interval in iterations (default: 7000)
-C  Resume from checkpoint (optional)
-n  Dry run: print commands, don't execute (0/1, default: 0)
-h  Show help
```

### Examples

```bash
# Minimal (assumes ./gaussian-splatting exists)
./run_gaussian_splatting.sh -s ./colmap_workspace/dense

# Custom GS repo path
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -r ~/src/gaussian-splatting

# More iterations for higher quality
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -i 50000 \
    -d 500,20000

# Dark interior (aggressively prune floaters)
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -i 50000 \
    -d 500,15000 \
    -W 1024

# High-end setup (24GB VRAM)
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -i 50000 \
    -W 2560 \
    -c 3500

# Dry run (preview all commands)
./run_gaussian_splatting.sh -s ./colmap_workspace/dense -n 1

# Resume from checkpoint
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -C ./gs_output/chkpnt_30000.pth
```

### Output

```
gs_output/
├── point_cloud/
│   ├── iteration_7000/point_cloud.ply      ← checkpoint
│   ├── iteration_14000/point_cloud.ply     ← checkpoint
│   └── iteration_30000/point_cloud.ply     ← FINAL (your output!)
├── test/                                    ← rendered views
├── results.json                             ← PSNR/SSIM/LPIPS metrics
└── cameras.json
```

### Live Monitoring

During training, view live progress in the SIBR viewer:

```bash
# In a separate terminal:
./gaussian-splatting/SIBR_viewers/bin/SIBR_gaussianViewer_app \
    -m gs_output
```

Or connect to `http://127.0.0.1:6009` if using a web viewer.

### Tuning Tips

| Problem | Fix |
|---------|-----|
| CUDA out of memory | Lower `-W 1024`, reduce `-i 15000` |
| Floaters/noise | Lower `--densify-grad-threshold` to 0.00015 in script, or edit stage 3 |
| Training too slow | Lower `-i 15000` for draft, increase later for final |
| Blurry result | Increase `-i 50000`, check COLMAP registration |

---

## Common Workflows

### Scenario 1: Quick Draft

Fast iteration for testing:

```bash
# Extract fewer frames
./extract_frames.sh -i building.mp4 -s 0.50

# Quick COLMAP
./run_colmap.sh -i ./frames -t 8

# Draft training (15 min)
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -i 15000 -W 1024
```

### Scenario 2: High-Quality Output

Best quality, takes longer:

```bash
# Extract more frames
./extract_frames.sh -i building.mp4 -s 0.20

# Full COLMAP
./run_colmap.sh -i ./frames -v ~/vocab_tree_flickr100K_words256K.bin

# High-quality training (2 hours)
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -i 50000 \
    -d 500,25000
```

### Scenario 3: Dark Interior

Low-light room walkthrough:

```bash
# Extract with lower threshold
./extract_frames.sh -i basement.mp4 -s 0.25

# COLMAP with more features
# (edit stage_colmap.sh line 120: peak_threshold 0.001 instead of 0.003)
./run_colmap.sh -i ./frames -t 16

# Training with tight densification
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -i 50000 \
    -d 500,12000 \
    -W 1024
```

### Scenario 4: Resume After Crash

Training crashed on iteration 28,000?

```bash
# Option A: Restart training (skips extract + COLMAP)
# Re-run the same command — it will skip completed stages
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -i 50000

# Option B: Resume from checkpoint
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -i 50000 \
    -C ./gs_output/chkpnt_28000.pth
```

---

## Viewing Results

### SIBR Viewer (Real-Time)

```bash
./gaussian-splatting/SIBR_viewers/bin/SIBR_gaussianViewer_app \
    -m gs_output
```

**Controls:**
- Mouse drag: rotate
- Scroll: zoom
- Space: pause/resume
- S: save screenshot

### Web Viewers (Drag & Drop)

1. Go to https://antimatter15.com/splat/
2. Drag `gs_output/point_cloud/iteration_30000/point_cloud.ply` onto the page

No upload needed; runs in your browser.

### Convert to .splat (Web-Friendly)

```bash
npm install -g gsplat-tools
gsplat-tools convert \
    gs_output/point_cloud/iteration_30000/point_cloud.ply \
    gs_output/scene.splat
```

The `.splat` file is 10–20× smaller and loads much faster in browsers.

---

## Prerequisites

### Required Tools

```bash
# Check if installed:
ffmpeg -version
colmap -h
nvidia-smi

# Install (macOS):
brew install ffmpeg colmap

# Install (Ubuntu/Debian):
sudo apt-get install ffmpeg colmap

# Install CUDA:
https://developer.nvidia.com/cuda-downloads
```

### Gaussian Splatting Repo

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
cd gaussian-splatting
conda env create -f environment.yml
conda activate gaussian_splatting
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('OK')"
```

### Vocab Tree (Optional)

```bash
wget https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin
# Pass to run_colmap.sh with -v
```

---

## Troubleshooting

### No frames extracted

**Cause:** Scene threshold too high.

**Fix:**
```bash
./extract_frames.sh -i building.mp4 -s 0.20
```

Gradually lower threshold until you get 50–200 frames.

---

### <50% COLMAP registration

**Cause:** Weak feature matching (low lighting, repetitive textures).

**Fix:**
```bash
# Add vocab tree
./run_colmap.sh -i ./frames -v ~/vocab_tree_flickr100K_words256K.bin

# Or extract more frames
./extract_frames.sh -i building.mp4 -s 0.20
./run_colmap.sh -i ./frames
```

---

### CUDA out of memory during training

**Cause:** Images too large for GPU.

**Fix:**
```bash
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -W 1024
```

Reduce resolution cap or iterations.

---

### Training too slow

**Cause:** Resolution too high or too many iterations.

**Fix:**
```bash
# Draft with fewer iterations
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -i 15000 -W 1024
```

---

### Floaters / noisy Gaussians

**Cause:** Over-densification.

**Fix:**

Edit `run_gaussian_splatting.sh` line ~180, change:
```bash
--Mapper.densify_grad_threshold 0.0002    # lower = more Gaussians
```

to:
```bash
--Mapper.densify_grad_threshold 0.0003    # higher = fewer, cleaner Gaussians
```

Then re-run training.

---

## Performance Expectations

| Stage | Time | Notes |
|-------|------|-------|
| Extract frames | 1–5 min | Depends on video length |
| COLMAP feature | 5–30 min | GPU-accelerated |
| COLMAP match | 5–15 min | Sequential is fast |
| COLMAP mapper | 30–120 min | Depends on frame count |
| GS training (30K iters) | 30–90 min | RTX 3060 is typical |
| **Total** | 1.5–4 hours | Typical for 5–10 min video |

**Speedup:** RTX 4090 is ~2–3× faster than RTX 3060.

---

## Example: Full Pipeline with Tuning

```bash
# Start with a 5-minute building walkthrough video

# 1. Extract frames (target: ~100–150 frames)
./extract_frames.sh -i building.mp4 -s 0.30 -o ./my_frames

# 2. Check frame count
ls ./my_frames | wc -l  # Should be 50–200

# 3. COLMAP reconstruction
./run_colmap.sh -i ./my_frames -f 2133 -v ~/vocab_tree.bin

# 4. Check registration
# Look at COLMAP output — should show "Registered X / Y images"
# Target: >80%

# 5. Train 3DGS
./run_gaussian_splatting.sh -s ./colmap_workspace/dense \
    -i 40000 \
    -d 500,16000 \
    -W 1600

# 6. View result
./gaussian-splatting/SIBR_viewers/bin/SIBR_gaussianViewer_app \
    -m ./gs_output
```

---

## Camera Geometry (Google Pixel)

The Blackmagic Camera App locks to Pixel's main wide lens:

- Equivalent focal length: ~24 mm
- Focal ratio: **0.529**
- Auto-focal (pixels) = `image_width × 0.529`

Examples:
- 4032 px wide → **2133 px**
- 3840 px wide (4K) → **2031 px**
- 1920 px wide (FHD) → **1016 px**

Override with `-f` in `run_colmap.sh`:

```bash
./run_colmap.sh -i ./frames -f 2133
```

---

## Tips & Best Practices

1. **Good lighting matters:** Dark interiors are harder; use daylight or consistent artificial light.

2. **Steady camera motion:** Smooth pans, not jerky handheld footage.

3. **Cover the space:** Walk around corners; revisit rooms from different angles.

4. **Dry-run before committing:** Use `-n 1` flag in `run_gaussian_splatting.sh` to preview commands.

5. **Monitor COLMAP registration:** If <80% of frames register, loop closure (vocab tree) usually helps.

6. **Start with draft settings:** Quick iteration with lower thresholds, then increase quality.

---

## References

- **ffmpeg:** https://ffmpeg.org
- **COLMAP:** https://colmap.github.io
- **3D Gaussian Splatting:** https://github.com/graphdeco-inria/gaussian-splatting

---

**Happy splatting!** 🎯✨
