# gs_pipeline

**Video → 3D Gaussian Splatting**, end-to-end.

Optimised for Google Pixel + Blackmagic Camera App interior building footage.

```
building.mp4
    │
    ▼  Stage 1: ffmpeg scene-change extraction
frames/frame_000001.jpg  …  frame_000312.jpg
    │
    ▼  Stage 2: COLMAP sparse reconstruction
colmap/sparse/0/cameras.bin  images.bin  points3D.bin
colmap/dense/images/  sparse/          ← 3DGS source path
    │
    ▼  Stage 3: 3D Gaussian Splatting training
gaussian/point_cloud/iteration_30000/point_cloud.ply
```

---

## Prerequisites

### System tools
| Tool | Version | Install |
|------|---------|---------|
| ffmpeg + ffprobe | ≥ 5.0 | https://ffmpeg.org/download.html |
| COLMAP | ≥ 3.8 | Build locally; requires Boost, Eigen3, OpenImageIO dev/tools, SQLite3, libsuitesparse dev, Ceres dev, CGAL dev, Qt6 SVG dev, GLEW dev, and METIS dev packages |
| CUDA GPU | any | Required for default `--gs-backend cuda` |
| ROCm GPU stack | any | Required for `--gs-backend rocm` |
| Miniconda | any | https://docs.conda.io/en/latest/miniconda.html |

### CUDA gaussian-splatting repo
```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
cd gaussian-splatting
conda env create -f environment.yml
conda activate gaussian_splatting
# Verify CUDA extensions compiled:
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('OK')"
```

### ROCm backend notes

- Use `--gs-backend rocm` to switch Stage 3 backend selection.
- For ROCm mode, `--gs-repo` should point to a GSplat-compatible checkout (for example `ROCm/gsplat`) rather than the Graphdeco CUDA repository.
- If you keep a local `./rocm-gsplat` checkout in the project root, the CLI will prefer it automatically for `--gs-backend rocm`.
- Recommended local workflow on this machine:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.5.1+rocm6.2 torchvision==0.20.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --index-url https://download.pytorch.org/whl/rocm6.2
python main.py building.mp4 --gs-backend rocm --env-runner uv --uv-python .venv/bin/python --gs-repo ./rocm-gsplat
```

- Verify ROCm PyTorch and gsplat in your environment:

```bash
.venv/bin/python -c "import torch; print(torch.version.hip)"
.venv/bin/python -c "import gsplat; print(gsplat.__version__)"
```

- Non-goal: this pipeline does not implement a remote CUDA backend.

### Local COLMAP build

If COLMAP is not already on `PATH`, build it locally and add the install prefix
to `PATH` before running the pipeline:

Required build dependencies:
- Boost development files
- Eigen3 development files
- OpenImageIO development files
- OpenImageIO tools package (provides `iconvert`)
- SQLite3 development files
- SuiteSparse / CHOLMOD development files (`libsuitesparse-dev`)
- Ceres development files (`libceres-dev`)
- CGAL development files (`libcgal-dev`)
- Qt6 development files (`qt6-base-dev`, `qt6-declarative-dev`)
- Qt6 SVG development files (`qt6-svg-dev`)
- GLEW development files (`libglew-dev`)
- METIS development files (`libmetis-dev`)

Before COLMAP can run, build and install ONNX Runtime `v1.24.4` from source so
that `libonnxruntime.so.1` is available in the same prefix or on the dynamic
linker path:

```bash
git clone --recursive --branch v1.24.4 https://github.com/microsoft/onnxruntime.git
cmake -S onnxruntime/cmake -B onnxruntime/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build onnxruntime/build -j
cmake --install onnxruntime/build
```

```bash
git clone --recursive https://github.com/colmap/colmap.git
cmake -S colmap -B colmap/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build colmap/build -j
cmake --install colmap/build

export PATH="$HOME/.local/bin:$PATH"
colmap --version
```

### Vocab tree (optional — improves loop closure in revisited rooms)
```bash
# Download the 256K-word tree (~800 MB):
wget https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin
# Pass with: --vocab-tree vocab_tree_flickr100K_words256K.bin
```

---

## Quick start

```bash
# Clone / copy this project
cd gs_pipeline

# Run the full pipeline (CUDA default assumes ./gaussian-splatting)
python main.py building.mp4

# Specify the repo location
python main.py building.mp4 -o ./output --gs-repo ~/gaussian-splatting

# CUDA backend (default) via conda
python main.py building.mp4 --gs-backend cuda --env-runner conda --conda-env gaussian_splatting

# ROCm backend via conda
python main.py building.mp4 --gs-backend rocm --env-runner conda --rocm-env gaussian_splatting_rocm

# ROCm backend via uv (explicit interpreter)
python main.py building.mp4 --gs-backend rocm --gs-repo ~/rocm-gsplat --env-runner uv --uv-python .venv/bin/python

# With vocab tree for better loop closure
python main.py building.mp4 --vocab-tree vocab_tree_flickr100K_words256K.bin
```

---

## All options

```
usage: gs_pipeline [-h] [-o OUTPUT] [--gs-repo GS_REPO] [--conda-env CONDA_ENV]
                   [--dry-run] [--resume]
                   [--skip-extract] [--skip-colmap] [--skip-training] [--skip-render]
                   [--scene-threshold T] [--min-gap S] [--max-gap S] [--frame-quality Q]
                   [--camera-model {SIMPLE_RADIAL,RADIAL,OPENCV,FULL_OPENCV}]
                   [--focal-px F] [--vocab-tree PATH] [--colmap-gpu IDX] [--colmap-threads N]
                   [--iterations N] [--densify-start N] [--densify-end N]
                   [--densify-grad-threshold T] [--resolution-cap PX]
                   [--test-holdout N] [--checkpoint-interval N] [--viewer-port PORT]
                   video
```

### Frame extraction

| Flag | Default | Notes |
|------|---------|-------|
| `--scene-threshold` | `0.35` | ffmpeg scene score (0–1). **Lower → more frames**. Raise if you have too many near-duplicates. |
| `--min-gap` | `0.5s` | Minimum seconds between any two extracted frames. Prevents burst extraction during fast pans. |
| `--max-gap` | `3.0s` | Force a frame if this much time passes even with low scene score. Prevents gaps during slow crawls. |
| `--frame-quality` | `2` | JPEG qscale (1=best, 31=worst). |

### COLMAP

| Flag | Default | Notes |
|------|---------|-------|
| `--camera-model` | `OPENCV` | `OPENCV` models radial + tangential distortion (k1,k2,p1,p2) — correct for phone lenses. |
| `--focal-px` | auto | Focal length in pixels. Auto-estimated from Pixel sensor geometry if omitted (`width × 0.529`). |
| `--vocab-tree` | none | Path to a vocab tree binary. Enables loop-closure detection for revisited rooms. |
| `--colmap-gpu` | `0` | GPU index (-1 = CPU). |
| `--colmap-threads` | all | CPU threads for COLMAP. |

### Gaussian Splatting

| Flag | Default | Notes |
|------|---------|-------|
| `--iterations` | `30000` | Total training steps. Raise to 50000 for higher quality. |
| `--densify-start` | `500` | Iteration to begin Gaussian densification. |
| `--densify-end` | `15000` | Iteration to stop densification. Set to half of `--iterations` to avoid over-densifying flat surfaces. |
| `--densify-grad-threshold` | `0.0002` | Gradient threshold for splitting Gaussians. Lower = more Gaussians (slower, higher quality). |
| `--resolution-cap` | `1600` | Cap longest image side (pixels). Guards VRAM — Pixel 4K frames use ~16GB without this. |
| `--test-holdout` | `8` | Hold out every Nth frame for PSNR/SSIM evaluation. |
| `--checkpoint-interval` | `7000` | Save a checkpoint PLY every N iterations (0 = final only). |
| `--viewer-port` | `6009` | Port for the live SIBR training monitor. |

---

## Output structure

```
gs_pipeline_output/
├── frames/
│   ├── frame_000001.jpg          ← scene-change extracted frames
│   └── frame_000NNN.jpg
├── colmap/
│   ├── database.db               ← COLMAP feature/match database
│   ├── sparse/
│   │   └── 0/
│   │       ├── cameras.bin       ← camera intrinsics + extrinsics
│   │       ├── images.bin        ← per-image pose
│   │       ├── points3D.bin      ← sparse point cloud
│   │       └── txt/              ← human-readable copies
│   └── dense/                    ← 3DGS source path
│       ├── images/               ← undistorted frames
│       └── sparse/               ← camera model copy
└── gaussian/
    ├── point_cloud/
    │   ├── iteration_7000/
    │   │   └── point_cloud.ply   ← checkpoint splat
    │   └── iteration_30000/
    │       └── point_cloud.ply   ← final splat  ← THIS IS YOUR OUTPUT
    ├── test/                     ← rendered eval images
    ├── results.json              ← PSNR / SSIM / LPIPS metrics
    ├── cameras.json
    ├── input.ply
    └── cfg_args
```

---

## Resuming an interrupted run

```bash
# Resume from wherever it stopped — skips stages whose outputs already exist
python main.py building.mp4 --resume
```

## Running individual stages

```bash
# Only extract frames
python main.py building.mp4 --skip-colmap --skip-training --skip-render

# Only run COLMAP (frames already done)
python main.py building.mp4 --skip-extract --skip-training --skip-render

# Only train (COLMAP already done)
python main.py building.mp4 --skip-extract --skip-colmap
```

## Dry run (print commands, don't execute)

```bash
python main.py building.mp4 --dry-run
```

---

## Viewing the result

### SIBR real-time viewer (bundled with gaussian-splatting)
```bash
./gaussian-splatting/SIBR_viewers/bin/SIBR_gaussianViewer_app \
    -m gs_pipeline_output/gaussian
```

### Web viewers (drag & drop the PLY)
- https://antimatter15.com/splat/
- https://playcanvas.com/viewer

### Convert to `.splat` (smaller, web-friendly)
```bash
npx gsplat-tools convert \
    gs_pipeline_output/gaussian/point_cloud/iteration_30000/point_cloud.ply \
    gs_pipeline_output/scene.splat
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No frames extracted | Scene threshold too high | `--scene-threshold 0.20` |
| Too many frames (slow COLMAP) | Threshold too low | `--scene-threshold 0.50` |
| <80% frames registered | Weak feature matches | Add `--vocab-tree`, check lighting |
| CUDA OOM during training | Phone images too large | `--resolution-cap 1024` |
| Floaters / noisy Gaussians | Over-densification | `--densify-grad-threshold 0.00015` |
| Blurry result | Not enough iterations | `--iterations 50000` |
| Conda env not found | Wrong env name or path | `--conda-env <name>` |

---

## Camera geometry — Google Pixel + Blackmagic

The Blackmagic Camera App always locks to the **primary wide lens**:
- Equivalent focal length: ~24 mm
- Actual EFL: ~3.65 mm
- Sensor width: ~6.9 mm (1/1.31" on Pixel 8)
- Focal ratio used: `3.65 / 6.9 ≈ 0.529`
- Auto-estimated focal (px) = `image_width_px × 0.529`

For a 4032-pixel-wide frame: `4032 × 0.529 ≈ 2133 px`

Override with `--focal-px 2133` if you know your exact model.
COLMAP's bundle adjustment will refine this during reconstruction.

---

## Project structure

```
gs_pipeline/
├── main.py                    ← CLI entry point
├── requirements.txt
├── README.md
├── config/
│   ├── __init__.py
│   └── settings.py            ← PipelineConfig dataclass (all parameters)
└── pipeline/
    ├── __init__.py
    ├── runner.py              ← Orchestrator: pre-flight + stage sequencing
    ├── utils.py               ← Logging, command runner, ffprobe, COLMAP helpers
    ├── stage_extract.py       ← Stage 1: ffmpeg frame extraction
    ├── stage_colmap.py        ← Stage 2: COLMAP SfM reconstruction
    └── stage_gaussian.py      ← Stage 3: 3DGS training + eval
```
