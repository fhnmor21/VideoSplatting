# Quick Guide: Stage Permutations

This guide shows CLI command examples for running specific stage combinations.

Pipeline stages:

- Stage 1: frame extraction
- Stage 2: COLMAP reconstruction
- Stage 3: Gaussian training (and optional render/metrics)

All examples assume you are in the repository root and use:

```bash
./.venv/bin/python python/main.py
```

Replace these placeholders in commands below:

- `<VIDEO>`: input video path (required by CLI even when Stage 1 is skipped)
- `<OUT>`: pipeline output root directory

---

## Full Pipeline (1 + 2 + 3)

```bash
./.venv/bin/python python/main.py <VIDEO> --output <OUT>
```

---

## Stage 1 Only

```bash
./.venv/bin/python python/main.py <VIDEO> --output <OUT> --skip-colmap --skip-training --skip-render
```

---

## Stage 2 Only

```bash
./.venv/bin/python python/main.py <VIDEO> --output <OUT> --skip-extract --skip-training --skip-render
```

Recommended explicit resource limits for Stage 2:

```bash
./.venv/bin/python python/main.py <VIDEO> --output <OUT> --skip-extract --skip-training --skip-render --colmap-gpu 0 --colmap-threads 16
```

Requirements for Stage 2 only:

- Frames must already exist in `<OUT>/frames`
- Frame names should follow `frame_*.jpg`

If your frames are in another folder, point `<OUT>/frames` to it first (example symlink):

```bash
ln -s /path/to/my_frames <OUT>/frames
```

---

## Stage 3 Only

```bash
./.venv/bin/python python/main.py <VIDEO> --output <OUT> --skip-extract --skip-colmap
```

Requirements for Stage 3 only:

- COLMAP dense output must already exist at `<OUT>/colmap/dense`

---

## Stage 1 + 2

```bash
./.venv/bin/python python/main.py <VIDEO> --output <OUT> --skip-training --skip-render
```

---

## Stage 2 + 3

```bash
./.venv/bin/python python/main.py <VIDEO> --output <OUT> --skip-extract
```

Requirements:

- Frames must already exist in `<OUT>/frames`

---

## Stage 1 + 3 (Uncommon)

```bash
./.venv/bin/python python/main.py <VIDEO> --output <OUT> --skip-colmap
```

Requirements:

- Existing valid COLMAP output must already exist in `<OUT>/colmap/dense`

---

## Train Without Render/Metrics

Run all stages but skip render+metrics step at the end:

```bash
./.venv/bin/python python/main.py <VIDEO> --output <OUT> --skip-render
```

---

## Resume an Interrupted Run

```bash
./.venv/bin/python python/main.py <VIDEO> --output <OUT> --resume
```

The pipeline will skip stages whose outputs already exist.

---

## Helpful Stage 2 Options

```bash
--colmap-gpu 0
--colmap-threads 16
--camera-model OPENCV
--vocab-tree /path/to/vocab_tree_flickr100K_words256K.bin
```

---

## All CLI Arguments

### Positional

- `video`: Input video file path.

### General

- `-o`, `--output`: Root output directory (`frames/`, `colmap/`, `gaussian/` are created inside).
- `--gs-repo`: Path to gaussian-splatting/gsplat repository.
- `--conda-env`: Conda environment name used for CUDA backend commands.
- `--env-runner`: Environment runner for Stage 3 (`conda` or `uv`).
- `--uv-python`: Python interpreter path used when `--env-runner uv`.
- `--dry-run`: Print commands only; do not execute.
- `--resume`: Skip stages with existing outputs.

### Stage Selection

- `--skip-extract`: Skip Stage 1 (frame extraction).
- `--skip-colmap`: Skip Stage 2 (COLMAP).
- `--skip-training`: Skip Stage 3 training.
- `--skip-render`: Skip render + metrics evaluation step.

### Frame Extraction (Stage 1)

- `--scene-threshold`: Scene-change score threshold (lower means more frames).
- `--min-gap`: Minimum seconds between extracted frames.
- `--max-gap`: Maximum seconds before forcing a frame.
- `--frame-quality`: JPEG quality value for extracted frames.

### COLMAP (Stage 2)

- `--camera-model`: COLMAP camera model (`SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `FULL_OPENCV`).
- `--focal-px`: Known focal length in pixels (optional; auto-estimated if omitted).
- `--vocab-tree`: Path to COLMAP vocab tree binary (optional loop closure improvement).
- `--colmap-gpu`: GPU index for COLMAP (`0` or higher enables COLMAP GPU operations, `-1` disables GPU and runs CPU-only).
- `--colmap-threads`: Number of CPU threads for COLMAP. If not set, the pipeline now uses a safer default (about half of available CPU cores, minimum 4) instead of all cores.

### Gaussian Splatting (Stage 3)

- `--gs-backend`: Gaussian backend (`cuda` or `rocm`).
- `--rocm-env`: Conda environment name used for ROCm backend commands.
- `--iterations`: Total training iterations.
- `--densify-start`: Iteration where densification starts.
- `--densify-end`: Iteration where densification stops.
- `--densify-grad-threshold`: Gradient threshold controlling densification aggressiveness.
- `--resolution-cap`: Maximum image side length used for training.
- `--test-holdout`: Hold out every Nth image for evaluation.
- `--checkpoint-interval`: Save checkpoint every N iterations (`0` for final-only).
- `--viewer-port`: Viewer port for live training monitor (`0` to disable).
