# VideoSplatting Python Implementation Specification

## Purpose

This Python application implements an end-to-end video-to-3D Gaussian Splatting pipeline tuned for handheld interior captures (Google Pixel + Blackmagic Camera app). It orchestrates three stages:

1. Frame extraction with `ffmpeg`
2. Sparse reconstruction and undistortion with `COLMAP`
3. Training and evaluation with the upstream Inria `gaussian-splatting` repository

The pipeline is designed to be resumable, configurable from CLI flags, and safe to dry-run.

## Runtime Entry Point

- **File:** `python/main.py`
- **Primary responsibilities:**
  - Parse CLI arguments (`parse_args`)
  - Resolve filesystem paths and validate input video existence
  - Build a single `PipelineConfig` object
  - Execute the pipeline through `PipelineRunner`
  - Return POSIX-style exit code (`0` success, `1` failure)

## Core Architecture

The implementation follows an orchestrator + staged worker structure:

- `PipelineConfig` (`python/settings.py`) is the central typed configuration and path derivation layer.
- `PipelineRunner` (`python/runner.py`) performs preflight checks and sequential stage execution.
- Stage classes encapsulate concrete work:
  - `FrameExtractor` (`python/stage_extract.py`)
  - `ColmapReconstructor` (`python/stage_colmap.py`)
  - `GaussianTrainer` (`python/stage_gaussian.py`)
- Shared command/log/metadata helpers are in `python/utils.py`.

Each stage exposes a `run() -> bool` contract. The runner aborts on the first hard failure and prints a stage summary table.

## Configuration Model

- **File:** `python/settings.py`
- **Type:** `@dataclass PipelineConfig`

`PipelineConfig` stores:

- Top-level execution options (`video`, `output_root`, `gs_repo`, `dry_run`, `resume`)
- Stage switches (`run_extract`, `run_colmap`, `run_training`, `run_render`)
- Stage-specific tuning values for extraction, COLMAP, and training
- Derived properties for all important output/script paths

Notable derived helpers:

- `best_sparse_model`: picks the COLMAP sub-model with the largest `images.bin`
- `estimate_focal_px`: computes focal prior using Pixel geometry ratio (`0.529`)
- `opencv_camera_params`: generates COLMAP OPENCV intrinsics/distortion initializer
- `ensure_dirs`: creates all required output directories before execution

## Stage Specifications

### Stage 1: Frame Extraction

- **File:** `python/stage_extract.py`
- **Class:** `FrameExtractor`
- **External tools:** `ffmpeg`, `ffprobe`

Behavior:

- Validates `ffmpeg` availability
- Supports resume skip when `frame_*.jpg` already exists
- Probes source video metadata for logging (`duration`, `fps`, `resolution`)
- Uses a select-expression that keeps frames when:
  - scene score exceeds threshold and minimum temporal spacing is respected, or
  - maximum spacing threshold is reached (safety net)
- Emits advisory warnings when extracted cadence suggests sparse coverage or over-sampling

### Stage 2: COLMAP Reconstruction

- **File:** `python/stage_colmap.py`
- **Class:** `ColmapReconstructor`
- **External tool:** `colmap`

Sub-steps:

1. `feature_extractor` (SIFT tuned for low-texture interiors)
2. `sequential_matcher` (video-order matching with overlap windows)
3. Optional `vocab_tree_matcher` (loop closure, non-fatal on failure)
4. `mapper` (incremental SfM)
5. Optional `model_converter` to TXT (non-fatal on failure)
6. `image_undistorter` to produce dense COLMAP layout for 3DGS

Behavioral details:

- Auto-computes camera priors from first frame dimensions when possible
- Supports resume behavior for features, sparse model, and undistorted image outputs
- Validates reconstruction quality by reporting registered image count

### Stage 3: Gaussian Splatting Training

- **File:** `python/stage_gaussian.py`
- **Class:** `GaussianTrainer`
- **External dependency:** upstream `gaussian-splatting` repo and conda environment

Behavior:

- Validates repository/script presence (`train.py`, optional `render.py`, `metrics.py`)
- Locates `conda.sh` and executes commands in the configured conda environment when available
- Validates COLMAP dense input structure (`images/` + `sparse/`)
- Builds `train.py` command line from config values
- Supports checkpoint interval expansion and final iteration inclusion
- Supports resume skip when final PLY already exists
- Optionally runs render and metrics pass, then prints result summary and viewer hints

## Utilities and Process Infrastructure

- **File:** `python/utils.py`

Provides:

- Structured, timestamped console logging
- Command execution wrappers:
  - `run(...)` for regular subprocess calls
  - `run_in_conda(...)` for environment-activated execution
- Metadata probing helpers:
  - `probe_video(...)`
  - `probe_image(...)`
- Reconstruction/output helpers:
  - `count_registered_images(...)`
  - `count_ply_points(...)`
- Tool discovery helpers (`check_tool`, `find_conda_sh`)

Error policy:

- Hard failures raise or return `False` at stage level
- Select non-critical operations are explicitly non-fatal (e.g., vocab-tree pass, TXT export, eval rendering/metrics)

## Data and Output Contract

Given an input video, the pipeline writes under `output_root`:

- `frames/`: extracted JPEG keyframes
- `colmap/database.db`: features/matches database
- `colmap/sparse/<id>/`: sparse model(s)
- `colmap/dense/`: undistorted images and sparse model copy for training
- `gaussian/point_cloud/iteration_<N>/point_cloud.ply`: checkpoint/final output
- Optional evaluation artifacts (`gaussian/test/`, `gaussian/results.json`)

## Execution Controls

CLI controls support:

- Stage skipping (`--skip-extract`, `--skip-colmap`, `--skip-training`, `--skip-render`)
- Resuming partial runs (`--resume`)
- No-op command preview (`--dry-run`)
- Fine-grained tuning for extraction cadence, COLMAP behavior, and 3DGS optimization

## Extensibility Notes

The implementation is designed for straightforward extension:

- Add new stages by implementing a class with `run() -> bool` and wiring it in `PipelineRunner`
- Add new config values in `PipelineConfig` and surface them in `main.py` arguments
- Reuse `utils.run(...)` and logging helpers for consistent command visibility and failure handling

## API Reference

### `python/main.py`

- `parse_args() -> argparse.Namespace`: Parses CLI flags for stage selection, tool paths, and tuning parameters.
- `main() -> int`: Validates input, builds `PipelineConfig`, runs `PipelineRunner`, and returns process exit code.

### `python/settings.py`

- `class PipelineConfig`: Central dataclass carrying runtime options, stage flags, tuning values, and derived paths.
- `PipelineConfig.frames_dir -> Path`: Output directory for extracted keyframes.
- `PipelineConfig.colmap_workspace -> Path`: Root directory for all COLMAP artifacts.
- `PipelineConfig.colmap_database -> Path`: Path to COLMAP SQLite database.
- `PipelineConfig.colmap_sparse -> Path`: Directory containing sparse model subdirectories.
- `PipelineConfig.colmap_dense -> Path`: Undistorted COLMAP output used as 3DGS source.
- `PipelineConfig.gs_output -> Path`: Root directory for gaussian-splatting outputs.
- `PipelineConfig.best_sparse_model -> Optional[Path]`: Selects most complete sparse model candidate.
- `PipelineConfig.cpu_threads -> int`: Returns configured thread count or system default.
- `PipelineConfig.estimate_focal_px(image_width: int) -> float`: Estimates focal length in pixels from Pixel lens ratio.
- `PipelineConfig.opencv_camera_params(image_width: int, image_height: int) -> str`: Builds COLMAP OPENCV camera parameter string.
- `PipelineConfig.train_script -> Path`: Path to upstream `train.py` script.
- `PipelineConfig.render_script -> Path`: Path to upstream `render.py` script.
- `PipelineConfig.metrics_script -> Path`: Path to upstream `metrics.py` script.
- `PipelineConfig.final_ply -> Path`: Expected path to final PLY at configured iteration.
- `PipelineConfig.ensure_dirs() -> None`: Creates required output directories before execution.

### `python/runner.py`

- `class StageResult`: Dataclass for stage name, success/skipped flags, elapsed time, and optional error text.
- `class PipelineRunner`: Stage orchestrator handling preflight, stage execution, and summary reporting.
- `PipelineRunner.run() -> bool`: Executes enabled stages in order and aborts on first hard failure.
- `PipelineRunner._run_extract() -> bool`: Runs frame extraction stage adapter.
- `PipelineRunner._run_colmap() -> bool`: Runs COLMAP stage adapter.
- `PipelineRunner._run_gaussian() -> bool`: Runs Gaussian training stage adapter.
- `PipelineRunner._preflight() -> bool`: Verifies required tools, GPU visibility, and repository prerequisites.
- `PipelineRunner._detect_gpu() -> Optional[str]`: Queries NVIDIA GPU name and memory via `nvidia-smi`.
- `PipelineRunner._print_banner() -> None`: Prints startup configuration banner.
- `PipelineRunner._print_stage_summary() -> None`: Prints stage status and timing table.

### `python/stage_extract.py`

- `class FrameExtractor`: Implements Stage 1 extraction from source video to keyframe images.
- `FrameExtractor.run() -> bool`: Runs extraction flow with resume behavior and output validation.
- `FrameExtractor._build_command(frames_dir: Path) -> list`: Builds the `ffmpeg` select-filter command for adaptive frame capture.
- `FrameExtractor._print_video_info(info: dict) -> None`: Logs source metadata and active extraction parameters.

### `python/stage_colmap.py`

- `class ColmapReconstructor`: Implements Stage 2 sparse reconstruction and undistortion.
- `ColmapReconstructor.run() -> bool`: Runs full COLMAP pipeline and validates model availability.
- `ColmapReconstructor._feature_extract(db: Path, imgs: Path, cam_params: str) -> bool`: Populates feature database with SIFT descriptors.
- `ColmapReconstructor._sequential_match(db: Path) -> bool`: Performs sequential feature matching across nearby frames.
- `ColmapReconstructor._vocab_tree_match(db: Path) -> bool`: Performs optional loop-closure matching using vocab tree.
- `ColmapReconstructor._mapper(db: Path, imgs: Path) -> bool`: Executes incremental SfM mapping.
- `ColmapReconstructor._export_txt(model: Path) -> bool`: Exports sparse model in TXT format (non-critical).
- `ColmapReconstructor._undistort(model: Path, imgs: Path) -> bool`: Produces undistorted image set and matching sparse metadata.

### `python/stage_gaussian.py`

- `class GaussianTrainer`: Implements Stage 3 training and optional evaluation.
- `GaussianTrainer.run() -> bool`: Validates inputs, runs training, then optionally render/metrics.
- `GaussianTrainer._validate_repo() -> bool`: Checks upstream repository and required scripts.
- `GaussianTrainer._train() -> bool`: Launches `train.py` with configured arguments.
- `GaussianTrainer._build_train_cmd() -> List`: Builds training CLI with iterations, densification, and checkpoint settings.
- `GaussianTrainer._render() -> None`: Runs held-out view rendering when available.
- `GaussianTrainer._metrics() -> None`: Runs metrics computation and prints `results.json` values when present.
- `GaussianTrainer._print_summary(image_count: int) -> None`: Prints output locations, stats, and troubleshooting hints.

### `python/utils.py`

- `class CommandError(RuntimeError)`: Raised for non-zero subprocess exits when `check=True`.
- `_ts() -> str`: Generates current-time log prefix text.
- `log_header(title: str) -> None`: Prints section header block.
- `log_info(msg: str) -> None`: Prints informational log line.
- `log_success(msg: str) -> None`: Prints success log line.
- `log_warn(msg: str) -> None`: Prints warning log line to `stderr`.
- `log_error(msg: str) -> None`: Prints error log line to `stderr`.
- `log_cmd(args: Sequence) -> None`: Prints shell-formatted command preview.
- `run(args: Sequence, *, dry_run=False, cwd=None, env=None, check=True, capture=False) -> subprocess.CompletedProcess`: Executes commands with optional capture and dry-run support.
- `run_in_conda(conda_sh: Path, env_name: str, args: Sequence, *, dry_run=False, cwd=None) -> subprocess.CompletedProcess`: Executes command under `conda activate` shell context.
- `find_conda_sh() -> Optional[Path]`: Locates `conda.sh` in common install prefixes.
- `probe_video(video: Path) -> dict`: Returns duration/fps/resolution metadata via `ffprobe`.
- `probe_image(image: Path) -> Tuple[Optional[int], Optional[int]]`: Returns image width/height via `ffprobe`.
- `count_registered_images(model_dir: Path) -> Optional[int]`: Counts registered images from COLMAP model outputs.
- `count_ply_points(ply_path: Path) -> Optional[int]`: Reads vertex count from PLY header.
- `check_tool(name: str, install_hint: str = "") -> bool`: Verifies executable presence in `PATH`.
- `format_duration(seconds: float) -> str`: Formats elapsed seconds as `h m s`, `m s`, or `s`.
