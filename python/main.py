#!/usr/bin/env python3
"""
gs_pipeline — Video to 3D Gaussian Splatting, end-to-end.

Optimised for Google Pixel + Blackmagic Camera App interior building footage.

Usage:
    python main.py <video> [options]
    python main.py --help
"""

import argparse
import sys
import time
from pathlib import Path

from pipeline.runner import PipelineRunner
from config.settings import PipelineConfig


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pipeline execution and tuning controls."""
    p = argparse.ArgumentParser(
        prog="gs_pipeline",
        description="Video → Frames → COLMAP → 3D Gaussian Splatting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Required
    # ------------------------------------------------------------------ #
    p.add_argument("video", type=Path, help="Input video file")

    # ------------------------------------------------------------------ #
    # General
    # ------------------------------------------------------------------ #
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./gs_pipeline_output"),
        help="Root output directory (frames/, colmap/, gaussian/ placed inside)",
    )
    p.add_argument(
        "--gs-repo",
        type=Path,
        default=Path("./gaussian-splatting"),
        help="Path to the cloned gaussian-splatting repository",
    )
    p.add_argument(
        "--conda-env",
        default="gaussian_splatting",
        help="Conda environment name that has the GS dependencies installed",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every command without executing anything",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages whose outputs already exist (resume interrupted run)",
    )

    # ------------------------------------------------------------------ #
    # Stage selection
    # ------------------------------------------------------------------ #
    stages = p.add_argument_group("Stage selection (all enabled by default)")
    stages.add_argument(
        "--skip-extract", action="store_true", help="Skip frame extraction"
    )
    stages.add_argument(
        "--skip-colmap", action="store_true", help="Skip COLMAP reconstruction"
    )
    stages.add_argument("--skip-training", action="store_true", help="Skip GS training")
    stages.add_argument(
        "--skip-render", action="store_true", help="Skip eval render + metrics"
    )

    # ------------------------------------------------------------------ #
    # Frame extraction
    # ------------------------------------------------------------------ #
    fe = p.add_argument_group("Frame extraction")
    fe.add_argument(
        "--scene-threshold",
        type=float,
        default=0.35,
        metavar="T",
        help="ffmpeg scene-change score to trigger frame save (0–1). "
        "Lower = more frames.",
    )
    fe.add_argument(
        "--min-gap",
        type=float,
        default=0.5,
        metavar="S",
        help="Minimum seconds between any two extracted frames",
    )
    fe.add_argument(
        "--max-gap",
        type=float,
        default=3.0,
        metavar="S",
        help="Maximum seconds before a frame is forced even with low scene score",
    )
    fe.add_argument(
        "--frame-quality",
        type=int,
        default=2,
        metavar="Q",
        help="JPEG quality for extracted frames (1=best, 31=worst)",
    )

    # ------------------------------------------------------------------ #
    # COLMAP
    # ------------------------------------------------------------------ #
    cm = p.add_argument_group("COLMAP reconstruction")
    cm.add_argument(
        "--camera-model",
        default="OPENCV",
        choices=["SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"],
        help="COLMAP camera model. OPENCV (k1,k2,p1,p2) is best for phone lenses.",
    )
    cm.add_argument(
        "--focal-px",
        type=float,
        default=None,
        metavar="F",
        help="Known focal length in pixels. Auto-estimated from Pixel sensor "
        "geometry if omitted.",
    )
    cm.add_argument(
        "--vocab-tree",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a COLMAP vocab tree binary for loop-closure detection. "
        "Download: https://demuc.de/colmap/#dataset",
    )
    cm.add_argument(
        "--colmap-gpu",
        type=int,
        default=0,
        metavar="IDX",
        help="GPU index for COLMAP feature extraction/matching (-1 = CPU)",
    )
    cm.add_argument(
        "--colmap-threads",
        type=int,
        default=None,
        metavar="N",
        help="CPU threads for COLMAP (default: all available)",
    )

    # ------------------------------------------------------------------ #
    # Gaussian Splatting
    # ------------------------------------------------------------------ #
    gs = p.add_argument_group("Gaussian Splatting training")
    gs.add_argument(
        "--iterations",
        type=int,
        default=30000,
        metavar="N",
        help="Total training iterations",
    )
    gs.add_argument(
        "--densify-start",
        type=int,
        default=500,
        metavar="N",
        help="Iteration to start Gaussian densification",
    )
    gs.add_argument(
        "--densify-end",
        type=int,
        default=15000,
        metavar="N",
        help="Iteration to stop Gaussian densification",
    )
    gs.add_argument(
        "--densify-grad-threshold",
        type=float,
        default=0.0002,
        metavar="T",
        help="Gradient threshold for densification. Lower = more Gaussians.",
    )
    gs.add_argument(
        "--resolution-cap",
        type=int,
        default=1600,
        metavar="PX",
        help="Cap longest image side to this many pixels before training "
        "(VRAM guard for large phone images)",
    )
    gs.add_argument(
        "--test-holdout",
        type=int,
        default=8,
        metavar="N",
        help="Hold out every Nth image as a test/eval split",
    )
    gs.add_argument(
        "--checkpoint-interval",
        type=int,
        default=7000,
        metavar="N",
        help="Save a checkpoint every N iterations (0 = only at end)",
    )
    gs.add_argument(
        "--viewer-port",
        type=int,
        default=6009,
        metavar="PORT",
        help="Port for the live training viewer (0 = disable)",
    )

    return p.parse_args()


def main() -> int:
    """Build pipeline config from CLI arguments and execute the pipeline."""
    args = parse_args()

    # Resolve all paths to absolutes up front
    video = args.video.resolve()
    if not video.exists():
        print(f"ERROR: Video file not found: {video}", file=sys.stderr)
        return 1

    output_root = args.output.resolve()
    gs_repo = args.gs_repo.resolve()
    vocab_tree = args.vocab_tree.resolve() if args.vocab_tree else None

    config = PipelineConfig(
        video=video,
        output_root=output_root,
        gs_repo=gs_repo,
        conda_env=args.conda_env,
        dry_run=args.dry_run,
        resume=args.resume,
        # Stage flags
        run_extract=not args.skip_extract,
        run_colmap=not args.skip_colmap,
        run_training=not args.skip_training,
        run_render=not args.skip_render,
        # Frame extraction
        scene_threshold=args.scene_threshold,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        frame_quality=args.frame_quality,
        # COLMAP
        camera_model=args.camera_model,
        focal_px=args.focal_px,
        vocab_tree=vocab_tree,
        colmap_gpu=args.colmap_gpu,
        colmap_threads=args.colmap_threads,
        # Gaussian Splatting
        iterations=args.iterations,
        densify_start=args.densify_start,
        densify_end=args.densify_end,
        densify_grad_threshold=args.densify_grad_threshold,
        resolution_cap=args.resolution_cap,
        test_holdout=args.test_holdout,
        checkpoint_interval=args.checkpoint_interval,
        viewer_port=args.viewer_port,
    )

    runner = PipelineRunner(config)
    t0 = time.time()
    success = runner.run()
    elapsed = time.time() - t0

    mins, secs = divmod(int(elapsed), 60)
    hours, mins = divmod(mins, 60)
    duration = f"{hours}h {mins}m {secs}s" if hours else f"{mins}m {secs}s"

    if success:
        print(f"\n✓ Pipeline finished in {duration}")
        return 0
    else:
        print(f"\n✗ Pipeline failed after {duration}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
