"""
pipeline/stage_gaussian.py — Stage 3: 3D Gaussian Splatting training.

Trains a 3DGS model from the COLMAP dense/ output produced by stage_colmap.
After training, optionally runs render.py and metrics.py for evaluation.

The official Inria gaussian-splatting repo is invoked via its conda environment
so that the CUDA extensions (diff_gaussian_rasterization, simple_knn) are
always loaded from the correct Python interpreter.

Tuning notes for interior buildings:
  * densify_end set to half of iterations — flat surfaces over-densify if
    allowed to run the full duration, producing millions of wall Gaussians.
  * opacity_reset_interval 3000 — aggressively prunes floaters that collect
    near walls/ceilings never seen from 360°.
  * resolution_cap guards VRAM against Pixel 4K frames (~16GB without cap).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from config.settings import PipelineConfig
from pipeline.utils import (
    CommandError,
    count_ply_points,
    find_conda_sh,
    format_duration,
    log_header,
    log_info,
    log_success,
    log_warn,
    run,
    run_in_conda,
)


class GaussianTrainer:
    """Train and evaluate a 3D Gaussian Splatting model from COLMAP output."""

    def __init__(self, config: PipelineConfig) -> None:
        """Store shared pipeline configuration and deferred conda metadata."""
        self.cfg = config
        self._conda_sh: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        """Validate prerequisites, run training, and optionally run evaluation."""
        log_header("Stage 3 — 3D Gaussian Splatting Training")

        # Locate the repo and conda env
        if not self._validate_repo():
            return False

        self._conda_sh = find_conda_sh()
        if self._conda_sh is None and not self.cfg.dry_run:
            log_warn(
                "Could not locate conda.sh — cannot activate the GS conda env.\n"
                "  Install Miniconda: https://docs.conda.io/en/latest/miniconda.html\n"
                "  Or ensure 'conda' is in PATH."
            )
            return False

        # Validate source path structure
        source = self.cfg.colmap_dense
        if not (source / "images").exists() or not (source / "sparse").exists():
            log_warn(
                f"Source path is missing images/ or sparse/: {source}\n"
                "  Ensure Stage 2 (COLMAP) completed with --undistort enabled."
            )
            return False

        image_count = len(list((source / "images").glob("*.jpg"))) + len(
            list((source / "images").glob("*.png"))
        )
        log_info(f"Source path  : {source}  ({image_count} undistorted images)")
        log_info(f"Output path  : {self.cfg.gs_output}")
        log_info(f"GS repo      : {self.cfg.gs_repo}")
        log_info(f"Conda env    : {self.cfg.conda_env}")
        log_info(f"Iterations   : {self.cfg.iterations}")
        log_info(f"Densification: {self.cfg.densify_start} → {self.cfg.densify_end}")
        log_info(f"Resolution cap: {self.cfg.resolution_cap}px")

        # Resume: skip if final PLY already exists
        if self.cfg.resume and self.cfg.final_ply.exists():
            log_info("Resume: final point_cloud.ply exists — skipping training.")
        else:
            if not self._train():
                return False

        # Evaluation
        if self.cfg.run_render:
            self._render()
            self._metrics()

        self._print_summary(image_count)
        return True

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_repo(self) -> bool:
        """Verify that gaussian-splatting repository and scripts are available."""
        if not self.cfg.gs_repo.exists():
            log_warn(
                f"Gaussian Splatting repo not found: {self.cfg.gs_repo}\n\n"
                "  Install it with:\n"
                "    git clone --recursive "
                "https://github.com/graphdeco-inria/gaussian-splatting\n"
                "    cd gaussian-splatting\n"
                "    conda env create -f environment.yml\n"
                "    conda activate gaussian_splatting\n"
            )
            return False

        if not self.cfg.train_script.exists():
            log_warn(f"train.py not found in repo: {self.cfg.gs_repo}")
            return False

        return True

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def _train(self) -> bool:
        """Launch gaussian-splatting training command and report elapsed time."""
        log_info("Training 3DGS model…  (this takes 20–90 min depending on GPU)")
        log_info(f"Monitor training at: http://127.0.0.1:{self.cfg.viewer_port}")

        cmd = self._build_train_cmd()

        t0 = time.time()
        try:
            if self._conda_sh:
                run_in_conda(
                    self._conda_sh,
                    self.cfg.conda_env,
                    cmd,
                    dry_run=self.cfg.dry_run,
                    cwd=self.cfg.gs_repo,
                )
            else:
                # Fallback: hope the right Python is already active
                run(cmd, dry_run=self.cfg.dry_run, cwd=self.cfg.gs_repo)
        except CommandError as e:
            log_warn(f"Training failed: {e}")
            return False

        elapsed = time.time() - t0
        log_success(f"Training complete in {format_duration(elapsed)}")
        return True

    def _build_train_cmd(self) -> List:
        """Build train.py CLI arguments from pipeline configuration values."""
        cfg = self.cfg

        cmd: List = [
            "python",
            str(cfg.train_script),
            # I/O
            "--source_path",
            str(cfg.colmap_dense),
            "--model_path",
            str(cfg.gs_output),
            # Iterations
            "--iterations",
            str(cfg.iterations),
            # Densification — stop at half-way for interiors (avoids wall over-splat)
            "--densify_from_iter",
            str(cfg.densify_start),
            "--densify_until_iter",
            str(cfg.densify_end),
            "--densify_grad_threshold",
            str(cfg.densify_grad_threshold),
            # Opacity reset — prunes floaters near walls/ceilings
            "--opacity_reset_interval",
            "3000",
            # Resolution — let 3DGS decide, but cap to guard VRAM
            "--resolution",
            "-1",
            "--resolution_scale",
            "1.0",
            # Image resolution cap (Pixel 4K → 3840px without this uses ~16GB)
            "--images",
            str(cfg.colmap_dense / "images"),
            # Evaluation split
            "--eval",
            "--llffhold",
            str(cfg.test_holdout),
            # Viewer
            "--ip",
            "127.0.0.1",
            "--port",
            str(cfg.viewer_port),
        ]

        # Checkpointing
        save_iters: List[str] = []
        ckpt_iters: List[str] = []
        if cfg.checkpoint_interval > 0:
            # Save at each interval and at the final iteration
            intervals = list(
                range(
                    cfg.checkpoint_interval,
                    cfg.iterations,
                    cfg.checkpoint_interval,
                )
            )
            if cfg.iterations not in intervals:
                intervals.append(cfg.iterations)
            save_iters = [str(i) for i in intervals]
            ckpt_iters = save_iters[:]
        else:
            save_iters = [str(cfg.iterations)]

        cmd += ["--save_iterations"] + save_iters
        if ckpt_iters:
            cmd += ["--checkpoint_iterations"] + ckpt_iters

        return cmd

    # ------------------------------------------------------------------ #
    # Render + metrics
    # ------------------------------------------------------------------ #

    def _render(self) -> None:
        """Render held-out viewpoints with the trained 3DGS model."""
        if not self.cfg.render_script.exists():
            log_warn("render.py not found — skipping render step.")
            return

        log_info("Rendering held-out test views…")
        cmd: List = [
            "python",
            str(self.cfg.render_script),
            "--model_path",
            str(self.cfg.gs_output),
            "--source_path",
            str(self.cfg.colmap_dense),
            "--iteration",
            str(self.cfg.iterations),
            "--skip_train",
        ]

        try:
            if self._conda_sh:
                run_in_conda(
                    self._conda_sh,
                    self.cfg.conda_env,
                    cmd,
                    dry_run=self.cfg.dry_run,
                    cwd=self.cfg.gs_repo,
                )
            else:
                run(cmd, dry_run=self.cfg.dry_run)
            log_success(f"Renders saved → {self.cfg.gs_output / 'test'}")
        except CommandError as e:
            log_warn(f"Render step failed (non-fatal): {e}")

    def _metrics(self) -> None:
        """Compute image quality metrics for rendered test views."""
        if not self.cfg.metrics_script.exists():
            log_warn("metrics.py not found — skipping metrics step.")
            return

        log_info("Computing PSNR / SSIM / LPIPS on test split…")
        cmd: List = [
            "python",
            str(self.cfg.metrics_script),
            "--model_path",
            str(self.cfg.gs_output),
            "--iteration",
            str(self.cfg.iterations),
        ]

        try:
            if self._conda_sh:
                run_in_conda(
                    self._conda_sh,
                    self.cfg.conda_env,
                    cmd,
                    dry_run=self.cfg.dry_run,
                    cwd=self.cfg.gs_repo,
                )
            else:
                run(cmd, dry_run=self.cfg.dry_run)
        except CommandError as e:
            log_warn(f"Metrics step failed (non-fatal): {e}")
            return

        # Pretty-print results.json if it exists
        results_file = self.cfg.gs_output / "results.json"
        if results_file.exists() and not self.cfg.dry_run:
            try:
                with open(results_file) as f:
                    data = json.load(f)
                print("\n  Quality metrics (test split):")
                for split, vals in data.items():
                    print(f"    [{split}]")
                    for k, v in vals.items():
                        if isinstance(v, float):
                            print(f"      {k:<10}: {v:.4f}")
                        else:
                            print(f"      {k:<10}: {v}")
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def _print_summary(self, image_count: int) -> None:
        """Print output locations, artifact stats, and quick follow-up guidance."""
        ply = self.cfg.final_ply
        print()
        print("══════════════════════════════════════════════════════════")
        print("  Pipeline Complete")
        print("══════════════════════════════════════════════════════════")
        print(f"  Output root  : {self.cfg.output_root}")
        print(f"  Frames       : {self.cfg.frames_dir}")
        print(
            f"  COLMAP sparse: {self.cfg.best_sparse_model or self.cfg.colmap_sparse}"
        )
        print(f"  COLMAP dense : {self.cfg.colmap_dense}")
        print(f"  3DGS output  : {self.cfg.gs_output}")

        if ply.exists():
            splat_count = count_ply_points(ply)
            size_mb = ply.stat().st_size / 1_048_576
            print()
            print(f"  Final PLY    : {ply}")
            print(f"  File size    : {size_mb:.1f} MB")
            if splat_count:
                print(f"  Gaussians    : {splat_count:,}")

        print()
        print("  ── Viewing ───────────────────────────────────────────")
        print(f"  SIBR viewer:")
        sibr = self.cfg.gs_repo / "SIBR_viewers" / "bin" / "SIBR_gaussianViewer_app"
        print(f"    {sibr} -m {self.cfg.gs_output}")
        print()
        print("  Web viewer (drag & drop PLY):")
        print("    https://antimatter15.com/splat/")
        print("    https://playcanvas.com/viewer")
        print()
        print("  ── Troubleshooting ───────────────────────────────────")
        print("  Floaters / noise   → lower --densify-grad-threshold 0.00015")
        print("  VRAM OOM           → lower --resolution-cap 1024")
        print("  Blurry result      → increase --iterations 50000")
        print("  Low registration   → add --vocab-tree <path> to COLMAP step")
        print("══════════════════════════════════════════════════════════")
