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
from pipeline.gaussian_backends import backend_for
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
    run_in_uv,
)


class GaussianTrainer:
    """Train and evaluate a 3D Gaussian Splatting model from COLMAP output."""

    def __init__(self, config: PipelineConfig) -> None:
        """Store shared pipeline configuration and deferred conda metadata."""
        self.cfg = config
        self.backend = backend_for(config)
        self._conda_sh: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        """Validate prerequisites, run training, and optionally run evaluation."""
        log_header("Stage 3 — 3D Gaussian Splatting Training")

        # Validate backend and repo
        if not self.backend.validate():
            return False

        if self.cfg.env_runner == "conda":
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
        log_info(f"Backend      : {self.backend.name}")
        if self.cfg.env_runner == "conda":
            log_info(f"Conda env    : {self.backend.env_name}")
        else:
            log_info(f"UV python    : {self.cfg.uv_python}")
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
    # Training
    # ------------------------------------------------------------------ #

    def _train(self) -> bool:
        """Launch gaussian-splatting training command and report elapsed time."""
        log_info("Training 3DGS model…  (this takes 20–90 min depending on GPU)")
        log_info(f"Monitor training at: http://127.0.0.1:{self.cfg.viewer_port}")

        cmd = self.backend.build_train_cmd()

        t0 = time.time()
        try:
            if self.cfg.env_runner == "conda" and self._conda_sh:
                run_in_conda(
                    self._conda_sh,
                    self.backend.env_name,
                    cmd,
                    dry_run=self.cfg.dry_run,
                    cwd=self.cfg.gs_repo,
                )
            elif self.cfg.env_runner == "uv":
                run_in_uv(
                    self.cfg.uv_python,
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

    # ------------------------------------------------------------------ #
    # Render + metrics
    # ------------------------------------------------------------------ #

    def _render(self) -> None:
        """Render held-out viewpoints with the trained 3DGS model."""
        if not self.cfg.render_script.exists():
            log_warn("render.py not found — skipping render step.")
            return

        log_info("Rendering held-out test views…")
        cmd = self.backend.build_render_cmd()

        try:
            if self.cfg.env_runner == "conda" and self._conda_sh:
                run_in_conda(
                    self._conda_sh,
                    self.backend.env_name,
                    cmd,
                    dry_run=self.cfg.dry_run,
                    cwd=self.cfg.gs_repo,
                )
            elif self.cfg.env_runner == "uv":
                run_in_uv(
                    self.cfg.uv_python,
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
        cmd = self.backend.build_metrics_cmd()

        try:
            if self.cfg.env_runner == "conda" and self._conda_sh:
                run_in_conda(
                    self._conda_sh,
                    self.backend.env_name,
                    cmd,
                    dry_run=self.cfg.dry_run,
                    cwd=self.cfg.gs_repo,
                )
            elif self.cfg.env_runner == "uv":
                run_in_uv(
                    self.cfg.uv_python,
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
