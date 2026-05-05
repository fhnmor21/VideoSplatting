"""
pipeline/runner.py — Orchestrates all pipeline stages in sequence.

Responsible for:
  * Pre-flight checks (tools installed, paths exist, GPU available)
  * Calling each stage in order
  * Collecting timing per stage
  * Printing a final summary table
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import PipelineConfig
from pipeline.stage_extract import FrameExtractor
from pipeline.stage_colmap import ColmapReconstructor
from pipeline.stage_gaussian import GaussianTrainer
from pipeline.utils import (
    format_duration, log_error, log_header, log_info,
    log_success, log_warn,
)


@dataclass
class StageResult:
    name: str
    success: bool
    skipped: bool = False
    elapsed: float = 0.0
    error: Optional[str] = None


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.results: List[StageResult] = []

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        self._print_banner()

        if not self._preflight():
            return False

        self.cfg.ensure_dirs()

        stages = [
            ("Frame Extraction",          self.cfg.run_extract,  self._run_extract),
            ("COLMAP Reconstruction",     self.cfg.run_colmap,   self._run_colmap),
            ("Gaussian Splatting",        self.cfg.run_training, self._run_gaussian),
        ]

        pipeline_ok = True
        for name, enabled, fn in stages:
            if not enabled:
                self.results.append(StageResult(name=name, success=True, skipped=True))
                log_info(f"Skipping stage: {name}")
                continue

            t0 = time.time()
            try:
                ok = fn()
            except Exception as exc:
                ok = False
                log_error(f"Unexpected error in '{name}': {exc}")

            elapsed = time.time() - t0
            self.results.append(StageResult(
                name=name,
                success=ok,
                elapsed=elapsed,
            ))

            if not ok:
                log_error(f"Stage '{name}' failed — aborting pipeline.")
                pipeline_ok = False
                break

        self._print_stage_summary()
        return pipeline_ok

    # ------------------------------------------------------------------ #
    # Stage runners
    # ------------------------------------------------------------------ #

    def _run_extract(self) -> bool:
        return FrameExtractor(self.cfg).run()

    def _run_colmap(self) -> bool:
        return ColmapReconstructor(self.cfg).run()

    def _run_gaussian(self) -> bool:
        return GaussianTrainer(self.cfg).run()

    # ------------------------------------------------------------------ #
    # Pre-flight checks
    # ------------------------------------------------------------------ #

    def _preflight(self) -> bool:
        log_header("Pre-flight Checks")
        ok = True

        # Required tools
        tools = {"ffmpeg": "https://ffmpeg.org/download.html",
                 "ffprobe": "https://ffmpeg.org/download.html",
                 "colmap": "https://colmap.github.io/install.html"}

        for tool, hint in tools.items():
            if shutil.which(tool):
                log_success(f"{tool} found")
            else:
                log_warn(f"{tool} not found in PATH — install: {hint}")
                ok = False  # hard requirement

        # GPU check (advisory only — we don't abort without a GPU)
        gpu_info = self._detect_gpu()
        if gpu_info:
            log_success(f"GPU: {gpu_info}")
        else:
            log_warn(
                "No NVIDIA GPU detected. COLMAP and 3DGS training both require CUDA.\n"
                "  COLMAP can run on CPU (slow) but 3DGS cannot."
            )

        # GS repo check
        if self.cfg.run_training:
            if self.cfg.gs_repo.exists() and self.cfg.train_script.exists():
                log_success(f"GS repo: {self.cfg.gs_repo}")
            else:
                log_warn(
                    f"gaussian-splatting repo not found at: {self.cfg.gs_repo}\n"
                    "  Clone it with:\n"
                    "    git clone --recursive "
                    "https://github.com/graphdeco-inria/gaussian-splatting\n"
                    "    cd gaussian-splatting && conda env create -f environment.yml"
                )
                ok = False

        # Vocab tree advisory
        if self.cfg.vocab_tree and not self.cfg.vocab_tree.exists():
            log_warn(
                f"Vocab tree not found: {self.cfg.vocab_tree}\n"
                "  Download from https://demuc.de/colmap/#dataset\n"
                "  Continuing without loop-closure detection."
            )
            # Non-fatal — clear it so downstream code doesn't try to use it
            self.cfg.vocab_tree = None  # type: ignore[assignment]

        if ok:
            log_success("All required tools present.")
        return ok

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _detect_gpu(self) -> Optional[str]:
        if not shutil.which("nvidia-smi"):
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            return lines[0] if lines else None
        except Exception:
            return None

    def _print_banner(self) -> None:
        cfg = self.cfg
        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║     Video → 3D Gaussian Splatting Pipeline               ║")
        print("║     Google Pixel + Blackmagic Camera — Interior Building ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()
        print(f"  Video        : {cfg.video}")
        print(f"  Output root  : {cfg.output_root}")
        print(f"  GS repo      : {cfg.gs_repo}")
        print(f"  Conda env    : {cfg.conda_env}")
        print(f"  Iterations   : {cfg.iterations}")
        print(f"  Resolution   : cap {cfg.resolution_cap}px")
        print(f"  Dry run      : {cfg.dry_run}")
        print(f"  Resume       : {cfg.resume}")
        enabled = []
        if cfg.run_extract:  enabled.append("extract")
        if cfg.run_colmap:   enabled.append("colmap")
        if cfg.run_training: enabled.append("train")
        if cfg.run_render:   enabled.append("render")
        print(f"  Stages       : {', '.join(enabled)}")
        print()

    def _print_stage_summary(self) -> None:
        print()
        print("┌─────────────────────────────────────────────────────────┐")
        print("│  Stage Summary                                          │")
        print("├────────────────────────────┬────────────┬───────────────┤")
        print("│ Stage                      │ Status     │ Time          │")
        print("├────────────────────────────┼────────────┼───────────────┤")
        for r in self.results:
            if r.skipped:
                status = "skipped"
                t = "—"
            elif r.success:
                status = "✓ ok"
                t = format_duration(r.elapsed)
            else:
                status = "✗ FAILED"
                t = format_duration(r.elapsed)
            print(f"│ {r.name:<26}  │ {status:<10} │ {t:<13} │")
        print("└────────────────────────────┴────────────┴───────────────┘")
