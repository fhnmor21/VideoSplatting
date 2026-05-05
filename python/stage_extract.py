"""
pipeline/stage_extract.py — Stage 1: Smart frame extraction from video.

Uses ffmpeg's scene-change detection score to extract frames only when there
has been enough visual change from the previous frame.  This naturally adapts
to variable camera speed: slow pans produce fewer frames, fast motion
produces more — ensuring good overlap throughout without frame duplication.

The approach:
  select filter keeps a frame when EITHER:
    (A) scene score > threshold  AND  time since last kept frame >= min_gap
    (B) time since last kept frame >= max_gap  (safety net for slow passages)
"""

from __future__ import annotations

import time
from pathlib import Path

from config.settings import PipelineConfig
from pipeline.utils import (
    CommandError, check_tool, log_header, log_info,
    log_success, log_warn, probe_video, run,
)


class FrameExtractor:
    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        log_header("Stage 1 — Frame Extraction")

        if not check_tool("ffmpeg", "Install: https://ffmpeg.org/download.html"):
            return False

        frames_dir = self.cfg.frames_dir

        # Resume: skip if frames already exist
        if self.cfg.resume:
            existing = list(frames_dir.glob("frame_*.jpg"))
            if existing:
                log_info(f"Resume: found {len(existing)} existing frames — skipping extraction.")
                return True

        # Probe the video for metadata
        video_info = probe_video(self.cfg.video)
        self._print_video_info(video_info)

        # Build and run the ffmpeg command
        cmd = self._build_command(frames_dir)
        log_info("Extracting frames (this may take a while for long videos)…")
        t0 = time.time()

        try:
            run(cmd, dry_run=self.cfg.dry_run)
        except CommandError as e:
            log_warn(f"ffmpeg exited with an error: {e}")
            # ffmpeg returns non-zero on some warnings; check if we got frames
            pass

        elapsed = time.time() - t0

        # Count output
        frames = list(frames_dir.glob("frame_*.jpg"))
        n = len(frames)

        if n == 0 and not self.cfg.dry_run:
            log_warn("No frames were extracted!")
            log_warn(f"  Try lowering --scene-threshold (current: {self.cfg.scene_threshold})")
            log_warn(f"  Or lower --max-gap (current: {self.cfg.max_gap}s)")
            return False

        log_success(f"Extracted {n} frames in {elapsed:.1f}s  →  {frames_dir}")

        # Advisory: rough average gap
        duration = video_info.get("duration")
        if duration and n > 0:
            avg_gap = duration / n
            log_info(f"Average gap between frames: {avg_gap:.1f}s")
            if avg_gap > 5:
                log_warn(
                    f"Average gap is {avg_gap:.1f}s — coverage may be sparse. "
                    f"Consider lowering --scene-threshold or --max-gap."
                )
            elif avg_gap < 0.3:
                log_warn(
                    f"Average gap is only {avg_gap:.2f}s — many near-duplicate frames. "
                    f"Consider raising --scene-threshold or --min-gap."
                )

        return True

    # ------------------------------------------------------------------ #
    # Build ffmpeg command
    # ------------------------------------------------------------------ #

    def _build_command(self, frames_dir: Path) -> list:
        """
        Build the ffmpeg select-filter command.

        The select expression:
          gt(scene, T) * gte(t - prev_selected_t, MIN)    → scene-change gate
          + gte(t - prev_selected_t, MAX)                  → max-gap safety net

        Commas inside the expression must be escaped as \\, for ffmpeg's
        filter-graph parser.
        """
        T   = self.cfg.scene_threshold
        MIN = self.cfg.min_gap
        MAX = self.cfg.max_gap
        Q   = self.cfg.frame_quality

        # Build the select expression (escaped commas)
        select_expr = (
            f"gt(scene\\,{T})*gte(t-prev_selected_t\\,{MIN})"
            f"+gte(t-prev_selected_t\\,{MAX})"
        )

        output_pattern = str(frames_dir / "frame_%06d.jpg")

        return [
            "ffmpeg",
            "-i", str(self.cfg.video),
            "-vf", f"select='{select_expr}'",
            "-vsync", "vfr",       # variable frame rate — only output selected frames
            "-qscale:v", str(Q),   # JPEG quality
            "-frame_pts", "true",  # embed PTS in output (helps with ordering checks)
            output_pattern,
        ]

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #

    def _print_video_info(self, info: dict) -> None:
        log_info(f"Video      : {self.cfg.video.name}")
        if info.get("width"):
            log_info(f"Resolution : {info['width']} × {info['height']}")
        if info.get("fps"):
            log_info(f"Frame rate : {info['fps']:.3f} fps")
        if info.get("duration"):
            log_info(f"Duration   : {info['duration']:.1f}s")
        log_info(f"Threshold  : scene > {self.cfg.scene_threshold}")
        log_info(f"Min gap    : {self.cfg.min_gap}s")
        log_info(f"Max gap    : {self.cfg.max_gap}s")
        log_info(f"Output dir : {self.cfg.frames_dir}")
