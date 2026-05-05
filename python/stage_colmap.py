"""
pipeline/stage_colmap.py — Stage 2: COLMAP sparse reconstruction.

Runs the full COLMAP SfM pipeline:
  2a. feature_extractor    — SIFT keypoints per image
  2b. sequential_matcher   — exploits video capture order
  2c. vocab_tree_matcher   — optional second pass for loop closure
  2d. mapper               — incremental SfM → sparse point cloud + cameras
  2e. model_converter      — exports human-readable text model
  2f. image_undistorter    — undistorts images for 3DGS input

All parameters are tuned for:
  * Google Pixel + Blackmagic Camera App (24mm equiv, OPENCV distortion)
  * Interior building (low texture, variable lighting, short baselines)
  * Sequential walkthrough (no turntable, one contiguous path)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from config.settings import PipelineConfig
from pipeline.utils import (
    CommandError, check_tool, count_registered_images,
    log_header, log_info, log_success, log_warn,
    probe_image, run,
)


class ColmapReconstructor:
    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        log_header("Stage 2 — COLMAP Sparse Reconstruction")

        if not check_tool("colmap", "Install: https://colmap.github.io/install.html"):
            return False

        frames_dir = self.cfg.frames_dir
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            log_warn(f"No frames found in {frames_dir} — did Stage 1 complete?")
            return False

        log_info(f"Input frames : {len(frame_files)} images in {frames_dir}")

        # Detect image dimensions for camera priors
        img_w, img_h = probe_image(frame_files[0])
        if img_w and img_h:
            log_info(f"Image size   : {img_w} × {img_h}")

        # Build camera parameter string
        cam_params = ""
        if img_w and img_h:
            cam_params = self.cfg.opencv_camera_params(img_w, img_h)
            focal = self.cfg.focal_px or self.cfg.estimate_focal_px(img_w)
            log_info(f"Focal prior  : {focal:.0f} px  (auto-estimated from Pixel geometry)")
            log_info(f"Camera params: {cam_params}")
        else:
            log_warn("Could not determine image dimensions — COLMAP will estimate focal length.")

        # Run sub-stages
        t0 = time.time()

        db   = self.cfg.colmap_database
        imgs = self.cfg.frames_dir

        if not self._feature_extract(db, imgs, cam_params):
            return False
        if not self._sequential_match(db):
            return False
        if self.cfg.vocab_tree:
            self._vocab_tree_match(db)   # optional; failure is non-fatal
        if not self._mapper(db, imgs):
            return False

        best = self.cfg.best_sparse_model
        if best is None:
            log_warn("Mapper produced no sub-model — reconstruction failed.")
            return False

        n_reg = count_registered_images(best)
        log_success(
            f"Sparse model : {best}  "
            f"({n_reg}/{len(frame_files)} images registered)"
        )

        if n_reg is not None and n_reg < len(frame_files) * 0.8:
            log_warn(
                f"Only {n_reg}/{len(frame_files)} frames registered (<80%). "
                "Consider: lower --scene-threshold, add --vocab-tree, "
                "or check lighting."
            )

        if not self._export_txt(best):
            return False
        if not self._undistort(best, imgs):
            return False

        elapsed = time.time() - t0
        log_success(f"COLMAP complete in {elapsed:.0f}s")
        return True

    # ------------------------------------------------------------------ #
    # 2a — Feature extraction
    # ------------------------------------------------------------------ #

    def _feature_extract(self, db: Path, imgs: Path, cam_params: str) -> bool:
        log_info("2a — Feature extraction (SIFT)…")

        # Resume: skip if database already has features
        if self.cfg.resume and db.exists() and db.stat().st_size > 65_536:
            log_info("Resume: database exists — skipping feature extraction.")
            return True

        cmd: List = [
            "colmap", "feature_extractor",
            "--database_path", db,
            "--image_path",    imgs,

            # Camera model
            "--ImageReader.camera_model",  self.cfg.camera_model,
            "--ImageReader.single_camera", "1",  # all frames = same phone/lens

            # SIFT — tuned for dark, low-texture interiors
            "--SiftExtraction.use_gpu",             "1",
            "--SiftExtraction.gpu_index",            str(self.cfg.colmap_gpu),
            "--SiftExtraction.num_threads",          str(self.cfg.cpu_threads),
            "--SiftExtraction.max_num_features",    "8192",
            "--SiftExtraction.first_octave",        "-1",    # finer scale detection
            "--SiftExtraction.peak_threshold",      "0.003", # more kps on low-contrast walls
            "--SiftExtraction.edge_threshold",      "10",
            "--SiftExtraction.domain_size_pooling", "1",     # better descriptors on flat surfaces
        ]

        if cam_params:
            cmd += ["--ImageReader.camera_params", cam_params]

        try:
            run(cmd, dry_run=self.cfg.dry_run)
        except CommandError as e:
            log_warn(f"Feature extraction failed: {e}")
            return False

        log_success("Feature extraction done.")
        return True

    # ------------------------------------------------------------------ #
    # 2b — Sequential matcher
    # ------------------------------------------------------------------ #

    def _sequential_match(self, db: Path) -> bool:
        log_info("2b — Sequential matching (video order)…")

        cmd: List = [
            "colmap", "sequential_matcher",
            "--database_path", db,

            "--SiftMatching.use_gpu",      "1",
            "--SiftMatching.gpu_index",     str(self.cfg.colmap_gpu),
            "--SiftMatching.num_threads",   str(self.cfg.cpu_threads),
            "--SiftMatching.max_ratio",    "0.80",   # Lowe ratio test
            "--SiftMatching.max_distance", "0.7",
            "--SiftMatching.cross_check",  "1",

            # Geometric verification — require solid overlap
            "--TwoViewGeometry.min_num_inliers", "15",

            # Sequential window — match each frame against 20 neighbours each side;
            # quadratic_overlap also matches at 2×, 4×, 8× steps to bridge gaps
            # that extract_frames.sh may have created.
            "--SequentialMatching.overlap",           "20",
            "--SequentialMatching.quadratic_overlap",  "1",
            "--SequentialMatching.loop_detection",     "0",  # handled separately below
        ]

        try:
            run(cmd, dry_run=self.cfg.dry_run)
        except CommandError as e:
            log_warn(f"Sequential matching failed: {e}")
            return False

        log_success("Sequential matching done.")
        return True

    # ------------------------------------------------------------------ #
    # 2c — Vocab tree matcher (optional, for loop closure)
    # ------------------------------------------------------------------ #

    def _vocab_tree_match(self, db: Path) -> bool:
        vt = self.cfg.vocab_tree
        if vt is None or not vt.exists():
            return True   # nothing to do

        log_info(f"2c — Vocab tree matching (loop closure)  [{vt.name}]…")

        cmd: List = [
            "colmap", "vocab_tree_matcher",
            "--database_path", db,

            "--SiftMatching.use_gpu",     "1",
            "--SiftMatching.gpu_index",    str(self.cfg.colmap_gpu),
            "--SiftMatching.num_threads",  str(self.cfg.cpu_threads),
            "--SiftMatching.max_ratio",   "0.80",

            "--VocabTreeMatching.vocab_tree_path",  str(vt),
            "--VocabTreeMatching.num_images",       "50",
        ]

        try:
            run(cmd, dry_run=self.cfg.dry_run)
            log_success("Vocab tree matching done.")
        except CommandError as e:
            log_warn(f"Vocab tree matching failed (non-fatal): {e}")

        return True

    # ------------------------------------------------------------------ #
    # 2d — Incremental mapper
    # ------------------------------------------------------------------ #

    def _mapper(self, db: Path, imgs: Path) -> bool:
        log_info("2d — Incremental mapper (SfM)…")
        log_info("     This is the longest COLMAP step — progress logged below.")

        sparse_out = self.cfg.colmap_sparse

        # Resume: skip if a sub-model already exists
        if self.cfg.resume and self.cfg.best_sparse_model is not None:
            log_info("Resume: sparse model exists — skipping mapper.")
            return True

        cmd: List = [
            "colmap", "mapper",
            "--database_path", db,
            "--image_path",    imgs,
            "--output_path",   sparse_out,

            # Initialisation — require a solid seed pair (avoids planar-wall init)
            "--Mapper.init_min_num_inliers",      "100",
            "--Mapper.init_max_forward_motion",   "0.95",

            # Registration — looser thresholds for interior (short baseline)
            "--Mapper.abs_pose_min_num_inliers",   "30",
            "--Mapper.abs_pose_min_inlier_ratio",  "0.25",

            # Bundle adjustment
            "--Mapper.ba_local_num_images",        "6",
            "--Mapper.ba_global_images_ratio",     "1.1",
            "--Mapper.ba_global_points_ratio",     "1.1",

            # Accept wide-angle focal priors (Pixel main cam sits outside default range)
            "--Mapper.min_focal_length_ratio",     "0.1",
            "--Mapper.max_focal_length_ratio",    "10.0",
            "--Mapper.max_extra_param",            "1.0",

            "--Mapper.num_threads", str(self.cfg.cpu_threads),
        ]

        try:
            run(cmd, dry_run=self.cfg.dry_run)
        except CommandError as e:
            log_warn(f"Mapper failed: {e}")
            return False

        return True

    # ------------------------------------------------------------------ #
    # 2e — Export text model
    # ------------------------------------------------------------------ #

    def _export_txt(self, model: Path) -> bool:
        log_info("2e — Exporting text model…")
        txt_dir = model / "txt"
        txt_dir.mkdir(exist_ok=True)

        cmd: List = [
            "colmap", "model_converter",
            "--input_path",  model,
            "--output_path", txt_dir,
            "--output_type", "TXT",
        ]
        try:
            run(cmd, dry_run=self.cfg.dry_run)
        except CommandError as e:
            log_warn(f"model_converter failed (non-fatal): {e}")
            return True   # not critical for 3DGS

        log_success(f"Text model  → {txt_dir}")
        return True

    # ------------------------------------------------------------------ #
    # 2f — Image undistorter
    # ------------------------------------------------------------------ #

    def _undistort(self, model: Path, imgs: Path) -> bool:
        log_info("2f — Image undistortion…")

        dense_out = self.cfg.colmap_dense

        # Resume: skip if undistorted images already exist
        if self.cfg.resume and (dense_out / "images").exists():
            undist_imgs = list((dense_out / "images").glob("*.jpg")) + \
                          list((dense_out / "images").glob("*.png"))
            if undist_imgs:
                log_info("Resume: undistorted images exist — skipping undistortion.")
                return True

        cmd: List = [
            "colmap", "image_undistorter",
            "--image_path",   imgs,
            "--input_path",   model,
            "--output_path",  dense_out,
            "--output_type",  "COLMAP",   # produces sparse/ + images/ for 3DGS
        ]
        try:
            run(cmd, dry_run=self.cfg.dry_run)
        except CommandError as e:
            log_warn(f"image_undistorter failed: {e}")
            return False

        log_success(f"Undistorted → {dense_out}")
        return True
