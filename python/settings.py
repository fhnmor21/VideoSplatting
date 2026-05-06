"""
config/settings.py — Centralised configuration dataclass.

All pipeline stages read from a single PipelineConfig instance, which is
constructed once in main.py from CLI arguments and then passed around.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Central configuration object shared by all pipeline stages."""

    # ------------------------------------------------------------------ #
    # Top-level
    # ------------------------------------------------------------------ #
    video: Path
    output_root: Path
    gs_repo: Path
    conda_env: str = "gaussian_splatting"
    dry_run: bool = False
    resume: bool = False

    # ------------------------------------------------------------------ #
    # Stage switches
    # ------------------------------------------------------------------ #
    run_extract: bool = True
    run_colmap: bool = True
    run_training: bool = True
    run_render: bool = True

    # ------------------------------------------------------------------ #
    # Frame extraction
    # ------------------------------------------------------------------ #
    scene_threshold: float = 0.35
    min_gap: float = 0.5
    max_gap: float = 3.0
    frame_quality: int = 2  # ffmpeg JPEG qscale (1 = best)

    # ------------------------------------------------------------------ #
    # COLMAP
    # ------------------------------------------------------------------ #
    camera_model: str = "OPENCV"
    focal_px: Optional[float] = None
    vocab_tree: Optional[Path] = None
    colmap_gpu: int = 0
    colmap_threads: Optional[int] = None

    # Google Pixel main-lens geometry constants
    # actual_focal_mm / sensor_width_mm  =  3.65 / 6.9
    PIXEL_FOCAL_RATIO: float = field(default=0.5290, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Gaussian Splatting
    # ------------------------------------------------------------------ #
    iterations: int = 30_000
    densify_start: int = 500
    densify_end: int = 15_000
    densify_grad_threshold: float = 0.0002
    resolution_cap: int = 1600
    test_holdout: int = 8
    checkpoint_interval: int = 7_000
    viewer_port: int = 6009

    # ------------------------------------------------------------------ #
    # Derived paths (computed on first access via properties)
    # ------------------------------------------------------------------ #
    @property
    def frames_dir(self) -> Path:
        """Return the directory where extracted frames are written."""
        return self.output_root / "frames"

    @property
    def colmap_workspace(self) -> Path:
        """Return the root COLMAP workspace directory."""
        return self.output_root / "colmap"

    @property
    def colmap_database(self) -> Path:
        """Return the COLMAP SQLite database path."""
        return self.colmap_workspace / "database.db"

    @property
    def colmap_sparse(self) -> Path:
        """Return the directory that stores sparse COLMAP models."""
        return self.colmap_workspace / "sparse"

    @property
    def colmap_dense(self) -> Path:
        """Output of image_undistorter — used as the 3DGS source path."""
        return self.colmap_workspace / "dense"

    @property
    def gs_output(self) -> Path:
        """Return the output directory for Gaussian Splatting artifacts."""
        return self.output_root / "gaussian"

    @property
    def best_sparse_model(self) -> Optional[Path]:
        """Return the COLMAP sub-model directory with the most registered images."""
        sparse = self.colmap_sparse
        if not sparse.exists():
            return None
        sub_models = sorted(sparse.iterdir()) if sparse.exists() else []
        sub_models = [d for d in sub_models if d.is_dir()]
        if not sub_models:
            return None

        # The mapper names them 0, 1, 2 … in registration order; 0 is usually largest.
        # To be safe, pick the one whose images.bin is biggest.
        def model_size(d: Path) -> int:
            """Return byte size of images.bin for sparse model ranking."""
            ib = d / "images.bin"
            return ib.stat().st_size if ib.exists() else 0

        return max(sub_models, key=model_size)

    @property
    def cpu_threads(self) -> int:
        """Return configured COLMAP thread count or an OS-derived default."""
        if self.colmap_threads is not None:
            return self.colmap_threads
        return os.cpu_count() or 4

    # ------------------------------------------------------------------ #
    # Derived camera params
    # ------------------------------------------------------------------ #
    def estimate_focal_px(self, image_width: int) -> float:
        """
        Estimate focal length in pixels from Google Pixel sensor geometry.
        focal_px = image_width * (actual_EFL_mm / sensor_width_mm)
                 = image_width * (3.65 / 6.9)
        """
        return round(image_width * self.PIXEL_FOCAL_RATIO)

    def opencv_camera_params(self, image_width: int, image_height: int) -> str:
        """
        Build the COLMAP --ImageReader.camera_params string for the OPENCV model.
        Format: fx,fy,cx,cy,k1,k2,p1,p2
        Distortion coefficients start at 0; bundle adjustment will refine them.
        """
        focal = self.focal_px or self.estimate_focal_px(image_width)
        cx = image_width / 2.0
        cy = image_height / 2.0
        return f"{focal:.1f},{focal:.1f},{cx:.1f},{cy:.1f},0,0,0,0"

    # ------------------------------------------------------------------ #
    # Paths for GS repo scripts
    # ------------------------------------------------------------------ #
    @property
    def train_script(self) -> Path:
        """Return the path to the upstream gaussian-splatting train script."""
        return self.gs_repo / "train.py"

    @property
    def render_script(self) -> Path:
        """Return the path to the upstream gaussian-splatting render script."""
        return self.gs_repo / "render.py"

    @property
    def metrics_script(self) -> Path:
        """Return the path to the upstream gaussian-splatting metrics script."""
        return self.gs_repo / "metrics.py"

    @property
    def final_ply(self) -> Path:
        """Return the expected path to the final trained point cloud PLY."""
        return (
            self.gs_output
            / "point_cloud"
            / f"iteration_{self.iterations}"
            / "point_cloud.ply"
        )

    def ensure_dirs(self) -> None:
        """Create all output directories that must exist before the pipeline starts."""
        for d in (
            self.frames_dir,
            self.colmap_workspace,
            self.colmap_sparse,
            self.colmap_dense,
            self.gs_output,
        ):
            d.mkdir(parents=True, exist_ok=True)
