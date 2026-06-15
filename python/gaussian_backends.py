"""Backend adapters for gaussian-splatting training execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Callable, List, Sequence
import shutil

from config.settings import PipelineConfig
from pipeline.utils import CommandError, copy_file, log_warn


@dataclass
class BaseGaussianBackend:
    cfg: PipelineConfig
    name: str
    env_name: str

    def validate(self) -> bool:
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

    def build_train_cmd(self) -> List[str]:
        cfg = self.cfg
        cmd: List[str] = [
            "python",
            str(cfg.train_script),
            "--source_path",
            str(cfg.colmap_dense),
            "--model_path",
            str(cfg.gs_output),
            "--iterations",
            str(cfg.iterations),
            "--densify_from_iter",
            str(cfg.densify_start),
            "--densify_until_iter",
            str(cfg.densify_end),
            "--densify_grad_threshold",
            str(cfg.densify_grad_threshold),
            "--opacity_reset_interval",
            "3000",
            "--resolution",
            "-1",
            "--resolution_scale",
            "1.0",
            "--images",
            str(cfg.colmap_dense / "images"),
            "--eval",
            "--llffhold",
            str(cfg.test_holdout),
            "--ip",
            "127.0.0.1",
            "--port",
            str(cfg.viewer_port),
        ]

        save_iters: List[str]
        ckpt_iters: List[str] = []
        if cfg.checkpoint_interval > 0:
            intervals = list(
                range(cfg.checkpoint_interval, cfg.iterations, cfg.checkpoint_interval)
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

    def train(self, runner: Callable[[Sequence[str]], None]) -> bool:
        try:
            runner(self.build_train_cmd())
        except CommandError:
            return False
        return True

    def build_render_cmd(self) -> List[str]:
        return [
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

    def render(self, runner: Callable[[Sequence[str]], None]) -> bool:
        if not self.cfg.render_script.exists():
            log_warn("render.py not found — skipping render step.")
            return False
        try:
            runner(self.build_render_cmd())
        except CommandError:
            return False
        return True

    def build_metrics_cmd(self) -> List[str]:
        return [
            "python",
            str(self.cfg.metrics_script),
            "--model_path",
            str(self.cfg.gs_output),
            "--iteration",
            str(self.cfg.iterations),
        ]

    def metrics(self, runner: Callable[[Sequence[str]], None]) -> bool:
        if not self.cfg.metrics_script.exists():
            log_warn("metrics.py not found — skipping metrics step.")
            return False
        try:
            runner(self.build_metrics_cmd())
        except CommandError:
            return False
        return True

    def finalize_outputs(self) -> bool:
        return self.cfg.final_ply.exists()


class CudaGaussianBackend(BaseGaussianBackend):
    def __init__(self, cfg: PipelineConfig) -> None:
        super().__init__(cfg=cfg, name="cuda", env_name=cfg.conda_env)


class RocmGaussianBackend(BaseGaussianBackend):
    def __init__(self, cfg: PipelineConfig) -> None:
        super().__init__(cfg=cfg, name="rocm", env_name=cfg.rocm_env)

    def validate(self) -> bool:
        if not self.cfg.rocm_env:
            log_warn("ROCm backend selected but --rocm-env is empty.")
            return False

        if self.cfg.env_runner == "uv" and not self.cfg.uv_python:
            log_warn("ROCm + uv runner requires --uv-python <path>.")
            return False

        if not self.cfg.gs_repo.exists():
            log_warn(f"ROCm GSplat repository not found: {self.cfg.gs_repo}")
            return False

        train_script = self.cfg.gs_repo / self.cfg.gsplat_train_script
        if not train_script.exists():
            log_warn(
                f"GSplat training script not found: {train_script}\n"
                "  Set --gs-repo to a ROCm gsplat checkout or update gsplat_train_script."
            )
            return False

        if not self.cfg.dry_run and not Path("/opt/rocm").exists():
            log_warn(
                "ROCm backend selected but /opt/rocm was not found.\n"
                "  Ensure ROCm runtime + ROCm PyTorch are installed.\n"
                "  Verify with: python -c \"import torch; print(torch.version.hip)\""
            )
            return False

        if not self.cfg.dry_run and self.cfg.env_runner == "uv":
            try:
                torch_check = subprocess.run(
                    [
                        self.cfg.uv_python,
                        "-c",
                        "import torch; print(torch.version.hip)",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if torch_check.returncode != 0 or not torch_check.stdout.strip():
                    log_warn(
                        "ROCm backend requires a ROCm-enabled PyTorch environment.\n"
                        "  Verify with: python -c \"import torch; print(torch.version.hip)\""
                    )
                    return False

                gsplat_check = subprocess.run(
                    [self.cfg.uv_python, "-c", "import gsplat"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if gsplat_check.returncode != 0:
                    log_warn(
                        "ROCm backend requires gsplat import to succeed.\n"
                        "  Verify with: python -c \"import gsplat\""
                    )
                    return False

                raster_probe = (
                    "import torch; "
                    "from gsplat.rendering import rasterization; "
                    "device='cuda'; "
                    "means=torch.zeros((1,3), device=device); "
                    "quats=torch.tensor([[1.0,0.0,0.0,0.0]], device=device); "
                    "scales=torch.ones((1,3), device=device)*0.01; "
                    "opacities=torch.ones((1,), device=device)*0.5; "
                    "colors=torch.ones((1,1,3), device=device); "
                    "viewmats=torch.eye(4, device=device).unsqueeze(0); "
                    "Ks=torch.tensor([[[100.0,0.0,64.0],[0.0,100.0,64.0],[0.0,0.0,1.0]]], device=device); "
                    "rasterization(means, quats, scales, opacities, colors, viewmats, Ks, 128, 128)"
                )
                raster_check = subprocess.run(
                    [self.cfg.uv_python, "-c", raster_probe],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env={**os.environ, "HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
                )
                if raster_check.returncode != 0:
                    stderr = (raster_check.stderr or "").strip()
                    detail = f"\n  Details: {stderr.splitlines()[-1]}" if stderr else ""
                    log_warn(
                        "ROCm gsplat runtime probe failed before training.\n"
                        "  This environment can import gsplat but crashes when launching rasterization kernels.\n"
                        "  Verify with: python -c \"from gsplat.rendering import rasterization; ...\""
                        f"{detail}"
                    )
                    return False
            except Exception as exc:
                log_warn(f"ROCm runtime validation failed: {exc}")
                return False

        return True

    def build_train_cmd(self) -> List[str]:
        return [
            "python",
            str(self.cfg.gs_repo / self.cfg.gsplat_train_script),
            "default",
            "--data-dir",
            str(self.cfg.colmap_dense),
            "--data-factor",
            "1",
            "--result-dir",
            str(self.cfg.rocm_backend_output_dir),
            "--max-steps",
            str(self.cfg.iterations),
            "--disable-viewer",
            "--save-ply",
        ]

    def build_render_cmd(self) -> List[str]:
        script = self.cfg.gs_repo / "examples" / "render.py"
        return [
            "python",
            str(script),
            "--model_dir",
            str(self.cfg.rocm_backend_output_dir),
        ]

    def build_metrics_cmd(self) -> List[str]:
        script = self.cfg.gs_repo / "examples" / "metrics.py"
        return [
            "python",
            str(script),
            "--model_dir",
            str(self.cfg.rocm_backend_output_dir),
        ]

    def finalize_outputs(self) -> bool:
        candidate = self.cfg.rocm_backend_output_dir / "point_cloud.ply"
        if candidate.exists():
            copy_file(candidate, self.cfg.final_ply)
            return self.cfg.final_ply.exists()

        ply_dir = self.cfg.rocm_backend_output_dir / "ply"
        if not ply_dir.exists():
            return False

        ply_files = sorted(ply_dir.glob("point_cloud_*.ply"))
        if not ply_files:
            return False

        latest = max(ply_files, key=lambda p: p.stat().st_mtime)
        self.cfg.final_ply.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest, self.cfg.final_ply)
        return self.cfg.final_ply.exists()


def backend_for(cfg: PipelineConfig) -> BaseGaussianBackend:
    if cfg.gs_backend == "rocm":
        return RocmGaussianBackend(cfg)
    return CudaGaussianBackend(cfg)
