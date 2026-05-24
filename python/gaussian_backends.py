"""Backend adapters for gaussian-splatting training execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from config.settings import PipelineConfig
from pipeline.utils import log_warn


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

    def build_metrics_cmd(self) -> List[str]:
        return [
            "python",
            str(self.cfg.metrics_script),
            "--model_path",
            str(self.cfg.gs_output),
            "--iteration",
            str(self.cfg.iterations),
        ]


class CudaGaussianBackend(BaseGaussianBackend):
    def __init__(self, cfg: PipelineConfig) -> None:
        super().__init__(cfg=cfg, name="cuda", env_name=cfg.conda_env)


class RocmGaussianBackend(BaseGaussianBackend):
    def __init__(self, cfg: PipelineConfig) -> None:
        super().__init__(cfg=cfg, name="rocm", env_name=cfg.rocm_env)

    def validate(self) -> bool:
        if not super().validate():
            return False

        if not self.cfg.rocm_env:
            log_warn("ROCm backend selected but --rocm-env is empty.")
            return False

        if self.cfg.env_runner == "uv" and not self.cfg.uv_python:
            log_warn("ROCm + uv runner requires --uv-python <path>.")
            return False

        if not self.cfg.dry_run and not Path("/opt/rocm").exists():
            log_warn(
                "ROCm backend selected but /opt/rocm was not found.\n"
                "  Ensure ROCm runtime + ROCm PyTorch are installed.\n"
                "  Verify with: python -c \"import torch; print(torch.version.hip)\""
            )
            return False

        return True


def backend_for(cfg: PipelineConfig) -> BaseGaussianBackend:
    if cfg.gs_backend == "rocm":
        return RocmGaussianBackend(cfg)
    return CudaGaussianBackend(cfg)
