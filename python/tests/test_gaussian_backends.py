import unittest
import sys
import tempfile
from pathlib import Path
import shutil
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PipelineConfig
from pipeline.gaussian_backends import backend_for
import pipeline.gaussian_backends as gaussian_backends
import pipeline.stage_gaussian as stage_gaussian
from pipeline.stage_gaussian import GaussianTrainer


class TestGaussianBackends(unittest.TestCase):
    def setUp(self) -> None:
        self._temps = []

    def tearDown(self) -> None:
        for tmp in self._temps:
            tmp.cleanup()

    def make_cfg(self, backend: str) -> PipelineConfig:
        tmp = tempfile.TemporaryDirectory()
        self._temps.append(tmp)
        repo = Path(tmp.name) / "gs_repo"
        repo.mkdir(parents=True, exist_ok=True)
        (repo / "train.py").write_text("print('train')\n", encoding="utf-8")
        (repo / "render.py").write_text("print('render')\n", encoding="utf-8")
        (repo / "metrics.py").write_text("print('metrics')\n", encoding="utf-8")

        return PipelineConfig(
            video=Path("/tmp/in.mp4"),
            output_root=Path("/tmp/out"),
            gs_repo=repo,
            gs_backend=backend,
            dry_run=True,
        )

    def test_backend_factory_returns_rocm_backend(self):
        cfg = self.make_cfg("rocm")
        backend = backend_for(cfg)
        self.assertEqual(backend.name, "rocm")

    def test_backend_factory_returns_cuda_adapter_by_default(self):
        cfg = self.make_cfg("cuda")
        backend = backend_for(cfg)
        self.assertEqual(backend.name, "cuda")
        self.assertTrue(hasattr(backend, "train"))
        self.assertTrue(hasattr(backend, "finalize_outputs"))

    def test_cuda_train_command_uses_train_script(self):
        cfg = self.make_cfg("cuda")
        backend = backend_for(cfg)
        cmd = backend.build_train_cmd()
        self.assertIn("train.py", " ".join(cmd))

    def test_rocm_backend_requires_rocm_env_name(self):
        cfg = self.make_cfg("rocm")
        cfg.rocm_env = ""
        backend = backend_for(cfg)
        self.assertFalse(backend.validate())

    def test_uv_runner_requires_python_path(self):
        cfg = self.make_cfg("rocm")
        cfg.env_runner = "uv"
        cfg.uv_python = ""
        backend = backend_for(cfg)
        self.assertFalse(backend.validate())

    def test_default_runner_is_conda(self):
        cfg = self.make_cfg("cuda")
        self.assertEqual(cfg.env_runner, "conda")

    def test_rocm_backend_builds_train_command_shape(self):
        cfg = self.make_cfg("rocm")
        backend = backend_for(cfg)
        cmd = backend.build_train_cmd()
        self.assertEqual(cmd[:2], ["python", str(cfg.gs_repo / cfg.gsplat_train_script)])
        self.assertEqual(cmd[2], "default")
        self.assertIn("--data-dir", cmd)
        self.assertIn(str(cfg.colmap_dense), cmd)
        self.assertIn("--data-factor", cmd)
        self.assertIn("1", cmd)
        self.assertIn("--result-dir", cmd)
        self.assertIn(str(cfg.rocm_backend_output_dir), cmd)
        self.assertIn("--max-steps", cmd)
        self.assertIn(str(cfg.iterations), cmd)
        self.assertIn("--disable-viewer", cmd)

    def test_cuda_finalize_outputs_keeps_existing_layout(self):
        cfg = self.make_cfg("cuda")
        backend = backend_for(cfg)
        cfg.gs_output.mkdir(parents=True, exist_ok=True)
        target = cfg.final_ply
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ply\n", encoding="utf-8")
        self.assertTrue(backend.finalize_outputs())

    def test_stage3_train_fails_when_finalize_missing_final_ply(self):
        cfg = self.make_cfg("rocm")
        cfg.env_runner = "uv"
        cfg.uv_python = "/usr/bin/python3"
        trainer = GaussianTrainer(cfg)
        trainer.backend.train = lambda runner: True
        trainer.backend.finalize_outputs = lambda: False
        self.assertFalse(trainer._train())

    def test_stage3_exec_sets_rocm_override_for_uv_runner(self):
        cfg = self.make_cfg("rocm")
        cfg.env_runner = "uv"
        cfg.uv_python = "/usr/bin/python3"
        trainer = GaussianTrainer(cfg)

        captured = {}

        original = GaussianTrainer._exec.__globals__["run_in_uv"]

        def fake_run_in_uv(python_bin, args, *, dry_run=False, cwd=None, env=None):
            captured["python_bin"] = python_bin
            captured["args"] = list(args)
            captured["dry_run"] = dry_run
            captured["cwd"] = cwd
            captured["env"] = env

        try:
            GaussianTrainer._exec.__globals__["run_in_uv"] = fake_run_in_uv
            trainer._exec(["python", "-c", "print('ok')"])
        finally:
            GaussianTrainer._exec.__globals__["run_in_uv"] = original

        self.assertEqual(captured.get("python_bin"), cfg.uv_python)
        self.assertEqual(captured.get("args"), ["python", "-c", "print('ok')"])
        self.assertEqual(captured.get("env"), {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"})

    def test_rocm_finalize_maps_backend_ply_to_canonical_path(self):
        cfg = self.make_cfg("rocm")
        backend = backend_for(cfg)
        fixture = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "rocm_backend_output"
            / "point_cloud.ply"
        )
        out = cfg.rocm_backend_output_dir / "point_cloud.ply"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixture, out)
        self.assertTrue(backend.finalize_outputs())
        self.assertTrue(cfg.final_ply.exists())

    def test_rocm_finalize_copies_latest_trainer_ply_from_ply_dir(self):
        cfg = self.make_cfg("rocm")
        backend = backend_for(cfg)
        backend_ply = cfg.rocm_backend_output_dir / "ply" / "point_cloud_29999.ply"
        backend_ply.parent.mkdir(parents=True, exist_ok=True)
        backend_ply.write_text("ply\n", encoding="utf-8")

        self.assertTrue(backend.finalize_outputs())
        self.assertTrue(cfg.final_ply.exists())
        self.assertEqual(cfg.final_ply.read_text(encoding="utf-8"), "ply\n")

    def test_rocm_uv_validate_runs_runtime_rasterization_probe(self):
        cfg = self.make_cfg("rocm")
        cfg.dry_run = False
        cfg.env_runner = "uv"
        cfg.uv_python = "/usr/bin/python3"
        train_script = cfg.gs_repo / cfg.gsplat_train_script
        train_script.parent.mkdir(parents=True, exist_ok=True)
        train_script.write_text("print('trainer')\n", encoding="utf-8")
        backend = backend_for(cfg)

        calls = []
        original = gaussian_backends.subprocess.run

        def fake_run(args, **kwargs):
            calls.append((list(args), kwargs))
            if "torch.version.hip" in args[2]:
                return SimpleNamespace(returncode=0, stdout="7.0\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        try:
            gaussian_backends.subprocess.run = fake_run
            self.assertTrue(backend.validate())
        finally:
            gaussian_backends.subprocess.run = original

        joined = "\n".join(" ".join(cmd) for cmd, _ in calls)
        self.assertIn("import torch; print(torch.version.hip)", joined)
        self.assertIn("import gsplat", joined)
        self.assertIn("from gsplat.rendering import rasterization", joined)


if __name__ == "__main__":
    unittest.main()
