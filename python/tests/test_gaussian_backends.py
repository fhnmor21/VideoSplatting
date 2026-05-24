import unittest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PipelineConfig
from pipeline.gaussian_backends import backend_for
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
        self.assertIn("--source_path", cmd)
        self.assertIn("--model_path", cmd)

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


if __name__ == "__main__":
    unittest.main()
