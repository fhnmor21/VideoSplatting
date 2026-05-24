import unittest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PipelineConfig
from pipeline.gaussian_backends import backend_for


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


if __name__ == "__main__":
    unittest.main()
