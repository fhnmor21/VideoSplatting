import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


class TestCliBackendFlags(unittest.TestCase):
    def test_default_backend_and_runner_are_cuda_conda(self):
        with patch("sys.argv", ["main.py", "input.mp4"]):
            args = main.parse_args()
        self.assertEqual(args.gs_backend, "cuda")
        self.assertEqual(args.env_runner, "conda")

    def test_default_backend_is_cuda(self):
        with patch("sys.argv", ["main.py", "input.mp4"]):
            args = main.parse_args()
        self.assertEqual(args.gs_backend, "cuda")

    def test_rocm_backend_and_env_flags_parse(self):
        with patch(
            "sys.argv",
            [
                "main.py",
                "input.mp4",
                "--gs-backend",
                "rocm",
                "--rocm-env",
                "gaussian_splatting_rocm",
            ],
        ):
            args = main.parse_args()
        self.assertEqual(args.gs_backend, "rocm")
        self.assertEqual(args.rocm_env, "gaussian_splatting_rocm")

    def test_uv_runner_flags_parse(self):
        with patch(
            "sys.argv",
            [
                "main.py",
                "input.mp4",
                "--env-runner",
                "uv",
                "--uv-python",
                "/opt/venvs/gs/bin/python",
            ],
        ):
            args = main.parse_args()
        self.assertEqual(args.env_runner, "uv")
        self.assertEqual(args.uv_python, "/opt/venvs/gs/bin/python")


if __name__ == "__main__":
    unittest.main()
