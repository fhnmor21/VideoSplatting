import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PipelineConfig


class TestSettingsThreads(unittest.TestCase):
    def test_cpu_threads_defaults_to_half_cores_when_high_core_count(self):
        cfg = PipelineConfig(
            video=Path("/tmp/in.mp4"),
            output_root=Path("/tmp/out"),
            gs_repo=Path("/tmp/repo"),
        )
        with patch("config.settings.os.cpu_count", return_value=64):
            self.assertEqual(cfg.cpu_threads, 32)

    def test_cpu_threads_respects_explicit_colmap_threads(self):
        cfg = PipelineConfig(
            video=Path("/tmp/in.mp4"),
            output_root=Path("/tmp/out"),
            gs_repo=Path("/tmp/repo"),
            colmap_threads=12,
        )
        with patch("config.settings.os.cpu_count", return_value=64):
            self.assertEqual(cfg.cpu_threads, 12)


if __name__ == "__main__":
    unittest.main()
