import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PipelineConfig
from pipeline.stage_extract import FrameExtractor


class TestFrameExtractionFilter(unittest.TestCase):
    def test_select_filter_bootstraps_first_frame_when_prev_selected_t_is_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = PipelineConfig(
                video=Path("/tmp/in.mp4"),
                output_root=root / "out",
                gs_repo=root / "repo",
            )
            cmd = FrameExtractor(cfg)._build_command(cfg.frames_dir)

        vf_arg = cmd[cmd.index("-vf") + 1]
        self.assertIn("isnan(prev_selected_t)", vf_arg)


if __name__ == "__main__":
    unittest.main()
