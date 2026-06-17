import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PipelineConfig
from pipeline.stage_colmap import ColmapReconstructor


class TestColmapCommands(unittest.TestCase):
    def test_feature_extractor_uses_new_colmap_feature_extraction_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            (frames / "frame_000001.jpg").write_bytes(b"x")
            cfg = PipelineConfig(
                video=Path("/tmp/in.mp4"),
                output_root=root / "out",
                gs_repo=root / "repo",
            )
            recon = ColmapReconstructor(cfg)

            with patch("stage_colmap.run") as run_mock:
                with patch("stage_colmap.log_success"), patch("stage_colmap.log_warn"), patch(
                    "stage_colmap.log_info"
                ):
                    run_mock.return_value = None
                    recon._feature_extract(root / "db.db", frames, "")

        cmd = run_mock.call_args.args[0]
        self.assertIn("--FeatureExtraction.use_gpu", cmd)
        self.assertIn("--FeatureExtraction.gpu_index", cmd)
        self.assertNotIn("--SiftExtraction.use_gpu", cmd)

    def test_sequential_matcher_uses_new_colmap_feature_matching_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = PipelineConfig(
                video=Path("/tmp/in.mp4"),
                output_root=root / "out",
                gs_repo=root / "repo",
            )
            recon = ColmapReconstructor(cfg)

            with patch("stage_colmap.run") as run_mock:
                with patch("stage_colmap.log_success"), patch("stage_colmap.log_warn"), patch(
                    "stage_colmap.log_info"
                ):
                    run_mock.return_value = None
                    recon._sequential_match(root / "db.db")

        cmd = run_mock.call_args.args[0]
        self.assertIn("--FeatureMatching.use_gpu", cmd)
        self.assertIn("--FeatureMatching.gpu_index", cmd)
        self.assertNotIn("--SiftMatching.use_gpu", cmd)
        self.assertEqual(cmd[cmd.index("--FeatureMatching.use_gpu") + 1], "1")

    def test_feature_extractor_disables_gpu_when_colmap_gpu_is_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            (frames / "frame_000001.jpg").write_bytes(b"x")
            cfg = PipelineConfig(
                video=Path("/tmp/in.mp4"),
                output_root=root / "out",
                gs_repo=root / "repo",
                colmap_gpu=-1,
            )
            recon = ColmapReconstructor(cfg)

            with patch("stage_colmap.run") as run_mock:
                with patch("stage_colmap.log_success"), patch("stage_colmap.log_warn"), patch(
                    "stage_colmap.log_info"
                ):
                    run_mock.return_value = None
                    recon._feature_extract(root / "db.db", frames, "")

        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--FeatureExtraction.use_gpu") + 1], "0")

    def test_mapper_uses_current_colmap_ba_global_frames_ratio_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = PipelineConfig(
                video=Path("/tmp/in.mp4"),
                output_root=root / "out",
                gs_repo=root / "repo",
            )
            recon = ColmapReconstructor(cfg)

            with patch("stage_colmap.run") as run_mock:
                with patch("stage_colmap.log_success"), patch("stage_colmap.log_warn"), patch(
                    "stage_colmap.log_info"
                ):
                    run_mock.return_value = None
                    recon._mapper(root / "db.db", root / "frames")

        cmd = run_mock.call_args.args[0]
        self.assertIn("--Mapper.ba_global_frames_ratio", cmd)
        self.assertNotIn("--Mapper.ba_global_images_ratio", cmd)


if __name__ == "__main__":
    unittest.main()
