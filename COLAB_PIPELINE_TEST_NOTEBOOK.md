# Google Colab Notebook: Test the VideoSplatting Pipeline

This document gives you a cell-by-cell notebook script to test the current `gs_pipeline` implementation in Colab.

Use each code block as a separate Jupyter cell, in order.

## Cell 1 - Optional: Verify GPU Runtime

```python
!nvidia-smi
```

## Cell 2 - Install System Dependencies (FFmpeg + COLMAP)

```bash
%%bash
set -euo pipefail

apt-get update
apt-get install -y ffmpeg colmap git wget

echo "Installed versions:"
ffmpeg -version | head -n 1
colmap -h >/dev/null && echo "COLMAP installed"
```

## Cell 3 - Clone This Pipeline Repository

```bash
%%bash
set -euo pipefail

if [ ! -d /content/VideoSplatting ]; then
  git clone https://github.com/<YOUR_USER_OR_ORG>/VideoSplatting.git /content/VideoSplatting
fi

ls -la /content/VideoSplatting
```

## Cell 4 - Clone gaussian-splatting and Create Conda Env

```bash
%%bash
set -euo pipefail

cd /content

if [ ! -d gaussian-splatting ]; then
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
fi

if [ ! -d /usr/local/miniconda ]; then
  wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p /usr/local/miniconda
fi

source /usr/local/miniconda/etc/profile.d/conda.sh
conda config --set always_yes yes --set changeps1 no

if ! conda env list | grep -q "gaussian_splatting"; then
  conda env create -f /content/gaussian-splatting/environment.yml
fi

conda activate gaussian_splatting
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('CUDA extensions OK')"
```

## Cell 5 - Upload Your Input Video

```python
from google.colab import files

uploaded = files.upload()
video_name = next(iter(uploaded.keys()))
print("Uploaded:", video_name)
```

## Cell 6 - (Optional) Download COLMAP Vocab Tree for Better Loop Closure

```bash
%%bash
set -euo pipefail

if [ ! -f /content/vocab_tree_flickr100K_words256K.bin ]; then
  wget -O /content/vocab_tree_flickr100K_words256K.bin https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin
fi

ls -lh /content/vocab_tree_flickr100K_words256K.bin
```

## Cell 7 - Run a Fast Pipeline Smoke Test

This uses lower iterations so you can verify the full pipeline quickly.

```bash
%%bash
set -euo pipefail

source /usr/local/miniconda/etc/profile.d/conda.sh

cd /content/VideoSplatting/python

VIDEO_FILE="/content/${video_name}"
OUT_DIR="/content/gs_pipeline_test_output"
GS_REPO="/content/gaussian-splatting"

python main.py "${VIDEO_FILE}" \
  -o "${OUT_DIR}" \
  --gs-repo "${GS_REPO}" \
  --conda-env gaussian_splatting \
  --scene-threshold 0.35 \
  --min-gap 0.5 \
  --max-gap 3.0 \
  --iterations 3000 \
  --densify-start 200 \
  --densify-end 1500 \
  --resolution-cap 1024 \
  --checkpoint-interval 1000 \
  --test-holdout 8
```

## Cell 8 - Run a Higher-Quality Pass (Optional)

Use this only after the smoke test succeeds.

```bash
%%bash
set -euo pipefail

source /usr/local/miniconda/etc/profile.d/conda.sh

cd /content/VideoSplatting/python

VIDEO_FILE="/content/${video_name}"
OUT_DIR="/content/gs_pipeline_full_output"
GS_REPO="/content/gaussian-splatting"

python main.py "${VIDEO_FILE}" \
  -o "${OUT_DIR}" \
  --gs-repo "${GS_REPO}" \
  --conda-env gaussian_splatting \
  --vocab-tree /content/vocab_tree_flickr100K_words256K.bin \
  --iterations 30000 \
  --densify-start 500 \
  --densify-end 15000 \
  --resolution-cap 1600 \
  --checkpoint-interval 7000 \
  --test-holdout 8
```

## Cell 9 - Inspect Outputs

```bash
%%bash
set -euo pipefail

OUT_DIR="/content/gs_pipeline_test_output"

echo "Frames:"
ls -1 "${OUT_DIR}/frames" | wc -l

echo "Sparse model files:"
ls -la "${OUT_DIR}/colmap/sparse/0" || true

echo "Dense source layout:"
ls -la "${OUT_DIR}/colmap/dense" || true

echo "Gaussian output:"
find "${OUT_DIR}/gaussian" -maxdepth 4 -type f | sed 's|^|  |'
```

## Cell 10 - Confirm Final PLY Exists

```python
from pathlib import Path

out_dir = Path('/content/gs_pipeline_test_output/gaussian/point_cloud')
ply_files = sorted(out_dir.glob('iteration_*/point_cloud.ply'))

print('PLY checkpoints found:', len(ply_files))
for p in ply_files:
    print('-', p)

if not ply_files:
    raise RuntimeError('No point_cloud.ply found. Check previous cell logs for failures.')
```

## Cell 11 - Package Results for Download

```bash
%%bash
set -euo pipefail

cd /content
tar -czf gs_pipeline_test_output.tar.gz gs_pipeline_test_output
ls -lh gs_pipeline_test_output.tar.gz
```

```python
from google.colab import files
files.download('/content/gs_pipeline_test_output.tar.gz')
```

---

## Notes About the Current Implementation

- Entry point is `python/main.py` and orchestrates frame extraction, COLMAP, then 3DGS training.
- It expects `--gs-repo` to point at a valid clone of the Inria `gaussian-splatting` repo.
- Stage 3 runs `train.py`/`render.py`/`metrics.py` via `conda activate <env>`.
- If Colab memory is tight, lower `--resolution-cap` and `--iterations`.
- For interrupted runs, add `--resume` to reuse previously generated stage outputs.
