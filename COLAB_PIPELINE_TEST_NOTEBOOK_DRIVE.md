# Google Colab Notebook (Google Drive): Test the VideoSplatting Pipeline

This variant runs the pipeline using Google Drive paths so your input/output survives Colab runtime resets.

Use each code block as a separate Jupyter cell, in order.

## Cell 1 - Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 2 - Set Your Paths

```python
from pathlib import Path

# Edit these to match your Drive layout
DRIVE_ROOT = Path('/content/drive/MyDrive')
WORK_ROOT = DRIVE_ROOT / 'videosplatting_colab'
REPO_DIR = WORK_ROOT / 'VideoSplatting'
GS_REPO_DIR = WORK_ROOT / 'gaussian-splatting'
INPUT_VIDEO = WORK_ROOT / 'input' / 'building.mp4'  # put your video here
OUT_DIR = WORK_ROOT / 'outputs' / 'gs_pipeline_output'

for p in [WORK_ROOT, WORK_ROOT / 'input', WORK_ROOT / 'outputs']:
    p.mkdir(parents=True, exist_ok=True)

print('WORK_ROOT:', WORK_ROOT)
print('INPUT_VIDEO expected at:', INPUT_VIDEO)
```

## Cell 3 - Optional: Verify GPU Runtime

```python
!nvidia-smi
```

## Cell 4 - Install System Dependencies

```bash
%%bash
set -euo pipefail

apt-get update
apt-get install -y ffmpeg colmap git wget

ffmpeg -version | head -n 1
colmap -h >/dev/null && echo "COLMAP installed"
```

## Cell 5 - Clone Repositories to Drive

```python
import subprocess

repo_url = 'https://github.com/<YOUR_USER_OR_ORG>/VideoSplatting.git'

if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', repo_url, str(REPO_DIR)], check=True)

if not GS_REPO_DIR.exists():
    subprocess.run([
        'git', 'clone', '--recursive',
        'https://github.com/graphdeco-inria/gaussian-splatting.git',
        str(GS_REPO_DIR)
    ], check=True)

print('Pipeline repo:', REPO_DIR)
print('GS repo:', GS_REPO_DIR)
```

## Cell 6 - Install Miniconda and Build gaussian_splatting Env

```bash
%%bash
set -euo pipefail

if [ ! -d /usr/local/miniconda ]; then
  wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p /usr/local/miniconda
fi

source /usr/local/miniconda/etc/profile.d/conda.sh
conda config --set always_yes yes --set changeps1 no

if ! conda env list | grep -q "gaussian_splatting"; then
  conda env create -f /content/drive/MyDrive/videosplatting_colab/gaussian-splatting/environment.yml
fi

conda activate gaussian_splatting
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('CUDA extensions OK')"
```

## Cell 7 - Upload Video into Drive Input Folder (If Needed)

```python
from google.colab import files
import shutil

uploaded = files.upload()
if uploaded:
    local_name = next(iter(uploaded.keys()))
    target = INPUT_VIDEO
    shutil.move(local_name, str(target))
    print('Moved uploaded file to:', target)
else:
    print('No upload this time.')

if not INPUT_VIDEO.exists():
    raise FileNotFoundError(f'Missing input video: {INPUT_VIDEO}')
```

## Cell 8 - Optional Vocab Tree Download (Saved to Drive)

```python
import subprocess

VOCAB_TREE = WORK_ROOT / 'vocab_tree_flickr100K_words256K.bin'
if not VOCAB_TREE.exists():
    subprocess.run([
        'wget',
        '-O', str(VOCAB_TREE),
        'https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin'
    ], check=True)

print('Vocab tree:', VOCAB_TREE)
```

## Cell 9 - Run Fast Smoke Test Pipeline

```bash
%%bash
set -euo pipefail

source /usr/local/miniconda/etc/profile.d/conda.sh

VIDEO_FILE="/content/drive/MyDrive/videosplatting_colab/input/building.mp4"
OUT_DIR="/content/drive/MyDrive/videosplatting_colab/outputs/gs_pipeline_output"
PIPELINE_DIR="/content/drive/MyDrive/videosplatting_colab/VideoSplatting/python"
GS_REPO="/content/drive/MyDrive/videosplatting_colab/gaussian-splatting"

cd "${PIPELINE_DIR}"

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

## Cell 10 - Optional Full-Quality Run

```bash
%%bash
set -euo pipefail

source /usr/local/miniconda/etc/profile.d/conda.sh

VIDEO_FILE="/content/drive/MyDrive/videosplatting_colab/input/building.mp4"
OUT_DIR="/content/drive/MyDrive/videosplatting_colab/outputs/gs_pipeline_full"
PIPELINE_DIR="/content/drive/MyDrive/videosplatting_colab/VideoSplatting/python"
GS_REPO="/content/drive/MyDrive/videosplatting_colab/gaussian-splatting"
VOCAB_TREE="/content/drive/MyDrive/videosplatting_colab/vocab_tree_flickr100K_words256K.bin"

cd "${PIPELINE_DIR}"

python main.py "${VIDEO_FILE}" \
  -o "${OUT_DIR}" \
  --gs-repo "${GS_REPO}" \
  --conda-env gaussian_splatting \
  --vocab-tree "${VOCAB_TREE}" \
  --iterations 30000 \
  --densify-start 500 \
  --densify-end 15000 \
  --resolution-cap 1600 \
  --checkpoint-interval 7000 \
  --test-holdout 8
```

## Cell 11 - Inspect Output Artifacts

```python
from pathlib import Path

pc_root = Path('/content/drive/MyDrive/videosplatting_colab/outputs/gs_pipeline_output/gaussian/point_cloud')
ply_files = sorted(pc_root.glob('iteration_*/point_cloud.ply'))

print('PLY checkpoints found:', len(ply_files))
for p in ply_files:
    print('-', p)

if not ply_files:
    raise RuntimeError('No point_cloud.ply found. Review Stage 1-3 logs above.')
```

## Cell 12 - Optional: Resume an Interrupted Run

```bash
%%bash
set -euo pipefail

source /usr/local/miniconda/etc/profile.d/conda.sh

VIDEO_FILE="/content/drive/MyDrive/videosplatting_colab/input/building.mp4"
OUT_DIR="/content/drive/MyDrive/videosplatting_colab/outputs/gs_pipeline_output"
PIPELINE_DIR="/content/drive/MyDrive/videosplatting_colab/VideoSplatting/python"
GS_REPO="/content/drive/MyDrive/videosplatting_colab/gaussian-splatting"

cd "${PIPELINE_DIR}"

python main.py "${VIDEO_FILE}" \
  -o "${OUT_DIR}" \
  --gs-repo "${GS_REPO}" \
  --conda-env gaussian_splatting \
  --resume
```

---

## Notes

- Replace `https://github.com/<YOUR_USER_OR_ORG>/VideoSplatting.git` with your real repo URL.
- The pipeline entrypoint used here is `VideoSplatting/python/main.py`.
- Drive I/O is slower than local `/content`; this setup favors persistence over speed.
