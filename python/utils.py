"""
pipeline/utils.py — Shared utilities used by all pipeline stages.

Covers:
  * Structured console logging with timestamps and stage headers
  * Command execution (with dry-run support and live stdout streaming)
  * ffprobe helpers for video/image metadata
  * Conda environment activation for subprocess calls
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"


def _ts() -> str:
    """Return current local time formatted for log prefixes."""
    return time.strftime("%H:%M:%S")


def log_header(title: str) -> None:
    """Print a prominent section header for stage-level logging."""
    bar = "─" * 60
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{CYAN}{bar}{RESET}")


def log_info(msg: str) -> None:
    """Print an informational log line with timestamp."""
    print(f"{DIM}[{_ts()}]{RESET} {msg}")


def log_success(msg: str) -> None:
    """Print a success log line with timestamp and checkmark."""
    print(f"{DIM}[{_ts()}]{RESET} {GREEN}✓{RESET} {msg}")


def log_warn(msg: str) -> None:
    """Print a warning log line to stderr."""
    print(f"{DIM}[{_ts()}]{RESET} {YELLOW}⚠ {msg}{RESET}", file=sys.stderr)


def log_error(msg: str) -> None:
    """Print an error log line to stderr."""
    print(f"{DIM}[{_ts()}]{RESET} {RED}✗ {msg}{RESET}", file=sys.stderr)


def log_cmd(args: Sequence) -> None:
    """Print a shell-like representation of a command being executed."""
    joined = " \\\n    ".join(str(a) for a in args)
    print(f"{DIM}$ {joined}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Command execution
# ─────────────────────────────────────────────────────────────────────────────


class CommandError(RuntimeError):
    """Raised when a subprocess exits with a non-zero return code."""


def run(
    args: Sequence,
    *,
    dry_run: bool = False,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a command, streaming its output live to the terminal.

    Args:
        args:     Command and arguments (list).
        dry_run:  If True, print the command and return a fake result.
        cwd:      Working directory.
        env:      Environment variables (merged with os.environ if provided).
        check:    Raise CommandError on non-zero exit.
        capture:  If True, capture stdout/stderr and return them in result.

    Returns:
        subprocess.CompletedProcess
    """
    str_args = [str(a) for a in args]
    log_cmd(str_args)

    if dry_run:
        return subprocess.CompletedProcess(str_args, 0, stdout="", stderr="")

    merged_env = None
    if env:
        merged_env = {**os.environ, **env}

    if capture:
        result = subprocess.run(
            str_args,
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    else:
        # Stream output live
        result = subprocess.run(str_args, cwd=cwd, env=merged_env)

    if check and result.returncode != 0:
        raise CommandError(
            f"Command exited with code {result.returncode}:\n  {' '.join(str_args)}"
        )
    return result


def run_in_conda(
    conda_sh: Path,
    env_name: str,
    args: Sequence,
    *,
    dry_run: bool = False,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """
    Run a command inside a conda environment by sourcing conda.sh first.
    Uses bash -c so that 'conda activate' works in a non-interactive shell.
    """
    str_args = [str(a) for a in args]
    cmd_str = " ".join(str_args)
    bash_cmd = f'source "{conda_sh}" && conda activate "{env_name}" && {cmd_str}'
    return run(
        ["bash", "-c", bash_cmd],
        dry_run=dry_run,
        cwd=cwd,
    )


def find_conda_sh() -> Optional[Path]:
    """Locate the conda.sh init script across common installation prefixes."""
    candidates = [
        Path.home() / "miniconda3/etc/profile.d/conda.sh",
        Path.home() / "anaconda3/etc/profile.d/conda.sh",
        Path("/opt/conda/etc/profile.d/conda.sh"),
        Path("/usr/local/conda/etc/profile.d/conda.sh"),
    ]

    # Also derive from `conda info --base` if conda is in PATH
    if shutil.which("conda"):
        try:
            result = subprocess.run(
                ["conda", "info", "--base"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            base = result.stdout.strip()
            if base:
                candidates.insert(0, Path(base) / "etc/profile.d/conda.sh")
        except Exception:
            pass

    for c in candidates:
        if c.exists():
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ffprobe helpers
# ─────────────────────────────────────────────────────────────────────────────


def probe_video(video: Path) -> dict:
    """
    Return a dict with keys: duration, fps, width, height, frame_count.
    Falls back gracefully to None for any value that can't be determined.
    """
    if not shutil.which("ffprobe"):
        return {}
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(video),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(result.stdout)
    except Exception:
        return {}

    info: dict = {}
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            info["width"] = stream.get("width")
            info["height"] = stream.get("height")
            # fps as fraction e.g. "30000/1001"
            fps_str = stream.get("r_frame_rate", "")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                try:
                    info["fps"] = float(num) / float(den)
                except ZeroDivisionError:
                    info["fps"] = None
            info["frame_count"] = stream.get("nb_frames")
            break
    try:
        info["duration"] = float(data.get("format", {}).get("duration", 0)) or None
    except (TypeError, ValueError):
        info["duration"] = None
    return info


def probe_image(image: Path) -> Tuple[Optional[int], Optional[int]]:
    """Return (width, height) of an image via ffprobe, or (None, None)."""
    if not shutil.which("ffprobe"):
        return None, None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(image),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        parts = result.stdout.strip().split(",")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# COLMAP binary helpers
# ─────────────────────────────────────────────────────────────────────────────


def count_registered_images(model_dir: Path) -> Optional[int]:
    """
    Count how many images were registered in a COLMAP sparse model.
    Reads images.bin (binary) or images.txt (text), whichever exists.
    """
    bin_file = model_dir / "images.bin"
    txt_file = model_dir / "images.txt"

    if bin_file.exists():
        try:
            with open(bin_file, "rb") as f:
                (count,) = struct.unpack("<Q", f.read(8))
            return count
        except Exception:
            pass

    if txt_file.exists():
        try:
            lines = [
                l
                for l in txt_file.read_text().splitlines()
                if l.strip() and not l.startswith("#")
            ]
            # images.txt has 2 lines per image (header + points)
            return len(lines) // 2
        except Exception:
            pass

    return None


def count_ply_points(ply_path: Path) -> Optional[int]:
    """Read the vertex count from a PLY file header."""
    try:
        with open(ply_path, "rb") as f:
            for raw_line in f:
                line = raw_line.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except Exception:
        pass
    return None


def check_tool(name: str, install_hint: str = "") -> bool:
    """Return True if `name` is found in PATH, otherwise log an error."""
    if shutil.which(name):
        return True
    msg = f"'{name}' not found in PATH."
    if install_hint:
        msg += f"\n  {install_hint}"
    log_error(msg)
    return False


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as a compact human-readable duration string."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
