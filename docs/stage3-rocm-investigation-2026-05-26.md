# Stage 3 ROCm Investigation Report (2026-05-26)

## Objective

Get Stage 3 (Gaussian training/rasterization path) working end-to-end for:

- Input: `video/test.mp4`
- Backend: `--gs-backend rocm`
- Resume flow: `--resume`
- Runtime target: AMD RX 7600 (`gfx1102`)

## Environment and Execution Context

- Project root: `/var/home/bazzite/Data/Develop/VideoSplatting`
- Active ROCm env: `.venv-rocm7` (`torch 2.10.0+rocm7.0`)
- PATH expectations during pipeline runs:
  - `PATH=/home/bazzite/.local/bin:$PATH`
  - `LD_LIBRARY_PATH=/home/bazzite/.local/lib:$LD_LIBRARY_PATH`
- ROCm test override used during probes:
  - `HSA_OVERRIDE_GFX_VERSION=11.0.0`

## Scope of Attempts

Work focused on removing blockers in ROCm gsplat source builds and validating runtime kernel execution.

Primary patch work was done in temporary clones:

- `/tmp/opencode/rocm-gsplat-b1` (branch `release/1.5.3b1`)
- `/tmp/opencode/rocm-gsplat-main` (branch `main`)

## Chronological Attempt Log

### 1) Baseline Stage 3 and backend integration state

Existing pipeline-side changes remained in place and were not rolled back:

- `python/stage_extract.py`
- `python/stage_colmap.py`
- `python/gaussian_backends.py`
- related tests

These were considered stable enough to proceed; the active blockers were in gsplat ROCm build/runtime.

### 2) Initial source build failures: missing GLM headers in HIP path

Observed errors in hipified compile path for GLM includes (`glm/gtc/type_ptr.hpp` and related `.inl`/detail files).

Actions:

- Patched `setup.py` in both b1 and main clones to include GLM path:
  - `gsplat/cuda/csrc/third_party/glm`
- Added synchronization logic to copy GLM tree from CUDA third-party path into HIP path before build.

Outcome:

- Missing-header class of failures was reduced/eliminated, but additional compile-time GLM logic failures surfaced.

### 3) GLM compiler/platform detection conflicts under HIP

Observed failures consistent with wrong preprocessor path selection for HIP builds (CUDA-oriented branch precedence).

Actions:

- Added GLM platform-branch patching in copied HIP-side headers (notably `glm/simd/platform.h`).
- Adjusted GLM/HIP macro handling in build flags.

Outcome:

- Compile advanced beyond prior GLM platform checks.

### 4) Hardcoded architecture mismatch (`gfx942`) vs RX 7600 (`gfx1102`)

Build system contained hardcoded offload arch settings incompatible with the target card.

Actions:

- Replaced hardcoded `--offload-arch=gfx942` with env-driven architecture from `PYTORCH_ROCM_ARCH`.
- Standardized builds with `PYTORCH_ROCM_ARCH=gfx1102`.

Outcome:

- Architecture mismatch blocker removed.

### 5) Device compilation issue in camera code (`std::array::at`)

Observed HIP/device compilation errors in `Cameras.cuh` due to `.at()` usage in device code.

Actions:

- Replaced `.at()` indexing with `[]` in b1 camera code path.

Outcome:

- This compile blocker cleared.

### 6) rocPRIM wavefront/static-assert failures

Observed repeated static assertions indicating wavefront/warp-size assumptions incompatible with target/runtime behavior.

Actions:

- Patched many ROCm-specific raster/utility paths from 64-lane assumptions to 32-lane logic.
- Updated related helper logic in raster source and utility include files.

Outcome:

- Build progressed further but introduced secondary symbol/instantiation issues.

### 7) Template instantiation breakage after bulk wave edits

Two related issues were hit:

1. duplicate explicit instantiations (`__INS__(32)`) after bulk substitutions
2. later, missing `<64U>` launcher symbol at import/runtime linkage

Actions:

- Removed duplicate explicit instantiations.
- Re-added/restored needed `__INS__(64)` entries for raster source where missing.

Outcome:

- Rebuild reached successful install state.

### 8) Successful editable build/install in ROCm7 env

Command that succeeded:

```bash
PYTORCH_ROCM_ARCH=gfx1102 .venv-rocm7/bin/python -m pip install --no-build-isolation -e /tmp/opencode/rocm-gsplat-b1
```

Outcome:

- Editable wheel built and installed (`amd_gsplat-1.0.0+28e641b`).

### 9) Post-build import validation

Checks performed:

- `import gsplat.csrc` succeeded.
- Symbol presence check succeeded (`CameraModelType` available).

Outcome:

- Import/link stage largely healthy after symbol restoration.

### 10) Runtime kernel probe failures (hard blocker)

Minimal GPU rasterization probe using `gsplat.rendering.rasterization(...)` did not complete successfully.

Observed behavior:

- Direct inline probe calls hung until timeout in some runs.
- Timed subprocess probe (`/tmp/opencode/raster_probe.py`) exited with `-11` (segmentation fault).

Outcome:

- Compile/import success did not translate to stable runtime kernel execution.
- This remains the current Stage 3 blocker.

## Key Errors and Signals Collected

- Missing GLM includes/files in HIP tree.
- GLM platform/compiler macro conflicts under HIP.
- Device compile incompatibility from `.at()` usage.
- rocPRIM wavefront static assertions.
- Duplicate/missing kernel template instantiations (`32/64` variants).
- Runtime segmentation fault (`exitcode -11`) in rasterization probe.

## Why Stage 3 Is Still Blocked

Stage 3 depends on reliable rasterization kernel execution. Current state is:

- Build: passing
- Extension import: passing
- Runtime rasterization kernel execution: failing (hang/segfault)

Because the minimal rasterization probe fails, full Stage 3 runs were intentionally not retried as a success path.

## Files/Areas Most Impacted During Investigation

- `/tmp/opencode/rocm-gsplat-b1/setup.py`
- `/tmp/opencode/rocm-gsplat-main/setup.py`
- `/tmp/opencode/rocm-gsplat-b1/gsplat/cuda/include/Cameras.cuh`
- `/tmp/opencode/rocm-gsplat-b1/gsplat/cuda/include/Utils.cuh`
- `/tmp/opencode/rocm-gsplat-b1/gsplat/cuda/csrc/RasterizeToPixels*.cu`
- `/tmp/opencode/rocm-gsplat-main/gsplat/cuda/csrc/RasterizeToPixels*.cu`
- `/tmp/opencode/rocm-gsplat-b1/gsplat/cuda/_backend.py` (observed `_C` fallback path behavior earlier)

## Commands Used for Latest Verification

Build/install:

```bash
PYTORCH_ROCM_ARCH=gfx1102 /home/bazzite/Data/Develop/VideoSplatting/.venv-rocm7/bin/python -m pip install --no-build-isolation -e /tmp/opencode/rocm-gsplat-b1
```

Import check:

```bash
/home/bazzite/Data/Develop/VideoSplatting/.venv-rocm7/bin/python -c "import gsplat.csrc as c; print('imported', hasattr(c,'CameraModelType'))"
```

Runtime probe (failing):

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 /home/bazzite/Data/Develop/VideoSplatting/.venv-rocm7/bin/python /tmp/opencode/raster_probe.py
```

## Final Status (as of 2026-05-26)

Stage 3 ROCm path is **not yet operational** on this stack for RX 7600 because rasterization runtime fails after successful source build and import.

Investigation advanced the state from "cannot build" to "builds and imports but runtime crashes," which narrows future work to kernel/runtime compatibility rather than packaging/include issues.
