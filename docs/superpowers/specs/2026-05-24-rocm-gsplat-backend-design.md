# ROCm GSplat Backend Design

## Goal

Enable Stage 3 to run with a ROCm/GSplat backend as an alternative to the existing CUDA/Graphdeco backend while preserving the pipeline's canonical output contract, especially:

- `gaussian/point_cloud/iteration_<N>/point_cloud.ply`

## Scope

### In scope

- Backend-trainer abstraction for Stage 3.
- Two concrete trainer implementations:
  - CUDA Graphdeco trainer (existing behavior)
  - ROCm GSplat trainer (new behavior)
- Canonical output normalization so downstream pipeline behavior is unchanged.
- Keep work on branch `feat/gaussian-backend-rocm-uv`.
- Validation and docs updates for ROCm + `uv` usage.

### Out of scope

- Remote CUDA backend execution.
- Rewriting Stage 1 (extract) or Stage 2 (COLMAP).
- Guaranteeing every GSplat metric/render artifact exactly matches Graphdeco internals; only canonical pipeline outputs are guaranteed.

## Current State Summary

- Stage 3 currently assumes Graphdeco scripts (`train.py`, `render.py`, `metrics.py`).
- Existing branch already introduced backend and env-runner flags (`--gs-backend`, `--env-runner`, `--uv-python`) and adapter scaffolding.
- ROCm + `uv` environment setup works.
- Upstream Graphdeco extension install remains CUDA-specific and fails for ROCm.
- ROCm GSplat package (`amd_gsplat`) installs and imports in the prepared `uv` env.

## Architecture

### High-level design

Keep `GaussianTrainer` as the stable Stage 3 orchestrator and delegate backend-specific actions to a trainer interface.

### Interface

Define a backend trainer contract:

- `validate() -> bool`
- `train() -> bool`
- `render() -> bool` (non-fatal on failure)
- `metrics() -> bool` (non-fatal on failure)
- `finalize_outputs() -> bool`

### Implementations

- `CudaGraphdecoTrainer`
  - Uses existing Graphdeco command flow.
  - Preserves current behavior as default and regression baseline.

- `RocmGsplatTrainer`
  - Uses GSplat-compatible training/eval flow inside selected runner (`uv` or `conda`, where configured).
  - Stores backend-native artifacts under a backend-specific working directory (for example `gaussian/backend_rocm/`).
  - Converts/maps backend artifacts into canonical output paths expected by the pipeline.

### Selection rules

- `cfg.gs_backend == "cuda"` -> `CudaGraphdecoTrainer`
- `cfg.gs_backend == "rocm"` -> `RocmGsplatTrainer`

Environment runner selection remains independent:

- `cfg.env_runner == "conda"` -> conda activation path
- `cfg.env_runner == "uv"` -> explicit Python interpreter path (`cfg.uv_python`)

## Data Flow

1. `GaussianTrainer.run()` performs shared Stage 3 checks:
   - Source path shape (`colmap/dense/images`, `colmap/dense/sparse`)
   - Resume check against canonical `cfg.final_ply`
2. Selected backend `validate()` is executed.
3. Backend `train()` runs.
4. Backend `finalize_outputs()` ensures canonical output structure exists.
5. Optional eval path:
   - If `cfg.run_render` true: call backend `render()` then `metrics()`.
6. Shared summary printer uses canonical locations.

## Canonical Output Contract

The following path is mandatory for all backends:

- `cfg.final_ply` => `gaussian/point_cloud/iteration_<iterations>/point_cloud.ply`

Rules:

- `finalize_outputs()` must create this file if training succeeded.
- If missing after finalize, Stage 3 fails.
- Resume logic (`--resume`) remains based on this path only.

Optional artifacts (render outputs/metrics) may differ internally by backend, but where produced they should be mapped to current pipeline locations (`gaussian/test`, `gaussian/results.json`) for compatibility.

## Configuration and CLI

Keep existing flags and semantics:

- `--gs-backend {cuda,rocm}`
- `--env-runner {conda,uv}`
- `--uv-python <path>`
- `--conda-env`, `--rocm-env`

Only add new flags if strictly required by implementation and impossible to infer from current config defaults.

## Error Handling

### Fatal errors

- Selected backend validation fails.
- Training command fails.
- Canonical final PLY missing after `finalize_outputs()`.

### Non-fatal errors

- Backend render step fails.
- Backend metrics step fails.

### Actionable diagnostics

Backend-specific errors must include:

- what check failed,
- command or dependency required to fix it,
- verification command (for ROCm: `python -c "import torch; print(torch.version.hip)"`, plus `import gsplat` check).

## Testing Strategy

### Unit tests

- Backend selector returns correct trainer implementation.
- Runner dispatch selects `run_in_conda` vs `run_in_uv` correctly.
- ROCm finalize path maps a backend-native sample output to canonical `cfg.final_ply`.
- Resume behavior remains keyed to canonical final PLY.

### CLI tests

- Parse and validate backend/runner flags.
- Ensure default path remains CUDA + conda.

### Dry-run smoke tests

- CUDA path command plan
- ROCm + uv path command plan

### Regression checks

- CUDA output layout remains unchanged.
- Existing Stage 3 summary and resume behavior remain unchanged for CUDA.

## Implementation Plan Readiness

This design is scoped to one feature area (Stage 3 backend abstraction and contract-preserving ROCm integration) and is ready to convert into a step-by-step implementation plan.

## Risks and Mitigations

- GSplat artifact layout may differ from Graphdeco.
  - Mitigation: mandatory `finalize_outputs()` normalization step.
- ROCm environment variability across hosts.
  - Mitigation: explicit validation checks and actionable diagnostics.
- Drift between CUDA and ROCm behavior.
  - Mitigation: shared Stage 3 orchestration + contract-based tests.
