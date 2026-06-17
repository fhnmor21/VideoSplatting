# ROCm Trainer CLI Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ROCm Gaussian backend invoke `rocm-gsplat/examples/simple_trainer.py` with the `tyro` CLI shape it actually expects.

**Architecture:** Keep the fix in the backend adapter layer so the pipeline stays unchanged. Translate the pipeline config into the trainer's subcommand-based CLI, preserve the existing output directory contract, and verify the generated command with a regression test.

**Tech Stack:** Python, unittest, existing pipeline backend adapters

---

### Task 1: Update ROCm command translation

**Files:**
- Modify: `python/gaussian_backends.py:219-229`

- [ ] **Step 1: Write the failing test**

```python
def test_rocm_backend_builds_tyro_command_shape(self):
    cfg = self.make_cfg("rocm")
    backend = backend_for(cfg)
    cmd = backend.build_train_cmd()

    self.assertEqual(cmd[:2], ["python", str(cfg.gs_repo / cfg.gsplat_train_script)])
    self.assertEqual(cmd[2], "default")
    self.assertIn("--data-dir", cmd)
    self.assertIn(str(cfg.colmap_dense), cmd)
    self.assertIn("--result-dir", cmd)
    self.assertIn(str(cfg.rocm_backend_output_dir), cmd)
    self.assertIn("--max-steps", cmd)
    self.assertIn(str(cfg.iterations), cmd)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest python.tests.test_gaussian_backends.TestGaussianBackends.test_rocm_backend_builds_tyro_command_shape -v`
Expected: FAIL because `build_train_cmd()` still emits legacy `--data_dir` / `--output_dir` / `--iterations` flags without the `default` subcommand.

- [ ] **Step 3: Write minimal implementation**

```python
    def build_train_cmd(self) -> List[str]:
        return [
            "python",
            str(self.cfg.gs_repo / self.cfg.gsplat_train_script),
            "default",
            "--data-dir",
            str(self.cfg.colmap_dense),
            "--result-dir",
            str(self.cfg.rocm_backend_output_dir),
            "--max-steps",
            str(self.cfg.iterations),
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest python.tests.test_gaussian_backends.TestGaussianBackends.test_rocm_backend_builds_tyro_command_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/gaussian_backends.py python/tests/test_gaussian_backends.py docs/superpowers/plans/2026-05-25-rocm-trainer-cli-fix.md
git commit -m "fix: translate rocm trainer cli"
```
