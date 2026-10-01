# fractal-microsam-segmentation-task

microSAM segmentation inference for Fractal

## Running

```bash
pixi run -e dev test
```

Runs the test suite (mocked micro-SAM, no GPU needed). Expect all tests to pass.

```bash
pixi run -e dev create-manifest
```

Regenerates `__FRACTAL_MANIFEST__.json` from the task signature. Re-run after any parameter change.

## GPU requirement

The task requires a CUDA GPU and fails fast otherwise (set `allow_cpu=True` to override for
small tests). The linux-64 pixi environment pins a CUDA build of pytorch via
`[tool.pixi.system-requirements]`; the log at the start of each task run reports the
torch/CUDA build and visible GPU.
