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

## Model choices and the missing Tiny models

The Tiny models (`vit_t`, `vit_t_lm`, `vit_t_em_organelles`, MobileSAM-based) are not
offered. They need the `mobile_sam` package, which is not on PyPI, and the conda-forge
package would pull in conda pytorch. Conda CUDA pytorch cannot be installed by Fractal
(its `pixi install --frozen` runs on a GPU-less server and conda CUDA builds require the
`__cuda` virtual package), so the task uses PyPI CUDA wheels, and Tiny support would need a
git-sourced dependency. On a GPU the speed gain of Tiny over Basic is small. To restore
them: add `mobile-sam` (pinned commit) to the pixi pypi-dependencies and re-add the three
enum entries in `utils_segmentation.py`.

## GPU architectures and the torch wheel

On linux-64 the task uses torch CUDA 12.6 wheels (`pyproject.toml`, PyTorch index `cu126`).
Do not move to `cu128`: those wheels dropped Volta kernels (V100, compute capability 7.0),
and the cluster mixes V100 and A40 nodes. `select_device` fails immediately, with the
supported architectures in the message, if the GPU is not covered by the installed wheel.
PyTorch 2.14 is the last release with prebuilt cu126 (Volta) wheels, hence `torch<2.15`.
