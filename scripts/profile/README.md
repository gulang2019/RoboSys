# Profiling environment

For the complete Armory/LIBERO, OpenPI, and FlashRT installation, run `bash setup.sh`
from the repository root. It installs Ubuntu system packages and CUDA Toolkit 12.8
when missing; a working NVIDIA driver remains a prerequisite. Missing checkpoints are downloaded and converted automatically. The standalone script bootstraps uv if needed.

Run from the RoboSys checkout on Linux x86_64. Requires Git, curl, a C++
compiler, CUDA Toolkit 12.8 and a compatible NVIDIA driver. The default build
targets RTX 4090 (SM89); use `GPU_ARCH=120` for RTX 5090. This script does not
install system packages. It downloads and converts missing pi05_libero checkpoints
by default; set `SKIP_CHECKPOINT=1` for environment setup only. It also prepares `.venv-openpi`
for OpenPI inference, CUDA graph profiling, and checkpoint conversion, using OpenPI's own `uv.lock` and Python 3.11.

```bash
source scripts/profile/setup.sh
```

The script creates `.venv-profile` with Python 3.11, installs the pinned CUDA
12.8 Python dependencies, installs the vendored FlashRT in editable mode, and
builds its kernels. Sourcing then activates the environment and sets `PYTHONPATH`,
`CUDA_HOME`, `CUDAToolkit_ROOT`, `LD_LIBRARY_PATH`, and the CUDA executable path in your current Bash
or zsh shell. Your working directory and shell options are preserved. Setup failure
returns an error without activating the environment. Repeated runs reuse the
environment and CMake build, without duplicating the added path entries.
Running with `bash scripts/profile/setup.sh` still performs setup only.
The CUDA 12.8 `lib64` directory takes priority in `LD_LIBRARY_PATH`: FlashRT
loads `libcudart.so` by name, and selecting an older system runtime can segfault
during CUDA graph creation. Setup verifies the loaded runtime reports 12.8.
Attention kernels are built only for `GPU_ARCH`, avoiding the upstream
multi-architecture default's SM121 target, which CUDA 12.8 cannot compile.

```bash
CUDA_HOME=/usr/local/cuda-12.8 GPU_ARCH=89 BUILD_JOBS=4 source scripts/profile/setup.sh
# Only prepare Python packages, for hardware profiling or a later kernel build:
BUILD_FLASHRT=0 source scripts/profile/setup.sh
# Skip the separate conversion environment when it is not needed:
INSTALL_OPENPI=0 source scripts/profile/setup.sh
# Install packages on a machine without a GPU/toolkit:
BUILD_FLASHRT=0 SKIP_GPU_CHECK=1 source scripts/profile/setup.sh
```

Run from the repository root after setup:

```bash
python -m robort.profile.main --help
ROBORT_TEST_CUDA=1 python -m pytest tests/test_profile_cuda.py -q
```

Live tests require GPU access; they preserve the current power limit. Full
profiling may require permission to change NVML power limits. FlashRT policy
profiling additionally requires `--model-dir` pointing to a PyTorch checkpoint
containing `model.safetensors`; an Orbax `params/` directory is insufficient.

## Convert an OpenPI checkpoint

OpenPI requires Transformers 4.53.2 with its model patches, so it is installed
in `.venv-openpi` rather than changing `.venv-profile`. Setup replaces patched
files without modifying uv's hard-linked cache and verifies the converter's
`--help` command. Sourcing setup still activates `.venv-profile`.

Run conversion explicitly with the other interpreter from the repository root:

```bash
JAX_PLATFORMS=cpu .venv-openpi/bin/python \
  scripts/convert_jax_model_to_pytorch.py \
  --checkpoint_dir "$PWD/checkpoints/pi05_libero" \
  --config_name pi05_libero \
  --output_path "$PWD/checkpoints/pi05_libero_pytorch" \
  --precision bfloat16
```

Setup skips conversion when `model.safetensors` and normalization assets already
exist. It reuses an existing JAX checkpoint, or downloads it when missing. Conversion
runs on CPU and includes normalization assets. Failed conversions do not publish a
partial destination. Existing incomplete destinations are preserved and reported.
Override paths with `CKPT_DIR` and `PYTORCH_CKPT_DIR`; `INSTALL_OPENPI=0` disables
conversion along with the OpenPI environment.

If `3rdparty/openpi` is missing, setup clones revision
`215abfb217dbac7d5f1273282331b9b1866c0479`. An existing checkout must match this revision; incompatible checkouts are left untouched.

## Updating dependencies

OpenPI uses its upstream frozen `uv.lock` plus the pinned profiling packages in
`requirements-openpi.txt`. Setup checks dependency compatibility in both environments.

Edit `requirements.in`, then regenerate the lockfile from the repository root
using uv 0.12.13 (the version used to generate it):

```bash
uv pip compile scripts/profile/requirements.in --python-version 3.11 \
  --python-platform x86_64-unknown-linux-gnu --torch-backend cu128 \
  -o scripts/profile/requirements-cu128.txt
```

Commit both requirements files. The Torch backend flag selects the CUDA wheel
index during both compilation and installation. FlashRT follows the repository's
submodule revision; CUTLASS is cloned at v4.4.2 when missing. Existing source
checkouts are preserved. The lockfile covers Python dependencies; the system
CUDA toolkit, driver and C++ compiler remain machine prerequisites.
