#!/usr/bin/env bash
# Ubuntu/Linux CUDA setup for the Robort LIBERO benchmark.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help ]]; then
    cat <<'HELP'
Usage: bash setup.sh

Initialize pinned Git dependencies, apply benchmark patches, install Python
3.11 and the locked Armory environment, configure LIBERO, download pi05_libero,
and check CUDA/imports/initial states. Does not launch the sweep.
Run after cloning RoboSys; requires Linux, internet and an NVIDIA CUDA 12 driver.
Do not run while a benchmark is using 3rdparty/armory/.venv.

Environment options:
  SKIP_SYSTEM_DEPS=1    Skip Ubuntu apt packages (otherwise uses sudo).
  SKIP_GPU_CHECK=1      Prepare files/environment without an accessible GPU.
  SKIP_CHECKPOINT=1     Skip downloading the checkpoint.
  CKPT_DIR=...         Destination (default: checkpoints/pi05_libero).
  LIBERO_CONFIG_PATH=...  Config directory (default: data/libero_config).
HELP
    exit 0
fi
[[ $# == 0 ]] || { echo 'Use --help for usage.' >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || { echo 'This setup targets Linux/CUDA.' >&2; exit 1; }
cd "$ROOT"
if [[ "${SKIP_SYSTEM_DEPS:-0}" != 1 ]]; then
    sudo apt-get update
    sudo apt-get install -y git git-lfs curl ca-certificates build-essential cmake pkg-config \
        libegl1 libgl1 libosmesa6-dev libglew-dev libglfw3-dev libgles2-mesa-dev \
        libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg
fi
if [[ "${SKIP_GPU_CHECK:-0}" != 1 ]]; then
    nvidia-smi
fi
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    installer="$(mktemp)"
    trap 'rm -f -- "$installer"' EXIT
    curl --fail --location https://astral.sh/uv/0.12.13/install.sh --output "$installer"
    sh "$installer"
    rm -f -- "$installer"
    trap - EXIT
fi
uv --version
# Initialize only dependencies needed by this benchmark, not every research repo.
git submodule update --init --recursive -- 3rdparty/armory
git submodule update --init -- 3rdparty/AutoHorizon
ARMORY="$ROOT/3rdparty/armory"
LIBERO="$ROOT/3rdparty/AutoHorizon/third_party/libero"
LIBERO_REV=8f1084e3132a39270c3a13ebe37270a43ece2a01
if [[ ! -e "$LIBERO" ]]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO"
    git -C "$LIBERO" checkout --detach "$LIBERO_REV"
fi
[[ "$(git -C "$LIBERO" rev-parse HEAD)" == "$LIBERO_REV" ]] || {
    echo "Existing LIBERO checkout differs from pinned $LIBERO_REV; leaving it untouched." >&2
    exit 1
}
apply_patch_once() {
    local checkout=$1 patch=$2
    if git -C "$checkout" apply --check "$patch" 2>/dev/null; then
        git -C "$checkout" apply "$patch"
    elif ! git -C "$checkout" apply --reverse --check "$patch" 2>/dev/null; then
        echo "Cannot apply $patch: checkout contains incompatible edits." >&2
        exit 1
    fi
}
apply_patch_once "$ARMORY" "$ROOT/scripts/benchmark/armory-batch-warmup.patch"
apply_patch_once "$LIBERO" "$ROOT/scripts/benchmark/libero-init-states.patch"
# Armory owns the complete dependency lock, including CUDA JAX and simulation.
GIT_LFS_SKIP_SMUDGE=1 uv sync --project "$ARMORY" --python 3.11 --frozen \
    --extra server --extra evaluation --extra libero
PYTHON="$ARMORY/.venv/bin/python"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/data/libero_config}"
export PYTHONPATH="$ROOT/src:$LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export ROBOSYS_LIBERO_ROOT="$LIBERO"
"$PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml
base = Path(os.environ['ROBOSYS_LIBERO_ROOT']) / 'libero/libero'
config = Path(os.environ['LIBERO_CONFIG_PATH'])
config.mkdir(parents=True, exist_ok=True)
paths = dict(benchmark_root=base, bddl_files=base/'bddl_files',
             init_states=base/'init_files', assets=base/'assets', datasets=base.parent/'datasets')
paths['datasets'].mkdir(exist_ok=True)
(config/'config.yaml').write_text(yaml.safe_dump({k: str(v.resolve()) for k, v in paths.items()}))
PY
if [[ "${SKIP_CHECKPOINT:-0}" != 1 ]]; then
    "$PYTHON" "$ROOT/download_openpi05.py" --destination "${CKPT_DIR:-$ROOT/checkpoints/pi05_libero}"
fi
"$PYTHON" - <<'PY'
import os
import jax
import robort.policy, robort.server, robort.client
import imageio_ffmpeg
from libero.libero import benchmark
if os.environ.get('SKIP_GPU_CHECK') != '1':
    assert any(device.platform == 'gpu' for device in jax.devices()), 'JAX cannot access CUDA'
    print('JAX devices:', jax.devices())
for name in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90'):
    suite = benchmark.get_benchmark_dict()[name]()
    for task_id in range(suite.n_tasks):
        suite.get_task_init_states(task_id)
    print(name, suite.n_tasks, 'tasks: initial states loaded')
print('FFmpeg:', imageio_ffmpeg.get_ffmpeg_exe())
PY
printf '\nSetup complete. Launch the sweep:\n'
printf 'cd %q\n' "$ROOT"
printf 'LIBERO_CONFIG_PATH=%q CKPT_DIR=%q bash benchmark/suc_rate_profile.sh\n' \
    "$LIBERO_CONFIG_PATH" "${CKPT_DIR:-$ROOT/checkpoints/pi05_libero}"
