#!/usr/bin/env bash
if [[ -n "${ZSH_VERSION:-}" || "${BASH_SOURCE[0]:-}" != "$0" ]]; then
    _robosys_thor_profile_activate() {
        local root
        if [[ -n "${ZSH_VERSION:-}" ]]; then
            root="${(%):-%x}"
        else
            root="${BASH_SOURCE[0]}"
        fi
        root="$(cd -- "$(dirname -- "$root")/../.." && pwd)" || return
        bash "$root/scripts/profile/setup-thor.sh" "$@" || return
        [[ "${1:-}" != --help ]] || return 0
        source "$root/.venv-profile-thor/bin/activate" || return
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
        export CUDAToolkit_ROOT="$CUDA_HOME"
        export JAX_PLATFORMS=cpu
        case ":$PATH:" in
            *":$CUDA_HOME/bin:"*) ;;
            *) export PATH="$CUDA_HOME/bin:$PATH" ;;
        esac
        local cuda_lib="$CUDA_HOME/targets/sbsa-linux/lib"
        case "${LD_LIBRARY_PATH:-}" in
            "$cuda_lib"|"$cuda_lib":*) ;;
            *) export LD_LIBRARY_PATH="$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
        case ":${PYTHONPATH:-}:" in
            *":$root/src:"*) ;;
            *) export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" ;;
        esac
        printf '\nThor profiling environment activated: %s\n' "$VIRTUAL_ENV"
    }
    _robosys_thor_profile_activate "$@"
    return $?
fi

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == --help ]]; then
    cat <<'HELP'
Usage: source scripts/profile/setup-thor.sh
Create and activate .venv-profile-thor for OpenPI and FlashRT profiling on NVIDIA Thor.
Running with bash performs setup only; sourcing also activates the environment.

Requires Linux aarch64, Python 3.11, CUDA 13.x, Git, curl and a compiler.
FlashRT is built from the pinned submodule for Thor SM110.

Options:
  CUDA_HOME=/usr/local/cuda-13.0
  TORCH_SPEC=torch==2.9.0+cu130
  TORCH_INDEX=https://download.pytorch.org/whl/cu130
  BUILD_JOBS=4
  SKIP_GPU_CHECK=1       Install without executing a CUDA smoke test.
  SKIP_CHECKPOINT=1      Do not download and convert pi05_libero.
  CKPT_DIR=...           JAX checkpoint input/download directory.
  PYTORCH_CKPT_DIR=...   Converted checkpoint destination.
HELP
    exit 0
fi
[[ $# == 0 ]] || { echo 'Use --help for usage.' >&2; exit 1; }
[[ "$(uname -s)-$(uname -m)" == Linux-aarch64 ]] || {
    echo 'This setup targets Linux aarch64 NVIDIA Thor.' >&2; exit 1;
}

cd "$ROOT"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDAToolkit_ROOT="$CUDA_HOME"
CUDA_LIB="$CUDA_HOME/targets/sbsa-linux/lib"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || {
    echo "CUDA compiler not found: $CUDA_HOME/bin/nvcc" >&2; exit 1;
}
"$CUDA_HOME/bin/nvcc" --version | grep -Eq 'release 13\.' || {
    echo "Thor requires CUDA 13.x; CUDA_HOME is $CUDA_HOME" >&2; exit 1;
}
[[ -d "$CUDA_LIB" ]] || {
    echo "Thor SBSA CUDA libraries not found: $CUDA_LIB" >&2; exit 1;
}
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu

if ! command -v uv >/dev/null; then
    installer="$(mktemp)"
    trap 'rm -f -- "$installer"' EXIT
    curl --fail --location https://astral.sh/uv/0.12.13/install.sh --output "$installer"
    sh "$installer"
    rm -f -- "$installer"
    trap - EXIT
fi

OPENPI="$ROOT/3rdparty/openpi"
OPENPI_REV=215abfb217dbac7d5f1273282331b9b1866c0479
if [[ ! -e "$OPENPI" ]]; then
    git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI"
    git -C "$OPENPI" checkout --detach "$OPENPI_REV"
fi
[[ "$(git -C "$OPENPI" rev-parse HEAD)" == "$OPENPI_REV" ]] || {
    echo 'OpenPI revision differs from the pinned version; leaving it untouched.' >&2
    exit 1
}

PROFILE_ENV="$ROOT/.venv-profile-thor"
[[ -x "$PROFILE_ENV/bin/python" ]] || uv venv "$PROFILE_ENV" --python 3.11
PYTHON="$PROFILE_ENV/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Expected Python 3.11"'

# Install CUDA PyTorch first.  OpenPI is installed --no-deps below so its
# torch==2.7.1 / jax[cuda12] metadata cannot replace the Thor packages.
uv pip install --python "$PYTHON" \
    --index "${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}" \
    "${TORCH_SPEC:-torch==2.9.0+cu130}"
uv pip install --python "$PYTHON" -r "$ROOT/scripts/profile/requirements-thor.txt"
uv pip install --python "$PYTHON" --no-deps \
    -e "$OPENPI/packages/openpi-client" -e "$OPENPI"

FLASHRT="$ROOT/3rdparty/FlashRT"
if [[ ! -f "$FLASHRT/pyproject.toml" ]]; then
    git submodule update --init -- 3rdparty/FlashRT
fi
uv pip install --python "$PYTHON" --no-deps -e "$FLASHRT"
if [[ ! -e "$FLASHRT/third_party/cutlass" ]]; then
    git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git \
        "$FLASHRT/third_party/cutlass"
fi
"$PROFILE_ENV/bin/cmake" -S "$FLASHRT" -B "$FLASHRT/build-profile-thor" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$PROFILE_ENV/bin/ninja" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DFA2_ARCH_NATIVE_ONLY=ON \
    -DPython3_EXECUTABLE="$PYTHON" -DGPU_ARCH=110
"$PROFILE_ENV/bin/cmake" --build "$FLASHRT/build-profile-thor" \
    --parallel "${BUILD_JOBS:-4}"

# OpenPI's PyTorch implementation depends on its Transformers replacements.
"$PYTHON" - "$OPENPI" <<'PY'
from pathlib import Path
import shutil
import sys
import tempfile
import transformers

assert transformers.__version__ == "4.53.2", "OpenPI patches require Transformers 4.53.2"
patches = Path(sys.argv[1]) / "src/openpi/models_pytorch/transformers_replace"
destination = Path(transformers.__file__).parent
for source in patches.rglob("*.py"):
    target = destination / source.relative_to(patches)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        staged = Path(temporary) / source.name
        shutil.copy2(source, staged)
        staged.replace(target)
print("OpenPI Transformers patches installed")
PY

# Do not run `uv pip check` here: the pinned OpenPI project metadata requests
# torch==2.7.1, but SM110 support requires the CUDA 13 build installed above.
# The import and real-GPU smoke tests below are the compatibility checks for
# this derived environment.
JAX_PLATFORMS=cpu "$PYTHON" "$ROOT/scripts/convert_jax_model_to_pytorch.py" --help >/dev/null
JAX_PLATFORMS=cpu "$PYTHON" - "${SKIP_GPU_CHECK:-0}" <<'PY'
import sys
import torch
import pynvml
from cuda.bindings import driver
from robort.policies.openpi_cuda_graph import CudaGraphOpenPIPolicy
from robort.profile.runner import Runner
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
if sys.argv[1] != "1":
    assert torch.cuda.is_available(), "PyTorch cannot access CUDA; use SKIP_GPU_CHECK=1 for installation only"
    capability = torch.cuda.get_device_capability(0)
    assert capability == (11, 0), f"Expected NVIDIA Thor SM110, found SM{capability[0]}{capability[1]}"
    # Force execution so a wheel missing SM110 kernels fails during setup.
    result = torch.ones(1, device="cuda") + 1
    torch.cuda.synchronize()
    assert result.item() == 2
    print("GPU:", torch.cuda.get_device_name(0), "capability:", capability)
PY

if [[ "${SKIP_CHECKPOINT:-0}" != 1 ]]; then
    JAX_PLATFORMS=cpu "$PYTHON" "$ROOT/scripts/profile/checkpoints.py" \
        --checkpoint-dir "${CKPT_DIR:-$ROOT/checkpoints/pi05_libero}" \
        --output-dir "${PYTORCH_CKPT_DIR:-$ROOT/checkpoints/pi05_libero_pytorch}"
fi

printf '\nThor setup complete. Source scripts/profile/setup-thor.sh to activate it.\n'
printf 'OpenPI and FlashRT SM110 profiling backends are ready.\n'
