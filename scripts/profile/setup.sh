#!/usr/bin/env bash
if [[ -n "${ZSH_VERSION:-}" || "${BASH_SOURCE[0]:-}" != "$0" ]]; then
    _robosys_profile_activate() {
        local root
        if [[ -n "${ZSH_VERSION:-}" ]]; then
            root="${(%):-%x}"
        else
            root="${BASH_SOURCE[0]}"
        fi
        root="$(cd -- "$(dirname -- "$root")/../.." && pwd)" || return
        # Run installation separately so its exit/set/cd cannot affect this shell.
        bash "$root/scripts/profile/setup.sh" "$@" || return
        [[ "${1:-}" != --help ]] || return 0
        source "$root/.venv-profile/bin/activate" || return
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
        export CUDAToolkit_ROOT="$CUDA_HOME"
        # FlashRT loads libcudart.so by name; CUDA_HOME/PATH do not select it.
        case "${LD_LIBRARY_PATH:-}" in
            "$CUDA_HOME/lib64"|"$CUDA_HOME/lib64":*) ;;
            *) export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
        case ":$PATH:" in
            *":$CUDA_HOME/bin:"*) ;;
            *) export PATH="$CUDA_HOME/bin:$PATH" ;;
        esac
        case ":${PYTHONPATH:-}:" in
            *":$root/src:"*) ;;
            *) export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" ;;
        esac
        printf '\nProfiling environment activated: %s\n' "$VIRTUAL_ENV"
    }
    _robosys_profile_activate "$@"
    return $?
fi

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == --help ]]; then
    cat <<'HELP'
Usage: source scripts/profile/setup.sh
Create .venv-profile and build FlashRT for Linux x86_64 / Python 3.11 / CUDA 12.8.
Also create .venv-openpi for OpenPI profiling and checkpoint conversion.
Sourcing also activates the environment and configures paths in Bash or zsh.
Running with bash performs setup only. The caller's working directory is preserved.
Requires git, curl, a C++ compiler, CUDA 12.8 and a compatible NVIDIA driver.
Options: CUDA_HOME=/usr/local/cuda-12.8 GPU_ARCH=89 BUILD_JOBS=4
Set BUILD_FLASHRT=0 to install the Python environment without compiling FlashRT.
Set INSTALL_OPENPI=0 to skip the OpenPI environment.
Set SKIP_GPU_CHECK=1 to install without an accessible GPU.
Download and convert pi05_libero when missing (requires INSTALL_OPENPI=1).
Set SKIP_CHECKPOINT=1 to skip checkpoint preparation.
Paths: CKPT_DIR=checkpoints/pi05_libero
       PYTORCH_CKPT_DIR=checkpoints/pi05_libero_pytorch
HELP
    exit 0
fi
[[ $# == 0 ]] || { echo 'Use --help for usage.' >&2; exit 1; }
[[ "$(uname -s)-$(uname -m)" == Linux-x86_64 ]] || {
    echo 'This lockfile targets Linux x86_64.' >&2; exit 1;
}
cd "$ROOT"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDAToolkit_ROOT="$CUDA_HOME"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    installer="$(mktemp)"
    trap 'rm -f -- "$installer"' EXIT
    curl --fail --location https://astral.sh/uv/0.12.13/install.sh --output "$installer"
    sh "$installer"
    rm -f -- "$installer"
    trap - EXIT
fi
if [[ "${BUILD_FLASHRT:-1}" != 0 ]]; then
    "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12\.8,'
    command -v c++ >/dev/null
fi
FLASHRT="$ROOT/3rdparty/FlashRT"
if [[ ! -f "$FLASHRT/pyproject.toml" ]]; then
    git submodule update --init -- 3rdparty/FlashRT
fi
bash "$ROOT/scripts/profile/apply-flashrt-patch.sh" "$FLASHRT"
PROFILE_ENV="$ROOT/.venv-profile"
[[ -x "$PROFILE_ENV/bin/python" ]] || uv venv "$PROFILE_ENV" --python 3.11
"$PROFILE_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Expected Python 3.11"'
uv pip sync --python "$PROFILE_ENV/bin/python" --torch-backend cu128 \
    scripts/profile/requirements-cu128.txt
uv pip check --python "$PROFILE_ENV/bin/python"
if [[ "${INSTALL_OPENPI:-1}" != 0 ]]; then
    OPENPI="$ROOT/3rdparty/openpi"
    if [[ ! -e "$OPENPI" ]]; then
        git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI"
        git -C "$OPENPI" checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479
    fi
    [[ "$(git -C "$OPENPI" rev-parse HEAD)" == 215abfb217dbac7d5f1273282331b9b1866c0479 ]] || {
        echo 'OpenPI revision differs from the pinned version; leaving it untouched.' >&2
        exit 1
    }
    UV_PROJECT_ENVIRONMENT="$ROOT/.venv-openpi" GIT_LFS_SKIP_SMUDGE=1 \
        uv sync --project "$OPENPI" --python 3.11 --frozen --no-dev
    # OpenPI's gemma_pytorch.py imports pytest for its cache type annotation.
    uv pip install --python "$ROOT/.venv-openpi/bin/python" \
        -r "$ROOT/scripts/profile/requirements-openpi.txt"
    uv pip check --python "$ROOT/.venv-openpi/bin/python"
    "$ROOT/.venv-openpi/bin/python" - "$OPENPI" <<'PY'
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
    # Replace files rather than modifying uv's potentially hard-linked cache.
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        staged = Path(temporary) / source.name
        shutil.copy2(source, staged)
        staged.replace(target)
print("OpenPI Transformers patches installed")
PY
    JAX_PLATFORMS=cpu "$ROOT/.venv-openpi/bin/python" \
        "$ROOT/scripts/convert_jax_model_to_pytorch.py" --help >/dev/null
    JAX_PLATFORMS=cpu "$ROOT/.venv-openpi/bin/python" -c \
        'from robort.policies.openpi_cuda_graph import CudaGraphOpenPIPolicy; from robort.profile.runner import Runner; import pynvml; from cuda.bindings import driver'
    printf '\nOpenPI conversion environment ready: %s\n' "$ROOT/.venv-openpi/bin/python"
    if [[ "${SKIP_CHECKPOINT:-0}" != 1 ]]; then
        JAX_PLATFORMS=cpu "$ROOT/.venv-openpi/bin/python" \
            "$ROOT/scripts/profile/checkpoints.py" \
            --checkpoint-dir "${CKPT_DIR:-$ROOT/checkpoints/pi05_libero}" \
            --output-dir "${PYTORCH_CKPT_DIR:-$ROOT/checkpoints/pi05_libero_pytorch}"
    fi
fi
if [[ "${BUILD_FLASHRT:-1}" != 0 ]]; then
    if [[ ! -e "$FLASHRT/third_party/cutlass" ]]; then
        git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git \
            "$FLASHRT/third_party/cutlass"
    fi
    "$PROFILE_ENV/bin/cmake" -S "$FLASHRT" -B "$FLASHRT/build-profile" -G Ninja \
        -DCMAKE_MAKE_PROGRAM="$PROFILE_ENV/bin/ninja" \
        -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
        -DFA2_ARCH_NATIVE_ONLY=ON \
        -DPython3_EXECUTABLE="$PROFILE_ENV/bin/python" -DGPU_ARCH="${GPU_ARCH:-89}"
    "$PROFILE_ENV/bin/cmake" --build "$FLASHRT/build-profile" --parallel "${BUILD_JOBS:-4}"
    "$PROFILE_ENV/bin/python" -c 'from flash_rt.frontends.torch.pi05_rtx_fp16 import Pi05TorchFrontendRtxFP16; from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx'
fi
"$PROFILE_ENV/bin/python" - "${BUILD_FLASHRT:-1}" "${SKIP_GPU_CHECK:-0}" <<'PY'
import sys
import ctypes
import torch
import pynvml
from cuda.bindings import driver
from robort.profile.runner import Runner

if sys.argv[1] != "0":
    runtime = ctypes.CDLL("libcudart.so")
    version = ctypes.c_int()
    runtime.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaRuntimeGetVersion.restype = ctypes.c_int
    assert runtime.cudaRuntimeGetVersion(ctypes.byref(version)) == 0
    assert version.value == 12080, f"FlashRT requires CUDA runtime 12.8; loaded {version.value}. Check LD_LIBRARY_PATH."
    print("FlashRT CUDA runtime:", version.value)
if sys.argv[2] != "1":
    assert torch.cuda.is_available(), "PyTorch cannot access CUDA; use SKIP_GPU_CHECK=1 for installation only"
print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
PY
printf '\nSetup complete. Source scripts/profile/setup.sh to activate in your shell.\n'
