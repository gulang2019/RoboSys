#!/usr/bin/env bash
# Native Ubuntu setup for AutoHorizon's LIBERO sweep. Run as your regular user.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUPPORT="$ROOT/scripts/autohorizon"
AUTO="$ROOT/3rdparty/AutoHorizon"
if [[ "${1:-}" == --help ]]; then
    cat <<'EOF'
Usage: bash setup_autohorizon.sh

Installs Ubuntu packages (sudo), uv, Python 3.11 server and Python 3.8
simulator environments; downloads and converts pi05_libero; checks inference.
Requires a working NVIDIA driver/GPU, internet, and substantial free disk/RAM
for two ML environments and both JAX/PyTorch checkpoints.

Options through environment variables:
  SKIP_SYSTEM_DEPS=1   Skip apt when system packages are already installed.
  CHECKPOINT_ROOT=... Checkpoint directory (default: RoboSys/checkpoints).
  LIBERO_REV=...      Override the pinned LIBERO revision.

Does not start a long-running server or benchmark. Prints launch commands.
Existing environments are synced. Existing checkpoints are validated/reused.
EOF
    exit 0
fi
[[ $# == 0 ]] || { echo 'Use --help for usage.' >&2; exit 1; }
[[ $EUID != 0 ]] || { echo 'Run as your normal user, not sudo bash.' >&2; exit 1; }

if [[ "${SKIP_SYSTEM_DEPS:-0}" != 1 ]]; then
    sudo apt-get update
    sudo apt-get install -y git git-lfs wget ca-certificates python3 \
        build-essential cmake pkg-config linux-libc-dev \
        libosmesa6-dev libgl1 libglew-dev libglfw3-dev libgles2-mesa-dev \
        libglib2.0-0 libsm6 libxrender1 libxext6
fi
command -v nvidia-smi >/dev/null || { echo 'Install the NVIDIA driver before setup.' >&2; exit 1; }
nvidia-smi
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    installer="$(mktemp)"
    wget -q https://astral.sh/uv/0.12.13/install.sh -O "$installer"
    sh "$installer"
    rm -f -- "$installer"
fi
uv --version

git -C "$ROOT" submodule update --init --recursive -- 3rdparty/AutoHorizon
LIBERO_REV="${LIBERO_REV:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
if [[ ! -d "$AUTO/third_party/libero" ]]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$AUTO/third_party/libero"
    git -C "$AUTO/third_party/libero" checkout "$LIBERO_REV"
fi
[[ -f "$AUTO/third_party/libero/requirements.txt" ]] || { echo 'Incomplete LIBERO checkout.' >&2; exit 1; }
cd "$AUTO"
if git apply --check "$SUPPORT/evaluator-results.patch" 2>/dev/null; then
    git apply "$SUPPORT/evaluator-results.patch"
elif ! git apply --reverse --check "$SUPPORT/evaluator-results.patch" 2>/dev/null; then
    echo 'Evaluator differs from the supported revision; cannot safely apply results patch.' >&2
    exit 1
fi
if [[ ! -f success_rate_vs_plan_steps.sh ]]; then
    cp "$SUPPORT/success_rate_vs_plan_steps.sh" success_rate_vs_plan_steps.sh
fi

# Isolated source builds need their own setuptools constraint on Python 3.8.
cat > examples/libero/build-constraints.txt <<'EOF'
setuptools==75.3.0
EOF
python3 - <<'PY'
from pathlib import Path
p = Path('third_party/libero/requirements.txt')
text = p.read_text()
if 'robosuite==1.4.0' in text:
    p.write_text(text.replace('robosuite==1.4.0', 'robosuite==1.4.1'))
elif 'robosuite==1.4.1' not in text:
    raise SystemExit('Unexpected LIBERO robosuite requirement; inspect it before continuing.')
PY

GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 --frozen
# Apply BEFORE converting weights, so pi05 adaptive normalization is present.
.venv/bin/python - <<'PY'
import pathlib, shutil, transformers
assert transformers.__version__ == '4.53.2', transformers.__version__
shutil.copytree('src/openpi/models_pytorch/transformers_replace',
                pathlib.Path(transformers.__file__).parent, dirs_exist_ok=True)
PY

if [[ ! -x examples/libero/.venv/bin/python ]]; then
    uv venv --python 3.8 examples/libero/.venv
fi
UV_BUILD_CONSTRAINT="$AUTO/examples/libero/build-constraints.txt" \
uv pip install --python examples/libero/.venv/bin/python \
    -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
    -e packages/openpi-client -e third_party/libero \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy unsafe-best-match

export LIBERO_CONFIG_PATH="$AUTO/.libero"
mkdir -p "$LIBERO_CONFIG_PATH" results/libero e_step_files
.venv/bin/python - <<'PY'
import os, pathlib, yaml
base = pathlib.Path('third_party/libero/libero/libero').resolve()
paths = dict(benchmark_root=base, bddl_files=base/'bddl_files',
             init_states=base/'init_files', assets=base/'assets', datasets=base.parent/'datasets')
pathlib.Path(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml').write_text(
    yaml.safe_dump({k: str(v) for k, v in paths.items()}))
PY
MUJOCO_GL=egl examples/libero/.venv/bin/python - <<'PY'
import imageio, yaml, torch, robosuite, openpi_client
from libero.libero import benchmark
for name in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
    suite = benchmark.get_benchmark_dict()[name]()
    for task_id in range(suite.n_tasks):
        suite.get_task_init_states(task_id)
    print(name, suite.n_tasks, 'tasks: initial states loaded')
PY

export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints}"
mkdir -p "$CHECKPOINT_ROOT"
CHECKPOINT_ROOT="$(cd "$CHECKPOINT_ROOT" && pwd)"
export CHECKPOINT_ROOT
export OPENPI_DATA_HOME="$CHECKPOINT_ROOT"
.venv/bin/python "$SUPPORT/prepare_checkpoint.py"

# Small runtime check: load the converted model and run one dummy observation.
.venv/bin/python - <<'PY'
import os, pathlib
import torch
from openpi.policies.libero_policy import make_libero_example
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA; check GPU/driver compatibility.'
policy = create_trained_policy(get_config('pi05_libero'),
    pathlib.Path(os.environ['CHECKPOINT_ROOT'])/'pi05_libero_pytorch_fixed')
obs = make_libero_example()
obs.update(episode=0, multi_actions=0)
result = policy.infer(obs)
print('Inference passed. Action shape:', result['actions'].shape)
PY

printf '\nSetup complete. Start the sweep (it starts and stops its own server):\n'
printf 'cd %q\n' "$AUTO"
printf 'OPENPI_DATA_HOME=%q LIBERO_CONFIG_PATH=%q CKPT_DIR=%q bash success_rate_vs_plan_steps.sh\n' \
    "$CHECKPOINT_ROOT" "$LIBERO_CONFIG_PATH" "$CHECKPOINT_ROOT/pi05_libero_pytorch_fixed"
