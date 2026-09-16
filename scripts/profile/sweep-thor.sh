#!/usr/bin/env bash
# Sweep RoboSys profiling across Jetson AGX Thor nvpmodel modes.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == --help ]]; then
    cat <<'HELP'
Usage: bash scripts/profile/sweep-thor.sh

Sweep modes 0-3, restoring the original nvpmodel mode on exit. Profiling runs
as the current user; sudo is used only for `nvpmodel -m`.

Environment options:
  NVP_MODES="0 1 2 3"          nvpmodel mode IDs
  BACKENDS="openpi flash_rt"   backends to run
  OPENPI_EXECUTIONS="graph"     use "eager graph" to run both
  FLASHRT_EXECUTIONS="graph"    use "eager graph" to run both
  SM_PERCS="0.5 1.0"
  BATCH_SIZES="1 2 3 4 5 6 7 8"
  NUM_WARMUP=10
  NUM_ITER=100
  SETTLE_SECONDS=10             delay after each mode switch
  OUTPUT_ROOT=profile/thor-nvpmodel-YYYYmmdd_HHMMSS
  PYTORCH_CKPT_DIR=checkpoints/pi05_libero_pytorch
  ALLOW_BUSY_GPU=1              bypass the compute-process safety check

Examples:
  # Complete sweep:
  bash scripts/profile/sweep-thor.sh

  # Quick OpenPI-only validation:
  BACKENDS=openpi SM_PERCS=1.0 BATCH_SIZES=1 NUM_WARMUP=3 NUM_ITER=10 \
    bash scripts/profile/sweep-thor.sh
HELP
    exit 0
fi
[[ $# == 0 ]] || { echo 'Use --help for usage.' >&2; exit 1; }
[[ "$(uname -s)-$(uname -m)" == Linux-aarch64 ]] || {
    echo 'This sweep targets Linux aarch64 NVIDIA Thor.' >&2; exit 1;
}
command -v nvpmodel >/dev/null || { echo 'nvpmodel is not installed.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is not installed.' >&2; exit 1; }

cd "$ROOT"
PYTHON="$ROOT/.venv-profile-thor/bin/python"
[[ -x "$PYTHON" ]] || {
    echo 'Missing .venv-profile-thor; run source scripts/profile/setup-thor.sh first.' >&2
    exit 1
}
CHECKPOINT="${PYTORCH_CKPT_DIR:-$ROOT/checkpoints/pi05_libero_pytorch}"
[[ -f "$CHECKPOINT/model.safetensors" ]] || {
    echo "Missing PyTorch checkpoint: $CHECKPOINT/model.safetensors" >&2
    exit 1
}

read -r -a modes <<< "${NVP_MODES:-0 1 2 3}"
read -r -a backends <<< "${BACKENDS:-openpi flash_rt}"
read -r -a openpi_executions <<< "${OPENPI_EXECUTIONS:-graph}"
read -r -a flashrt_executions <<< "${FLASHRT_EXECUTIONS:-graph}"
read -r -a sm_percs <<< "${SM_PERCS:-0.5 1.0}"
read -r -a batch_sizes <<< "${BATCH_SIZES:-1 2 3 4 5 6 7 8}"
NUM_WARMUP="${NUM_WARMUP:-10}"
NUM_ITER="${NUM_ITER:-100}"
SETTLE_SECONDS="${SETTLE_SECONDS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/profile/thor-nvpmodel-$(date +%Y%m%d_%H%M%S)}"

for mode in "${modes[@]}"; do
    [[ "$mode" =~ ^[0-3]$ ]] || { echo "Invalid nvpmodel mode: $mode" >&2; exit 1; }
done
for backend in "${backends[@]}"; do
    [[ "$backend" == openpi || "$backend" == flash_rt ]] || {
        echo "Invalid backend: $backend" >&2; exit 1;
    }
done
for execution in "${openpi_executions[@]}"; do
    [[ "$execution" == eager || "$execution" == graph ]] || {
        echo "Invalid OpenPI execution mode: $execution" >&2; exit 1;
    }
done
for execution in "${flashrt_executions[@]}"; do
    [[ "$execution" == eager || "$execution" == graph ]] || {
        echo "Invalid FlashRT execution mode: $execution" >&2; exit 1;
    }
done
for value in "$NUM_WARMUP" "$NUM_ITER" "$SETTLE_SECONDS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "Expected a nonnegative integer: $value" >&2; exit 1; }
done
(( NUM_ITER > 0 )) || { echo 'NUM_ITER must be positive.' >&2; exit 1; }

if [[ "${ALLOW_BUSY_GPU:-0}" != 1 ]]; then
    busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | sed '/^[[:space:]]*$/d' || true)"
    [[ -z "$busy" ]] || {
        echo "GPU compute processes are active (PIDs: $(tr '\n' ' ' <<< "$busy"))." >&2
        echo 'Stop them or set ALLOW_BUSY_GPU=1 if this is intentional.' >&2
        exit 1
    }
fi

original_mode="$(nvpmodel -q | tail -n 1 | tr -d '[:space:]')"
[[ "$original_mode" =~ ^[0-9]+$ ]] || {
    echo "Could not parse current nvpmodel mode: $original_mode" >&2; exit 1;
}

sudo -v
restore_mode() {
    status=$?
    trap - EXIT
    if ! sudo nvpmodel -m "$original_mode"; then
        echo "WARNING: failed to restore nvpmodel mode $original_mode" >&2
        status=1
    fi
    exit "$status"
}
trap restore_mode EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDAToolkit_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$ROOT/.venv-profile-thor/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
mkdir -p "$OUTPUT_ROOT"

run_profile() {
    local mode=$1 backend=$2 execution=$3 output=$4
    local graph_flag=--no-use-cuda-graph
    local num_views=2
    [[ "$execution" != graph ]] || graph_flag=--use-cuda-graph
    mkdir -p "$output"
    "$PYTHON" -m robort.profile.main \
        --model-name pi05_libero \
        --backend "$backend" \
        --model-dir "$CHECKPOINT" \
        --num-views "$num_views" \
        --image-resolution 224 \
        --precision "$( [[ "$backend" == openpi ]] && echo bf16 || echo fp16 )" \
        --batch-sizes "${batch_sizes[@]}" \
        --prompt-len 10 \
        --num-steps 10 \
        --chunk-size 10 \
        --power-perc 1.0 \
        --sm-perc "${sm_percs[@]}" \
        --num-warmup "$NUM_WARMUP" \
        --num-iter "$NUM_ITER" \
        "$graph_flag" \
        --output-dir "$output" 2>&1 | tee "$output/run.log"
    [[ -s "$output/policy_profile.csv" ]] || {
        echo "No policy profiles were produced for $backend ($execution)." >&2
        echo "See $output/run.log for skipped configurations or backend errors." >&2
        return 1
    }
    printf '%s\n' "$mode" > "$output/nvpmodel_mode"
    printf '%s\n' "$execution" > "$output/execution_mode"
}

for mode in "${modes[@]}"; do
    echo "Switching Thor to nvpmodel mode $mode"
    sudo nvpmodel -m "$mode"
    sleep "$SETTLE_SECONDS"
    mode_root="$OUTPUT_ROOT/mode_$mode"
    mkdir -p "$mode_root"
    nvpmodel -q > "$mode_root/nvpmodel.txt"
    nvidia-smi -q > "$mode_root/nvidia-smi-before.txt"

    for backend in "${backends[@]}"; do
        if [[ "$backend" == openpi ]]; then
            for execution in "${openpi_executions[@]}"; do
                run_profile "$mode" openpi "$execution" \
                    "$mode_root/openpi-$execution"
            done
        else
            # The current FlashRT public policy supports only batch size 1.
            saved_batch_sizes=("${batch_sizes[@]}")
            batch_sizes=(1)
            for execution in "${flashrt_executions[@]}"; do
                run_profile "$mode" flash_rt "$execution" \
                    "$mode_root/flash_rt-$execution"
            done
            batch_sizes=("${saved_batch_sizes[@]}")
        fi
    done
    nvidia-smi -q > "$mode_root/nvidia-smi-after.txt"
done

# Merge results for analysis while retaining per-run source files and logs.
"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import csv
from pathlib import Path
import sys

root = Path(sys.argv[1])
for filename in ("hardware_profile.csv", "policy_profile.csv"):
    rows = []
    columns = ["nvpmodel_mode", "execution"]
    for path in sorted(root.glob(f"mode_*/*/{filename}")):
        mode = path.parents[1].name.removeprefix("mode_")
        execution = path.parent.name
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                tagged = {"nvpmodel_mode": mode, "execution": execution, **row}
                rows.append(tagged)
                columns.extend(key for key in tagged if key not in columns)
    if rows:
        target = root / f"all_{filename}"
        with target.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {target}")
PY

printf '\nThor nvpmodel sweep complete: %s\n' "$OUTPUT_ROOT"
