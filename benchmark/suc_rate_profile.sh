#!/usr/bin/env bash
# Start one local Robort server, sweep all tasks, and stop the server on exit.
# Example: TRIALS=1 PLAN_STEPS="1 5" INFERENCE_DELAYS="0 1" USE_RTC="false true" bash benchmark/suc_rate_profile.sh
# PLAN_STEPS sets the request interval; INFERENCE_DELAYS sets delivery latency in control steps.
# RTC uses the default JAX checkpoint; the PyTorch and Triton backends do not support it.
set -euo pipefail
sweep_started=$SECONDS
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

POLICY_PYTHON="${POLICY_PYTHON:-$PWD/3rdparty/armory/.venv/bin/python}"
SIM_PYTHON="${SIM_PYTHON:-$PWD/3rdparty/armory/.venv/bin/python}"
CKPT_DIR="${CKPT_DIR:-$PWD/checkpoints/pi05_libero}"
MODEL_NAME="${MODEL_NAME:-pi05_libero}"
GPU_ID="${GPU_ID:-0}"
POLICY_HOST="127.0.0.1"
POLICY_PORT="${POLICY_PORT:-$((8000 + 10#$GPU_ID))}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-20}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-10}"
read -r -a batch_sizes <<< "${BATCH_SIZES:-$MAX_BATCH_SIZE}"
BATCH_TIMEOUT="${BATCH_TIMEOUT:-0.01}"
TRIALS="${TRIALS:-10}"
seed="${SEED:-7}"
read -r -a suites <<< "${SUITES:-libero_spatial libero_object libero_goal libero_10 libero_90}"
read -r -a steps <<< "${PLAN_STEPS:-1 4 7 10}"
read -r -a delays <<< "${INFERENCE_DELAYS:-0 1 2 4}"
read -r -a rtc_modes <<< "${USE_RTC:-false true}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/results/libero/suc_rate_sweep_$(date +%Y%m%d_%H%M%S)}"
SERVER_LOG="${SERVER_LOG:-$OUTPUT_ROOT/server.log}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PWD/data/libero_config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$PWD/src:$PWD/3rdparty/AutoHorizon/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"

for step in "${steps[@]}"; do
    [[ "$step" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid plan steps: $step" >&2; exit 1; }
done
for delay in "${delays[@]}"; do
    [[ "$delay" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid inference delay: $delay" >&2; exit 1; }
done
for rtc in "${rtc_modes[@]}"; do
    [[ "$rtc" == false || "$rtc" == true ]] || { echo "Invalid USE_RTC value: $rtc (expected false or true)" >&2; exit 1; }
done
for suite in "${suites[@]}"; do
    case "$suite" in
        libero_spatial|libero_object|libero_goal|libero_10|libero_90) ;;
        *) echo "Invalid suite: $suite" >&2; exit 1 ;;
    esac
done
for value in "$TRIALS" "$MAX_PARALLEL_TASKS" "$MAX_BATCH_SIZE"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Expected positive count, got: $value" >&2; exit 1; }
done
if (( ${#steps[@]} == 0 || ${#delays[@]} == 0 || ${#rtc_modes[@]} == 0 || ${#suites[@]} == 0 )); then
    echo "Sweep dimensions must not be empty" >&2
    exit 1
fi

# Refuse an occupied port instead of accidentally evaluating another server.
"$SIM_PYTHON" - "$POLICY_PORT" <<'PYPORT'
import socket
import sys
with socket.socket() as sock:
    sock.bind(("0.0.0.0", int(sys.argv[1])))
PYPORT

mkdir -p "$OUTPUT_ROOT" "$(dirname -- "$SERVER_LOG")"
server_pid=""
cleanup() {
    exit_status=$?
    trap - EXIT INT TERM
    if [[ -n "$server_pid" ]]; then
        echo "Stopping policy server (PID $server_pid)"
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for ((attempt=0; attempt<10; attempt++)); do
            kill -0 -- "-$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    exit "$exit_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid "$POLICY_PYTHON" -u -m robort.server \
    --host-addr "$POLICY_HOST" --port "$POLICY_PORT" \
    --policy-config.model-name "$MODEL_NAME" --policy-config.model-dir "$CKPT_DIR" \
    --policy-config.max-batch-size "$MAX_BATCH_SIZE" \
    --policy-config.batch-sizes "${batch_sizes[@]}" \
    --timeout "$BATCH_TIMEOUT" >"$SERVER_LOG" 2>&1 &
server_pid=$!
echo "Starting policy server on GPU $GPU_ID, port $POLICY_PORT; log: $SERVER_LOG"

# Robort accepts websocket connections after model loading and warmup.
if ! "$SIM_PYTHON" - "$POLICY_PORT" "$server_pid" "$SERVER_START_TIMEOUT" <<'PYREADY'
import os
import sys
import time
from websockets.sync.client import connect
from websockets.exceptions import WebSocketException

port, pid, timeout = map(int, sys.argv[1:])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("Policy server exited during startup")
    try:
        with connect(f"ws://127.0.0.1:{port}", open_timeout=2, close_timeout=2):
            print("Policy server is ready")
            break
    except (OSError, TimeoutError, WebSocketException):
        pass
    time.sleep(1)
else:
    raise SystemExit(f"Policy server was not ready within {timeout} seconds")
PYREADY
then
    tail -n 50 "$SERVER_LOG" >&2
    exit 1
fi

# Discover the usable horizon and validate RTC with a supported combination.
"$SIM_PYTHON" - "$POLICY_HOST" "$POLICY_PORT" "$OUTPUT_ROOT/action_horizon.txt" "${steps[*]}" "${delays[*]}" "${rtc_modes[@]}" <<'PY'
from contextlib import closing
from pathlib import Path
import sys
import numpy as np
from benchmark.libero import ACTION_HORIZON
from robort.client import WebsocketClientPolicy
from robort.schemas import InferenceRequest

host, port = sys.argv[1], int(sys.argv[2])
horizon_path = Path(sys.argv[3])
steps = list(map(int, sys.argv[4].split()))
delays = list(map(int, sys.argv[5].split()))
with closing(WebsocketClientPolicy(host=host, port=port)) as client:
    observation = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "Pick up the black bowl.", "episode": 0, "multi_actions": 0,
    }
    result = client.infer(InferenceRequest(observation=observation))
    horizon = result.actions.shape[0]
    usable_horizon = min(horizon, ACTION_HORIZON)
    valid_pairs = [(s, d) for s in steps for d in delays if s + d <= usable_horizon]
    if "true" in sys.argv[6:] and valid_pairs:
        steps, delay = max(valid_pairs, key=lambda pair: sum(pair))
        required = steps + delay
        if result.rtc_prev_actions is None:
            raise SystemExit("RTC requires a server that returns model-space rtc_prev_actions")
        result = client.infer(InferenceRequest(
            observation=observation, inference_type="rtc",
            previous_action=result.rtc_prev_actions, rtc_s_param=steps,
            rtc_d_param=delay, max_execution_horizon=required,
        ))
        if len(result.actions) < required:
            raise SystemExit("RTC response is too short for the requested sweep")
horizon_path.write_text(f"{usable_horizon}\n")
print(f"Server prediction horizon: {horizon}; benchmark usable horizon: {usable_horizon}")
PY
read -r action_horizon < "$OUTPUT_ROOT/action_horizon.txt"

# Estimate work in parallel task waves, so libero_90 gets more weight than
# the 10-task suites. Actual runtime also varies with episode length and RTC.
task_waves() {
    local tasks=10
    [[ "$1" != libero_90 ]] || tasks=90
    echo "$(( (tasks + MAX_PARALLEL_TASKS - 1) / MAX_PARALLEL_TASKS ))"
}
format_duration() {
    local seconds=$1
    printf '%dh %02dm %02ds' "$((seconds / 3600))" "$((seconds / 60 % 60))" "$((seconds % 60))"
}
combinations=0
for step in "${steps[@]}"; do
    for delay in "${delays[@]}"; do
        if (( step + delay > action_horizon )); then
            echo "Skipping plan_steps=$step, inference_delay=$delay (all suites/RTC modes): requires $((step + delay)) actions, available $action_horizon" | tee -a "$OUTPUT_ROOT/progress.log"
            continue
        fi
        combinations=$((combinations + ${#rtc_modes[@]}))
    done
done
if (( combinations == 0 )); then
    echo "No supported combinations to evaluate."
    exit 0
fi
total_runs=$(( ${#suites[@]} * combinations ))
total_work=0
for suite in "${suites[@]}"; do
    total_work=$((total_work + $(task_waves "$suite") * combinations))
done
completed_runs=0
completed_work=0
evaluation_seconds=0
echo "Sweep: $total_runs runs; elapsed $(format_duration "$((SECONDS - sweep_started))"); ETA available after the first run."

for suite in "${suites[@]}"; do
    for step in "${steps[@]}"; do
        for delay in "${delays[@]}"; do
            if (( step + delay > action_horizon )); then continue; fi
            for rtc in "${rtc_modes[@]}"; do
                run_dir="$OUTPUT_ROOT/$suite/steps_$step/delay_$delay/rtc_$rtc/seed_$seed"
                if [[ -e "$run_dir" ]]; then
                    echo "Refusing to overwrite existing run: $run_dir" >&2
                    exit 1
                fi
                mkdir -p "$run_dir"
                rtc_flag=--args.no-use-rtc
                if [[ "$rtc" == true ]]; then rtc_flag=--args.use-rtc; fi
                echo "[$((completed_runs + 1))/$total_runs] Evaluating $suite: plan_steps=$step, inference_delay=$delay, use_rtc=$rtc, seed=$seed, trials/task=$TRIALS"
                run_started=$SECONDS
                "$SIM_PYTHON" -u -m benchmark.libero \
                    --args.host "$POLICY_HOST" --args.port "$POLICY_PORT" \
                    --args.task-suite-name "$suite" \
                    --args.replan-steps "$step" --args.inference-delay "$delay" "$rtc_flag" \
                    --args.max-parallel-tasks "$MAX_PARALLEL_TASKS" \
                    --args.num-trials-per-task "$TRIALS" --args.seed "$seed" \
                    --args.video-out-path "$run_dir/videos" \
                    --args.results-path "$run_dir/results.json" \
                    --args.fail-on-error 2>&1 | tee "$run_dir/eval.log"
                run_seconds=$((SECONDS - run_started))
                evaluation_seconds=$((evaluation_seconds + run_seconds))
                completed_runs=$((completed_runs + 1))
                completed_work=$((completed_work + $(task_waves "$suite")))
                remaining_seconds=$((evaluation_seconds * (total_work - completed_work) / completed_work))
                echo "Progress: $completed_runs/$total_runs runs | last run $(format_duration "$run_seconds") | elapsed $(format_duration "$((SECONDS - sweep_started))") | estimated remaining $(format_duration "$remaining_seconds")" | tee -a "$OUTPUT_ROOT/progress.log"
            done
        done
    done
done

"$SIM_PYTHON" - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
settings = ["suite", "plan_steps", "inference_delay", "use_rtc", "seed"]
with (root / "per_task.csv").open("w", newline="") as tasks_file, \
     (root / "success_rate_profile.csv").open("w", newline="") as summary_file:
    tasks_writer = csv.writer(tasks_file)
    summary_writer = csv.writer(summary_file)
    tasks_writer.writerow(settings + ["task_id", "task", "successes", "episodes", "success_rate"])
    summary_writer.writerow(settings + ["successes", "episodes", "success_rate", "success_percent"])
    for path in sorted(root.glob("*/steps_*/delay_*/rtc_*/seed_*/results.json")):
        data = json.loads(path.read_text())
        config = [data["suite"], data["replan_steps"], data["inference_delay"], data["use_rtc"], data["seed"]]
        summary_writer.writerow(config + [data["successes"], data["episodes"], data["success_rate"], data["success_rate"] * 100])
        for task in data["tasks"]:
            tasks_writer.writerow(config + [task["task_id"], task["task"], task["successes"],
                                            task["episodes"], task["success_rate"]])
print(f"Results saved to {root}")
PY
echo "Sweep completed in $(format_duration "$((SECONDS - sweep_started))")."
