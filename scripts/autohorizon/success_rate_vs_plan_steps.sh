#!/usr/bin/env bash
# Starts its own local policy server and stops it when the sweep exits.
# Example quick check: TRIALS=1  PLAN_STEPS="1 5" bash success_rate_vs_plan_steps.sh
# Figure 1 covers four suites; optionally include libero_90 through SUITES.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

POLICY_PYTHON="${POLICY_PYTHON:-$PWD/.venv/bin/python}"
CKPT_DIR="${CKPT_DIR:-$PWD/../../checkpoints/pi05_libero_pytorch_fixed}"
GPU_ID="${GPU_ID:-0}"
SIM_PYTHON="${SIM_PYTHON:-$PWD/examples/libero/.venv/bin/python}"
POLICY_HOST="127.0.0.1"
POLICY_PORT="${POLICY_PORT:-$((8000 + 10#$GPU_ID))}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

TRIALS="${TRIALS:-10}"
seed="${SEED:-7}"
read -r -a suites <<< "${SUITES:-libero_spatial libero_object libero_goal libero_10}"
read -r -a steps <<< "${PLAN_STEPS:-1 4 7 10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/results/libero/horizon_sweep_$(date +%Y%m%d_%H%M%S)}"
SERVER_LOG="${SERVER_LOG:-$OUTPUT_ROOT/server.log}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$PWD/third_party/libero:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

max_steps=0
for step in "${steps[@]}"; do
    [[ "$step" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid horizon: $step" >&2; exit 1; }
    if (( step > max_steps )); then max_steps="$step"; fi
done

# Refuse an occupied port instead of accidentally evaluating another server.
"$SIM_PYTHON" - "$POLICY_PORT" <<'PYPORT'
import socket
import sys
with socket.socket() as sock:
    sock.bind(("0.0.0.0", int(sys.argv[1])))
PYPORT

mkdir -p "$OUTPUT_ROOT" e_step_files "$(dirname -- "$SERVER_LOG")"
server_pid=""
cleanup() {
    exit_status=$?
    trap - EXIT INT TERM
    if [[ -n "$server_pid" ]]; then
        echo "Stopping policy server (PID $server_pid)"
        # setsid gives this server its own process group, including workers.
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

setsid "$POLICY_PYTHON" -u scripts/serve_policy.py \
    --env LIBERO --port "$POLICY_PORT" \
    policy:checkpoint --policy.config pi05_libero \
    --policy.dir "$CKPT_DIR" >"$SERVER_LOG" 2>&1 &
server_pid=$!
echo "Starting policy server on GPU $GPU_ID, port $POLICY_PORT; log: $SERVER_LOG"

# The HTTP endpoint becomes available after the policy has loaded.
if ! "$SIM_PYTHON" - "$POLICY_PORT" "$server_pid" "$SERVER_START_TIMEOUT" <<'PYREADY'
import os
import sys
import time
import urllib.error
import urllib.request
port, pid, timeout = map(int, sys.argv[1:])
deadline = time.monotonic() + timeout
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("Policy server exited during startup")
    try:
        with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
            if response.status == 200:
                print("Policy server is ready")
                break
    except (OSError, urllib.error.URLError):
        pass
    time.sleep(1)
else:
    raise SystemExit(f"Policy server was not ready within {timeout} seconds")
PYREADY
then
    tail -n 50 "$SERVER_LOG" >&2
    exit 1
fi

# Verify the actual server output, rather than assuming its configuration.
"$SIM_PYTHON" - "$POLICY_HOST" "$POLICY_PORT" "$max_steps" <<'PY'
import sys
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

host, port, required = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
client = WebsocketClientPolicy(host=host, port=port)
result = client.infer({
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(8, dtype=np.float32),
    "prompt": "Pick up the black bowl.", "episode": 0, "multi_actions": 0,
})
horizon = result["actions"].shape[0]
if horizon < required:
    raise SystemExit(f"Server predicts {horizon} actions; sweep requires >= {required}. "
                     "Restart the server with the appropriate action_horizon.")
print(f"Server prediction horizon: {horizon}")
PY

for suite in "${suites[@]}"; do
    for step in "${steps[@]}"; do
        run_dir="$OUTPUT_ROOT/$suite/steps_$step/seed_$seed"
        if [[ -e "$run_dir" ]]; then
            echo "Refusing to overwrite existing run: $run_dir" >&2
            exit 1
        fi
        mkdir -p "$run_dir"
        echo "Evaluating $suite: execution horizon=$step, seed=$seed, trials/task=$TRIALS"
        "$SIM_PYTHON" -u examples/libero/main.py \
            --args.host "$POLICY_HOST" --args.port "$POLICY_PORT" \
            --args.task-suite-name "$suite" \
            --args.replan-steps "$step" \
            --args.num-trials-per-task "$TRIALS" --args.seed "$seed" \
            --args.video-out-path "$run_dir/videos" \
            --args.results-path "$run_dir/results.json" \
            --args.fail-on-error 2>&1 | tee "$run_dir/eval.log"
        # The existing evaluator uses np.save, which appends .npy to .npz.
        cp "e_step_files/${suite}_e_steps.npz.npy" "$run_dir/execution_steps.npy"
    done
done

"$SIM_PYTHON" - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import pathlib
import statistics
import sys
from collections import defaultdict

root = pathlib.Path(sys.argv[1])
groups = defaultdict(list)
with (root / "per_task.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["suite", "plan_steps", "seed", "task_id", "task", "successes", "episodes", "success_rate"])
    for path in sorted(root.glob("*/steps_*/seed_*/results.json")):
        data = json.loads(path.read_text())
        groups[data["suite"], data["replan_steps"]].append(data["success_rate"] * 100)
        for task in data["tasks"]:
            writer.writerow([data["suite"], data["replan_steps"], data["seed"],
                             task["task_id"], task["task"], task["successes"],
                             task["episodes"], task["success_rate"]])
with (root / "success_rate_vs_plan_steps.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["suite", "plan_steps", "mean_success_percent", "std_success_percent", "repeats"])
    for (suite, steps), rates in sorted(groups.items()):
        writer.writerow([suite, steps, statistics.mean(rates),
                         statistics.stdev(rates) if len(rates) > 1 else 0.0, len(rates)])
print(f"Results saved to {root}")
PY
