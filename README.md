# TODO
- [x] Implement the batched policy server.
- [] Update `benchmark/suc_rate_profile.sh` to multiple gpus.  
- [ ] Profile the utility function as a function of delay and replan steps.
    - [] Libero, OpenPI (Siyuan Working on it)
    - [] Libero, Groot-N1.6
    - [] Aloha Sim, OpenPI0 (Xunyuan, can you work on this)

# Robort LIBERO benchmark

A batched policy server adapted from Armory and a LIBERO success-rate sweep over
request intervals, simulated inference delays, and RTC conditioning. Small
inference groups are padded to configured batch sizes to limit JAX compilation
shapes. Responses retain the original request order and model-space RTC history.

## Setup

On Ubuntu/Linux with a working NVIDIA driver compatible with CUDA 12:

```bash
git clone https://github.com/gulang2019/RoboSys.git
cd RoboSys
bash setup.sh
```

Setup installs system packages through `sudo`, installs `uv` if needed,
initializes the pinned Armory and AutoHorizon submodules, clones a pinned LIBERO
checkout, applies the patches in `scripts/benchmark/`, and syncs Armory's locked
Python 3.11 environment. It configures LIBERO under `data/libero_config`, downloads
the JAX `pi05_libero` checkpoint, and checks imports, CUDA, and all task initial
states. It does not launch evaluation or validate full model inference.

Allow substantial disk space for the CUDA environment and the checkpoint (the
OpenPI download cache and destination each hold a copy). Run `bash setup.sh --help`
for options, including `SKIP_SYSTEM_DEPS=1`, `SKIP_CHECKPOINT=1`, and
`SKIP_GPU_CHECK=1`. Existing compatible patches and checkpoints are reused.
Do not run setup while a sweep uses the same environment.

The separate `setup_autohorizon.sh` workflow is for the older AutoHorizon
experiment; it is not required for this benchmark. Its simulator checkout is
reused, but its Python environment and evaluator changes are not needed.

## Run

```bash
bash benchmark/suc_rate_profile.sh
```

The script starts a local server, validates its prediction horizon and RTC,
runs supported combinations, writes CSV reports, and stops the server on exit.
The default sweep covers all five LIBERO suites, plan steps `1 4 7 10`, delays
`0 1 2 4`, RTC off/on, and 10 trials per task at seed 7. Unsupported combinations
(`plan_steps + inference_delay > usable action horizon`) are logged and skipped.
With a 10-action horizon this gives 120 suite/configuration runs.

Outputs are under `results/libero/suc_rate_sweep_<timestamp>/`: `server.log`,
`progress.log`, per-run `eval.log`, `results.json`, videos, and final
`success_rate_profile.csv` / `per_task.csv`. The ETA updates after each completed
run and weights suites by parallel task waves; episode lengths and configuration
changes can make early estimates inaccurate. Existing run directories are never
overwritten; automatic resume is not implemented.

Example small run:

```bash
TRIALS=1 SUITES=libero_spatial PLAN_STEPS="1 4" \
INFERENCE_DELAYS="0 2" USE_RTC="false true" \
bash benchmark/suc_rate_profile.sh
```

Use `GPU_ID`, `CKPT_DIR`, `OUTPUT_ROOT`, `MAX_PARALLEL_TASKS`, `MAX_BATCH_SIZE`,
`BATCH_SIZES`, and `BATCH_TIMEOUT` to configure execution. `BATCH_SIZES="4 10"`
warms those sizes and pads each SYNC/RTC group to the smallest fitting size;
the largest bucket must equal `MAX_BATCH_SIZE`. By default the only bucket is
`MAX_BATCH_SIZE` (10). Server log batch sizes count real requests before padding.
RTC requires the JAX checkpoint, rather than a PyTorch checkpoint.

### Examples to play with 
```bash
.venv-openpi/bin/python examples/robort_inference.py
```
Its good to step through the program with vscode. Script provided in `.vscode/launch.json`

### Local or websocket evaluation clients

The evaluator accepts `--client-args` as a JSON object (Python dictionary literals
are also accepted). Use `type` to select `local` or `websocket`; omitting it
selects `websocket` and preserves the existing `--args.host` / `--args.port` flags.
Host and port inside the dictionary override those flags.

The full `bash setup.sh` also installs the simulator dependencies into
`.venv-openpi`, preserving its patched Transformers 4.53.2 and Torch 2.7.1.
To augment an existing OpenPI environment and LIBERO checkout only:

```bash
bash setup.sh --openpi-libero
```

This checks package compatibility, the Transformers patch, all LIBERO initial
states, and CUDA availability. Use `SKIP_GPU_CHECK=1` for setup without a GPU.
It reuses the system rendering libraries installed by the full setup.

For local PyTorch OpenPI inference, run from the repository root in an environment
with both OpenPI and LIBERO/MuJoCo dependencies installed:

```bash
PYTHONPATH=src:3rdparty/AutoHorizon/third_party/libero \
LIBERO_CONFIG_PATH="$PWD/data/libero_config" MUJOCO_GL=egl JAX_PLATFORMS=cpu \
.venv-openpi/bin/python -m benchmark.libero \
  --client-args '{"type":"local","model_dir":"checkpoints/pi05_libero_pytorch","device":"cuda","num_steps":10}' \
  --args.task-suite-name libero_spatial --args.num-trials-per-task 1
```

The local client loads one model per evaluation and reuses it across tasks and
episodes. Tasks run sequentially regardless of `--args.max-parallel-tasks`, since
the policy reuses mutable caches. RTC is unsupported. Local options are
`model_dir`, `model_name`, `device`, `num_steps`, `encode_keep_rate`, and `debug_dir`; the checkpoint must contain
PyTorch weights and the normalization assets required by OpenPI.

With `encode_keep_rate` enabled, debugging images are stored in a new
`debug/run-*/` directory, whose path is logged. Closing the local client at the
end of evaluation retains only `combined.mp4`, `scores.cdf.png`, and `attention.cdf.png`. The video has
labeled global/wrist rows and full/binary/shaded/text-attention columns, synchronized by inference
index. Each CDF overlays a labeled line per inference step. `scores.cdf.png` uses
the L2-based visual embedding MSE; the initial full refresh has no difference
scores. `attention.cdf.png` uses text-to-image attention, including the initial
full refresh. Intermediate PNGs and individual camera videos are removed only
after all final outputs are saved successfully.
Binary panels black out skipped patches; shaded panels use continuous score-based
brightness. Text-attention panels average attention from non-padding language tokens
(including any tokenized state) to each image patch over encoder layers and heads.
Pruned query positions are mapped back to their original token slots. Higher
attention is brighter, with a shared scale across both cameras for each frame.
Collecting these attention matrices adds debugging memory overhead.
Videos contain one frame per policy inference at 10 FPS (not simulator
real time). Frame numbering continues across episodes; each episode starts with a
full refresh. If using `OpenPIPolicy` directly, call `policy.export_debug_videos()`
when finished. PNG generation and final video encoding add debugging overhead.
Set `debug_dir` in `--client-args` for deterministic `episode_000/`, `episode_001/`,
etc. directories. Each episode then has its own video and two CDFs; prior episodes
are finalized on reset, and the final episode is finalized on close. Existing
episode directories are never overwritten.

### Pruning sweep

```bash
.venv-openpi/bin/python -m benchmark.sweep_pruning \
  --gpus 0 1 --output-dir results/libero/pruning_sweep
```

This keeps one worker process and one loaded model per GPU across configurations.
Each configuration resets episode state and updates pruning/decoding settings; a
failed worker is restarted before the next configuration. The 64 configurations cover
keep rates `1/6, 1/3, 1/2, 2/3` and decoding steps `2, 4, 7, 10`, with 10 episodes
per configuration (640 episodes total). The four tasks span:

| Suite / task ID | Task | Diversity |
|---|---|---|
| `libero_spatial / 0` | Bowl between plate and ramekin → plate | Spatial relation and distractors |
| `libero_object / 7` | Milk → basket | Different object geometry and receptacle |
| `libero_goal / 5` | Push plate to front of stove | Non-grasping contact manipulation |
| `libero_10 / 3` | Bowl → bottom drawer, then close it | Multi-stage articulated-object interaction |

The sweep keeps replan interval 5, seed 7, and the first 10 initial states fixed;
PyTorch and NumPy are seeded for each configuration. Change `--checkpoint`,
`--episodes`, `--seed`, or `--replan-steps` if needed. These artifacts are for
success-rate/debugging comparisons, not latency measurement.

`success_rates.csv` is updated atomically after each run, including failed and
pending configurations. Rates are fractions in `[0, 1]`; failed runs have blank
rates, rather than being counted as zero-success evaluations. `manifest.json`
records the full experiment settings. Example output:

```text
results/libero/pruning_sweep/
  manifest.json
  success_rates.csv
  libero_spatial/task_000_bowl_between_objects/keep_1of6/decode_02/
    attempt_001/
      command.json
      eval.log
      results.json
      rollouts/                  # standard episode videos, success/failure in filenames
      debug/episode_000/
        combined.mp4
        scores.cdf.png
        attention.cdf.png
      debug/episode_001/...
```

To resume, repeat the command with `--resume`. Completed runs are skipped only
when results and all episode artifacts exist. Incomplete/failed runs restart all
episodes in a new `attempt_002/` directory, preserving previous logs. Settings
must match the saved manifest. `--dry-run` writes the manifest and 64-row CSV
without starting a GPU job; use `--resume` with that output directory to execute.

For an existing websocket server, replace the client arguments with:

```bash
--client-args '{"type":"websocket","host":"127.0.0.1","port":8000}'
```

Websocket clients also accept `api_key` and retain parallel task evaluation and
server-provided RTC support. The shell sweep continues to use websocket clients.

For lower GPU memory usage (same evaluation coverage, reduced concurrency):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false MAX_BATCH_SIZE=1 BATCH_SIZES=1 \
MAX_PARALLEL_TASKS=2 bash benchmark/suc_rate_profile.sh
```

## Validation

After setup:

```bash
3rdparty/armory/.venv/bin/python -m pytest tests -q
```

Tests cover scheduling, client/server serialization, batching and padding,
RTC history, simulation delays, initial-state loading, and sweep control flow.
Most tests use fake policies/environments; they do not establish GPU memory
requirements or model success rates. Use the small run above for GPU validation.

## Further experiments

- Measure success rate under jitter.
- Compare RTC with normalized action chunking.
- Profile batching throughput and latency.
