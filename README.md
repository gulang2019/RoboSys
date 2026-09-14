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
