# FlashRT GPU profiling

The runner currently supports the BF16 Pi0.5 LIBERO FlashRT backend on RTX
SM89/SM120, with two 224×224 views. Use `backend="flash_rt"`.

```python
from robort.profile.runner import Runner
from robort.profile.schemas import HardwareConfig, PolicyConfig, RunnerConfig

policy = PolicyConfig(
    backend="flash_rt", precision="bf16",
    model_dir="checkpoints/pi05_libero_pytorch",
    batch_sizes={"embed": [1, 2], "encode": [1], "decode": [1, 4]},
    use_cuda_graph=True,
)
hardware = [HardwareConfig(1.0, 1.0), HardwareConfig(1.0, 0.5)]
with Runner(RunnerConfig(num_warmup=3, num_iter=20)) as runner:
    runner.init_backend(policy, hardware)
    for hw in hardware:
        result = runner.profile_policy(hw, policy)
        print(result.stages["decode"].lat[4])  # mean/std milliseconds
```

Initialization loads one policy and captures its batch-size variants on each
unique device/SM-fraction stream. Power settings reuse these streams. All hardware
configurations for a policy must refer to the same CUDA device. Initializing the
next policy configuration drains and releases the previous policy before loading
its replacement; green streams persist until `close()`.

`batch_sizes` maps stages to sizes. By default only the three GPU stages are
measured, at B=1. Include `preprocess` or `postprocess` to measure the CPU stages.
Missing GPU stages get a B=1 pipeline for input preparation. A profiling call can
select a subset of initialized GPU sizes and can select CPU sizes independently.

The runner generates each stage's inputs outside measurement, executing upstream
stages in supported batches. Partial batches are padded and excess outputs are
discarded. Thus decode at B=16 does not require encode or embed graphs at B=16.
The runner retains intermediate tensors and completes input preparation before
measurement. It serializes work and owns all synchronization; the backend does
not manage dependency events.

## Measurements

- Latency is public-stage wall time: host dispatch, input/output copies, compute,
  and completion on the selected stream. It is not GPU graph-only latency.
- Loading, autotuning, capture, input preparation, and warmup are excluded.
- Each stage is measured independently. Stage means are not concurrent pipeline
  throughput or an end-to-end request benchmark.
- Memory is peak additional Torch CUDA allocation per call. Preallocated stage
  buffers, weights, graph storage, and native GEMM workspaces are excluded.
- Energy uses device-wide endpoint power samples. Short stages may be below the
  telemetry refresh interval; unsupported readings produce NaN.
- `HardwareConfig` binds to the current GPU. SM allocations follow CUDA green
  context rounding; power limits are restored after each profiling run.
- Close drains streams, releases graph handles and policies, then destroys green
  streams. Model parameter/FLOP/weight-memory metadata is currently unavailable.

## Command line and tests

FlashRT is pinned to the upstream commit `eaf90192`; the decoder override hook
is shipped in `patches/flashrt-pi05-decoder-hook.patch`. The setup script applies
it idempotently. For an existing environment, apply it without reinstalling:

```sh
git submodule update --init -- 3rdparty/FlashRT
bash scripts/profile/apply-flashrt-patch.sh
```

If pulling the earlier `5c03d88` revision reports `not our ref 1400361a...`, first
fetch the RoboSys fix without recursive submodule fetching:

```sh
git -c submodule.recurse=false pull --no-recurse-submodules
```

Then run the submodule update and patch commands above. Local edits in the
submodule are preserved; resolve any checkout/patch conflicts before proceeding.

Set `LD_LIBRARY_PATH` before launching Python. Activating `.venv-profile` alone
is insufficient: FlashRT loads the unversioned `libcudart.so` independently of
PyTorch. On this machine, omitting the path selects CUDA 11.7 and can crash graph
capture even though `torch.version.cuda` reports 12.8. The backend now checks this
runtime mismatch before loading model weights.

```sh
PYTHONPATH=src LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 \
  .venv-profile/bin/python -m robort.profile.main \
  --backend flash_rt --precision bf16 --num-views 2 --image-resolution 224 \
  --embed-batch-sizes 1 2 --encode-batch-sizes 1 --decode-batch-sizes 1 4 \
  --sm-perc 1.0 0.5 --num-warmup 3 --num-iter 20

LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 ROBORT_TEST_FLASHRT_PROFILE=1 \
  .venv-profile/bin/python -m pytest tests/test_flashrt_profile.py -q
```

The sweep writes one CSV row per hardware configuration, policy configuration,
stage and batch size. The opt-in GPU tests exercise eager and captured execution
on full/half-SM green streams with the real checkpoint.
