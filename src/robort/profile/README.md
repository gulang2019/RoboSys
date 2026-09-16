# GPU profiling

`Runner.profile_policy(hardware, policy)` returns `{batch_size: PolicyProfile}`
for every size in `policy.batch_sizes` (default `[1]`). Each result contains
`preprocess`, `embed`, `encode`, `decode` and `postprocess` stage measurements.
The policy CSV contains one row per configuration, scalar `batch_size`, and stage.
Existing singleton `batch_sizes` CSV values are migrated when results are merged.

```python
from robort.profile.runner import Runner
from robort.profile.schemas import HardwareConfig, PolicyConfig, RunnerConfig

policy = PolicyConfig(
    backend="openpi", model_dir="checkpoints/pi05_libero_pytorch",
    batch_sizes=[1, 2], num_steps=10, use_cuda_graph=True,
)
with Runner(RunnerConfig(num_warmup=3, num_iter=20)) as runner:
    full = runner.profile_policy(HardwareConfig(1.0, 1.0), policy)
    half = runner.profile_policy(HardwareConfig(1.0, 0.5), policy)
    print(full[2].stages["decode"].lat_mean)  # milliseconds for batch size 2
```

## Model and compilation lifetime

The runner owns a `PolicyCache`. It retains one checkpoint/device's model and
compiled execution variants across calls. Changing power, SM fraction, batch size,
or graph/eager selection does not reload the OpenPI model. Batch shapes reuse the
same compiled functions; each new shape may need an initial compilation. A changed
execution configuration (for example denoising steps) gets another compiled
variant sharing the loaded weights. A changed checkpoint/model/device evicts the
previous model. Configurations are copied, so caller mutations cannot corrupt
existing variants. Checkpoint files replaced in place require `runner.close()`
before the next call.

Loading and initial compilation occur on the device's default stream. Compiled
functions and model storage therefore outlive each experiment's green stream.
Each graph run creates a `CudaGraphOpenPIPolicy` using the cached compiled stages,
captures fresh graphs on that run's green stream, and releases them before the
stream/context is destroyed. The original `OpenPIPolicy` stays eager. Graph capture
and warmup still occur per run, outside the measured samples; repeated identical
runs reuse the Torch compiler results. Call `runner.close()` (or use its context
manager) to release cache ownership. The runner is synchronous and not thread-safe.

`use_cuda_graph=False` uses the cached eager policy. Graph profiling supports
OpenPI and FlashRT on Thor SM110. OpenPI model precision, action horizon and
image/token shapes come from its training configuration/transforms; the generic
profile metadata fields do not override those model properties.

## Measurement conventions

- Each iteration executes all five stages in dependency order. Latencies include
  Python dispatch, transfers and completion on the selected stream. Loading,
  compilation, capture, warmup and telemetry overhead are excluded. These stage
  means are not an end-to-end throughput measurement.
- Model parameter/FLOP/memory metadata is currently NaN for this five-stage API.
- Energy estimates use endpoint device-power samples, including idle power and
  other GPU activity. Very short stages may be below the telemetry refresh rate.
- `HardwareConfig` binds to the current GPU. Power fractions use the NVML default
  power limit; SM fractions are rounded according to CUDA green-context rules.
  Each run restores the previous power limit, including on failure.
- First-time compilation can take minutes. Compilation may change floating-point
  results relative to the original eager model; graph parity is checked against
  the same compiled functions without capture.

## Environment and tests

Use `.venv-openpi` with CUDA PyTorch, OpenPI and its checkpoint. Profiling also
requires CUDA driver bindings and NVML (installed by `scripts/profile/setup.sh`):

```sh
uv pip install --python .venv-openpi/bin/python cuda-python==12.9.7 nvidia-ml-py==13.610.43
JAX_PLATFORMS=cpu .venv-openpi/bin/python -m pytest \
  tests/test_policy_profile.py tests/test_profile_policy_cache.py tests/test_profile_sweep.py -q
ROBORT_TEST_OPENPI_PROFILE=1 JAX_PLATFORMS=cpu .venv-openpi/bin/python -m pytest \
  tests/test_openpi_profile_cache.py -q -s
```

The real-checkpoint test profiles multiple batch sizes at full/half SM allocation,
checks checkpoint load and compiler call counts, and verifies that Dynamo's count
of compiled graphs does not increase across repeated hardware runs. It retains the
current NVML power limit. The sweep entry point is `python -m robort.profile.main`;
`--batch-sizes 1 2` produces separate CSV rows per batch size and stage.
