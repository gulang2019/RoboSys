# Profiling validation

Validated on an RTX 4090 after running `bash scripts/profile/setup.sh`.
Setup passed both dependency checks, native FlashRT imports, CUDA runtime checks,
and reuse of the existing pi05_libero checkpoint. System packages and checkpoint
conversion were not rerun in this validation.

OpenPI produced 20 stage rows: batches 1 and 2, full and half SM allocation,
with CUDA graphs enabled. FlashRT produced 5 stage rows: batch 1, full SM
allocation, FP16 eager execution. All stage mean latencies and energy estimates
are finite, and no hardware or policy configuration was skipped. Each run used
one warmup and two measured iterations; these are execution checks, not stable
performance benchmarks. Parameter/FLOP/memory estimates remain unavailable (NaN).

Reproduce from the repository root:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=0 \
  .venv-openpi/bin/python -m robort.profile.main \
  --backend openpi --power-perc 1.0 --sm-perc 1.0 0.5 \
  --num-views 2 --image-resolution 224 --precision bf16 --prompt-len 200 \
  --batch-sizes 1 2 --num-steps 3 --use-cuda-graph \
  --num-warmup 1 --num-iter 2 --output-dir profile/commit-validation/openpi

PYTHONPATH=src LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 CUDA_VISIBLE_DEVICES=0 \
  .venv-profile/bin/python -m robort.profile.main \
  --backend flash_rt --power-perc 1.0 --sm-perc 1.0 \
  --num-views 2 --image-resolution 224 --precision fp16 --prompt-len 200 \
  --batch-sizes 1 --num-steps 3 --no-use-cuda-graph \
  --num-warmup 1 --num-iter 2 --output-dir profile/commit-validation/flashrt
```

Tests: 129 profiling/setup/CUDA tests passed (3 optional tests skipped), plus
17 OpenPI unit tests passed. Shell syntax and staged whitespace checks passed.
