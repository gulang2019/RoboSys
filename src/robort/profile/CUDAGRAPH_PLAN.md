# CUDA graphs for the five-stage runner

The runner now measures the public policy stages in dependency order, once per
stage per iteration. Setup and at least one full initialization pass are outside
measurement. No model or native handle is deep-copied. Public-method timings
include CPU dispatch, transfers and completion, while stage metadata without an
inference-side provider is NaN. `use_cuda_graph=True` currently raises explicitly.
The older `profile/policies.py` graph backend remains separate from this runner.

## 1. Separate host work from capture-safe GPU functions

Keep `preprocess` and `postprocess` eager. Within FlashRT `embed`, perform prompt
tokenization, pipeline selection, CPU image normalization and input upload outside
capture. Capture `pipeline.vision_encoder` only. For `encode`, capture language
embedding restoration and `pipeline.transformer_encoder`. For `decode`, capture
copying a persistent noise input into the diffusion buffer followed by all steps
of `pipeline.transformer_decoder`. Generate noise and download actions outside
capture. Preserve the existing public method names and stage-order checks; their
Python logic must execute on every call, outside graph replay.

The captured bodies must not contain stream/device synchronization, `.cpu()`,
NumPy conversion, tokenizer calls, dynamic pipeline construction or telemetry.
Keep synchronization in the public wrappers/profiler after replay. Do not wrap
entire current `embed`/`decode` methods in `torch.cuda.graph`.

## 2. Give the policy an explicit prepared execution owner

Add a preparation method accepting the supplied stream and an example request.
Allocate persistent input/output/noise storage and initialize the prompt pipeline
before capture. Keep weights, attention buffers, GEMM runner and graph-owned
storage alive for the complete execution lifetime. Reuse the capture/cleanup
pattern in `_prepare_stage_targets`, but do not call its synchronizing target
wrapper from inside a capture.

Start with three independent graph pools (vision, encoder, decoder). Warm the
bodies in dependency order on the supplied green stream, capture each on that
same stream, then replay the full chain before collecting measurements. Publish
prepared execution only after all captures succeed; reset partial graphs on error.
An explicit `close()` must synchronize, reset graphs, and release execution
buffers before `prepare_hardware_env` destroys the stream/context. Weight caching
can follow later, with weights loaded outside temporary green contexts.

## 3. Key graphs by execution shape and lifetime

Start with one graph set for one fixed configuration: device, precision, view
count, actual tokenized prompt length, batch size, chunk size and denoising steps.
The inference prompt length is determined by tokenization; it is not necessarily
`PolicyConfig.prompt_len`. Prompt values with the same shape can update persistent
language storage. A changed shape requires a newly prepared pipeline/graph set.
Never reuse a graph after its underlying pipeline/storage or green context has
been released. Rebuild per hardware-environment invocation initially; avoid a
global graph cache until ownership is explicit.

## 4. Wire the flag and keep measurement boundaries comparable

With `use_cuda_graph=False`, execute the same GPU bodies eagerly. With it enabled,
replay the three prepared graphs. Continue measuring all five public stages with
the runner's wall-clock/energy profiler, including the same host work and copies
in both modes. Report graph preparation separately if needed. GPU-event timings
can be added as a separate metric; do not substitute them for public-stage wall
latency or sum stage means and call that end-to-end throughput. CPU-only stages'
energy remains device-wide telemetry, not energy attributable to CPU computation.

OpenPI needs separate preparation: static tensors/cache storage, a fixed Python
`for` loop for diffusion instead of a GPU-tensor `while` condition, and removal
of dynamic cache allocations from captured bodies. Implement FlashRT first and
reject unsupported graph backends explicitly.

## 5. Acceptance checks

- Compare eager and graph raw actions using identical images, prompts and noise.
- Change images, state, prompt contents and noise between replays to expose stale inputs.
- Test changed token length, chunk size and step count rebuild the execution.
- Run repeated contexts at full and half SM allocation; verify current stream and
  resource teardown on success, capture failure and inference failure.
- Measure eager versus replay using identical iteration counts and power/SM settings;
  exclude loading, warmup and capture from the reported stage samples.

PyTorch's constraints require stable addresses/shapes and exclude CPU work and
CPU/GPU synchronization from capture:
https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
