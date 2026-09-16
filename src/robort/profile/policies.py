"""Prepared, synchronous profiling targets; optional runtimes load on demand."""

from collections.abc import Callable
from contextlib import ExitStack
from copy import copy
from dataclasses import replace
from importlib import import_module
from math import prod
from os import environ
from pathlib import Path

from ..policies.flash_rt import flashrt_hardware
from .schemas import PolicyConfig, StageProfile

PolicyTargets = dict[str, Callable[[], None]]


class OpenPIBackend:
    def prepare_policies(self, policy_config: PolicyConfig, stream=None):
        raise NotImplementedError("OpenPI profiling is not implemented")


class FlashRTBackend:
    """Pi0.5 RTX/Torch component profiling with real checkpoint weights.

    Supports batch size 1, 224px images and FP16 or FP8 (BF16 residuals).
    Inputs are synthetic, including exactly ``prompt_len`` language embeddings.
    Calibration on synthetic images is for performance, not accuracy evaluation.
    Targets replay separate CUDA graphs by default; set use_cuda_graph=False
    for eager execution. Both modes use the supplied stream (or a private stream)
    and wait for completion.
    Capture and warmup happen during preparation, outside profiling. VLM includes
    restoring language inputs; action includes restoring fixed diffusion noise
    and all denoising steps. Preprocessing/model loading are outside the targets.
    The targets share buffers and must not be invoked concurrently.

    Metadata: num_params counts checkpoint tensor elements; FLOPs estimates
    dense GEMMs and attention (multiply-add = 2), excluding elementwise ops and
    setup-time style computation. Weight memory is retained per-component CUDA
    storage, including original and quantized copies. Activation memory is the
    shared resident pipeline/input/attention footprint, repeated for each stage
    (not additive). All memory uses decimal GB and excludes opaque library
    workspaces, allocator reserves and graph pools.
    """

    def __init__(self):
        self._last_policy_config = None
        self._cache_key = None
        self._prepared = None
        self._model = None
        self._model_key = None

    def prepare_policies(self, policy_config: PolicyConfig, stream=None) -> tuple[PolicyTargets, dict[str, StageProfile]]:
        config = replace(policy_config)  # Do not cache a caller-owned mutable config.
        self._validate(config)
        checkpoint = Path(config.model_dir or f"checkpoints/{config.model_name}_pytorch").expanduser().resolve()
        weights_path = checkpoint / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"FlashRT requires a PyTorch checkpoint: {weights_path}")

        torch = import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("FlashRT profiling requires CUDA")
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        hardware = flashrt_hardware(capability)
        if stream is not None and stream.device != torch.device("cuda", device):
            raise ValueError("Profiling stream must belong to the current CUDA device")
        overrides = tuple(sorted((name, value) for name, value in environ.items()
                                 if name.startswith(("FLASHRT_", "FVK_"))))
        thor_shape = ((config.num_views, config.prompt_len, config.num_steps, config.chunk_size)
                      if hardware == "thor" else None)
        model_key = (str(checkpoint), config.model_name, config.precision, device,
                     hardware, thor_shape, overrides)
        if model_key != self._model_key:
            self.release_policies()
            self._model = self._load_model(config, checkpoint, torch, capability)
            self._model_key = model_key
        key = (config, model_key, stream)
        if key != self._cache_key:
            self.release_execution()
            prepared = self._prepare(config, self._model, torch, stream)
            # Publish only after successful initialization.
            self._prepared = prepared
            self._cache_key = key
            self._last_policy_config = replace(config)
        targets, profile = self._prepared
        # Results are mutable; each experiment gets fresh result containers.
        return dict(targets), {name: replace(stage) for name, stage in profile.items()}

    def release_policies(self):
        """Release execution resources and evict the resident model weights."""
        try:
            self.release_execution()
        finally:
            self._model = None
            self._model_key = None

    def release_execution(self):
        """Release stream-dependent resources while retaining model weights."""
        if self._prepared is not None:
            targets, _ = self._prepared
            self._prepared = None
            self._cache_key = None
            with ExitStack() as cleanup:
                for target in targets.values():
                    cleanup.callback(target.close)

    @staticmethod
    def _validate(config):
        if config.backend != "flash_rt":
            raise ValueError("FlashRTBackend requires backend='flash_rt'")
        if config.model_name not in ("pi05", "pi05_libero"):
            raise NotImplementedError("FlashRT profiling currently supports pi05/pi05_libero")
        if config.precision not in ("fp16", "fp8"):
            raise NotImplementedError("FlashRT profiling supports fp16 and fp8 (BF16 residuals)")
        if (config.batch_sizes or [1]) != [1] or str(config.image_resolution) != "224":
            raise NotImplementedError("FlashRT profiling requires batch_size=1 and image_resolution=224")
        if type(config.use_cuda_graph) is not bool:
            raise ValueError("use_cuda_graph must be a boolean")
        for name in ("num_views", "prompt_len", "num_steps", "chunk_size"):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _load_model(config, checkpoint, torch, capability):
        fp16 = config.precision == "fp16"
        hardware = flashrt_hardware(capability)
        if hardware == "thor":
            module = import_module("flash_rt.frontends.torch.pi05_thor")
            frontend_type = getattr(module, "Pi05TorchFrontendThor")
            with torch.inference_mode(), torch.cuda.stream(torch.cuda.default_stream()):
                frontend = frontend_type(
                    checkpoint, num_views=config.num_views,
                    use_cuda_graph=False, autotune=0, use_fp8=not fp16,
                )
                if (frontend.steps != config.num_steps
                        or frontend.Sa != config.chunk_size):
                    raise NotImplementedError(
                        "FlashRT Thor fixes num_steps/chunk_size to "
                        f"{frontend.steps}/{frontend.Sa}; got "
                        f"{config.num_steps}/{config.chunk_size}"
                    )
                frontend._robort_hardware = "thor"
                torch.cuda.synchronize()
            params = _checkpoint_params(checkpoint)
            return module, frontend, params
        suffix = "_fp16" if fp16 else ""
        module = import_module(f"flash_rt.frontends.torch.pi05_rtx{suffix}")
        frontend_type = getattr(module, "Pi05TorchFrontendRtxFP16" if fp16 else "Pi05TorchFrontendRtx")
        # Model storage must outlive each experiment's green stream. Initialize
        # on the device's default stream and finish uploads before publishing.
        with torch.inference_mode(), torch.cuda.stream(torch.cuda.default_stream()):
            frontend = frontend_type(
                checkpoint, num_views=1, max_prompt_len=1,
                chunk_size=1, num_steps=1,
                cache_frames=1, use_fp8=not fp16,
                hardware=f"rtx_sm{capability[0]}{capability[1]}",
            )
            precision = frontend._pipeline_precision_kwargs()
            if (bool(precision["use_fp8"]) != (not fp16)
                    or bool(precision["use_fp8_decoder"]) != (not fp16)
                    or any(value for name, value in precision.items() if name.startswith("use_int8"))):
                raise ValueError("FlashRT environment overrides conflict with requested precision")
            # At one step the frontend scales these by -1. Undo that exactly;
            # each execution derives its own projections from this base copy.
            for name in ("decoder_action_out_proj_w", "decoder_action_out_proj_b"):
                frontend._ckpt_bf16[name] = -frontend._ckpt_bf16[name]
            torch.cuda.synchronize()
        # The vendor constructor also allocates execution resources. Keep only
        # its model state; every run receives fresh attention/GEMM/input buffers.
        for name in ("attn_backend", "gemm", "_img_buf", "_noise_buf", "_noise_out",
                     "_precomputed_styles"):
            setattr(frontend, name, None)
        params = _checkpoint_params(checkpoint)
        return module, frontend, params

    @staticmethod
    def _execution_frontend(config, module, model, torch):
        """Share immutable weights, with all mutable execution state owned here."""
        frontend = copy(model)
        frontend.num_views = config.num_views
        frontend.max_prompt_len = config.prompt_len
        frontend.chunk_size = config.chunk_size
        frontend._num_steps = config.num_steps
        frontend._prompt_pipeline_cache = {}
        frontend.latency_records = []
        frontend._ckpt_bf16 = dict(model._ckpt_bf16)
        for name in ("decoder_action_out_proj_w", "decoder_action_out_proj_b"):
            frontend._ckpt_bf16[name] = model._ckpt_bf16[name] * (-1.0 / config.num_steps)
        frontend._precomputed_styles = module._precompute_decoder_styles(
            frontend._ckpt_bf16, config.chunk_size, num_steps=config.num_steps)
        attention_kwargs = {"dtype": module.bf16} if config.precision == "fp16" else {}
        frontend.attn_backend = module.RtxFlashAttnBackend(
            num_views=config.num_views, encoder_seq_max=config.num_views * 256 + config.prompt_len,
            chunk_size=config.chunk_size, num_encoder_layers=module.ENC_L, **attention_kwargs)
        frontend.gemm = frontend.fvk.GemmRunner()
        frontend._img_buf = torch.empty(config.num_views, 224, 224, 3, dtype=module.bf16, device="cuda")
        frontend._noise_buf = torch.empty(config.chunk_size, 32, dtype=module.bf16, device="cuda")
        frontend._noise_out = torch.empty(config.chunk_size, 32, dtype=module.bf16, device="cuda")
        return frontend

    @staticmethod
    def _prepare(config, model_state, torch, stream=None):
        import numpy as np

        module, model, params = model_state
        if getattr(model, "_robort_hardware", None) == "thor":
            return FlashRTBackend._prepare_thor(
                config, module, model, params, torch, stream)
        stream = stream if stream is not None else torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(stream):
            frontend = FlashRTBackend._execution_frontend(config, module, model, torch)
            precision = frontend._pipeline_precision_kwargs()
            # Mirror set_prompt's construction without tokenization changing the
            # requested sequence length. Keep private FlashRT API usage here.
            pipeline = module.Pi05Pipeline(
                gemm=frontend.gemm, fvk=frontend.fvk, attn_backend=frontend.attn_backend,
                weights=frontend._build_pipeline_weights(), num_views=config.num_views,
                max_prompt_len=config.prompt_len, chunk_size=config.chunk_size,
                num_steps=config.num_steps, **precision,
            )
            frontend.pipeline = pipeline
            frontend.current_prompt_len = config.prompt_len
            indices = torch.arange(config.prompt_len, device="cuda") % frontend.embedding_weight.shape[0]
            embeds = frontend.embedding_weight[indices].contiguous()
            pipeline.set_language_embeds(embeds.view(torch.uint16).cpu().numpy())
            rng = np.random.default_rng(0)
            images = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                      for _ in range(config.num_views)]
            frontend.calibrate([{"images": images}])
            torch.cuda.synchronize()
            # The FP16 frontend calls its float16 dtype "bf16" internally.
            noise = torch.from_numpy(rng.standard_normal((config.chunk_size, 32)).astype(np.float32)).to(
                device="cuda", dtype=module.bf16)
            torch.cuda.synchronize()

            def vision():
                pipeline.vision_encoder(stream=stream.cuda_stream)

            def vlm():
                pipeline._copy_lang_embeds_to_encoder_x(stream=stream.cuda_stream)
                pipeline.transformer_encoder(stream=stream.cuda_stream)

            def action():
                frontend._copy_tensor_to_pipeline_buf_stream(noise, pipeline.input_noise_buf, stream.cuda_stream)
                pipeline.transformer_decoder(stream=stream.cuda_stream)

            from flash_rt.core.cuda_buffer import CudaBuffer
            weight_bytes = _stage_weight_bytes(frontend, torch, model=model)
            # All these buffers remain resident during every component call.
            # Report the shared footprint for each stage; do not sum it across stages.
            activation_bytes = _buffer_bytes(vars(pipeline), CudaBuffer)
            activation_bytes += _storage_bytes([
                vars(frontend.attn_backend), frontend._img_buf, frontend._noise_buf,
                frontend._noise_out, embeds, noise,
            ], torch)
            flops = _pi05_stage_flops(config)
            unknown = float("nan")
            profile = {
                name: StageProfile(params[name], flops[name], weight_bytes[name] / 1e9,
                                   activation_bytes / 1e9, unknown, unknown, unknown, unknown)
                for name in params
            }
            # Capture last so metadata failures cannot leave owned graphs behind.
            targets = _prepare_stage_targets(
                {"vis": vision, "vlm": vlm, "action": action},
                torch, stream, use_cuda_graph=config.use_cuda_graph,
            )
        return targets, profile

    @staticmethod
    def _prepare_thor(config, module, frontend, params, torch, stream=None):
        """Build SM110 stage targets from FlashRT's native Thor primitives."""
        import numpy as np

        stream = stream if stream is not None else torch.cuda.Stream()
        rng = np.random.default_rng(0)
        with torch.inference_mode(), torch.cuda.stream(stream):
            # The Thor frontend accepts token ids, which preserves the profile's
            # exact requested language length without tokenizer-dependent text.
            vocab = frontend.embedding_weight.shape[0]
            frontend.set_prompt(np.arange(config.prompt_len, dtype=np.int64) % vocab)
            images = rng.integers(
                0, 256, (config.num_views, 224, 224, 3), dtype=np.uint8)
            np.copyto(frontend._infer_images_u8_np, images)
            frontend._img_u8_buf.upload(frontend._infer_images_u8_np)
            noise = torch.from_numpy(
                rng.standard_normal((config.chunk_size, 32)).astype(np.float16)
            ).to(device="cuda")
            torch.cuda.synchronize()

            enc_bufs, enc_weights, enc_dims = frontend._runtime_encoder_spec()
            dec_bufs, dec_weights, dec_dims = frontend._runtime_decoder_spec()

            def vision():
                stream_int = stream.cuda_stream
                frontend._patch_embed_ops(stream_int, uint8_input=True)
                module.siglip_forward(
                    frontend._gemm, module.fvk, frontend._sig_bufs,
                    frontend._sig_weights, frontend._sig_dims,
                    stream=stream_int, attn=frontend._attn,
                    use_fp8=frontend.use_fp8,
                )
                frontend._postln_project_ops(stream_int)

            def vlm():
                frontend._Kc.zero_()
                frontend._Vc.zero_()
                module.encoder_forward(
                    frontend._gemm, module.fvk, enc_bufs, enc_weights,
                    enc_dims, stream=stream.cuda_stream, attn=frontend._attn,
                    use_fp8=frontend.use_fp8,
                )

            def action():
                frontend._g_noise.copy_(noise)
                module.decoder_forward(
                    frontend._ctx, module.fvk, dec_bufs, dec_weights,
                    dec_dims, stream=stream.cuda_stream, attn=frontend._attn,
                    use_fp8=frontend.use_fp8,
                )

            from flash_rt.core.cuda_buffer import CudaBuffer
            resident_bytes = _storage_bytes(vars(frontend), torch)
            resident_bytes += _buffer_bytes(vars(frontend), CudaBuffer)
            weight_bytes = {name: params[name] * (2 if config.precision == "fp16" else 1)
                            for name in params}
            activation_bytes = max(0, resident_bytes - sum(weight_bytes.values()))
            flops = _pi05_stage_flops(config)
            unknown = float("nan")
            profile = {
                name: StageProfile(params[name], flops[name], weight_bytes[name] / 1e9,
                                   activation_bytes / 1e9, unknown, unknown,
                                   unknown, unknown)
                for name in params
            }
            targets = _prepare_stage_targets(
                {"vis": vision, "vlm": vlm, "action": action},
                torch, stream, use_cuda_graph=config.use_cuda_graph,
            )
        return targets, profile


class _StageTarget:
    def __init__(self, name, stages, torch, stream):
        self.name, self.stages = name, stages
        self.torch, self.stream = torch, stream
        self.graph = None

    def __call__(self):
        if self.stages is None:
            raise RuntimeError("Profiling target has been released")
        torch, stream = self.torch, self.stream
        with torch.cuda.device(stream.device), torch.inference_mode(), torch.cuda.stream(stream):
            if self.graph is None:
                self.stages[self.name]()
            else:
                self.graph.replay()
        stream.synchronize()

    def close(self):
        try:
            try:
                if self.stream is not None:
                    self.stream.synchronize()
            finally:
                if self.graph is not None:
                    self.graph.reset()
        finally:
            self.graph = None
            self.stages = None
            self.stream = None


def _prepare_stage_targets(stages, torch, stream, *, use_cuda_graph):
    """Warm, capture and replay on the same stream; explicitly release graphs.

    Each graph has its own pool. Targets retain upstream buffers and weights
    until close(), which must be called before the supplied stream is destroyed.
    """
    stages = dict(stages)
    targets = {name: _StageTarget(name, stages, torch, stream) for name in stages}
    try:
        for _ in range(3 if use_cuda_graph else 1):
            for target in targets.values():
                target()
        if use_cuda_graph:
            for name, fn in stages.items():
                with torch.cuda.device(stream.device), torch.inference_mode():
                    target = targets[name]
                    target.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(target.graph, stream=stream):
                        fn()
                    target()
        return targets
    except BaseException:
        with ExitStack() as cleanup:
            for target in targets.values():
                cleanup.callback(target.close)
        raise


def _checkpoint_stage(name):
    name = name.removeprefix("model.")
    if name.startswith("paligemma_with_expert.paligemma.model.vision_tower."):
        return "vis"
    if name.startswith("paligemma_with_expert.paligemma."):
        return "vlm"
    if name.startswith(("paligemma_with_expert.gemma_expert.", "time_mlp_", "action_in_proj.", "action_out_proj.")):
        return "action"
    raise ValueError(f"Cannot assign checkpoint tensor to a profiling stage: {name}")


def _checkpoint_params(checkpoint):
    from safetensors import safe_open

    params = {name: 0 for name in ("vis", "vlm", "action")}
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as weights:
        for name in weights.keys():
            params[_checkpoint_stage(name)] += prod(weights.get_slice(name).get_shape())
    return params


def _weight_stage(name):
    if name == "vision_projector_w" or name == "embedding_weight" or name.startswith("encoder_"):
        return "vlm"
    if name.startswith("vision_"):
        return "vis"
    if name.startswith("decoder_"):
        return "action"
    raise ValueError(f"Cannot assign runtime weight to a profiling stage: {name}")


def _stage_weight_bytes(frontend, torch, model=None):
    weights = {name: [] for name in ("vis", "vlm", "action")}
    for name, value in frontend._ckpt_bf16.items():
        weights[_weight_stage(name)].append(value)
    if model is not None:
        for name, value in model._ckpt_bf16.items():
            weights[_weight_stage(name)].append(value)
    # Quantization dictionaries contain pointers; resolve them to owned tensors.
    for kind in ("fp8", "int8"):
        tensors = {t.data_ptr(): t for t in getattr(frontend, f"_{kind}_store")}
        for name, pointers in getattr(frontend, f"_{kind}_weights").items():
            weights[_weight_stage(name)].extend(tensors[pointer] for pointer in pointers)
    return {name: _storage_bytes(values, torch) for name, values in weights.items()}


def _leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _storage_bytes(value, torch):
    storages = {}
    for item in _leaves(value):
        if isinstance(item, torch.Tensor) and item.is_cuda:
            storage = item.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def _buffer_bytes(value, buffer_type):
    return sum({item.ptr.value: item.nbytes for item in _leaves(value)
                if isinstance(item, buffer_type)}.values())


def _pi05_stage_flops(config):
    """Dense GEMM + full attention estimate for the vendored Pi0.5 dimensions."""
    views, patches = config.num_views, 256
    vision_tokens = views * patches
    seq = vision_tokens + config.prompt_len
    chunk = config.chunk_size
    # SigLIP: patch projection, QKV/output, two FFN projections, per-view attention.
    vision = 2 * vision_tokens * 588 * 1152
    vision += 27 * (2 * vision_tokens * (4 * 1152**2 + 2 * 1152 * 4304)
                    + 4 * views * patches**2 * 1152)
    # Gemma: eight query heads and one KV head, each with head dimension 256.
    encoder = 2 * vision_tokens * 1152 * 2048
    encoder += 18 * (2 * seq * (2048 * (2048 + 2 * 256) + 2048**2 + 3 * 2048 * 16384)
                     + 4 * seq**2 * 2048)
    decoder = 2 * chunk * (32 * 1024 + 1024 * 32)
    decoder += 18 * (2 * chunk * (1024 * (2048 + 2 * 256) + 2048 * 1024 + 3 * 1024 * 4096)
                     + 4 * chunk * (seq + chunk) * 2048)
    return {"vis": vision, "vlm": encoder, "action": config.num_steps * decoder}


class PolicyManager:
    def __init__(self):
        self.backends = {"openpi": OpenPIBackend(), "flash_rt": FlashRTBackend()}

    def prepare_policies(self, policy_config: PolicyConfig,
                         stream=None) -> tuple[PolicyTargets, dict[str, StageProfile]]:
        try:
            backend = self.backends[policy_config.backend]
        except KeyError:
            raise ValueError(f"Unknown profiling backend: {policy_config.backend!r}") from None
        return backend.prepare_policies(policy_config, stream=stream)

    def release_policies(self):
        with ExitStack() as cleanup:
            for backend in self.backends.values():
                release = getattr(backend, "release_policies", None)
                if release is not None:
                    cleanup.callback(release)

    def release_execution(self):
        """Release per-run resources, keeping backend model caches resident."""
        with ExitStack() as cleanup:
            for backend in self.backends.values():
                release = getattr(backend, "release_execution", None)
                if release is not None:
                    cleanup.callback(release)
