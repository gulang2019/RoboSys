"""Prepared, synchronous profiling targets; optional runtimes load on demand."""

from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from math import prod
from pathlib import Path

from .schemas import PolicyConfig, PolicyProfile

PolicyTargets = dict[str, Callable[[], None]]


class OpenPIBackend:
    def prepare_policies(self, policy_config: PolicyConfig):
        raise NotImplementedError("OpenPI profiling is not implemented")


class FlashRTBackend:
    """Pi0.5 RTX/Torch component profiling with real checkpoint weights.

    Supports batch size 1, 224px images and FP16 or FP8 (BF16 residuals).
    Inputs are synthetic, including exactly ``prompt_len`` language embeddings.
    Calibration on synthetic images is for performance, not accuracy evaluation.
    Targets run eagerly on a private stream and wait for completion. VLM includes
    restoring language inputs; action includes restoring fixed diffusion noise
    and all denoising steps. Preprocessing/model loading are outside the targets.
    The targets share buffers and must not be invoked concurrently.

    Metadata: num_params counts checkpoint tensor elements; FLOPs estimates
    dense GEMMs and attention (multiply-add = 2), excluding elementwise ops and
    setup-time style computation. Memory uses decimal GB: retained CUDA tensor
    storage for weights and explicit pipeline buffers for activations/workspace;
    it excludes opaque library workspaces, allocator reserves and graph pools.
    """

    def __init__(self):
        self._last_policy_config = None
        self._cache_key = None
        self._prepared = None

    def prepare_policies(self, policy_config: PolicyConfig) -> tuple[PolicyTargets, PolicyProfile]:
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
        if capability not in ((8, 9), (12, 0)):
            raise NotImplementedError("FlashRT profiling currently supports RTX SM89/SM120")
        key = (config, str(checkpoint), device)
        if key != self._cache_key:
            prepared = self._prepare(config, checkpoint, torch, capability)
            # Publish only after successful initialization.
            self._prepared = prepared
            self._cache_key = key
            self._last_policy_config = replace(config)
        targets, profile = self._prepared
        # Results are mutable; each experiment gets fresh result containers.
        return dict(targets), replace(profile, stages={})

    @staticmethod
    def _validate(config):
        if config.backend != "flash_rt":
            raise ValueError("FlashRTBackend requires backend='flash_rt'")
        if config.model_name not in ("pi05", "pi05_libero"):
            raise NotImplementedError("FlashRT profiling currently supports pi05/pi05_libero")
        if config.implementation != "torch":
            raise NotImplementedError("FlashRT profiling currently supports implementation='torch'")
        if config.precision not in ("fp16", "fp8"):
            raise NotImplementedError("FlashRT profiling supports fp16 and fp8 (BF16 residuals)")
        if config.batch_size != 1 or str(config.image_resolution) != "224":
            raise NotImplementedError("FlashRT profiling requires batch_size=1 and image_resolution=224")
        for name in ("num_views", "prompt_len", "num_steps", "chunk_size"):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _prepare(config, checkpoint, torch, capability):
        import numpy as np

        fp16 = config.precision == "fp16"
        suffix = "_fp16" if fp16 else ""
        module = import_module(f"flash_rt.frontends.torch.pi05_rtx{suffix}")
        frontend_type = getattr(module, "Pi05TorchFrontendRtxFP16" if fp16 else "Pi05TorchFrontendRtx")
        with torch.inference_mode():
            frontend = frontend_type(
                checkpoint, num_views=config.num_views, max_prompt_len=config.prompt_len,
                chunk_size=config.chunk_size, num_steps=config.num_steps,
                cache_frames=1, use_fp8=not fp16,
                hardware=f"rtx_sm{capability[0]}{capability[1]}",
            )
            precision = frontend._pipeline_precision_kwargs()
            if (bool(precision["use_fp8"]) != (not fp16)
                    or bool(precision["use_fp8_decoder"]) != (not fp16)
                    or any(value for name, value in precision.items() if name.startswith("use_int8"))):
                raise ValueError("FlashRT environment overrides conflict with requested precision")
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
            stream = torch.cuda.Stream()
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

            def synchronized(fn):
                def run():
                    with torch.cuda.device(stream.device), torch.inference_mode(), torch.cuda.stream(stream):
                        fn()
                    stream.synchronize()
                return run

            targets = {name: synchronized(fn) for name, fn in
                       (("vis", vision), ("vlm", vlm), ("action", action))}
            # Establish valid upstream outputs for independently repeated targets.
            for target in targets.values():
                target()
            from safetensors import safe_open
            with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as weights:
                num_params = sum(prod(weights.get_slice(name).get_shape()) for name in weights.keys())
            from flash_rt.core.cuda_buffer import CudaBuffer
            weight_bytes = _storage_bytes(vars(frontend), torch)
            buffer_bytes = _buffer_bytes(vars(pipeline), CudaBuffer)
            profile = PolicyProfile(num_params, _pi05_flops(config), weight_bytes / 1e9,
                                    buffer_bytes / 1e9, {})
        return targets, profile


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


def _pi05_flops(config):
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
    return vision + encoder + config.num_steps * decoder


class PolicyManager:
    def __init__(self):
        self.backends = {"openpi": OpenPIBackend(), "flash_rt": FlashRTBackend()}

    def prepare_policies(self, policy_config: PolicyConfig) -> tuple[PolicyTargets, PolicyProfile]:
        try:
            backend = self.backends[policy_config.backend]
        except KeyError:
            raise ValueError(f"Unknown profiling backend: {policy_config.backend!r}") from None
        return backend.prepare_policies(policy_config)
