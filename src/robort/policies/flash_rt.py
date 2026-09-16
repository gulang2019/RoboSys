"""Minimal eager, synchronous FlashRT Pi0.5 LIBERO inference."""

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .config import PolicyConfig
from .vla_base import VLABasePolicy
from ..schemas import InferenceResponse


def flashrt_hardware(capability):
    """Return FlashRT's hardware family for a CUDA compute capability."""
    if capability == (11, 0):
        return "thor"
    if capability in ((8, 9), (12, 0)):
        return f"rtx_sm{capability[0]}{capability[1]}"
    raise NotImplementedError(
        f"FlashRT does not support CUDA capability SM{capability[0]}{capability[1]}"
    )


class FlashRTPolicy(VLABasePolicy):
    """Single-request FP16 inference on RTX SM89/SM120 or Thor SM110.

    Stages share FlashRT's resident buffers. Call preprocess, embed, encode,
    decode, postprocess in order; do not interleave requests or threads.
    Embedding runs SigLIP; the vision projection is part of encode in FlashRT.
    CUDA graphs and batching are intentionally outside this minimal backend.
    """

    def __init__(self, policy_config: PolicyConfig, device: str | None = None):
        super().__init__(policy_config, device)
        if policy_config.model_name not in ("pi05", "pi05_libero"):
            raise NotImplementedError("FlashRTPolicy supports Pi0.5 LIBERO checkpoints")
        if policy_config.precision != "fp16":
            raise NotImplementedError("FlashRTPolicy supports fp16 only")
        if (policy_config.batch_sizes or [1]) != [1] or policy_config.num_views != 2 or str(policy_config.image_resolution) != "224":
            raise NotImplementedError("FlashRTPolicy requires batch_size=1, num_views=2 and image_resolution=224")
        if policy_config.num_steps < 1 or policy_config.chunk_size < 1:
            raise ValueError("num_steps and chunk_size must be positive")
        self.device = torch.device(self.device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FlashRTPolicy requires CUDA")
        checkpoint = Path(policy_config.model_dir or f"checkpoints/{policy_config.model_name}_pytorch")
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint / "model.safetensors")
        with torch.cuda.device(self.device), torch.inference_mode():
            capability = torch.cuda.get_device_capability()
            self.hardware = flashrt_hardware(capability)
            if self.hardware == "thor":
                from flash_rt.frontends.torch import pi05_thor as frontend_module
                self._frontend_module = frontend_module
                self.frontend = frontend_module.Pi05TorchFrontendThor(
                    checkpoint, num_views=2, use_cuda_graph=False,
                    autotune=0, use_fp8=False,
                )
                if (self.frontend.steps != policy_config.num_steps
                        or self.frontend.Sa != policy_config.chunk_size):
                    raise NotImplementedError(
                        "FlashRT Thor fixes num_steps/chunk_size to "
                        f"{self.frontend.steps}/{self.frontend.Sa}; got "
                        f"{policy_config.num_steps}/{policy_config.chunk_size}"
                    )
            else:
                from flash_rt.frontends.torch.pi05_rtx_fp16 import Pi05TorchFrontendRtxFP16
                self._frontend_module = None
                self.frontend = Pi05TorchFrontendRtxFP16(
                    checkpoint, num_views=2, chunk_size=policy_config.chunk_size,
                    num_steps=policy_config.num_steps, max_prompt_len=policy_config.prompt_len,
                    cache_frames=1, use_fp8=False, hardware=self.hardware,
                )
            torch.cuda.synchronize()
        self._active = None
        self._stage = None

    def _check(self, value, stage):
        if value is not self._active or self._stage != stage:
            raise ValueError(f"Expected the current {stage} result; stages must run in order")

    def preprocess(self, observations):
        if not observations:
            raise ValueError("preprocess requires a nonempty batch")
        if len(observations) != 1:
            raise ValueError("FlashRTPolicy supports batch_size=1 only")
        request = observations[0]
        if request.inference_type != "sync":
            raise ValueError("FlashRTPolicy only supports sync requests")
        obs = request.observation
        images = [np.asarray(obs[key]) for key in ("observation/image", "observation/wrist_image")]
        if any(im.shape != (224, 224, 3) or im.dtype != np.uint8 for im in images):
            raise ValueError("Expected two uint8 HWC images with shape (224, 224, 3)")
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite LIBERO state with shape (8,)")
        stats = self.frontend.norm_stats["state"]
        q01, q99 = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        state = (2 * (state - q01[:8]) / (q99[:8] - q01[:8] + 1e-6) - 1).astype(np.float32)
        # OpenPI normalizes the eight LIBERO coordinates, then pads to action_dim.
        state = np.pad(state, (0, 24))
        prompt = obs["prompt"]
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        result = SimpleNamespace(state=state[None], images=[im.copy() for im in images], prompt=prompt)
        self._active, self._stage = result, "preprocess"
        return result

    @torch.inference_mode()
    def embed(self, observation):
        self._check(observation, "preprocess")
        with torch.cuda.device(self.device):
            frontend = self.frontend
            frontend.set_prompt(observation.prompt, state=observation.state[0])
            if self.hardware == "thor":
                images = np.stack(observation.images)
                np.copyto(frontend._infer_images_u8_np, images)
                frontend._img_u8_buf.upload(frontend._infer_images_u8_np)
                stream = torch.cuda.current_stream()
                stream_int = stream.cuda_stream
                module = self._frontend_module
                frontend._patch_embed_ops(stream_int, uint8_input=True)
                module.siglip_forward(
                    frontend._gemm, module.fvk, frontend._sig_bufs,
                    frontend._sig_weights, frontend._sig_dims,
                    stream=stream_int, attn=frontend._attn,
                    use_fp8=frontend.use_fp8,
                )
                frontend._postln_project_ops(stream_int)
                stream.synchronize()
                result = {"state": observation.state, "frontend": frontend}
                self._active, self._stage = result, "embed"
                return result
            pipeline = frontend.pipeline
            stream = torch.cuda.current_stream()
            frontend._fill_img_buf({"images": observation.images})
            frontend._copy_tensor_to_pipeline_buf(frontend._img_buf, pipeline.input_images_buf)
            pipeline.vision_encoder(stream=stream.cuda_stream)
            stream.synchronize()
        result = {"state": observation.state, "pipeline": pipeline,
                  "vision_shape": (1, pipeline.vision_seq, 1152)}
        self._active, self._stage = result, "embed"
        return result

    @torch.inference_mode()
    def encode(self, embeddings):
        self._check(embeddings, "embed")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream()
            if self.hardware == "thor":
                frontend = embeddings["frontend"]
                module = self._frontend_module
                buffers, weights, dims = frontend._runtime_encoder_spec()
                frontend._Kc.zero_()
                frontend._Vc.zero_()
                module.encoder_forward(
                    frontend._gemm, module.fvk, buffers, weights, dims,
                    stream=stream.cuda_stream, attn=frontend._attn,
                    use_fp8=frontend.use_fp8,
                )
                stream.synchronize()
                result = {"state": embeddings["state"], "frontend": frontend}
                self._active, self._stage = result, "encode"
                return result
            pipeline = embeddings["pipeline"]
            # Encoder overwrites its input, so restore language embeds every time.
            pipeline._copy_lang_embeds_to_encoder_x(stream=stream.cuda_stream)
            pipeline.transformer_encoder(stream=stream.cuda_stream)
            stream.synchronize()
        result = {"state": embeddings["state"], "pipeline": pipeline}
        self._active, self._stage = result, "encode"
        return result

    @torch.inference_mode()
    def decode(self, context, noise=None):
        self._check(context, "encode")
        with torch.cuda.device(self.device):
            if self.hardware == "thor":
                frontend = context["frontend"]
                shape = (1, frontend.Sa, 32)
                if noise is None:
                    noise = torch.randn(frontend.Sa, 32, device=self.device, dtype=torch.float16)
                else:
                    noise = torch.as_tensor(noise, device=self.device, dtype=torch.float16)
                    if tuple(noise.shape) != shape:
                        raise ValueError(f"Expected noise shape {shape}")
                    noise = noise[0]
                frontend._g_noise.copy_(noise)
                buffers, weights, dims = frontend._runtime_decoder_spec()
                stream = torch.cuda.current_stream()
                self._frontend_module.decoder_forward(
                    frontend._ctx, self._frontend_module.fvk,
                    buffers, weights, dims, stream=stream.cuda_stream,
                    attn=frontend._attn, use_fp8=frontend.use_fp8,
                )
                stream.synchronize()
                actions = frontend._g_noise.float().cpu().numpy()[None]
                result = {"state": context["state"], "actions": actions}
                self._active, self._stage = result, "decode"
                return result
            frontend, pipeline = self.frontend, context["pipeline"]
            stream = torch.cuda.current_stream()
            shape = (1, self.config.chunk_size, 32)
            if noise is None:
                frontend._noise_buf.normal_()
            else:
                noise = torch.as_tensor(noise, device=self.device, dtype=torch.float16)
                if tuple(noise.shape) != shape:
                    raise ValueError(f"Expected noise shape {shape}")
                frontend._noise_buf.copy_(noise[0])
            frontend._copy_tensor_to_pipeline_buf(frontend._noise_buf, pipeline.input_noise_buf)
            pipeline.transformer_decoder(stream=stream.cuda_stream)
            frontend._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(frontend._noise_out.data_ptr()), pipeline.input_noise_buf.ptr,
                frontend._noise_out.numel() * 2, 3, stream.cuda_stream,
            )
            stream.synchronize()
            actions = frontend._noise_out.float().cpu().numpy()[None]
        result = {"state": context["state"], "actions": actions}
        self._active, self._stage = result, "decode"
        return result

    def postprocess(self, outputs):
        from flash_rt.core.utils.actions import unnormalize_actions, LIBERO_ACTION_DIM
        actions = unnormalize_actions(outputs["actions"], self.frontend.norm_stats)
        return [InferenceResponse(actions=action[:, :LIBERO_ACTION_DIM].copy()) for action in actions]

    def infer(self, observations):
        return super().infer(observations) if observations else []
