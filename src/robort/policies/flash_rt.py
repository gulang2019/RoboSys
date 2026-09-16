"""Minimal eager, synchronous FlashRT Pi0.5 LIBERO inference."""

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .config import PolicyConfig
from .vla_base import VLABasePolicy
from ..schemas import InferenceResponse


class FlashRTPolicy(VLABasePolicy):
    """Single-request FP16 inference on RTX SM89/SM120.

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
            if capability not in ((8, 9), (12, 0)):
                raise NotImplementedError("FlashRTPolicy requires RTX SM89/SM120")
            from flash_rt.frontends.torch.pi05_rtx_fp16 import Pi05TorchFrontendRtxFP16
            self.frontend = Pi05TorchFrontendRtxFP16(
                checkpoint, num_views=2, chunk_size=policy_config.chunk_size,
                num_steps=policy_config.num_steps, max_prompt_len=policy_config.prompt_len,
                cache_frames=1, use_fp8=False,
                hardware=f"rtx_sm{capability[0]}{capability[1]}",
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
