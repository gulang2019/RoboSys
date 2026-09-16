"""Per-stream CUDA graph execution for the decomposed FlashRT Thor policy."""

from contextlib import ExitStack

import numpy as np
import torch

from .vla_base import VLABasePolicy


class _CapturedStage:
    """Own one graph captured and replayed on an externally owned stream."""

    def __init__(self, body, stream):
        self.body = body
        self.stream = stream
        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.device(stream.device), torch.inference_mode(), torch.cuda.stream(stream):
                for _ in range(3):
                    body()
                stream.synchronize()
                with torch.cuda.graph(self.graph, stream=stream):
                    body()
            stream.synchronize()
        except BaseException:
            self.close()
            raise

    def __call__(self):
        if self.graph is None:
            raise RuntimeError("Captured FlashRT stage has been closed")
        with (torch.cuda.device(self.stream.device), torch.inference_mode(),
              torch.cuda.stream(self.stream)):
            self.graph.replay()
        self.stream.synchronize()

    def close(self):
        graph, self.graph = self.graph, None
        if graph is not None:
            try:
                if self.stream is not None:
                    self.stream.synchronize()
            finally:
                graph.reset()
        self.body = None
        self.stream = None


class CudaGraphFlashRTPolicy(VLABasePolicy):
    """Replay FlashRT Thor's three GPU stages as independent CUDA graphs.

    Host-side request validation, prompt/image/noise updates and action download
    still run for every call. Graphs own only the static-shape GPU bodies and are
    captured on the profiler's green stream. Close this wrapper before that
    stream and its green context are destroyed.
    """

    def __init__(self, policy, stream):
        super().__init__(policy.config, policy.device)
        if getattr(policy, "hardware", None) != "thor":
            raise NotImplementedError(
                "FlashRT CUDA graph profiling currently supports Thor SM110 only")
        device = torch.device(self.device)
        if device.type != "cuda" or (device.index is not None and device != stream.device):
            raise ValueError("Capture stream must belong to the policy CUDA device")
        self.policy = policy
        self.frontend = policy.frontend
        self.stream = stream
        self._graphs = {}
        self._active = self._stage = None
        self._shape = None
        try:
            self._prepare()
        except BaseException:
            self.close()
            raise

    def _prepare(self):
        frontend = self.frontend
        module = self.policy._frontend_module
        observation = self.policy.preprocess([self.policy.make_example_input()])
        with torch.cuda.device(self.stream.device), torch.inference_mode(), torch.cuda.stream(self.stream):
            frontend.set_prompt(observation.prompt, state=observation.state[0])
            self._shape = (frontend.Se, frontend.current_prompt_len)
            images = np.stack(observation.images)
            np.copyto(frontend._infer_images_u8_np, images)
            frontend._img_u8_buf.upload(frontend._infer_images_u8_np)
            enc_bufs, enc_weights, enc_dims = frontend._runtime_encoder_spec()
            dec_bufs, dec_weights, dec_dims = frontend._runtime_decoder_spec()

            def vision():
                stream_int = self.stream.cuda_stream
                frontend._patch_embed_ops(stream_int, uint8_input=True)
                module.siglip_forward(
                    frontend._gemm, module.fvk, frontend._sig_bufs,
                    frontend._sig_weights, frontend._sig_dims,
                    stream=stream_int, attn=frontend._attn,
                    use_fp8=frontend.use_fp8,
                )
                frontend._postln_project_ops(stream_int)

            def encode():
                frontend._Kc.zero_()
                frontend._Vc.zero_()
                module.encoder_forward(
                    frontend._gemm, module.fvk, enc_bufs, enc_weights,
                    enc_dims, stream=self.stream.cuda_stream,
                    attn=frontend._attn, use_fp8=frontend.use_fp8,
                )

            def decode():
                module.decoder_forward(
                    frontend._ctx, module.fvk, dec_bufs, dec_weights,
                    dec_dims, stream=self.stream.cuda_stream,
                    attn=frontend._attn, use_fp8=frontend.use_fp8,
                )

            self._graphs["embed"] = _CapturedStage(vision, self.stream)
            self._graphs["embed"]()
            self._graphs["encode"] = _CapturedStage(encode, self.stream)
            self._graphs["encode"]()
            self._graphs["decode"] = _CapturedStage(decode, self.stream)
        self.stream.synchronize()
        # set_prompt() builds FlashRT's fused inference graphs as a setup side
        # effect. This wrapper uses its own per-stage graphs, so release those
        # fused graphs while their capture context is still alive. Keeping them
        # in the cached frontend would outlive the profiler's green context.
        for name in ("_siglip_graph", "_siglip_u8_graph", "_enc_ae_graph"):
            graph = getattr(frontend, name, None)
            if graph is not None:
                graph.reset()
                setattr(frontend, name, None)
        self.policy._active = self.policy._stage = None

    def _check(self, value, stage):
        if self._graphs is None:
            raise RuntimeError("CUDA graph policy has been closed")
        if value is not self._active or self._stage != stage:
            raise ValueError(f"Expected the current {stage} result; stages must run in order")

    def preprocess(self, observations):
        result = self.policy.preprocess(observations)
        self._active, self._stage = result, "preprocess"
        return result

    def embed(self, observation):
        self._check(observation, "preprocess")
        frontend = self.frontend
        with torch.cuda.device(self.stream.device), torch.inference_mode(), torch.cuda.stream(self.stream):
            frontend.set_prompt(observation.prompt, state=observation.state[0])
            if (frontend.Se, frontend.current_prompt_len) != self._shape:
                raise ValueError(
                    "FlashRT prompt shape changed; create a new CUDA graph policy")
            images = np.stack(observation.images)
            np.copyto(frontend._infer_images_u8_np, images)
            frontend._img_u8_buf.upload(frontend._infer_images_u8_np)
        self._graphs["embed"]()
        result = {"state": observation.state, "frontend": frontend}
        self._active, self._stage = result, "embed"
        return result

    def encode(self, embedding):
        self._check(embedding, "embed")
        self._graphs["encode"]()
        result = {"state": embedding["state"], "frontend": self.frontend}
        self._active, self._stage = result, "encode"
        return result

    def decode(self, context, noise=None):
        self._check(context, "encode")
        frontend = self.frontend
        shape = (1, frontend.Sa, 32)
        with torch.cuda.device(self.stream.device), torch.inference_mode(), torch.cuda.stream(self.stream):
            if noise is None:
                noise = torch.randn(frontend.Sa, 32, device=self.device, dtype=torch.float16)
            else:
                noise = torch.as_tensor(noise, device=self.device, dtype=torch.float16)
                if tuple(noise.shape) != shape:
                    raise ValueError(f"Expected noise shape {shape}")
                noise = noise[0]
            frontend._g_noise.copy_(noise)
        self._graphs["decode"]()
        actions = frontend._g_noise.float().cpu().numpy()[None]
        result = {"state": context["state"], "actions": actions}
        self._active, self._stage = result, "decode"
        return result

    def postprocess(self, outputs):
        self._check(outputs, "decode")
        result = self.policy.postprocess(outputs)
        self._active, self._stage = result, "postprocess"
        return result

    def close(self):
        graphs, self._graphs = self._graphs, None
        if graphs is not None:
            with ExitStack() as cleanup:
                for graph in graphs.values():
                    cleanup.callback(graph.close)
        self.policy = self.frontend = self.stream = None
        self._active = self._stage = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
