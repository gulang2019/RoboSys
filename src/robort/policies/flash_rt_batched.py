"""BF16 RTX Pi0.5 stages with shared storage and one graph per batch size.

One caller per stage; different stages may run concurrently on separate streams.
Stage outputs own their storage. The controller owns dependency waits, input
lifetimes and serialization of each stage, including its input/output copies.
Compile/close require the controller to drain outstanding work first.
"""
import ctypes
from types import SimpleNamespace
import weakref

import numpy as np
import torch

from flash_rt import flash_rt_fa2, flash_rt_kernels
from flash_rt.core.cuda_buffer import _cudart
from flash_rt.core.cuda_graph import CUDAGraph, _check
from flash_rt.core.utils.actions import unnormalize_actions
from flash_rt.core.utils.pi05_prompt import format_pi05_prompt, PI05_STATE_PROMPT_MAX_LEN
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
from flash_rt.hardware.rtx.attn_backend_batched_pi05 import RtxFlashAttnBatchedBackendPi05
from flash_rt.models.pi05.pipeline_rtx import (
    VIS_D, VIS_H, ENC_D, ENC_H, ENC_L, DEC_D, DEC_H,
)
from flash_rt.models.pi05.pipeline_rtx_batched import Pi05BatchedPipeline
from flash_rt.utils.paligemma_tokenizer import load_paligemma_sentencepiece

from .config import PolicyConfig
from .vla_base import VLABasePolicy
from ..schemas import InferenceRequest, InferenceResponse


def _validate_cuda_runtime():
    # FlashRT's ctypes loader resolves libcudart.so independently of PyTorch.
    # An old runtime can segfault when ending capture on a CUDA green stream.
    version = ctypes.c_int()
    _check(_cudart.cudaRuntimeGetVersion(ctypes.byref(version)))
    major, minor = version.value // 1000, version.value % 1000 // 10
    torch_cuda = torch.version.cuda
    if version.value < 12060 or major != int(torch_cuda.split(".")[0]):
        raise RuntimeError(
            f"FlashRT loaded CUDA runtime {major}.{minor}; this backend requires CUDA 12.6+ "
            f"with the same major version as PyTorch (CUDA {torch_cuda}). "
            f"Set LD_LIBRARY_PATH=/usr/local/cuda-{torch_cuda}/lib64:$LD_LIBRARY_PATH "
            "before starting Python, or source scripts/profile/setup.sh.")


def _spec(num_views=2, max_prompt_len=PI05_STATE_PROMPT_MAX_LEN, chunk_size=10, num_steps=10):
    return SimpleNamespace(num_views=num_views, max_prompt_len=max_prompt_len,
                           chunk_size=chunk_size, num_steps=num_steps,
                           vision_seq=num_views * 256, encoder_seq_len=num_views * 256 + max_prompt_len)


def _storage(capacity, model_spec, device, shapes):
    if type(capacity) is not int or capacity < 1:
        raise ValueError("buffer capacity must be a positive integer")
    return SimpleNamespace(capacity=capacity, spec=model_spec, tensors={
        name: torch.zeros(shape, dtype=torch.bfloat16, device=device)
        for name, shape in shapes.items()
    })


def _ptr(tensor):
    return SimpleNamespace(ptr=ctypes.c_void_p(tensor.data_ptr()))


class _StageAttention(RtxFlashAttnBatchedBackendPi05):
    def __init__(self, batch_size, shared_buffers, *args, **kwargs):
        self.init_buffers(shared_buffers, batch_size)

    @classmethod
    def allocate_buffers(cls, bsz, model_spec=None, device="cuda"):
        s = model_spec or _spec()
        if cls.stage == "embed":
            n, q, heads, dim, kv = bsz * s.num_views, 256, 16, 72, 256
        else:
            n, q, heads, dim = bsz, (s.encoder_seq_len if cls.stage == "encode" else s.chunk_size), 8, 256
            kv = s.encoder_seq_len + (s.chunk_size if cls.stage == "decode" else 0)
        shapes = {"q": (n, q, heads, dim), "o": (n, q, heads, dim)}
        shape_kv = (n, kv, heads, dim) if cls.stage == "embed" else (ENC_L, n, kv, 1, dim)
        shapes.update(k=shape_kv, v=shape_kv)
        storage = _storage(bsz, s, device, shapes)
        storage.tensors["lse"] = torch.zeros(n, heads, ((q + 127) // 128) * 128, device=device)
        if cls.stage != "embed":
            storage.tensors["lengths"] = torch.full((bsz,), s.encoder_seq_len, dtype=torch.int32, device=device)
            if cls.stage == "decode":
                storage.tensors["positions"] = torch.full((bsz,), s.encoder_seq_len, dtype=torch.int32, device=device)
        return storage

    def init_buffers(self, shared_buffers, batch_size):
        if not 1 <= batch_size <= shared_buffers.capacity:
            raise ValueError("batch size exceeds attention capacity")
        self.storage, self.B = shared_buffers, batch_size
        n = batch_size * shared_buffers.spec.num_views if self.stage == "embed" else batch_size
        self.t = {k: (v[:, :n] if k in ("k", "v") and self.stage != "embed" else v[:n])
                  for k, v in shared_buffers.tensors.items()}
        self._fa2, self._fa2_fwd = flash_rt_fa2, flash_rt_fa2.fwd_bf16
        self._num_sms = torch.cuda.get_device_properties(self.t["q"].device).multi_processor_count

    @property
    def batch_size(self):
        return self.B

    def get_ptrs_b2(self):
        if self.stage == "embed":
            return {"vis_" + k.upper(): self.t[k].data_ptr() for k in ("q", "k", "v")}
        k, v, q = self.t["k"], self.t["v"], self.t["q"]
        return {"enc_K": k.data_ptr(), "enc_V": v.data_ptr(),
                "enc_k_layer_stride_bytes": k.stride(0) * 2,
                "enc_v_layer_stride_bytes": v.stride(0) * 2,
                "enc_k_sample_stride_bytes": k.stride(1) * 2,
                "enc_v_sample_stride_bytes": v.stride(1) * 2,
                "enc_q_sample_stride_bytes": q.stride(0) * 2,
                ("enc_Q" if self.stage == "encode" else "dec_Q"): q.data_ptr()}

    def run(self, stream: int, layer_idx=0):
        t = self.t
        if self.stage == "embed":
            self._call_fvk_fa2(t["q"], t["k"], t["v"], t["o"], t["lse"], stream=stream)
        else:
            self._call_fvk_fa2_seqused(t["q"], t["k"][layer_idx], t["v"][layer_idx],
                                      t["o"], t["lse"], t["lengths"], stream=stream)
        return t["o"].data_ptr()

    def run_batched(self, site, layer_idx, q_seq, *, kv_seq=None, stream=0):
        return self.run(stream, layer_idx)


class EmbedAttnBackend(_StageAttention):
    stage = "embed"
    # Keep the original skeleton's singular spelling as an alias.
    @classmethod
    def allocate_buffer(cls, bsz, model_spec=None, device="cuda"):
        return cls.allocate_buffers(bsz, model_spec, device)


class EncodeAttnBackend(_StageAttention):
    stage = "encode"


class DecodeAttnBackend(_StageAttention):
    stage = "decode"


class _StagePipeline(Pi05BatchedPipeline):
    def __init__(self, batch_size, shared_buffers, *, attn_backend, gemm, fvk, weights, model_spec):
        self.attn, self.gemm, self.fvk, self.weights = attn_backend, gemm, fvk, weights
        self.__dict__.update(vars(model_spec))
        self.use_fp8 = self.use_fp8_decoder = self.fp8_calibrated = False
        self._cudart, self._graphs = _cudart, None
        self._attn_ptrs_b2 = attn_backend.get_ptrs_b2()
        self.init_buffers(shared_buffers, batch_size)

    def init_buffers(self, shared_buffers, batch_size):
        if not 1 <= batch_size <= shared_buffers.capacity:
            raise ValueError("batch size exceeds pipeline capacity")
        self.storage, self.B = shared_buffers, batch_size
        self.t = shared_buffers.tensors
        self.bufs = {k: _ptr(v) for k, v in self.t.items()}
        self._bias_zero_buf = self.bufs["bias_zero"]
        if "rms" in self.t:
            self._rms_ones_enc = self._rms_ones_dec = self.bufs["rms"]

    def _inputs(self, items):
        if len(items) != self.B:
            raise ValueError(f"expected {self.B} inputs, got {len(items)}")
        # Only retain metadata needed by get_output, not copied images/KV.
        self.items = [{"language": item.get("language"), "noise": item.get("noise")} for item in items]

    def _copy(self, dst, src):
        src = torch.as_tensor(src, device=dst.device)
        if src.shape != dst.shape or not src.is_floating_point():
            raise ValueError(f"expected floating tensor with shape {tuple(dst.shape)}")
        dst.copy_(src)

    def autotune_gemms(self):
        """Tune the exact BF16 GEMMs used by this stage, once per runner/shape."""
        runner = self.gemm
        seen = self.storage.tuned_shapes
        class Tuner:
            def bf16_nn(_, a, w, out, m, n, k, stream=0):
                if (m, n, k) not in seen:
                    _check(_cudart.cudaStreamSynchronize(ctypes.c_void_p(stream)))
                    runner.autotune_bf16_nn(a, w, out, m, n, k)
                    seen.add((m, n, k))
                runner.bf16_nn(a, w, out, m, n, k, stream=stream)
        try:
            self.gemm = Tuner()
            self.run(torch.cuda.current_stream().cuda_stream)
        finally:
            self.gemm = runner


def _pipeline_storage(stage, bsz, model_spec, device):
    s = model_spec or _spec()
    v, e, d = s.vision_seq, s.encoder_seq_len, s.chunk_size
    if stage == "embed":
        shapes = {"observation_images_normalized_b2": (bsz, s.num_views, 224, 224, 3),
                  "vision_x_b2": (bsz, v, VIS_D), "vision_x_norm_b2": (bsz, v, VIS_D),
                  "vision_hidden_b2": (bsz, v, VIS_H),
                  "vision_pos_embed_expanded_b2": (bsz, v, VIS_D), "bias_zero": (bsz * v * VIS_H,)}
    elif stage == "encode":
        shapes = {"vision_x_b2": (bsz, v, VIS_D), "vision_x_norm_b2": (bsz, v, VIS_D),
                  "encoder_x_b2": (bsz, e, ENC_D), "encoder_x_norm_b2": (bsz, e, ENC_D),
                  "encoder_hidden_b2": (bsz, e, ENC_H), "encoder_gate_merged_b2": (bsz, e, ENC_H),
                  "bias_zero": (bsz * v * ENC_D,), "rms": (ENC_D,)}
    else:
        shapes = {"diffusion_noise_b2": (bsz, d, 32), "decoder_x_b2": (bsz, d, DEC_D),
                  "decoder_action_buf_b2": (bsz, d, 32), "x_normed_buf_b2": (bsz, d, DEC_D),
                  "decoder_rope_weights": (bsz, d, 256),
                  "gate_buf_b2": (bsz, d, DEC_D), "decoder_hidden_b2": (bsz, d, DEC_H),
                  "decoder_gate_merged_b2": (bsz, d, DEC_H),
                  "bias_zero": (bsz * d * DEC_D,), "rms": (DEC_D,)}
    storage = _storage(bsz, s, device, shapes)
    storage.tuned_shapes = set()
    t = storage.tensors
    # These temporaries have disjoint lifetimes; alias flat contiguous scratch.
    if stage == "embed":
        for key in ("vision_QKV_b2", "vision_patches_b2"):
            t[key] = t["vision_hidden_b2"]
    else:
        t[f'{"encoder" if stage == "encode" else "decoder"}_QKV_b2'] = t[f'{"encoder" if stage == "encode" else "decoder"}_gate_merged_b2']
        t["rms"].fill_(1)
    return storage


class EmbedPipeline(_StagePipeline):
    @staticmethod
    def allocate_buffers(bsz, model_spec=None, device="cuda"):
        return _pipeline_storage("embed", bsz, model_spec, device)

    def set_input(self, _input: list):
        self._inputs(_input)
        for b, item in enumerate(_input):
            self._copy(self.t["observation_images_normalized_b2"][b], item["images"])

    def get_output(self) -> list:
        return [{"vision": self.t["vision_x_b2"][b].clone(),
                          "language": item["language"], "noise": item.get("noise")}
                         for b, item in enumerate(self.items)]

    def run(self, stream: int):
        self.vision_encoder_batched(stream)


class EncodePipeline(_StagePipeline):
    @staticmethod
    def allocate_buffers(bsz, model_spec=None, device="cuda"):
        return _pipeline_storage("encode", bsz, model_spec, device)

    def set_input(self, _input: list):
        self._inputs(_input)
        self.t["encoder_x_b2"][:self.B].zero_()
        for b, item in enumerate(_input):
            self._copy(self.t["vision_x_b2"][b], item["vision"])
            language = item["language"]
            n = len(language)
            if not 0 < n <= self.max_prompt_len:
                raise ValueError("language length exceeds prompt capacity or is empty")
            self._copy(self.t["encoder_x_b2"][b, self.vision_seq:self.vision_seq+n], language)
            self.attn.t["lengths"][b].fill_(self.vision_seq + n)

    def get_output(self) -> list:
        return [{"k": self.attn.t["k"][:, b, :self.vision_seq + len(item["language"])].clone(),
                          "v": self.attn.t["v"][:, b, :self.vision_seq + len(item["language"])].clone(),
                          "prefix_len": self.vision_seq + len(item["language"]), "noise": item.get("noise")}
                         for b, item in enumerate(self.items)]

    def run(self, stream: int):
        self.transformer_encoder_batched(stream)


class DecodePipeline(_StagePipeline):
    @staticmethod
    def allocate_buffers(bsz, model_spec=None, device="cuda"):
        return _pipeline_storage("decode", bsz, model_spec, device)

    def set_input(self, _input: list):
        self._inputs(_input)
        for b, item in enumerate(_input):
            n = item["prefix_len"]
            if not self.vision_seq < n <= self.encoder_seq_len:
                raise ValueError("invalid prefix length")
            for key in ("k", "v"):
                self._copy(self.attn.t[key][:, b, :n], item[key])
            self.t["decoder_rope_weights"][b].copy_(self.t["rope_table"][n:n+self.chunk_size])
            self.attn.t["positions"][b].fill_(n)
            self.attn.t["lengths"][b].fill_(n + self.chunk_size)
            noise = item.get("noise")
            if noise is None:
                self.t["diffusion_noise_b2"][b].normal_()
            else:
                self._copy(self.t["diffusion_noise_b2"][b], noise)

    def get_output(self) -> list:
        return [{"actions": self.t["diffusion_noise_b2"][b].clone()} for b in range(self.B)]

    def run(self, stream: int):
        self.transformer_decoder_batched(stream)

    def _style_slice_ptr(self, buf_name, step, layer=None):
        # Physical step/layer strides use capacity, never the active batch size.
        t = self.t[buf_name]
        return t[step].data_ptr() if layer is None else t[step, layer].data_ptr()

    def _decoder_qkv_rope_batched(self, i, enc_seq, ds, stream):
        for b in range(self.B):
            self.fvk.qkv_split_rope_devpos(
                self.bufs["decoder_QKV_b2"].ptr.value + b * ds * 2560 * 2,
                self.t["decoder_rope_weights"][b].data_ptr(),
                self.attn.t["q"][b].data_ptr(), self.attn.t["k"][i, b].data_ptr(),
                self.attn.t["v"][i, b].data_ptr(), self.attn.t["positions"][b].data_ptr(),
                ds, 2048, 256, 256, 256, stream=stream)


def _destroy_graph(exec_handle, graph_handle, device):
    with torch.cuda.device(device):
        _cudart.cudaGraphExecDestroy(exec_handle)
        _cudart.cudaGraphDestroy(graph_handle)


class FlashRTFrontEnd(Pi05TorchFrontendRtx):
    def __init__(self, batch_sizes: dict[str, list[int]], *args, **kwargs):
        classes = {"embed": (EmbedPipeline, EmbedAttnBackend), "encode": (EncodePipeline, EncodeAttnBackend),
                   "decode": (DecodePipeline, DecodeAttnBackend)}
        if set(batch_sizes) != set(classes) or any(
            not sizes or any(type(b) is not int or b < 1 for b in sizes) for sizes in batch_sizes.values()
        ):
            raise ValueError("batch_sizes must specify positive sizes for embed, encode and decode")
        kwargs.update(use_fp8=False, use_cuda_graph=False)
        super().__init__(*args, **kwargs)
        if self._vision_pool_factor != 1 or self._vision_num_layers != 27:
            raise ValueError("batched stages require full SigLIP without pooling")
        self.model_spec = _spec(self.num_views, self.max_prompt_len, self.chunk_size, self._num_steps)
        self.tokenizer = load_paligemma_sentencepiece()
        self.pipelines, self.stage_storage = {}, {}
        weights, device = self._build_pipeline_weights(), self.embedding_weight.device
        # Drop unused monolithic frontend attention/input allocations. Reuse its
        # first GemmRunner for embed; encode/decode have independent workspaces.
        self.attn_backend = self._img_buf = self._noise_buf = self._noise_out = None
        s = self.model_spec
        phase = np.arange(s.encoder_seq_len+s.chunk_size, dtype=np.float64)[:, None] / (
            10000 ** (np.arange(0, 256, 2, dtype=np.float64) / 256))
        rope = torch.as_tensor(np.stack([np.cos(phase), np.sin(phase)], axis=-1).reshape(-1, 256),
                               dtype=torch.bfloat16, device=device)
        for stage, (pipeline_cls, attn_cls) in classes.items():
            sizes = sorted(set(batch_sizes[stage]))
            capacity = max(sizes)
            gemm = self.gemm if stage == "embed" else self.fvk.GemmRunner()
            attn_buffers = attn_cls.allocate_buffers(capacity, s, device)
            buffers = pipeline_cls.allocate_buffers(capacity, s, device)
            if stage == "embed":
                buffers.tensors["vision_pos_embed_expanded_b2"].copy_(
                    self._ckpt_bf16["vision_position_embedding"].reshape(1, 256, VIS_D).repeat(capacity, s.num_views, 1))
            else:
                buffers.tensors["encoder_rope_weights" if stage == "encode" else "rope_table"] = rope
            if stage == "decode":
                buffers.tensors["decoder_rope_weights"].copy_(rope[s.encoder_seq_len:s.encoder_seq_len+s.chunk_size])
                for name in ("style_attn", "style_ffn", "style_final"):
                    pre = torch.from_numpy(self._precomputed_styles[name]).view(torch.bfloat16).to(device)
                    buffers.tensors["decoder_"+name] = pre.unsqueeze(-3).expand(
                        *pre.shape[:-2], capacity, *pre.shape[-2:]).contiguous()
            self.stage_storage[stage] = (buffers, attn_buffers)
            self.pipelines[stage] = {
                b: pipeline_cls(b, buffers, attn_backend=attn_cls(b, attn_buffers),
                                gemm=gemm, fvk=self.fvk, weights=weights, model_spec=s)
                for b in sizes
            }
        torch.cuda.current_stream().synchronize()

    def get_pipeline(self, stage, bsz):
        try:
            return self.pipelines[stage][bsz]
        except KeyError:
            raise ValueError(f"unsupported stage/batch size: {stage}/{bsz}") from None

    def validate_streams(self, streams):
        if set(streams) != set(self.pipelines):
            raise ValueError("streams must contain embed, encode and decode")
        for stage_streams in streams.values():
            if not isinstance(stage_streams, (list, tuple)) or not stage_streams:
                raise ValueError("each stage requires a nonempty list of streams")
            if any(not isinstance(stream, torch.cuda.Stream) or
                   stream.device != self.embedding_weight.device for stream in stage_streams):
                raise ValueError("stream must belong to the model device")

    def compile(self, streams: dict[str, list]):
        """Capture missing stream variants; caller must drain all stages first."""
        self.validate_streams(streams)
        if any(stream.cuda_stream == 0 for group in streams.values() for stream in group):
            raise ValueError("graph capture requires non-default streams")
        for stage, variants in self.pipelines.items():
            for pipeline in variants.values():
                if pipeline._graphs is None:
                    pipeline._graphs = {}
                for stream in streams[stage]:
                    if stream.cuda_stream in pipeline._graphs:
                        continue
                    handle = ctypes.c_void_p(stream.cuda_stream)
                    with torch.cuda.device(stream.device), torch.inference_mode(), torch.cuda.stream(stream):
                        pipeline.autotune_gemms()
                        for _ in range(3):
                            pipeline.run(stream.cuda_stream)
                        stream.synchronize()
                        graph = CUDAGraph()
                        graph.begin_capture(handle)
                        pipeline.run(stream.cuda_stream)
                        graph.end_capture(handle)
                        graph._release = weakref.finalize(
                            graph, _destroy_graph, graph._graph_exec, graph._graph, stream.device)
                        stream.synchronize()
                        pipeline._graphs[stream.cuda_stream] = graph

    def close(self):
        """Release graph handles after the controller has drained all stages."""
        for variants in self.pipelines.values():
            for pipeline in variants.values():
                for graph in (pipeline._graphs or {}).values():
                    graph._release()
                pipeline._graphs = None


class FlashRTPolicy(VLABasePolicy):
    """List-based LIBERO policy. Serialize calls within each stage.

    Optional initial noise is observation["noise"] with shape (chunk_size, 32),
    or context["noise"] before decode. The controller selects streams, orders
    dependencies and retains inputs until consumption completes. GPU stage
    calls do not wait or generate events. Postprocess returns CPU results.
    """
    def __init__(self, policy_config: PolicyConfig, device=None, streams=None):
        super().__init__(policy_config, device, streams)
        self.device = torch.device(self.device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("FlashRT batched policy requires CUDA")
        if policy_config.precision != "bf16" or policy_config.model_name not in ("pi05", "pi05_libero"):
            raise ValueError("FlashRT batched policy requires Pi0.5 BF16")
        if policy_config.num_views != 2 or str(policy_config.image_resolution) != "224":
            raise ValueError("LIBERO preprocessing requires two 224x224 views")
        if policy_config.chunk_size < 1 or policy_config.num_steps < 1:
            raise ValueError("chunk_size and num_steps must be positive")
        if not hasattr(Pi05BatchedPipeline, "_decoder_qkv_rope_batched"):
            raise RuntimeError("FlashRT needs the RoboSys decoder hook. Run "
                               "bash scripts/profile/apply-flashrt-patch.sh before loading the policy.")
        _validate_cuda_runtime()
        with torch.cuda.device(self.device), torch.inference_mode():
            capability = torch.cuda.get_device_capability()
            if capability not in ((8, 9), (12, 0)):
                raise ValueError("requires RTX SM89 or SM120")
            self.policy = FlashRTFrontEnd(
                policy_config.batch_sizes or {s: [1] for s in ("embed", "encode", "decode")},
                policy_config.model_dir or "checkpoints/pi05_libero_pytorch",
                num_views=policy_config.num_views, chunk_size=policy_config.chunk_size,
                num_steps=policy_config.num_steps,
                max_prompt_len=max(policy_config.prompt_len, PI05_STATE_PROMPT_MAX_LEN),
                hardware=f"rtx_sm{capability[0]}{capability[1]}",
            )
            if streams is None:
                common = torch.cuda.Stream()
                streams = {stage: [common] for stage in self.policy.pipelines}
            self.streams = streams
            self.policy.validate_streams(self.streams)
            if policy_config.use_cuda_graph:
                self.compile(self.streams)

    def compile(self, streams):
        self.policy.compile(streams)
        self.streams = streams

    def preprocess(self, observations, stream=None):
        if not observations:
            raise ValueError("preprocess requires a nonempty batch")
        result = []
        stats = self.policy.norm_stats["state"]
        lo, hi = np.asarray(stats["q01"])[:8], np.asarray(stats["q99"])[:8]
        for request in observations:
            if request.inference_type != "sync":
                raise ValueError("only sync inference is supported")
            obs = request.observation
            images = [np.asarray(obs[k]) for k in ("observation/image", "observation/wrist_image")]
            if any(im.shape != (224, 224, 3) or im.dtype != np.uint8 for im in images):
                raise ValueError("expected uint8 HWC images of shape (224,224,3)")
            state = np.asarray(obs["observation/state"], dtype=np.float32)
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError("expected finite state of shape (8,)")
            if not isinstance(obs["prompt"], str):
                raise ValueError("prompt must be a string")
            state = np.pad((2 * (state-lo) / (hi-lo+1e-6) - 1).astype(np.float32), (0, 24))
            tokens = self.policy.tokenizer.Encode(format_pi05_prompt(obs["prompt"], state), add_bos=True)
            if len(tokens) > self.policy.max_prompt_len:
                raise ValueError("state-conditioned prompt exceeds token capacity")
            result.append({"images": np.stack(images).astype(np.float32) / 127.5 - 1,
                           "tokens": tokens, "noise": obs.get("noise")})
        return result

    @torch.inference_mode()
    def _execute(self, stage, items, stream):
        pipeline = self.policy.get_pipeline(stage, len(items))
        if stream not in self.streams[stage]:
            raise ValueError(f"unregistered stream for {stage}")
        with torch.cuda.device(stream.device), torch.cuda.stream(stream):
            if stage == "embed":
                items = [dict(item, language=torch.nn.functional.embedding(
                    torch.as_tensor(item["tokens"], device=stream.device), self.policy.embedding_weight) * ENC_D**0.5)
                         for item in items]
            pipeline.set_input(items)
            if pipeline._graphs is None:
                pipeline.run(stream.cuda_stream)
            else:
                pipeline._graphs[stream.cuda_stream].replay(ctypes.c_void_p(stream.cuda_stream))
            return pipeline.get_output()

    def embed(self, observations, stream):
        return self._execute("embed", observations, stream)

    def encode(self, embeddings, stream):
        return self._execute("encode", embeddings, stream)

    def decode(self, context, stream):
        return self._execute("decode", context, stream)

    def postprocess(self, actions, stream=None):
        """Blocking CPU conversion; caller must order GPU inputs onto stream."""
        stream = stream if stream is not None else torch.cuda.current_stream(self.device)
        result = []
        for item in actions:
            with torch.cuda.device(stream.device), torch.cuda.stream(stream):
                raw = item["actions"].cpu().float().numpy()
            result.append(InferenceResponse(unnormalize_actions(raw, self.policy.norm_stats)[:, :7],
                                            rtc_prev_actions=raw))
        return result

    def infer(self, observations: list[InferenceRequest], stream=None) -> list[InferenceResponse]:
        if not observations:
            return []
        if stream is None:
            raise ValueError("infer requires an explicit stream registered for every GPU stage")
        for stage in self.policy.pipelines:
            if stream not in self.streams[stage]:
                raise ValueError(f"unregistered stream for {stage}")
        values = observations
        for stage in ("preprocess", "embed", "encode", "decode", "postprocess"):
            values = getattr(self, stage)(values, stream=stream)
        return values

    def make_example_input(self, bsz: int, stage="all", stream=None):
        if stage not in ("all", "preprocess", "embed", "encode", "decode", "postprocess"):
            raise ValueError(f"unknown stage: {stage}")
        requests = [InferenceRequest({"observation/image": np.zeros((224, 224, 3), np.uint8),
                    "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
                    "observation/state": np.zeros(8, np.float32), "prompt": "do something useful"}) for _ in range(bsz)]
        values = [requests]
        for name in ("preprocess", "embed", "encode", "decode", "postprocess"):
            if name == stage:
                return values[-1]
            if name in self.policy.pipelines and stream is None:
                raise ValueError("make_example_input requires an explicit stream for GPU stages")
            values.append(getattr(self, name)(values[-1], stream=stream))
        return tuple(values)
