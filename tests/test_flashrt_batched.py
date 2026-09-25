"""Real-GPU coverage for the decoupled FlashRT backend.

Run: LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 .venv-profile/bin/python -m pytest tests/test_flashrt_batched.py -v
Override ROBORT_FLASHRT_CHECKPOINT to select a converted Pi0.5 LIBERO checkpoint.
"""
import ctypes
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)
backend = pytest.importorskip("robort.policies.flash_rt_batched")
from robort.policies.config import PolicyConfig
from robort.schemas import InferenceRequest

CKPT = Path(os.environ.get("ROBORT_FLASHRT_CHECKPOINT", "checkpoints/pi05_libero_pytorch"))


@pytest.fixture(scope="module")
def policy():
    if not (CKPT / "model.safetensors").exists():
        pytest.skip("Pi0.5 checkpoint missing")
    if torch.cuda.get_device_capability() not in ((8, 9), (12, 0)):
        pytest.skip("RTX SM89/SM120 required")
    config = PolicyConfig(precision="bf16", model_dir=str(CKPT), num_steps=10,
                          batch_sizes={"embed": [1, 2, 3], "encode": [1, 2], "decode": [1, 2, 4]})
    common = torch.cuda.Stream()
    streams = {stage: [torch.cuda.Stream(), common] for stage in config.batch_sizes}
    result = backend.FlashRTPolicy(config, "cuda:0", streams=streams)
    yield Controller(result)
    torch.cuda.synchronize()
    result.policy.close()
    del result
    gc.collect()
    torch.cuda.empty_cache()


class Controller:
    """Test scheduler: all dependency/serialization events live outside backend."""
    def __init__(self, backend_policy):
        self.backend = backend_policy
        self.done = {}
        self.retained = []

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def submit(self, stage, items, stream=None):
        stream = stream if stream is not None else self.streams[stage][0]
        with torch.cuda.stream(stream):
            if stage in self.done:
                stream.wait_event(self.done[stage])
            for item in items:
                if "ready" in item:
                    stream.wait_event(item["ready"])
            outputs = getattr(self.backend, stage)(items, stream=stream)
            assert all("ready" not in item for item in outputs)
            done = torch.cuda.Event()
            done.record(stream)
        self.done[stage] = done
        # Keep tensors alive until all consuming work has completed.
        self.retained.append((done, items))
        return [dict(item, ready=done) for item in outputs]

    def embed(self, items, stream=None):
        return self.submit("embed", items, stream)

    def encode(self, items, stream=None):
        return self.submit("encode", items, stream)

    def decode(self, items, stream=None):
        return self.submit("decode", items, stream)

    def postprocess(self, items):
        for item in items:
            item["ready"].synchronize()
        return self.backend.postprocess(items)

    def infer(self, observations):
        if not observations:
            return []
        # External GPU noise may be produced on the caller's current stream.
        ready = torch.cuda.Event()
        ready.record()
        data = [dict(item, ready=ready) for item in self.preprocess(observations)]
        return self.postprocess(self.decode(self.encode(self.embed(data))))

    def make_example_input(self, bsz, stage="all"):
        # This convenience helper uses one registered stream for its full chain.
        for done in self.done.values():
            done.synchronize()
        common = self.streams["embed"][1]
        values = self.backend.make_example_input(bsz, stage, stream=common)
        common.synchronize()
        return values


def requests(n=2):
    rng = np.random.default_rng(33)
    return [InferenceRequest({
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.linspace(-0.1, 0.1, 8, dtype=np.float32) + i * 0.1,
        "prompt": "pick up the cup" if i == 0 else "move the green cube beside the red bowl on the left",
        "noise": rng.standard_normal((10, 32)).astype(np.float32),
    }) for i in range(n)]


def raw(items):
    for item in items:
        item["ready"].synchronize()
    return torch.stack([item["actions"].float().cpu() for item in items]).numpy()


@pytest.mark.parametrize("attn_cls,pipeline_cls,stage", [
    (backend.EmbedAttnBackend, backend.EmbedPipeline, "embed"),
    (backend.EncodeAttnBackend, backend.EncodePipeline, "encode"),
    (backend.DecodeAttnBackend, backend.DecodePipeline, "decode"),
])
def test_allocate_bind_capacity_and_minimal_buffers(attn_cls, pipeline_cls, stage):
    spec = backend._spec(max_prompt_len=8, chunk_size=4)
    a = attn_cls.allocate_buffers(3, spec)
    views = [attn_cls(b, a) for b in (1, 2, 3)]
    assert all(v.batch_size == b for v, b in zip(views, (1, 2, 3)))
    assert len({v.t["q"].data_ptr() for v in views}) == 1
    assert len({v.get_ptrs_b2().get("enc_k_layer_stride_bytes") for v in views}) == 1
    with pytest.raises(ValueError, match="capacity"):
        attn_cls(4, a)
    p = pipeline_cls.allocate_buffers(3, spec)
    assert not any("fp8" in key or "int8" in key for key in p.tensors)
    if stage == "embed":
        assert not any("encoder" in key or "decoder" in key for key in p.tensors)
        assert p.tensors["vision_QKV_b2"].data_ptr() == p.tensors["vision_hidden_b2"].data_ptr()
        assert backend.EmbedAttnBackend.allocate_buffer(1, spec).capacity == 1
    else:
        name = "encoder" if stage == "encode" else "decoder"
        assert p.tensors[name+"_QKV_b2"].data_ptr() == p.tensors[name+"_gate_merged_b2"].data_ptr()
    with pytest.raises(ValueError, match="capacity"):
        pipeline_cls.allocate_buffers(0, spec)


def test_initialization_registry_weights_runners_and_graphs(policy):
    frontend = policy.policy
    variants = frontend.pipelines
    assert frontend.attn_backend is None  # no unused monolithic attention buffers
    assert len({id(v[1].gemm) for v in variants.values()}) == 3
    assert len({id(p.weights) for v in variants.values() for p in v.values()}) == 1
    for stage, entries in variants.items():
        a, b = entries[1], entries[2]
        assert a.gemm is b.gemm and a.storage is b.storage
        assert a.attn.storage is b.attn.storage
        assert len(a._graphs) == len(policy.streams[stage])
        for stream in policy.streams[stage]:
            ga, gb = a._graphs[stream.cuda_stream], b._graphs[stream.cuda_stream]
            assert ga is not gb and ga.captured and gb.captured
        assert all(a.bufs[k].ptr.value == b.bufs[k].ptr.value for k in a.bufs)
        assert frontend.get_pipeline(stage, 1) is a
        with pytest.raises(ValueError, match="pipeline capacity"):
            a.init_buffers(a.storage, a.storage.capacity + 1)
    for args in (("other", 1), ("embed", 99)):
        with pytest.raises(ValueError, match="unsupported"):
            frontend.get_pipeline(*args)


def test_compile_and_autotune_are_idempotent(policy):
    frontend = policy.policy
    before = [dict(p._graphs) for v in frontend.pipelines.values() for p in v.values()]
    sizes = {s: len(v[1].storage.tuned_shapes) for s, v in frontend.pipelines.items()}
    policy.compile(policy.streams)
    assert before == [p._graphs for v in frontend.pipelines.values() for p in v.values()]
    for stage, variants in frontend.pipelines.items():
        with torch.cuda.stream(policy.streams[stage][0]), torch.inference_mode():
            variants[1].autotune_gemms()
        assert len(variants[1].storage.tuned_shapes) == sizes[stage]
    with pytest.raises(ValueError, match="streams"):
        frontend.compile({})
    other = dict(policy.streams, embed=torch.cuda.Stream())
    with pytest.raises(ValueError, match="nonempty list"):
        frontend.compile(other)
    torch.cuda.synchronize()


def test_preprocess_state_tokens_and_validation(policy):
    req = requests()
    images_before = req[0].observation["observation/image"].copy()
    data = policy.preprocess(req)
    assert len(data) == 2 and len(data[0]["tokens"]) != len(data[1]["tokens"])
    np.testing.assert_allclose(data[0]["images"][0], images_before.astype(np.float32)/127.5-1)
    np.testing.assert_array_equal(req[0].observation["observation/image"], images_before)
    changed = requests(1)
    changed[0].observation["observation/state"] += 0.6
    assert policy.preprocess(changed)[0]["tokens"] != data[0]["tokens"]
    with pytest.raises(ValueError, match="nonempty"):
        policy.preprocess([])
    for key, value, error in [("observation/state", np.zeros(7), "state"),
                              ("observation/state", np.full(8, np.nan), "state"),
                              ("observation/image", np.zeros((4, 4, 3)), "images"),
                              ("prompt", None, "prompt"),
                              ("prompt", "hello " * 500, "capacity")]:
        bad = requests(1)
        bad[0].observation[key] = value
        with pytest.raises(ValueError, match=error):
            policy.preprocess(bad)
    bad = requests(1)
    bad[0].inference_type = "rtc"
    with pytest.raises(ValueError, match="sync"):
        policy.preprocess(bad)


def test_stage_inputs_outputs_lifetime_and_noise(policy):
    data = policy.preprocess(requests())
    embeddings = policy.embed(data)
    assert all(x["vision"].shape == (512, 1152) for x in embeddings)
    embeddings[0]["ready"].synchronize()
    saved = embeddings[0]["vision"].clone()
    contexts = policy.encode(embeddings)
    assert contexts[0]["prefix_len"] != contexts[1]["prefix_len"]
    for context in contexts:
        assert context["k"].shape == (18, context["prefix_len"], 1, 256)
    actions = policy.decode(contexts)
    baseline = raw(actions)
    assert np.isfinite(baseline).all()
    assert not np.array_equal(baseline[0], baseline[1])
    # Subsequent use of all stage buffers must not invalidate old outputs.
    policy.infer(requests(1))
    np.testing.assert_array_equal(raw(actions), baseline)
    torch.testing.assert_close(embeddings[0]["vision"], saved, rtol=0, atol=0)
    np.testing.assert_array_equal(raw(policy.decode(contexts)), baseline)
    np.testing.assert_array_equal(data[0]["noise"], requests()[0].observation["noise"])
    no_noise = [dict(x, noise=None) for x in contexts]
    assert not np.array_equal(raw(policy.decode(no_noise)), raw(policy.decode(no_noise)))
    flipped = [dict(x, noise=-x["noise"]) for x in contexts]
    assert not np.array_equal(raw(policy.decode(flipped)), baseline)


def test_rebatch_permute_and_graph_eager_parity(policy):
    inputs = policy.preprocess(requests())
    contexts = policy.encode(policy.embed(inputs))
    baseline = raw(policy.decode(contexts))
    reversed_actions = raw(policy.decode(list(reversed(contexts))))
    np.testing.assert_allclose(reversed_actions, baseline[::-1], atol=0.02, rtol=0.02)
    serial = np.concatenate([raw(policy.decode(policy.encode(policy.embed([item])))) for item in inputs])
    np.testing.assert_allclose(serial, baseline, atol=0.04, rtol=0.02)
    graphs = [(p, p._graphs) for v in policy.policy.pipelines.values() for p in v.values()]
    try:
        for p, _ in graphs:
            p._graphs = None
        eager = raw(policy.decode(policy.encode(policy.embed(inputs))))
    finally:
        for p, graph in graphs:
            p._graphs = graph
    np.testing.assert_array_equal(eager, baseline)


def test_concurrent_stages_and_stream_handoffs(policy):
    data = policy.preprocess(requests())
    embeddings = policy.embed(data)
    contexts = policy.encode(embeddings)
    reference = raw(policy.decode(contexts))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for _ in range(3):
            emb = pool.submit(policy.embed, data)
            enc = pool.submit(policy.encode, embeddings)
            dec = pool.submit(policy.decode, contexts)
            new_embeddings, new_contexts, new_actions = emb.result(), enc.result(), dec.result()
            np.testing.assert_array_equal(raw(new_actions), reference)
            np.testing.assert_array_equal(raw(policy.decode(new_contexts)), reference)
            np.testing.assert_array_equal(raw(policy.decode(policy.encode(new_embeddings))), reference)
    # GPU noise produced on a nondefault caller stream is also ordered.
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        req = requests(1)
        req[0].observation["noise"] = torch.zeros(10, 32, device="cuda", dtype=torch.bfloat16).t().contiguous().t()
        gpu_result = policy.infer(req)[0].actions
    req[0].observation["noise"] = np.zeros((10, 32), np.float32)
    np.testing.assert_array_equal(policy.infer(req)[0].actions, gpu_result)


def test_native_single_sample_parity(policy):
    """Independent original FlashRT pipeline, exact-length prefixes, same weights/noise."""
    from flash_rt.hardware.rtx.attn_backend import RtxFlashAttnBackend
    from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
    data = policy.preprocess(requests())
    embeddings = policy.embed(data)
    actual = raw(policy.decode(policy.encode(embeddings)))
    for b, item in enumerate(embeddings):
        item["ready"].synchronize()
        n = len(item["language"])
        attn = RtxFlashAttnBackend(2, 512+n, 10)
        runner = backend.flash_rt_kernels.GemmRunner()
        native = Pi05Pipeline(runner, backend.flash_rt_kernels, attn,
                              policy.policy._build_pipeline_weights(), num_views=2,
                              max_prompt_len=n, chunk_size=10, num_steps=10,
                              use_fp8=False, use_fp8_decoder=False)
        native.set_language_embeds(item["language"].view(torch.uint16).cpu().numpy())
        images = torch.as_tensor(data[b]["images"], device="cuda", dtype=torch.bfloat16)
        noise = torch.as_tensor(data[b]["noise"], device="cuda", dtype=torch.bfloat16)
        policy.policy._copy_tensor_to_pipeline_buf(images, native.input_images_buf)
        policy.policy._copy_tensor_to_pipeline_buf(noise, native.input_noise_buf)
        native.forward(stream=torch.cuda.current_stream().cuda_stream)
        ref_bits = native.input_noise_buf.download_new((10, 32), np.uint16)
        expected = torch.from_numpy(ref_bits).view(torch.bfloat16).float().numpy()
        np.testing.assert_allclose(actual[b], expected, atol=0.05, rtol=0.025)
        del native, attn, runner


def test_pipeline_input_validation(policy):
    data = policy.preprocess(requests(1))
    embeddings = policy.embed(data)
    contexts = policy.encode(embeddings)
    for stage in ("embed", "encode", "decode"):
        with torch.cuda.stream(policy.streams[stage][0]), pytest.raises(ValueError, match="expected 1 inputs"):
            policy.policy.get_pipeline(stage, 1).set_input([])
    bad = [dict(data[0], images=np.zeros((2, 10, 10, 3), np.float32))]
    with pytest.raises(ValueError, match="shape"):
        policy.embed(bad)
    bad = [dict(embeddings[0], language=embeddings[0]["language"][:0])]
    with pytest.raises(ValueError, match="language length"):
        policy.encode(bad)
    with pytest.raises(ValueError, match="prefix"):
        policy.decode([dict(contexts[0], prefix_len=99999)])
    for noise in (np.zeros((10, 7), np.float32), np.zeros((10, 32), np.int32)):
        with pytest.raises(ValueError, match="floating tensor"):
            policy.decode([dict(contexts[0], noise=noise)])


def test_postprocess_infer_and_example_inputs(policy):
    req = requests(1)
    result = policy.infer(req)
    assert result[0].actions.shape == (10, 7)
    assert result[0].rtc_prev_actions.shape == (10, 32)
    assert policy.infer([]) == []
    for stage, key in (("embed", "tokens"), ("encode", "vision"),
                       ("decode", "k"), ("postprocess", "actions")):
        assert key in policy.make_example_input(1, stage)[0]
    assert isinstance(policy.make_example_input(1, "preprocess")[0], InferenceRequest)
    all_values = policy.make_example_input(1)
    assert len(all_values) == 6 and len(all_values[-1]) == 1
    with pytest.raises(ValueError, match="unknown stage"):
        policy.make_example_input(1, "missing")


@pytest.mark.parametrize("sizes", [{}, {"embed": [1]}, {s: [] for s in ("embed", "encode", "decode")},
                                   {s: [0] for s in ("embed", "encode", "decode")}])
def test_invalid_batch_configuration(sizes):
    with pytest.raises(ValueError, match="batch_sizes"):
        backend.FlashRTFrontEnd(sizes, CKPT)


@pytest.mark.parametrize("changes,error", [({"precision": "fp16"}, "BF16"),
    ({"model_name": "other"}, "Pi0.5"), ({"num_views": 3}, "two"), ({"num_steps": 0}, "positive")])
def test_invalid_policy_configuration(changes, error):
    cfg = PolicyConfig(precision="bf16")
    for key, value in changes.items():
        setattr(cfg, key, value)
    with pytest.raises(ValueError, match=error):
        backend.FlashRTPolicy(cfg, "cuda:0")


def test_different_stage_capacities_and_batch_sizes(policy):
    data = policy.preprocess(requests(3))
    embeddings = policy.embed(data)  # B=3
    contexts = policy.encode(embeddings[:2]) + policy.encode(embeddings[2:])  # B=2 then B=1
    expanded = [contexts[2], contexts[0], contexts[1], contexts[0]]
    batched = raw(policy.decode(expanded))  # B=4, a different stage capacity
    serial = np.concatenate([raw(policy.decode([item])) for item in expanded])
    np.testing.assert_allclose(batched, serial, atol=0.04, rtol=0.025)
    assert policy.policy.pipelines["embed"][1].storage.capacity == 3
    assert policy.policy.pipelines["encode"][1].storage.capacity == 2
    assert policy.policy.pipelines["decode"][1].storage.capacity == 4


def test_factory_dispatch(monkeypatch):
    from robort.policies import create_policy
    sentinel = object()
    config = PolicyConfig(backend="flash_rt", precision="bf16")
    streams = {stage: [object()] for stage in ("embed", "encode", "decode")}
    def construct(cfg, device, streams=None):
        assert streams is not None
        assert cfg is config and device == "cuda:0"
        return sentinel
    monkeypatch.setattr(backend, "FlashRTPolicy", construct)
    assert create_policy(config, "cuda:0", streams=streams) is sentinel


def test_stream_switching_and_backend_event_contract(policy, monkeypatch):
    torch.cuda.synchronize()
    data = policy.preprocess(requests())
    contexts = policy.encode(policy.embed(data))
    baseline = raw(policy.decode(contexts))
    old_outputs = policy.embed(data)
    old_vision = old_outputs[0]["vision"]
    old_outputs[0]["ready"].synchronize()
    saved = old_vision.clone()
    torch.cuda.synchronize()
    # Backend must neither wait on incoming events nor produce outgoing events.
    class ForbiddenEvent:
        def synchronize(self):
            raise AssertionError("backend owns no waits")
    stream = policy.streams["embed"][1]
    poison = [dict(item, ready=ForbiddenEvent()) for item in data]
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda.Stream, "wait_event", lambda *a, **k: pytest.fail("backend wait"))
        patch.setattr(torch.cuda, "Event", lambda *a, **k: pytest.fail("backend event"))
        outputs = policy.backend.embed(poison, stream=stream)
    stream.synchronize()
    assert all("ready" not in item for item in outputs)
    torch.testing.assert_close(old_vision, saved, rtol=0, atol=0)
    # Change stream and batch size while old outputs remain live.
    for index in (1, 0, 1):
        emb = policy.embed(data, policy.streams["embed"][index])
        enc = policy.encode(emb, policy.streams["encode"][index])
        actions = policy.decode(enc, policy.streams["decode"][index])
        np.testing.assert_array_equal(raw(actions), baseline)
        single = policy.decode([contexts[0]], policy.streams["decode"][1-index])
        np.testing.assert_allclose(raw(single)[0], baseline[0], atol=0.04, rtol=0.025)


def test_explicit_stream_validation_and_convenience(policy):
    torch.cuda.synchronize()
    data = policy.preprocess(requests(1))
    with pytest.raises(ValueError, match="unregistered"):
        policy.backend.embed(data, torch.cuda.Stream())
    with pytest.raises(ValueError, match="explicit stream"):
        policy.backend.infer(requests(1))
    with pytest.raises(ValueError, match="explicit stream"):
        policy.backend.make_example_input(1, "encode")
    for bad in ({}, dict(policy.streams, embed=[]), dict(policy.streams, embed=[object()])):
        with pytest.raises(ValueError):
            policy.compile(bad)
    shared = policy.streams["embed"][1]
    result = policy.backend.infer(requests(1), stream=shared)
    expected = policy.infer(requests(1))
    np.testing.assert_array_equal(result[0].actions, expected[0].actions)
    assert policy.backend.infer([]) == []


def test_fresh_eager_policy_and_close(policy):
    # Run last: reuse loaded weights but exercise constructor with capture disabled.
    from unittest.mock import patch
    from dataclasses import replace
    torch.cuda.synchronize()
    frontend = policy.policy
    graphs = [g for variants in frontend.pipelines.values() for p in variants.values()
              for g in p._graphs.values()]
    frontend.close()
    assert all(not graph._release.alive for graph in graphs)
    assert all(p._graphs is None for variants in frontend.pipelines.values() for p in variants.values())
    frontend.close()  # idempotent
    with patch.object(backend, "FlashRTFrontEnd", return_value=frontend):
        eager = backend.FlashRTPolicy(replace(policy.config, use_cuda_graph=False), "cuda:0")
    assert all(isinstance(streams, list) and len(streams) == 1 for streams in eager.streams.values())
    stream = eager.streams["embed"][0]
    actual = eager.infer(requests(1), stream=stream)
    eager.compile(eager.streams)
    captured = eager.infer(requests(1), stream=stream)
    np.testing.assert_array_equal(actual[0].actions, captured[0].actions)


@pytest.mark.parametrize("runtime,accepted", [(11070, False), (12040, False),
                                               (12080, True), (13010, False)])
def test_cuda_runtime_validation(monkeypatch, runtime, accepted):
    from types import SimpleNamespace
    def get_version(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = runtime
        return 0
    monkeypatch.setattr(backend, "_cudart", SimpleNamespace(cudaRuntimeGetVersion=get_version))
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    if accepted:
        backend._validate_cuda_runtime()
    else:
        with pytest.raises(RuntimeError, match="LD_LIBRARY_PATH"):
            backend._validate_cuda_runtime()
