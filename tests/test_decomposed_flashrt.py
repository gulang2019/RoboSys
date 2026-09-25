"""Validation and optional real-GPU parity for decomposed FlashRT inference."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
from robort.policies import PolicyConfig, create_policy
from robort.policies.flash_rt import FlashRTPolicy, flashrt_hardware
from robort.schemas import InferenceRequest


def request():
    return InferenceRequest({
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(8, np.float32),
        "prompt": "pick up the red block",
    })


@pytest.mark.parametrize(('capability', 'hardware'), [
    ((8, 9), 'rtx_sm89'), ((11, 0), 'thor'), ((12, 0), 'rtx_sm120'),
])
def test_flashrt_hardware_dispatch(capability, hardware):
    assert flashrt_hardware(capability) == hardware


def test_flashrt_hardware_dispatch_rejects_unknown_capability():
    with pytest.raises(NotImplementedError, match='SM87'):
        flashrt_hardware((8, 7))


@pytest.fixture
def policy():
    policy = FlashRTPolicy.__new__(FlashRTPolicy)
    policy.frontend = SimpleNamespace(norm_stats={"state": {"q01": [-2] * 8, "q99": [2] * 8}})
    policy._active = policy._stage = None
    return policy


def test_preprocess_normalizes_pads_and_preserves_input(policy):
    req = request()
    req.observation["observation/state"][:] = 1
    obs = policy.preprocess([req])
    np.testing.assert_allclose(obs.state[0, :8], 0.5, atol=1e-6)
    np.testing.assert_array_equal(obs.state[0, 8:], 0)
    assert obs.state.shape == (1, 32)
    np.testing.assert_array_equal(req.observation["observation/state"], 1)
    policy.preprocess([request()])
    with pytest.raises(ValueError, match="current preprocess"):
        policy.embed(obs)


def test_unsupported_requests(policy):
    assert policy.infer([]) == []
    with pytest.raises(ValueError, match="nonempty"):
        policy.preprocess([])
    with pytest.raises(ValueError, match="batch_size=1"):
        policy.preprocess([request(), request()])
    req = request()
    req.inference_type = "rtc"
    with pytest.raises(ValueError, match="sync"):
        policy.preprocess([req])
    req = request()
    req.observation["observation/image"] = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(ValueError, match="HWC"):
        policy.preprocess([req])


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown policy backend"):
        create_policy(PolicyConfig(backend="missing"))


@pytest.mark.skipif(os.environ.get("ROBORT_TEST_FLASHRT") != "1", reason="requires FlashRT, RTX GPU and checkpoint")
def test_gpu_parity_with_native_pipeline():
    import torch

    policy = FlashRTPolicy(PolicyConfig(backend="flashrt", use_cuda_graph=False))
    req = request()
    noise = torch.randn(1, 10, 32, device="cuda", dtype=torch.float16)
    for prompt in ("pick up the red block", "close the drawer"):
        req.observation["prompt"] = prompt
        obs = policy.preprocess([req])
        context = policy.encode(policy.embed(obs))
        outputs = policy.decode(context, noise=noise)
        assert np.isfinite(outputs["actions"]).all()
        responses = policy.postprocess(outputs)
        assert responses[0].actions.shape == (10, 7)
        frontend = policy.frontend
        # Restore identical noise and images; native forward reruns all stages.
        with torch.inference_mode():
            frontend._noise_buf.copy_(noise[0])
            frontend._copy_tensor_to_pipeline_buf(frontend._noise_buf, frontend.pipeline.input_noise_buf)
            torch.cuda.synchronize()
            frontend.pipeline.forward()
            import ctypes
            frontend._cudart.cudaMemcpy(
                ctypes.c_void_p(frontend._noise_out.data_ptr()),
                frontend.pipeline.input_noise_buf.ptr, frontend._noise_out.numel() * 2, 3,
            )
            native = frontend._noise_out.float().cpu().numpy()[None]
        np.testing.assert_allclose(outputs["actions"], native, rtol=0, atol=0)


@pytest.mark.parametrize('sizes', [None, [1], {'embed': [1], 'encode': [1], 'decode': [1]}])
def test_constructor_accepts_single_batch_config_before_cuda_check(sizes):
    with pytest.raises(RuntimeError, match='requires CUDA'):
        FlashRTPolicy(PolicyConfig(batch_sizes=sizes, precision='fp16'), device='cpu')


def test_constructor_rejects_multiple_batches():
    with pytest.raises(NotImplementedError, match='batch_size=1'):
        FlashRTPolicy(PolicyConfig(batch_sizes=[1, 2], precision='fp16'), device='cpu')
