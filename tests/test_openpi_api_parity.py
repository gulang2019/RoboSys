"""Real-checkpoint parity against OpenPI's public Policy.infer API.

Run from the repo root (requires the OpenPI environment and a CUDA GPU):
    ROBORT_TEST_OPENPI=1 PYTHONPATH=src .venv-openpi/bin/python -m pytest \
        tests/test_openpi_api_parity.py -v

Optional overrides: ROBORT_OPENPI_CHECKPOINT and ROBORT_OPENPI_DEVICE.
The reference uses the same loaded weights but OpenPI's own complete pipeline.
Single requests compare directly with the public API. Batches additionally use
OpenPI's native batched sampler to separate decomposition errors from BF16
batch-size rounding differences. API batch-vs-single drift is recorded in JUnit
properties (add --junitxml=/tmp/openpi-parity.xml to save it).
"""

import copy
import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ROBORT_TEST_OPENPI") != "1",
    reason="Set ROBORT_TEST_OPENPI=1 to run real-checkpoint GPU parity tests",
)
torch = pytest.importorskip("torch")
pytest.importorskip("openpi")

from robort.policies import openpi as policy_module
from robort.policies.openpi import OpenPIPolicy
from robort.schemas import InferenceRequest, PolicyConfig


@pytest.fixture(scope="module")
def policy():
    checkpoint = Path(os.environ.get(
        "ROBORT_OPENPI_CHECKPOINT",
        Path(__file__).resolve().parents[1] / "checkpoints/pi05_libero_pytorch",
    ))
    # Explicitly opted-in runs must fail if prerequisites are missing.
    assert (checkpoint / "model.safetensors").is_file(), checkpoint
    assert torch.cuda.is_available(), "CUDA GPU required for real-model parity tests"
    # Compare eager computations: compilation is a separate optimization contract.
    get_config = policy_module.get_config

    def eager_config(name):
        config = get_config(name)
        return dataclasses.replace(
            config, model=dataclasses.replace(config.model, pytorch_compile_mode=None)
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(policy_module, "get_config", eager_config)
        result = OpenPIPolicy(
            PolicyConfig(model_dir=str(checkpoint), max_batch_size=2),
            device=os.environ.get("ROBORT_OPENPI_DEVICE", "cuda:0"),
        )
    return result


def make_requests(batch_size):
    rng = np.random.default_rng(731)
    prompts = ["pick up the red block", "open the drawer and put the small bowl inside"]
    return [InferenceRequest({
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.uniform(-0.2, 0.2, 8).astype(np.float32),
        "prompt": prompts[i],
    }) for i in range(batch_size)]


@pytest.mark.parametrize("num_steps", [1, 3, 10])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_stages_match_openpi(policy, monkeypatch, record_property, batch_size, num_steps):
    monkeypatch.setattr(policy.config, "num_sample_steps", num_steps)
    monkeypatch.setitem(policy.policy._sample_kwargs, "num_steps", num_steps)
    requests = make_requests(batch_size)
    model_config = policy._model.config
    noise = np.random.default_rng(123).standard_normal(
        (batch_size, model_config.action_horizon, model_config.action_dim), dtype=np.float32
    )

    observation = policy.preprocess(copy.deepcopy(requests))
    embedding = policy.get_embeddings(observation)
    context = policy.encode(embedding)
    outputs = policy.decode(context, noise=torch.from_numpy(noise.copy()).to(policy.device))
    actual = policy.postprocess(outputs)
    assert len(actual) == batch_size
    assert not outputs["actions"].requires_grad

    # Observe the API's sampler boundary without replacing its computation.
    # This checks raw model-space actions too, before output transforms can hide errors.
    captured = []
    original_sampler = policy.policy._sample_actions

    def capture_sampler(device, obs, **kwargs):
        actions = original_sampler(device, obs, **kwargs)
        captured.append((obs, actions.detach().cpu()))
        return actions

    # The public API only accepts one request. For batches, also compare against
    # OpenPI's unmodified batched sampler at the same batch shape. BF16 kernels
    # can round differently when separate single requests are batched together.
    if batch_size > 1:
        native_batch = original_sampler(
            policy.device, observation,
            noise=torch.from_numpy(noise.copy()).to(policy.device), num_steps=num_steps,
        )
        torch.testing.assert_close(outputs["actions"], native_batch, rtol=1e-5, atol=1e-5)

    monkeypatch.setattr(policy.policy, "_sample_actions", capture_sampler)
    for i, request in enumerate(requests):
        expected = policy.policy.infer(copy.deepcopy(request.observation), noise=noise[i].copy())
        reference_obs, reference_actions = captured[-1]
        torch.testing.assert_close(observation.state[i:i + 1], reference_obs.state, rtol=0, atol=0)
        for key in observation.images:
            torch.testing.assert_close(observation.images[key][i:i + 1], reference_obs.images[key], rtol=0, atol=0)
            torch.testing.assert_close(observation.image_masks[key][i:i + 1], reference_obs.image_masks[key])
        torch.testing.assert_close(observation.tokenized_prompt[i:i + 1], reference_obs.tokenized_prompt)
        torch.testing.assert_close(observation.tokenized_prompt_mask[i:i + 1], reference_obs.tokenized_prompt_mask)
        if batch_size > 1:
            # Record the API's batch-size drift; do not relax parity tolerances
            # to accommodate it. Raw batch parity was checked above.
            drift = (native_batch[i:i + 1].cpu() - reference_actions).abs().max().item()
            record_property(f"request_{i}_batch_vs_single_max_abs_error", drift)
            expected = policy.policy._output_transform({
                "state": reference_obs.state[0].cpu().numpy(),
                "actions": native_batch[i].cpu().numpy(),
            })
        else:
            torch.testing.assert_close(outputs["actions"][i:i + 1].cpu(), reference_actions, rtol=1e-5, atol=1e-5)
        assert actual[i].actions.shape == expected["actions"].shape
        assert np.isfinite(actual[i].actions).all()
        np.testing.assert_allclose(actual[i].actions, expected["actions"], rtol=1e-5, atol=1e-5)


def test_infer_matches_public_api_with_seeded_noise(policy):
    # Exercise the convenience pipeline and its internally sampled noise as well.
    request = make_requests(1)[0]
    with torch.random.fork_rng(devices=[torch.device(policy.device)]):
        torch.manual_seed(456)
        actual, = policy.infer([copy.deepcopy(request)])
        torch.manual_seed(456)
        expected = policy.policy.infer(copy.deepcopy(request.observation))
    np.testing.assert_allclose(actual.actions, expected["actions"], rtol=1e-5, atol=1e-5)
