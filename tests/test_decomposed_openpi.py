"""Regression checks for stage boundaries without loading model weights."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openpi")
from robort.policies.openpi import OpenPIPolicy
from robort.schemas import InferenceRequest, PolicyConfig


@pytest.fixture
def policy():
    policy = OpenPIPolicy.__new__(OpenPIPolicy)
    policy.config = PolicyConfig(num_sample_steps=4)
    policy.device = "cpu"
    policy.policy = SimpleNamespace(
        _input_transform=lambda data: {
            "state": data["state"],
            "image": {"camera": np.zeros((4, 4, 3), dtype=np.uint8)},
            "image_mask": {"camera": np.True_},
        },
        _output_transform=lambda data: {"actions": data["actions"] + data["state"][0]},
    )
    return policy


def test_preprocess_stacks_transformed_requests(policy):
    requests = [InferenceRequest({"state": np.full(8, i, np.float32)}) for i in range(2)]
    obs = policy.preprocess(requests)
    assert obs.state.shape == (2, 8)
    assert obs.state[:, 0].tolist() == [0, 1]
    assert obs.images["camera"].shape == (2, 3, 4, 4)
    assert obs.images["camera"].min() == -1
    assert obs.image_masks["camera"].dtype == torch.bool
    assert set(requests[0].observation) == {"state"}


def test_postprocess_preserves_batch_order(policy):
    outputs = {"state": torch.tensor([[1.0], [7.0]]), "actions": torch.zeros(2, 3, 2)}
    responses = policy.postprocess(outputs)
    assert len(responses) == 2
    np.testing.assert_array_equal(responses[0].actions, np.ones((3, 2)))
    np.testing.assert_array_equal(responses[1].actions, np.full((3, 2), 7))
    assert outputs["actions"].shape == (2, 3, 2)


def test_decode_steps_and_disables_autograd(policy):
    times = []
    def denoise(state, masks, cache, actions, time):
        assert not torch.is_grad_enabled()
        times.append(time[0].item())
        return torch.ones_like(actions)
    policy._model = SimpleNamespace(denoise_step=denoise)
    context = {"state": torch.zeros(2, 8), "prefix_pad_masks": torch.ones(2, 3), "past_key_values": object()}
    outputs = policy.decode(context, noise=torch.zeros(2, 3, 2))
    assert times == [1.0, 0.75, 0.5, 0.25]
    torch.testing.assert_close(outputs["actions"], -torch.ones(2, 3, 2))


def test_batch_validation(policy):
    assert policy.infer([]) == []
    with pytest.raises(ValueError, match="nonempty"):
        policy.preprocess([])
    with pytest.raises(ValueError, match="exceeds"):
        policy.preprocess([InferenceRequest({})] * 11)
    with pytest.raises(ValueError, match="sync"):
        policy.preprocess([InferenceRequest({}, inference_type="rtc")])


def test_invalid_step_count():
    with pytest.raises(ValueError, match="num_sample_steps"):
        OpenPIPolicy(PolicyConfig(num_sample_steps=0))
