from unittest.mock import Mock

import numpy as np
import pytest

from armory.serving.rtc import InferType
from armory.serving.schemas import SlotData
from robort import policy as policy_module
from robort.schemas import InferenceRequest, InferenceResponse, PolicyConfig, ServerConfig


@pytest.fixture
def policy(monkeypatch):
    backend = Mock()
    monkeypatch.setattr(policy_module, "OpenPiPolicyFactory", Mock(return_value=lambda: backend))
    return policy_module.create_policy(PolicyConfig())


@pytest.mark.parametrize("model,factory_name", [
    ("pi05_libero", "OpenPiPolicyFactory"),
    ("gr00t-n1.7", "Gr00tPolicyFactory"),
])
def test_factory_selection_and_warmup(monkeypatch, model, factory_name):
    backend = Mock()
    factory = Mock(return_value=lambda: backend)
    monkeypatch.setattr(policy_module, factory_name, factory)
    config = PolicyConfig(model_name=model, model_dir="example/checkpoint", max_batch_size=3)

    result = policy_module.create_policy(config)

    env = policy_module.EnvMode.LIBERO_20
    if model == "pi05_libero":
        factory.assert_called_once_with(model, config.model_dir, config.num_sample_steps, env)
    else:
        factory.assert_called_once_with(model, env, config.model_dir)
    if model == "pi05_libero":
        backend.warmup.assert_called_once_with(3, batch_sizes=[3])
    else:
        backend.warmup.assert_called_once_with(3)
    assert result._policy is backend


@pytest.mark.parametrize("mode,expected", [
    ("sync", InferType.SYNC),
    ("vlash", InferType.VLASH),
])
def test_non_rtc_conversion(policy, mode, expected):
    observation = {"state": np.zeros(7)}
    slot = policy._convert_to_armory(InferenceRequest(observation, inference_type=mode))
    assert isinstance(slot, SlotData)
    assert slot.observation is observation
    assert slot.infer_type == expected
    assert slot.params is None
    assert slot.noise is None


@pytest.mark.parametrize("mode", ["rtc", "inference_time_rtc"])
def test_rtc_conversion(policy, mode):
    previous = np.zeros((10, 7), dtype=np.float32)
    request = InferenceRequest({}, mode, previous, 2, 3, 8)
    slot = policy._convert_to_armory(request)
    assert slot.infer_type == InferType.INFERENCE_TIME_RTC
    assert slot.params.prev_action is previous
    assert slot.params.s_param == 2
    assert slot.params.d_param == 3
    assert slot.max_execution_horizon == 8


def test_batch_preserves_order_and_actions(policy):
    requests = [InferenceRequest({"state": np.full(7, i)}) for i in range(2)]
    actions = [np.full((10, 7), i) for i in range(2)]
    policy._policy.infer_batch.return_value = [{"actions": a} for a in actions] + [{"actions": actions[0]}] * 8

    responses = policy.infer_batch(requests)

    [slots] = policy._policy.infer_batch.call_args.args
    assert len(slots) == 10
    for slot, request, response, action in zip(slots, requests, responses, actions):
        assert slot.observation is request.observation
        assert isinstance(response, InferenceResponse)
        assert response.actions is action


def test_first_rtc_request_has_no_conditioning(policy):
    slot = policy._convert_to_armory(InferenceRequest({}, inference_type="rtc"))
    assert slot.infer_type == InferType.INFERENCE_TIME_RTC
    assert slot.params is None


@pytest.mark.parametrize("flag", ["_is_pytorch_model", "_is_triton_optimized"])
def test_backends_that_ignore_rtc_are_rejected(policy, flag):
    setattr(policy._policy, flag, True)
    with pytest.raises(ValueError, match="RTC requires the OpenPI JAX backend"):
        policy._convert_to_armory(InferenceRequest({}, inference_type="rtc"))


def test_rtc_model_space_actions_are_returned(policy):
    actions = np.zeros((10, 7))
    raw = np.ones((10, 32))
    policy._policy.infer_batch.return_value = [{"actions": actions, "rtc_prev_actions": raw}] * 10
    response, = policy.infer_batch([InferenceRequest({})])
    assert response.actions is actions
    assert response.rtc_prev_actions is raw


def test_server_config_defaults_are_independent():
    first, second = ServerConfig(), ServerConfig()
    first.policy_config.max_batch_size = 2
    assert first.max_batch_size == 2
    assert second.max_batch_size == 10


def test_mixed_batch_pads_each_mode_and_restores_order(policy):
    previous = np.ones((10, 32), dtype=np.float32)
    requests = [
        InferenceRequest({"id": 0}, "rtc", previous, 4, 2, 6),
        InferenceRequest({"id": 1}),
        InferenceRequest({"id": 2}, "rtc"),
        InferenceRequest({"id": 3}, "vlash"),
    ]
    def infer(slots):
        assert len(slots) == 10
        modes = {isinstance(slot.params, policy_module.RTCParams) for slot in slots}
        assert len(modes) == 1
        if True in modes:
            for slot in slots:
                assert slot.params.prev_action.shape == (10, 32)
                assert slot.params.prev_action.dtype == np.float32
                assert slot.max_execution_horizon == 6
        return [{"actions": slot.observation["id"]} for slot in slots]
    policy._policy.infer_batch.side_effect = infer
    responses = policy.infer_batch(requests)
    assert [response.actions for response in responses] == [0, 1, 2, 3]
    assert policy._policy.infer_batch.call_count == 2
    assert len(requests) == 4


def test_empty_and_oversized_batches(policy):
    assert policy.infer_batch([]) == []
    with pytest.raises(ValueError, match="exceeds"):
        policy.infer_batch([InferenceRequest({})] * 11)
    policy._policy.infer_batch.assert_not_called()


@pytest.mark.parametrize("sizes", [[0, 10], [2], [11], [-1, 10]])
def test_invalid_padding_buckets(sizes):
    with pytest.raises(ValueError, match="batch_sizes"):
        policy_module.create_policy(PolicyConfig(batch_sizes=sizes))


def test_unsorted_buckets_use_smallest_fitting_size(monkeypatch):
    backend = Mock()
    monkeypatch.setattr(policy_module, "OpenPiPolicyFactory", Mock(return_value=lambda: backend))
    policy = policy_module.create_policy(PolicyConfig(batch_sizes=[10, 4, 4]))
    backend.warmup.assert_called_once_with(10, batch_sizes=[4, 10])
    backend.infer_batch.side_effect = lambda slots: [{"actions": 0} for slot in slots]
    assert len(policy.infer_batch([InferenceRequest({})] * 2)) == 2
    assert len(backend.infer_batch.call_args.args[0]) == 4
