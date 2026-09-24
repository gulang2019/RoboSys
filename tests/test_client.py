from unittest.mock import Mock

import numpy as np
import pytest

from robort import client as client_module
from robort import utils
from robort.schemas import InferenceRequest, InferenceResponse


@pytest.mark.parametrize("mode", ["sync", "rtc"])
def test_connect_and_infer_server_response(monkeypatch, mode):
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    websocket = Mock()
    raw = np.ones((3, 32), dtype=np.float32)
    websocket.recv.return_value = utils.packb(InferenceResponse(actions, raw))
    connect = Mock(return_value=websocket)
    monkeypatch.setattr(client_module.websockets.sync.client, "connect", connect)

    client = client_module.WebsocketClientPolicy("localhost", 8000)
    assert client.get_server_metadata() == {}
    websocket.recv.assert_not_called()
    request = InferenceRequest({"state": np.zeros(8)}, inference_type=mode,
                               previous_action=raw if mode == "rtc" else None)
    response = client.infer(request)

    assert connect.call_args.args == ("ws://localhost:8000",)
    np.testing.assert_array_equal(response.actions, actions)
    np.testing.assert_array_equal(response.rtc_prev_actions, raw)
    payload = utils.unpackb(websocket.send.call_args.args[0])
    assert payload["inference_type"] == mode
    if mode == "rtc":
        np.testing.assert_array_equal(payload["previous_action"], raw)
    np.testing.assert_array_equal(payload["observation"]["state"], np.zeros(8))
    client.close()
    websocket.close.assert_called_once_with()


def test_server_error_is_reported(monkeypatch):
    websocket = Mock()
    websocket.recv.return_value = "inference failed"
    monkeypatch.setattr(client_module.websockets.sync.client, "connect", Mock(return_value=websocket))
    client = client_module.WebsocketClientPolicy()
    with pytest.raises(RuntimeError, match="inference failed"):
        client.infer(InferenceRequest({}))


def test_local_client_uses_policy_batch_interface(monkeypatch):
    from benchmark import client as benchmark_client

    response = InferenceResponse(np.ones((10, 7)))
    policy = Mock()
    policy.infer.return_value = [response]
    factory = Mock(return_value=policy)
    monkeypatch.setattr(benchmark_client, "create_policy", factory)
    client = benchmark_client.LocalClientPolicy(model_dir="checkpoint", device="cpu", num_steps=4)
    config = factory.call_args.args[0]
    assert (config.backend, config.model_dir, config.num_steps) == ("openpi", "checkpoint", 4)
    assert factory.call_args.kwargs == {"device": "cpu"}
    request = InferenceRequest({"prompt": "pick up the cup"})
    assert client.infer(request) is response
    policy.infer.assert_called_once_with([request])
    with pytest.raises(ValueError, match="sync"):
        client.infer(InferenceRequest({}, inference_type="rtc"))
    client.close()
    policy.export_debug_videos.assert_called_once_with()
    client.close()
    policy.export_debug_videos.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        client.infer(request)


@pytest.mark.parametrize("value", [
    '{"type": "local", "num_steps": 4}',
    "{'type': 'local', 'num_steps': 4}",
])
def test_parse_client_dictionary(value):
    from benchmark.client import parse_client_args

    assert parse_client_args(value) == ("local", {"num_steps": 4})
    assert parse_client_args('{}') == ("websocket", {})
    assert parse_client_args('{"host": "localhost", "port": 9000}') == (
        "websocket", {"host": "localhost", "port": 9000})


@pytest.mark.parametrize("value", [
    '[]', '{1: "local"}', '{"type": "other"}', '{"type": "local", "host": "localhost"}',
    '{"unknown": 1}', '__import__("os").getcwd()',
])
def test_invalid_client_arguments(value):
    from benchmark.client import parse_client_args

    with pytest.raises(ValueError):
        parse_client_args(value)


def test_local_client_names_episode_artifacts(monkeypatch, tmp_path):
    from benchmark import client as benchmark_client

    policy = Mock()
    monkeypatch.setattr(benchmark_client, 'create_policy', Mock(return_value=policy))
    client = benchmark_client.LocalClientPolicy(debug_dir=str(tmp_path))
    client.reset()
    client.reset()
    assert [call.args[0] for call in policy.start_debug_episode.call_args_list] == [
        tmp_path / 'episode_000', tmp_path / 'episode_001']
    client.close()
    policy.export_debug_videos.assert_called_once_with()


def test_local_client_reconfigures_without_loading_weights(monkeypatch, tmp_path):
    from benchmark import client as benchmark_client
    from robort.policies import PolicyConfig

    policy = Mock(config=PolicyConfig())
    factory = Mock(return_value=policy)
    monkeypatch.setattr(benchmark_client, 'create_policy', factory)
    client = benchmark_client.LocalClientPolicy(debug_dir=str(tmp_path / 'first'))
    client.reset()
    client.configure_run(num_steps=2, encode_keep_rate=1/3, debug_dir=str(tmp_path / 'second'))
    client.reset()
    factory.assert_called_once()
    assert client.policy is policy
    assert policy.config.num_steps == 2
    assert policy.config.num_sample_steps is None
    assert policy.config.encode_keep_rate == 1/3
    policy.reset.assert_called_once()
    policy.export_debug_videos.assert_called_once()
    assert policy.start_debug_episode.call_args.args == (tmp_path / 'second/episode_000',)
