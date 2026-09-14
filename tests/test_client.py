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
