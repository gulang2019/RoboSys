import asyncio
from contextlib import asynccontextmanager
import threading

import numpy as np

from armory_client import msgpack_numpy
from robort import server as server_module
from robort.schemas import InferenceRequest, InferenceResponse, PolicyConfig, ServerConfig


class FakePolicy:
    def __init__(self):
        self.batches = []
        self.thread_ids = []

    def infer_batch(self, batch):
        self.batches.append(list(batch))
        self.thread_ids.append(threading.get_ident())
        return [InferenceResponse(req.observation["state"]) for req in batch]


def make_server(monkeypatch, batch_size=2, timeout=0.01):
    policy = FakePolicy()
    monkeypatch.setattr(server_module, "create_policy", lambda config: policy)
    config = ServerConfig(timeout=timeout, policy_config=PolicyConfig(max_batch_size=batch_size))
    return server_module.PolicyServer(config), policy


@asynccontextmanager
async def running_worker(server):
    task = asyncio.create_task(server._loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def request(value):
    return InferenceRequest({"state": np.full(7, value, dtype=np.float32)})


def test_batch_limit_order_and_client_routing(monkeypatch):
    server, policy = make_server(monkeypatch)

    async def check():
        first, qa = await server.add_request()
        second, qb = await server.add_request()
        assert first != second
        assert qa is not qb
        for client, value in [(first, 1), (second, 2), (first, 3)]:
            await server.add_payload(client, request(value))
        async with running_worker(server):
            a1 = await asyncio.wait_for(qa.get(), 1)
            b1 = await asyncio.wait_for(qb.get(), 1)
            a2 = await asyncio.wait_for(qa.get(), 1)
        for response, value in [(a1, 1), (b1, 2), (a2, 3)]:
            np.testing.assert_array_equal(response.actions, np.full(7, value))
        assert [len(batch) for batch in policy.batches] == [2, 1]
        assert all(tid != threading.get_ident() for tid in policy.thread_ids)

    asyncio.run(check())


def test_timeout_dispatches_partial_batch(monkeypatch):
    server, policy = make_server(monkeypatch, batch_size=10)

    async def check():
        client, queue = await server.add_request()
        await server.add_payload(client, request(4))
        async with running_worker(server):
            response = await asyncio.wait_for(queue.get(), 1)
        np.testing.assert_array_equal(response.actions, np.full(7, 4))
        assert len(policy.batches) == 1

    asyncio.run(check())


def test_disconnected_client_does_not_break_other_responses(monkeypatch):
    server, _ = make_server(monkeypatch)

    async def check():
        gone, abandoned_queue = await server.add_request()
        live, queue = await server.add_request()
        await server.add_payload(gone, request(1))
        await server.add_payload(live, request(2))
        server.finish_request(gone)
        async with running_worker(server):
            response = await asyncio.wait_for(queue.get(), 1)
        assert gone not in server.response_queues
        assert abandoned_queue.empty()
        np.testing.assert_array_equal(response.actions, np.full(7, 2))

    asyncio.run(check())


def test_websocket_serialization_and_normal_disconnect(monkeypatch):
    server, _ = make_server(monkeypatch, batch_size=1)
    monkeypatch.setattr(server_module, "PolicyServer", lambda config: server)

    class FakeWebSocket:
        def __init__(self):
            self.sent = asyncio.Queue()

        async def messages(self):
            yield msgpack_numpy.packb(request(5))
            encoded = await asyncio.wait_for(self.sent.get(), 1)
            response = msgpack_numpy.unpackb(encoded)
            np.testing.assert_array_equal(response["actions"], np.full(7, 5))

        def __aiter__(self):
            return self.messages()

        async def send(self, message):
            assert isinstance(message, bytes)
            await self.sent.put(message)

    class FakeListener:
        def __init__(self, handler, host, port):
            assert (host, port) == (server.config.host_addr, server.config.port)
            self.handler = handler

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def serve_forever(self):
            await asyncio.wait_for(self.handler(FakeWebSocket()), 1)
            assert not server.response_queues

    monkeypatch.setattr(server_module, "serve", FakeListener)
    asyncio.run(asyncio.wait_for(server_module.main(server.config), 2))
