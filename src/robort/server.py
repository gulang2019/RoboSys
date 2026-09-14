import logging
import time
import uuid

import asyncio
import tyro
from websockets.asyncio.server import serve

import robort.utils as utils
from robort.policy import create_policy
from robort.schemas import (
    ServerConfig,
    InferenceRequest
)

logger = logging.getLogger(__name__)


class PolicyServer:
    def __init__(self, config: ServerConfig):
        logger.info("Loading policy %s from %s", config.policy_config.model_name, config.policy_config.model_dir)
        self.policy = create_policy(config.policy_config)
        logger.info("Policy loaded and warmed up")
        self.max_batch_size = config.policy_config.max_batch_size
        self.config = config
        self.response_queues: dict[str, asyncio.queues.Queue] = {}
        self.payloads = []

    async def add_request(self):
        req_id = str(uuid.uuid4())
        q = asyncio.queues.Queue()
        self.response_queues[req_id] = q
        logger.info("Client connected: %s (clients=%d)", req_id, len(self.response_queues))
        return req_id, q

    async def add_payload(self, req_id, payload: InferenceRequest):
        self.payloads.append((req_id, payload))
        logger.info("Request queued: client=%s mode=%s pending=%d", req_id, payload.inference_type, len(self.payloads))

    async def _loop(self):
        next_sch_time = time.time() + self.config.timeout

        while True:
            current_time = time.time()
            if current_time >= next_sch_time or \
                len(self.payloads) >= self.config.max_batch_size:
                next_sch_time = current_time + self.config.timeout
                if len(self.payloads):
                    batch_payloads = self.payloads[:self.config.max_batch_size]
                    self.payloads = self.payloads[self.config.max_batch_size:]
                    req_ids, batch = zip(*batch_payloads)
                    logger.info("Inference started: batch_size=%d pending=%d", len(batch), len(self.payloads))
                    started = time.perf_counter()
                    responses = await asyncio.to_thread(self.policy.infer_batch, batch)
                    logger.info("Inference completed: batch_size=%d duration_ms=%.1f", len(batch), (time.perf_counter() - started) * 1000)
                    for req_id, response in zip(req_ids, responses):
                        if req_id in self.response_queues:
                            self.response_queues[req_id].put_nowait(response)
            await asyncio.sleep(0.001)

    def finish_request(self, req_id):
        self.response_queues.pop(req_id)
        logger.info("Client disconnected: %s (clients=%d)", req_id, len(self.response_queues))


async def main(args):
    async def send(ws, req_id, q):
        while True:
            response = await q.get()
            await ws.send(utils.packb(response))
            logger.info("Response sent: client=%s actions_shape=%s", req_id, response.actions.shape)

    async def recv(ws, req_id):
        async for payload in ws:
            await policy_server.add_payload(
                req_id, InferenceRequest(**utils.unpackb(payload))
            )

    async def handler(ws):
        req_id, response_queue = await policy_server.add_request()
        try:
            async with asyncio.TaskGroup() as group:
                sender = group.create_task(send(ws, req_id, response_queue))
                try:
                    await recv(ws, req_id)
                finally:
                    sender.cancel()
        finally:
            policy_server.finish_request(req_id)

    policy_server = PolicyServer(args)
    async with asyncio.TaskGroup() as group:
        worker = group.create_task(policy_server._loop())
        try:
            async with serve(handler, args.host_addr, args.port) as server:
                logger.info("Listening on ws://%s:%d (max_batch_size=%d timeout=%.3fs)", args.host_addr, args.port, args.max_batch_size, args.timeout)
                await server.serve_forever()
        finally:
            worker.cancel()
            logger.info("Server stopping")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(main(tyro.cli(ServerConfig)))
