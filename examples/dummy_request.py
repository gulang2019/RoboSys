import asyncio

import numpy as np
from armory_client import msgpack_numpy
from websockets.asyncio.client import connect


async def main():
    request = {
        "observation": {
            "state": np.zeros(8, dtype=np.float32),
            "image": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": "pick up the black bowl",
        },
        "inference_type": "sync",
    }
    async with connect("ws://127.0.0.1:8000") as ws:
        await ws.send(msgpack_numpy.packb(request))
        print("Dummy request sent; waiting for actions...", flush=True)
        response = msgpack_numpy.unpackb(await ws.recv())
        print("Actions shape:", response["actions"].shape)
        print(response["actions"])


if __name__ == "__main__":
    asyncio.run(main())
