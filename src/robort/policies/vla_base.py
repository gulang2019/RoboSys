import torch


from .config import PolicyConfig
from ..schemas import (InferenceRequest,
                            InferenceResponse)

class VLABasePolicy:
    def __init__(self, policy_config: PolicyConfig, device: str):
        self.config = policy_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess(self, observations):
        ...

    def embed(self, observations):
        ...

    def encode(self, embeddings):
        ...

    def decode(self, context):
        ...

    def postprocess(self, actions):
        ...

    def infer(self, observations: list[InferenceRequest]) -> list[InferenceResponse]:
        observations = self.preprocess(observations)
        embeddings = self.embed(observations)
        context = self.encode(embeddings)
        actions = self.decode(context)
        actions = self.postprocess(actions)
        return actions

    def make_example_input(self, bsz: int, stage = 'all') -> tuple | object:
        import numpy as np

        requests = [InferenceRequest(observation={
            'observation/image': np.zeros((224, 224, 3), dtype=np.uint8),
            'observation/wrist_image': np.zeros((224, 224, 3), dtype=np.uint8),
            'observation/state': np.zeros(8, dtype=np.float32),
            'prompt': 'do something useful',
        }) for _ in range(bsz)]
        observation = self.preprocess(requests)
        embedding = self.embed(observation)
        context = self.encode(embedding)
        decoded = self.decode(context)
        if stage == 'all':
            return requests, observation, embedding, context, decoded, self.postprocess(decoded)
        return {
            'preprocess': requests, 
            'embed': observation,
            'encode': embedding,
            'decode': context, 
            'postprocess': decoded
        }.get(stage, None)
