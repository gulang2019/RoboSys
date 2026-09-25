import torch


from .config import PolicyConfig
from ..schemas import (InferenceRequest,
                            InferenceResponse)

class VLABasePolicy:
    def __init__(self, policy_config: PolicyConfig, device: str, streams: dict[str, list] | None = None):
        '''
        streams: StageName -> list[Streams], the list of streams executable of the stage.
        at runtime, the stream is configured by external runtime.
        '''
        self.streams = streams
        self.config = policy_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess(self, observations, stream):
        ...

    def embed(self, observations, stream):
        ...

    def encode(self, embeddings, stream):
        ...

    def decode(self, context, stream):
        ...

    def postprocess(self, actions, stream):
        ...

    def infer(self, observations: list[InferenceRequest], stream=None) -> list[InferenceResponse]:
        kwargs = {} if stream is None else {"stream": stream}
        observations = self.preprocess(observations, **kwargs)
        embeddings = self.embed(observations, **kwargs)
        context = self.encode(embeddings, **kwargs)
        actions = self.decode(context, **kwargs)
        actions = self.postprocess(actions, **kwargs)
        return actions

    def make_example_input(self, bsz: int, stage = 'all', stream=None) -> tuple | object:
        import numpy as np

        requests = [InferenceRequest(observation={
            'observation/image': np.zeros((224, 224, 3), dtype=np.uint8),
            'observation/wrist_image': np.zeros((224, 224, 3), dtype=np.uint8),
            'observation/state': np.zeros(8, dtype=np.float32),
            'prompt': 'do something useful',
        }) for _ in range(bsz)]
        kwargs = {} if stream is None else {"stream": stream}
        observation = self.preprocess(requests, **kwargs)
        embedding = self.embed(observation, **kwargs)
        context = self.encode(embedding, **kwargs)
        decoded = self.decode(context, **kwargs)
        if stage == 'all':
            return requests, observation, embedding, context, decoded, self.postprocess(decoded, **kwargs)
        return {
            'preprocess': requests, 
            'embed': observation,
            'encode': embedding,
            'decode': context, 
            'postprocess': decoded
        }.get(stage, None)
