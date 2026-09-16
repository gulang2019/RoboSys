"""Optional CUDA graph execution for an unchanged, eager OpenPIPolicy."""

from contextlib import ExitStack

import torch
from openpi.models.model import Observation
from transformers.cache_utils import DynamicCache

from .cuda_graph import capture_torch
from .vla_base import VLABasePolicy


def _observation_tensors(observation):
    return {name: value for name, value in vars(observation).items() if value is not None}


def _context_tensors(context):
    return {**context, 'past_key_values': context['past_key_values'].to_legacy_cache()}


def _context_object(context):
    return {**context, 'past_key_values': DynamicCache.from_legacy_cache(context['past_key_values'])}


def _sampling_steps(config):
    return config.num_steps if config.num_sample_steps is None else config.num_sample_steps


class _TensorStages:
    """Adapt OpenPI objects at the tensor-only capture boundary."""

    def __init__(self, policy):
        self.policy = policy

    def embed(self, observation):
        return self.policy.embed(Observation(**observation))

    def encode(self, embedding):
        return _context_tensors(self.policy.encode(embedding))

    def decode(self, context, noise):
        return self.policy.decode(_context_object(context), noise=noise)


class CompiledOpenPIStages:
    """Reusable compiled functions, independent of CUDA graph/stream lifetime."""

    def __init__(self, policy):
        self.policy = policy
        adapters = _TensorStages(policy)
        self.functions = {
            name: torch.compile(getattr(adapters, name), options={'triton.cudagraphs': False})
            for name in ('embed', 'encode', 'decode')
        }
        self._prepared_sizes = set()

    @torch.inference_mode()
    def prepare(self, batch_sizes):
        """Compile new shapes before capture on a long-lived, caller-owned stream."""
        from torch.utils import _pytree

        def owned(tree):
            return _pytree.tree_map(lambda tensor: tensor.clone(), tree)

        for size in batch_sizes:
            if size in self._prepared_sizes:
                continue
            observation = owned(_observation_tensors(self.policy.preprocess(
                [self.policy.make_example_input() for _ in range(size)])))
            embedding = self.functions['embed'](observation=observation)
            context = self.functions['encode'](embedding=owned(embedding))
            model = self.policy._model
            noise = model.sample_noise(
                (size, model.config.action_horizon, model.config.action_dim), self.policy.device)
            self.functions['decode'](context=owned(context), noise=noise)
            torch.cuda.current_stream().synchronize()
            self._prepared_sizes.add(size)


class CudaGraphOpenPIPolicy(VLABasePolicy):
    """Own graph execution without changing the wrapped policy's methods/state.

    Construction captures embed/encode/decode for the requested batch sizes.
    Supply compiled_stages to reuse compilation across stream lifetimes; otherwise
    a new CompiledOpenPIStages instance is created. Pre/postprocessing delegate to the eager policy. Inputs have
    fixed shapes/dtypes; outputs share storage across replays. The wrapper is
    not thread-safe. Close it before destroying its stream. The underlying
    policy stays usable throughout, including after this wrapper is closed.
    """

    def __init__(self, policy, stream, *, batch_sizes=None, compiled_stages=None):
        super().__init__(policy.config, policy.device)
        sizes = batch_sizes if batch_sizes is not None else self.config.batch_sizes
        sizes = [1] if sizes is None else sizes
        if not sizes or any(type(n) is not int or not 1 <= n <= self.config.max_batch_size for n in sizes):
            raise ValueError('batch_sizes must contain positive integers <= max_batch_size')
        device = torch.device(self.device)
        if device.type != 'cuda' or (device.index is not None and device != stream.device):
            raise ValueError('Capture stream must belong to the policy CUDA device')
        self.policy = policy
        self._graphs = {}
        self._steps = _sampling_steps(self.config)
        if compiled_stages is not None and compiled_stages.policy is not policy:
            raise ValueError('Compiled stages belong to a different policy')
        compiled_stages = compiled_stages if compiled_stages is not None else CompiledOpenPIStages(policy)
        compiled = compiled_stages.functions
        try:
            for size in sorted(set(sizes)):
                stages = self._graphs[size] = {}
                batch = [policy.make_example_input() for _ in range(size)]
                observation = _observation_tensors(policy.preprocess(batch))
                stages['embed'] = capture_torch(compiled['embed'], {'observation': observation}, stream)
                embedding = stages['embed'](observation=observation)
                stages['encode'] = capture_torch(compiled['encode'], {'embedding': embedding}, stream)
                context = stages['encode'](embedding=embedding)
                noise = self._sample_noise(size)
                stages['decode'] = capture_torch(compiled['decode'], {'context': context, 'noise': noise}, stream)
                stages['decode'](context=context, noise=noise)
            stream.synchronize()
        except BaseException:
            self.close()
            raise

    def _graph(self, stage, state):
        if self._graphs is None:
            raise RuntimeError('CUDA graph policy has been closed')
        if _sampling_steps(self.config) != self._steps:
            raise ValueError('Sampling steps changed; create a new CUDA graph policy')
        size = state.shape[0]
        if size not in self._graphs:
            raise ValueError(f'Batch size {size} was not captured; captured sizes: {list(self._graphs)}')
        return self._graphs[size][stage]

    def _sample_noise(self, size):
        model = self.policy._model
        return model.sample_noise((size, model.config.action_horizon, model.config.action_dim), self.device)

    def preprocess(self, observations):
        return self.policy.preprocess(observations)

    def embed(self, observation):
        return self._graph('embed', observation.state)(observation=_observation_tensors(observation))

    def encode(self, embedding):
        return _context_object(self._graph('encode', embedding['state'])(embedding=embedding))

    def decode(self, context, noise=None):
        graph = self._graph('decode', context['state'])
        if noise is None:
            noise = self._sample_noise(context['state'].shape[0])
        return graph(context=_context_tensors(context), noise=noise)

    def postprocess(self, outputs):
        return self.policy.postprocess(outputs)

    def infer(self, observations):
        return super().infer(observations) if observations else []

    def close(self):
        graphs, self._graphs = self._graphs, None
        if graphs is not None:
            with ExitStack() as cleanup:
                for stages in graphs.values():
                    for graph in stages.values():
                        cleanup.callback(graph.close)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
