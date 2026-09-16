"""Weights/compiled functions persist; execution graphs remain per stream."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip('torch')
from robort.policies.config import PolicyConfig
from robort.profile.policy_cache import PolicyCache
from robort.profile.runner import Runner
from robort.profile.schemas import RunnerConfig


@pytest.fixture
def cache_env(monkeypatch):
    from robort.profile import policy_cache as module
    stream = SimpleNamespace(synchronize=Mock())
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, 'default_stream', lambda *args: stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: stream)
    factory = Mock(side_effect=lambda config, device: SimpleNamespace(
        config=config, device=device, _model=object()))
    monkeypatch.setattr(module, 'create_policy', factory)
    return PolicyCache(), factory


def test_weights_reused_across_settings_and_config_is_copied(cache_env):
    cache, factory = cache_env
    config = PolicyConfig(use_cuda_graph=False, batch_sizes=[1])
    first, _ = cache.get(config, 'cuda:0', [1])
    second, _ = cache.get(replace(config, batch_sizes=[2]), 'cuda:0', [2])
    assert first is second
    config.num_steps = 3
    third, _ = cache.get(config, 'cuda:0', [1])
    assert third is not first
    assert third._model is first._model
    assert first.config.num_steps == 10
    factory.assert_called_once()
    cache.close()
    assert cache._model is None and not cache._variants


def test_model_or_device_change_invalidates_cache(cache_env):
    cache, factory = cache_env
    config = PolicyConfig(use_cuda_graph=False)
    first, _ = cache.get(config, 'cuda:0', [1])
    second, _ = cache.get(replace(config, model_dir='/different/checkpoint'), 'cuda:0', [1])
    third, _ = cache.get(config, 'cuda:1', [1])
    assert len({id(p._model) for p in (first, second, third)}) == 3
    assert factory.call_count == 3


def test_compiled_functions_reused_and_graph_execution_closed(cache_env, monkeypatch):
    pytest.importorskip('openpi')
    from robort.policies import openpi_cuda_graph as module
    cache, factory = cache_env
    compiler = Mock(side_effect=lambda policy: SimpleNamespace(policy=policy, prepare=Mock()))
    monkeypatch.setattr(module, 'CompiledOpenPIStages', compiler)
    wrappers = []
    class GraphPolicy:
        def __init__(self, policy, stream, **kwargs):
            self.policy, self.stream, self.compiled = policy, stream, kwargs['compiled_stages']
            self.closed = False
            wrappers.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            self.closed = True
    monkeypatch.setattr(module, 'CudaGraphOpenPIPolicy', GraphPolicy)
    runner = Runner(RunnerConfig())
    runner.policy_cache = cache
    config = PolicyConfig(use_cuda_graph=True)
    streams = [SimpleNamespace(device='cuda:0') for _ in range(2)]
    for stream in streams:
        with runner._execution(config, stream, [1]) as graph:
            assert not graph.closed
        assert graph.closed
    factory.assert_called_once()
    compiler.assert_called_once()
    assert wrappers[0].compiled is wrappers[1].compiled
    assert wrappers[0].stream is not wrappers[1].stream
    with pytest.raises(RuntimeError):
        with runner._execution(config, streams[0], [1]):
            raise RuntimeError('inference failure')
    assert wrappers[-1].closed
    runner.close()
    assert cache._model is None


def test_flashrt_graph_request_does_not_compile_openpi_stages(cache_env):
    cache, _ = cache_env
    config = PolicyConfig(backend='flash_rt', use_cuda_graph=True)
    policy, compiled = cache.get(config, 'cuda:0', [1])
    assert policy is not None
    assert compiled is None
