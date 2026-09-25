from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch

from robort import policies
from robort.policies import compiler_utils
from robort.profile.runner import Runner
from robort.profile.schemas import PolicyConfig, RunnerConfig


@pytest.mark.parametrize('backend', ['openpi', 'flash_rt'])
@pytest.mark.parametrize('use_cuda_graph', [False, True])
def test_replacing_backend_releases_policy_before_loading_next(monkeypatch, use_cuda_graph, backend):
    stream = Mock()
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, 'default_stream', lambda *args: stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: stream)
    events, references = [], []

    class Policy:
        def __init__(self):
            self.cycle = self

    def create(*args, **kwargs):
        if references:
            assert references[-1]() is None
        policy = Policy()
        references.append(weakref.ref(policy))
        events.append('load')
        return policy

    class Compiler:
        def __init__(self, **kwargs):
            self.policy = None

        def compile(self, policy):
            self.policy = policy
            return policy

        def close(self):
            events.append('close')
            self.policy = None

    monkeypatch.setattr(policies, 'create_policy', create)
    monkeypatch.setattr(compiler_utils, 'Compiler', Compiler)
    runner = Runner(RunnerConfig())
    runner._streams[(0, 1.)] = stream
    config = PolicyConfig(backend=backend, use_cuda_graph=use_cuda_graph)
    hardware = SimpleNamespace(_device_index=0, sm_perc=1., power_perc=1., num_sms=128)
    try:
        runner.init_backend(config, [hardware])
        runner.init_backend(config, [hardware])
        assert events == (['load', 'close', 'load'] if use_cuda_graph and backend == 'openpi' else ['load', 'load'])
        assert runner._streams[(0, 1.)] is stream
    finally:
        runner.close()
