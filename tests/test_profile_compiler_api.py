"""Profiler ownership and the policy-level compilation API."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from robort.policies.compiler_utils import Compiler
from robort.profile.runner import Runner
from robort.profile.schemas import PolicyConfig, RunnerConfig


def test_runner_reuses_policy_compiler_and_streams_until_close(monkeypatch):
    from contextlib import contextmanager
    from robort import policies
    from robort.policies import compiler_utils
    from robort.profile import runner as module
    stream = Mock(device=torch.device('cuda:0'))
    monkeypatch.setattr(torch.cuda, 'device', lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, 'default_stream', lambda *a: stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *a: stream)
    events = []

    @contextmanager
    def green(*args):
        events.append('allocate')
        yield stream
        events.append('destroy')

    @contextmanager
    def hardware_env(config, stream=None):
        assert stream is not None
        yield stream

    monkeypatch.setattr(module, '_green_stream', green)
    monkeypatch.setattr(module, 'prepare_hardware_env', hardware_env)
    policy = object()
    factory = Mock(return_value=policy)
    monkeypatch.setattr(policies, 'create_policy', factory)
    compiler = Mock()
    compiler.compile.return_value = policy
    compiler.close.side_effect = lambda: events.append('close graphs')
    constructor = Mock(return_value=compiler)
    monkeypatch.setattr(compiler_utils, 'Compiler', constructor)
    config = PolicyConfig(batch_sizes=[1, 2])
    hw = SimpleNamespace(_device_index=0, num_sms=16, power_perc=1., sm_perc=.5)
    runner = Runner(RunnerConfig())
    runner._profile_batch = Mock(return_value='profile')
    runner.init_backend(config, [hw], [1, 2])
    for power in (1., .8):
        hw.power_perc = power
        assert runner.profile_policy(hw, config) == {1: 'profile', 2: 'profile'}
    factory.assert_called_once()
    compiler.compile.assert_called_once_with(policy, [1, 2])
    compiler.close.assert_not_called()
    assert events == ['allocate']
    runner.close()
    runner.close()
    assert events == ['allocate', 'close graphs', 'destroy']
    assert not runner.backends
    assert not hasattr(runner, 'policy_cache')



def test_compile_policy_uses_six_example_outputs():
    examples = [object() for _ in range(6)]
    policy = SimpleNamespace(make_example_input=Mock(return_value=examples),
                             embed=Mock(), encode=Mock(), decode=Mock())
    compiler = Compiler.__new__(Compiler)
    compiler._compile = Mock(side_effect=['embed', 'encode', 'decode'])
    methods = [policy.embed, policy.encode, policy.decode]
    assert compiler.compile(policy, [1, 2]) is policy
    policy.make_example_input.assert_called_once_with(1)
    assert [c.args for c in compiler._compile.call_args_list] == [
        (method, example, [1, 2]) for method, example in zip(methods, examples[1:4])]
    assert (policy.embed, policy.encode, policy.decode) == ('embed', 'encode', 'decode')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compile_policy_dispatches_on_supplied_stream():
    stream = torch.cuda.Stream()
    x = torch.ones(1, 4, device=stream.device)
    policy = SimpleNamespace(
        make_example_input=lambda bsz: ([], x, x + 1, x + 2, x + 3, []),
        embed=lambda x: x + 1, encode=lambda x: x + 1, decode=lambda x: x + 1)
    with Compiler(device=stream.device, streams=[stream]) as compiler:
        compiler.compile(policy, [1, 2])
        with torch.cuda.stream(stream):
            for size in (1, 2):
                source = torch.ones(size, 4, device=stream.device)
                torch.testing.assert_close(policy.decode(policy.encode(policy.embed(source))), source + 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('mismatches,fallback', [(5, False), (6, True)])
def test_precision_gate_allows_point_five_percent(monkeypatch, mismatches, fallback):
    def operator(x):
        return x + 1

    def compile_with_error(fn, **kwargs):
        def compiled(x):
            result = fn(x)
            result[:, :mismatches] += 1
            return result
        return compiled

    monkeypatch.setattr(torch, 'compile', compile_with_error)
    stream = torch.cuda.Stream()
    x = torch.zeros(1, 1000, device=stream.device)
    with Compiler(device=stream.device, streams=[stream]) as compiler:
        warning = pytest.warns(RuntimeWarning, match='limit 0.5%') if fallback else nullcontext()
        with warning:
            dispatch = compiler._compile(operator, x)
        program = dispatch.programs[(stream.cuda_stream, 1)]
        assert program.graph is not None
        assert (program._function is operator) == fallback
        with torch.cuda.stream(stream):
            result = dispatch(x)
            expected = operator(x)
            if not fallback:
                expected[:, :mismatches] += 1
            torch.testing.assert_close(result, expected)
