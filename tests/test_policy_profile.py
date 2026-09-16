from contextlib import contextmanager
from dataclasses import asdict
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robort.profile.policies import FlashRTBackend
from robort.profile.profiler import Profiler
from robort.profile.runner import Runner
from robort.profile.schemas import HardwareConfig, PolicyConfig, RunnerConfig, StageProfile


def stage_metadata(params=100):
    return StageProfile(params, 200, 0.1, 0.2, 0, 0, 0, 0)


def hardware(power=1, sm=1):
    return SimpleNamespace(power_perc=power, sm_perc=sm)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_units_statistics_and_telemetry_overhead():
    clock = Clock()
    powers = iter([10, 30, 50])

    def power():
        clock.advance(10)  # Expensive telemetry must not inflate latency.
        return next(powers)

    sync = Mock()
    with Profiler(power_w=power, synchronize=sync, clock=clock) as p:
        clock.advance(1)
        p.tick()
        clock.advance(3)
        p.tick()
    assert p.latencies == [1000, 3000]
    assert p.energies == [20, 120]
    result = p.to_profile(stage_metadata())
    assert result.lat_mean == 2000
    assert result.lat_std == 1000
    assert result.energy_mean == 70
    assert result.energy_std == 50
    assert sync.call_count == 3


def test_synchronization_is_included():
    clock = Clock()
    with Profiler(power_w=lambda: 10, synchronize=lambda: clock.advance(2), clock=clock) as p:
        clock.advance(1)
        p.tick()
    assert p.latencies == [3000]
    assert p.energies == [30]


@pytest.mark.parametrize('power', [Mock(side_effect=RuntimeError('unsupported')),
                                   lambda: float('nan'), lambda: float('inf'), lambda: -1])
def test_missing_energy_keeps_latency(power):
    clock = Clock()
    with Profiler(power_w=power, synchronize=lambda: None, clock=clock) as p:
        clock.advance(1)
        p.tick()
    result = p.to_profile(stage_metadata())
    assert result.lat_mean == 1000
    assert result.lat_std == 0
    assert math.isnan(result.energy_mean)
    assert math.isnan(result.energy_std)


def test_reuse_exceptions_and_context_validation():
    clock = Clock()
    p = Profiler(power_w=lambda: 10, synchronize=lambda: None, clock=clock)
    with pytest.raises(RuntimeError, match='active'):
        p.tick()
    with pytest.raises(ValueError, match='no completed'):
        p.to_profile(stage_metadata())
    with pytest.raises(LookupError):
        with p:
            clock.advance(1)
            p.tick()
            raise LookupError('model failed')
    assert len(p.latencies) == 1
    with p:
        assert p.latencies == []
        with pytest.raises(RuntimeError, match='already active'):
            with p:
                pass
        clock.advance(2)
        p.tick()
    assert p.to_profile(stage_metadata()).lat_mean == 2000
    other = Profiler()
    assert other.latencies == []


def test_default_cuda_device_and_power_units(monkeypatch):
    import sys
    clock = Clock()
    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 2,
                           power_draw=Mock(return_value=20_000), synchronize=Mock())
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    with Profiler(clock=clock) as p:
        clock.advance(2)
        p.tick()
    assert p.energies == [40]
    assert all(call.args == (2,) for call in cuda.synchronize.call_args_list)
    assert all(call.args == (2,) for call in cuda.power_draw.call_args_list)


def test_cpu_default(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'torch', None)
    clock = Clock()
    with Profiler(clock=clock) as p:
        clock.advance(1)
        p.tick()
    assert p.latencies == [1000]
    assert math.isnan(p.energies[0])


@pytest.fixture
def experiment(monkeypatch):
    clock = Clock()
    runner = Runner(RunnerConfig(num_warmup=2, num_iter=3))
    runner.profiler = Profiler(power_w=lambda: 10, synchronize=lambda: None, clock=clock)
    names = ['preprocess', 'embed', 'encode', 'decode', 'postprocess']
    calls = []
    def stage(name):
        calls.append(name)
        clock.advance(names.index(name) + 1)
    targets = {name: Mock(side_effect=lambda name=name: stage(name)) for name in names}
    events = []
    stream = Mock()

    @contextmanager
    def environment(config):
        events.append('enter')
        try:
            yield stream
        finally:
            events.append('exit')

    monkeypatch.setattr('robort.profile.runner.prepare_hardware_env', environment)
    runner._test_policy = object()
    @contextmanager
    def execution(*args):
        yield runner._test_policy
    runner._execution = execution
    runner.make_model_functions = Mock(side_effect=lambda *args: dict(targets))
    runner._test_events, runner._test_stream, runner._test_calls = events, stream, calls
    config = PolicyConfig(backend='flash_rt', use_cuda_graph=False)
    return runner, targets, config


def test_runner_stage_order_warmup_statistics_and_reuse(experiment):
    runner, targets, config = experiment
    result = runner.profile_policy(hardware(), config)[1]
    assert list(result.stages) == list(targets)
    assert runner._test_calls == list(targets) * 5
    assert all(fn.call_count == 5 for fn in targets.values())
    for i, stage in enumerate(result.stages.values(), 1):
        assert stage.lat_mean == 1000 * i
        assert stage.energy_mean == 10 * i
        assert stage.lat_std == 0
        assert math.isnan(stage.num_params)
    assert len(runner.profiler.latencies) == 3
    runner.make_model_functions.assert_called_once_with(runner._test_policy, 1)
    assert runner._test_events == ['enter', 'exit']
    assert runner.profile_policy(hardware(), config)[1].stages['decode'].lat_mean == 4000


@pytest.mark.parametrize('name,value', [('num_iter', 0), ('num_iter', -1),
                                       ('num_iter', 1.5), ('num_iter', True),
                                       ('num_warmup', -1), ('num_warmup', False)])
def test_invalid_iterations_before_preparation(experiment, name, value):
    runner, _, config = experiment
    setattr(runner.args, name, value)
    with pytest.raises(ValueError, match=name):
        runner.profile_policy(hardware(), config)
    runner.make_model_functions.assert_not_called()


@pytest.mark.parametrize('field', ['power_perc', 'sm_perc'])
@pytest.mark.parametrize('partition', [-1, 2, float('nan')])
def test_invalid_hardware_before_preparation(experiment, partition, field):
    runner, _, config = experiment
    hw = hardware()
    setattr(hw, field, partition)
    with pytest.raises(ValueError, match=field):
        runner.profile_policy(hw, config)
    runner.make_model_functions.assert_not_called()


def test_runner_failure_and_recovery(experiment):
    runner, targets, config = experiment
    targets['embed'].side_effect = RuntimeError('model failed')
    with pytest.raises(RuntimeError, match='model failed'):
        runner.profile_policy(hardware(), config)
    targets['decode'].assert_not_called()
    assert runner._test_events == ['enter', 'exit']
    targets['embed'].side_effect = None
    assert runner.profile_policy(hardware(), config)[1].stages


def test_empty_stages(experiment):
    runner, _, config = experiment
    runner.make_model_functions.side_effect = lambda *args: {}
    with pytest.raises(ValueError, match='no profiling stages'):
        runner.profile_policy(hardware(), config)


def test_preparation_failure_is_not_masked(experiment):
    runner, _, config = experiment
    runner.make_model_functions.side_effect = RuntimeError('preparation')
    with pytest.raises(RuntimeError, match='preparation'):
        runner.profile_policy(hardware(), config)
    assert runner._test_events == ['enter', 'exit']
    runner._test_stream.synchronize.assert_called_once()


def test_graph_request_is_explicitly_rejected(experiment):
    runner, _, config = experiment
    config.use_cuda_graph = True
    with pytest.raises(NotImplementedError, match='openpi backend'):
        runner.profile_policy(hardware(), config)
    runner.make_model_functions.assert_not_called()


def test_zero_warmup_still_initializes_outside_timing(experiment):
    runner, targets, config = experiment
    runner.args.num_warmup = 0
    result = runner.profile_policy(hardware(), config)[1]
    assert runner._test_calls == list(targets) * 4
    assert result.stages['decode'].lat_mean == 4000


def test_native_outputs_are_passed_by_identity_and_released(monkeypatch):
    import weakref
    active = []
    events = []
    class Handle:
        def __deepcopy__(self, memo):
            raise AssertionError('Native handles must not be copied')
    class Policy:
        def make_example_input(self):
            return [Handle()]
        def preprocess(self, inputs):
            active[:] = [Handle()]
            return active[0]
        def step(self, value):
            assert value is active[0]
            active[:] = [Handle()]
            return active[0]
        embed = encode = decode = postprocess = step
        def __del__(self):
            events.append('policy released')
    policy = Policy()
    reference = weakref.ref(policy)
    runner = Runner(RunnerConfig())
    functions = runner.make_model_functions(policy, 1)
    del policy
    for _ in range(2):
        for fn in functions.values():
            fn()
    del fn
    functions.clear()
    assert reference() is None
    assert events == ['policy released']


def test_profiler_uses_supplied_stream(monkeypatch):
    clock = Clock()
    stream = Mock()
    stream.synchronize.side_effect = lambda: clock.advance(2)
    p = Profiler(power_w=lambda: 10, clock=clock)
    with p.measure(stream):
        clock.advance(1)
        p.tick()
    assert p.latencies == [3000]
    assert stream.synchronize.call_count == 2


def test_sync_failure_exits_environment(experiment):
    runner, _, config = experiment
    runner._test_stream.synchronize.side_effect = RuntimeError('sync failed')
    with pytest.raises(RuntimeError, match='sync failed'):
        runner.profile_policy(hardware(), config)
    assert runner._test_events == ['enter', 'exit']


def test_profiles_each_batch_size_separately(experiment):
    runner, targets, config = experiment
    config.batch_sizes = [2, 1, 2]
    profiles = runner.profile_policy(hardware(), config)
    assert list(profiles) == [1, 2]
    assert profiles[1].policy_config.batch_sizes == [1]
    assert profiles[2].policy_config.batch_sizes == [2]
    assert config.batch_sizes == [2, 1, 2]
    assert runner._test_calls == list(targets) * 10
    assert [call.args[1] for call in runner.make_model_functions.call_args_list] == [1, 2]
