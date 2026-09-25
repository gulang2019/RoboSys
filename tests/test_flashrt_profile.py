"""FlashRT profiler orchestration and opt-in real checkpoint/green-stream smoke."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
import math
import os
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch

from robort.profile import runner as module
from robort.profile.runner import Runner
from robort.profile.schemas import HardwareConfig, PolicyConfig, RunnerConfig


def hardware(sm=1., device=0):
    return SimpleNamespace(_device_index=device, sm_perc=sm, power_perc=1., num_sms=128)


def config(**kwargs):
    return PolicyConfig(backend='flash_rt', precision='bf16', **kwargs)


class Policy:
    def __init__(self, cfg, streams, events):
        self.config, self.streams, self.events = cfg, streams, events
        self.calls = []
        self.cycle = self  # teardown must collect cyclic owners before replacement
        self.policy = SimpleNamespace(close=lambda: events.append('close'))

    def make_example_input(self, bsz, stage, stream):
        assert stage == 'preprocess'
        return list(range(bsz))

    def __getattr__(self, stage):
        if stage not in Runner.stages:
            raise AttributeError(stage)
        def call(items, stream):
            if stage in Runner.gpu_stages:
                assert len(items) in self.config.batch_sizes[stage]
                assert stream in self.streams[stage]
            self.calls.append((stage, len(items), tuple(items)))
            return list(items)
        return call


@pytest.fixture
def setup(monkeypatch):
    events, created, refs = [], [], []
    @contextmanager
    def green(torch, device, sms):
        events.append(('create_stream', device, sms))
        stream = Mock(device=torch.device('cuda', device))
        stream.synchronize.side_effect = lambda: events.append('sync')
        created.append(stream)
        try:
            yield stream
        finally:
            events.append('destroy_stream')
    def create(cfg, device, streams):
        assert set(streams) == set(Runner.gpu_stages)
        if refs:
            assert refs[-1]() is None, 'previous model must be released before loading'
        events.append('load')
        policy = Policy(cfg, streams, events)
        refs.append(weakref.ref(policy))
        return policy
    monkeypatch.setattr(module, '_green_stream', green)
    monkeypatch.setattr('robort.policies.create_policy', create)
    monkeypatch.setattr(module, 'prepare_hardware_env', lambda *a, **kw: nullcontext(kw['stream']))
    runner = Runner(RunnerConfig(num_warmup=0, num_iter=2))
    samples = SimpleNamespace(latencies=[1., 3.], energies=[2., 4.], memories=[0.1, 0.3], tick=Mock())
    runner.profiler = SimpleNamespace(measure=lambda **kw: nullcontext(samples))
    yield runner, events, created
    runner.close()


def test_init_reuse_replace_cleanup(setup):
    runner, events, streams = setup
    cfg = config()
    original = deepcopy(cfg)
    runner.init_backend(cfg, [hardware(), hardware(), hardware(.5)])
    assert cfg == original
    assert len(streams) == 2
    policy = runner.backends['flash_rt']
    assert policy.config.model_dir == 'checkpoints/pi05_libero_pytorch'
    assert all(group == streams for group in policy.streams.values())
    del policy
    runner.init_backend(cfg, [hardware(), hardware(.5)])
    assert len(streams) == 2
    assert events.count('load') == 2 and events.count('close') == 1
    assert events.index('sync') < events.index('close')
    runner.close()
    assert events.index('close') < events.index('destroy_stream')
    assert not runner.backends and not runner._configs and not runner._streams
    runner.close()


def test_stage_preparation_rebatches_and_pads(setup):
    runner, _, streams = setup
    cfg = config(batch_sizes={'embed': [2], 'encode': [1], 'decode': [4]})
    runner.init_backend(cfg, [hardware()])
    policy = runner.backends['flash_rt']
    assert runner._example_input(policy, 5, 'decode', streams[0]) == list(range(5))
    assert [b for stage, b, _ in policy.calls if stage == 'embed'] == [2, 2, 2]
    assert [b for stage, b, _ in policy.calls if stage == 'encode'] == [1] * 5
    assert runner._example_input(policy, 3, 'postprocess', streams[0]) == list(range(3))
    assert policy.calls[-1][0:2] == ('decode', 4)
    streams[0].synchronize.assert_called()


def test_selected_stages_sizes_and_metadata(setup):
    runner, _, _ = setup
    cfg = config(batch_sizes={'embed': [1, 2], 'encode': [1], 'decode': [1, 4]})
    runner.init_backend(cfg, [hardware()])
    requested = replace(cfg, batch_sizes={'preprocess': [3], 'decode': [4], 'postprocess': [3]})
    result = runner.profile_policy(hardware(), requested)
    assert result.policy_config.batch_sizes == requested.batch_sizes
    assert list(result.stages) == list(requested.batch_sizes)
    assert result.stages['decode'].lat == {4: (2., 1.)}
    assert result.stages['decode'].energy == {4: (3., 1.)}
    policy = runner.backends['flash_rt']
    # Two measured B=4 calls and three B=1 preparation calls for postprocess.
    assert sum(stage == 'decode' for stage, _, _ in policy.calls) == 5
    with pytest.raises(ValueError, match='uninitialized batch'):
        runner.profile_policy(hardware(), replace(cfg, batch_sizes={'decode': [2]}))
    with pytest.raises(ValueError, match='differs'):
        runner.profile_policy(hardware(), replace(cfg, num_steps=3))
    with pytest.raises(ValueError, match='stream was not initialized'):
        runner.profile_policy(hardware(.5), cfg)


@pytest.mark.parametrize('changes', [dict(backend='openpi'), dict(batch_sizes={}),
    dict(batch_sizes={'bad': [1]}), dict(batch_sizes={'embed': [0]}),
    dict(batch_sizes={'decode': [True]}), dict(use_cuda_graph='yes')])
def test_invalid_config_rejected_before_loading(setup, changes):
    runner, events, _ = setup
    with pytest.raises(ValueError):
        runner.init_backend(replace(config(), **changes), [hardware()])
    assert not events


def test_invalid_hardware_and_runner_args(setup):
    runner, events, _ = setup
    for configs in ([], [hardware(), hardware(device=1)]):
        with pytest.raises(ValueError):
            runner.init_backend(config(), configs)
    runner.args.num_iter = 0
    with pytest.raises(ValueError, match='num_iter'):
        runner.init_backend(config(), [hardware()])
    assert not events


def test_failure_drains_and_close_releases(setup, monkeypatch):
    runner, events, streams = setup
    cfg = config()
    runner.init_backend(cfg, [hardware()])
    policy = runner.backends['flash_rt']
    def fail(*args, **kwargs):
        events.append('failure')
        raise RuntimeError('stage failed')
    monkeypatch.setattr(policy, 'encode', fail, raising=False)
    with pytest.raises(RuntimeError, match='stage failed'):
        runner.profile_policy(hardware(), cfg)
    assert events[-1] == 'sync'
    runner.close()
    assert events.index('failure') < events.index('close') < events.index('destroy_stream')


@pytest.mark.skipif(os.environ.get('ROBORT_TEST_FLASHRT_PROFILE') != '1',
                    reason='opt-in real FlashRT checkpoint and CUDA green streams')
@pytest.mark.parametrize('use_graph', [False, True])
def test_real_flashrt_green_stream_profile(use_graph):
    cfg = config(use_cuda_graph=use_graph, batch_sizes={
        'preprocess': [3], 'embed': [2], 'encode': [1], 'decode': [4], 'postprocess': [3]})
    hardware_configs = [HardwareConfig(1., 1.), HardwareConfig(1., .5)]
    with Runner(RunnerConfig(num_warmup=1, num_iter=2)) as runner:
        runner.init_backend(cfg, hardware_configs)
        policy = runner.backends['flash_rt']
        for hw in hardware_configs:
            result = runner.profile_policy(hw, cfg)
            assert list(result.stages) == list(cfg.batch_sizes)
            assert all(math.isfinite(mean) and mean > 0
                       for stage in result.stages.values() for mean, _ in stage.lat.values())
            assert runner.backends['flash_rt'] is policy
            print({'graph': use_graph, 'sm_perc': hw.sm_perc,
                   'latency_ms': {name: values.lat for name, values in result.stages.items()}})
        # Reuse initialized batches while profiling only the selected stage.
        subset = runner.profile_policy(hardware_configs[0], replace(cfg, batch_sizes={'decode': [4]}))
        assert list(subset.stages) == ['decode']


def test_flashrt_sweep_defaults_and_csv(setup, tmp_path, monkeypatch):
    import csv
    from robort.profile import main as sweep
    runner, events, _ = setup
    defaults = dict(sweep.POLICY_CHOICES)
    assert defaults['backend'] == ['flash_rt']
    assert defaults['num_views'] == [2] and defaults['image_resolution'] == [224]
    def detect(hw):
        hw._device_index = 0
        hw.hardware_name = 'test'
        hw.num_sms = 128
        hw.mem_cap_gb = 24
        hw.max_power_w = 450
    monkeypatch.setattr(HardwareConfig, '__post_init__', detect)
    monkeypatch.setattr(sweep, 'Runner', lambda args: runner)
    runner.profile_hardware = Mock(side_effect=RuntimeError('skip hardware microbenchmarks'))
    choices = {'batch_sizes': [{'embed': [2], 'encode': [1], 'decode': [1, 4]}],
               'num_steps': [2, 3]}
    args = RunnerConfig(output_dir=str(tmp_path))
    sweep.main(args, {'sm_perc': [1., .5]}, choices)
    with (tmp_path / 'policy_profile.csv').open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 16
    assert {row['backend'] for row in rows} == {'flash_rt'}
    assert {row['batch_size'] for row in rows if row['stage'] == 'decode'} == {'1', '4'}
    assert events.count('load') == events.count('close') == 2
    sweep.main(args, {'sm_perc': [1., .5]}, choices)
    with (tmp_path / 'policy_profile.csv').open() as source:
        assert len(list(csv.DictReader(source))) == 16  # repeated sweep merges rows
