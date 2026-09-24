from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from robort.profile.profiler import Profiler
from robort.profile.runner import Runner
from robort.profile.schemas import PolicyConfig, RunnerConfig


@pytest.fixture
def clean_cuda():
    # Reclaim graph/compile objects from earlier tests before measuring deltas.
    import gc
    gc.collect()
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_peak_additional_memory_includes_freed_temporaries(clean_cuda):
    stream = torch.cuda.Stream()
    profiler = Profiler()
    with torch.cuda.stream(stream):
        resident = torch.ones(1_000_000, device='cuda')

        def operation():
            temporary = torch.ones_like(resident)
            return temporary + resident

        with profiler.measure(stream=stream):
            for _ in range(3):
                operation()
                profiler.tick()
        values = profiler.memories
        assert values == pytest.approx([8_000_000 / 1e9] * 3, abs=2048 / 1e9)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = operation()
        with profiler.measure(stream=stream):
            for _ in range(2):
                graph.replay()
                profiler.tick()
        values = profiler.memories
        assert values == [0, 0]  # Captured buffers are already allocated.
        torch.testing.assert_close(output, torch.full_like(output, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_profile_batch_returns_per_stage_sizes_and_warms_up(clean_cuda):
    stream = torch.cuda.Stream()
    runner = Runner(RunnerConfig(num_warmup=2, num_iter=3))
    runner.profiler = Profiler(power_w=lambda: 10)
    calls = []
    examples = []

    def make_input(size, stage):
        examples.append((stage, size))
        return torch.ones(size, 1024, device='cuda')

    def operation(x):
        assert not torch.is_grad_enabled()
        calls.append(x.shape[0])
        return x + 1

    policy = SimpleNamespace(config=PolicyConfig(batch_sizes={'embed': [1, 2], 'decode': [4]}),
                             make_example_input=make_input, embed=operation, decode=operation)
    hardware = SimpleNamespace(_device_index=stream.device.index, sm_perc=1., power_perc=1.)
    with torch.cuda.stream(stream):
        profile = runner._profile_batch(policy, hardware, stream)
    assert profile.policy_config == policy.config
    assert profile.policy_config is not policy.config
    assert examples == [('embed', 1), ('embed', 2), ('decode', 4)]
    assert calls == [1] * 5 + [2] * 5 + [4] * 5  # warmup + combined timing/memory
    for name, sizes in policy.config.batch_sizes.items():
        assert set(profile.stages[name].lat) == set(sizes)
        for size in sizes:
            mean, std = profile.stages[name].mem_fp_activation_gb[size]
            assert mean == pytest.approx(size * 1024 * 4 / 1e9)
            assert std == pytest.approx(0)


def test_profile_policy_passes_correct_arguments(monkeypatch):
    runner = Runner(RunnerConfig())
    config = PolicyConfig(model_dir='checkpoint', batch_sizes={'embed': [1]}, use_cuda_graph=False)
    policy = SimpleNamespace(config=config)
    hardware = SimpleNamespace(_device_index=0, sm_perc=1., power_perc=1.)
    stream = Mock()
    runner.backends[config.backend] = policy
    runner._configs[config.backend] = config
    runner._streams[(0, 1.)] = stream
    runner._profile_batch = Mock(return_value='profile')

    @contextmanager
    def environment(*args, **kwargs):
        yield stream

    monkeypatch.setattr('robort.profile.runner.prepare_hardware_env', environment)
    assert runner.profile_policy(hardware, config) == 'profile'
    runner._profile_batch.assert_called_once_with(policy, hardware, stream)
    stream.synchronize.assert_called_once()


def test_memory_statistics_written_to_csv(tmp_path, monkeypatch):
    import csv
    from dataclasses import fields
    from robort.profile import main as sweep
    from robort.profile.schemas import HardwareConfig, PolicyProfile, StageProfile

    def initialize(config):
        config.hardware_name = 'test'
        config.num_sms = 16
        config.mem_cap_gb = 24
        config.max_power_w = 450

    monkeypatch.setattr(HardwareConfig, '__post_init__', initialize)
    hardware = HardwareConfig(1., 1.)
    config = PolicyConfig(batch_sizes={'decode': [2]}, use_cuda_graph=False)
    result = PolicyProfile(hardware, config, {'decode': StageProfile(
        lat={2: (3., 0.1)}, energy={2: (4., 0.2)}, mem_fp_activation_gb={2: (0.5, 0.01)})})
    runner = Mock()
    runner.profile_hardware.side_effect = RuntimeError('skip hardware')
    runner.profile_policy.return_value = result
    monkeypatch.setattr(sweep, 'Runner', lambda _: runner)
    sweep.main(RunnerConfig(output_dir=str(tmp_path)),
               {'sm_perc': [1.], 'power_perc': [1.]},
               {f.name: [getattr(config, f.name)] for f in fields(config)})
    with (tmp_path / 'policy_profile.csv').open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert rows[0]['stage'] == 'decode'
    assert rows[0]['batch_size'] == '2'
    assert float(rows[0]['mem_fp_activation_gb_mean']) == 0.5
    assert float(rows[0]['mem_fp_activation_gb_std']) == 0.01
    runner.close.assert_called_once()


def test_openpi_stage_example_uses_small_upstream_batch():
    from robort.policies.openpi import OpenPIPolicy

    policy = OpenPIPolicy.__new__(OpenPIPolicy)
    policy.device = 'cpu'
    policy.preprocess = Mock(side_effect=lambda requests: {'state': torch.ones(len(requests), 8)})
    policy.embed = SimpleNamespace(function=Mock(side_effect=lambda x: x))
    policy.encode = SimpleNamespace(function=Mock(side_effect=lambda x: x))
    policy.decode = Mock(side_effect=AssertionError('decode must not run to prepare decode input'))
    policy._model = SimpleNamespace(config=SimpleNamespace(action_horizon=4, action_dim=8),
                                    sample_noise=lambda shape, device: torch.zeros(shape))
    example = policy.make_example_input(32, stage='decode')
    assert len(policy.preprocess.call_args.args[0]) == 1
    assert example['state'].shape == (32, 8)
    assert example['noise'].shape == (32, 4, 8)
    assert example['state'].is_contiguous()
    policy.decode.assert_not_called()


def test_tick_excludes_memory_bookkeeping_and_resets_baseline(monkeypatch):
    import sys

    clock = [0.]
    allocated = [100]
    peak = [100]

    def read(value):
        clock[0] += 10  # Deliberately expensive bookkeeping.
        return value[0]

    def reset(device):
        clock[0] += 10
        peak[0] = allocated[0]

    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                           memory_allocated=lambda device: read(allocated),
                           max_memory_allocated=lambda device: read(peak),
                           reset_peak_memory_stats=reset)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    profiler = Profiler(power_w=lambda: 10, synchronize=lambda: None, clock=lambda: clock[0])
    with profiler:
        clock[0] += 1
        allocated[0], peak[0] = 120, 150
        profiler.tick()
        clock[0] += 2
        allocated[0], peak[0] = 120, 130
        profiler.tick()
        assert profiler.latencies == [1000, 2000]
        assert profiler.energies == [10, 20]
        assert profiler.memories == [50 / 1e9, 10 / 1e9]
    with profiler:
        assert profiler.memories == []
        # Exiting without a tick must not record a sample.
    assert profiler.memories == []


def test_cpu_tick_reports_unavailable_memory(monkeypatch):
    import math
    import sys

    monkeypatch.setitem(sys.modules, 'torch', None)
    with Profiler(power_w=lambda: 10, synchronize=lambda: None) as profiler:
        profiler.tick()
    assert len(profiler.memories) == 1
    assert math.isnan(profiler.memories[0])
