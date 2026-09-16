"""Real-model cache/green-stream integration: ROBORT_TEST_OPENPI_PROFILE=1."""

import os
from dataclasses import replace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('ROBORT_TEST_OPENPI_PROFILE') != '1',
                                reason='requires OpenPI checkpoint, CUDA and NVML')


def test_repeated_graph_profiles_reuse_weights_and_compilation(monkeypatch):
    import torch
    import pynvml
    from torch._dynamo.utils import counters
    from robort.profile import policy_cache
    from robort.profile.environment import nvml_device_handle
    from robort.profile.runner import Runner
    from robort.profile.schemas import HardwareConfig, PolicyConfig, RunnerConfig

    hardware = HardwareConfig(1, 1)
    pynvml.nvmlInit()
    try:
        before_power = pynvml.nvmlDeviceGetPowerManagementLimit(
            nvml_device_handle(torch, pynvml, hardware._device_index))
    finally:
        pynvml.nvmlShutdown()
    hardware.power_perc = before_power / (hardware.max_power_w * 1000)
    factory = Mock(wraps=policy_cache.create_policy)
    compile_fn = Mock(wraps=torch.compile)
    monkeypatch.setattr(policy_cache, 'create_policy', factory)
    monkeypatch.setattr(torch, 'compile', compile_fn)
    config = PolicyConfig(model_dir='checkpoints/pi05_libero_pytorch',
                          batch_sizes=[1, 2], num_steps=3, use_cuda_graph=True)
    with Runner(RunnerConfig(num_warmup=1, num_iter=2)) as runner:
        first = runner.profile_policy(hardware, config)
        compile_calls = compile_fn.call_count
        unique_graphs = counters['stats']['unique_graphs']
        model = runner.policy_cache._model._model
        for sm in [1, 0.5]:
            hardware.sm_perc = sm
            results = runner.profile_policy(hardware, config)
            assert list(results) == [1, 2]
            for size, result in results.items():
                assert result.policy_config.batch_sizes == [size]
                assert list(result.stages) == ['preprocess', 'embed', 'encode', 'decode', 'postprocess']
                assert all(stage.lat_mean > 0 for stage in result.stages.values())
            assert runner.policy_cache._model._model is model
            assert compile_fn.call_count == compile_calls
            assert counters['stats']['unique_graphs'] == unique_graphs
        # Switching back to eager uses the same weights without compilation.
        runner.profile_policy(hardware, replace(config, use_cuda_graph=False, batch_sizes=[1]))
        assert runner.policy_cache._model._model is model
        assert compile_fn.call_count == compile_calls
        factory.assert_called_once()
        print('First graph profiles (ms):', {n: {s: round(v.lat_mean, 3) for s, v in p.stages.items()}
                                           for n, p in first.items()})
    assert runner.policy_cache._model is None
    pynvml.nvmlInit()
    try:
        assert pynvml.nvmlDeviceGetPowerManagementLimit(
            nvml_device_handle(torch, pynvml, hardware._device_index)) == before_power
    finally:
        pynvml.nvmlShutdown()
