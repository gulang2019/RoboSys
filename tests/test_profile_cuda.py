"""Opt-in GPU integration tests; leave the existing NVML power limit unchanged."""

import math
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('ROBORT_TEST_CUDA') != '1',
                                reason='set ROBORT_TEST_CUDA=1 for live CUDA profiling tests')


@pytest.mark.parametrize('sm_fraction', [0.5, 1.0])
def test_live_green_runner(sm_fraction, monkeypatch):
    import pynvml
    import torch
    from cuda.bindings import driver

    from robort.profile.environment import _cuda_call, nvml_device_handle
    from robort.profile.runner import Runner
    from robort.profile.schemas import HardwareConfig, PolicyConfig, RunnerConfig

    config = HardwareConfig(1, sm_fraction)
    device = config._device_index
    pynvml.nvmlInit()
    try:
        handle = nvml_device_handle(torch, pynvml, device)
        before = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        if before > config.max_power_w * 1000:
            pytest.skip('current limit exceeds the default; test does not modify power')
        config.power_perc = before / (config.max_power_w * 1000)
        allocated = []

        class SyntheticPolicy:
            def __init__(self):
                self.a = torch.ones((256, 256), device=device)
                self.b = torch.ones_like(self.a)

            def make_example_input(self):
                return self.a

            def preprocess(self, value):
                stream = torch.cuda.current_stream(device)
                ctx = _cuda_call(driver.cuStreamGetCtx, stream.cuda_stream)
                resource = _cuda_call(driver.cuCtxGetDevResource, ctx,
                                     driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
                allocated.append(resource.sm.smCount)
                return value[0]

            def embed(self, value):
                return torch.mm(value, self.b)

            encode = decode = embed

            def postprocess(self, value):
                assert torch.isfinite(value).all()
                return value.cpu().numpy()

        monkeypatch.setattr('robort.profile.policy_cache.create_policy', lambda *args, **kwargs: SyntheticPolicy())
        runner = Runner(RunnerConfig(2, 5))
        policy = PolicyConfig(use_cuda_graph=False)
        for _ in range(2):
            result = runner.profile_policy(config, policy)[1]
            assert list(result.stages) == ['preprocess', 'embed', 'encode', 'decode', 'postprocess']
            for measured in result.stages.values():
                assert measured.lat_mean > 0
                assert math.isfinite(measured.energy_mean) and measured.energy_mean > 0
                assert math.isnan(measured.flops)
        runner.close()
        assert all(count >= math.ceil(config.num_sms * sm_fraction) for count in allocated)
        assert all(count <= config.num_sms for count in allocated)
        if sm_fraction == 0.5:
            assert all(count < config.num_sms for count in allocated)
        assert pynvml.nvmlDeviceGetPowerManagementLimit(handle) == before
    finally:
        pynvml.nvmlShutdown()
