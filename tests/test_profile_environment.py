from contextlib import contextmanager
from dataclasses import asdict
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from robort.profile.environment import _green_stream, prepare_hardware_env
from robort.profile.schemas import HardwareConfig


@pytest.fixture
def devices(monkeypatch):
    events = []
    torch, nvml = MagicMock(), MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.current_device.return_value = 1
    torch.cuda.get_device_properties.return_value = SimpleNamespace(
        name='NVIDIA GeForce RTX 4090', multi_processor_count=128,
        total_memory=24_000_000_000, uuid='reordered-uuid')
    nvml.nvmlDeviceGetHandleByUUID.return_value = 'physical-gpu'
    nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 450_000
    nvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (150_000, 600_000)
    current = [300_000]
    nvml.nvmlDeviceGetPowerManagementLimit.side_effect = lambda handle: current[0]

    def set_power(handle, value):
        assert handle == 'physical-gpu'
        events.append(('power', value))
        current[0] = value

    nvml.nvmlDeviceSetPowerManagementLimit.side_effect = set_power
    nvml.nvmlShutdown.side_effect = lambda: events.append('shutdown')
    stream = Mock()
    stream.synchronize.side_effect = lambda: events.append('sync')

    @contextmanager
    def green(torch_arg, device, requested):
        assert torch_arg is torch
        events.append(('green', device, requested))
        try:
            yield stream
        finally:
            events.append('destroy')

    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'pynvml', nvml)
    monkeypatch.setattr('robort.profile.environment._green_stream', green)
    return SimpleNamespace(torch=torch, nvml=nvml, stream=stream, events=events, current=current)


def test_schema_detects_device_and_uses_nvml_default(devices):
    config = HardwareConfig(0.8, 0.5)
    assert asdict(config) == dict(power_perc=0.8, sm_perc=0.5,
                                 hardware_name='NVIDIA GeForce RTX 4090', num_sms=128,
                                 mem_cap_gb=24, max_power_w=450)
    assert config._device_index == 1
    devices.nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with('GPU-reordered-uuid')
    devices.nvml.nvmlDeviceGetHandleByIndex.assert_not_called()
    devices.nvml.nvmlShutdown.assert_called_once()


@pytest.mark.parametrize('field', ['power_perc', 'sm_perc'])
@pytest.mark.parametrize('value', [0, -0.1, 1.1, float('nan'), float('inf'), True, '0.5'])
def test_invalid_fractions_before_device_access(devices, field, value):
    args = dict(power_perc=1, sm_perc=1)
    args[field] = value
    with pytest.raises(ValueError, match=field):
        HardwareConfig(**args)
    devices.nvml.nvmlInit.assert_not_called()


def test_schema_query_failure_shuts_down(devices):
    devices.nvml.nvmlDeviceGetPowerManagementDefaultLimit.side_effect = RuntimeError('NVML query')
    with pytest.raises(RuntimeError, match='NVML query'):
        HardwareConfig(1, 1)
    devices.nvml.nvmlShutdown.assert_called_once()


def test_unsupported_power_control_allows_full_power(monkeypatch):
    class NotSupported(Exception):
        pass

    events = []
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.current_device.return_value = 0
    torch.cuda.get_device_properties.return_value = SimpleNamespace(
        name='NVIDIA Thor', multi_processor_count=20,
        total_memory=128_000_000_000, uuid='thor-uuid')
    nvml = MagicMock()
    nvml.NVMLError_NotSupported = NotSupported
    nvml.nvmlDeviceGetPowerManagementDefaultLimit.side_effect = NotSupported()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'pynvml', nvml)

    config = HardwareConfig(1, 1)
    assert config.max_power_w != config.max_power_w  # NaN

    stream = Mock()
    stream.synchronize.side_effect = lambda: events.append('sync')

    @contextmanager
    def green(*args):
        yield stream

    monkeypatch.setattr('robort.profile.environment._green_stream', green)
    with prepare_hardware_env(config):
        pass
    nvml.nvmlDeviceGetPowerManagementLimitConstraints.assert_not_called()
    nvml.nvmlDeviceGetPowerManagementLimit.assert_not_called()
    nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()

    config.power_perc = 0.8
    with pytest.raises(NotImplementedError, match='power_perc=1.0'):
        with prepare_hardware_env(config):
            pass


@pytest.mark.parametrize('failure', [False, True])
def test_power_apply_restore_and_stream_scope(devices, failure):
    config = HardwareConfig(0.8, 0.501)
    devices.events.clear()
    # Changing the current CUDA device must not redirect an existing config.
    devices.torch.cuda.current_device.return_value = 0
    try:
        with prepare_hardware_env(config) as stream:
            assert stream is devices.stream
            assert devices.current[0] == 360_000
            devices.torch.cuda.stream.assert_called_once_with(stream)
            devices.events.append('body')
            if failure:
                raise LookupError('inference failed')
    except LookupError:
        assert failure
    assert devices.current[0] == 300_000
    assert devices.events == [('green', 1, 65), ('power', 360_000), 'body',
                              'sync', ('power', 300_000), 'destroy', 'shutdown']
    devices.torch.cuda.device.assert_called_once_with(1)


def test_supplied_stream_is_reused_without_destroying_it(devices):
    config = HardwareConfig(0.8, 0.5)
    devices.stream.device = devices.torch.device('cuda', config._device_index)
    devices.events.clear()
    for _ in range(2):
        with prepare_hardware_env(config, stream=devices.stream) as stream:
            assert stream is devices.stream
            assert devices.current[0] == 360_000
        assert devices.current[0] == 300_000
    assert devices.events == [('power', 360_000), 'sync', ('power', 300_000), 'shutdown'] * 2


def test_full_power_is_applied_and_restored(devices):
    with prepare_hardware_env(HardwareConfig(1, 1)):
        assert devices.current[0] == 450_000
    assert devices.current[0] == 300_000


def test_unchanged_power_does_not_require_set_permission(devices):
    devices.current[0] = 450_000
    with prepare_hardware_env(HardwareConfig(1, 1)):
        pass
    devices.nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()


def test_below_minimum_fails_without_changing_power(devices):
    with pytest.raises(ValueError, match='outside NVML range'):
        with prepare_hardware_env(HardwareConfig(0.2, 0.5)):
            pytest.fail('must not yield')
    devices.nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()
    assert not any(isinstance(e, tuple) and e[0] == 'green' for e in devices.events)


def test_permission_error_propagates_and_destroys_stream(devices):
    devices.nvml.nvmlDeviceSetPowerManagementLimit.side_effect = PermissionError('NVML denied')
    with pytest.raises(PermissionError, match='NVML denied'):
        with prepare_hardware_env(HardwareConfig(0.8, 0.5)):
            pytest.fail('must not yield')
    assert devices.events[-2:] == ['destroy', 'shutdown']


def test_readback_mismatch_restores_power(devices):
    devices.nvml.nvmlDeviceGetPowerManagementLimit.side_effect = [300_000, 350_000]
    with pytest.raises(RuntimeError, match='instead of'):
        with prepare_hardware_env(HardwareConfig(0.8, 0.5)):
            pytest.fail('must not yield')
    assert devices.current[0] == 300_000


def test_restore_failure_still_releases_resources(devices):
    devices.nvml.nvmlDeviceSetPowerManagementLimit.side_effect = [None, RuntimeError('restore failed')]
    devices.nvml.nvmlDeviceGetPowerManagementLimit.side_effect = [300_000, 360_000]
    with pytest.raises(RuntimeError, match='restore failed'):
        with prepare_hardware_env(HardwareConfig(0.8, 0.5)):
            pass
    assert devices.events[-2:] == ['destroy', 'shutdown']


def test_green_creation_failure_never_changes_power(devices, monkeypatch):
    @contextmanager
    def fail(*args):
        raise RuntimeError('unsupported green context')
        yield
    monkeypatch.setattr('robort.profile.environment._green_stream', fail)
    with pytest.raises(RuntimeError, match='unsupported green'):
        with prepare_hardware_env(HardwareConfig(0.8, 0.5)):
            pass
    devices.nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()
    assert devices.events[-1] == 'shutdown'


@pytest.fixture
def cuda_driver(monkeypatch):
    events = []
    driver = SimpleNamespace(
        CUdevResourceType=SimpleNamespace(CU_DEV_RESOURCE_TYPE_SM=1),
        CUgreenCtxCreate_flags=SimpleNamespace(CU_GREEN_CTX_DEFAULT_STREAM=1),
        CUstream_flags=SimpleNamespace(CU_STREAM_NON_BLOCKING=1))
    resource = SimpleNamespace(sm=SimpleNamespace(smCount=128))
    group = SimpleNamespace(sm=SimpleNamespace(smCount=66))
    values = dict(cuDeviceGet=(0, 1), cuDeviceGetDevResource=(0, resource),
                  cuDevSmResourceSplitByCount=(0, [group], 1, None),
                  cuDevResourceGenerateDesc=(0, 'descriptor'), cuGreenCtxCreate=(0, 'green'),
                  cuGreenCtxStreamCreate=(0, 1234), cuStreamDestroy=(0,), cuGreenCtxDestroy=(0,))
    for name, result in values.items():
        def operation(*args, _name=name, _result=result):
            events.append(_name)
            return _result
        operation.__name__ = name
        setattr(driver, name, Mock(side_effect=operation, __name__=name))
    monkeypatch.setitem(sys.modules, 'cuda', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'cuda.bindings', SimpleNamespace(driver=driver))
    torch = MagicMock()
    torch.cuda.ExternalStream.return_value.synchronize.side_effect = lambda: events.append('sync')
    return driver, torch, events


def test_green_driver_rounding_and_owned_cleanup(cuda_driver):
    driver, torch, events = cuda_driver
    with _green_stream(torch, 1, 65) as stream:
        assert stream is torch.cuda.ExternalStream.return_value
    driver.cuDevSmResourceSplitByCount.assert_called_once()
    assert driver.cuDevSmResourceSplitByCount.call_args.args[-1] == 65
    torch.cuda.ExternalStream.assert_called_once_with(1234, device=1)
    assert events[-3:] == ['sync', 'cuStreamDestroy', 'cuGreenCtxDestroy']


def test_stream_creation_failure_destroys_green_context(cuda_driver):
    driver, torch, events = cuda_driver
    driver.cuGreenCtxStreamCreate.side_effect = RuntimeError('stream failed')
    with pytest.raises(RuntimeError, match='stream failed'):
        with _green_stream(torch, 1, 65):
            pass
    assert events[-1] == 'cuGreenCtxDestroy'
    driver.cuStreamDestroy.assert_not_called()


def test_green_synchronization_failure_still_destroys_handles(cuda_driver):
    driver, torch, events = cuda_driver
    torch.cuda.ExternalStream.return_value.synchronize.side_effect = RuntimeError('sync failed')
    with pytest.raises(RuntimeError, match='sync failed'):
        with _green_stream(torch, 1, 65):
            pass
    assert events[-2:] == ['cuStreamDestroy', 'cuGreenCtxDestroy']


def test_unsupported_nvml_telemetry_returns_nan(monkeypatch):
    import math
    from robort.profile.environment import power_draw_w

    class NVMLError(Exception):
        pass

    monkeypatch.setitem(sys.modules, 'pynvml', SimpleNamespace(NVMLError=NVMLError))
    torch = SimpleNamespace(cuda=SimpleNamespace(power_draw=Mock(side_effect=NVMLError('unsupported'))))
    assert math.isnan(power_draw_w(torch, 0))
    torch.cuda.power_draw.side_effect = ValueError('programming error')
    with pytest.raises(ValueError, match='programming error'):
        power_draw_w(torch, 0)
