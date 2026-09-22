"""Owned CUDA green streams and reversible, device-wide NVML power limits."""

from contextlib import ExitStack, contextmanager
import logging
import math
from numbers import Real
from threading import RLock

_LOG = logging.getLogger(__name__)
_ENV_LOCK = RLock()


def validate_hardware_config(config):
    for name in ("power_perc", "sm_perc"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be a finite fraction in (0, 1]")


def nvml_device_handle(torch, nvml, device):
    """CUDA ordinals may be reordered by CUDA_VISIBLE_DEVICES; use GPU UUIDs."""
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    if not uuid.startswith(("GPU-", "MIG-")):
        uuid = f"GPU-{uuid}"
    return nvml.nvmlDeviceGetHandleByUUID(uuid)


def power_draw_w(torch, device):
    """Keep latency profiling usable when GPU power telemetry is unsupported."""
    try:
        return float(torch.cuda.power_draw(device)) / 1000
    except (AttributeError, ImportError, RuntimeError):
        return float("nan")
    except Exception as error:
        import pynvml

        if isinstance(error, pynvml.NVMLError):
            return float("nan")
        raise


def _cuda_call(function, *args):
    status, *values = function(*args)
    if int(status) != 0:
        raise RuntimeError(f"{function.__name__} failed: {status}")
    if len(values) == 1:
        return values[0]
    return tuple(values)


@contextmanager
def _green_stream(torch, device, requested_sms):
    # Own the handles directly: external Torch streams do not destroy CUDA
    # streams, and cuGreenCtxDestroy does not destroy its streams either.
    try:
        from cuda.bindings import driver
    except ImportError as error:
        raise RuntimeError("Green contexts require cuda-python with cuda.bindings.driver "
                           "and a CUDA 12.6+ compatible NVIDIA driver") from error
    with ExitStack() as cleanup:
        torch.cuda.init()
        cu_device = _cuda_call(driver.cuDeviceGet, device)
        kind = driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        resource = _cuda_call(driver.cuDeviceGetDevResource, cu_device, kind)
        # Let the driver apply architecture-specific minimums and alignment.
        groups, count, _ = _cuda_call(driver.cuDevSmResourceSplitByCount,
                                      1, resource, 0, requested_sms)
        if count != 1:
            raise RuntimeError(f"Cannot allocate a green context with {requested_sms} SMs")
        actual_sms = groups[0].sm.smCount
        if actual_sms < requested_sms or actual_sms > resource.sm.smCount:
            raise RuntimeError(f"Unexpected green-context SM allocation: {actual_sms}")
        descriptor = _cuda_call(driver.cuDevResourceGenerateDesc, groups, 1)
        green = _cuda_call(driver.cuGreenCtxCreate, descriptor, cu_device,
                          driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM)
        cleanup.callback(_cuda_call, driver.cuGreenCtxDestroy, green)
        raw_stream = _cuda_call(driver.cuGreenCtxStreamCreate, green,
                               driver.CUstream_flags.CU_STREAM_NON_BLOCKING, 0)
        cleanup.callback(_cuda_call, driver.cuStreamDestroy, raw_stream)
        stream = torch.cuda.ExternalStream(int(raw_stream), device=device)
        cleanup.callback(stream.synchronize)
        _LOG.info("Green context on CUDA %s: requested %s SMs, allocated %s",
                  device, requested_sms, actual_sms)
        yield stream


@contextmanager
def prepare_hardware_env(hardware_config, stream=None):
    """Select a green stream, apply power_perc * max_power_w, then restore both.

    max_power_w is the NVML default (rated) power limit, not an overclock limit.
    Fractions must be positive; out-of-range power limits fail without clamping.
    SM counts round upward according to the CUDA driver's allocation rules.
    The yielded stream is current for Torch operations. Explicit backend launches
    and CUDA graph capture/replay must use it.
    All work must finish before exit. A supplied stream remains caller-owned;
    otherwise the temporary stream and green context are destroyed on exit.

    Power control requires NVML permissions. Errors propagate, including failures
    to restore power. Calls are serialized within this process; other processes
    can still alter the device-wide limit or compete for GPU resources.
    """
    validate_hardware_config(hardware_config)
    import torch
    import pynvml

    if not torch.cuda.is_available():
        raise RuntimeError("Hardware profiling requires a CUDA device")
    device = hardware_config._device_index
    with _ENV_LOCK, ExitStack() as cleanup:
        cleanup.enter_context(torch.cuda.device(device))
        pynvml.nvmlInit()
        cleanup.callback(pynvml.nvmlShutdown)
        handle = nvml_device_handle(torch, pynvml, device)
        power_control = math.isfinite(hardware_config.max_power_w)
        if not power_control and hardware_config.power_perc != 1:
            raise NotImplementedError(
                "This GPU does not support NVML power-limit control; use power_perc=1.0"
            )
        if power_control:
            minimum, maximum = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            requested_mw = math.ceil(
                hardware_config.power_perc * hardware_config.max_power_w * 1000)
            if not minimum <= requested_mw <= maximum:
                raise ValueError(f"Requested power limit {requested_mw / 1000:g} W is outside "
                                 f"NVML range [{minimum / 1000:g}, {maximum / 1000:g}] W")
            previous_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        # Initialize a working green context before modifying device-wide state.
        torch.cuda.synchronize(device)
        if stream is None:
            stream = cleanup.enter_context(_green_stream(
                torch, device, math.ceil(hardware_config.num_sms * hardware_config.sm_perc)))
        elif stream.device != torch.device("cuda", device):
            raise ValueError("Supplied stream must belong to the hardware device")
        if power_control and requested_mw != previous_mw:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, requested_mw)
            cleanup.callback(pynvml.nvmlDeviceSetPowerManagementLimit, handle, previous_mw)
            actual_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            if actual_mw != requested_mw:
                raise RuntimeError(f"NVML applied {actual_mw} mW instead of {requested_mw} mW")
        cleanup.enter_context(torch.cuda.stream(stream))
        # Run this before restoring power, including on a failing model call.
        cleanup.callback(stream.synchronize)
        yield stream
