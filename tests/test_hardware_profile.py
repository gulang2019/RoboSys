from contextlib import contextmanager
import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import unittest

from robort.profile.hardware import profile_hardware
from robort.profile.schemas import HardwareConfig


class HardwareProfileTests(unittest.TestCase):
    def setUp(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.cuda.current_device.return_value = 1
        torch.cuda.device_count.return_value = 2
        torch.cuda.can_device_access_peer.return_value = False
        torch.cuda.mem_get_info.return_value = (1024**3, 2 * 10**9)
        torch.cuda.get_device_properties.return_value = SimpleNamespace(
            name="Test GPU", multi_processor_count=80, total_memory=2 * 10**9, uuid="test-uuid",
        )
        torch.cuda.power_draw.side_effect = [10_000, 20_000, 30_000, 25_000] + [40_000] * 10
        torch.cuda.get_device_capability.return_value = (8, 9)
        torch.cuda.is_bf16_supported.return_value = True
        torch.backends.cuda.matmul.allow_tf32 = True
        self.torch = torch
        nvml = MagicMock()
        nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 450_000
        modules = patch.dict(sys.modules, torch=torch, pynvml=nvml)
        self.stream = MagicMock()
        self.active = False

        @contextmanager
        def environment(config):
            self.active = True
            try:
                yield self.stream
            finally:
                self.active = False

        env_patch = patch("robort.profile.hardware.prepare_hardware_env", environment)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        torch.mm.side_effect = lambda *args, **kwargs: self.assertTrue(self.active)

        clock = patch("robort.profile.hardware.perf_counter", side_effect=range(20))
        modules.start()
        clock.start()
        self.addCleanup(modules.stop)
        self.addCleanup(clock.stop)

    def test_all_fields_and_units(self):
        config = HardwareConfig(1, 1.0)
        result = profile_hardware(config)
        assert result.hardware_config is config
        assert result.hardware_config.hardware_name == "Test GPU"
        assert result.hardware_config.num_sms == 80
        assert result.hardware_config.mem_cap_gb == 2
        assert result.idle_power_w == 10
        assert result.hardware_config.max_power_w == 450
        size = 64 * 1024**2
        self.assertAlmostEqual(result.mem_bw_gbs, 2 * size / 0.1 / 1e9)
        self.assertAlmostEqual(result.c2g_bw_gbs, size / 0.1 / 1e9)
        self.assertAlmostEqual(result.g2c_bw_gbs, size / 0.1 / 1e9)
        for name in ("fp32", "fp16", "bf16", "int8", "fp8"):
            self.assertAlmostEqual(getattr(result, f"gflops_{name}"), 2 * 2048**3 / 0.1 / 1e9)
        assert result.g2g_bw_gbs is None
        assert self.stream.synchronize.call_count > 1
        assert self.active is False
        assert self.torch.mm.call_count == 39
        assert self.torch._int_mm.call_count == 13
        assert self.torch._scaled_mm.call_count == 13
        assert self.torch.backends.cuda.matmul.allow_tf32 is True
        self.torch.autocast.assert_called_once_with(device_type="cuda", enabled=False)
        assert [call.kwargs["dtype"] for call in self.torch.randn.call_args_list[:3]] == [
            self.torch.float32, self.torch.float16, self.torch.bfloat16]


    def test_peer_transfer(self):
        self.torch.cuda.can_device_access_peer.return_value = True
        result = profile_hardware(HardwareConfig(1, 1))
        self.assertAlmostEqual(result.g2g_bw_gbs, 64 * 1024**2 / 0.1 / 1e9)
        self.torch.cuda.can_device_access_peer.assert_called_once_with(1, 0)
        assert any(call.args == (0,) for call in self.torch.cuda.synchronize.call_args_list)


    def test_unsupported_power(self):
        self.torch.cuda.power_draw.side_effect = RuntimeError("Not supported")
        result = profile_hardware(HardwareConfig(1, 1))
        assert math.isnan(result.idle_power_w)
        assert result.gflops_fp32 > 0


    def test_no_cuda(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA"):
            profile_hardware(HardwareConfig(1, 1))


    def test_invalid_partition(self):
        for partition in [-0.1, 1.1, float("nan")]:
            with self.subTest(partition=partition), self.assertRaisesRegex(ValueError, "sm_perc"):
                profile_hardware(HardwareConfig(1, partition))


    def test_no_free_memory(self):
        self.torch.cuda.mem_get_info.return_value = (0, 2 * 10**9)
        with self.assertRaisesRegex(RuntimeError, "Insufficient free CUDA memory"):
            profile_hardware(HardwareConfig(1, 1))

    def test_unsupported_dtypes(self):
        self.torch.cuda.is_bf16_supported.return_value = False
        self.torch.cuda.get_device_capability.return_value = (8, 0)
        self.torch._int_mm.side_effect = RuntimeError("not supported")
        result = profile_hardware(HardwareConfig(1, 1))
        for name in ("bf16", "int8", "fp8"):
            assert math.isnan(getattr(result, f"gflops_{name}"))
        assert result.gflops_fp16 > 0
        self.torch._scaled_mm.assert_not_called()

    def test_compute_error_restores_tf32(self):
        self.torch._scaled_mm.side_effect = RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            profile_hardware(HardwareConfig(1, 1))
        assert self.torch.backends.cuda.matmul.allow_tf32 is True
