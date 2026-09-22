from dataclasses import dataclass, field
import math

from ..policies import PolicyConfig


# TODO: the profile should be maintained in a tree structure.

@dataclass
class RunnerConfig:
    num_warmup: int = 3
    num_iter: int = 20
    output_dir: str = "profile"

@dataclass
class HardwareConfig:
    """Configuration bound to the current CUDA device at construction time."""

    power_perc: float # (0, 1] fraction of the default NVML power limit
    sm_perc: float # (0, 1] requested fraction of physical SMs
    hardware_name: str = field(init=False)
    num_sms: int = field(init=False) # Physical count; green allocations round up.
    mem_cap_gb: float = field(init=False) # Decimal GB
    max_power_w: float = field(init=False) # NVML default limit, in Watts

    def __post_init__(self):
        from robort.profile.environment import nvml_device_handle, validate_hardware_config

        validate_hardware_config(self)
        import torch
        import pynvml

        if not torch.cuda.is_available():
            raise RuntimeError("Hardware profiling requires a CUDA device")
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        self._device_index = device
        self.hardware_name = properties.name
        self.num_sms = properties.multi_processor_count
        self.mem_cap_gb = properties.total_memory / 1e9
        pynvml.nvmlInit()
        try:
            handle = nvml_device_handle(torch, pynvml, device)
            try:
                self.max_power_w = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000
            except Exception as error:
                unsupported = getattr(pynvml, "NVMLError_NotSupported", None)
                if not isinstance(unsupported, type) or not isinstance(error, unsupported):
                    raise
                # Jetson AGX Thor exposes instantaneous power telemetry but
                # not NVML's desktop/server power-limit control interface.
                self.max_power_w = math.nan
        finally:
            pynvml.nvmlShutdown()


# @dataclass
# class PolicyConfig:
#     model_name: str
#     backend: str # flash_rt, openpi
#     num_views: int
#     image_resolution: str # e.g. 224, 384, 512, etc.
#     batch_size: int
#     precision: str # fp16, bf16, int8, etc.
#     prompt_len: int
#     model_dir: str | None = None  # Defaults to checkpoints/{model_name}_pytorch.
#     num_steps: int = 10 # number of sampling steps
#     chunk_size: int = 10 # number of action chunk size
#     use_cuda_graph: bool = True  # Capture/replay each FlashRT profiling stage.

@dataclass
class HardwareProfile:
    hardware_config: HardwareConfig
    idle_power_w: float # in Watts
    mem_bw_gbs: float # in GB/s

    gflops_fp32: float # in GFLOPS
    gflops_fp16: float # in GFLOPS
    gflops_bf16: float # in GFLOPS
    gflops_int8: float # in GOPS (integer operations)
    gflops_fp8: float # in GFLOPS

    c2g_bw_gbs: float # in GB/s
    g2c_bw_gbs: float # in GB/s
    g2g_bw_gbs: float | None # in GB/s

@dataclass
class StageProfile:
    """Stage metadata plus wall latency (ms) and device energy (J) statistics.

    Standard deviations use the population convention. Unmeasured statistics
    and unavailable memory measurements are NaN.
    """
    num_params: int = None 
    flops: int = None 
    mem_fp_weight_gb: float = None 
    # Peak additional Torch CUDA allocation per call (decimal GB); excludes
    # preallocated inputs/caches/graph pools and non-Torch native allocations.
    mem_fp_activation_gb: dict[int, tuple[float, float]] = field(default_factory = dict)
    lat: dict[int, tuple[float, float]] = field(default_factory = dict) # batch size -> latency 
    energy: dict[int, tuple[float, float]] = field(default_factory = dict) # batch size -> latency

@dataclass
class PolicyProfile:
    hardware_config: HardwareConfig
    policy_config: PolicyConfig
    stages: dict[str, StageProfile]
