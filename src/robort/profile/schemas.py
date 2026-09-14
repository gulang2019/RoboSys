from dataclasses import dataclass

@dataclass
class RunnerConfig:
    num_warmup: int = 10
    num_iter: int = 100

@dataclass
class HardwareConfig:
    power_mode: str
    sm_partition: float # [0, 1] fraction of SMs to use

@dataclass
class PolicyConfig:
    model_name: str
    backend: str # flash_rt, openpi, etc.
    implementation: str # jax, torch, etc.
    num_views: int
    image_resolution: str # e.g. 224, 384, 512, etc.
    batch_size: int
    precision: str # fp16, bf16, int8, etc.
    prompt_len: int
    model_dir: str | None = None  # Defaults to checkpoints/{model_name}_pytorch.
    num_steps: int = 10
    chunk_size: int = 10

@dataclass
class HardwareProfile:
    hardware_config: HardwareConfig
    hardware_name: str
    num_sms: int
    max_power_w: float # in Watts
    idle_power_w: float # in Watts
    mem_bw_gbs: float # in GB/s
    gflops: float # in TFLOPS
    mem_cap_gb: float # in GB
    c2g_bw_gbs: float # in GB/s
    g2c_bw_gbs: float # in GB/s
    g2g_bw_gbs: float # in GB/s

@dataclass
class DistrProfile: 
    mean: float
    std: float
    p20: float 
    p50: float
    p80: float
    p99: float
    
@dataclass
class PolicyProfile:
    num_params: int
    flops_per_inference: int 
    mem_fp_model_weight_gb: float
    mem_fp_activation_gb: float
    stages: dict[str, DistrProfile]
    
