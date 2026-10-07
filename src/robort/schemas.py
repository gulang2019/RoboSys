import numpy as np
from dataclasses import dataclass, field, asdict

STAGES = ('preprocess', 'embed', 'encode', 'decode', 'postprocess')

@dataclass
class PolicyConfig:
    model_name: str = "pi05_libero"
    backend: str = "openpi"
    num_views: int = 2
    image_resolution: str = "224"
    precision: str = "fp16"
    prompt_len: int = 20
    model_dir: str | None = None
    num_steps: int = 10
    chunk_size: int = 10
    use_cuda_graph: bool = True
    batch_sizes: dict[str, list[int]] | None = None
    sm_partitions: dict[str, float] | None = None
    # Compatibility with the existing OpenPI inference configuration.
    num_sample_steps: int | None = None
    use_torch_compile: bool = True
    # Sparsity config
    encode_keep_rate: float | None = None


@dataclass
class InferenceRequest:
    observation: dict
    inference_type: str = "sync" # rtc/vlash/sync
    previous_action: np.ndarray | None = None
    rtc_s_param: int | None = None
    rtc_d_param: int | None = None
    max_execution_horizon: int = 0

    def to_dict(self)->dict:
        return asdict(self)

@dataclass
class InferenceResponse:
    actions: np.ndarray
    # Model-space actions used for RTC conditioning, before output transforms.
    rtc_prev_actions: np.ndarray | None = None



@dataclass
class ServerConfig:
    host_addr: str = "127.0.0.1"
    port: int = 8000
    timeout: float = 1.0
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)
