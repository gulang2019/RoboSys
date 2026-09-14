import numpy as np
from dataclasses import dataclass, field, asdict

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
class PolicyConfig:
    model_name: str = "pi05_libero"
    model_dir: str = "checkpoints/pi05_libero"
    env_name: str = "libero_20"
    max_batch_size: int = 10
    batch_sizes: list[int] = field(default_factory=list)
    num_sample_steps: int = 10
    action_horizon: int = 10

@dataclass
class ServerConfig:
    host_addr: str = "127.0.0.1"
    port: int = 8000
    timeout: float = 1.0
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)

    @property
    def max_batch_size(self):
        return self.policy_config.max_batch_size
