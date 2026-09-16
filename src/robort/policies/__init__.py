"""Decomposed VLA policies; backend dependencies load only when requested."""

from .config import PolicyConfig

__all__ = ["PolicyConfig", "VLABasePolicy", "OpenPIPolicy", "CudaGraphOpenPIPolicy", "FlashRTPolicy", "create_policy"]


def __getattr__(name):
    if name == "OpenPIPolicy":
        from .openpi import OpenPIPolicy
        return OpenPIPolicy
    if name == "CudaGraphOpenPIPolicy":
        from .openpi_cuda_graph import CudaGraphOpenPIPolicy
        return CudaGraphOpenPIPolicy
    if name == "FlashRTPolicy":
        from .flash_rt import FlashRTPolicy
        return FlashRTPolicy
    if name == "VLABasePolicy":
        from .vla_base import VLABasePolicy
        return VLABasePolicy
    raise AttributeError(name)


def create_policy(policy_config: PolicyConfig, device: str | None = None):
    if policy_config.backend in ("flash_rt", "flashrt"):
        from .flash_rt import FlashRTPolicy
        return FlashRTPolicy(policy_config, device)
    if policy_config.backend == "openpi":
        from .openpi import OpenPIPolicy
        return OpenPIPolicy(policy_config, device)
    raise ValueError(f"Unknown policy backend: {policy_config.backend!r}")
