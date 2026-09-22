"""Configuration shared by inference and profiling backends."""

from dataclasses import dataclass


@dataclass
class PolicyConfig:
    model_name: str = "pi05_libero"
    backend: str = "openpi"
    num_views: int = 2
    image_resolution: str = "224"
    precision: str = "fp16"
    prompt_len: int = 200
    model_dir: str | None = None
    num_steps: int = 10
    chunk_size: int = 10
    use_cuda_graph: bool = True
    batch_sizes: dict[str, list[int]] | None = None
    # Compatibility with the existing OpenPI inference configuration.
    num_sample_steps: int | None = None
    use_torch_compile: bool = True
