"""Local and websocket clients for the LIBERO evaluator."""

import ast
import inspect
import json

from robort.client import WebsocketClientPolicy
from robort.policies import PolicyConfig, create_policy
from robort.schemas import InferenceRequest, InferenceResponse


class LocalClientPolicy:
    """Adapt the decomposed OpenPI batch interface to a single request."""

    def __init__(
        self,
        model_dir: str = "checkpoints/pi05_libero_pytorch",
        model_name: str = "pi05_libero",
        device: str | None = None,
        num_steps: int = 10,
        encode_keep_rate: float | None = None,
        debug_dir: str | None = None,
    ):
        self.policy = create_policy(
            PolicyConfig(
                backend="openpi", model_name=model_name, model_dir=model_dir,
                num_steps=num_steps, use_cuda_graph=False,
                encode_keep_rate=encode_keep_rate,
            ),
            device=device,
        )
        self.debug_dir = debug_dir
        self._episode = 0

    def configure_run(self, *, num_steps, encode_keep_rate, debug_dir):
        """Reuse weights while starting a fresh sweep configuration."""
        self.policy.export_debug_videos()
        self.policy.config.num_steps = num_steps
        self.policy.config.num_sample_steps = None
        self.policy.config.encode_keep_rate = encode_keep_rate
        self.policy.reset()
        self.debug_dir = debug_dir
        self._episode = 0

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if self.policy is None:
            raise RuntimeError("Local client is closed")
        if request.inference_type != "sync":
            raise ValueError("Local OpenPI client only supports sync requests; disable RTC")
        return self.policy.infer([request])[0]

    def reset(self) -> None:
        if self.debug_dir is None:
            self.policy.reset()
        else:
            from pathlib import Path

            self.policy.start_debug_episode(Path(self.debug_dir) / f'episode_{self._episode:03d}')
            self._episode += 1

    def close(self) -> None:
        if self.policy is not None:
            try:
                self.policy.export_debug_videos()
            finally:
                self.policy = None


def parse_client_args(value: str) -> tuple[str, dict]:
    """Parse a JSON object or Python dictionary literal without executing code."""
    try:
        options = json.loads(value)
    except json.JSONDecodeError:
        try:
            options = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("--client-args must be a JSON object or Python dictionary literal") from exc
    if not isinstance(options, dict) or not all(isinstance(key, str) for key in options):
        raise ValueError("--client-args must be a dictionary with string keys")
    client_type = options.pop("type", "websocket")
    if client_type not in ("local", "websocket"):
        raise ValueError("Client type must be 'local' or 'websocket'")
    client_class = LocalClientPolicy if client_type == "local" else WebsocketClientPolicy
    try:
        inspect.signature(client_class).bind(**options)
    except TypeError as exc:
        raise ValueError(f"Invalid {client_type} client arguments: {exc}") from exc
    return client_type, options
