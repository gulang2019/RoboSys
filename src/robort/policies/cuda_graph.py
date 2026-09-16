"""CUDA graph capture for inference functions with Torch tensor input trees."""

import torch
from torch.utils import _pytree


def _flatten_inputs(inputs, device):
    if not isinstance(inputs, dict):
        raise TypeError("inputs must be a dictionary of keyword arguments")
    leaves, spec = _pytree.tree_flatten(inputs)
    for tensor in leaves:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("all input tree leaves must be Torch tensors")
        if tensor.device != device or tensor.layout != torch.strided:
            raise ValueError(f"inputs must be strided CUDA tensors on {device}")
    return leaves, spec


class _CapturedTorch:
    def __init__(self, fn, stream, graph, inputs, spec, outputs):
        # Keep the callable alive too: it may own captured weights/buffers.
        self.fn, self.stream, self.graph = fn, stream, graph
        self.inputs, self.spec, self.outputs = inputs, spec, outputs

    @torch.inference_mode()
    def __call__(self, **kwargs):
        if self.graph is None:
            raise RuntimeError("Captured function has been closed")
        incoming, spec = _flatten_inputs(kwargs, self.stream.device)
        if spec != self.spec:
            raise ValueError("Input tree structure must match the captured inputs")
        # Validate everything before modifying any captured input storage.
        for source, target in zip(incoming, self.inputs):
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError("Input shape and dtype must match the captured inputs")
        with torch.cuda.device(self.stream.device):
            caller = torch.cuda.current_stream()
            self.stream.wait_stream(caller)
            with torch.cuda.stream(self.stream):
                for source, target in zip(incoming, self.inputs):
                    target.copy_(source)
                    # Protect sources that are freed before the copy finishes.
                    source.record_stream(self.stream)
                self.graph.replay()
            # Outputs can immediately be consumed on the calling stream.
            caller.wait_stream(self.stream)
        return self.outputs

    def close(self):
        """Release graph/storage before the supplied stream is destroyed."""
        if self.graph is not None:
            try:
                self.stream.synchronize()
            finally:
                self.graph.reset()
                self.graph = None
                self.inputs = self.outputs = self.fn = None
                self.stream = None


def capture_torch(fn, example_inputs, stream, *, warmups=3):
    """Return an inference-only callable replaying ``fn(**example_inputs)``.

    Inputs are a keyword dictionary containing tensor leaves and PyTorch pytree
    containers (dict/list/tuple, etc.). Every tensor must be on stream.device.
    Shapes, dtypes and tree structure are fixed; source strides may vary because
    replay copies into persistent storage. Inputs must not rely on aliasing
    between different leaves: each leaf gets its own allocation.

    ``fn`` must be capture-safe and launch all GPU work on the current stream.
    Python side effects occur during warmup/capture, not replay. Outputs reuse
    captured storage and are overwritten on later calls; clone to retain them.
    The returned callable is not thread-safe and must be closed before destroying
    an externally owned stream. No autograd graph is built.
    """
    if type(warmups) is not int or warmups < 1:
        raise ValueError("warmups must be a positive integer")
    leaves, spec = _flatten_inputs(example_inputs, stream.device)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.device(stream.device), torch.inference_mode():
            caller = torch.cuda.current_stream()
            stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                static = [tensor.clone() for tensor in leaves]
                inputs = _pytree.tree_unflatten(static, spec)
                for tensor in leaves:
                    tensor.record_stream(stream)
                for _ in range(warmups):
                    # Functions may mutate inputs; every warmup starts fresh.
                    for target, source in zip(static, leaves):
                        target.copy_(source)
                    fn(**inputs)
                for target, source in zip(static, leaves):
                    target.copy_(source)
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    outputs = fn(**inputs)
            caller.wait_stream(stream)
        return _CapturedTorch(fn, stream, graph, static, spec, outputs)
    except BaseException:
        graph.reset()
        raise
