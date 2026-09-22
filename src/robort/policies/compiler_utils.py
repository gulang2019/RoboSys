"""Compile inference operators and dispatch fixed-shape CUDA graph replays.

Inputs may contain dicts, lists, tuples, dataclasses, scalar constants, and
initialized Transformers StaticCache instances. Dynamic caches are unsupported.
All non-scalar tensors use batch dimension
zero; a compile example has batch size one. Scalar tensors retain their shape.
Operators must be capture-safe and must not depend on Python side effects or
aliasing between input leaves. Replay copies every tensor, including KV caches.
Operators may mutate tensor contents but must not replace input fields or leaves.
Outputs reuse storage: clone them before a subsequent call if needed. Dispatchers
are inference-only and not thread-safe. Keep the compiler open while using them.
"""

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Callable
import warnings

import torch
from transformers.cache_utils import StaticCache

from .vla_base import VLABasePolicy


@dataclass(frozen=True)
class TreeSpec:
    kind: str
    metadata: tuple
    children: tuple = ()
    constructor: Callable | None = field(default=None, compare=False, repr=False)

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        if self.kind != other.kind or self.metadata != other.metadata or \
            self.children != other.children:
            return False
        return True 


def flatten_tree(tree):
    """Return leaves and comparable structure without modifying the input tree."""
    leaves = []

    def visit(node):
        if isinstance(node, torch.Tensor):
            leaves.append(node)
            return TreeSpec(kind = 'tensor', 
                            metadata = (tuple(node.shape), node.dtype),
                            children = ())
        if node is None or isinstance(node, (int, float, bool, str)):
            leaves.append(node)
            return TreeSpec(kind = 'constant',
                            metadata = (type(node), node),
                            children = ())
        if isinstance(node, dict):
            return TreeSpec(kind = 'dict',
                            metadata = tuple(node.keys()),
                            children =  tuple(visit(v) for v in node.values()),
                            constructor= lambda metadata, children: {k:v for k, v in zip(metadata, children)})
        if isinstance(node, (list, tuple)):
            return TreeSpec(kind = 'sequence',
                            metadata = type(node),
                            children =  tuple(visit(v) for v in node),
                            constructor= lambda metadata, children: metadata(*children) if hasattr(metadata, '_fields') else metadata(children))
        if is_dataclass(node) and not isinstance(node, type):
            names = tuple(f.name for f in fields(node))

            # Avoid the mess-up from post-init 
            def rebuild_dataclass(metadata, children):
                result = object.__new__(metadata[0])
                for name, value in zip(metadata[1], children):
                    object.__setattr__(result, name, value)
                return result

            return TreeSpec(kind = 'dataclass', 
                            metadata = (type(node), names),
                            children = tuple(visit(getattr(node, name)) for name in names),
                            constructor = rebuild_dataclass)
        if isinstance(node, StaticCache):
            assert hasattr(node, 'key_cache')
            assert hasattr(node, 'value_cache')
            # Snapshot the non-buffer attributes; tensor structure is recorded
            # by the children. Neither tensors nor the callable enter metadata.
            metadata = (type(node), tuple(
                (name, value) for name, value in sorted(vars(node).items())
                if name not in ('key_cache', 'value_cache')
            ))

            def rebuild_cache(metadata, children):
                cache_type, attributes = metadata
                # The supplied tensors are already allocated. Do not invoke
                # StaticCache.__init__, which would allocate new KV buffers.
                cache = object.__new__(cache_type)
                vars(cache).update(attributes)
                cache.key_cache, cache.value_cache = children
                if cache.key_cache:
                    cache.max_batch_size = cache.key_cache[0].shape[0]
                return cache

            return TreeSpec(kind = 'cache', 
                            metadata = metadata,
                            children = (visit(node.key_cache), visit(node.value_cache)),
                            constructor = rebuild_cache)
        raise TypeError(f'Unsupported input tree type: {type(node).__name__}')

    spec = visit(tree)
    return leaves, spec


def reconstruct(leaves, spec):
    """Rebuild a tree with new containers/cache objects and the supplied leaves."""
    iterator = iter(leaves)

    def build(node: TreeSpec):
        if node.kind in ('tensor', 'constant'):
            try:
                return next(iterator)
            except StopIteration as error:
                raise ValueError('Too few leaves for tree specification') from error
        children = [build(x) for x in node.children]
        return node.constructor(metadata = node.metadata, children = children)

    result = build(spec)
    sentinel = object()
    if next(iterator, sentinel) is not sentinel:
        raise ValueError('Too many leaves for tree specification')
    return result

def check(stage, actual, expected):
    actual_leaves, actual_spec = flatten_tree(actual)
    expected_leaves, expected_spec = flatten_tree(expected)
    assert actual_spec == expected_spec, f"{stage}: output structure differs"
    for actual_tensor, expected_tensor in zip(actual_leaves, expected_leaves):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-2, atol=2e-2)
    print(f"{stage}: compiled CUDA graph matches eager", flush=True)


@dataclass
class Program:
    buffer: list[torch.Tensor]
    output: Any
    graph: torch.cuda.CUDAGraph | None
    stream: torch.cuda.Stream
    input: Any = None


@dataclass
class ProgramDispatcher:
    device: torch.device
    spec: TreeSpec
    programs: dict[tuple[float, int], Program]
    function: Any  # Original callable, also used to prepare eager profiling inputs.

    @torch.inference_mode()
    def __call__(self, input):
        if not self.programs:
            raise RuntimeError('Compiled program has been closed')
        leaves, spec = flatten_tree(input)
        tensors = [t for t in leaves if isinstance(t, torch.Tensor)]
        batched = [t for t in tensors if t.ndim]
        if not batched:
            raise ValueError('At least one batched tensor is required')
        batch_size = batched[0].shape[0]
        if any(t.shape[0] != batch_size for t in batched):
            raise ValueError('All non-scalar tensors must have the same batch size')

        def for_batch(node):
            metadata = node.metadata
            if node.kind == 'tensor' and metadata[0]:
                metadata = ((batch_size, *metadata[0][1:]), metadata[1])
            elif node.kind == 'cache':
                metadata = (metadata[0], tuple(
                    (name, batch_size if name == 'max_batch_size' else value)
                    for name, value in metadata[1]))
            return replace(node, metadata=metadata, children=tuple(for_batch(c) for c in node.children))
        stream = torch.cuda.current_stream(self.device)
        if spec != for_batch(self.spec):
            raise ValueError('Input tree structure/constants, shape and dtype must match the compiled inputs')
        try:
            program = self.programs[(stream.cuda_stream, batch_size)]
        except KeyError as error:
            raise ValueError(f'No program for SM partition {stream.cuda_stream}, batch size {batch_size}') from error
        for source, target in zip(tensors, program.buffer):
            if source.shape != target.shape or source.dtype != target.dtype or source.device != target.device or source.layout != target.layout:
                raise ValueError('Input shape and dtype, device and layout must match the compiled inputs')
        with torch.cuda.device(program.stream.device):
            with torch.cuda.stream(program.stream):
                for source, target in zip(tensors, program.buffer):
                    target.copy_(source)
                program.graph.replay()
        return program.output

    def close(self):
        for program in self.programs.values():
            program.stream.synchronize()
            if program.graph is not None:
                program.graph.reset()
        self.programs.clear()
        self.function = None


class Compiler:
    """Compile each batch size, then warm up and capture on each stream.

    PyTorch does not expose green-context allocation in supported versions.
    Supply caller-owned CUDA streams; close programs before destroying them.
    Each program allows at most 0.5% mismatches per tensor (rtol=atol=0.02).
    A mismatch warns and captures the original function instead.
    use_torch_compile=False captures the original function directly.
    Graphs share a pool and execute serially. Outputs are allocated outside
    capture, so pool storage is temporary and variants may replay in any order.
    """

    def __init__(self, device=None, streams=None, use_torch_compile=True):
        self.device = torch.device(device if device is not None else 'cuda')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.streams = list(streams) if streams is not None else [torch.cuda.Stream(device=self.device)]
        self.use_torch_compile = use_torch_compile
        self._dispatchers = []
        self._pool = torch.cuda.graph_pool_handle()

    @torch.inference_mode()
    def _compile(self, func, batch_one_input, batch_sizes=(1,)) -> ProgramDispatcher:
        if not self.streams:
            raise RuntimeError('Compiler has been closed')
        sizes = tuple(batch_sizes)
        if not sizes or any(type(b) is not int or b < 1 for b in sizes) or len(set(sizes)) != len(sizes):
            raise ValueError('batch_sizes must contain unique positive integers')
        leaves, spec = flatten_tree(batch_one_input)
        tensors = [t for t in leaves if isinstance(t, torch.Tensor)]
        if not any(t.ndim for t in tensors):
            raise ValueError('At least one batched tensor is required')
        for tensor in tensors:
            if tensor.device != self.device or tensor.layout != torch.strided:
                raise ValueError(f'Inputs must be strided CUDA tensors on {self.device}')
            if tensor.ndim and tensor.shape[0] != 1:
                raise ValueError('Compile example tensors must have batch size one')
        
        compiled = func
        if self.use_torch_compile:
            # Batch shapes and guarded policy state can require multiple variants.
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 64, 4 * len(sizes) * len(self.streams))
            torch._dynamo.config.accumulated_recompile_limit = max(
                torch._dynamo.config.accumulated_recompile_limit, torch._dynamo.config.recompile_limit)
            compiled = torch.compile(func, dynamic=True, options={'triton.cudagraphs': False})
        dispatcher = ProgramDispatcher(self.device, spec, {}, func)
        max_size = max(sizes)
        shared_buffers = [
            torch.empty(
                (max_size, *t.shape[1:]) if t.ndim else (),
                dtype=t.dtype,
                device=t.device,
            )
            for t in tensors
        ]

        try:
            with torch.cuda.device(self.device):
                for stream in self.streams:
                    for size in sizes:
                        sources = [t.expand(size, *t.shape[1:]) if t.ndim else t for t in tensors]
                        buffers = [
                            buffer[:size] if tensor.ndim else buffer
                            for tensor, buffer in zip(tensors, shared_buffers)
                        ]
                        for target, source in zip(buffers, sources):
                            target.copy_(source)
                        buffer_iter = iter(buffers)
                        static_input = reconstruct(
                            [next(buffer_iter) if isinstance(t, torch.Tensor) else t for t in leaves], spec)
                        run = compiled
                        if self.use_torch_compile:
                            # Snapshot before compiled execution can overwrite aliased
                            # outputs (including the input KV cache).
                            with torch.random.fork_rng(devices=[self.device.index]):
                                expected, expected_spec = flatten_tree(func(static_input))
                                expected = [t.clone() if isinstance(t, torch.Tensor) else t for t in expected]
                            for target, source in zip(buffers, sources):
                                target.copy_(source)
                            with torch.random.fork_rng(devices=[self.device.index]):
                                actual, actual_spec = flatten_tree(compiled(static_input))
                            try:
                                if actual_spec != expected_spec:
                                    raise AssertionError('Output structure, shape, dtype or constants differ')
                                for value, reference in zip(actual, expected):
                                    if isinstance(value, torch.Tensor):
                                        mismatches = (~torch.isclose(value, reference, rtol=2e-2, atol=2e-2)).sum().item()
                                        if mismatches > reference.numel() * 0.005:
                                            raise AssertionError(f'{mismatches}/{reference.numel()} elements exceed tolerance (limit 0.5%)')
                            except AssertionError as error:
                                warnings.warn(
                                    f'{getattr(func, "__qualname__", type(func).__name__)}: '
                                    f'compiled output differs from eager (rtol=atol=0.02, '
                                    f'batch={size}, Stream={stream.cuda_stream}); '
                                    f'falling back to CUDA graph capture of the original function. {error}',
                                    RuntimeWarning, stacklevel=2,
                                )
                                run = func
                            del actual, actual_spec, expected, expected_spec
                        
                        with torch.cuda.stream(stream):
                            stream.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream, pool=self._pool):
                                output = run(static_input)
                            program = Program(buffers, output, graph, stream)
                            dispatcher.programs[(stream.cuda_stream, size)] = program
        except BaseException:
            dispatcher.close()
            raise
        self._dispatchers.append(dispatcher)
        return dispatcher

    @torch.inference_mode()
    def compile(self, policy: VLABasePolicy) -> VLABasePolicy:
        _, observation, embedding, decode_input, _, _ = policy.make_example_input(1)
        for name, example in (('embed', observation), ('encode', embedding), ('decode', decode_input)):
            setattr(policy, name, self._compile(getattr(policy, name), example, policy.config.batch_sizes[name]))
        return policy

    def close(self):
        for dispatcher in self._dispatchers:
            dispatcher.close()
        self._dispatchers.clear()
        self.streams.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
