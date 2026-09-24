"""Tree handling and real compiled CUDA graph replay regressions."""
from collections import namedtuple
from dataclasses import dataclass, field

import pytest

torch = pytest.importorskip('torch')
from robort.policies.compiler_utils import Compiler, flatten_tree, reconstruct


@dataclass
class Input:
    x: object
    optional: object = None


def test_tree_roundtrip_and_structure():
    tree = Input({'x': [1, (2, 3)], 'empty': [], 'dict': {}, 'tuple': ()})
    leaves, spec = flatten_tree(tree)
    assert leaves == [1, 2, 3, None]
    assert reconstruct(leaves, spec) == tree
    assert reconstruct(leaves, spec) is not tree
    assert flatten_tree(tree)[1] == spec
    assert flatten_tree(Input({'other': [1, 2, 3]}))[1] != spec
    with pytest.raises(ValueError, match='Too few'):
        reconstruct(leaves[:-1], spec)
    with pytest.raises(ValueError, match='Too many'):
        reconstruct(leaves + [4], spec)


def test_namedtuple_and_frozen_dataclass_non_init_fields():
    Pair = namedtuple('Pair', 'tensor label')

    @dataclass(frozen=True, slots=True)
    class Record:
        pair: object
        derived: int = field(init=False, default=7)

    source = Record(Pair(torch.ones(1, 2), 'input'))
    leaves, spec = flatten_tree(source)
    replacement = torch.zeros(1, 2)
    rebuilt = reconstruct([replacement, *leaves[1:]], spec)
    assert isinstance(rebuilt.pair, Pair)
    assert rebuilt.pair.tensor is replacement
    assert rebuilt.pair.label == 'input'
    assert rebuilt.derived == 7
    assert source.pair.tensor is not replacement


def test_cache_roundtrip_does_not_modify_source():
    transformers = pytest.importorskip('transformers')
    cache = transformers.StaticCache(
        transformers.LlamaConfig(num_hidden_layers=1, num_attention_heads=2,
                                 hidden_size=8, max_position_embeddings=4),
        max_batch_size=1, device='cpu')
    leaves, spec = flatten_tree(cache)
    rebuilt = reconstruct([t + 1 for t in leaves], spec)
    assert rebuilt is not cache
    for t in flatten_tree(cache)[0]:
        assert torch.count_nonzero(t) == 0
    for t in flatten_tree(rebuilt)[0]:
        assert torch.all(t == 1)
    assert flatten_tree(rebuilt)[1] == spec


def test_dynamic_cache_is_rejected():
    transformers = pytest.importorskip('transformers')
    with pytest.raises(TypeError, match='Unsupported input tree type: DynamicCache'):
        flatten_tree({'cache': transformers.DynamicCache()})


def test_static_cache_spec_uses_metadata_not_constructor_or_tensor_values(monkeypatch):
    transformers = pytest.importorskip('transformers')
    config = transformers.LlamaConfig(num_hidden_layers=1, num_attention_heads=2,
                                      hidden_size=8, max_position_embeddings=4)
    first = transformers.StaticCache(config, max_batch_size=1, device='cpu')
    second = transformers.StaticCache(config, max_batch_size=1, device='cpu')
    first.prefix_length = second.prefix_length = 2
    leaves, spec = flatten_tree(first)
    second.key_cache[0].fill_(7)
    other_spec = flatten_tree(second)[1]
    assert spec.constructor is not other_spec.constructor
    assert spec == other_spec
    assert hash(spec) == hash(other_spec)
    second.prefix_length = 3
    assert spec != flatten_tree(second)[1]

    def unexpected_init(*args, **kwargs):
        raise AssertionError('Reconstruction must not allocate a new cache')
    monkeypatch.setattr(transformers.StaticCache, '__init__', unexpected_init)
    first.prefix_length = 4  # Reconstruction uses the metadata snapshot.
    rebuilt = reconstruct(leaves, spec)
    assert rebuilt.prefix_length == 2
    assert rebuilt.key_cache is not first.key_cache
    assert rebuilt.key_cache[0] is first.key_cache[0]
    assert rebuilt.value_cache[0] is first.value_cache[0]


def test_static_cache_subclass_is_preserved():
    transformers = pytest.importorskip('transformers')
    class PrefixCache(transformers.StaticCache):
        pass

    cache = PrefixCache(
        transformers.LlamaConfig(num_hidden_layers=1, num_attention_heads=2,
                                 hidden_size=8, max_position_embeddings=4),
        max_batch_size=1, device='cpu')
    cache.prefix_length = 2
    leaves, spec = flatten_tree(cache)
    rebuilt = reconstruct([t.clone() for t in leaves], spec)
    assert isinstance(rebuilt, PrefixCache)
    assert rebuilt.prefix_length == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compile_batch_dispatch_replay_and_validation():
    device = 'cuda:0'
    def operator(inputs):
        return {'result': inputs.x['x'].sin() * inputs.x['scale'] + inputs.x['offset']}
    example = Input({'x': torch.randn(1, 8, device=device),
                     'offset': torch.tensor(0.5, device=device), 'scale': 2})
    with Compiler(device=device) as compiler:
        dispatch = compiler.compile(operator, example, [1, 3])
        for size in (1, 3, 1):
            source = Input({'x': torch.randn(size, 8, device=device),
                            'offset': torch.tensor(1.5, device=device), 'scale': 2})
            torch.testing.assert_close(dispatch(source), operator(source))
        with pytest.raises(ValueError, match='structure'):
            dispatch({'x': example.x})
        with pytest.raises(ValueError, match='shape and dtype'):
            dispatch(Input({**example.x, 'x': torch.randn(1, 9, device=device)}))
        with pytest.raises(ValueError, match='constants'):
            dispatch(Input({**example.x, 'scale': 3}))
        with pytest.raises(ValueError, match='No program'):
            dispatch(example, .5)
    with pytest.raises(RuntimeError, match='closed'):
        dispatch(example)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_mutating_inputs_reset_and_cross_stream():
    def operator(inputs):
        inputs['x'].add_(2)
        return inputs['x'] * 3
    source = {'x': torch.ones(1, 4, device='cuda')}
    with Compiler() as compiler:
        dispatch = compiler.compile(operator, source)
        consumer = torch.cuda.Stream()
        consumer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(consumer):
            for value in (1, 5):
                source['x'].fill_(value)
                output = dispatch(source)
                torch.testing.assert_close(output, torch.full_like(output, (value + 2) * 3))
                torch.testing.assert_close(source['x'], torch.full_like(source['x'], value))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_static_cache_replay_copies_values():
    transformers = pytest.importorskip('transformers')
    cache = transformers.StaticCache(
        transformers.LlamaConfig(num_hidden_layers=1, num_attention_heads=2,
                                 hidden_size=8, max_position_embeddings=4),
        max_batch_size=1, device='cuda:0')
    def operator(cache):
        return cache.key_cache[0] + cache.value_cache[0]
    with Compiler(device='cuda:0') as compiler:
        dispatch = compiler.compile(operator, cache, [1, 2])
        cache.key_cache[0].fill_(3)
        cache.value_cache[0].fill_(7)
        torch.testing.assert_close(dispatch(cache), operator(cache))
        larger = transformers.StaticCache(
            transformers.LlamaConfig(num_hidden_layers=1, num_attention_heads=2,
                                     hidden_size=8, max_position_embeddings=4),
            max_batch_size=2, device='cuda:0')
        larger.key_cache[0].fill_(4)
        larger.value_cache[0].fill_(9)
        torch.testing.assert_close(dispatch(larger), operator(larger))


def test_fractional_partitions_require_external_streams():
    with pytest.raises(ValueError, match='caller-supplied'):
        Compiler((0.5,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_precision_fallback_is_per_batch_and_handles_aliased_outputs(monkeypatch):
    def operator(x):
        return x.add_(1)

    def inaccurate_compile(fn, **kwargs):
        assert 'backend' not in kwargs
        def compiled(x):
            result = fn(x)
            # Batch one remains within tolerance, proving compiled is retained.
            return result.add_(0.001 if x.shape[0] == 1 else 1)
        return compiled

    monkeypatch.setattr(torch, 'compile', inaccurate_compile)
    source = torch.zeros(1, 4, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with Compiler(streams=[stream]) as compiler, torch.cuda.stream(stream):
        with pytest.warns(RuntimeWarning, match='batch=2.*capture of the original function') as caught:
            dispatch = compiler._compile(operator, source, [1, 2])
        assert len(caught) == 1
        assert torch._dynamo.config.recompile_limit >= 64
        assert dispatch.programs[(stream.cuda_stream, 1)].graph is not None
        fallback = dispatch.programs[(stream.cuda_stream, 2)]
        assert fallback.graph is not None
        assert fallback._function is operator
        buffer_address = fallback.buffer[0].data_ptr()
        torch.testing.assert_close(source, torch.zeros_like(source))
        for batch, increment in ((1, 1.001), (2, 1.0)):
            for value in (0, 3):
                inputs = torch.full((batch, 4), value, device='cuda', dtype=torch.float32)
                torch.testing.assert_close(dispatch(inputs), inputs + increment)
                torch.testing.assert_close(inputs, torch.full_like(inputs, value))
        assert fallback.buffer[0].data_ptr() == buffer_address
