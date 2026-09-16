"""Real CUDA tests for tensor-tree graph capture (no model checkpoint needed)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from robort.policies.cuda_graph import capture_torch


def test_rejects_invalid_inputs_before_cuda_allocation():
    stream = SimpleNamespace(device=torch.device('cuda:0'))
    with pytest.raises(TypeError, match='dictionary'):
        capture_torch(lambda: None, [], stream)
    with pytest.raises(TypeError, match='leaves'):
        capture_torch(lambda **kw: None, {'x': {'bad': 3}}, stream)
    with pytest.raises(ValueError, match='CUDA tensors'):
        capture_torch(lambda x: x, {'x': torch.ones(2)}, stream)
    for warmups in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='warmups'):
            capture_torch(lambda: None, {}, stream, warmups=warmups)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@cuda
def test_nested_inputs_changed_values_and_shared_outputs():
    stream = torch.cuda.Stream()
    calls = []
    def fn(batch, bias):
        calls.append(1)
        assert not torch.is_grad_enabled()
        value = batch['images'][0] * batch['scale'][0] + bias
        return {'actions': value, 'sum': (value.sum(),)}
    examples = {'batch': {'images': [torch.ones(4, device='cuda')],
                          'scale': (torch.tensor(2., device='cuda'),)},
                'bias': torch.zeros(4, device='cuda')}
    replay = capture_torch(fn, examples, stream)
    try:
        assert len(calls) == 4  # Three warmups and capture; no Python on replay.
        first = replay(**examples)
        saved = first['actions'].clone()
        examples['batch']['images'][0].fill_(5)
        examples['bias'].fill_(3)
        second = replay(**examples)
        assert first is second
        torch.testing.assert_close(saved, torch.full((4,), 2., device='cuda'))
        torch.testing.assert_close(second['actions'], torch.full((4,), 13., device='cuda'))
        torch.testing.assert_close(second['sum'][0], torch.tensor(52., device='cuda'))
        assert len(calls) == 4
    finally:
        replay.close()
    replay.close()
    with pytest.raises(RuntimeError, match='closed'):
        replay(**examples)


@cuda
def test_cross_stream_inputs_outputs_and_noncontiguous_sources():
    capture_stream, caller = torch.cuda.Stream(), torch.cuda.Stream()
    replay = capture_torch(lambda x: x.square(), {'x': torch.ones((4, 3), device='cuda')}, capture_stream)
    try:
        with torch.cuda.stream(caller):
            x = torch.arange(12., device='cuda').reshape(3, 4).t()
            result = replay(x=x).clone()
            expected = x.square()
        caller.synchronize()
        torch.testing.assert_close(result, expected)
    finally:
        replay.close()


@cuda
def test_mutating_function_does_not_modify_callers_input():
    def fn(x):
        x.add_(1)
        return x
    x = torch.ones(4, device='cuda', requires_grad=True)
    replay = capture_torch(fn, {'x': x}, torch.cuda.Stream())
    try:
        for _ in range(2):
            torch.testing.assert_close(replay(x=x), torch.full_like(x, 2))
        torch.testing.assert_close(x, torch.ones_like(x))
    finally:
        replay.close()


@cuda
def test_replay_validation_and_recovery():
    x = torch.ones(4, device='cuda')
    replay = capture_torch(lambda x: x * 2, {'x': x}, torch.cuda.Stream())
    try:
        for kwargs, error, match in [
            ({'other': x}, ValueError, 'structure'),
            ({'x': [x]}, ValueError, 'structure'),
            ({'x': 2}, TypeError, 'leaves'),
            ({'x': x.cpu()}, ValueError, 'CUDA tensors'),
            ({'x': x.double()}, ValueError, 'dtype'),
            ({'x': x[:2]}, ValueError, 'shape'),
        ]:
            with pytest.raises(error, match=match):
                replay(**kwargs)
        torch.testing.assert_close(replay(x=x), x * 2)
    finally:
        replay.close()


@cuda
def test_capture_exception_does_not_prevent_next_capture():
    x = torch.ones(4, device='cuda')
    calls = 0
    def fail_during_capture(x):
        nonlocal calls
        calls += 1
        result = x + 1
        if calls == 4:
            raise RuntimeError('capture failed')
        return result
    stream = torch.cuda.Stream()
    with pytest.raises(RuntimeError, match='capture failed'):
        capture_torch(fail_during_capture, {'x': x}, stream)
    replay = capture_torch(lambda x: x + 2, {'x': x}, stream)
    try:
        torch.testing.assert_close(replay(x=x), x + 2)
    finally:
        replay.close()
