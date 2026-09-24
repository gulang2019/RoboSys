"""Shared graph pools preserve outputs across interleaved batch/stream replay."""
from contextlib import ExitStack

import pytest
import torch

from robort.policies.compiler_utils import Compiler
from robort.profile.environment import _green_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('green', [False, True])
@pytest.mark.parametrize('compiled', [False, True])
def test_shared_pool_replay(green, compiled, monkeypatch):
    # Exercise allocator reuse with distinct intermediate and output tensors.
    if not compiled:
        monkeypatch.setattr(torch, 'compile', lambda *a, **kw: pytest.fail('torch.compile must be skipped'))
    device = torch.cuda.current_device()
    with ExitStack() as resources:
        streams = [resources.enter_context(_green_stream(torch, device, 64))
                   if green else torch.cuda.Stream() for _ in range(2)]
        x = torch.ones(1, 1024, device='cuda')
        compiler = resources.enter_context(Compiler(streams=streams, use_torch_compile=compiled))
        first = compiler._compile(lambda x: (x.sin() + 2).cos(), x, [1, 3])
        second = compiler._compile(lambda x: (x * 2).sin() + 1, x, [1, 3])
        programs = [p for d in (first, second) for p in d.programs.values()]
        assert len({p.graph.pool() for p in programs}) == 1
        retained = []
        # Different replay order from capture; switch streams without CPU waits.
        for index, size in [(1, 3), (0, 1), (1, 1), (0, 3)]:
            with torch.cuda.stream(streams[index]):
                source = torch.full((size, 1024), index + size, device='cuda', dtype=torch.float32)
                output = first(source)
                result = second(output)
                expected = ((source.sin() + 2).cos() * 2).sin() + 1
                retained.append((result, expected))
        torch.cuda.synchronize()
        for result, expected in retained:
            torch.testing.assert_close(result, expected)
