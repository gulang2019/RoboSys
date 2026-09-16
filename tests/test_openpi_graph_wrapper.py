"""Graph execution owns its state and never replaces eager policy methods."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('openpi')
from robort.policies.config import PolicyConfig
from robort.policies import openpi_cuda_graph as module


@pytest.fixture
def eager():
    return SimpleNamespace(
        config=PolicyConfig(num_sample_steps=4), device='cuda:0',
        make_example_input=Mock(return_value=object()),
        preprocess=Mock(return_value=SimpleNamespace(state=torch.zeros(1, 8))),
        embed=Mock(), encode=Mock(), decode=Mock(), postprocess=Mock(),
        _model=SimpleNamespace(config=SimpleNamespace(action_horizon=10, action_dim=32),
                               sample_noise=lambda shape, device: torch.zeros(shape)),
    )


@pytest.mark.parametrize('sizes', [[], [0], [-1], [True], [1.5], [11]])
def test_validates_batch_sizes_before_capture(eager, sizes):
    with pytest.raises(ValueError, match='batch_sizes'):
        module.CudaGraphOpenPIPolicy(eager, SimpleNamespace(device=torch.device('cuda:0')), batch_sizes=sizes)


def test_dispatch_and_close_are_separate_from_eager(eager):
    wrapper = module.CudaGraphOpenPIPolicy.__new__(module.CudaGraphOpenPIPolicy)
    wrapper.policy, wrapper.config, wrapper._steps = eager, eager.config, 4
    one, two = Mock(), Mock()
    wrapper._graphs = {1: {'embed': one}, 2: {'embed': two}}
    assert wrapper._graph('embed', torch.zeros(1, 8)) is one
    assert wrapper._graph('embed', torch.zeros(2, 8)) is two
    with pytest.raises(ValueError, match='not captured'):
        wrapper._graph('embed', torch.zeros(3, 8))
    eager.config.num_sample_steps = 2
    with pytest.raises(ValueError, match='steps changed'):
        wrapper._graph('embed', torch.zeros(1, 8))
    wrapper.close()
    wrapper.close()
    one.close.assert_called_once()
    two.close.assert_called_once()
    with pytest.raises(RuntimeError, match='closed'):
        wrapper._graph('embed', torch.zeros(1, 8))
    assert not hasattr(eager, '_graphs')


@pytest.mark.parametrize('fail', [False, True])
def test_capture_ownership_and_failure_cleanup(eager, monkeypatch, fail):
    before = dict(vars(eager))
    stream = SimpleNamespace(device=torch.device('cuda:0'), synchronize=Mock())
    monkeypatch.setattr(torch, 'compile', lambda fn, **kwargs: fn)
    targets = [Mock(return_value={}) for _ in range(3)]
    capture = Mock(side_effect=[*targets[:2], RuntimeError('capture failure') if fail else targets[2]])
    monkeypatch.setattr(module, 'capture_torch', capture)
    if fail:
        with pytest.raises(RuntimeError, match='capture failure'):
            module.CudaGraphOpenPIPolicy(eager, stream)
        targets[0].close.assert_called_once()
        targets[1].close.assert_called_once()
    else:
        with module.CudaGraphOpenPIPolicy(eager, stream) as wrapper:
            wrapper.preprocess([])
            wrapper.postprocess({})
            eager.postprocess.assert_called_once_with({})
            assert wrapper.infer([]) == []
        for target in targets:
            target.close.assert_called_once()
    assert vars(eager) == before
