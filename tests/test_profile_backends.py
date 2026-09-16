from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from robort.profile.policies import (FlashRTBackend, PolicyManager, _prepare_stage_targets,
                                    _checkpoint_stage, _pi05_stage_flops, _stage_weight_bytes)
from robort.profile.schemas import PolicyConfig, StageProfile


def config():
    return PolicyConfig(model_name='pi05', backend='flash_rt', num_views=2,
                        image_resolution='224', batch_sizes=[1], precision='fp16', prompt_len=16)


@pytest.mark.parametrize('capture', [False, True])
def test_stage_execution_capture_and_replay_use_supplied_stream(capture):
    torch = MagicMock()
    stream = Mock(device='cuda:1')
    active = []

    @contextmanager
    def use_stream(selected):
        assert selected is stream
        active.append(selected)
        try:
            yield
        finally:
            active.pop()

    torch.cuda.stream.side_effect = use_stream
    graphs = []

    def new_graph():
        graph = Mock()
        graph.replay.side_effect = lambda: assert_active()
        graphs.append(graph)
        return graph

    def assert_active():
        assert active[-1] is stream

    @contextmanager
    def capture_graph(graph, *, stream):
        with use_stream(stream):
            yield

    torch.cuda.CUDAGraph.side_effect = new_graph
    torch.cuda.graph.side_effect = capture_graph
    stages = {name: Mock(side_effect=assert_active) for name in ('vis', 'vlm', 'action')}
    targets = _prepare_stage_targets(stages, torch, stream, use_cuda_graph=capture)
    for target in targets.values():
        target()
    assert all(stage.call_count == (4 if capture else 2) for stage in stages.values())
    if capture:
        assert all(graph.replay.call_count == 2 for graph in graphs)
    for target in targets.values():
        target.close()
        with pytest.raises(RuntimeError, match='released'):
            target()
    assert all(graph.reset.call_count == 1 for graph in graphs)


def test_capture_error_releases_all_created_graphs():
    torch = MagicMock()
    stream = Mock()
    first, second = Mock(), Mock()
    torch.cuda.CUDAGraph.side_effect = [first, second]
    torch.cuda.graph.side_effect = [MagicMock(), RuntimeError('capture failed')]
    with pytest.raises(RuntimeError, match='capture failed'):
        _prepare_stage_targets({'vis': Mock(), 'action': Mock()}, torch, stream, use_cuda_graph=True)
    first.reset.assert_called_once()
    second.reset.assert_called_once()


def test_cache_is_bound_to_stream_and_results_are_copied(tmp_path, monkeypatch):
    (tmp_path / 'model.safetensors').touch()
    cfg = replace(config(), model_dir=str(tmp_path))
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.current_device.return_value = 1
    torch.cuda.get_device_capability.return_value = (8, 9)
    torch.device.return_value = 'cuda:1'
    monkeypatch.setattr('robort.profile.policies.import_module', lambda _: torch)
    backend = FlashRTBackend()
    metadata = {'vis': StageProfile(1, 2, 3, 4, 0, 0, 0, 0)}
    target = Mock()
    backend._load_model = Mock(return_value=object())
    backend._prepare = Mock(return_value=({'vis': target}, metadata))
    first, second = Mock(device='cuda:1'), Mock(device='cuda:1')
    _, result = backend.prepare_policies(cfg, stream=first)
    result['vis'].num_params = 99
    _, repeated = backend.prepare_policies(cfg, stream=first)
    assert repeated['vis'].num_params == 1
    assert backend._prepare.call_count == 1
    backend.prepare_policies(cfg, stream=second)
    assert backend._prepare.call_count == 2
    assert backend._prepare.call_args.args[-1] is second
    assert target.close.call_count == 1
    backend.release_execution()
    assert target.close.call_count == 2
    backend.prepare_policies(cfg, stream=first)
    assert backend._prepare.call_count == 3
    backend._load_model.assert_called_once()
    backend.release_policies()
    assert backend._model is None
    assert backend._prepared is None


def test_manager_passes_stream_and_releases():
    manager = PolicyManager()
    backend = Mock()
    manager.backends['flash_rt'] = backend
    stream = object()
    cfg = config()
    manager.prepare_policies(cfg, stream=stream)
    backend.prepare_policies.assert_called_once_with(cfg, stream=stream)
    manager.release_policies()
    backend.release_policies.assert_called_once()


@pytest.mark.parametrize('name,stage', [
    ('paligemma_with_expert.paligemma.model.vision_tower.x', 'vis'),
    ('model.paligemma_with_expert.paligemma.model.multi_modal_projector.x', 'vlm'),
    ('paligemma_with_expert.paligemma.lm_head.weight', 'vlm'),
    ('paligemma_with_expert.gemma_expert.model.layers.0.x', 'action'),
    ('time_mlp_in.weight', 'action'), ('action_out_proj.weight', 'action'),
])
def test_checkpoint_metadata_assignment(name, stage):
    assert _checkpoint_stage(name) == stage


def test_unknown_checkpoint_metadata_fails():
    with pytest.raises(ValueError, match='Cannot assign'):
        _checkpoint_stage('unexpected.weight')


def test_flop_scaling_by_stage():
    cfg = config()
    baseline = _pi05_stage_flops(cfg)
    more_steps = _pi05_stage_flops(replace(cfg, num_steps=20))
    assert all(value > 0 for value in baseline.values())
    assert more_steps['action'] == 2 * baseline['action']
    assert more_steps['vis'] == baseline['vis']
    assert more_steps['vlm'] == baseline['vlm']
    more_views = _pi05_stage_flops(replace(cfg, num_views=4))
    assert more_views['vis'] == 2 * baseline['vis']
    assert more_views['vlm'] > baseline['vlm']


def test_stage_weight_memory_deduplicates_and_includes_quantized_storage():
    class Tensor:
        is_cuda = True

        def __init__(self, ptr, size):
            self.ptr, self.size = ptr, size

        def data_ptr(self):
            return self.ptr

        def untyped_storage(self):
            return SimpleNamespace(data_ptr=self.data_ptr, nbytes=lambda: self.size)

    a, b, c, quant, scale = [Tensor(i, size) for i, size in enumerate([20, 40, 60, 10, 4])]
    frontend = SimpleNamespace(
        _ckpt_bf16={'vision_w': a, 'vision_alias': a, 'encoder_w': b, 'decoder_w': c},
        _fp8_weights={'vision_projector_w': (3, 4)}, _fp8_store=[quant, scale],
        _int8_weights={}, _int8_store=[])
    assert _stage_weight_bytes(frontend, SimpleNamespace(Tensor=Tensor)) == {
        'vis': 20, 'vlm': 54, 'action': 60}


def test_release_failure_still_closes_other_targets():
    backend = FlashRTBackend()
    first, second = Mock(), Mock()
    second.close.side_effect = RuntimeError('reset failed')
    backend._prepared = {'first': first, 'second': second}, {}
    with pytest.raises(RuntimeError, match='reset failed'):
        backend.release_policies()
    first.close.assert_called_once()
    second.close.assert_called_once()
    assert backend._prepared is None


@pytest.fixture
def cached_backend(tmp_path, monkeypatch):
    (tmp_path / 'model.safetensors').touch()
    cfg = replace(config(), model_dir=str(tmp_path))
    torch = MagicMock()
    torch.cuda.current_device.return_value = 0
    torch.cuda.get_device_capability.return_value = (8, 9)
    monkeypatch.setattr('robort.profile.policies.import_module', lambda _: torch)
    backend = FlashRTBackend()
    backend._load_model = Mock(side_effect=lambda *args: object())
    backend._prepare = Mock(side_effect=lambda *args: ({'vis': Mock()}, {}))
    return backend, cfg, torch


@pytest.mark.parametrize('changes', [
    {'prompt_len': 32}, {'num_views': 3}, {'num_steps': 20},
    {'chunk_size': 20}, {'use_cuda_graph': False},
])
def test_execution_changes_reuse_model(cached_backend, changes):
    backend, cfg, _ = cached_backend
    targets, _ = backend.prepare_policies(cfg)
    backend.prepare_policies(replace(cfg, **changes))
    backend._load_model.assert_called_once()
    assert backend._prepare.call_count == 2
    targets['vis'].close.assert_called_once()


@pytest.mark.parametrize('change', ['precision', 'checkpoint', 'device', 'environment', 'model_name'])
def test_model_changes_evict_cache(cached_backend, change, tmp_path, monkeypatch):
    backend, cfg, torch = cached_backend
    targets, _ = backend.prepare_policies(cfg)
    if change == 'precision':
        cfg.precision = 'fp8'
    elif change == 'checkpoint':
        other = tmp_path / 'other'
        other.mkdir()
        (other / 'model.safetensors').touch()
        cfg.model_dir = str(other)
    elif change == 'device':
        torch.cuda.current_device.return_value = 1
    elif change == 'environment':
        monkeypatch.setenv('FVK_PI05_RTX_FORCE_BF16', '1')
    else:
        cfg.model_name = 'pi05_libero'
    backend.prepare_policies(cfg)
    assert backend._load_model.call_count == 2
    targets['vis'].close.assert_called_once()


def test_execution_failure_keeps_model_for_retry(cached_backend):
    backend, cfg, _ = cached_backend
    backend._prepare.side_effect = RuntimeError('preparation failed')
    with pytest.raises(RuntimeError, match='preparation failed'):
        backend.prepare_policies(cfg)
    assert backend._prepared is None
    backend.release_execution()
    backend._prepare.side_effect = lambda *args: ({'vis': Mock()}, {})
    backend.prepare_policies(cfg)
    backend._load_model.assert_called_once()
    backend.release_policies()
    backend.prepare_policies(cfg)
    assert backend._load_model.call_count == 2


@pytest.mark.parametrize('precision', ['fp16', 'fp8'])
def test_execution_frontend_shares_weights_but_rebuilds_runtime(precision):
    model = SimpleNamespace(
        _ckpt_bf16={'decoder_action_out_proj_w': 8.0,
                    'decoder_action_out_proj_b': 4.0, 'vision_w': object()},
        fvk=SimpleNamespace(GemmRunner=Mock(side_effect=object)),
        _prompt_pipeline_cache={}, latency_records=[],
        calibrated=False, pipeline=None,
    )
    module = SimpleNamespace(
        _precompute_decoder_styles=Mock(side_effect=lambda *args, **kw: object()),
        RtxFlashAttnBackend=Mock(side_effect=lambda **kw: object()), bf16='dtype', ENC_L=18)
    torch = SimpleNamespace(empty=Mock(side_effect=lambda *args, **kw: object()))
    cfg = replace(config(), precision=precision, num_steps=2)
    first = FlashRTBackend._execution_frontend(cfg, module, model, torch)
    second = FlashRTBackend._execution_frontend(
        replace(cfg, num_steps=4, chunk_size=20, prompt_len=32), module, model, torch)
    assert first._ckpt_bf16['vision_w'] is model._ckpt_bf16['vision_w']
    assert first._ckpt_bf16['decoder_action_out_proj_w'] == -4
    assert second._ckpt_bf16['decoder_action_out_proj_w'] == -2
    assert model._ckpt_bf16['decoder_action_out_proj_w'] == 8
    assert model._ckpt_bf16['decoder_action_out_proj_b'] == 4
    assert first.attn_backend is not second.attn_backend
    assert first.gemm is not second.gemm
    assert first._noise_buf is not second._noise_buf
    assert first._precomputed_styles is not second._precomputed_styles
    first.calibrated = True
    assert second.calibrated is False
    assert model.calibrated is False
    assert second.chunk_size == 20
    assert second.max_prompt_len == 32
    assert module._precompute_decoder_styles.call_args.kwargs == {'num_steps': 4}
    assert module.RtxFlashAttnBackend.call_args.kwargs['encoder_seq_max'] == 544


def test_manager_releases_execution_separately():
    manager = PolicyManager()
    backend = Mock()
    manager.backends['flash_rt'] = backend
    manager.release_execution()
    backend.release_execution.assert_called_once()
    backend.release_policies.assert_not_called()


@pytest.mark.parametrize('precision', ['fp16', 'fp8'])
def test_model_load_unscales_weights_and_drops_execution_resources(precision, tmp_path, monkeypatch):
    import sys

    fp8 = precision == 'fp8'
    frontend = SimpleNamespace(
        _ckpt_bf16={'decoder_action_out_proj_w': -8.0, 'decoder_action_out_proj_b': -4.0},
        _pipeline_precision_kwargs=lambda: {'use_fp8': fp8, 'use_fp8_decoder': fp8},
    )
    constructor = Mock(return_value=frontend)
    module = SimpleNamespace(Pi05TorchFrontendRtx=constructor, Pi05TorchFrontendRtxFP16=constructor)
    monkeypatch.setattr('robort.profile.policies.import_module', lambda _: module)
    weights = MagicMock()
    weights.__enter__.return_value = weights
    weights.keys.return_value = ['time_mlp_in.weight']
    weights.get_slice.return_value.get_shape.return_value = (2, 3)
    monkeypatch.setitem(sys.modules, 'safetensors', SimpleNamespace(safe_open=Mock(return_value=weights)))
    torch = MagicMock()
    loaded_module, model, params = FlashRTBackend._load_model(
        replace(config(), precision=precision, num_steps=20), tmp_path, torch, (8, 9))
    assert loaded_module is module
    assert model._ckpt_bf16['decoder_action_out_proj_w'] == 8
    assert model._ckpt_bf16['decoder_action_out_proj_b'] == 4
    assert constructor.call_args.kwargs['num_steps'] == 1
    assert constructor.call_args.kwargs['use_fp8'] == fp8
    assert params == {'vis': 0, 'vlm': 0, 'action': 6}
    for name in ('attn_backend', 'gemm', '_img_buf', '_noise_buf', '_noise_out', '_precomputed_styles'):
        assert getattr(model, name) is None
    torch.cuda.stream.assert_called_once_with(torch.cuda.default_stream.return_value)
    torch.cuda.synchronize.assert_called_once()


def test_model_load_failure_is_retryable(cached_backend):
    backend, cfg, _ = cached_backend
    backend._load_model.side_effect = RuntimeError('load failed')
    with pytest.raises(RuntimeError, match='load failed'):
        backend.prepare_policies(cfg)
    assert backend._model is None
    assert backend._model_key is None
    backend._prepare.assert_not_called()
    backend._load_model.side_effect = lambda *args: object()
    backend.prepare_policies(cfg)
    assert backend._model is not None
