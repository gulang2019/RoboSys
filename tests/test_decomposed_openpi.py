"""Regression checks for stage boundaries without loading model weights."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openpi")
from robort.policies.openpi import OpenPIPolicy
from robort.schemas import InferenceRequest, PolicyConfig


@pytest.fixture
def policy():
    policy = OpenPIPolicy.__new__(OpenPIPolicy)
    policy.config = PolicyConfig(num_sample_steps=4)
    policy.device = "cpu"
    policy.policy = SimpleNamespace(
        _input_transform=lambda data: {
            "state": data["state"],
            "image": {"camera": np.zeros((4, 4, 3), dtype=np.uint8)},
            "image_mask": {"camera": np.True_},
        },
        _output_transform=lambda data: {"actions": data["actions"] + data["state"][0]},
    )
    return policy


def test_preprocess_stacks_transformed_requests(policy):
    requests = [InferenceRequest({"state": np.full(8, i, np.float32)}) for i in range(2)]
    obs = policy.preprocess(requests)
    assert obs.state.shape == (2, 8)
    assert obs.state[:, 0].tolist() == [0, 1]
    assert obs.images["camera"].shape == (2, 3, 4, 4)
    assert obs.images["camera"].min() == -1
    assert obs.image_masks["camera"].dtype == torch.bool
    assert set(requests[0].observation) == {"state"}


def test_postprocess_preserves_batch_order(policy):
    outputs = {"state": torch.tensor([[1.0], [7.0]]), "actions": torch.zeros(2, 3, 2)}
    responses = policy.postprocess(outputs)
    assert len(responses) == 2
    np.testing.assert_array_equal(responses[0].actions, np.ones((3, 2)))
    np.testing.assert_array_equal(responses[1].actions, np.full((3, 2), 7))
    assert outputs["actions"].shape == (2, 3, 2)


def test_decode_steps_and_disables_autograd(policy):
    times = []
    def denoise(state, masks, cache, actions, time):
        assert not torch.is_grad_enabled()
        times.append(time[0].item())
        return torch.ones_like(actions)
    policy._model = SimpleNamespace(denoise_step=denoise)
    context = {"state": torch.zeros(2, 8), "prefix_pad_masks": torch.ones(2, 3), "past_key_values": object()}
    outputs = policy.decode(context, noise=torch.zeros(2, 3, 2))
    assert times == [1.0, 0.75, 0.5, 0.25]
    torch.testing.assert_close(outputs["actions"], -torch.ones(2, 3, 2))


def test_batch_validation(policy):
    assert policy.infer([]) == []
    with pytest.raises(ValueError, match="nonempty"):
        policy.preprocess([])
    with pytest.raises(ValueError, match="exceeds"):
        policy.preprocess([InferenceRequest({})] * 11)
    with pytest.raises(ValueError, match="sync"):
        policy.preprocess([InferenceRequest({}, inference_type="rtc")])


def test_invalid_step_count():
    with pytest.raises(ValueError, match="num_sample_steps"):
        OpenPIPolicy(PolicyConfig(num_sample_steps=0))


def test_encode_reuses_static_cache_and_overwrites_prefix(policy):
    from transformers import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaModel
    from transformers.cache_utils import DynamicCache, StaticCache

    model = GemmaModel(GemmaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=32, max_position_embeddings=32,
    )).eval()
    model.config._attn_implementation = 'eager'
    policy._model = SimpleNamespace(paligemma_with_expert=SimpleNamespace(
        paligemma=SimpleNamespace(language_model=model)))
    policy._caches = {}

    def embedding(batch, length):
        return {
            'state': torch.zeros(batch, 8),
            'prefix_embeds': torch.randn(batch, length, 16),
            'prefix_position_ids': torch.arange(length).expand(batch, -1),
            'prefix_att_2d_masks_4d': torch.zeros(batch, 1, length, length),
            'prefix_pad_masks': torch.ones(batch, length, dtype=torch.bool),
        }

    first = policy.encode(embedding(1, 4))['past_key_values']
    assert isinstance(first, StaticCache)
    assert first.key_cache[0].shape[2] == model.config.max_position_embeddings
    pointers = [(k.data_ptr(), v.data_ptr()) for k, v in first.to_legacy_cache()]
    for batch, length in ((1, 4), (2, 4), (1, 6), (1, 4)):
        inputs = embedding(batch, length)
        with torch.no_grad():
            expected = model(
                inputs_embeds=inputs['prefix_embeds'],
                position_ids=inputs['prefix_position_ids'],
                attention_mask=inputs['prefix_att_2d_masks_4d'],
                past_key_values=DynamicCache(), use_cache=True,
            ).past_key_values
        cache = policy.encode(inputs)['past_key_values']
        for actual_pair, expected_pair in zip(cache.to_legacy_cache(), expected.to_legacy_cache()):
            torch.testing.assert_close(actual_pair, expected_pair)
        assert cache.get_seq_length() == length
        if batch == 1:
            assert cache is first
            assert [(k.data_ptr(), v.data_ptr()) for k, v in cache.to_legacy_cache()] == pointers
        # The read-only suffix path used by OpenPI must also accept this cache.
        before = [(k.clone(), v.clone()) for k, v in cache.to_legacy_cache()]
        suffix_inputs = {
            'inputs_embeds': torch.randn(batch, 2, 16),
            'attention_mask': torch.zeros(batch, 1, 2, length + 2),
            'position_ids': torch.arange(length, length + 2).expand(batch, -1),
            'use_cache': False,
        }
        with torch.no_grad():
            suffix = model(**suffix_inputs, past_key_values=cache)
            reference = model(**suffix_inputs, past_key_values=expected)
        torch.testing.assert_close(suffix.last_hidden_state, reference.last_hidden_state)
        assert suffix.last_hidden_state.shape == (batch, 2, 16)
        torch.testing.assert_close(cache.to_legacy_cache(), tuple(before))
    assert set(policy._caches) == {1, 2}
    with pytest.raises(ValueError, match='capacity'):
        policy.encode(embedding(1, 33))


@pytest.mark.parametrize('keep_rate', [0.0, 0.5, 1.0])
def test_pruned_encode_preserves_unselected_cache_and_saves_cdf(policy, monkeypatch, tmp_path, keep_rate):
    from transformers import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaModel
    from robort.policies.openpi import _StaticPrefixCache
    from PIL import Image

    monkeypatch.chdir(tmp_path)
    model = GemmaModel(GemmaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=32, max_position_embeddings=16,
    )).eval()
    policy._model = SimpleNamespace(paligemma_with_expert=SimpleNamespace(
        paligemma=SimpleNamespace(language_model=model)))
    policy.config.encode_keep_rate = keep_rate
    policy._num_visual_tokens = lambda: 4
    policy.reset()
    cache = _StaticPrefixCache(model.config, max_batch_size=1, max_cache_len=16,
                              device='cpu', dtype=torch.float32)
    cache.prefix_length = 6
    prefix = torch.randn(1, 6, 16)
    embedding = {
        'state': torch.zeros(1, 8), 'prefix_embeds': prefix,
        'prefix_position_ids': torch.arange(6)[None],
        'prefix_att_2d_masks_4d': torch.zeros(1, 1, 6, 6),
        'prefix_pad_masks': torch.ones(1, 6, dtype=torch.bool),
        'past_key_values': cache,
    }
    embedding['debug_images'] = []
    attention_scores = []
    monkeypatch.setattr('robort.policies.openpi._save_pruning_images',
                        lambda *args, **kwargs: attention_scores.append(kwargs['attention_scores']))
    positions = []
    hook = model.layers[0].register_forward_pre_hook(
        lambda module, args, kwargs: positions.append(kwargs['cache_position'].clone()),
        with_kwargs=True,
    )
    policy.encode(embedding)
    assert positions[-1].tolist() == list(range(6))
    assert attention_scores[-1].shape == (4,)
    assert torch.isfinite(attention_scores[-1]).all()
    before = [(k.clone(), v.clone()) for k, v in cache.to_legacy_cache()]
    baseline = policy._prev_visual.clone()
    # Increasing changes make the selected visual positions deterministic.
    prefix[0, :, 0] += torch.arange(1, 7)
    policy.encode(embedding)
    selected = list(range(4 - round(keep_rate * 4), 6))
    assert positions[-1].tolist() == selected
    unselected = list(range(4 - round(keep_rate * 4)))
    for (old_k, old_v), (new_k, new_v) in zip(before, cache.to_legacy_cache()):
        torch.testing.assert_close(new_k[:, :, unselected], old_k[:, :, unselected], rtol=0, atol=0)
        torch.testing.assert_close(new_v[:, :, unselected], old_v[:, :, unselected], rtol=0, atol=0)
    assert not torch.equal(cache[0][0][:, :, selected], before[0][0][:, :, selected])
    torch.testing.assert_close(policy._prev_visual, baseline)
    from robort.policies.openpi import _save_score_cdf
    assert set(policy._score_history) == {2}
    _save_score_cdf(policy._score_history, tmp_path / 'scores.cdf.png')
    with Image.open(tmp_path / 'scores.cdf.png') as figure:
        assert figure.format == 'PNG'
        figure.verify()
    policy.reset()
    policy.encode(embedding)
    assert positions[-1].tolist() == list(range(6))
    hook.remove()


@pytest.mark.parametrize('scores', [torch.tensor([0., 1., 2., 3., 3., 2., 1., 0.]), torch.zeros(8)])
def test_pruning_images_shade_by_score(monkeypatch, tmp_path, scores):
    from PIL import Image
    from robort.policies.openpi import _save_pruning_images

    monkeypatch.chdir(tmp_path)
    images = [torch.zeros(1, 3, 28, 28), torch.ones(1, 28, 28, 3)]
    _save_pruning_images(images, scores, 2)
    levels = [0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25] if scores.any() else [0.25] * 8
    for view, (name, brightness) in enumerate([('global', 128), ('wrist', 255)]):
        with Image.open(tmp_path / f'debug/2.{name}.png') as image:
            np.testing.assert_array_equal(np.asarray(image), np.full((28, 28, 3), brightness))
        with Image.open(tmp_path / f'debug/2.{name}.masked.png') as image:
            pixels = np.asarray(image)
            for patch in range(4):
                row, col = divmod(patch, 2)
                expected = round(brightness * levels[view * 4 + patch])
                assert np.all(pixels[row * 14:(row + 1) * 14, col * 14:(col + 1) * 14] == expected)
    _save_pruning_images(images, None, 1)
    for name in ('global', 'wrist'):
        assert (tmp_path / f'debug/1.{name}.png').read_bytes() == (
            tmp_path / f'debug/1.{name}.masked.png').read_bytes()


@pytest.mark.parametrize('fail_cdf', [False, True])
def test_debug_video_export_preserves_frame_order_and_masks(policy, tmp_path, monkeypatch, fail_cdf):
    import imageio.v2 as imageio
    from PIL import Image
    from robort.policies.openpi import _save_pruning_images

    policy._debug_output = tmp_path
    policy._iter = 10
    policy.reset()
    assert policy._iter == 10
    # Write out of order to exercise numeric frame sorting.
    for iteration, value in [(10, 1.0), (2, 0.0)]:
        images = [torch.full((1, 3, 28, 28), value)] * 2
        _save_pruning_images(images, torch.arange(8).float(), iteration,
                             indices=torch.tensor([0, 5]), output=tmp_path,
                             attention_scores=torch.arange(8).float())
    for camera, column in [('global', 0), ('wrist', 14)]:
        with Image.open(tmp_path / f'2.{camera}.binary.png') as image:
            expected = np.zeros((28, 28, 3), dtype=np.uint8)
            expected[:14, column:column + 14] = 128
            np.testing.assert_array_equal(np.asarray(image), expected)
    policy._score_history = {2: np.array([0., 1.]), 10: np.array([1., 2.])}
    if fail_cdf:
        def fail(*args):
            raise RuntimeError('CDF write failed')
        monkeypatch.setattr('robort.policies.openpi._save_score_cdf', fail)
        with pytest.raises(RuntimeError, match='CDF write failed'):
            policy.export_debug_videos()
        assert (tmp_path / '2.global.png').exists()
        assert (tmp_path / '10.wrist.binary.png').exists()
        assert not (tmp_path / 'combined.mp4').exists()
        return
    (tmp_path / 'global.full.mp4').write_bytes(b'old individual video')
    policy.export_debug_videos()
    assert {p.name for p in tmp_path.iterdir()} == {'combined.mp4', 'scores.cdf.png', 'attention.cdf.png'}
    with Image.open(tmp_path / 'scores.cdf.png') as figure:
        figure.verify()
    policy.export_debug_videos()  # Repeated close/export must preserve the final files.
    videos = list(tmp_path.glob('*.mp4'))
    assert len(videos) == 1
    for video in videos:
        with imageio.get_reader(video) as reader:
            frames = list(reader.iter_data())
            assert reader.get_meta_data()['fps'] == 10
        assert len(frames) == 2
        assert frames[0].shape == ((104, 112, 3) if video.name == 'combined.mp4' else (28, 28, 3))
        assert frames[0].mean() < frames[1].mean()


def test_text_attention_uses_selected_valid_text_queries():
    from robort.policies.openpi import _text_attention_scores

    # Physical prefix: four visual slots, two valid text tokens, one padding token.
    # Pruned query rows represent visual slot 2, text slots 4/5, and padding slot 6.
    indices = torch.tensor([2, 4, 5, 6])
    mask = torch.tensor([[True, True, False, True, True, True, False]])
    layer = torch.full((1, 2, 4, 7), 1000.)
    layer[0, :, 1, :4] = torch.tensor([1., 2., 3., 4.])
    layer[0, :, 2, :4] = torch.tensor([3., 4., 5., 6.])
    actual = _text_attention_scores((layer, layer * 3), indices, mask, 4)
    torch.testing.assert_close(actual, torch.tensor([4., 6., 0., 10.]))
    mask[:, 4:] = False
    torch.testing.assert_close(_text_attention_scores((layer,), indices, mask, 4), torch.zeros(4))
