"""PI0/PI0.5 attention masks can be constructed during raw CUDA capture."""
from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('pi05', [False, True])
@pytest.mark.parametrize('compiled', [False, True])
@torch.inference_mode()
def test_suffix_mask_cuda_capture(pi05, compiled):
    width, horizon, batch = 8, 4, 2
    linear = lambda inp, out: torch.nn.Linear(inp, out).cuda().eval()
    model = SimpleNamespace(
        pi05=pi05, config=SimpleNamespace(action_horizon=horizon),
        _apply_checkpoint=lambda fn, *args: fn(*args),
        state_proj=linear(3, width), action_in_proj=linear(3, width),
        action_time_mlp_in=linear(2 * width, width), action_time_mlp_out=linear(width, width),
        time_mlp_in=linear(width, width), time_mlp_out=linear(width, width))
    state = torch.randn(batch, 3, device='cuda')
    actions = torch.randn(batch, horizon, 3, device='cuda')
    time = torch.ones(batch, device='cuda')
    operation = lambda state, actions, time: PI0Pytorch.embed_suffix(model, state, actions, time)
    if compiled:
        operation = torch.compile(operation, dynamic=True, fullgraph=True,
                                  options={'triton.cudagraphs': False})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = PI0Pytorch.embed_suffix(model, state, actions, time)
        operation(state, actions, time)  # Compile and initialize before capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = operation(state, actions, time)
        graph.replay()
        stream.synchronize()
        torch.testing.assert_close(actual, expected)
        mask = [1, 0, 0, 0] if pi05 else [1, 1, 0, 0, 0]
        torch.testing.assert_close(actual[2], torch.tensor([mask] * batch, device='cuda', dtype=actual[2].dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_prefix_mask_cuda_capture():
    model = SimpleNamespace(
        _apply_checkpoint=lambda fn, *args: fn(*args),
        paligemma_with_expert=SimpleNamespace(embed_image=lambda x: x, embed_language_tokens=lambda x: x))
    images = [torch.ones(2, 3, 8, device='cuda')] * 2
    masks = [torch.ones(2, dtype=torch.bool, device='cuda')] * 2
    tokens = torch.ones(2, 4, 8, device='cuda')
    lang_mask = torch.ones(2, 4, dtype=torch.bool, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = PI0Pytorch.embed_prefix(model, images, masks, tokens, lang_mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = PI0Pytorch.embed_prefix(model, images, masks, tokens, lang_mask)
        graph.replay()
        stream.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[2], torch.zeros(2, 10, dtype=torch.bool, device='cuda'))
