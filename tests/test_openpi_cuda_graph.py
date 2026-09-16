"""Opt-in real-checkpoint graph tests: ROBORT_TEST_OPENPI_GRAPH=1 JAX_PLATFORMS=cpu."""

import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('openpi')
from robort.policies.openpi import OpenPIPolicy
from robort.policies.openpi_cuda_graph import CudaGraphOpenPIPolicy, _observation_tensors
from robort.schemas import PolicyConfig

pytestmark = pytest.mark.skipif(os.environ.get('ROBORT_TEST_OPENPI_GRAPH') != '1',
                                reason='requires CUDA, OpenPI and checkpoint')


@pytest.fixture(scope='module')
def policy():
    policy = OpenPIPolicy(PolicyConfig(
        model_dir='checkpoints/pi05_libero_pytorch', batch_sizes=[1, 2],
        max_batch_size=2, num_steps=3), 'cuda:0')
    stream = torch.cuda.Stream()
    with CudaGraphOpenPIPolicy(policy, stream) as graphed:
        assert not hasattr(policy, "_graphs")
        assert not hasattr(policy, "compile")
        yield graphed


@pytest.mark.parametrize('size', [1, 2])
def test_graph_matches_compiled_execution_and_updates_inputs(policy, size):
    previous = None
    for seed in [13, 27]:
        rng = np.random.default_rng(seed)
        requests = [policy.make_example_input() for _ in range(size)]
        for req in requests:
            for key in ('observation/image', 'observation/wrist_image'):
                req.observation[key] = rng.integers(0, 256, (224, 224, 3), np.uint8)
            req.observation['observation/state'] = rng.uniform(-0.2, 0.2, 8).astype(np.float32)
            req.observation['prompt'] = 'pick up the red block' if seed == 13 else 'open the drawer'
        obs = policy.preprocess(requests)
        noise = torch.from_numpy(rng.standard_normal((size, policy.policy._model.config.action_horizon,
                                                      policy.policy._model.config.action_dim)).astype(np.float32)).cuda()
        actual = policy.decode(policy.encode(policy.embed(obs)), noise=noise)['actions'].clone()
        # Run precisely the compiled GPU functions without CUDA graph replay,
        # separating graph correctness from Inductor's numerical differences.
        with torch.inference_mode():
            stages = policy._graphs[size]
            embedding = stages['embed'].fn(observation=_observation_tensors(obs))
            context = stages['encode'].fn(embedding=embedding)
            reference = stages['decode'].fn(context=context, noise=noise)['actions']
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
        assert actual.isfinite().all()
        if previous is not None:
            assert not torch.equal(actual, previous)
        previous = actual
        responses = policy.postprocess({'state': obs.state, 'actions': actual})
        assert len(responses) == size
        assert responses[0].actions.shape == (policy.policy._model.config.action_horizon, 7)


def test_replay_generates_fresh_noise(policy):
    obs = policy.preprocess([policy.make_example_input()])
    context = policy.encode(policy.embed(obs))
    first = policy.decode(context)['actions'].clone()
    second = policy.decode(context)['actions'].clone()
    assert not torch.equal(first, second)


def test_close_leaves_original_policy_usable(policy):
    eager = policy.policy
    obs = eager.preprocess([eager.make_example_input()])
    noise = torch.zeros(1, eager._model.config.action_horizon, eager._model.config.action_dim, device=eager.device)
    before = eager.decode(eager.encode(eager.embed(obs)), noise=noise)['actions']
    policy.close()
    with pytest.raises(RuntimeError, match='closed'):
        policy.embed(obs)
    after = eager.decode(eager.encode(eager.embed(obs)), noise=noise)['actions']
    torch.testing.assert_close(before, after, rtol=0, atol=0)
