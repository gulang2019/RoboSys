"""Check torch.compile + CUDA graph parity for all PyTorch OpenPI policy."""

import argparse
from pathlib import Path

import torch
import time 
import numpy as np

from robort.schemas import PolicyConfig, InferenceRequest
from robort.policies import create_policy
from robort.policies.compiler_utils import Compiler, flatten_tree, reconstruct

from contextlib import contextmanager 

@contextmanager
def timer(scope: str):
    try:
        start = time.perf_counter()
        yield 
    finally:
        elapsed = time.perf_counter() - start
        print('%s takes %.2f ms' % (scope, elapsed * 1000))

def _time(f: callable):
    import torch 
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        f()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 10
    return elapsed

def make_example_input(policy, bsz: int):
    requests = [InferenceRequest(observation={
        'observation/image': np.zeros((224, 224, 3), dtype=np.uint8),
        'observation/wrist_image': np.zeros((224, 224, 3), dtype=np.uint8),
        'observation/state': np.zeros(8, dtype=np.float16),
        'prompt': 'do something useful',
    }) for _ in range(bsz)]

    observation = policy.preprocess(requests)
    embedding = policy.embed(observation)
    encoded = policy.encode(embedding)
    noise = policy._model.sample_noise(
            (bsz, policy._model.config.action_horizon, policy._model.config.action_dim), policy.device)
    decode_input = {**encoded, 'noise': noise}
    decoded = policy.decode(decode_input)
    actions = policy.postprocess(decoded)
    return observation, embedding, decode_input, decoded, actions

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).resolve().parents[1] / 'checkpoints/pi05_libero_pytorch')
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-sizes', type=int, nargs = '+', default=[1,2,4,8,16])
    parser.add_argument('--num-steps', type=int, default=1)
    parser.add_argument('--backend', type=str, choices=['openpi'], default='openpi')
    args = parser.parse_args()

    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error('--batch-sizes must be unique')
    if not (args.checkpoint / 'model.safetensors').is_file():
        parser.error('--checkpoint must contain model.safetensors (converted PyTorch weights)')

    policy_config = PolicyConfig(
        model_dir=str(args.checkpoint), num_steps=args.num_steps,
        batch_sizes=args.batch_sizes, max_batch_size=max(args.batch_sizes),
        backend=args.backend, use_cuda_graph=False,
    )
    policy = create_policy(policy_config, device=args.device)
    print('dtype', policy._model.config.dtype)
    
    observation_b1, embedding_b1, encoded_b1, decoded_b1, actions_b1 = make_example_input(policy, 1)

    def check(stage, actual, expected):
        actual_leaves, actual_spec = flatten_tree(actual)
        expected_leaves, expected_spec = flatten_tree(expected)
        assert actual_spec == expected_spec, f"{stage}: output structure differs"
        for actual_tensor, expected_tensor in zip(actual_leaves, expected_leaves):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-2, atol=2e-2)
        print(f"{stage}: compiled CUDA graph matches eager", flush=True)

    example_ios = {}
    # Explicit noise makes decoder comparisons deterministic.
    with torch.inference_mode(), Compiler(device=policy.device) as compiler:
        for bsz in args.batch_sizes:
            observation, embedding, encoded, decoded, actions = make_example_input(policy, bsz)
            example_ios[bsz] = {
                "embed": (observation, embedding),
                "encode": (embedding, {k: v for k, v in encoded.items() if k != 'noise'}),
                "decode": (encoded, decoded)
            }

        for name, function, batch_one_input in [
            ("embed", policy.embed, observation_b1),
            ("encode", policy.encode, embedding_b1),
            ("decode", policy.decode, encoded_b1),
        ]:
            compiled = compiler._compile(function, batch_one_input, args.batch_sizes)
            for bsz in args.batch_sizes:
                i, o = example_ios[bsz][name]
                check(name + ' bsz %s' % bsz, compiled(i), o)
                print('%s before compilation' % name, _time(lambda : function(i)))
                print('%s after compilation' % name, _time(lambda : compiled(i)))


if __name__ == '__main__':
    main()
