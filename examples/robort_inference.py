"""Run decomposed OpenPI or FlashRT inference on synthetic LIBERO observations."""

import argparse
from pathlib import Path

import numpy as np

from robort.schemas import PolicyConfig, InferenceRequest
from robort.policies import create_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).resolve().parents[1] / 'checkpoints/pi05_libero_pytorch')
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--backend', type=str, choices=['openpi', 'flashrt'], default='openpi')
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 1 if args.backend == 'flashrt' else 2
    if args.backend == 'flashrt' and args.batch_size != 1:
        parser.error('FlashRT currently supports --batch-size 1 only')
    if args.batch_size < 1 or args.num_steps < 1:
        parser.error('--batch-size and --num-steps must be positive')
    if not (args.checkpoint / 'model.safetensors').is_file():
        parser.error('--checkpoint must contain model.safetensors (converted PyTorch weights)')

    policy_config = PolicyConfig(
        model_dir=str(args.checkpoint), num_steps=args.num_steps,
        batch_sizes={'encode': [args.batch_size], 'decode': [args.batch_size], 'embed': [args.batch_size]},
        backend=args.backend, use_cuda_graph=False,
    )
    policy = create_policy(policy_config, device=args.device)
    requests = [InferenceRequest(observation={
        'observation/image': np.zeros((224, 224, 3), dtype=np.uint8),
        'observation/wrist_image': np.zeros((224, 224, 3), dtype=np.uint8),
        'observation/state': np.zeros(8, dtype=np.float32),
        'prompt': 'do something useful',
    }) for _ in range(args.batch_size)]

    observation = policy.preprocess(requests)
    print('1. Preprocessed state:', tuple(observation.state.shape), flush=True)
    embedding = policy.embed(observation)
    if args.backend == 'flashrt':
        print('2. Vision embeddings:', embedding['vision_shape'], flush=True)
    else:
        print('2. Prefix embeddings:', tuple(embedding['prefix_embeds'].shape), flush=True)
    context = policy.encode(embedding)
    print('3. Encoded prefix cache', flush=True)
    outputs = policy.decode(context)
    print('4. Decoded actions:', tuple(outputs['actions'].shape), flush=True)
    responses = policy.postprocess(outputs)
    for response in responses:
        if not np.isfinite(response.actions).all():
            raise RuntimeError('Inference returned non-finite actions')
    print('5. Postprocessed actions:', responses[0].actions.shape)
    print(responses[0])


if __name__ == '__main__':
    main()
