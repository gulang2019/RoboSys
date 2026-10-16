import numpy as np

from robort.schemas import InferenceResponse
from robort.utils import packb, unpackb


def serve(host: str = '127.0.0.1', port: int = 8010,
          model_dir: str = 'checkpoints/pi05_libero_pytorch',
          config_name: str | None = None):
    """Keep the pi05 model in its own Python/CUDA environment; accept batched requests."""
    import torch
    from websockets.sync.server import serve as websocket_serve
    from robort.policies import create_policy
    from robort.schemas import InferenceRequest, PolicyConfig
    from robort.utils import packb, unpackb

    if config_name:
        from openpi.policies.policy_config import create_trained_policy
        from openpi.training.config import get_config
        policy = create_trained_policy(get_config(config_name), model_dir)
    else:
        policy = create_policy(PolicyConfig(model_dir=model_dir, use_cuda_graph=False), 'cuda:0')

    def handle(connection):
        for message in connection:
            job = unpackb(message)
            if 'seed' in job:
                torch.manual_seed(job['seed'])
                if config_name and hasattr(policy, '_rng'):
                    import jax
                    policy._rng = jax.random.key(job['seed'])
                if not config_name:
                    policy.reset()
            if config_name:
                actions = []
                for i, obs in enumerate(job['observations']):
                    if 'observation_seeds' in job:
                        torch.manual_seed(job['observation_seeds'][i])
                        if hasattr(policy, '_rng'):
                            import jax
                            policy._rng = jax.random.key(job['observation_seeds'][i])
                    actions.append(policy.infer(obs)['actions'])
            else:
                actions = [response.actions for response in policy.infer(
                    [InferenceRequest(observation=obs) for obs in job['observations']])]
            connection.send(packb(np.stack(actions)))

    with websocket_serve(handle, host, port, compression=None, max_size=None) as server:
        print(f'pi05 ready on ws://{host}:{port}', flush=True)
        server.serve_forever()


class PolicyClient:
    def __init__(self, host, port):
        from websockets.sync.client import connect
        self.connection = connect(f'ws://{host}:{port}', compression=None, max_size=None)
        self.seeded = False

    def batch_inference(self, requests):
        job = {'observations': [request.observation for request in requests]}
        if not self.seeded:
            job['seed'] = requests[0].observation.get('seed', 7)
        self.connection.send(packb(job))
        actions = unpackb(self.connection.recv())
        if (not isinstance(actions, np.ndarray) or actions.ndim != 3
                or len(actions) != len(requests) or not np.isfinite(actions).all()):
            raise ValueError('Invalid batched policy response')
        self.seeded = True
        return [InferenceResponse(actions=row) for row in actions]

    def close(self):
        self.connection.close()




if __name__ == '__main__':
    import tyro
    tyro.cli(serve)
