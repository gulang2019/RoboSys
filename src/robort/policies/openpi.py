import torch
import numpy as np
import jax
from transformers.cache_utils import StaticCache

import openpi
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks, PI0Pytorch

from .vla_base import VLABasePolicy
from ..schemas import PolicyConfig, InferenceRequest, InferenceResponse

def _recursive_stack(list_of_dicts: list[dict]) -> dict:
    """Recursively stack a list of dicts-of-arrays into a single dict-of-batched-arrays."""
    result = {}
    for key in list_of_dicts[0]:
        values = [d[key] for d in list_of_dicts]
        if isinstance(values[0], dict):
            result[key] = _recursive_stack(values)
        elif isinstance(values[0], np.ndarray):
            result[key] = np.stack(values, axis=0)
        else:
            # scalars (np.bool_, np.float32, etc.) — convert to array
            result[key] = np.asarray(values)
    return result


class _StaticPrefixCache(StaticCache):
    """Fixed-capacity storage exposing only the current request's prefix."""

    prefix_length = 0

    def __getitem__(self, layer_idx):
        return (self.key_cache[layer_idx][:, :, :self.prefix_length],
                self.value_cache[layer_idx][:, :, :self.prefix_length])

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        super().update(key_states, value_states, layer_idx, cache_kwargs)
        return self[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.prefix_length

    def to_legacy_cache(self):
        return tuple(self[i] for i in range(len(self.key_cache)))

class OpenPIPolicy(VLABasePolicy):
    """Synchronous PyTorch inference split into independently callable stages."""

    def __init__(self, policy_config: PolicyConfig, device: str | None = None):
        super().__init__(policy_config, device)
        num_steps = policy_config.num_sample_steps
        if num_steps is None:
            num_steps = policy_config.num_steps
        if num_steps < 1:
            raise ValueError("num_sample_steps must be positive")
        config = get_config(policy_config.model_name)
        self.policy = create_trained_policy(
            config,
            policy_config.model_dir,
            sample_kwargs={'num_steps': num_steps},
            pytorch_device=self.device,
        )
        self._model = self.policy._model
        if not isinstance(self._model, PI0Pytorch):
            raise TypeError("Decomposed inference requires a PyTorch checkpoint containing model.safetensors")
        self._model.eval()
        self._caches = {}  # Allocate once per batch size on first use.


    def preprocess(self, observations: list[InferenceRequest]) -> openpi.models.model.Observation:
        if not observations:
            raise ValueError("preprocess requires a nonempty batch")
        if any(request.inference_type != "sync" for request in observations):
            raise ValueError("Decomposed OpenPI inference only supports sync requests")
        inputs = []
        for request in observations:
            # Transforms can modify the dictionary structure.
            observation = jax.tree.map(lambda x: x, request.observation)
            inputs.append(self.policy._input_transform(observation))
        batched = _recursive_stack(inputs)
        batched = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self.device), batched
        )
        return openpi.models.model.Observation.from_dict(batched)


    @torch.no_grad()
    def embed(self, observation: openpi.models.model.Observation) -> dict:

        images, img_masks, lang_tokens, lang_masks, state = self._model._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self._model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._model._prepare_attention_masks_4d(prefix_att_2d_masks)
        batch_size, prefix_length = prefix_embs.shape[:2]
        model = self._model.paligemma_with_expert.paligemma.language_model
        if prefix_length > model.config.max_position_embeddings:
            raise ValueError('Prefix exceeds the model cache capacity')
        if batch_size not in self._caches:
            self._caches[batch_size] = _StaticPrefixCache(
                model.config, max_batch_size=batch_size,
                max_cache_len=model.config.max_position_embeddings,
                device=prefix_embs.device, dtype=model.layers[0].self_attn.k_proj.weight.dtype,
            )
        self._caches[batch_size].prefix_length = prefix_length

        return {
            'state': state,
            'prefix_att_2d_masks_4d': prefix_att_2d_masks_4d,
            'prefix_embeds': prefix_embs,
            'prefix_position_ids': prefix_position_ids,
            'prefix_pad_masks': prefix_pad_masks,
            'past_key_values': self._caches[batch_size]
        }

    @torch.no_grad()
    def encode(self, embedding: dict) -> dict:
        # The output borrows the supplied cache's storage.
        model = self._model.paligemma_with_expert.paligemma.language_model
        model.config._attn_implementation = "eager"
        prefix = embedding['prefix_embeds']
        batch_size, prefix_length = prefix.shape[:2]

        # Call the language model directly: the OpenPI wrapper does not expose
        # cache_position. Always overwrite from zero; the cache adapter hides
        # unused slots, so no reset or zero-fill is needed.
        model.forward(
            attention_mask=embedding['prefix_att_2d_masks_4d'],
            position_ids=embedding['prefix_position_ids'],
            past_key_values=embedding['past_key_values'],
            cache_position=torch.arange(prefix_length, device=prefix.device),
            inputs_embeds=embedding['prefix_embeds'],
            use_cache=True,
            adarms_cond=None,
        )

        return {
            'state': embedding['state'],
            'prefix_pad_masks': embedding['prefix_pad_masks'],
            'past_key_values': embedding['past_key_values']
        }

    @torch.no_grad()
    def decode(self, context: dict) -> dict[str, torch.Tensor]:
        bsize = context['state'].shape[0]
        noise = context.get('noise', None)
        if noise is None:
            actions_shape = (bsize, self._model.config.action_horizon, self._model.config.action_dim)
            noise = self._model.sample_noise(actions_shape, self.device)

        num_steps = self.config.num_sample_steps
        if num_steps is None:
            num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        dt = torch.full((), dt, dtype=torch.float32, device=self.device)

        x_t = noise
        time = torch.ones((), dtype=torch.float32, device=self.device)
        for _ in range(num_steps):
            expanded_time = time.expand(bsize)
            v_t = self._model.denoise_step(
                context['state'],
                context['prefix_pad_masks'],
                context['past_key_values'],
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return {
            'state': context['state'],
            'actions': x_t
        }

    def postprocess(self, outputs: dict[str, torch.Tensor]) -> list[InferenceResponse]:
        bsz = outputs['actions'].shape[0]
        rets = []
        for i in range(bsz):
            output = jax.tree.map(lambda x: np.asarray(x[i, ...].detach().cpu()), outputs)
            output = self.policy._output_transform(output)
            rets.append(InferenceResponse(actions=output['actions']))
        return rets

    def infer(self, observations):
        return super().infer(observations) if observations else []

    def make_example_input(self, bsz: int, stage='all'):
        if stage not in ('all', 'preprocess', 'embed', 'encode', 'decode', 'postprocess'):
            raise ValueError(f'Unknown stage: {stage}')
        requested_bsz = bsz
        if stage not in ('all', 'preprocess'):
            bsz = 1  # Build upstream context once, even for large decode batches.

        def example(tree):
            from .compiler_utils import flatten_tree, reconstruct
            leaves, spec = flatten_tree(tree)
            return reconstruct([
                t.expand(requested_bsz, *t.shape[1:]).clone()
                if isinstance(t, torch.Tensor) and t.ndim else t
                for t in leaves
            ], spec)

        def run(name, inputs):
            method = getattr(self, name)
            # Example preparation must not require an upstream graph for this batch.
            if stage != 'all':
                method = getattr(method, 'function', method)
            return method(inputs)

        requests = [InferenceRequest(observation={
            'observation/image': np.zeros((224, 224, 3), dtype=np.uint8),
            'observation/wrist_image': np.zeros((224, 224, 3), dtype=np.uint8),
            'observation/state': np.zeros(8, dtype=np.float16),
            'prompt': 'do something useful',
        }) for _ in range(bsz)]
        if stage == 'preprocess':
            return requests
    
        observation = self.preprocess(requests)
        if stage == 'embed':
            return example(observation)
        embedding = run('embed', observation)
        if stage == 'encode':
            return example(embedding)
        encoded = run('encode', embedding)
        noise = self._model.sample_noise(
                (bsz, self._model.config.action_horizon, self._model.config.action_dim), self.device)
        decode_input = {**encoded, 'noise': noise}
        if stage == 'decode':
            return example(decode_input)
        decoded = run('decode', decode_input)
        if stage == 'postprocess':
            return example(decoded)
        actions = self.postprocess(decoded)
        return requests, observation, embedding, decode_input, decoded, actions
