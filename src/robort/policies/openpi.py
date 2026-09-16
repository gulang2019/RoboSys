import torch
import numpy as np
import jax

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


    def preprocess(self, observations: list[InferenceRequest]) -> openpi.models.model.Observation:
        if not observations:
            raise ValueError("preprocess requires a nonempty batch")
        if len(observations) > self.config.max_batch_size:
            raise ValueError("Batch exceeds max_batch_size")
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

        return {
            'state': state,
            'prefix_att_2d_masks_4d': prefix_att_2d_masks_4d,
            'prefix_embeds': prefix_embs,
            'prefix_position_ids': prefix_position_ids,
            'prefix_pad_masks': prefix_pad_masks
        }

    @torch.no_grad()
    def encode(self, embedding: dict) -> dict:
        self._model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self._model.paligemma_with_expert.forward(
            attention_mask=embedding['prefix_att_2d_masks_4d'],
            position_ids=embedding['prefix_position_ids'],
            past_key_values=None,
            inputs_embeds=[embedding['prefix_embeds'], None],
            use_cache=True,
        )

        return {
            'state': embedding['state'],
            'prefix_pad_masks': embedding['prefix_pad_masks'],
            'past_key_values': past_key_values
        }

    @torch.no_grad()
    def decode(self, context: dict, noise: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        bsize = context['state'].shape[0]
        if noise is None:
            actions_shape = (bsize, self._model.config.action_horizon, self._model.config.action_dim)
            noise = self._model.sample_noise(actions_shape, self.device)

        num_steps = self.config.num_sample_steps
        if num_steps is None:
            num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=self.device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=self.device)
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
