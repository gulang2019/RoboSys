import logging
from pathlib import Path
import tempfile

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

logger = logging.getLogger(__name__)


def _save_score_cdf(score_history: dict[int, np.ndarray], path: Path,
                    xlabel: str = 'Visual embedding MSE') -> None:
    """Save the empirical CDF without requiring a display or pyplot state."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(6, 4))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    for step, scores in sorted(score_history.items()):
        values = np.sort(scores.ravel())
        probabilities = np.arange(1, values.size + 1) / values.size
        ax.step(np.r_[values[0], values], np.r_[0, probabilities], where='post', label=f'Step {step}')
    if score_history:
        ax.legend(fontsize=6, ncol=max(1, (len(score_history) + 24) // 25),
                  loc='upper left', bbox_to_anchor=(1, 1))
    else:
        ax.text(0.5, 0.5, 'No scored steps (initial full refresh only)', ha='center', transform=ax.transAxes)
    ax.set(xlabel=xlabel, ylabel='CDF', ylim=(0, 1), title=xlabel)
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches='tight')


def _text_attention_scores(attentions, indices, pad_mask, visual_tokens):
    """Average valid text queries over layers and heads, retaining visual key order."""
    queries = (indices >= visual_tokens) & pad_mask[0].index_select(0, indices).bool()
    scores = torch.zeros(visual_tokens, device=indices.device, dtype=torch.float32)
    if queries.any():
        for attention in attentions:
            scores += attention[0, :, queries, :visual_tokens].float().mean(dim=(0, 1))
        scores /= len(attentions)
    return scores * pad_mask[0, :visual_tokens]


def _save_pruning_images(images, scores: torch.Tensor | None, iteration: int,
                         indices: torch.Tensor | None = None, output: Path = Path('debug'),
                         attention_scores: torch.Tensor | None = None) -> None:
    """Shade 14x14 patches by score, using the same scale for both cameras."""
    from PIL import Image

    output.mkdir(parents=True, exist_ok=True)
    selected = None if indices is None else indices.detach().cpu().numpy()
    weights = None
    if scores is not None:
        values = scores.detach().float().cpu().numpy()
        maximum = values.max()
        weights = values / maximum if maximum > 0 else np.zeros_like(values)
    attention_weights = None
    if attention_scores is not None:
        values = attention_scores.detach().float().cpu().numpy()
        maximum = values.max()
        attention_weights = values / maximum if maximum > 0 else np.zeros_like(values)
    offset = 0
    for name, image in zip(('global', 'wrist'), images):
        pixels = image[0].detach().float().cpu()
        if pixels.shape[0] == 3:
            pixels = pixels.permute(1, 2, 0)
        pixels = ((pixels + 1) * 127.5).round().clamp(0, 255).numpy().astype(np.uint8)
        height, width = pixels.shape[:2]
        rows, cols = height // 14, width // 14
        count = rows * cols
        brightness = np.ones(count) if weights is None else 0.25 + 0.75 * weights[offset:offset + count]
        brightness = brightness.reshape(rows, cols).repeat(14, axis=0).repeat(14, axis=1)
        shaded = (pixels * brightness[..., None]).round().clip(0, 255).astype(np.uint8)
        keep = np.ones(count, dtype=bool) if selected is None else np.zeros(count, dtype=bool)
        if selected is not None:
            keep[selected[(selected >= offset) & (selected < offset + count)] - offset] = True
        keep = keep.reshape(rows, cols).repeat(14, axis=0).repeat(14, axis=1)
        Image.fromarray(pixels).save(output / f'{iteration}.{name}.png')
        Image.fromarray(shaded).save(output / f'{iteration}.{name}.masked.png')
        Image.fromarray(pixels * keep[..., None]).save(output / f'{iteration}.{name}.binary.png')
        if attention_weights is not None:
            brightness = 0.25 + 0.75 * attention_weights[offset:offset + count]
            brightness = brightness.reshape(rows, cols).repeat(14, axis=0).repeat(14, axis=1)
            attended = (pixels * brightness[..., None]).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(attended).save(output / f'{iteration}.{name}.attention.png')
        offset += count

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
        self._debug_output = None
        self._iter = 0
        self.reset()
        logger.info(f"initialize openpi policy with {policy_config}")

    def _embed_seq_len(self):
        return self._num_visual_tokens() + self.config.prompt_len

    def _num_visual_tokens(self):
        return {
            '224': 256
        }[self.config.image_resolution] * 3

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

        embedding = {
            'state': state,
            'prefix_att_2d_masks_4d': prefix_att_2d_masks_4d,
            'prefix_embeds': prefix_embs,
            'prefix_position_ids': prefix_position_ids,
            'prefix_pad_masks': prefix_pad_masks,
            'past_key_values': self._caches[batch_size]
        }
        if batch_size == 1 and self.config.encode_keep_rate is not None:
            embedding['debug_images'] = images[:2]
        return embedding

    @torch.no_grad()
    def encode(self, embedding: dict) -> dict:
        # The output borrows the supplied cache's storage.
        model = self._model.paligemma_with_expert.paligemma.language_model
        model.config._attn_implementation = "eager"
        prefix = embedding['prefix_embeds'] # [B, SeqLen, H]
        batch_size, prefix_length = prefix.shape[:2]

        attention_mask = embedding['prefix_att_2d_masks_4d']
        position_ids = embedding['prefix_position_ids']
        input_embeds = embedding['prefix_embeds']
        indices = torch.arange(prefix_length, device=prefix.device)
        if 'debug_images' in embedding and getattr(self, '_debug_output', None) is None:
            Path('debug').mkdir(exist_ok=True)
            self._debug_output = Path(tempfile.mkdtemp(prefix='run-', dir='debug'))
            logger.info('Pruning debug output: %s', self._debug_output)
        if batch_size == 1 and self.config.encode_keep_rate is not None:
            selected = self.prune_encode_bsz1(prefix[0])
            if selected is not None:
                indices = selected
                attention_mask = attention_mask.index_select(2, indices)
                position_ids = position_ids.index_select(1, indices)
                input_embeds = input_embeds.index_select(1, indices)

        # Explicit cache positions preserve unselected KV slots during partial refresh.
        debug_attention = batch_size == 1 and 'debug_images' in embedding
        output = model.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=embedding['past_key_values'],
            cache_position=indices,
            inputs_embeds=input_embeds,
            output_attentions=debug_attention,
            use_cache=True,
            adarms_cond=None,
        )
        if batch_size == 1 and self.config.encode_keep_rate is not None and self._prev_visual is None:
            # Keep the first successfully encoded frame as the debugging baseline.
            self._prev_visual = prefix[0, :self._num_visual_tokens()].detach().clone()
        if batch_size == 1 and self.config.encode_keep_rate is not None and self._pruning_scores is not None:
            self._score_history[self._iter] = self._pruning_scores.detach().float().cpu().numpy().copy()
        if batch_size == 1 and self.config.encode_keep_rate is not None and 'debug_images' in embedding:
            text_attn = _text_attention_scores(output.attentions, indices,
                                               embedding['prefix_pad_masks'], self._num_visual_tokens())
            self._attention_history[self._iter] = text_attn.detach().float().cpu().numpy().copy()
            _save_pruning_images(embedding['debug_images'], self._pruning_scores, self._iter,
                                 indices=indices, output=self._debug_output, attention_scores=text_attn)

        return {
            'state': embedding['state'],
            'prefix_pad_masks': embedding['prefix_pad_masks'],
            'past_key_values': embedding['past_key_values']
        }

    @torch.no_grad()
    def prune_encode_bsz1(self, prefix):
        """Select visual tokens from a [sequence, hidden] prefix; keep all language tokens."""
        if not 0 <= self.config.encode_keep_rate <= 1:
            raise ValueError('encode_keep_rate must be between 0 and 1')
        self._iter += 1
        S, _ = prefix.shape
        V = self._num_visual_tokens()
        if S < V:
            raise ValueError('Prefix is shorter than the configured visual token count')
        visual = prefix[:V]
        self._pruning_scores = None
        if self._prev_visual is None:
            return None
        n_to_keep = round(self.config.encode_keep_rate * V)
        scores = (visual.float() - self._prev_visual.float()).square().mean(dim=-1)
        self._pruning_scores = scores
        indices = scores.topk(k=n_to_keep).indices.sort().values
        lang_indices = torch.arange(V,S,device = prefix.device, dtype = torch.long)
        return torch.cat([indices, lang_indices])

    def reset(self):
        self._prev_visual = None
        self._pruning_scores = None
        # Keep filenames unique across episodes within a job.
        self._iter = getattr(self, '_iter', 0)
        self._score_history = getattr(self, '_score_history', {})
        self._attention_history = getattr(self, '_attention_history', {})

    def start_debug_episode(self, directory: str | Path) -> None:
        """Finish previous artifacts and start an explicitly named episode directory."""
        self.export_debug_videos()
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=False)
        self.reset()
        self._debug_output = output
        self._iter = 0
        self._score_history = {}
        self._attention_history = {}

    def export_debug_videos(self, fps: int = 10) -> None:
        """Encode the current job's camera PNGs, reading one frame at a time."""
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw

        output = getattr(self, '_debug_output', None)
        if output is None:
            return
        frames = sorted(output.glob('*.global.png'), key=lambda path: int(path.name.split('.')[0]))
        if frames:
            destination = output / 'combined.tmp.mp4'
            with imageio.get_writer(destination, fps=fps, codec='libx264', macro_block_size=1) as writer:
                for frame in frames:
                    iteration = frame.name.split('.')[0]
                    rows = []
                    for camera in ('global', 'wrist'):
                        tiles = []
                        for kind, suffix in (('full', ''), ('binary', '.binary'), ('shaded', '.masked'),
                                             ('text attention', '.attention')):
                            with Image.open(output / f'{iteration}.{camera}{suffix}.png') as image:
                                tile = Image.new('RGB', (image.width, image.height + 24), 'black')
                                tile.paste(image, (0, 24))
                                ImageDraw.Draw(tile).text((6, 5), f'{camera} / {kind}', fill='white')
                            tiles.append(np.asarray(tile))
                        rows.append(np.concatenate(tiles, axis=1))
                    writer.append_data(np.concatenate(rows, axis=0))
            cdf = output / 'scores.cdf.tmp.png'
            _save_score_cdf(getattr(self, '_score_history', {}), cdf)
            attention_cdf = output / 'attention.cdf.tmp.png'
            _save_score_cdf(getattr(self, '_attention_history', {}), attention_cdf,
                            xlabel='Text-to-image attention probability')
            destination.replace(output / 'combined.mp4')
            cdf.replace(output / 'scores.cdf.png')
            attention_cdf.replace(output / 'attention.cdf.png')
            # Remove intermediates only after the video and both CDFs exist.
            for artifact in output.iterdir():
                parts = artifact.name.split('.')
                is_frame = (parts[0].isdigit() and artifact.suffix == '.png'
                            and (len(parts) == 2 or (parts[1] in ('global', 'wrist')
                                 and (len(parts) == 3 or (len(parts) == 4 and parts[2] in ('masked', 'binary', 'attention'))))))
                is_video = artifact.name in {f'{camera}.{kind}.mp4'
                    for camera in ('global', 'wrist') for kind in ('full', 'binary', 'shaded')}
                if is_frame or is_video:
                    artifact.unlink()
            logger.info('Saved combined.mp4, scores.cdf.png and attention.cdf.png in %s', output)

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
