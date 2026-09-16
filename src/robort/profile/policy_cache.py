"""Resident model and compiled-function cache; no per-run CUDA graphs live here."""

from copy import copy, deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from robort.policies import create_policy


class PolicyCache:
    """Keep one checkpoint/device resident, with compiled execution variants.

    Hardware power/SM settings and graph-vs-eager selection do not invalidate
    weights. Batch shapes reuse compiled functions, compiling each new shape once.
    A changed execution configuration gets a fresh policy view sharing the model.
    Switching checkpoint/device evicts the previous model to bound GPU residency.
    """

    def __init__(self):
        self._model_key = None
        self._model = None
        self._variants = {}

    def get(self, config, device, batch_sizes):
        config = deepcopy(config)
        device = torch.device(device)
        config.model_dir = str(Path(config.model_dir or f'checkpoints/{config.model_name}_pytorch').expanduser().resolve())
        model_key = (config.backend, config.model_name, config.model_dir, str(device))
        values = asdict(config)
        for key in ('use_cuda_graph', 'batch_sizes'):
            values.pop(key)
        variant_key = tuple(sorted(values.items()))
        # Optional backends can embed execution settings in their constructor.
        if config.backend != 'openpi':
            model_key += (variant_key,)
        with torch.cuda.device(device), torch.cuda.stream(torch.cuda.default_stream(device)):
            if model_key != self._model_key:
                self.close()
                model = create_policy(config, device=str(device))
                torch.cuda.current_stream().synchronize()
                self._model, self._model_key = model, model_key
            if variant_key not in self._variants:
                policy = copy(self._model)
                policy.config = config
                self._variants[variant_key] = [policy, None]
            entry = self._variants[variant_key]
            if config.use_cuda_graph:
                from robort.policies.openpi_cuda_graph import CompiledOpenPIStages
                if entry[1] is None:
                    entry[1] = CompiledOpenPIStages(entry[0])
                entry[1].prepare(batch_sizes)
            return entry[0], entry[1]

    def close(self):
        self._variants.clear()
        self._model = None
        self._model_key = None
