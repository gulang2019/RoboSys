"""Synchronous profiling of the FlashRT five-stage policy interface."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from math import ceil
import gc

import numpy as np
import torch
import tqdm

from robort.profile.schemas import HardwareConfig, HardwareProfile, PolicyConfig, PolicyProfile, RunnerConfig, StageProfile
from robort.profile.profiler import Profiler
from robort.profile.hardware import profile_hardware
from robort.profile.environment import validate_hardware_config, prepare_hardware_env, _green_stream


class Runner:
    """Own one FlashRT policy and its green streams; calls are serialized."""

    stages = ('preprocess', 'embed', 'encode', 'decode', 'postprocess')
    gpu_stages = ('embed', 'encode', 'decode')

    def __init__(self, args: RunnerConfig):
        self.args = args
        self.profiler = Profiler()
        self.backends = {}
        self._configs = {}
        self._streams = {}
        self._resources = ExitStack()

    def profile_hardware(self, hardware_config: HardwareConfig) -> HardwareProfile:
        return profile_hardware(hardware_config)

    def _config(self, policy_config):
        config = deepcopy(policy_config)
        if config.backend not in ('flash_rt', 'flashrt'):
            raise ValueError('profiling currently supports only the flash_rt backend')
        config.backend = 'flash_rt'
        config.model_dir = config.model_dir or f'checkpoints/{config.model_name}_pytorch'
        if type(config.use_cuda_graph) is not bool:
            raise ValueError('use_cuda_graph must be a boolean')
        if config.batch_sizes is None:
            config.batch_sizes = {stage: [1] for stage in self.gpu_stages}
        if not isinstance(config.batch_sizes, dict) or not config.batch_sizes:
            raise ValueError('batch_sizes must be a nonempty stage-to-sizes mapping')
        for stage, sizes in config.batch_sizes.items():
            if stage not in self.stages:
                raise ValueError(f'Unknown profiling stage: {stage}')
            if not isinstance(sizes, (list, tuple)) or not sizes or any(
                type(size) is not int or size < 1 for size in sizes
            ):
                raise ValueError('batch_sizes must contain positive integer sizes')
            config.batch_sizes[stage] = sorted(set(sizes))
        for name, minimum in (('num_warmup', 0), ('num_iter', 1)):
            value = getattr(self.args, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        return config

    def init_backend(self, policy_config, hardware_configs):
        """Load and capture all requested GPU batch/stream variants once."""
        from robort.policies import create_policy

        config = self._config(policy_config)
        hardware_configs = list(hardware_configs)
        if not hardware_configs:
            raise ValueError('hardware_configs must not be empty')
        device_index = hardware_configs[0]._device_index
        for hw in hardware_configs:
            validate_hardware_config(hw)
            if hw._device_index != device_index:
                raise ValueError('one initialized FlashRT policy requires a single CUDA device')
        self._release_backend()
        streams = []
        for hw in hardware_configs:
            key = (hw._device_index, hw.sm_perc)
            if key not in self._streams:
                self._streams[key] = self._resources.enter_context(
                    _green_stream(torch, hw._device_index, ceil(hw.num_sms * hw.sm_perc)))
            if self._streams[key] not in streams:
                streams.append(self._streams[key])
        # CPU stages are measurable but have no FlashRT pipeline/graph registry.
        backend_config = deepcopy(config)
        backend_config.batch_sizes = {stage: config.batch_sizes.get(stage, [1]) for stage in self.gpu_stages}
        device = torch.device('cuda', device_index)
        policy = create_policy(backend_config, device=device,
                               streams={stage: list(streams) for stage in self.gpu_stages})
        self.backends[config.backend] = policy
        self._configs[config.backend] = config

    def _example_input(self, policy, bsz, stage, stream):
        """Assemble a stage batch using only supported upstream batch sizes."""
        items = policy.make_example_input(bsz, stage='preprocess', stream=stream)
        retained = [items]
        try:
            for upstream in self.stages[:self.stages.index(stage)]:
                if upstream == 'preprocess':
                    items = policy.preprocess(items, stream=stream)
                else:
                    sizes = policy.config.batch_sizes[upstream]
                    outputs = []
                    offset = 0
                    while offset < len(items):
                        remaining = len(items) - offset
                        size = max((b for b in sizes if b <= remaining), default=min(sizes))
                        chunk = items[offset:offset + size]
                        count = len(chunk)
                        # Padding is preparation only; discard padded outputs.
                        chunk += [chunk[-1]] * (size - count)
                        outputs.extend(getattr(policy, upstream)(chunk, stream=stream)[:count])
                        offset += count
                    items = outputs
                retained.append(items)
            return items
        finally:
            # Keep all upstream tensors alive until their async consumers finish.
            # Preparation and this wait are outside the measurement interval.
            stream.synchronize()

    @torch.inference_mode()
    def _profile_batch(self, policy, hardware_config, stream, config=None):
        config = config if config is not None else policy.config
        result = PolicyProfile(deepcopy(hardware_config), deepcopy(config), {})
        for stage, sizes in tqdm.tqdm(config.batch_sizes.items(), desc='Stages'):
            measurements = StageProfile()
            for bsz in tqdm.tqdm(sizes, desc=f'{stage} batch sizes'):
                example = self._example_input(policy, bsz, stage, stream)
                operation = getattr(policy, stage)
                for _ in range(self.args.num_warmup):
                    operation(example, stream=stream)
                with self.profiler.measure(stream=stream) as measured:
                    for _ in range(self.args.num_iter):
                        operation(example, stream=stream)
                        measured.tick()
                for field, samples in (('lat', measured.latencies), ('energy', measured.energies),
                                       ('mem_fp_activation_gb', measured.memories)):
                    getattr(measurements, field)[bsz] = (float(np.mean(samples)), float(np.std(samples)))
            result.stages[stage] = measurements
        return result

    def profile_policy(self, hardware_config: HardwareConfig, policy_config: PolicyConfig) -> PolicyProfile:
        """Measure public-stage wall latency, including copies and completion.

        Loading, capture, input preparation and warmup are outside measurement.
        Only requested stages/sizes are measured; GPU sizes must be initialized.
        """
        validate_hardware_config(hardware_config)
        config = self._config(policy_config)
        if config.backend not in self.backends:
            raise ValueError('Call init_backend before profiling')
        initialized = self._configs[config.backend]
        for name, value in asdict(config).items():
            if name != 'batch_sizes' and value != getattr(initialized, name):
                raise ValueError(f'{name} differs from the initialized policy')
        policy = self.backends[config.backend]
        for stage, sizes in config.batch_sizes.items():
            if stage in self.gpu_stages and not set(sizes) <= set(policy.config.batch_sizes[stage]):
                raise ValueError(f'uninitialized batch size for {stage}')
        key = (hardware_config._device_index, hardware_config.sm_perc)
        if key not in self._streams:
            raise ValueError('hardware stream was not initialized')
        stream = self._streams[key]
        if any(stream not in policy.streams[stage] for stage in self.gpu_stages):
            raise ValueError('hardware stream was not initialized for this policy')
        with prepare_hardware_env(hardware_config, stream=stream):
            try:
                return self._profile_batch(policy, hardware_config, stream, config)
            finally:
                stream.synchronize()

    def _release_backend(self):
        for stream in self._streams.values():
            stream.synchronize()
        try:
            while self.backends:
                _, policy = self.backends.popitem()
                try:
                    policy.policy.close()
                finally:
                    del policy
        finally:
            self.backends.clear()
            self._configs.clear()
            gc.collect()

    def close(self):
        try:
            self._release_backend()
        finally:
            self._resources.close()
            self._streams.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
