"""Profile the public five-stage policy interface in dependency order."""

from contextlib import ExitStack
from copy import deepcopy
from math import ceil
from dataclasses import asdict
import gc
import numpy as np
import torch

from robort.profile.schemas import HardwareConfig, HardwareProfile, PolicyConfig, PolicyProfile, RunnerConfig, StageProfile
from robort.profile.profiler import Profiler
from robort.profile.hardware import profile_hardware, prepare_hardware_env
from robort.profile.environment import validate_hardware_config, _green_stream
from ..policies.vla_base import VLABasePolicy
import tqdm

class Runner:
    """Synchronous profiler reusing one resident policy per initialized backend."""

    stages = ('preprocess', 'embed', 'encode', 'decode', 'postprocess')

    def __init__(self, args: RunnerConfig):
        self.args = args
        self.profiler = Profiler()
        self.backends = {}
        self._configs = {}
        self._compilers = {}
        self._streams = {}
        self._resources = ExitStack()

    def profile_hardware(self, hardware_config: HardwareConfig) -> HardwareProfile:
        return profile_hardware(hardware_config)

    def init_backend(self, policy_config, hardware_configs):
        """Load once and capture all batch/stream combinations before benchmarking."""
        import torch
        from robort.policies import create_policy

        config = deepcopy(policy_config)
        if config.backend in self.backends:
            previous_compiler = self._compilers.pop(config.backend, None)
            if previous_compiler is not None:
                previous_compiler.close()
            del self.backends[config.backend]
            self._configs.pop(config.backend, None)
            gc.collect()
        if config.use_cuda_graph and config.backend not in ('openpi', 'flash_rt', 'flashrt'):
            raise NotImplementedError('The Torch compiler currently supports OpenPI only')
        devices = {hw._device_index for hw in hardware_configs}
        if len(devices) != 1:
            raise ValueError('A resident backend requires one CUDA device')
        device = torch.device('cuda', devices.pop())
        config.model_dir = config.model_dir or f'checkpoints/{config.model_name}_pytorch'
        streams = {}
        for hw in hardware_configs:
            validate_hardware_config(hw)
            key = (hw._device_index, hw.sm_perc)
            if key not in self._streams:
                self._streams[key] = self._resources.enter_context(
                    _green_stream(torch, hw._device_index, ceil(hw.num_sms * hw.sm_perc)))
            streams[key] = self._streams[key]
        compiler = None
        try:
            with torch.cuda.device(device), torch.cuda.stream(torch.cuda.default_stream(device)):
                policy = create_policy(config, device=str(device))
                if config.use_cuda_graph and config.backend == 'openpi':
                    from robort.policies.compiler_utils import Compiler
                    compiler = Compiler(device=device, streams=list(streams.values()),
                                        use_torch_compile=config.use_torch_compile)
                    policy = compiler.compile(policy)
                torch.cuda.current_stream().synchronize()
        except BaseException:
            if compiler is not None:
                compiler.close()
            raise
        self.backends[config.backend] = policy
        self._configs[config.backend] = deepcopy(config)
        if compiler is not None:
            self._compilers[config.backend] = compiler


    @torch.inference_mode()
    def _profile_batch(self, policy: VLABasePolicy, hardware_config: HardwareConfig, stream):
        policy_profile = PolicyProfile(deepcopy(hardware_config), deepcopy(policy.config), {})
        with ExitStack() as cleanup:
            targets = None
            if policy.config.backend in ('flash_rt', 'flashrt'):
                targets = policy.profile_functions(stream)
                for target in targets.values():
                    cleanup.callback(target.close)
            for stage, batch_sizes in tqdm.tqdm(policy.config.batch_sizes.items(), desc='Stages'):
                if stage not in self.stages:
                    raise ValueError(f'Unknown profiling stage: {stage}')
                stage_profile = StageProfile()
                for bsz in tqdm.tqdm(batch_sizes, desc=f'{stage} batch sizes'):
                    if type(bsz) is not int or bsz < 1:
                        raise ValueError('batch_sizes must contain positive integers')
                    if targets is not None:
                        model_fn = targets[stage]
                    else:
                        example_input = policy.make_example_input(bsz, stage=stage)
                        model_fn = lambda: getattr(policy, stage)(example_input)
                    for _ in range(max(1, self.args.num_warmup)):
                        model_fn()
                    with self.profiler.measure(stream=stream) as p:
                        for _ in range(self.args.num_iter):
                            model_fn()
                            p.tick()
                    stage_profile.lat[bsz] = (float(np.mean(p.latencies)), float(np.std(p.latencies)))
                    stage_profile.energy[bsz] = (float(np.mean(p.energies)), float(np.std(p.energies)))
                    stage_profile.mem_fp_activation_gb[bsz] = (
                        (float('nan'), float('nan')) if targets is not None else
                        (float(np.mean(p.memories)), float(np.std(p.memories))))
                policy_profile.stages[stage] = stage_profile
        return policy_profile

    def profile_policy(self,
                       hardware_config: HardwareConfig,
                       policy_config: PolicyConfig) -> PolicyProfile:
        """Return selected stage measurements per batch size (default [1]).

        Model loading, compilation, capture and warmup are outside measurement.
        Policies, compiled graphs and streams persist across runs until close().
        Timings include
        public-method CPU work/transfers/synchronization. Unknown metadata is NaN.
        """
        validate_hardware_config(hardware_config)
        for name, minimum in (("num_warmup", 0), ("num_iter", 1)):
            value = getattr(self.args, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(policy_config.use_cuda_graph) is not bool:
            raise ValueError('use_cuda_graph must be a boolean')
        if (policy_config.use_cuda_graph
                and policy_config.backend not in ('openpi', 'flash_rt', 'flashrt')):
            raise NotImplementedError(
                f'CUDA graph profiling does not support backend {policy_config.backend!r}')
        config = deepcopy(policy_config)
        if config.model_dir is None:
            config.model_dir = f"checkpoints/{config.model_name}_pytorch"
        if config.backend not in self.backends:
            raise ValueError(f'Call init_backend for {config.backend!r} before profiling')
        initialized = self._configs[config.backend]
        for name, value in asdict(config).items():
            if name not in ('batch_sizes', 'max_batch_size') and value != getattr(initialized, name):
                raise ValueError(f'{name} differs from the initialized {config.backend} policy')
        policy = self.backends[config.backend]
        stream = self._streams[(hardware_config._device_index, hardware_config.sm_perc)]
        with prepare_hardware_env(hardware_config, stream=stream):
            try:
                return self._profile_batch(policy, hardware_config, stream)
            finally:
                stream.synchronize()

    def close(self):
        try:
            for compiler in self._compilers.values():
                compiler.close()
        finally:
            self._compilers.clear()
            self.backends.clear()
            self._configs.clear()
            self._resources.close()
            self._streams.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
