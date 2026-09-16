"""Profile the public five-stage policy interface in dependency order."""

from contextlib import contextmanager
from copy import deepcopy

from robort.profile.schemas import HardwareConfig, HardwareProfile, PolicyConfig, PolicyProfile, RunnerConfig, StageProfile
from robort.profile.profiler import Profiler
from robort.profile.hardware import profile_hardware, prepare_hardware_env
from robort.profile.environment import validate_hardware_config
from robort.profile.policy_cache import PolicyCache


class Runner:
    """Reusable, synchronous profiler. Call close() to evict resident models."""

    def __init__(self, args: RunnerConfig):
        self.args = args
        self.profiler = Profiler()
        self.policy_cache = PolicyCache()

    def profile_hardware(self, hardware_config: HardwareConfig) -> HardwareProfile:
        return profile_hardware(hardware_config)

    @contextmanager
    def _execution(self, config, stream, sizes):
        policy, compiled = self.policy_cache.get(config, stream.device, sizes)
        if config.use_cuda_graph:
            from robort.policies.openpi_cuda_graph import CudaGraphOpenPIPolicy
            with CudaGraphOpenPIPolicy(policy, stream, batch_sizes=sizes,
                                       compiled_stages=compiled) as graphed:
                yield graphed
        else:
            yield policy

    @staticmethod
    def make_model_functions(policy, batch_size):
        """Ordered calls sharing live outputs; native handles are never copied."""
        inputs = [policy.make_example_input() for _ in range(batch_size)]
        values = {}

        def preprocess():
            values.clear()
            values['observation'] = policy.preprocess(inputs)

        def embed():
            values['embedding'] = policy.embed(values.pop('observation'))

        def encode():
            values['context'] = policy.encode(values.pop('embedding'))

        def decode():
            values['actions'] = policy.decode(values.pop('context'))

        def postprocess():
            values['responses'] = policy.postprocess(values.pop('actions'))

        return dict(preprocess=preprocess, embed=embed, encode=encode,
                    decode=decode, postprocess=postprocess)

    def _profile_batch(self, policy, size, hardware_config, policy_config, stream):
        model_fns = self.make_model_functions(policy, size)
        model_fn = None
        try:
            if not model_fns:
                raise ValueError('Policy returned no profiling stages')
            # Initialize lazy eager state even when num_warmup=0.
            for _ in range(max(1, self.args.num_warmup)):
                for model_fn in model_fns.values():
                    model_fn()
            samples = {name: ([], []) for name in model_fns}
            for _ in range(self.args.num_iter):
                for name, model_fn in model_fns.items():
                    with self.profiler.measure(stream=stream) as p:
                        model_fn()
                        p.tick()
                    latencies, energies = samples[name]
                    latencies.extend(p.latencies)
                    energies.extend(p.energies)
            config = deepcopy(policy_config)
            config.batch_sizes = [size]
            result = PolicyProfile(deepcopy(hardware_config), config, {})
            metadata = StageProfile(*([float('nan')] * 8))
            for name, (latencies, energies) in samples.items():
                self.profiler.latencies = latencies
                self.profiler.energies = energies
                result.stages[name] = self.profiler.to_profile(metadata)
            return result
        finally:
            model_fn = None
            model_fns.clear()

    def profile_policy(self, hardware_config: HardwareConfig,
                       policy_config: PolicyConfig) -> dict[int, PolicyProfile]:
        """Return one five-stage profile per configured batch size (default [1]).

        Model loading, compilation, capture and warmup are outside measurement.
        Models/compiled functions persist in policy_cache; graphs are recreated
        on each green stream and closed before its destruction. Timings include
        public-method CPU work/transfers/synchronization. Unknown metadata is NaN.
        """
        validate_hardware_config(hardware_config)
        for name, minimum in (("num_warmup", 0), ("num_iter", 1)):
            value = getattr(self.args, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        sizes = policy_config.batch_sizes if policy_config.batch_sizes is not None else [1]
        if not sizes or any(type(n) is not int or not 1 <= n <= policy_config.max_batch_size for n in sizes):
            raise ValueError('batch_sizes must contain positive integers <= max_batch_size')
        if type(policy_config.use_cuda_graph) is not bool:
            raise ValueError('use_cuda_graph must be a boolean')
        if policy_config.use_cuda_graph and policy_config.backend != 'openpi':
            raise NotImplementedError('CUDA graph profiling currently supports the openpi backend only')
        config = deepcopy(policy_config)
        sizes = sorted(set(sizes))
        with prepare_hardware_env(hardware_config) as stream:
            try:
                with self._execution(config, stream, sizes) as policy:
                    return {size: self._profile_batch(policy, size, hardware_config, config, stream)
                            for size in sizes}
            finally:
                stream.synchronize()

    def close(self):
        self.policy_cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
