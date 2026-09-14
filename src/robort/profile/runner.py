from robort.profile.schemas import * 
from robort.profile.profiler import Profiler
from robort.profile.policies import PolicyManager

class Runner:
    def __init__(self, args: RunnerConfig):
        self.args = args
        self.profiler = Profiler()
        self.policy_manager = PolicyManager(policy_config=None)

    def prepare_hardware_config(self, hardware_config: HardwareConfig) -> 'contextlib':
        '''
        Set the hardware configuration for profiling.
        The return is the stream id.
        '''
        self.hardware_config = hardware_config
        # TODO: implement the logic. 
        # TODO: 
        return 0


    def profile_hardware(
            self,
            hardware_config: HardwareConfig) -> HardwareProfile:
        ... 

    def profile_policy(
            self,
            hardware_config: HardwareConfig,
            policy_config: PolicyConfig) -> PolicyProfile:
        with self.set_hardware_config(hardware_config=hardware_config):
            policies = self.policy_manager.prepare_policies(
                policy_config=policy_config)

            policy_profile = PolicyProfile()

            for model_name, model_fn in policies.items():
                for _ in range(self.args.num_warmup):
                    model_fn()

                with self.profiler as p:
                    for _ in range(self.args.num_iter):
                        model_fn()
                        p.tick()
                lat_distr, energy_distr = p.to_distr()

                policy_profile.__dict__[f'{model_name}_latency_ms'] = lat_distr
                policy_profile.__dict__[f'{model_name}_energy_j'] = energy_distr
        
        return policy_profile 