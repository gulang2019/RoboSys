from .env import Environment, EpisodeOutcome, EnvConfig, PHASE_TYPE


def create_env(env_config: EnvConfig):
    if env_config.type_ in ('conveyor_belt_pnp', 'reflexbench'):
        from .reflexbench_adaptor import ReflexBenchAdaptor
        return ReflexBenchAdaptor(env_config)
    if env_config.type_ == 'libero':
        from .libero_adaptor import LIBEROAdaptor
        return LIBEROAdaptor(env_config)
    raise NotImplementedError
