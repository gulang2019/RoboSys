import csv
import heapq
import json
import sys
import traceback
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from robort.schemas import InferenceRequest, InferenceResponse
from .envs import create_env, Environment, EnvConfig, EpisodeOutcome
from .robot import RobotConfig, create_robot, Robot

@dataclass
class PolicyServer:
    host: str
    port: int 
    _connector: object = field(init=False, repr=False)
    _pending: list = field(default_factory=list)
    _ready: list = field(default_factory = list)

    def __post_init__(self):
        from .serve import PolicyClient
        self._connector = PolicyClient(self.host, self.port)

    def schedule(self, env_id: int,
                obv: InferenceRequest, 
                obv_tick: int,
                deadline: int):
        heapq.heappush(self._pending, (deadline, env_id, obv_tick, obv))

    def maybe_run_inf(self, tick) -> list[tuple[int, int, InferenceResponse]]:
        if self._pending and tick >= self._pending[0][0]:
            deadlines, env_ids, obv_ticks, obvs = zip(*self._pending)
            self._pending.clear()
            results = self._connector.batch_inference(list(obvs))
            for deadline, env_id, obv_tick, result in zip(deadlines, env_ids, obv_ticks, results, strict=True):
                heapq.heappush(self._ready, (deadline, env_id, obv_tick, result))
        rets = []
        while self._ready and self._ready[0][0] <= tick:
            _, env_id, obv_tick, result = heapq.heappop(self._ready)
            rets.append((env_id, obv_tick, result))
        return rets 

def benchmark(env_config: EnvConfig,
            robot_configs: list[RobotConfig],
            n_episodes: int, 
            host: str, 
            port: str,
            output_path: str):

    if n_episodes < 1 or not robot_configs:
        raise ValueError('Expected positive episode count and at least one robot configuration')
    with ExitStack() as stack:
        np.random.seed(getattr(env_config, 'seed', 7))
        env = create_env(env_config)
        if hasattr(env, 'close'):
            stack.callback(env.close)

        def report_error(exc_type, exc, tb):
            if exc_type is not None:
                traceback.print_exception(exc_type, exc, tb)
                sys.stderr.flush()

        stack.push(report_error)

        s = PolicyServer(host, port)
        stack.callback(s._connector.close)
        progress = stack.enter_context(tqdm(total=n_episodes * len(robot_configs), desc='Episodes'))

        launched = [0 for _ in range(len(robot_configs))]
        n_finished = 0
        robots: list[Robot | None] = [None] * env.n_envs

        def hook():
            for eid, r in enumerate(robots):
                if r is not None:
                    r.update_phase(env.get_phase(eid))
            tick = env.get_tick()
            for eid, r in enumerate(robots):
                if r and r.need_sch_inf(tick):
                    r.maybe_sch_inf(tick, env.get_observation(eid))
            results = s.maybe_run_inf(tick)
            for env_id, obv_tick, infer_response in results:
                if robots[env_id] is None: continue
                robots[env_id].add_chunk(infer_response, obv_tick, tick)
            to_update = []
            for eid, r in enumerate(robots):
                if r is not None:
                    # the updated action
                    act = r.maybe_update_action(tick)
                    if act is not None: to_update.append((eid, act))
            env.update_actions(to_update)
        
        env.add_pre_action_hook(hook)

        def get_next_robo(env_id):
            try: 
                i, config = next((i, robot_config) for i, robot_config in enumerate(robot_configs) if launched[i] < n_episodes)
                episode_id = launched[i]
                launched[i] += 1
                return create_robot(config, 
                                    env_id = env_id,
                                    config_id = i,
                                    episode_id = episode_id,
                                    server = s, 
                                    tick_per_s = env.tick_per_s, 
                                    tick_per_act = env.tick_per_act,
                                    init_tick = env.get_tick(),
                                    output_path = output_path)
            except StopIteration:
                return None


        for i in range(env.n_envs):
            next_robot = get_next_robo(i)
            robots[i] = next_robot

        results = []
        while n_finished < n_episodes * len(robot_configs):
            _, _, terminated, truncated, _ = env.step()
            done = (terminated | truncated).cpu().numpy()
            for i in map(int, np.flatnonzero(done)):
                if robots[i] is not None:
                    res = env.reset(i)
                    robots[i].finish_episode(env.get_tick(), res)
                    robot_configs[robots[i].config_id].results.append(res)
                    n_finished += 1
                    robots[i] =  get_next_robo(i)
                    progress.update(1)
                    _save_results(output_path, env_config, robot_configs)
        _save_results(output_path, env_config, robot_configs)
        print(f'Results saved to {output_path}', flush=True)
    '''
    output_path/config.yaml
    output_path/summary.csv
    output_path/tracing/{config_id}_{episode_id}/metrics.json
    output_path/tracing/{config_id}_{episode_id}/visualization.html

    '''

def _save_results(output_path, env_config: EnvConfig, robot_configs: list[RobotConfig]):
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    episodes, rows = [], []
    for i, config in enumerate(robot_configs):
        count = len(config.results)
        total_elapsed_s = sum(x.duration_s for x in config.results)
        successes = sum(outcome.success for outcome in config.results)
        rows.append(dict(config_id=i,
                         successes=successes, episodes=count,
                         success_rate=successes / count if count else float('nan'),
                         success_tpt=successes / total_elapsed_s if total_elapsed_s > 0 else float('nan')))
        episodes.extend(dict(config_id=i, episode=j, **asdict(outcome))
                        for j, outcome in enumerate(config.results))
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['config_id', 'successes', 'episodes', 'success_rate', 'success_tpt'])
        writer.writeheader()
        writer.writerows(rows)
    (root / 'episodes.json').write_text(json.dumps(episodes, indent=2) + '\n')
    metadata = dict(environment=asdict(env_config),
                    robot_configs=[dict(type_=config.type_, kwargs=config.kwargs,
                                        enable_tracing=config.enable_tracing) for config in robot_configs],
                    action_timing='observation_relative_exact_start', inference_batching='earliest_deadline')
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

def _main(args=None):
    import argparse

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--config', type=Path)
    options, remaining = parser.parse_known_args(args)
    if options.config is not None:
        if remaining:
            parser.error('Use --config by itself; put benchmark settings in the configuration file')
        from .config import load_config
        benchmark(**load_config(options.config))
        return

    import tyro
    def cli(env_config: EnvConfig, robot_configs: str, n_episodes: int,
            host: str, port: str, output_path: str):
        """Use --config FILE for JSON/YAML, or pass --robot-configs as a JSON list."""
        configs = [RobotConfig(**config) for config in json.loads(robot_configs)]
        benchmark(env_config, configs, n_episodes, host, port, output_path)
        import yaml
        config = dict(env_config=asdict(env_config),
                      robot_configs=json.loads(robot_configs),
                      n_episodes=n_episodes, host=host, port=port,
                      output_path=output_path)
        with (Path(output_path) / 'config.yaml').open('w') as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    tyro.cli(cli, args=remaining)


if __name__ == '__main__':
    _main()
