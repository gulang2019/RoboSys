from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from robort.schemas import InferenceRequest
from .env import Environment, EpisodeOutcome, EnvConfig


@dataclass
class LIBEROConfig(EnvConfig):
    type_: str = 'libero'
    action_per_s: int = 20
    tick_per_s: int = 500
    task_suite_name: str = 'libero_spatial'
    task_id: int = 0
    num_envs: int = 1
    seed: int = 7
    resize_size: int = 224
    resolution: int = 256
    num_steps_wait: int = 10
    max_episode_steps: int | None = None


class LIBEROAdaptor(Environment):
    """Run LIBERO slots in lockstep, with one hook before every MuJoCo tick."""

    def __init__(self, config: LIBEROConfig):
        if not isinstance(config, LIBEROConfig):
            config = LIBEROConfig(**vars(config))
        super().__init__(config)
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        if config.num_envs < 1 or config.tick_per_s < 1 or config.action_per_s < 1:
            raise ValueError('Environment count and frequencies must be positive')
        self.tick_per_act
        limits = {'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300,
                  'libero_10': 520, 'libero_90': 400}
        self._max_steps = (limits[config.task_suite_name] if config.max_episode_steps is None
                           else config.max_episode_steps)
        if self._max_steps < 1 or config.num_steps_wait < 0:
            raise ValueError('Episode limit must be positive and warmup nonnegative')
        suite = benchmark.get_benchmark_dict()[config.task_suite_name]()
        task = suite.get_task(config.task_id)
        self._prompt = str(task.language)
        self._initial_states = suite.get_task_init_states(config.task_id)
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        self._envs = []
        self._hooks = []
        self._tick = 0
        self._next_episode = 0
        self._episodes = [0] * config.num_envs
        self._steps = [0] * config.num_envs
        self._actions = np.tile(np.array([0.] * 6 + [-1.]), (config.num_envs, 1))
        self._changed = [True] * config.num_envs
        self._outcomes = [EpisodeOutcome(i, False, False, False) for i in range(config.num_envs)]
        try:
            for i in range(config.num_envs):
                env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                    camera_heights=config.resolution, camera_widths=config.resolution,
                    control_freq=config.action_per_s, hard_reset=False,
                    horizon=self._max_steps + config.num_steps_wait + 1)
                self._envs.append(env)
                env.seed(config.seed + i)
                self.reset(i)
        except BaseException:
            self.close()
            raise

    @property
    def n_envs(self) -> int:
        return len(self._envs)

    def get_tick(self) -> int:
        return self._tick

    def add_pre_action_hook(self, fn):
        self._hooks.append(fn)

    def update_actions(self, actions: list[tuple[int, np.ndarray]]):
        for env_id, action in actions:
            action = np.asarray(action)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError('LIBERO actions must be seven finite values')
            self._actions[env_id] = action
            self._changed[env_id] = True

    def step(self):
        # This is robosuite's step loop, shared across slots so the hook sees
        # all robots at the same physics tick, including between control steps.
        active = [not (o.terminated or o.truncated) for o in self._outcomes]
        for i, env in enumerate(self._envs):
            if active[i]:
                env.env.timestep += 1
        for substep in range(self.tick_per_act):
            for hook in self._hooks:
                hook()
            for i, wrapper in enumerate(self._envs):
                if not active[i]:
                    continue
                env = wrapper.env
                env.sim.forward()
                env._pre_action(self._actions[i], policy_step=substep == 0 or self._changed[i])
                self._changed[i] = False
                env.sim.step()
                env._update_observables()
            self._tick += 1

        rewards = torch.zeros(self.n_envs)
        for i, wrapper in enumerate(self._envs):
            if not active[i]:
                continue
            env = wrapper.env
            env.cur_time += env.control_timestep
            reward, done, _ = env._post_action(self._actions[i])
            rewards[i] = reward
            self._steps[i] += 1
            success = bool(wrapper.check_success())
            timeout = not success and (done or self._steps[i] >= self._max_steps)
            self._outcomes[i] = EpisodeOutcome(i, success, success, bool(timeout))
        terminated = torch.tensor([o.terminated for o in self._outcomes])
        truncated = torch.tensor([o.truncated for o in self._outcomes])
        return None, rewards, terminated, truncated, {}

    def reset(self, env_id: int) -> EpisodeOutcome:
        outcome = self._outcomes[env_id]
        wrapper = self._envs[env_id]
        wrapper.reset()
        wrapper.set_init_state(self._initial_states[self._next_episode % len(self._initial_states)])
        env = wrapper.env
        dt = 1 / self.tick_per_s
        env.model_timestep = dt
        env.sim.model.opt.timestep = dt
        for robot in env.robots:
            robot.controller.model_timestep = dt
        self._actions[env_id] = [0.] * 6 + [-1.]
        for _ in range(self.config.num_steps_wait):
            wrapper.step(self._actions[env_id])
        self._episodes[env_id] = self._next_episode
        self._next_episode += 1
        self._steps[env_id] = 0
        self._changed[env_id] = True
        self._outcomes[env_id] = EpisodeOutcome(env_id, False, False, False)
        return outcome

    def get_observation(self, env_id: int) -> InferenceRequest:
        from openpi_client import image_tools

        # Force a snapshot: normal camera sampling runs at the control rate,
        # while requests can occur at any physics tick.
        obs = self._envs[env_id].env._get_observations(force_update=True)
        size = self.config.resize_size
        image = lambda name: image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs[name][::-1, ::-1]), size, size)).copy()
        quat = obs['robot0_eef_quat']
        w = float(np.clip(quat[3], -1., 1.))
        den = math.sqrt(1 - w * w)
        rotation = np.zeros(3) if math.isclose(den, 0.) else quat[:3] * (2 * math.acos(w) / den)
        state = np.concatenate((obs['robot0_eef_pos'], rotation, obs['robot0_gripper_qpos']))
        return InferenceRequest(observation={
            'observation/image': image('agentview_image'),
            'observation/wrist_image': image('robot0_eye_in_hand_image'),
            'observation/state': state,
            'prompt': self._prompt,
            'episode': self._episodes[env_id],
            'seed': self.config.seed,
            'multi_actions': 0,
        })

    def close(self):
        for env in self._envs:
            env.close()
        self._envs.clear()
