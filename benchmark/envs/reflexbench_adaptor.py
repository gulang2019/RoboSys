from dataclasses import dataclass
from typing import Callable
import math
import numpy as np

from robort.schemas import InferenceRequest
from .env import Environment, EpisodeOutcome, EnvConfig


@dataclass
class ReflexBenchConfig(EnvConfig):
    type_: str = 'conveyor_belt_pnp'
    action_per_s: int = 25
    tick_per_s: int = 1000
    task: str = 'ConveyorBeltPickAndPlace-Franka-DataCollection-v0'
    num_envs: int = 1
    seed: int = 7
    device: str = 'cuda:0'
    belt_speed_mps: float = .3
    episode_length_s: float | None = None
    headless: bool = True
    action_scale: float = .1
    prompt: str = 'pick up the cube from the moving conveyor and place it into the bin, leaving the macaroni-and-cheese package untouched'


class ReflexBenchAdaptor(Environment):
    def __init__(self, config: EnvConfig):
        if not isinstance(config, ReflexBenchConfig):
            config = ReflexBenchConfig(**vars(config))
        super().__init__(config)
        if (config.num_envs < 1 or config.action_per_s < 1 or config.tick_per_s < 1
                or config.tick_per_s % config.action_per_s):
            raise ValueError('Positive environment counts and an integral number of ticks per action are required')
        if not math.isfinite(config.action_scale) or config.action_scale <= 0:
            raise ValueError('Action scale must be finite and positive')
        if not math.isfinite(config.belt_speed_mps) or config.belt_speed_mps < 0:
            raise ValueError('Belt speed must be finite and nonnegative')

        from isaaclab.app import AppLauncher
        self._app = AppLauncher(headless=config.headless, enable_cameras=True, device=config.device).app
        self._env = None
        try:
            import gymnasium as gym
            import reflexbench.tasks  # Register tasks after Isaac Sim starts.
            from isaaclab_tasks.utils import parse_env_cfg

            cfg = parse_env_cfg(config.task, device=config.device, num_envs=config.num_envs)
            cfg.seed = config.seed
            cfg.sim.dt = 1 / config.tick_per_s
            cfg.decimation = self.tick_per_act
            cfg.sim.render_interval = max(1, round(.01 * config.tick_per_s))
            cfg.observations.policy.enable_corruption = False
            cfg.actions.arm_action.scale = config.action_scale
            if config.episode_length_s is not None:
                cfg.episode_length_s = config.episode_length_s
            if hasattr(cfg.events, 'print_pos'):
                cfg.events.print_pos = None
            if hasattr(cfg.events, 'reset_conveyor_velocity'):
                cfg.events.reset_conveyor_velocity.params['velocity_range'] = (-config.belt_speed_mps,) * 2

            self._env = gym.make(config.task, cfg=cfg)
            self._manager = self._env.unwrapped
            self._manager.enable_ik_override = False
            if config.task.startswith('ConveyorBeltPickAndPlace'):
                origins = self._manager.scene.env_origins
                self._manager.scene['fixed_cam'].set_world_poses_from_view(
                    origins + origins.new_tensor([.85, 1.8, 1.2]),
                    origins + origins.new_tensor([.85, 0., .75]))
            self._env.reset(seed=config.seed)
            self._robot = self._manager.scene['robot']
            self._arm_ids = self._robot.find_joints(['panda_joint.*'])[0]
            self._finger_ids = self._robot.find_joints(['panda_finger.*'])[0]
            self._targets = self._robot.data.joint_pos[:, self._arm_ids].clone()
            self._gripper = self._targets.new_ones(self.n_envs)
            self._command = self._targets.new_zeros((self.n_envs, 8))
            self._command[:, -1] = 1
            self._tick = int(self._manager._sim_step_counter)
            self._hooks = []
            self._outcomes = {}
            self._stepping = False
            self._closed = False
            self._original_apply = self._manager.action_manager.apply_action
            self._original_reset = self._manager._reset_idx
            self._manager.action_manager.apply_action = self._apply_action
            self._manager._reset_idx = self._reset_idx
        except BaseException:
            if self._env is not None:
                self._env.close()
            self._app.close()
            raise

    @property
    def n_envs(self) -> int:
        return self._manager.num_envs

    def get_tick(self) -> int:
        return self._tick

    def get_phase(self, env_id: int) -> int | str:
        return self._manager.task_phase[env_id].item()

    def add_pre_action_hook(self, fn: Callable[[], None]) -> None:
        self._hooks.append(fn)

    def _apply_action(self):
        # Isaac increments its counter before applying the action for that tick.
        self._tick = int(self._manager._sim_step_counter) - 1
        for hook in self._hooks:
            hook()
        # RelativeJointPositionAction adds current positions on every substep.
        self._command[:, :7] = (self._targets - self._robot.data.joint_pos[:, self._arm_ids]) / self.config.action_scale
        self._command[:, -1] = self._gripper
        self._manager.action_manager.process_action(self._command)
        self._original_apply()

    def update_actions(self, actions: list[tuple[int, np.ndarray]]):
        for env_id, action in actions:
            action = np.asarray(action)
            if action.shape != (8,) or not np.isfinite(action).all():
                raise ValueError(f'Expected a finite 8-dimensional joint action, got {action.shape}')
            self._targets[env_id] = self._targets.new_tensor(action[:7])
            self._gripper[env_id] = 1 if action[-1] > .5 else -1

    def step(self):
        self._stepping = True
        try:
            result = self._env.step(self._command)
            self._tick = int(self._manager._sim_step_counter)
            return result
        finally:
            self._stepping = False

    def _reset_idx(self, env_ids):
        # Capture terminal state before Isaac's automatic reset clears it.
        if self._stepping:
            for env_id in env_ids:
                i = int(env_id)
                success = (self._manager.valid_hits[i] > 0 if hasattr(self._manager, 'valid_hits') else
                           self._manager.termination_manager.get_term('task_completed')[i])
                self._outcomes[i] = EpisodeOutcome(i, bool(success),
                    bool(self._manager.reset_terminated[i]), bool(self._manager.reset_time_outs[i]))
        self._original_reset(env_ids)
        self._targets[env_ids] = self._robot.data.joint_pos[env_ids][:, self._arm_ids]
        self._gripper[env_ids] = 1

    def reset(self, env_id: int) -> EpisodeOutcome:
        if env_id in self._outcomes:
            return self._outcomes.pop(env_id)  # Already reset by env.step().
        self._manager.reset(env_ids=[env_id])
        return EpisodeOutcome(env_id, False, False, False)

    def get_observation(self, env_id: int) -> InferenceRequest:
        joint_pos = self._robot.data.joint_pos[env_id]
        state = np.empty(8, dtype=np.float32)
        state[:7] = joint_pos[self._arm_ids].cpu().numpy()
        state[-1] = float(joint_pos[self._finger_ids].mean() > .03)
        images = [self._manager.scene[name].data.output['rgb'][env_id, ..., :3]
                  .cpu().numpy().astype(np.uint8, copy=True) for name in ('fixed_cam', 'wrist_cam')]
        return InferenceRequest(observation={'observation/image': images[0],
            'observation/wrist_image': images[1], 'observation/state': state,
            'prompt': self.config.prompt, 'seed': self.config.seed})

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._manager.action_manager.apply_action = self._original_apply
        self._manager._reset_idx = self._original_reset
        try:
            self._env.close()
        finally:
            self._app.close()
