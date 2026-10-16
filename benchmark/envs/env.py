from dataclasses import dataclass
from typing import Callable

import numpy as np

from robort.schemas import InferenceRequest

PHASE_TYPE = int | str

@dataclass
class EpisodeOutcome:
    env_id: int
    success: bool
    terminated: bool
    truncated: bool
    duration_s: float = 0.0

@dataclass
class EnvConfig: 
    type_: str
    action_per_s: int
    tick_per_s: int

@dataclass 
class Environment:
    config: EnvConfig

    @property
    def tick_per_s(self):
        return self.config.tick_per_s

    @property
    def action_per_s(self):
        return self.config.action_per_s

    @property
    def tick_per_act(self):
        assert self.tick_per_s % self.action_per_s == 0
        return self.tick_per_s // self.action_per_s 
    
    @property 
    def n_envs(self) -> int:
        ... 
    

    def get_tick(self) -> int:
        ... 

    def get_phase(self, eid: int) -> PHASE_TYPE:
        ...

    def add_pre_action_hook(self, fn: Callable[[], None]) -> None:
        ...

    def update_actions(self, actions: list[tuple[int, np.ndarray]]):
        ...

    def step(self):
        ... 

    def reset(self, env_id: int) -> EpisodeOutcome: 
        '''reset env_id's environment and return if the task succeed'''
        ...

    def get_observation(self, env_id: int) -> InferenceRequest:
        ... 
