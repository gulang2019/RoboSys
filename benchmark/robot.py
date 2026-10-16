from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from robort.schemas import InferenceResponse

from .envs import EpisodeOutcome, PHASE_TYPE
from .tracer import Tracer

if TYPE_CHECKING:
    from .bench import PolicyServer
    from robort.schemas import InferenceRequest
    

@dataclass(kw_only = True)
class Robot:
    env_id: int
    config_id: int
    server: PolicyServer
    tick_per_s: int 
    tick_per_act: int
    init_tick: int

    _tracer: Tracer | None = None
    _phase: str | int | None = None
    _nxt_inf_t: int = 0
    _action_buffer: list[tuple[int, int, np.ndarray]] = field(default_factory=list) # action for 
    _action_timestamp: int = 0
    _executed_timestamp: int = 0
    _last_chunk_obv_tick: int = field(init = False) # the last tick scheduled

    def __post_init__(self):
        self._nxt_inf_t = math.ceil((self._get_replan_delay()) * self.tick_per_s) + self.init_tick
        self._last_chunk_obv_tick = self.init_tick


    @property
    def phase(self):
        return self._phase 

    def update_phase(self, phase):    
        self._phase = phase
    
    def _get_replan_delay(self):
        '''calculate the next replan delay in seconds'''
        ...
        
    def _get_infer_delay(self):
        '''calculate the delay of inference in seconds'''
        ...

    def need_sch_inf(self, tick):
        return tick >= self._nxt_inf_t or (self._tracer is not None and self._tracer.need_obv(tick))
    
    def maybe_sch_inf(self, tick: int,
                    obv: InferenceRequest):
        if self._tracer:
            self._tracer.update_obv(tick, self.phase, obv)

        if tick < self._nxt_inf_t: return 

        self._nxt_inf_t = math.ceil((self._get_replan_delay()) * self.tick_per_s) + tick
        self.server.schedule(self.env_id, obv, tick, tick + math.ceil(self._get_infer_delay() * self.tick_per_s))

    def add_chunk(self, response: InferenceResponse, obv_tick: int, cur_tick: int):
        # obv_tick -> obv_tick + tik_per_act -> obv_tick + 2 * tick_per_act -> cu
        if obv_tick < self._last_chunk_obv_tick: return
        self._last_chunk_obv_tick = obv_tick
        if self._tracer:
            self._tracer.update_chunk(cur_tick, obv_tick, self._action_timestamp, response)

        for i, act in enumerate(response.actions):
            heapq.heappush(self._action_buffer, (obv_tick + i * self.tick_per_act, - self._action_timestamp, act))
        self._action_timestamp += 1

    def maybe_update_action(self, tick: int):
        while self._action_buffer and tick >= self._action_buffer[0][0]:
            # action buffer is ready 
            launch_tick, timestamp_minus, action = heapq.heappop(self._action_buffer)
            timestamp = - timestamp_minus
            if self._executed_timestamp > timestamp:
                continue
            self._executed_timestamp = max(self._executed_timestamp, timestamp)
            if launch_tick == tick:
                if self._tracer: self._tracer.update_action(tick, timestamp, action)
                return action
        return None 

    def finish_episode(self, tick, result: EpisodeOutcome):
        result.duration_s = (tick - self.init_tick) / self.tick_per_s
        if self._tracer:
            self._tracer.finish(tick, result)
            self._tracer = None

def _sample_uniform_positive(mean, var):
    if not math.isfinite(mean) or not math.isfinite(var) or mean < 0 or var < 0:
        raise ValueError('Delay and jitter must be finite and nonnegative')
    return max(0.0, mean + (np.random.uniform(-var, var) if var else 0.0))

@dataclass
class PeriodicRobot(Robot):
    replan_steps: int
    delay: float
    replan_steps_jitter: float = 0
    delay_jitter: float = 0

    def _get_replan_delay(self):
        return _sample_uniform_positive(self.replan_steps, 
                        self.replan_steps_jitter) * self.tick_per_act / self.tick_per_s

    def _get_infer_delay(self):
        return _sample_uniform_positive(self.delay, self.delay_jitter) 

@dataclass(kw_only = True)
class PhaseAwareRobot(PeriodicRobot):
    replan_steps_by_phase: dict[str, int]
    delay_by_phase: dict[str, int]
    replan_steps_jitter_by_phase: dict[str, float]
    delay_jitter_by_phase: dict[str, float]

    def update_phase(self, phase):
        super().update_phase(phase)
        self.replan_steps = self.replan_steps_by_phase[self.phase]
        self.delay = self.delay_by_phase[self.phase]
        self.replan_steps_jitter = self.replan_steps_jitter_by_phase[self.phase]
        self.delay_jitter = self.delay_jitter_by_phase[self.phase]

@dataclass 
class RobotConfig:
    type_: str
    kwargs: dict
    enable_tracing: bool = False 
    results: list[EpisodeOutcome] = field(default_factory=list)

def create_robot(robot_config: RobotConfig,
                env_id,
                config_id,
                episode_id, 
                server,
                tick_per_s,
                tick_per_act,
                init_tick,
                output_path
                ):
    tracer = None
    if robot_config.enable_tracing:
        tracer = Tracer(init_tick = init_tick,
                        tick_per_s = tick_per_s,
                        tick_per_act = tick_per_act,
                        output_prefix = f'{output_path}/tracing/{config_id}_{episode_id}')
    if robot_config.type_ == 'periodic':
        return PeriodicRobot(**robot_config.kwargs, 
                             env_id = env_id,
                            config_id = config_id,
                            server = server, 
                            init_tick = init_tick,
                            tick_per_act=tick_per_act,
                            tick_per_s = tick_per_s,
                            _tracer = tracer)
    if robot_config.type_ == 'phase':
        return PhaseAwareRobot(**robot_config.kwargs, 
                            env_id = env_id,
                            config_id = config_id,
                            server = server, 
                            init_tick = init_tick,
                            tick_per_act=tick_per_act,
                            tick_per_s = tick_per_s,
                            _tracer = tracer)
    raise NotImplementedError 
