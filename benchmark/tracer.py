from __future__ import annotations

import base64
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass, field, is_dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

from .envs import EpisodeOutcome, PHASE_TYPE
from robort.schemas import InferenceResponse, InferenceRequest


def _image_uri(value):
    from PIL import Image

    image = Image.fromarray(value).convert('RGB')
    image.thumbnail((384, 384))
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=80)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def _json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


@dataclass
class Metrics:
    suc: bool
    t_s: float
    starvation_s: float
    delay_s: float
    delay_std_s: float
    replan_steps: float
    replan_steps_std: float
    by_phase: dict[PHASE_TYPE, 'Metrics']
    

@dataclass
class Tracer:
    init_tick: int
    tick_per_s: int
    tick_per_act: int 
    output_prefix: str
    fps: int = 25

    _obvs: list[tuple[int, PHASE_TYPE, InferenceRequest]] = field(default_factory=list)
    _inferences: list[tuple[int, int, int, InferenceResponse]] = field(default_factory=list)
    _actions: list[tuple[int, int, np.ndarray]] = field(default_factory=list)
    _outcome: EpisodeOutcome | None = None
    _finish_tick: int | None = None 
    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError('FPS must be positive')

    def need_obv(self, tick: int):
        return (tick - self.init_tick) % max(1, self.tick_per_s // self.fps) == 0

    def update_obv(self, tick:int, phase: PHASE_TYPE, obv: InferenceRequest):
        self._obvs.append((tick, phase, obv))

    def update_chunk(self, tick: int, obv_tick: int, chunk_id: int, chunk: InferenceResponse):
        self._inferences.append((tick, obv_tick, chunk_id, chunk))
        
    def update_action(self, tick: int, chunk_id: int, action: np.ndarray):
        self._actions.append((tick, chunk_id, action))
    
    def export_dict(self):
        """Write episode and per-phase Metrics as JSON; return its data."""
        payload = json.dumps(self._metrics(), default=_json_value, allow_nan=False, indent=2)
        self._write('json', payload)
        return json.loads(payload)

    def _metrics(self):
        end = self._finish_tick if self._finish_tick is not None else max(
            [self.init_tick] + [tick for tick, *_ in self._obvs + self._inferences + self._actions])
        observations = sorted(self._obvs, key=lambda item: item[0])
        ticks = [tick for tick, _, _ in observations]

        def phase_at(tick):
            index = bisect_right(ticks, tick) - 1
            return observations[index][1] if index >= 0 else None

        # A selected action covers one action period; subsequent holding is starvation.
        gaps, covered_until = [], self.init_tick
        for tick, _, _ in sorted(self._actions, key=lambda item: item[0]):
            start = max(self.init_tick, min(end, tick))
            if start > covered_until:
                gaps.append((covered_until, start))
            covered_until = max(covered_until, min(end, tick + self.tick_per_act))
        if covered_until < end:
            gaps.append((covered_until, end))

        windows_by_phase = {}
        for index, (tick, phase, _) in enumerate(observations):
            start, stop = max(self.init_tick, tick), min(end, ticks[index+1] if index+1 < len(ticks) else end)
            if stop > start:
                windows = windows_by_phase.setdefault(phase, [])
                if windows and windows[-1][1] == start:
                    windows[-1] = (windows[-1][0], stop)
                else:
                    windows.append((start, stop))

        chunks = sorted(self._inferences, key=lambda item: item[1])
        delays = [(phase_at(obv_tick), (tick-obv_tick)/self.tick_per_s)
                  for tick, obv_tick, _, _ in chunks]
        replans = [(phase_at(a[1]), (b[1]-a[1])/self.tick_per_act) for a, b in zip(chunks, chunks[1:])]

        def summarize(windows, delays, replans):
            return Metrics(
                suc=bool(self._outcome and self._outcome.success),
                t_s=sum(stop-start for start, stop in windows)/self.tick_per_s,
                starvation_s=sum(max(0, min(stop, b)-max(start, a))
                                 for start, stop in windows for a, b in gaps)/self.tick_per_s,
                delay_s=float(np.mean(delays)) if delays else 0.,
                delay_std_s=float(np.std(delays)) if delays else 0.,
                replan_steps=float(np.mean(replans)) if replans else 0.,
                replan_steps_std=float(np.std(replans)) if replans else 0., by_phase={})

        metrics = summarize([(self.init_tick, end)], [value for _, value in delays],
                            [value for _, value in replans])
        metrics.by_phase = {phase: summarize(windows, [v for p, v in delays if p == phase],
                                             [v for p, v in replans if p == phase])
                            for phase, windows in windows_by_phase.items()}
        return metrics

    def export_html(self):
        """Write a self-contained, interactive viewer for the recorded trace."""
        template = Path(__file__).with_suffix('.html').read_text()
        payload = json.dumps(self._data(), allow_nan=False).replace('<', '\\u003c')
        return self._write('html', template.replace('__TRACE_DATA__', payload))

    def _data(self):
        observations = []
        for tick, phase, request in self._obvs:
            observation = {key: _image_uri(value) if 'image' in key and isinstance(value, np.ndarray)
                           and value.ndim == 3 else value for key, value in request.observation.items()}
            observations.append(dict(tick=tick, phase=phase,
                                     request=dict(vars(request), observation=observation)))
        data = dict(schema_version=1,
            config=dict(name=Path(self.output_prefix).name, init_tick=self.init_tick,
                        tick_per_s=self.tick_per_s, tick_per_act=self.tick_per_act, fps=self.fps),
            finish_tick=self._finish_tick, outcome=self._outcome, observations=observations,
            inferences=[dict(tick=tick, obv_tick=obv_tick, chunk_id=chunk_id, **vars(chunk))
                        for tick, obv_tick, chunk_id, chunk in self._inferences],
            actions=[dict(tick=tick, chunk_id=chunk_id, action=action)
                     for tick, chunk_id, action in self._actions])
        return json.loads(json.dumps(data, default=_json_value, allow_nan=False))

    def _write(self, suffix, content):
        path = Path(f'{self.output_prefix}.{suffix}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def finish(self, tick: int, result: EpisodeOutcome):
        self._outcome = result
        self._finish_tick = tick
        self.export_dict()
        self.export_html()
