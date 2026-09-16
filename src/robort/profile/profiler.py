"""Wall-clock latency and sampled GPU energy for individual iterations."""

from contextlib import contextmanager
from dataclasses import replace
from math import isfinite
from statistics import fmean, pstdev
from time import perf_counter

from robort.profile.schemas import StageProfile
from robort.profile.environment import power_draw_w


class Profiler:
    """Record one completed iteration per tick; warmup belongs outside the context.

    Latency includes host dispatch and device synchronization, in milliseconds.
    Energy is estimated in joules using the mean of the endpoint power samples
    times latency. It includes idle/device-wide power, not just this process.
    Short iterations may be below the telemetry refresh interval. Unsupported
    telemetry produces NaN energy while retaining latency measurements.

    Optional callbacks supply power in watts, synchronization, and time in
    seconds. Defaults use the current Torch CUDA device when available; CPU
    measurements have no energy telemetry. Telemetry reads and bookkeeping are
    excluded from the next iteration's timing. Instances are reusable but cannot
    be nested or shared concurrently.
    """

    def __init__(self, *, power_w=None, synchronize=None, clock=perf_counter):
        self._power_callback = power_w
        self._sync_callback = synchronize
        self._clock = clock
        self._active = False
        self.energies = []
        self.latencies = []

    @contextmanager
    def measure(self, stream=None):
        """Measure on a supplied stream, including green-context work."""
        self._enter(stream)
        try:
            yield self
        finally:
            self.__exit__(None, None, None)

    def __enter__(self):
        return self._enter()

    def _enter(self, stream=None):
        if self._active:
            raise RuntimeError("Profiler is already active")
        self.energies = []
        self.latencies = []
        self._power = self._power_callback
        self._synchronize = self._sync_callback or (stream.synchronize if stream is not None else None)
        if self._power is None or self._synchronize is None:
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None and torch.cuda.is_available():
                device = stream.device if stream is not None else torch.cuda.current_device()
                if self._power is None:
                    self._power = lambda: power_draw_w(torch, device)
                if self._synchronize is None:
                    self._synchronize = lambda: torch.cuda.synchronize(device)
        if self._synchronize is None:
            self._synchronize = lambda: None
        self._synchronize()
        self._previous_power = self._read_power()
        self._start = self._clock()
        self._active = True
        return self

    def _read_power(self):
        if self._power is None:
            return float("nan")
        try:
            value = float(self._power())
        except (AttributeError, ImportError, RuntimeError):
            return float("nan")
        return value if isfinite(value) and value >= 0 else float("nan")

    def tick(self):
        if not self._active:
            raise RuntimeError("tick requires an active Profiler context")
        self._synchronize()
        elapsed = self._clock() - self._start
        if not isfinite(elapsed) or elapsed < 0:
            raise ValueError("Profiler clock must be finite and monotonic")
        power = self._read_power()
        self.latencies.append(elapsed * 1000)
        self.energies.append((self._previous_power + power) / 2 * elapsed)
        self._previous_power = power
        self._start = self._clock()

    def __exit__(self, exc_type, exc_value, traceback):
        # Only explicit ticks commit samples; a failed/unfinished call is omitted.
        self._active = False
        self._synchronize = None
        self._power = None
        return False

    def to_profile(self, metadata: StageProfile) -> StageProfile:
        """Copy stage metadata and fill mean/population-std measurements.

        Empty runs fail; any missing energy sample makes both energy statistics
        NaN rather than silently dropping iterations.
        """
        if not self.latencies:
            raise ValueError("Profiler has no completed iterations")
        lat_mean, lat_std = self._statistics(self.latencies)
        energy_mean, energy_std = self._statistics(self.energies)
        return replace(metadata, lat_mean=lat_mean, lat_std=lat_std,
                       energy_mean=energy_mean, energy_std=energy_std)

    @staticmethod
    def _statistics(values):
        if not all(isfinite(value) for value in values):
            return float("nan"), float("nan")
        return fmean(values), pstdev(values)
