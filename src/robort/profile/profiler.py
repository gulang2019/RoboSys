from robort.profile.schemas import *
from dataclasses import dataclass


class Profiler:
    energies: list[float] = []
    latencies: list[float] = []

    def __enter__(self):
        self.energies = []
        self.latencies = []
        return self

    def tick(self):
        # TODO: implement the logic to record energy and latency
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        # TODO: set the energy and latency profile
        ... 

    def to_distr(self) -> tuple[DistrProfile, DistrProfile]:
        # TODO: convert to distribution profile
        ...

