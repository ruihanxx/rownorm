import time
from dataclasses import dataclass


@dataclass
class SmoothedValue:
    value: float = 0.0
    count: int = 0

    def update(self, v: float, n: int = 1):
        self.value += v * n
        self.count += n

    @property
    def avg(self):
        return self.value / max(1, self.count)


class Timer:
    def __init__(self):
        self.start_t = time.time()

    def reset(self):
        self.start_t = time.time()

    def elapsed(self):
        return time.time() - self.start_t

