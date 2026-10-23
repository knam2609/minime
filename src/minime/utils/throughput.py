"""Throughput and model-FLOPs-utilisation measurement.

Tokens/second is the number that decides what is affordable, and MFU is the
number that says whether a slow run is the hardware's fault or ours.  Both go
in the metrics file for every run, because the optimiser experiment compares
wall-clock time and a 20% MFU difference between two arms would otherwise be
mistaken for an optimiser effect.
"""

from __future__ import annotations

import time
from collections import deque

# Dense half-precision peak throughput, FLOP/s.  Only used to express MFU as a
# percentage; an unknown device simply reports no MFU rather than a wrong one.
PEAK_FLOPS = {
    "Tesla T4": 65.0e12,
    "Tesla P100-PCIE-16GB": 18.7e12,
    "NVIDIA A100-SXM4-40GB": 312.0e12,
    "NVIDIA A10G": 125.0e12,
    "NVIDIA L4": 121.0e12,
    "NVIDIA GeForce RTX 4090": 165.2e12,
    "Apple Silicon (MPS)": 2.6e12,
    "CPU": None,
}


def peak_flops_for(device_name: str) -> float | None:
    if device_name in PEAK_FLOPS:
        return PEAK_FLOPS[device_name]
    for known, value in PEAK_FLOPS.items():
        if known.lower() in device_name.lower():
            return value
    return None


class ThroughputMeter:
    """Rolling tokens/s, deliberately ignoring the first step.

    Step 0 includes lazy CUDA/MPS kernel compilation and allocator warmup; it
    is several times slower than steady state and would skew a short run's
    reported throughput badly.
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self._durations: deque[float] = deque(maxlen=window)
        self._tokens: deque[int] = deque(maxlen=window)
        self._start: float | None = None
        self.total_tokens = 0
        self.total_seconds = 0.0
        self._steps = 0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self, tokens: int) -> float:
        if self._start is None:
            raise RuntimeError("ThroughputMeter.stop() called before start()")
        elapsed = time.perf_counter() - self._start
        self._start = None
        self._steps += 1
        if self._steps > 1:                    # skip warmup step
            self._durations.append(elapsed)
            self._tokens.append(tokens)
            self.total_tokens += tokens
            self.total_seconds += elapsed
        return elapsed

    @property
    def tokens_per_second(self) -> float:
        seconds = sum(self._durations)
        return sum(self._tokens) / seconds if seconds > 0 else 0.0

    @property
    def average_tokens_per_second(self) -> float:
        return self.total_tokens / self.total_seconds if self.total_seconds > 0 else 0.0

    def mfu(self, flops_per_token: float, device_name: str) -> float | None:
        peak = peak_flops_for(device_name)
        if not peak:
            return None
        return (self.tokens_per_second * flops_per_token) / peak


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
