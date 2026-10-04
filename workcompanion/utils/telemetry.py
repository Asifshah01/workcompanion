"""Lightweight in-process telemetry: latencies, retrieval stats, error counts.

Deliberately simple - the goal is visibility in the UI and in logs, not a full
metrics stack.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = ["Timing", "get_telemetry", "Timer"]


@dataclass(slots=True)
class Timing:
    """Aggregated timing samples for one operation name."""

    name: str
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def add(self, milliseconds: float) -> None:
        self.samples.append(float(milliseconds))

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def last_ms(self) -> float:
        return self.samples[-1] if self.samples else 0.0

    @property
    def average_ms(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = min(int(len(ordered) * 0.95), len(ordered) - 1)
        return ordered[index]


class Telemetry:
    """Collects timings and counters for the current process."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._timings: dict[str, Timing] = {}
        self._counters: dict[str, int] = defaultdict(int)

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        """Time a block of work under ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000)

    def record(self, name: str, milliseconds: float) -> None:
        if not self.enabled:
            return
        self._timings.setdefault(name, Timing(name)).add(milliseconds)

    def count(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        self._counters[name] += amount

    def snapshot(self) -> dict[str, Any]:
        """Serialisable snapshot for the UI / logs."""
        return {
            "timings": {
                name: {
                    "count": t.count,
                    "last_ms": round(t.last_ms, 1),
                    "avg_ms": round(t.average_ms, 1),
                    "p95_ms": round(t.p95_ms, 1),
                }
                for name, t in sorted(self._timings.items())
            },
            "counters": dict(sorted(self._counters.items())),
        }

    def reset(self) -> None:
        self._timings.clear()
        self._counters.clear()


class _Timer:
    """``with timer("llm"):`` helper mirroring :meth:`Telemetry.track`."""

    def __init__(self, telemetry: Telemetry, name: str) -> None:
        self._telemetry = telemetry
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._telemetry.record(self._name, (time.perf_counter() - self._start) * 1000)


_TELEMETRY = Telemetry()


def get_telemetry() -> Telemetry:
    """Return the process-wide telemetry collector."""
    return _TELEMETRY


def timer(name: str) -> _Timer:
    """Context manager timing an arbitrary operation."""
    return _Timer(_TELEMETRY, name)