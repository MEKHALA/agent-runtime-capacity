"""Span recording for agent turns.

The measurement that matters for the paper is the gap between *wall* time and
*CPU* time inside a single agent turn. On a per-millisecond-billed platform you
pay for wall time; an agent turn spends most of its wall time blocked on a
remote model or tool. Every span therefore records both clocks.

Threading note: `time.thread_time()` is per-thread CPU, which is what we want
when an agent runs its steps on one thread. If you instrument a framework that
fans work out across threads, a span's CPU figure covers only the thread that
opened it — see `Span.cpu_is_partial`.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterator


class SpanKind(str, Enum):
    """Why a span exists. Drives the wait-vs-compute split."""

    LLM_CALL = "llm_call"          # blocked on remote inference
    TOOL_CALL = "tool_call"        # blocked on a remote tool / API
    HUMAN_WAIT = "human_wait"      # blocked on a person
    RETRIEVAL = "retrieval"        # blocked on a vector store / DB
    STATE_IO = "state_io"          # reading or writing externalised state
    COMPUTE = "compute"            # genuine local work (parsing, planning glue)
    STEP = "step"                  # one iteration of the agent loop
    TURN = "turn"                  # the whole turn

    @property
    def is_remote_wait(self) -> bool:
        return self in _REMOTE_WAIT


_REMOTE_WAIT = frozenset(
    {SpanKind.LLM_CALL, SpanKind.TOOL_CALL, SpanKind.HUMAN_WAIT, SpanKind.RETRIEVAL, SpanKind.STATE_IO}
)


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    wall_start: float
    wall_end: float | None = None
    cpu_start: float = 0.0
    cpu_end: float | None = None
    thread_id: int = 0
    cpu_is_partial: bool = False
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def wall(self) -> float:
        if self.wall_end is None:
            raise ValueError(f"span {self.name!r} was never closed")
        return self.wall_end - self.wall_start

    @property
    def cpu(self) -> float:
        if self.cpu_end is None:
            raise ValueError(f"span {self.name!r} was never closed")
        return max(0.0, self.cpu_end - self.cpu_start)

    @property
    def blocked(self) -> float:
        """Wall time this span was not burning CPU on its own thread."""
        return max(0.0, self.wall - self.cpu)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["wall"] = self.wall
        d["cpu"] = self.cpu
        d["blocked"] = self.blocked
        return d


class Tracer:
    """Collects spans for one process. Not thread-safe across turns by design:
    start a Tracer per turn, or call `reset()` between turns."""

    def __init__(self) -> None:
        self._spans: list[Span] = []
        self._stack: list[str] = []
        self._lock = threading.Lock()
        self._root_thread = threading.get_ident()

    def reset(self) -> None:
        with self._lock:
            self._spans.clear()
            self._stack.clear()

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    @contextmanager
    def span(self, name: str, kind: SpanKind, **attrs: Any) -> Iterator[Span]:
        tid = threading.get_ident()
        s = Span(
            span_id=uuid.uuid4().hex[:12],
            parent_id=self._stack[-1] if self._stack else None,
            name=name,
            kind=kind,
            wall_start=time.monotonic(),
            cpu_start=time.thread_time(),
            thread_id=tid,
            cpu_is_partial=(tid != self._root_thread),
            attrs=dict(attrs),
        )
        with self._lock:
            self._spans.append(s)
            self._stack.append(s.span_id)
        try:
            yield s
        finally:
            s.wall_end = time.monotonic()
            s.cpu_end = time.thread_time()
            with self._lock:
                if self._stack and self._stack[-1] == s.span_id:
                    self._stack.pop()

    def record_state(self, state: Any, *, step: int) -> int:
        """Measure the serialised size of agent state at a step boundary.

        This is the input to the 'externalisation tax' argument: on a stateless
        substrate this many bytes cross the wire on every step, not once.
        """
        try:
            payload = json.dumps(state, default=str, separators=(",", ":"))
            size = len(payload.encode("utf-8"))
        except (TypeError, ValueError):
            size = len(repr(state).encode("utf-8"))
        with self._lock:
            self._spans.append(
                Span(
                    span_id=uuid.uuid4().hex[:12],
                    parent_id=self._stack[-1] if self._stack else None,
                    name=f"state@{step}",
                    kind=SpanKind.STATE_IO,
                    wall_start=time.monotonic(),
                    wall_end=time.monotonic(),
                    cpu_start=0.0,
                    cpu_end=0.0,
                    attrs={"step": step, "state_bytes": size, "measured_only": True},
                )
            )
        return size

    def to_json(self) -> str:
        return json.dumps([s.to_dict() for s in self._spans], indent=2)
