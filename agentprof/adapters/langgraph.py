"""Wrap LangGraph node callables so each node becomes a STEP span.

    from agentprof.adapters.langgraph import instrument_node

    builder.add_node("plan", instrument_node(plan_fn, tracer))

Two caveats to verify against your LangGraph version before publishing numbers:

* If nodes run concurrently, each span's CPU figure covers only its own thread
  (`Span.cpu_is_partial` is set); turn-level CPU will be understated.
* Checkpointer reads and writes are the state-externalisation cost the paper
  cares about. Wrap the checkpointer's put/get with `instrument_state` too, or
  they will not appear in the blocked-time breakdown.
"""
from __future__ import annotations

import functools
from typing import Any, Callable

from ..tracer import SpanKind, Tracer


def instrument_node(fn: Callable[..., Any], tracer: Tracer, *, name: str | None = None) -> Callable[..., Any]:
    label = name or getattr(fn, "__name__", "node")
    # Bug found running this against real LangGraph (see CALIBRATION-NOTES.md):
    # `step=-1` was hardcoded, so every call from a given node collapsed onto
    # the same "state@-1" span name and metrics.summarise's per-step size
    # stats couldn't distinguish one invocation of a node from the next.
    # Track calls to this wrapper instead, and stamp both the STEP span and
    # the state span with the real, increasing step index.
    call_count = 0

    @functools.wraps(fn)
    def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        step = call_count
        call_count += 1
        with tracer.span(label, SpanKind.STEP, step=step):
            out = fn(state, *args, **kwargs)
            tracer.record_state(out if out is not None else state, step=step)
            return out

    return wrapper


def instrument_state(fn: Callable[..., Any], tracer: Tracer, *, name: str = "checkpoint") -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with tracer.span(name, SpanKind.STATE_IO):
            return fn(*args, **kwargs)

    return wrapper
