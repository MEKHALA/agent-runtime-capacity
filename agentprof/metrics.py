"""Turn-level metrics derived from spans.

The headline figure is `billed_idle_fraction`: on a per-millisecond-billed
serverless platform you are charged for the whole wall clock of an invocation,
but only `cpu` seconds of it did any work. Everything else is time you paid for
while blocked on somebody else's network.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Any, Sequence

from .tracer import Span, SpanKind


@dataclass
class TurnMetrics:
    turn_wall_s: float
    turn_cpu_s: float
    billed_idle_s: float
    billed_idle_fraction: float
    wait_to_compute_ratio: float | None
    steps: int
    remote_calls: int
    longest_block_s: float
    longest_block_name: str
    blocked_by_kind_s: dict[str, float]
    state_bytes_total: int
    state_bytes_per_step_p50: int
    state_bytes_per_step_max: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _root(spans: Sequence[Span]) -> Span:
    turns = [s for s in spans if s.kind is SpanKind.TURN]
    if not turns:
        raise ValueError("no TURN span found — wrap the whole turn in one")
    if len(turns) > 1:
        raise ValueError(f"expected exactly one TURN span, found {len(turns)}")
    return turns[0]


def summarise(spans: Sequence[Span]) -> TurnMetrics:
    turn = _root(spans)
    notes: list[str] = []

    real = [s for s in spans if not s.attrs.get("measured_only")]
    steps = [s for s in real if s.kind is SpanKind.STEP]
    remote = [s for s in real if s.kind.is_remote_wait]

    # Only count leaf-level remote spans so nested waits aren't double counted.
    parent_ids = {s.parent_id for s in real if s.parent_id}
    leaf_remote = [s for s in remote if s.span_id not in parent_ids]
    if len(leaf_remote) != len(remote):
        notes.append(
            f"{len(remote) - len(leaf_remote)} nested remote span(s) excluded from "
            "the blocked-time breakdown to avoid double counting"
        )

    blocked_by_kind: dict[str, float] = {}
    for s in leaf_remote:
        blocked_by_kind[s.kind.value] = blocked_by_kind.get(s.kind.value, 0.0) + s.blocked

    if any(s.cpu_is_partial for s in real):
        notes.append(
            "some spans ran off the tracer's root thread; their CPU figures cover "
            "only their own thread, so turn CPU may be understated"
        )

    wall = turn.wall
    cpu = turn.cpu
    idle = max(0.0, wall - cpu)

    longest = max(leaf_remote, key=lambda s: s.blocked, default=None)

    state_spans = [s for s in spans if s.attrs.get("state_bytes") is not None]
    sizes = [int(s.attrs["state_bytes"]) for s in state_spans]

    return TurnMetrics(
        turn_wall_s=round(wall, 6),
        turn_cpu_s=round(cpu, 6),
        billed_idle_s=round(idle, 6),
        billed_idle_fraction=round(idle / wall, 6) if wall > 0 else 0.0,
        wait_to_compute_ratio=round(idle / cpu, 3) if cpu > 0 else None,
        steps=len(steps),
        remote_calls=len(leaf_remote),
        longest_block_s=round(longest.blocked, 6) if longest else 0.0,
        longest_block_name=longest.name if longest else "",
        blocked_by_kind_s={k: round(v, 6) for k, v in sorted(blocked_by_kind.items())},
        state_bytes_total=sum(sizes),
        state_bytes_per_step_p50=int(statistics.median(sizes)) if sizes else 0,
        state_bytes_per_step_max=max(sizes) if sizes else 0,
        notes=notes,
    )


def aggregate(runs: Sequence[TurnMetrics]) -> dict[str, Any]:
    """Across repeated turns of the same shape. Report medians, not means —
    latency distributions here are heavily right-skewed."""
    if not runs:
        raise ValueError("no runs to aggregate")

    def med(key: str) -> float:
        return round(statistics.median(getattr(r, key) for r in runs), 6)

    ratios = [r.wait_to_compute_ratio for r in runs if r.wait_to_compute_ratio is not None]
    walls = sorted(r.turn_wall_s for r in runs)

    return {
        "n_runs": len(runs),
        "turn_wall_s_p50": med("turn_wall_s"),
        "turn_wall_s_p95": round(walls[min(len(walls) - 1, int(0.95 * len(walls)))], 6),
        "turn_cpu_s_p50": med("turn_cpu_s"),
        "billed_idle_fraction_p50": med("billed_idle_fraction"),
        "wait_to_compute_ratio_p50": round(statistics.median(ratios), 3) if ratios else None,
        "steps_p50": med("steps"),
        "remote_calls_p50": med("remote_calls"),
        "state_bytes_per_step_p50": med("state_bytes_per_step_p50"),
    }
