"""Human-readable and machine-readable output."""

from __future__ import annotations

import json
from typing import Any, Sequence

from .metrics import TurnMetrics, aggregate


def _bar(fraction: float, width: int = 28) -> str:
    filled = max(0, min(width, round(fraction * width)))
    return "#" * filled + "." * (width - filled)


def turn_report(m: TurnMetrics, *, title: str = "turn") -> str:
    lines = [
        f"{title}",
        "-" * max(len(title), 46),
        f"  wall clock         {m.turn_wall_s:>10.3f} s",
        f"  cpu (own thread)   {m.turn_cpu_s:>10.3f} s",
        f"  billed idle        {m.billed_idle_s:>10.3f} s   {_bar(m.billed_idle_fraction)} "
        f"{m.billed_idle_fraction * 100:.1f}%",
    ]
    if m.wait_to_compute_ratio is not None:
        lines.append(f"  wait : compute     {m.wait_to_compute_ratio:>10.1f} : 1")
    lines += [
        f"  steps              {m.steps:>10d}",
        f"  remote calls       {m.remote_calls:>10d}",
        f"  longest block      {m.longest_block_s:>10.3f} s   ({m.longest_block_name})",
        f"  state / step p50   {m.state_bytes_per_step_p50:>10d} bytes",
        f"  state / step max   {m.state_bytes_per_step_max:>10d} bytes",
    ]
    if m.blocked_by_kind_s:
        lines.append("  blocked by kind:")
        for k, v in m.blocked_by_kind_s.items():
            lines.append(f"      {k:<14} {v:>8.3f} s")
    for n in m.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def cost_note(m: TurnMetrics, *, gb_seconds_price: float, memory_gb: float = 1.0) -> str:
    """Illustrate the billing inversion using a published per-GB-second price.

    Pass the real figure from the provider's pricing page; nothing is hardcoded
    here on purpose, because prices change and a stale constant in a paper is
    an easy reviewer objection.
    """
    billed = m.turn_wall_s * memory_gb * gb_seconds_price
    useful = m.turn_cpu_s * memory_gb * gb_seconds_price
    waste = billed - useful
    multiple = (billed / useful) if useful > 0 else float("inf")
    return (
        f"  at {gb_seconds_price:g} per GB-second and {memory_gb:g} GB:\n"
        f"      billed for this turn      {billed:.6f}\n"
        f"      attributable to work      {useful:.6f}\n"
        f"      paid while blocked        {waste:.6f}   ({multiple:.1f}x)"
    )


def aggregate_report(runs: Sequence[TurnMetrics], *, title: str) -> str:
    agg = aggregate(runs)
    lines = [f"{title}  (n={agg['n_runs']})", "-" * max(len(title) + 8, 46)]
    for k, v in agg.items():
        if k == "n_runs":
            continue
        lines.append(f"  {k:<28} {v if v is not None else 'n/a'}")
    return "\n".join(lines)


def to_json(runs: Sequence[TurnMetrics], *, shape: str) -> str:
    return json.dumps(
        {"shape": shape, "aggregate": aggregate(runs), "runs": [r.to_dict() for r in runs]},
        indent=2,
    )
