"""Synthetic agent turns with replayable latency.

Two jobs, deliberately:

1. It exercises the tracer so the harness can be tested without any framework
   or network access.
2. It *is* the latency-replay mock for the substrate benchmark. Calibrate the
   distributions once against real model and tool calls (see
   `LatencyProfile.from_samples`), save the profile, then replay it identically
   on Lambda, on a container and on a VM. That isolates the substrate as the
   only variable and makes the benchmark reproducible by anyone — which is
   worth more to reviewers than spending real tokens.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .tracer import SpanKind, Tracer


@dataclass
class LatencyProfile:
    """Lognormal latency, which is the usual shape for remote call latency."""

    name: str
    median_s: float
    sigma: float = 0.45          # log-space spread; ~0.45 gives a realistic tail
    floor_s: float = 0.005
    # Provenance for a fitted profile: who/what/when/where it was calibrated
    # against, e.g. {"provider": "bedrock", "model": "...", "region": "...",
    # "recorded_at": "...", "n_samples": 50, "conditions": "..."}. None for a
    # profile that was constructed by hand (placeholder or synthetic-for-testing).
    calibration_context: dict[str, Any] | None = None

    def sample(self, rng: random.Random) -> float:
        mu = math.log(max(self.median_s, 1e-6))
        return max(self.floor_s, rng.lognormvariate(mu, self.sigma))

    @classmethod
    def from_samples(
        cls,
        name: str,
        samples: Iterable[float],
        *,
        calibration_context: dict[str, Any] | None = None,
    ) -> "LatencyProfile":
        """Fit from real observed latencies. Record these once against the real
        model/tool endpoints, then replay forever.

        Pass `calibration_context` to record what was fitted against (provider,
        model, region, date, conditions) so the profile is self-documenting.
        """
        xs = [s for s in samples if s > 0]
        if len(xs) < 2:
            raise ValueError("need at least two positive samples to fit")
        logs = [math.log(x) for x in xs]
        mu = sum(logs) / len(logs)
        var = sum((l - mu) ** 2 for l in logs) / (len(logs) - 1)
        return cls(
            name=name,
            median_s=math.exp(mu),
            sigma=math.sqrt(var),
            calibration_context=calibration_context,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "median_s": self.median_s,
            "sigma": self.sigma,
            "floor_s": self.floor_s,
            "calibration_context": self.calibration_context,
        }


@dataclass
class TurnShape:
    """One of the workload shapes the paper characterises."""

    name: str
    steps: int
    llm: LatencyProfile
    tool: LatencyProfile
    tools_per_step: float = 1.0
    human_wait: LatencyProfile | None = None
    compute_ms_per_step: float = 0.229    # measured — see _MEASURED_COMPUTE_MS_PER_STEP below
    state_growth_bytes: int = 1800        # per step, roughly one exchange of history


# Calibrated 2026-09-03 against real endpoints (see scripts/record_latencies.py,
# data/latency_raw.json for the raw samples, profiles.json for the fit):
#   llm:  AWS Bedrock, us-east-1, cross-region inference profile
#         "us.anthropic.claude-haiku-4-5-20251001-v1:0", on-demand invocation
#         via the Converse API. 50 calls, one discarded warm-up, max_tokens
#         drawn log-uniformly from [48, 1536] per call (the range of a
#         realistic agent reasoning turn), actual completion_tokens recorded.
#   tool: 30 calls alternating https://api.github.com/zen and
#         https://httpbin.org/get, one discarded warm-up per endpoint.
# Conditions: no other load on the AWS account at calibration time.
# `long_research` and `human_in_loop` reuse the same llm/tool fit — the
# underlying model and tool endpoints don't change with shape, only the step
# count and structure do. `human_wait` for human_in_loop is NOT calibrated
# (no real approval-wait data was collected) and remains a placeholder.
_CALIBRATED_LLM = LatencyProfile(
    "llm",
    median_s=4.229560649420906,
    sigma=0.5783297827186541,
    calibration_context={
        "provider": "bedrock",
        "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "region": "us-east-1",
        "recorded_at": "2026-09-03T09:49:28.717296+00:00",
        "n_samples": 50,
        "lo_tokens": 48,
        "hi_tokens": 1536,
        "conditions": "us-east-1, cross-region inference profile, on-demand Bedrock, no other load on this AWS account",
    },
)
_CALIBRATED_TOOL = LatencyProfile(
    "tool",
    median_s=0.5671790479191061,
    sigma=0.8855069178730316,
    calibration_context={
        "provider": "http",
        "endpoints": ["https://api.github.com/zen", "https://httpbin.org/get"],
        "recorded_at": "2026-09-03T09:49:28.717296+00:00",
        "n_samples": 30,
        "conditions": "no other load on this AWS account at calibration time",
    },
)

# Measured 2026-09-03, REVISED 2026-09-04 — real per-step LangGraph FRAMEWORK
# CPU (not the invented 12ms placeholder). The 3 Sep pass (median 0.229ms,
# flat 2->40 steps) stubbed the model OBJECT, which skips the dominant,
# O(history) cost of a real agent step: converting messages to wire format,
# serialising the WHOLE accumulated conversation into the request body, and
# parsing the response back. That's why it looked flat — see
# CALIBRATION-NOTES.md, "Framework CPU measurement — review" and the 4 Sep
# entry below it for the full story. This revision mocks the model at the
# TRANSPORT layer instead (scripts/measure_framework_cpu.py +
# scripts/bedrock_transport_mock.py): a real `langchain_aws.ChatBedrockConverse`
# client does real message serialisation and response parsing; only a
# botocore `before-send` hook is faked, verified before trusting any number
# (`verify_transport_mock()` asserts request body size actually grows with
# history — 1631 -> 19617 bytes, step 2 -> 40 — before the sweep runs).
# langgraph 0.6.11, langchain-aws 0.2.35, langchain-core 0.3.86, Python 3.9.6,
# dedicated .venv, no paid API calls, no network, no real AWS credentials
# (dummy local strings only, used for SigV4 signing math that's never sent).
# 40-step conversations x 40 replicates x 3 checkpointer variants (none /
# InMemorySaver / SqliteSaver), agent_side_synchronous verified on every
# replicate INCLUDING the model-call thread this time (ChatBedrockConverse's
# whole call chain, botocore included, confirmed single-threaded — LangGraph's
# ToolNode is still the one place that genuinely isn't, handled as before by
# tracking its worker-thread CPU explicitly rather than assuming it away).
#
# Per-step CPU decomposes into three buckets (see CALIBRATION-NOTES.md item 3
# for the full distributions): (a) model-path serialisation + response
# parsing — now the dominant, GROWING term; (b) tool dispatch + pydantic
# validation — flat, unaffected by history, unchanged from the first pass;
# (c) checkpoint write — unaffected by this fix, InMemorySaver's ~6x growth
# 2->40 steps and SqliteSaver's end-of-graph write-batching both still stand,
# since checkpointing was never stubbed either time.
#
# The flat-across-history finding does NOT survive: (a)+(b), pooled across
# checkpointer choice, grows from ~0.52ms at step 2 (request ~1.6KB) to
# ~2.0ms at step 40 (request ~19.6KB) — a real, near-linear relationship with
# REQUEST PAYLOAD BYTES (~0.08 microseconds of CPU per byte, plus a ~0.48ms
# fixed per-call floor), not with step count as such — step count is
# incidental to this agent's particular message sizes. compute_ms_per_step
# below is therefore no longer one number shared by every shape: it's set per
# shape at THAT shape's own step count, not short_tool_loop's number reused
# everywhere as before. Checkpoint cost (bucket c) still is not folded in —
# that remains tracked separately via STATE_IO spans, per the original design.
_MEASURED_COMPUTE_MS_PER_STEP = 0.734          # short_tool_loop, 3 steps, ~2.0KB request
_MEASURED_COMPUTE_MS_PER_STEP_HUMAN_IN_LOOP = 0.806   # human_in_loop, 5 steps, ~3.1KB request
_MEASURED_COMPUTE_MS_PER_STEP_LONG_RESEARCH = 1.155   # long_research, 14 steps, ~7.5KB request —
    # measured with 1 tool call/step; long_research's real tools_per_step=2.0
    # would grow messages faster than this harness did, so treat this as a
    # floor for long_research specifically, not a tight estimate.

DEFAULT_SHAPES: dict[str, TurnShape] = {
    "short_tool_loop": TurnShape(
        name="short_tool_loop",
        steps=3,
        llm=_CALIBRATED_LLM,
        tool=_CALIBRATED_TOOL,
        compute_ms_per_step=_MEASURED_COMPUTE_MS_PER_STEP,
    ),
    "long_research": TurnShape(
        name="long_research",
        steps=14,
        llm=_CALIBRATED_LLM,
        tool=_CALIBRATED_TOOL,
        tools_per_step=2.0,
        state_growth_bytes=4200,
        compute_ms_per_step=_MEASURED_COMPUTE_MS_PER_STEP_LONG_RESEARCH,
    ),
    "human_in_loop": TurnShape(
        name="human_in_loop",
        steps=5,
        llm=_CALIBRATED_LLM,
        tool=_CALIBRATED_TOOL,
        human_wait=LatencyProfile("human", median_s=45.0, sigma=1.1),  # placeholder — not calibrated
        compute_ms_per_step=_MEASURED_COMPUTE_MS_PER_STEP_HUMAN_IN_LOOP,
    ),
}


def _burn_cpu(ms: float) -> None:
    """Do genuine local work, so CPU time moves. sleep() would not."""
    end = time.thread_time() + ms / 1000.0
    x = 0.0
    while time.thread_time() < end:
        for i in range(2000):
            x += math.sqrt(i + 1.0)
    return None


def run_turn(shape: TurnShape, tracer: Tracer, *, seed: int | None = None, sleep: bool = True) -> dict[str, Any]:
    """Execute one synthetic turn against the tracer.

    `sleep=False` skips the actual blocking (useful in tests); the spans are
    then shorter but structurally identical.
    """
    rng = random.Random(seed)
    state: dict[str, Any] = {"messages": [], "scratch": ""}

    def block(profile: LatencyProfile) -> float:
        d = profile.sample(rng)
        if sleep:
            time.sleep(d)
        return d

    with tracer.span(f"turn:{shape.name}", SpanKind.TURN, shape=shape.name):
        for step in range(shape.steps):
            with tracer.span(f"step{step}", SpanKind.STEP, step=step):
                with tracer.span("llm", SpanKind.LLM_CALL, step=step):
                    block(shape.llm)

                n_tools = int(shape.tools_per_step) + (
                    1 if rng.random() < (shape.tools_per_step % 1) else 0
                )
                for t in range(n_tools):
                    with tracer.span(f"tool{t}", SpanKind.TOOL_CALL, step=step):
                        block(shape.tool)

                if shape.human_wait is not None and step == shape.steps // 2:
                    with tracer.span("approval", SpanKind.HUMAN_WAIT, step=step):
                        block(shape.human_wait)

                with tracer.span("glue", SpanKind.COMPUTE, step=step):
                    _burn_cpu(shape.compute_ms_per_step)

                state["messages"].append("x" * shape.state_growth_bytes)
                tracer.record_state(state, step=step)

    return state


def save_profiles(shapes: dict[str, TurnShape], path: str | Path) -> None:
    payload = {
        name: {
            "steps": s.steps,
            "tools_per_step": s.tools_per_step,
            "compute_ms_per_step": s.compute_ms_per_step,
            "state_growth_bytes": s.state_growth_bytes,
            "llm": s.llm.to_dict(),
            "tool": s.tool.to_dict(),
            "human_wait": s.human_wait.to_dict() if s.human_wait else None,
        }
        for name, s in shapes.items()
    }
    Path(path).write_text(json.dumps(payload, indent=2))
