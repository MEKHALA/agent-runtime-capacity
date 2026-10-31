#!/usr/bin/env python3
"""Record real latency samples for agentprof calibration.

Two things get timed, each with a discarded warm-up call first (connection
setup + TLS handshake would otherwise skew sample #1):

  1. Model calls, with output length VARIED across the sample. LLM latency is
     dominated by output tokens, so 50 calls at a fixed tiny max_tokens would
     produce a distribution that looks nothing like real agent steps (which
     range from a one-line tool-call decision to a paragraph of reasoning).
     max_tokens is drawn log-uniformly across a realistic agent-turn range and
     the ACTUAL completion token count is recorded alongside each latency.

  2. Tool calls, against a couple of realistic remote HTTP endpoints.

Output: a JSON file (default data/latency_raw.json) with "llm" and "tool"
sample lists plus a "meta" block recording provider/model/region/conditions,
suitable as-is for a `calibration_context` when fitting with
`agentprof.LatencyProfile.from_samples`.

Providers are pluggable behind one small interface (`ModelProvider`) so
switching endpoints later is a --provider flag, not a rewrite:

    python3 scripts/record_latencies.py --dry-run                       # stub, no network, no cost
    python3 scripts/record_latencies.py --provider bedrock \\
        --model anthropic.claude-3-5-haiku-20241022-v1:0 --region us-east-1
    python3 scripts/record_latencies.py --provider anthropic --model claude-haiku-4-5-20251001
    python3 scripts/record_latencies.py --provider openai --model gpt-4o-mini

Real (non-dry-run, non-stub) runs print a cost estimate and ask for
confirmation before spending anything; pass --yes to skip the prompt once
you've seen the estimate and are happy with it.
"""
from __future__ import annotations

import abc
import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# Provider interface — the one small seam that makes --provider a flag
# --------------------------------------------------------------------------

@dataclass
class CompletionResult:
    prompt_tokens: int
    completion_tokens: int
    model: str
    region: str | None = None
    raw_usage: dict[str, Any] | None = None


class ModelProvider(abc.ABC):
    """Anything that can run one completion and report token usage.

    A new backend needs only this method implemented; the recording loop,
    warm-up, sampling and cost-estimation logic are provider-agnostic.
    """

    name: str = "base"

    def __init__(self, model_id: str, region: str | None = None):
        self.model_id = model_id
        self.region = region

    @abc.abstractmethod
    def complete(self, prompt: str, max_tokens: int) -> CompletionResult:
        ...


class StubProvider(ModelProvider):
    """No network, no credentials, no cost. Exercises the full recording path
    (warm-up, sampling, timing, progress printing, JSON output) for --dry-run."""

    name = "stub"

    def __init__(self, model_id: str = "stub-model", region: str | None = None, *, seed: int = 1234):
        super().__init__(model_id, region)
        self._rng = random.Random(seed)

    def complete(self, prompt: str, max_tokens: int) -> CompletionResult:
        # Fake latency that scales with requested length, like a real model,
        # scaled down so a dry run of 50+80 calls finishes in a few seconds.
        base = 0.01 + 0.0004 * max_tokens
        time.sleep(base * self._rng.uniform(0.7, 1.4))
        completion_tokens = self._rng.randint(max(1, int(max_tokens * 0.2)), max_tokens)
        prompt_tokens = max(8, len(prompt.split()))
        return CompletionResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model_id,
            region=self.region,
            raw_usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        )


class BedrockProvider(ModelProvider):
    """AWS Bedrock via the provider-agnostic Converse API — works the same way
    whether model_id points at an Anthropic, Meta, Amazon, or other model on
    Bedrock, so this class doesn't need to know which."""

    name = "bedrock"

    def __init__(self, model_id: str, region: str | None = None):
        if not region:
            raise ValueError("--region is required for --provider bedrock")
        super().__init__(model_id, region)
        import boto3  # deferred: only required for this provider

        self._client = boto3.client("bedrock-runtime", region_name=region)

    def complete(self, prompt: str, max_tokens: int) -> CompletionResult:
        resp = self._client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens},
        )
        usage = resp.get("usage", {})
        return CompletionResult(
            prompt_tokens=usage.get("inputTokens", 0),
            completion_tokens=usage.get("outputTokens", 0),
            model=self.model_id,
            region=self.region,
            raw_usage=dict(usage),
        )


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(self, model_id: str, region: str | None = None):
        super().__init__(model_id, region=None)
        import anthropic  # deferred: only required for this provider

        self._client = anthropic.Anthropic()

    def complete(self, prompt: str, max_tokens: int) -> CompletionResult:
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = resp.usage
        return CompletionResult(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            model=self.model_id,
            region=None,
            raw_usage={"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        )


class OpenAIProvider(ModelProvider):
    name = "openai"

    def __init__(self, model_id: str, region: str | None = None):
        super().__init__(model_id, region=None)
        import openai  # deferred: only required for this provider

        self._client = openai.OpenAI()

    def complete(self, prompt: str, max_tokens: int) -> CompletionResult:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = resp.usage
        return CompletionResult(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model=self.model_id,
            region=None,
            raw_usage={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
        )


PROVIDERS = {
    "stub": StubProvider,
    "bedrock": BedrockProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def build_provider(name: str, model_id: str, region: str | None) -> ModelProvider:
    cls = PROVIDERS[name]
    return cls(model_id, region)


# --------------------------------------------------------------------------
# Rough, approximate pricing — verify against the provider's current pricing
# page before trusting a cost estimate for anything but a sanity check.
# Override with --price-in-per-1k / --price-out-per-1k if you know better.
# --------------------------------------------------------------------------

APPROX_PRICE_PER_1K: dict[str, tuple[float, float]] = {
    # model id substring -> (input $/1k tokens, output $/1k tokens)
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-3-5-haiku": (0.0008, 0.004),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-haiku-4-5": (0.001, 0.005),  # applies to us.anthropic.claude-haiku-4-5-* inference profile ids too
    "claude-sonnet-5": (0.003, 0.015),
    "claude-opus-5": (0.015, 0.075),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}


def lookup_price(model_id: str) -> tuple[float, float] | None:
    for key, price in APPROX_PRICE_PER_1K.items():
        if key in model_id:
            return price
    return None


# --------------------------------------------------------------------------
# Prompts and output-length sampling for the LLM leg
# --------------------------------------------------------------------------

AGENT_PROMPTS = [
    "You are an autonomous coding agent. Given the failing test output below, "
    "explain the likely root cause and propose the next single tool call to "
    "make, with reasoning. Test output:\n\nAssertionError: expected 3 rows, got 0",
    "You are an agent planning a multi-step task. The user asked you to find "
    "and summarize the three most relevant files for implementing rate "
    "limiting in this repo. You have just run a grep and gotten 40 hits. "
    "Decide what to do next and explain your reasoning.",
    "Continue this agent transcript. The previous tool call returned a 404. "
    "Diagnose why, and either retry with a fix or choose a different "
    "approach. Be specific about what changed and why you expect it to work.",
    "Summarize the following log excerpt for a human reviewer, calling out "
    "anything that looks like an error, a retry, or unexpectedly high "
    "latency:\n\n[fetch] 200 OK 812ms\n[fetch] 200 OK 4210ms\n[fetch] 503 "
    "retrying...\n[fetch] 200 OK 340ms",
    "You are deciding whether the current plan for this agent turn is still "
    "on track after three steps, or whether to replan. Weigh the evidence "
    "and give a short recommendation with your reasoning.",
    "Write the next tool call as a single JSON object, then explain in one "
    "or two sentences why that tool and those arguments follow from the "
    "state so far.",
]


def sample_max_tokens(rng: random.Random, lo: int, hi: int) -> int:
    """Log-uniform across [lo, hi]: agent turns range from a short tool-call
    decision (tens of tokens) to a long reasoning chain (over a thousand)."""
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


# --------------------------------------------------------------------------
# Tool-call leg: real remote HTTP endpoints, no credentials needed
# --------------------------------------------------------------------------

DEFAULT_TOOL_ENDPOINTS: list[tuple[str, str]] = [
    ("github_zen", "https://api.github.com/zen"),
    ("httpbin_get", "https://httpbin.org/get"),
]

# Used in --dry-run instead of the real endpoints above, so a smoke test never
# implies real HTTP calls were made when the "tool" caller is actually stubbed.
STUB_TOOL_ENDPOINTS: list[tuple[str, str]] = [
    ("stub_endpoint_a", "stub://endpoint-a"),
    ("stub_endpoint_b", "stub://endpoint-b"),
]


def time_tool_call(url: str, *, timeout: float = 10.0) -> tuple[float, int]:
    req = urllib.request.Request(url, headers={"User-Agent": "agentprof-calibration/0.1"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
        status = resp.status
    return time.monotonic() - t0, status


def make_stub_tool_caller(seed: int = 4321):
    """No network: exercises the tool-call recording path for --dry-run."""
    rng = random.Random(seed)

    def _call(url: str, *, timeout: float = 10.0) -> tuple[float, int]:
        t0 = time.monotonic()
        time.sleep(rng.uniform(0.01, 0.08))
        return time.monotonic() - t0, 200

    return _call


# --------------------------------------------------------------------------
# Recording loops
# --------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_llm_samples(
    provider: ModelProvider,
    *,
    n: int,
    lo_tokens: int,
    hi_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    warmup_tokens = sample_max_tokens(rng, lo_tokens, hi_tokens)
    warmup_prompt = rng.choice(AGENT_PROMPTS)
    print(f"[llm] warm-up call (max_tokens={warmup_tokens}) ... ", end="", flush=True)
    provider.complete(warmup_prompt, warmup_tokens)
    print("done, discarded")

    price = lookup_price(provider.model_id)
    running_cost = 0.0
    samples: list[dict[str, Any]] = []

    for i in range(n):
        prompt = rng.choice(AGENT_PROMPTS)
        max_tokens = sample_max_tokens(rng, lo_tokens, hi_tokens)

        t0 = time.monotonic()
        result = provider.complete(prompt, max_tokens)
        latency_s = time.monotonic() - t0

        sample = {
            "latency_s": latency_s,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "max_tokens_requested": max_tokens,
            "model": result.model,
            "region": result.region,
            "timestamp": now_iso(),
        }
        samples.append(sample)

        cost_note = ""
        if price is not None:
            in_price, out_price = price
            call_cost = (result.prompt_tokens / 1000) * in_price + (result.completion_tokens / 1000) * out_price
            running_cost += call_cost
            cost_note = f"  running cost est ~${running_cost:.4f}"

        print(
            f"[llm] {i + 1}/{n}  latency={latency_s:.3f}s  "
            f"completion_tokens={result.completion_tokens} (max={max_tokens})"
            f"{cost_note}"
        )

    return samples


def record_tool_samples(
    endpoints: list[tuple[str, str]], *, n: int, caller=time_tool_call
) -> list[dict[str, Any]]:
    # One discarded warm-up per endpoint: same rationale as the LLM leg,
    # connection + TLS setup skews the first sample against each host.
    for name, url in endpoints:
        print(f"[tool] warm-up call to {name} ... ", end="", flush=True)
        try:
            caller(url)
            print("done, discarded")
        except urllib.error.URLError as e:
            print(f"failed ({e}) — still discarded, will retry in main loop")

    samples: list[dict[str, Any]] = []
    for i in range(n):
        name, url = endpoints[i % len(endpoints)]
        try:
            latency_s, status = caller(url)
        except urllib.error.URLError as e:
            print(f"[tool] {i + 1}/{n}  {name} FAILED: {e}")
            continue
        samples.append(
            {
                "latency_s": latency_s,
                "endpoint": name,
                "url": url,
                "status": status,
                "timestamp": now_iso(),
            }
        )
        print(f"[tool] {i + 1}/{n}  {name}  latency={latency_s:.3f}s  status={status}")

    return samples


# --------------------------------------------------------------------------
# Cost pre-flight and confirmation
# --------------------------------------------------------------------------

def preflight_cost_estimate(model_id: str, n_llm: int, lo_tokens: int, hi_tokens: int) -> float | None:
    price = lookup_price(model_id)
    if price is None:
        return None
    in_price, out_price = price
    # Worst case: every call maxes out prompt (~200 tokens, our prompts are
    # short) and hits hi_tokens of output.
    assumed_prompt_tokens = 200
    worst_call = (assumed_prompt_tokens / 1000) * in_price + (hi_tokens / 1000) * out_price
    return worst_call * n_llm


def confirm_or_abort(message: str, *, assume_yes: bool) -> None:
    if assume_yes:
        return
    reply = input(f"{message} Proceed? [y/N] ").strip().lower()
    if reply not in ("y", "yes"):
        print("aborted.")
        sys.exit(1)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=sorted(PROVIDERS), default="stub")
    p.add_argument("--model", help="model id (required for bedrock/anthropic/openai)")
    p.add_argument("--region", help="AWS region (bedrock only)")
    p.add_argument("--n-llm", type=int, default=50, help="number of LLM calls to time (>=50 required)")
    p.add_argument("--n-tool", type=int, default=30, help="number of tool calls to time (>=30 required)")
    p.add_argument("--lo-tokens", type=int, default=48, help="lower bound of max_tokens sampling range")
    p.add_argument("--hi-tokens", type=int, default=1536, help="upper bound of max_tokens sampling range")
    p.add_argument("--tool-endpoint", action="append", metavar="NAME=URL",
                    help="repeatable; overrides the default tool endpoints")
    p.add_argument("--out", default="data/latency_raw.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                    help="force the stub provider, skip cost confirmation, no network/cost")
    p.add_argument("--yes", action="store_true", help="skip the cost-confirmation prompt")
    p.add_argument("--price-in-per-1k", type=float, help="override input $/1k tokens for cost estimate")
    p.add_argument("--price-out-per-1k", type=float, help="override output $/1k tokens for cost estimate")
    p.add_argument("--conditions", default="", help="free-text note on conditions (time of day, load, etc.)")
    args = p.parse_args()

    if args.price_in_per_1k is not None and args.price_out_per_1k is not None:
        APPROX_PRICE_PER_1K[args.model or ""] = (args.price_in_per_1k, args.price_out_per_1k)

    provider_name = "stub" if args.dry_run else args.provider
    model_id = args.model or "stub-model"

    if provider_name != "stub" and not args.model:
        p.error(f"--model is required for --provider {provider_name}")
    if provider_name == "bedrock" and not args.region:
        p.error("--region is required for --provider bedrock")

    endpoints = STUB_TOOL_ENDPOINTS if args.dry_run else DEFAULT_TOOL_ENDPOINTS
    if args.tool_endpoint:
        endpoints = []
        for spec in args.tool_endpoint:
            name, _, url = spec.partition("=")
            if not url:
                p.error(f"--tool-endpoint must be NAME=URL, got {spec!r}")
            endpoints.append((name, url))

    n_llm, n_tool = args.n_llm, args.n_tool
    if args.dry_run:
        # Fast smoke test by default; still exercises every code path.
        n_llm = min(n_llm, 8) if args.n_llm == 50 else n_llm
        n_tool = min(n_tool, 6) if args.n_tool == 30 else n_tool
    else:
        if n_llm < 50:
            p.error("--n-llm must be >= 50 for a real recording (this is enforced deliberately)")
        if n_tool < 30:
            p.error("--n-tool must be >= 30 for a real recording (this is enforced deliberately)")

    print(f"provider={provider_name} model={model_id} region={args.region} "
          f"n_llm={n_llm} n_tool={n_tool} dry_run={args.dry_run}")

    if not args.dry_run and provider_name != "stub":
        est = preflight_cost_estimate(model_id, n_llm, args.lo_tokens, args.hi_tokens)
        if est is None:
            print(f"cost estimate: unknown pricing for model {model_id!r} — "
                  f"pass --price-in-per-1k/--price-out-per-1k for an estimate.")
            confirm_or_abort("No pricing on file, so no cost estimate is possible.", assume_yes=args.yes)
        else:
            print(f"worst-case cost estimate for {n_llm} LLM calls: ~${est:.3f} "
                  f"(approximate — verify against the provider's current pricing page)")
            confirm_or_abort(f"Estimated worst-case spend is ~${est:.3f}.", assume_yes=args.yes)

    provider = build_provider(provider_name, model_id, args.region)

    llm_samples = record_llm_samples(
        provider, n=n_llm, lo_tokens=args.lo_tokens, hi_tokens=args.hi_tokens, seed=args.seed
    )
    tool_caller = make_stub_tool_caller(args.seed) if args.dry_run else time_tool_call
    tool_samples = record_tool_samples(endpoints, n=n_tool, caller=tool_caller)

    payload = {
        "meta": {
            "provider": provider_name,
            "model": model_id,
            "region": args.region,
            "recorded_at": now_iso(),
            "n_llm_samples": len(llm_samples),
            "n_tool_samples": len(tool_samples),
            "lo_tokens": args.lo_tokens,
            "hi_tokens": args.hi_tokens,
            "tool_endpoints": [name for name, _ in endpoints],
            "conditions": args.conditions,
            "dry_run": args.dry_run,
        },
        "llm": llm_samples,
        "tool": tool_samples,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {len(llm_samples)} llm samples and {len(tool_samples)} tool samples to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
