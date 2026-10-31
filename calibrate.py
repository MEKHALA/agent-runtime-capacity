#!/usr/bin/env python3
"""Fit latency profiles from REAL calls, once. Then replay them forever.

This is the step that makes the substrate benchmark both cheap and
reproducible: you spend tokens here, once, and never again.

    # 1. record real latencies (you supply the call)
    # 2. fit and save
    python3 calibrate.py --demo            # shows the workflow with fake data

In your own code:

    from agentprof import LatencyProfile
    samples = []
    for _ in range(50):
        t0 = time.monotonic()
        client.chat.completions.create(model=..., messages=...)
        samples.append(time.monotonic() - t0)
    prof = LatencyProfile.from_samples("llm", samples)
    print(prof.to_dict())      # paste into your TurnShape

Fifty samples per endpoint is enough to fit a lognormal well. Record at the
time of day you care about; tail latency moves with load.
"""
from __future__ import annotations
import argparse, json, random, sys
from agentprof import LatencyProfile


def _extract_latencies(value: list) -> list[float]:
    """Accept either a flat list of floats, or scripts/record_latencies.py's
    richer list of {"latency_s": ..., ...} sample dicts."""
    if value and isinstance(value[0], dict):
        return [s["latency_s"] for s in value]
    return value


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", help="JSON file: {\"llm\": [1.2, 0.9, ...], \"tool\": [...]} "
                                      "or scripts/record_latencies.py's {\"meta\": {...}, \"llm\": [...], \"tool\": [...]}")
    p.add_argument("--out", default="profiles.json")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()

    calibration_context = None
    if a.demo:
        rng = random.Random(0)
        truth = LatencyProfile("llm", median_s=2.4, sigma=0.5)
        data = {"llm": [truth.sample(rng) for _ in range(200)]}
        print("demo: fitting 200 samples drawn from median=2.4s sigma=0.5")
    elif a.samples:
        data = json.load(open(a.samples))
        calibration_context = data.pop("meta", None)
    else:
        p.error("pass --samples FILE or --demo")

    fitted = {
        k: LatencyProfile.from_samples(k, _extract_latencies(v), calibration_context=calibration_context).to_dict()
        for k, v in data.items()
    }
    print(json.dumps(fitted, indent=2))
    with open(a.out, "w") as f:
        json.dump(fitted, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
