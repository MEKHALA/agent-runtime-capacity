#!/usr/bin/env python3
"""Summarize data/framework_cpu_raw.json (transport-mocked version):
distributions, not means; per-checkpointer bucket decomposition; and CPU
against request payload BYTES, not just step index (see CALIBRATION-NOTES.md
item 4 — bytes is the generalisable x-axis, step count is incidental)."""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path

CHECKPOINTS = [2, 5, 10, 20, 40]
BUCKETS = ["model_path_cpu_s", "tool_dispatch_cpu_s", "checkpoint_cpu_s", "total_cpu_s"]


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
    return xs[k]


def dist_ms(xs_s):
    xs = [x * 1000 for x in xs_s]
    return {
        "n": len(xs),
        "p10": round(pct(xs, 0.10), 5),
        "p50": round(pct(xs, 0.50), 5),
        "p90": round(pct(xs, 0.90), 5),
        "mean": round(statistics.mean(xs), 5) if xs else None,
    }


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/framework_cpu_raw.json"
    d = json.loads(Path(path).read_text())
    print(f"steps={d['steps']} replicates={d['replicates']}\n")

    print("=== bucket decomposition, pooled across all steps x replicates (ms) ===")
    for cp, runs in d["runs"].items():
        print(f"--- checkpointer={cp} ---")
        for bucket in BUCKETS:
            xs = [it[bucket] for r in runs for it in r["iterations"]]
            print(f"  {bucket:24s} {dist_ms(xs)}")
        print()

    print("--- model_path_cpu_ms vs step index (history length) ---")
    header = "checkpointer".ljust(12) + "".join(f"step={c:<10}" for c in CHECKPOINTS)
    print(header)
    for cp, runs in d["runs"].items():
        row = cp.ljust(12)
        for c in CHECKPOINTS:
            idx = c - 1
            vals = [r["iterations"][idx]["model_path_cpu_s"] * 1000 for r in runs if idx < len(r["iterations"])]
            row += f"{statistics.median(vals):<10.4f}" if vals else f"{'--':<10}"
        print(row)

    print("\n--- request_body_bytes vs step index (confirms the x we actually care about) ---")
    print(header)
    for cp, runs in d["runs"].items():
        row = cp.ljust(12)
        for c in CHECKPOINTS:
            idx = c - 1
            vals = [r["iterations"][idx]["request_body_bytes"] for r in runs if idx < len(r["iterations"])]
            row += f"{statistics.median(vals):<10.0f}" if vals else f"{'--':<10}"
        print(row)

    print("\n=== model_path_cpu_ms vs REQUEST PAYLOAD BYTES (pooled across checkpointers) ===")
    print("(bytes bucketed into round(bytes/250)*250-wide bins; this is the generalisable")
    print(" relationship — step count is incidental to this specific agent's message sizes)")
    all_iterations = [it for runs in d["runs"].values() for r in runs for it in r["iterations"]]
    by_bin: dict[int, list[float]] = {}
    for it in all_iterations:
        b = round(it["request_body_bytes"] / 250) * 250
        by_bin.setdefault(b, []).append(it["model_path_cpu_s"] * 1000)
    for b in sorted(by_bin):
        vals = by_bin[b]
        print(f"  ~{b:5d} bytes  (n={len(vals):4d})  median={statistics.median(vals):.4f}ms  "
              f"p90={pct(sorted(vals), 0.9):.4f}ms")

    # simple linear fit: cpu_ms ~ a + b * bytes, to quantify $/byte cost
    xs = [it["request_body_bytes"] for it in all_iterations]
    ys = [it["model_path_cpu_s"] * 1000 for it in all_iterations]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var = sum((x - mean_x) ** 2 for x in xs)
    slope = cov / var if var else 0.0
    intercept = mean_y - slope * mean_x
    print(f"\n  linear fit: model_path_cpu_ms ~= {intercept:.4f} + {slope:.6f} * bytes  (n={n})")
    print(f"  i.e. ~{slope*1000:.3f} microseconds of CPU per extra byte of request body")

    return 0


if __name__ == "__main__":
    sys.exit(main())
