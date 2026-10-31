#!/usr/bin/env python3
"""Characterise agent turn shapes. Phase 2 of the paper.

    python3 run_characterise.py --shape long_research --runs 5
    python3 run_characterise.py --all --runs 3 --fast --json out.json
"""
from __future__ import annotations
import argparse, sys
from agentprof import DEFAULT_SHAPES, Tracer, run_turn, summarise, report


def characterise(shape_name: str, runs: int, fast: bool, seed0: int = 0):
    shape = DEFAULT_SHAPES[shape_name]
    out = []
    for i in range(runs):
        t = Tracer()
        run_turn(shape, t, seed=seed0 + i, sleep=not fast)
        out.append(summarise(t.spans))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=sorted(DEFAULT_SHAPES))
    p.add_argument("--all", action="store_true")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--fast", action="store_true", help="skip real blocking (structure only)")
    p.add_argument("--price-per-gb-second", type=float, default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    names = sorted(DEFAULT_SHAPES) if a.all else [a.shape or "short_tool_loop"]
    blobs = []
    for name in names:
        rs = characterise(name, a.runs, a.fast)
        print()
        print(report.turn_report(rs[0], title=f"{name}  (single run)"))
        if a.price_per_gb_second:
            print(report.cost_note(rs[0], gb_seconds_price=a.price_per_gb_second))
        print()
        print(report.aggregate_report(rs, title=f"{name}  aggregate"))
        blobs.append(report.to_json(rs, shape=name))

    if a.json:
        with open(a.json, "w") as f:
            f.write("[" + ",".join(blobs) + "]")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
