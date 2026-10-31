"""Direct check of the capacity calculation (Little's law) against measured CPU.

The paper computes "CPU cores busy while N sessions are held" as
    N * c / (w + c)
from per-step measurements of wait w and CPU c. This script runs N real
LangGraph agent sessions concurrently on one machine and measures the CPU the
processes actually consume with the operating system's counters, so that the
calculation can be compared with a direct measurement.

Each session is the transport-mocked LangGraph agent of measure_framework_cpu.py
(40 steps, growing history), except that every model call now blocks for a wait
drawn from the 1,094 deployed LangGraph model calls (data/deployed_turns.csv),
via time.sleep inside the socket mock. Sessions run back-to-back in each worker
thread, so N sessions are held at all times. Workers are spread across several
processes so that Python's global interpreter lock does not serialise them.

For each N we report:
  measured   total process CPU over a fixed window / window length
  predicted  N * c / (w + c), with c measured separately (one session at a time,
             no wait) and w the mean of the waits actually drawn in the window
  little     (model calls completed per second in the window) * c

    .venv/bin/python scripts/validate_capacity.py
Output: results/capacity_validation.json
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

N_GRID = (50, 100, 200, 400)
PROCESSES = 5            # this machine has 6 cores; leave one for the OS
STEPS = 40
WARMUP_S, WINDOW_S = 150, 300
SEED = 20261028


def load_waits() -> list[float]:
    rows = csv.DictReader(l for l in open(ROOT / "data" / "deployed_turns.csv") if not l.startswith("#"))
    return [float(r["model_ms"]) / 1000 for r in rows if r["config"] == "c2-langgraph-agentcore"]


def _session(rng, waits, counters, lock, sleep: bool) -> None:
    """One 40-step LangGraph session over a socket mock that sleeps for a drawn wait."""
    import boto3
    from botocore.awsrequest import AWSResponse
    from langchain_core.messages import HumanMessage
    from agentprof import Tracer
    from bedrock_transport_mock import ScriptedConverseResponses, _FakeRawResponse
    from measure_framework_cpu import build_graph, build_llm_with_tools, build_tool_node, build_tools

    tracer = Tracer()
    client = boto3.session.Session().client("bedrock-runtime", region_name="us-east-1",
                                            aws_access_key_id="test", aws_secret_access_key="test")
    responses = ScriptedConverseResponses(n_tool_calls=STEPS)

    def intercept(request, **_):
        if sleep:
            w = waits[rng.randrange(len(waits))]
            time.sleep(w)
        else:
            w = 0.0
        with lock:
            counters["calls"] += 1
            counters["wait_s"] += w
        return AWSResponse(request.url, 200, {"Content-Type": "application/json"},
                           _FakeRawResponse(responses.next_body()))

    client.meta.events.register("before-send.bedrock-runtime.Converse", intercept)
    tools = build_tools(tracer)
    graph = build_graph(tracer, None, build_llm_with_tools(tracer, client, tools), build_tool_node(tools, []))
    graph.invoke({"messages": [HumanMessage(content="Work through this multi-step task.")]},
                 config={"configurable": {"thread_id": "v"}, "recursion_limit": STEPS * 3 + 10})


def worker(n_threads: int, seed: int, t_start: float, t_end: float, waits: list[float], q) -> None:
    import gc, random
    if os.environ.get("VC_GC_DISABLE") == "1":   # diagnostic: is the extra CPU garbage collection?
        gc.disable()
    counters, lock, stop = {"calls": 0, "wait_s": 0.0}, threading.Lock(), threading.Event()

    def loop(k: int) -> None:
        rng = random.Random(seed * 1000 + k)
        time.sleep(rng.uniform(0, 3.0))          # stagger starts
        while not stop.is_set():
            _session(rng, waits, counters, lock, sleep=True)

    for k in range(n_threads):
        threading.Thread(target=loop, args=(k,), daemon=True).start()
    time.sleep(max(0.0, t_start - time.time()))
    cpu0, calls0, wait0 = time.process_time(), counters["calls"], counters["wait_s"]
    time.sleep(max(0.0, t_end - time.time()))
    cpu1, calls1, wait1 = time.process_time(), counters["calls"], counters["wait_s"]
    q.put({"cpu_s": cpu1 - cpu0, "calls": calls1 - calls0, "wait_s": wait1 - wait0})
    q.close(); q.join_thread()                   # flush before exiting, or the result is lost
    stop.set()
    os._exit(0)                                  # abandon in-flight sessions


def baseline_c(n_sessions: int = 30) -> float:
    """CPU per model call, one session at a time, no wait: the c of the formula."""
    import random
    counters, lock = {"calls": 0, "wait_s": 0.0}, threading.Lock()
    rng = random.Random(SEED)
    _session(rng, [0.0], counters, lock, sleep=False)        # warm-up
    counters.update(calls=0, wait_s=0.0)
    t0 = time.process_time()
    for _ in range(n_sessions):
        _session(rng, [0.0], counters, lock, sleep=False)
    return (time.process_time() - t0) / counters["calls"]


def main(tag: str = "") -> None:
    waits = load_waits()
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    c = None
    out = {"processes": PROCESSES, "steps": STEPS, "warmup_s": WARMUP_S, "window_s": WINDOW_S,
           "machine": os.uname().machine, "runs": []}
    for n in N_GRID:
        # Measure c fresh before each N, in a child, so that it reflects the same interpreter state.
        p = ctx.Process(target=lambda qq: qq.put(baseline_c()), args=(q,)); p.start(); c = q.get(); p.join()
        print(f"N={n}: baseline c = {c * 1000:.2f} ms per model call; starting {n} sessions", flush=True)
        per = [n // PROCESSES + (1 if i < n % PROCESSES else 0) for i in range(PROCESSES)]
        t_start = time.time() + WARMUP_S; t_end = t_start + WINDOW_S
        procs = [ctx.Process(target=worker, args=(k, SEED + i, t_start, t_end, waits, q)) for i, k in enumerate(per)]
        for pr in procs:
            pr.start()
        res = [q.get() for _ in procs]
        for pr in procs:
            pr.join()
        cpu, calls, wsum = sum(r["cpu_s"] for r in res), sum(r["calls"] for r in res), sum(r["wait_s"] for r in res)
        w_mean = wsum / calls
        row = {"sessions": n, "c_ms": c * 1000, "w_mean_s": w_mean, "calls": calls,
               "measured_cores": cpu / WINDOW_S,
               "predicted_cores": n * c / (w_mean + c),
               "little_cores": calls / WINDOW_S * c}
        row["measured_over_predicted"] = row["measured_cores"] / row["predicted_cores"]
        out["runs"].append(row)
        print(f"N={n:4d}: c={row['c_ms']:.2f} ms, mean wait {w_mean:.2f} s | measured {row['measured_cores']:.3f} cores, "
              f"predicted {row['predicted_cores']:.3f}, from throughput {row['little_cores']:.3f} "
              f"(measured/predicted {row['measured_over_predicted']:.2f})", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    out["gc_disabled"] = os.environ.get("VC_GC_DISABLE") == "1"
    name = f"capacity_validation{('_' + tag) if tag else ''}.json"
    (ROOT / "results" / name).write_text(json.dumps(out, indent=1))
    print(f"wrote results/{name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
