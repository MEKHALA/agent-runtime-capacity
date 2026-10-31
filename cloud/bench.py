"""Transport-mock CPU benchmark, runnable inside a cloud platform.

Same measurement as scripts/measure_framework_cpu.py (LangGraph) and
scripts/measure_strands_cpu.py (Strands), packaged so that Lambda and AgentCore
Runtime can run it on their own vCPUs. Nothing here touches the network: the
model client's socket is faked, so no model is called and no tokens are spent.

    run(framework="langgraph" | "strands", steps=40, replicates=30)

returns {"environment": {...}, "runs": {"none": [ {replicate, iterations: [...]}, ... ]}}
in the schema the analysis scripts already read.
"""
from __future__ import annotations

import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / "scripts"):          # bundle layout: agentprof/, scripts/ at root
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def environment() -> dict:
    cpu_model = None
    info = _read("/proc/cpuinfo") or ""
    for line in info.splitlines():
        if line.lower().startswith(("model name", "cpu model")):
            cpu_model = line.split(":", 1)[1].strip()
            break
    if cpu_model is None and "CPU implementer" in info:      # arm64: no model name line
        part = [l.split(":", 1)[1].strip() for l in info.splitlines() if l.startswith("CPU part")]
        cpu_model = f"arm64 (CPU part {part[0] if part else '?'})"
    pkgs = {}
    for name in ("langgraph", "langgraph-checkpoint", "langchain-core", "langchain-aws",
                 "strands-agents", "boto3", "botocore", "pydantic"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_brand": cpu_model,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os_cpu_count": os.cpu_count(),
        "cgroup_cpu_max": _read("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max": _read("/sys/fs/cgroup/memory.max"),
        "lambda_memory_mb": os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE"),
        "execution_env": os.environ.get("AWS_EXECUTION_ENV"),
        "packages": pkgs,
    }


def _langgraph(steps: int, replicates: int) -> list[dict]:
    from measure_framework_cpu import framework_cpu_per_iteration, run_one_replicate
    runs = []
    for r in range(replicates):
        tracer, dispatch_log, request_log, synchronous = run_one_replicate(None, steps, thread_id=f"cloud-{r}")
        runs.append({"replicate": r, "synchronous": synchronous,
                     "iterations": framework_cpu_per_iteration(tracer, dispatch_log, request_log)})
    return runs


def _strands(steps: int, replicates: int) -> list[dict]:
    from measure_strands_cpu import run_one_replicate
    return [{"replicate": r, "synchronous": True, "iterations": run_one_replicate(steps)}
            for r in range(replicates)]


def run(framework: str = "langgraph", steps: int = 40, replicates: int = 30,
        warmup: int = 2, **_ignored) -> dict:
    """Run `warmup` discarded replicates (imports, lazy initialisation), then
    `replicates` measured ones. Refuses to report if the request body does not
    grow with the conversation: the mock would be at the wrong layer."""
    fn = {"langgraph": _langgraph, "strands": _strands}[framework]
    t0 = time.perf_counter()
    fn(steps, warmup)
    runs = fn(steps, replicates)
    first = runs[0]["iterations"]
    if first[-1]["request_body_bytes"] < 3 * first[1]["request_body_bytes"]:
        raise RuntimeError("request body did not grow with history: mock at the wrong layer")
    return {"framework": framework, "steps": steps, "replicates": replicates,
            "warmup_discarded": warmup, "wall_s": time.perf_counter() - t0,
            "all_synchronous": all(r["synchronous"] for r in runs),
            "environment": environment(), "runs": {"none": runs}}


if __name__ == "__main__":                       # local smoke test
    import json
    out = run(sys.argv[1] if len(sys.argv) > 1 else "langgraph", replicates=2)
    it = out["runs"]["none"][0]["iterations"]
    print(json.dumps(out["environment"], indent=1))
    print(out["framework"], "steps", len(it), "bytes", it[0]["request_body_bytes"], "->",
          it[-1]["request_body_bytes"], "model-path ms step 39:", round(it[-1]["model_path_cpu_s"] * 1000, 3))
