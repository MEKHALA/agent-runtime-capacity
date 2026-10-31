#!/usr/bin/env python3
"""Measure real per-step Strands framework CPU, with the model mocked at the
TRANSPORT layer: the Strands counterpart of measure_framework_cpu.py.

Same design as the LangGraph script, so the two are comparable:
  * a real strands BedrockModel over a real boto3 client; only the socket is
    faked, by the same botocore before-send hook (bedrock_transport_mock.py),
    returning the same scripted Converse responses (40 tool calls, then a
    final answer);
  * the same three tools, with the same argument schemas and return values;
  * no system prompt, max_tokens 256, full history retained (Strands' default
    sliding-window conversation manager is replaced by NullConversationManager,
    otherwise history is truncated and the payload stops growing).

Two differences are forced by Strands' architecture and stated here so they are
not mistaken for results:
  * Strands owns its agent loop and executes the tools itself, so there is no
    graph and no checkpointer: only the 'none' configuration exists.
  * BedrockModel runs the boto3 call in a worker thread (asyncio.to_thread) and
    processes the response on the event-loop thread. Per-thread CPU therefore
    splits one model call across two threads. We measure PROCESS CPU time over
    the window between Strands' own BeforeModelCallEvent and AfterModelCallEvent
    hooks instead. The loop is strictly sequential (SequentialToolExecutor, one
    tool per step), so nothing else runs inside that window.

    .venv/bin/python scripts/measure_strands_cpu.py --steps 40 --replicates 40
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import boto3
from strands import Agent, tool
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.hooks import (AfterModelCallEvent, AfterToolCallEvent, BeforeModelCallEvent,
                           BeforeToolCallEvent, HookProvider, HookRegistry)
from strands.models.bedrock import BedrockModel
from strands.tools.executors.sequential import SequentialToolExecutor

from bedrock_transport_mock import ScriptedConverseResponses, install_transport_mock
from measure_framework_cpu import MODEL_ID, _get_weather, _run_calculation, _search_docs


# Same names, parameters and descriptions as the LangGraph tools, so the tool
# schemas serialised into every request are equivalent.
@tool
def search_docs(query: str, top_k: int = 5) -> str:
    """Search internal docs for a query."""
    return _search_docs(query, top_k)


@tool
def get_weather(city: str, units: str = "metric") -> str:
    """Get current weather for a city."""
    return _get_weather(city, units)


@tool
def run_calculation(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    return _run_calculation(expression)


class CpuProbe(HookProvider):
    """Process CPU over each model call and each tool call, from Strands' own
    lifecycle hooks."""

    def __init__(self) -> None:
        self.model: list[float] = []
        self.tools: list[float] = []
        self._t = 0.0

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._start)
        registry.add_callback(AfterModelCallEvent, lambda e: self.model.append(time.process_time() - self._t))
        registry.add_callback(BeforeToolCallEvent, self._start)
        registry.add_callback(AfterToolCallEvent, lambda e: self.tools.append(time.process_time() - self._t))

    def _start(self, _e: Any) -> None:
        self._t = time.process_time()


def run_one_replicate(n_steps: int) -> list[dict[str, Any]]:
    session = boto3.Session(aws_access_key_id="test", aws_secret_access_key="test",
                            region_name="us-east-1")
    model = BedrockModel(model_id=MODEL_ID, boto_session=session, streaming=False, max_tokens=256)
    request_log: list[dict[str, Any]] = []
    install_transport_mock(model.client, ScriptedConverseResponses(n_tool_calls=n_steps), request_log)
    probe = CpuProbe()
    agent = Agent(model=model, tools=[search_docs, get_weather, run_calculation],
                  conversation_manager=NullConversationManager(),
                  tool_executor=SequentialToolExecutor(),
                  callback_handler=None, hooks=[probe])
    agent("Work through this multi-step task.")
    # Iteration = one model call that returned a tool call + that tool call.
    # The final answer call (index n_steps) is excluded, as in the LangGraph script.
    if len(probe.tools) != n_steps or len(probe.model) != n_steps + 1 or len(request_log) != n_steps + 1:
        raise RuntimeError(f"unexpected loop shape: model={len(probe.model)} tools={len(probe.tools)} "
                           f"requests={len(request_log)}")
    return [{"iteration": i, "request_body_bytes": request_log[i]["body_bytes"],
             "model_path_cpu_s": probe.model[i], "tool_dispatch_cpu_s": probe.tools[i],
             "checkpoint_cpu_s": 0.0, "total_cpu_s": probe.model[i] + probe.tools[i]}
            for i in range(n_steps)]


def verify_transport_mock(steps: int = 40) -> None:
    """Refuse to report anything unless the request body grows with history:
    proof that the mock sits below the serialisation we mean to measure."""
    it = run_one_replicate(steps)
    b0, b1 = it[1]["request_body_bytes"], it[-1]["request_body_bytes"]
    print(f"  request body {b0} -> {b1} bytes over {steps} steps")
    if b1 < 3 * b0:
        raise SystemExit("request body did not grow with history: mock is at the wrong layer")


def capture_environment() -> dict[str, Any]:
    pkgs = ["strands-agents", "boto3", "botocore", "pydantic"]
    return {"recorded_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "machine": platform.machine(), "processor": platform.processor(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "packages": {p: md.version(p) for p in pkgs},
            "cpu_clock": "time.process_time() over hook-delimited windows"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--replicates", type=int, default=40)
    p.add_argument("--out", default="data/strands_cpu_raw_x86.json")
    a = p.parse_args()
    print("=== verifying transport mock is at the right layer ===")
    verify_transport_mock(a.steps)
    env = capture_environment()
    runs = []
    for r in range(a.replicates):
        runs.append({"replicate": r, "synchronous": True, "iterations": run_one_replicate(a.steps)})
        if (r + 1) % 10 == 0:
            print(f"  replicate {r + 1}/{a.replicates}")
    out = {"environment": env, "steps": a.steps, "replicates": a.replicates,
           "runs": {"none": runs}, "all_synchronous": True}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
