#!/usr/bin/env python3
"""Measure REAL per-step LangGraph framework CPU (not latency), with the
model mocked at the TRANSPORT layer.

`compute_ms_per_step` in DEFAULT_SHAPES was an invented 12ms placeholder; a
first pass (see CALIBRATION-NOTES.md, "Framework CPU measurement — review")
measured 0.229ms/step by stubbing the model OBJECT — which skips the
dominant, O(history) cost of a real agent step: converting the message list
to wire format, serialising the whole accumulated conversation into the
request body, JSON-encoding it, and parsing the response back. That's why
the first pass looked flat across history length: the only thing left that
scaled with history was checkpointing.

This version fakes only the SOCKET: a real `langchain_aws.ChatBedrockConverse`
client (see bedrock_transport_mock.py) does all of that real work, and a
botocore `before-send` hook intercepts just before the bytes would leave the
process. No paid API calls, no network, no real AWS credentials — verified at
startup by `verify_transport_mock()`, which asserts request body size grows
with conversation length before any number from this script is trusted.

Per-step CPU is decomposed into three buckets that map onto the paper's
taxonomy:
    (a) model_path_cpu_s   — message serialisation + response parsing
                              (the whole 'agent' node; now dominated by the
                              real ChatBedrockConverse.invoke() call)
    (b) tool_dispatch_cpu_s — ToolNode dispatch + pydantic validation
                              (unchanged from the stub version — tool
                              dispatch was never stubbed)
    (c) checkpoint_cpu_s    — checkpointer put/put_writes/get_tuple
                              (unchanged — also never stubbed)

Must run under the project venv (has langgraph + langchain-aws installed):

    .venv/bin/python3 scripts/measure_framework_cpu.py --steps 40 --replicates 40
"""
from __future__ import annotations

import argparse
import ast
import json
import operator
import sys
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentprof import Tracer, SpanKind
from agentprof.adapters.langgraph import instrument_node, instrument_state
from agentprof.adapters.openai_like import instrument_chat, instrument_tool

from bedrock_transport_mock import (
    build_mocked_bedrock_client, ScriptedConverseResponses, install_transport_mock,
)

from pydantic import BaseModel, Field
from typing import Literal
from typing_extensions import TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # never actually called — see bedrock_transport_mock.py


# --------------------------------------------------------------------------
# Three real tools: pydantic args_schema (so BaseTool.invoke really validates
# and coerces), plausible business logic, non-trivial return payloads.
# --------------------------------------------------------------------------

class SearchDocsArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    top_k: int = Field(5, ge=1, le=20)


class WeatherArgs(BaseModel):
    city: str = Field(..., min_length=1, max_length=100)
    units: Literal["metric", "imperial"] = "metric"


class CalcArgs(BaseModel):
    expression: str = Field(..., pattern=r"^[0-9\.\+\-\*/\(\)\s]+$", max_length=100)


_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"disallowed expression node: {ast.dump(node)}")


def _search_docs(query: str, top_k: int = 5) -> str:
    results = [{"id": f"doc-{i}", "snippet": f"passage {i} relevant to {query!r}"} for i in range(top_k)]
    return json.dumps({"query": query, "results": results})


def _get_weather(city: str, units: str = "metric") -> str:
    return json.dumps({"city": city, "temp": 21.5, "units": units, "conditions": "cloudy"})


def _run_calculation(expression: str) -> str:
    value = _safe_eval(ast.parse(expression, mode="eval"))
    return json.dumps({"expression": expression, "value": value})


class _DispatchLoggingProxy:
    """LangGraph's ToolNode._func ALWAYS calls tool.invoke() through
    `get_executor_for_config(config)` — a thread pool — even for a single
    tool call in synchronous graph.invoke(). That means pydantic validation
    inside BaseTool.invoke() runs on a worker thread the tracer's root-thread
    STEP span never sees, and its CPU would silently vanish from a same-
    thread subtraction. This proxy sits in ToolNode.tools_by_name in place of
    the real tool (ToolNode only ever calls `.invoke()` on entries there —
    verified against installed langgraph's source, see CALIBRATION-NOTES.md)
    and brackets the FULL invoke (validation + our raw business logic
    together) with time.thread_time() on whatever thread actually runs it.
    In analysis we subtract the nested TOOL_CALL span's cpu (also on that
    same worker thread, so the subtraction is same-thread and valid) to
    isolate validation+packaging from business logic.

    A StructuredTool is a pydantic model and rejects `tool.invoke = ...`
    (no such declared field), which is why this is a separate proxy object
    rather than a monkeypatched attribute."""

    def __init__(self, wrapped: StructuredTool, dispatch_log: list[dict[str, Any]]):
        self._wrapped = wrapped
        self._dispatch_log = dispatch_log
        self.name = wrapped.name

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.thread_time()
        try:
            return self._wrapped.invoke(*args, **kwargs)
        finally:
            self._dispatch_log.append({
                "tool": self.name,
                "thread_id": threading.get_ident(),
                "cpu_s": time.thread_time() - t0,
            })


def build_tools(tracer: Tracer) -> list[StructuredTool]:
    """instrument_tool wraps only the RAW business logic — the nested
    TOOL_CALL span it opens covers just that call (our stand-in for what
    would, in production, be genuine remote execution on the tool's own
    server: real CPU here, but conceptually meant to net out to ~0 local
    cost). The dispatch-logging proxy (wired in by build_tool_node) separately
    captures the worker-thread CPU of everything ToolNode/BaseTool do around
    it — validation, coercion, ToolMessage packaging — which is the
    framework overhead we actually want to measure."""
    return [
        StructuredTool.from_function(
            func=instrument_tool(_search_docs, tracer, name="search_docs"),
            name="search_docs", description="Search internal docs for a query.",
            args_schema=SearchDocsArgs,
        ),
        StructuredTool.from_function(
            func=instrument_tool(_get_weather, tracer, name="get_weather"),
            name="get_weather", description="Get current weather for a city.",
            args_schema=WeatherArgs,
        ),
        StructuredTool.from_function(
            func=instrument_tool(_run_calculation, tracer, name="run_calculation"),
            name="run_calculation", description="Evaluate a basic arithmetic expression.",
            args_schema=CalcArgs,
        ),
    ]


def build_tool_node(tools: list[StructuredTool], dispatch_log: list[dict[str, Any]]) -> ToolNode:
    tool_node = ToolNode(tools)
    for name, t in list(tool_node.tools_by_name.items()):
        tool_node.tools_by_name[name] = _DispatchLoggingProxy(t, dispatch_log)
    return tool_node


class AgentState(TypedDict):
    messages: Annotated[Sequence[Any], add_messages]


def build_llm_with_tools(tracer: Tracer, client: Any, tools: list[StructuredTool]) -> Any:
    """Real ChatBedrockConverse over the transport-mocked client. .bind_tools
    makes every request carry real tool JSON schemas too, like a real agent."""
    llm = ChatBedrockConverse(model_id=MODEL_ID, client=client, max_tokens=256)
    return llm.bind_tools(tools)


def build_graph(tracer: Tracer, checkpointer: Any, llm_with_tools: Any, tool_node: ToolNode):
    wrapped_invoke = instrument_chat(lambda **kw: llm_with_tools.invoke(kw["messages"]), tracer, name="llm")

    def agent_fn(state: AgentState) -> dict[str, Any]:
        ai = wrapped_invoke(messages=state["messages"])
        return {"messages": [ai]}

    def tools_fn(state: AgentState) -> dict[str, Any]:
        return tool_node.invoke(state)

    builder = StateGraph(AgentState)
    builder.add_node("agent", instrument_node(agent_fn, tracer, name="agent"))
    builder.add_node("tools", instrument_node(tools_fn, tracer, name="tools"))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


def instrument_checkpointer(cp: Any, tracer: Tracer) -> Any:
    if cp is None:
        return None
    for method_name in ("put", "put_writes", "get_tuple"):
        original = getattr(cp, method_name)
        setattr(cp, method_name, instrument_state(original, tracer, name=f"checkpoint_{method_name}"))
    return cp


# --------------------------------------------------------------------------
# One replicate.
#
# Threading reality check (found empirically, see CALIBRATION-NOTES.md):
# LangGraph's own graph loop runs each node on the calling thread for our
# linear agent<->tools graph — genuinely synchronous, verified directly. But
# LangGraph's prebuilt ToolNode._func ALWAYS calls
# `with get_executor_for_config(config) as executor: executor.map(...)`,
# even for a single tool call, so tool validation+execution runs on a worker
# thread regardless. We can't make that go away without abandoning the real
# ToolNode (which would defeat "not a toy"), so instead of same-thread
# subtraction we track that worker-thread CPU explicitly via
# `_wrap_tool_dispatch` and sum cross-thread. The agent/LLM/checkpoint side
# stays genuinely single-threaded, verified per replicate below.
# --------------------------------------------------------------------------

def run_one_replicate(
    checkpointer: Any, n_steps: int, thread_id: str
) -> tuple[Tracer, list[dict[str, Any]], list[dict[str, Any]], bool]:
    tracer = Tracer()
    dispatch_log: list[dict[str, Any]] = []
    request_log: list[dict[str, Any]] = []

    client = build_mocked_bedrock_client()
    responses = ScriptedConverseResponses(n_tool_calls=n_steps)
    install_transport_mock(client, responses, request_log)

    tools = build_tools(tracer)
    tool_node = build_tool_node(tools, dispatch_log)
    llm_with_tools = build_llm_with_tools(tracer, client, tools)
    cp = instrument_checkpointer(checkpointer, tracer)
    graph = build_graph(tracer, cp, llm_with_tools, tool_node)

    root_thread = tracer._root_thread
    with tracer.span("turn", SpanKind.TURN):
        graph.invoke(
            {"messages": [HumanMessage(content="Work through this multi-step task.")]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": n_steps * 3 + 10,  # agent+tools per step, plus final agent call
            },
        )

    # Root-thread consistency only matters where we do CROSS-thread
    # subtraction: agent_step.cpu - llm.cpu (both must be root thread, and
    # empirically always are — verified again here, and separately for
    # ChatBedrockConverse/botocore in verify_transport_mock()). Checkpoint
    # puts/get_tuple ran off-thread too in testing (LangGraph backgrounds
    # checkpoint writes) — but we never subtract from them, only sum each
    # span's own same-thread-measured cpu, which is valid regardless of which
    # thread ran it. So only STEP and LLM_CALL spans, plus the before-send
    # interceptor's own thread, need to be checked here; TOOL_CALL and
    # checkpoint STATE_IO spans are handled by construction (see docstring).
    agent_side_synchronous = (
        all(s.thread_id == root_thread for s in tracer.spans if s.kind in (SpanKind.STEP, SpanKind.LLM_CALL))
        and all(r["thread_id"] == root_thread for r in request_log)
    )
    return tracer, dispatch_log, request_log, agent_side_synchronous


def framework_cpu_per_iteration(
    tracer: Tracer, dispatch_log: list[dict[str, Any]], request_log: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pair each 'agent' STEP span with the 'tools' STEP span that follows it,
    the dispatch_log entry from that iteration's tool call, and the
    request_log entry from that iteration's model call (all four advance in
    lockstep — one model call and one tool call per iteration in this
    harness) to decompose per-iteration CPU into three buckets that map onto
    the paper's taxonomy:

    (a) model_path_cpu_s = agent_step.cpu (root thread)
        The whole 'agent' node — now dominated by the REAL
        ChatBedrockConverse.invoke() call (message->wire conversion, full-
        history JSON serialisation, response parsing back into AIMessage),
        not subtracted out as it was when the model was a stub. The tiny
        LangGraph node-dispatch glue around that call is also in here (see
        llm_call_cpu_s / agent_node_overhead_cpu_s below for the split, kept
        for transparency though not used in the headline number).

    (b) tool_dispatch_cpu_s = tools_step.cpu (root thread: executor setup/
        dispatch/collect) + (dispatch_log.cpu_s - tool_call_span.cpu) (worker
        thread: pydantic validation + ToolMessage packaging, our raw business
        logic netted out via same-thread subtraction, which is valid since
        both operands are measured on the same worker thread within the same
        call). Unchanged from the stub version — tool dispatch was never
        stubbed.

    (c) checkpoint_cpu_s: every `checkpoint_*` STATE_IO span's own cpu
        (each one is internally same-thread valid, see instrument_state)
        attributed to whichever STEP most recently completed before it
        fired — LangGraph issues checkpoint writes right after a node
        returns, often on a background thread, so this is causal/positional
        attribution, not span-tree nesting. Unchanged from the stub version.

    request_body_bytes is the size of THIS iteration's serialised request —
    the generalisable x-axis (see CALIBRATION-NOTES.md item 4), since step
    count is incidental but payload size is what the work actually scales
    with.

    The final agent call (ends the loop, no tool_calls) has no paired tools
    step and is dropped, not folded into the sweep.
    """
    agent_steps = [s for s in tracer.spans if s.name == "agent" and s.kind is SpanKind.STEP]
    tools_steps = [s for s in tracer.spans if s.name == "tools" and s.kind is SpanKind.STEP]
    tool_call_spans = [s for s in tracer.spans if s.kind is SpanKind.TOOL_CALL]

    def nested_cpu(parent_span, kind) -> float:
        return sum(s.cpu for s in tracer.spans if s.parent_id == parent_span.span_id and s.kind is kind)

    checkpoint_cpu_by_step_id: dict[str, float] = {}
    last_step_span = None
    for s in tracer.spans:
        if s.kind is SpanKind.STEP:
            last_step_span = s
        elif s.kind is SpanKind.STATE_IO and s.name.startswith("checkpoint_") and last_step_span is not None:
            checkpoint_cpu_by_step_id[last_step_span.span_id] = (
                checkpoint_cpu_by_step_id.get(last_step_span.span_id, 0.0) + s.cpu
            )

    iterations = []
    n = min(len(agent_steps), len(tools_steps), len(dispatch_log), len(tool_call_spans), len(request_log))
    for i in range(n):
        agent_span, tools_span = agent_steps[i], tools_steps[i]

        llm_call_cpu = nested_cpu(agent_span, SpanKind.LLM_CALL)
        model_path_cpu = agent_span.cpu

        dispatch_only_cpu = dispatch_log[i]["cpu_s"] - tool_call_spans[i].cpu
        tool_dispatch_cpu = tools_span.cpu + dispatch_only_cpu

        checkpoint_cpu = (
            checkpoint_cpu_by_step_id.get(agent_span.span_id, 0.0)
            + checkpoint_cpu_by_step_id.get(tools_span.span_id, 0.0)
        )

        iterations.append({
            "iteration": i,
            "request_body_bytes": request_log[i]["body_bytes"],
            "model_path_cpu_s": model_path_cpu,
            "llm_call_cpu_s": llm_call_cpu,
            "agent_node_overhead_cpu_s": model_path_cpu - llm_call_cpu,
            "tool_dispatch_cpu_s": tool_dispatch_cpu,
            "checkpoint_cpu_s": checkpoint_cpu,
            "total_cpu_s": model_path_cpu + tool_dispatch_cpu + checkpoint_cpu,
        })
    return iterations


# --------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------

CHECKPOINTER_FACTORIES = {
    "none": lambda db_path: None,
    "memory": lambda db_path: InMemorySaver(),
}


def verify_transport_mock(steps: int = 40) -> None:
    """Item 2's requirement, non-negotiable: PROVE the mock is at the right
    layer before trusting any number from this script. If request body size
    doesn't grow with conversation length, the interception point is too
    early (e.g. before message serialisation) and every downstream CPU
    number would be measuring the wrong thing — exactly the error this
    whole re-run exists to correct. Abort loudly rather than proceed."""
    print("=== verifying transport mock is at the right layer ===")
    tracer, _dispatch_log, request_log, synchronous = run_one_replicate(
        None, n_steps=steps, thread_id="verify"
    )
    if not synchronous:
        print("*** FATAL: agent/llm work ran off the tracer's root thread during ")
        print("*** verification. Aborting before trusting anything downstream.")
        sys.exit(1)

    checkpoints = [c for c in (2, 5, 10, 20, 40) if c <= steps]
    sizes = []
    for c in checkpoints:
        idx = c - 1
        if idx >= len(request_log):
            print(f"  FATAL: only {len(request_log)} requests captured, need index {idx} for step {c}")
            sys.exit(1)
        b = request_log[idx]["body_bytes"]
        sizes.append(b)
        print(f"  step {c:3d}: request body = {b:6d} bytes")

    if not all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1)):
        print("\n*** FATAL: request body size is not monotonically growing with ***")
        print("*** conversation length. The interception point is too early    ***")
        print("*** (before message serialisation) or something else is wrong.  ***")
        print("*** Refusing to trust any CPU number from this harness until    ***")
        print("*** that's fixed — see CALIBRATION-NOTES.md item 2.             ***")
        sys.exit(1)

    growth = sizes[-1] / sizes[0]
    print(f"  body size grew {growth:.1f}x from step {checkpoints[0]} to step "
          f"{checkpoints[-1]} — mock confirmed at the right layer.\n")


def capture_environment() -> dict:
    """Record everything needed to reproduce or reinterpret these timings.

    CPU-time constants are meaningless without the machine and interpreter
    that produced them: a newer CPython alone moves them by tens of per cent.
    Captured automatically so the numbers can never drift from their context.
    """
    import platform, sys, subprocess, datetime

    def _sysctl(key: str) -> str:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    env = {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
    }
    if sys.platform == "darwin":
        env["cpu_brand"] = _sysctl("machdep.cpu.brand_string")
        env["physical_cores"] = _sysctl("hw.physicalcpu")
        env["logical_cores"] = _sysctl("hw.logicalcpu")
        env["memory_bytes"] = _sysctl("hw.memsize")

    pkgs = {}
    try:
        from importlib import metadata
        for name in ("langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite",
                     "langgraph-prebuilt", "langchain-core", "langchain-aws", "boto3"):
            try:
                pkgs[name] = metadata.version(name)
            except Exception:
                pass
    except Exception:
        pass
    env["packages"] = pkgs
    return env


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--replicates", type=int, default=8)
    p.add_argument("--out", default="data/framework_cpu_raw.json")
    p.add_argument("--sqlite-path", default="data/framework_cpu_checkpoints.sqlite")
    args = p.parse_args()

    verify_transport_mock(args.steps)

    env = capture_environment()
    print(f"environment: {env.get('cpu_brand', env['machine'])} | "
          f"{env['python_implementation']} {env['python_version']} | "
          f"langgraph {env['packages'].get('langgraph', '?')} | "
          f"langgraph-checkpoint {env['packages'].get('langgraph-checkpoint', '?')}")

    results: dict[str, Any] = {"environment": env, "steps": args.steps,
                               "replicates": args.replicates, "runs": {}}
    all_synchronous = True

    for cp_name in ("none", "memory", "sqlite"):
        print(f"=== checkpointer={cp_name} ===")
        runs = []
        for r in range(args.replicates):
            if cp_name == "sqlite":
                import sqlite3
                db_path = args.sqlite_path
                for suffix in ("", "-wal", "-shm"):
                    Path(db_path + suffix).unlink(missing_ok=True)
                conn = sqlite3.connect(db_path, check_same_thread=False)
                checkpointer = SqliteSaver(conn)
                checkpointer.setup()
            else:
                checkpointer = CHECKPOINTER_FACTORIES[cp_name](None)

            tracer, dispatch_log, request_log, synchronous = run_one_replicate(
                checkpointer, args.steps, thread_id=f"{cp_name}-{r}",
            )
            if cp_name == "sqlite":
                conn.close()
                for suffix in ("", "-wal", "-shm"):
                    Path(db_path + suffix).unlink(missing_ok=True)

            all_synchronous = all_synchronous and synchronous
            iterations = framework_cpu_per_iteration(tracer, dispatch_log, request_log)
            runs.append({"replicate": r, "synchronous": synchronous, "iterations": iterations})
            print(f"  replicate {r + 1}/{args.replicates}: "
                  f"{len(iterations)} iterations, agent_side_synchronous={synchronous}")

        results["runs"][cp_name] = runs

    results["all_synchronous"] = all_synchronous
    if not all_synchronous:
        print("\n*** WARNING: agent/llm/checkpoint work ran off the tracer's root thread ***")
        print("*** in at least one replicate. That side of the measurement has no cross- ***")
        print("*** thread correction (unlike the known ToolNode executor, see module ***")
        print("*** docstring), so time.thread_time() would be silently wrong. STOPPING ***")
        print("*** without writing a report figure.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        return 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}  (agent/llm/checkpoint side confirmed synchronous on the "
          f"tracer's root thread; ToolNode's executor-thread CPU is tracked explicitly)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
