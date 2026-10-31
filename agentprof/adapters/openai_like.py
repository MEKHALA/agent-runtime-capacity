"""Wrap any OpenAI-shaped chat client so model calls become LLM_CALL spans.

    from agentprof import Tracer
    from agentprof.adapters.openai_like import instrument_chat

    tracer = Tracer()
    client.chat.completions.create = instrument_chat(
        client.chat.completions.create, tracer
    )

Works for anything with the same call shape — Bedrock's converse API, LiteLLM,
vLLM's OpenAI-compatible server — because it only wraps the callable.
"""
from __future__ import annotations

import functools
from typing import Any, Callable

from ..tracer import SpanKind, Tracer


def instrument_chat(fn: Callable[..., Any], tracer: Tracer, *, name: str = "llm") -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with tracer.span(name, SpanKind.LLM_CALL, model=kwargs.get("model")) as s:
            out = fn(*args, **kwargs)
            usage = getattr(out, "usage", None)
            if usage is not None:
                s.attrs["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
                s.attrs["completion_tokens"] = getattr(usage, "completion_tokens", None)
            return out

    return wrapper


def instrument_tool(fn: Callable[..., Any], tracer: Tracer, *, name: str | None = None) -> Callable[..., Any]:
    label = name or getattr(fn, "__name__", "tool")

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with tracer.span(label, SpanKind.TOOL_CALL):
            return fn(*args, **kwargs)

    return wrapper
