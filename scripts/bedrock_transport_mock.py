"""A real langchain-aws ChatBedrockConverse client with only the SOCKET faked.

Registers a botocore `before-send` event handler, which fires after the SDK
has done all its real work — building the Converse request body from the
LangChain message list, JSON-encoding it, SigV4-signing it — and short-
circuits just before the bytes would hit a socket, returning a synthetic
response. Every real client-side cost this measurement cares about
(message→wire conversion, full-conversation serialisation, JSON encode,
response parse back into AIMessage) still runs for real; only the network is
fake. See CALIBRATION-NOTES.md, "Framework CPU measurement — review" for why
this replaces the previous model-object stub.

Credentials are local dummy strings ("test"/"test") used only so botocore's
SigV4 signer has something to sign with — signing happens before-send fires,
but the signed request is never transmitted, so no real credential ever
reaches a network or an AWS account.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable

import boto3
from botocore.awsrequest import AWSResponse


def build_mocked_bedrock_client(region: str = "us-east-1"):
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


class _FakeRawResponse:
    """Minimal file-like object botocore's HTTP layer accepts in place of a
    real urllib3 response: .read() once, .stream() for the streaming path."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self, *_a: Any, **_kw: Any) -> bytes:
        data, self._data = self._data, b""
        return data

    def stream(self, **_kw: Any):
        yield self._data


_TOOL_CYCLE = [
    ("search_docs", lambda i: {"query": f"topic {i}", "top_k": 3 + (i % 5)}),
    ("get_weather", lambda i: {"city": ["Paris", "Tokyo", "Nairobi", "Lima"][i % 4], "units": "metric"}),
    ("run_calculation", lambda i: {"expression": f"({i} + 3) * 2 / {(i % 4) + 1}"}),
]

_FINAL_ANSWER_TEXT = (
    "Based on everything gathered across the steps above, here is a complete "
    "summary: the searched documents were relevant, the weather lookups "
    "returned plausible conditions, and the calculations checked out. No "
    "further tool calls are needed to answer the original request."
)


class ScriptedConverseResponses:
    """Same scripted plan as the old StubModel (cycle through 3 tools, then a
    final answer) but emitting real Bedrock Converse wire-format JSON bodies,
    so response PARSING cost (not just request serialisation) is real too."""

    def __init__(self, n_tool_calls: int):
        self.n_tool_calls = n_tool_calls
        self.call_count = 0

    def next_body(self) -> bytes:
        i = self.call_count
        self.call_count += 1
        if i < self.n_tool_calls:
            name, args_fn = _TOOL_CYCLE[i % len(_TOOL_CYCLE)]
            content = [
                {"text": "Let me check that."},
                {"toolUse": {"toolUseId": f"call_{i}", "name": name, "input": args_fn(i)}},
            ]
            stop_reason = "tool_use"
        else:
            content = [{"text": _FINAL_ANSWER_TEXT}]
            stop_reason = "end_turn"
        payload = {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": stop_reason,
            "usage": {"inputTokens": 40 + i * 15, "outputTokens": 25, "totalTokens": 65 + i * 15},
        }
        return json.dumps(payload).encode("utf-8")


def install_transport_mock(
    client: Any, responses: ScriptedConverseResponses, request_log: list[dict[str, Any]]
) -> None:
    """Registers the interceptor. request_log gets one entry per intercepted
    call: body_bytes (the proof the mock is at the right layer — see
    verify_transport_mock below) and thread_id (proof it's single-threaded,
    unlike ToolNode's executor)."""

    def intercept(request: Any, **_kwargs: Any) -> AWSResponse:
        body = request.body if isinstance(request.body, (bytes, bytearray)) else (request.body or b"").encode()
        request_log.append({"body_bytes": len(body), "thread_id": threading.get_ident()})
        return AWSResponse(
            request.url, 200, {"Content-Type": "application/json"},
            _FakeRawResponse(responses.next_body()),
        )

    client.meta.events.register("before-send.bedrock-runtime.Converse", intercept)
