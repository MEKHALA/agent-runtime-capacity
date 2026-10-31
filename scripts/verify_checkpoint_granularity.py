"""Confirm empirically what InMemorySaver stores per write.

Claim under test (§5.2): LangGraph checkpointers write incrementally at CHANNEL
granularity, but store the entire value of each changed channel — no delta. For an
`add_messages` channel the stored blob is therefore the whole conversation, and grows
with every step.

Run under the project venv:   .venv/bin/python3 scripts/verify_checkpoint_granularity.py
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import empty_checkpoint

saver = InMemorySaver()
cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
msgs = []

for step in range(1, 7):
    msgs = msgs + [{"role": "assistant", "content": "x" * 500, "id": f"m{step}"}]
    ck = empty_checkpoint()
    ck["id"] = f"c{step}"
    ck["channel_values"] = {"messages": list(msgs), "counter": step}
    ck["channel_versions"] = {"messages": str(step), "counter": str(step)}
    # only `messages` is declared changed on steps > 1
    new_versions = {"messages": str(step)} if step > 1 else ck["channel_versions"]
    saver.put(cfg, ck, {"source": "loop", "step": step}, new_versions)

print(f"{'channel':>10}{'version':>9}{'blob bytes':>12}")
for (thread, ns, chan, ver), (typ, blob) in sorted(saver.blobs.items(), key=lambda kv: (kv[0][2], kv[0][3])):
    print(f"{chan:>10}{ver:>9}{len(blob):>12}")

msg_blobs = [len(b) for (t, n, c, v), (ty, b) in saver.blobs.items() if c == "messages"]
print(f"\nmessages blobs: {len(msg_blobs)}, sizes {min(msg_blobs)}..{max(msg_blobs)} bytes")
print("Each holds the FULL message list at that version — incremental per channel,")
print("whole-value within a channel. `counter` is written once; `messages` every step.")
