# Calibration status — 3 Sep 2026

## What is calibrated

50 real Bedrock calls (`us.anthropic.claude-haiku-4-5`, us-east-1, Converse API,
max_tokens log-uniform over [48, 1536]) and 30 real tool calls. Spend ~$0.044.

- `llm`  median 4.23s, sigma 0.58
- `tool` median 0.57s, sigma 0.89

## `compute_ms_per_step` — MEASURED, not assumed, as of today

The 29 Aug entry below proposed a sensitivity table over six *guessed* values
(12ms–600ms) because nobody had measured real framework CPU yet. That
measurement is now done: `scripts/measure_framework_cpu.py` drives a real
LangGraph `StateGraph` — actual `ToolNode` (not a hand-rolled dispatcher), 3
tools with pydantic `args_schema` (real JSON-schema validation via
`BaseTool.invoke`), a real checkpointer (none / `InMemorySaver` /
`SqliteSaver`) — for 40-step conversations × 40 replicates × 3 checkpointer
variants, no paid API calls (a stub model returns scripted tool calls; the
calibrated `llm` `LatencyProfile` above is sampled for a plausible number but
never actually slept for — this measures CPU, which sleep duration cannot
affect). `time.thread_time()` validity (genuinely single-threaded on the
tracer's root thread) was checked and confirmed on every one of the 120
replicates before any number was trusted — see "the one non-synchronous part"
below.

**Result: median 0.229ms/step** (p10 0.217, p90 0.245, n=1600), essentially
flat across checkpointer choice and across accumulated history length 2→40
steps. That is **~52x smaller than the 12ms placeholder this table used to
call "the invented placeholder"** — every row in the original sensitivity
table below, including the one explicitly labeled "the invented placeholder,"
overestimated real framework CPU.

Old table, for the record (turn CPU/idle%/wait:compute holding the calibrated
latencies fixed at 3 steps, idle_s ≈ 14.4s — an illustrative back-of-envelope
calc, not the live tool's actual output):

| compute/step | turn CPU | idle % | wait:compute | plausible source |
|---|---|---|---|---|
| 12 ms | 0.036s | 99.8% | 400:1 | the invented placeholder |
| 25 ms | 0.075s | 99.5% | 192:1 | thin wrapper, no validation |
| 50 ms | 0.150s | 99.0% | 96:1 | modest framework overhead |
| 150 ms | 0.450s | 97.0% | 32:1 | LangGraph-ish per node |
| 300 ms | 0.900s | 94.1% | 16:1 | pydantic + templating + callbacks |
| 600 ms | 1.800s | 88.9% | 8:1 | heavy framework, large state |

**What `python3 run_characterise.py --shape short_tool_loop --runs 20` actually
prints now** (real number, not a back-of-envelope calc — this is what changed
when `DEFAULT_SHAPES` was updated with the measured 0.229ms/step):
`turn_cpu_s_p50 = 0.004034s`, `wait_to_compute_ratio_p50 = 4338:1`. (Turn CPU
here is higher than the raw `3 x 0.229ms = 0.7ms` you'd naively expect —
`run_turn`'s own span bookkeeping and `_burn_cpu`'s busy-loop overshoot add
real, separate CPU on top of the measured LangGraph dispatch figure. That's
expected and fine; it's still ~9x smaller turn CPU, hence ~9x larger ratio,
than the 478:1 the invented 12ms placeholder produced.)

### Does this make the paper's claim stronger or weaker? Read this before quoting a ratio.

Both, in different senses — say so explicitly in the methodology section:

- **The underlying economic claim gets stronger, not weaker.** "You are billed
  for network wait, not compute" is now MORE true, empirically, not less: a
  real LangGraph agent with real tool schemas and a real checkpointer costs
  under a quarter of a millisecond of CPU per step. There was no slack in the
  12ms placeholder to begin with — the real number is smaller.
- **The specific ratio gets LESS credible to a reviewer, not more.** 4,338:1
  (the live `run_characterise.py` figure, n=20 real runs — up nearly 10x from
  the invented placeholder's 478:1) is an extraordinary number, and
  extraordinary numbers invite exactly the skepticism CALIBRATION-NOTES.md
  was written to get ahead of. A reviewer's first hypothesis for a number
  this large will be "the compute measurement is broken," not "billing is
  this lopsided." **Do not lead with the ratio.** Lead with the two
  measured quantities separately (median LLM latency, median framework CPU)
  and let the reader compute the ratio themselves if they want it — a
  reviewer trusts "4.23s vs 0.23ms, both independently measured and
  reproducible" far more than a pre-computed "4,338:1."
- **This measurement is a floor, not a ceiling.** The harness is a real but
  *minimal* LangGraph agent: 3 tools, no LangSmith tracing, no output
  parsers beyond tool-call schema validation, short message content, no
  retry/guardrail middleware. Production agents carry more of that, and it
  all adds real CPU. Say explicitly in the paper that 0.229ms/step is a
  lower bound for LangGraph's own dispatch overhead, not an upper bound for
  what agents cost in practice.
- **Locality still inflates the numerator** (see caveat 1 below), which
  *shrinks* the true ratio somewhat versus what's reported here — partially,
  not close to fully, offsetting the above.

### A secondary finding: checkpoint cost, not framework dispatch, is where state size actually bites

The sweep also varied checkpointer (none / `InMemorySaver` / `SqliteSaver`)
and history length (checkpoints at step 2, 5, 10, 20, 40). Two things worth
their own line in the paper:

- **Framework dispatch overhead (the 0.229ms above) does NOT grow with
  accumulated history** in this harness — flat from step 2 to step 40 for
  all three checkpointer variants. This contradicts the expectation stated
  when this measurement was proposed. Say so rather than quietly dropping
  the prediction.
- **`InMemorySaver`'s checkpoint cost DOES grow with history — about 6x from
  step 2 (~0.05ms) to step 40 (~0.30ms)** — it copies/serialises the whole
  accumulated state on every write, so cost scales with total state size.
  This is the state-externalisation tax the paper argues about, and it is
  real and measured, just not where the original guess placed it (framework
  dispatch, not checkpoint I/O).
- **`SqliteSaver` does NOT show the same clean per-step growth** — LangGraph
  defers/batches a large fraction of `put()`/`put_writes()` calls to a burst
  at graph completion rather than paying evenly per step (confirmed by
  inspecting wall-clock order of spans: ~24 extra `checkpoint_put` calls
  landed after the last real step in an 8-step trace). Report SQLite
  checkpoint cost as a whole-run average, not a per-step-index curve — the
  per-step attribution is real but lumpy for reasons that are about
  LangGraph's write-batching, not about state size.

### The one non-synchronous part, and how it's handled

LangGraph's own graph loop runs each node on the calling thread for this
linear agent↔tools graph — confirmed empirically, genuinely synchronous, no
special-casing needed for the agent/LLM/checkpoint-read side.

**LangGraph's prebuilt `ToolNode._func`, however, unconditionally dispatches
every tool call through `get_executor_for_config(config)` — a thread pool —
even for exactly one tool call in synchronous `graph.invoke()`.** Confirmed
by reading the installed `langgraph==0.6.11` source, not assumed. That means
pydantic validation inside `BaseTool.invoke()` runs on a worker thread the
tracer's root-thread STEP span never sees; a naive same-thread
`step.cpu - tool_call.cpu` subtraction would silently and invisibly drop
exactly the schema-validation cost this whole exercise exists to measure.

Fix: a proxy sits in `ToolNode.tools_by_name` (which only ever calls
`.invoke()` on its entries — verified against source) and brackets the full
worker-thread `invoke()` with `time.thread_time()` directly, logged
separately. Framework CPU for the tools side of a step is then computed as
`tools_step.cpu` (root thread: executor setup/dispatch/collect) `+`
`(worker_thread_invoke_cpu - tool_call_span.cpu)` (worker thread: validation
+ packaging, business logic netted out via same-thread subtraction, which
*is* valid since both operands are measured on the same worker thread within
the same call). Every replicate's agent/LLM/checkpoint spans were separately
confirmed on the tracer's root thread before any number here was trusted —
see `scripts/measure_framework_cpu.py`'s module docstring for the full
reasoning.

`agentprof/adapters/langgraph.py` had one real bug found running it against
real LangGraph for the first time: `instrument_node` hardcoded `step=-1` on
every call instead of tracking the real step index. Fixed, with new tests
(`tests/test_agentprof.py::TestLangGraphAdapter`, no LangGraph install
required — they exercise the same calling convention with fakes).

## Three further caveats for the methodology section

1. **Locality.** Recorded from a laptop in Asia/Calcutta to us-east-1, so the
   LLM median includes local network and geographic RTT. A Lambda co-located
   in us-east-1 sees less. This inflates the numerator, so the true in-region
   ratio is *lower* than measured — the same direction as the (now resolved)
   compute error, but this one still stands.
2. **Tool sigma 0.89 is a mixture artifact.** GitHub Zen and httpbin are
   unrelated endpoints; fitting one lognormal across both produces an
   inflated spread. Fit them separately, or state plainly that `tool` is a
   deliberate mixture.
3. **n=50, one time of day.** Adequate for a median, thin for a tail. Record
   a second sample at a different hour and check the medians agree before
   quoting any percentile.

`human_wait` remains an explicit uncalibrated placeholder — correctly so,
since there is nothing real to fit it against.

## Status

Both halves of the headline ratio (`llm`/`tool` latency, `compute_ms_per_step`
framework CPU) are now measured against real endpoints/real frameworks, not
invented. What remains before quoting a figure in the paper: decide how to
present the ratio given the credibility caveat above, and address the three
methodology caveats (especially locality and the tool-sigma mixture).

---

# Framework CPU measurement — review, 30 Aug 2026

Measured 0.229 ms/step (n=1600, flat across history 2→40 and across checkpointer).
Two predictions were wrong: the 12 ms placeholder, and my own estimate of
100–300 ms. But 0.229 ms is also wrong, in the other direction, and the reason
matters more than the number.

## The stub model bypasses the dominant per-step cost

The measurement replaces the *model object* with a stub returning scripted tool
calls. That is the right instinct for avoiding spend, but it removes the largest
CPU consumer in a real agent step:

- converting the message list into the provider's wire format
- **serialising the entire accumulated conversation into the request body**
- JSON-encoding that body
- parsing the response JSON back into message objects
- tokenisation / token counting where enabled

A real agent re-serialises its whole history on *every* step. That work is O(history)
and it is precisely the externalisation tax the paper argues about — and the stub
skips all of it.

## Which means the "flat across history" finding is probably an artifact

Framework dispatch overhead being flat 2→40 steps is reported as contradicting
expectation. More likely it is a property of the harness: the only remaining
history-dependent work was checkpointing, which is exactly where growth *was*
observed (InMemorySaver ~6x, step 2→40). Dispatch looked flat because the
history-dependent path was stubbed out.

Order-of-magnitude check: JSON-encoding a 40-turn conversation of roughly 200 KB
costs single-digit milliseconds, before message-object construction. So real
per-step CPU at long history is plausibly 10–30x the measured figure, and rising
with history rather than flat.

## The fix: mock at the transport layer, not the model layer

Keep zero spend, but let the real provider client do its real serialisation work
and fake only the network:

- `ChatBedrockConverse` with a **botocore Stubber**, or
- an `httpx.MockTransport` under the real client

Then every conversion, encode and parse runs for real; only the socket is fake.
Re-run the history sweep against that. If per-step CPU then grows with history,
that is a genuine finding and a figure in the paper. If it stays flat, the flat
result is real and much better evidenced.

## What stands regardless

- The `step=-1` adapter bug, found and fixed.
- The ToolNode thread-pool dispatch wrinkle, verified from source. Keep that
  reasoning in the methodology section; it is the kind of detail that earns trust.
- **InMemorySaver checkpoint cost growing ~6x with history.** This is the real
  state-externalisation measurement and it survives the critique, because
  checkpointing was never stubbed.
- SqliteSaver deferring writes to a burst at graph completion. Worth a sentence:
  on a serverless substrate a graph completes once per invocation, so that burst
  is paid on every turn rather than amortised.
- The recommendation not to lead with the ratio. Report 4.23 s and the measured
  CPU as two separate quantities and let the reader form the ratio. An extreme
  ratio reads as broken instrumentation, not as a finding.

## Status (superseded — see the transport-layer re-run below)

`_MEASURED_COMPUTE_MS_PER_STEP = 0.229` is a **lower bound from a stubbed model**,
not the per-step cost of a real agent. Label it that way in DEFAULT_SHAPES and in
any draft, until the transport-layer re-run replaces it.

---

# Framework CPU measurement — transport-layer re-run, 4 Sep 2026

Done. The critique above was correct: **the "flat across history" finding does
NOT survive.** Model-path CPU grows with request size; it only looked flat
because the model was stubbed at the object layer, which skips exactly the
work that scales with history.

## What changed

The model is now a real `langchain_aws.ChatBedrockConverse` client
(`scripts/bedrock_transport_mock.py`), doing genuine message→wire conversion,
full-conversation JSON serialisation, and response parsing. Only the socket
is fake: a botocore `before-send` event handler intercepts after signing,
before any bytes would leave the process, and returns a scripted response
built from the SAME tool-call plan the old model stub used. Dummy local
credentials (`"test"`/`"test"`) exist only so the SigV4 signer has something
to sign — never transmitted, no network, no real AWS account touched.

**Proof the mock is at the right layer** (`verify_transport_mock()`, run
automatically before every sweep, aborts loudly if this fails):

| step | request body |
|---|---|
| 2  | 1,631 bytes |
| 5  | 3,137 bytes |
| 10 | 5,219 bytes |
| 20 | 10,300 bytes |
| 40 | 19,617 bytes |

Monotonic, ~12x growth step 2→40, matching the full accumulated conversation
being re-serialised every call. Confirmed identical across all three
checkpointer variants (checkpointing doesn't touch the request body, as
expected). `ChatBedrockConverse.invoke()` → botocore → the before-send hook
was also re-verified single-threaded on every one of 120 replicates
(`agent_side_synchronous`) — LangGraph's `ToolNode` remains the only
genuinely multi-threaded part, handled exactly as before.

## Decomposition (item 3): three buckets, pooled across 40 steps x 40 replicates

| bucket | none | memory | sqlite |
|---|---|---|---|
| (a) model_path_cpu_ms (serialise+parse) | p50 1.213, p10 0.633, p90 1.836 | p50 1.249 | p50 1.256 |
| (b) tool_dispatch_cpu_ms (ToolNode+pydantic) | p50 0.207 | p50 0.225 | p50 0.226 |
| (c) checkpoint_cpu_ms | 0 (no checkpointer) | p50 0.157, p90 0.569 | p50 0, p90 0.209 (batched — see below) |
| total_cpu_ms | p50 1.419 | p50 1.670 | p50 1.539 |

**(a) is now the dominant, growing term** — roughly 6x the size of (b), and
unlike (b) or (c) it is NOT flat with history (below). (b) is unchanged from
the stub version, as expected: tool dispatch was never stubbed. (c) is also
unchanged in kind, though the exact multiplier moved slightly (this run: ~9x
InMemorySaver growth 2→40 vs ~6x on 3 Sep — same finding, run-to-run noise on
the multiplier, not a contradiction).

## CPU against request payload bytes (item 4) — the real relationship

Step count was never the right x-axis; it's incidental to this agent's
particular message sizes. Bytes is generalisable:

| request size | model_path_cpu_ms (median) |
|---|---|
| ~1.6 KB (step 2)  | 0.54 |
| ~3.1 KB (step 5)  | 0.63 |
| ~5.2 KB (step 10) | 0.83 |
| ~10.3 KB (step 20) | 1.23 |
| ~19.6 KB (step 40) | 2.03 |

Linear fit across all 4,800 samples: `model_path_cpu_ms ≈ 0.478 + 0.000081 *
bytes` — i.e. a ~0.48ms fixed floor per call (HTTP/SDK machinery, client-side
setup) plus ~0.08 microseconds of CPU per additional byte of request body.
R² isn't reported here but the per-checkpoint medians above are essentially
monotonic and smooth; this is a clean, near-linear relationship, not noise
dressed up as a trend.

One artifact worth naming rather than hiding: the very *first* call in each
replicate (smallest body, ~1KB) costs *more* CPU than the fit predicts —
plausibly lazy imports / regex compilation / cache warm-up inside
boto3/pydantic/langchain firing once. On a real serverless substrate that
cost lands on every cold-start invocation, not just once — which is itself a
relevant footnote for this paper, not just harness noise.

## Direct answer (item 5): does "flat across history" survive?

**No.** It was an artifact of stubbing the model object. With the model
mocked at the transport layer instead, per-step CPU clearly grows with
accumulated conversation size — see the table above. The growth is real,
well-evidenced (4,800 samples, monotonic, near-linear in bytes, verified
interception point), and belongs in the paper as a finding: *framework CPU
at a serverless-substrate-relevant scale (tens of KB of history) is several
times what it is at a cold start,* which is the opposite of what the 3 Sep
entry reported.

What DOES still stand from the stubbed measurement, unaffected by this fix:
tool dispatch cost (bucket b — never stubbed), the InMemorySaver
checkpoint-cost growth with history (bucket c — never stubbed, same
mechanism, still real), SqliteSaver's write-batching at graph completion
(same mechanism), the `step=-1` adapter bug and its fix, and the ToolNode
thread-pool wrinkle.

## Updated `compute_ms_per_step`

No longer one number shared by all three shapes — it now genuinely depends
on how much history a shape accumulates, so each `TurnShape` uses its own
measured value at its own step count (bucket a + bucket b, checkpointer=none;
checkpoint cost stays tracked separately via STATE_IO spans, unchanged
design):

| shape | steps | request size | compute_ms_per_step |
|---|---|---|---|
| short_tool_loop | 3 | ~2.0 KB | 0.734 |
| human_in_loop | 5 | ~3.1 KB | 0.806 |
| long_research | 14 | ~7.5 KB | 1.155 (floor — see caveat below) |

**long_research caveat:** measured with 1 tool call/step for consistency
with the other two shapes; `long_research`'s real `tools_per_step=2.0` would
grow the message list faster than this harness did, so 1.155ms understates
long_research specifically more than it understates the other two shapes.
Worth a dedicated 2-tools-per-step measurement before quoting long_research
numbers in the paper.

Still ~6-9x the invented 12ms placeholder was for the same reason as before
(the old number was too high, not too low) — but no longer flat, and no
longer one number.

## `run_characterise.py --shape short_tool_loop` — updated live figures

n=20 (stable): `turn_cpu_s_p50 = 0.005766s`, `wait_to_compute_ratio_p50 =
2,848:1`. Down from the transport-naive 4,338:1 (3 Sep, compute stubbed to
0.229ms — flat and too small) and still well above the invented placeholder's
478:1 (12ms — too large, but in the other direction). The mechanism matters
more than the point estimate: compute went from an unrealistically-flat
stub-derived number to a real, measured, history-growing one, and the ratio
moved to reflect that — that's the story worth telling, not 2,848 by itself.

## Revised credibility read

The 3 Sep entry argued "don't lead with the ratio, the underlying claim gets
stronger even as the number gets less credible." That argument gets MORE
right, not less, with this fix: the ratio moved back down some (compute is
real and non-trivial now, not a near-zero stub artifact) while the finding
underneath it got substantially more defensible — a real client, a verified
interception point, a measured and monotonic bytes relationship. That
combination — a less extreme ratio, backed by a harder-to-dispute
measurement — is a strictly better position for the paper than either the
invented placeholder or the first (stubbed) transport-free measurement.

---

# What the linear model implies — 3 Sep 2026

`model_path_cpu_ms ≈ 0.478 + 0.000081 × bytes` (~12 MB/s through the full
LangChain serialise → encode → parse → construct path). This, not any ratio, is
the paper's result: a mechanism, falsifiable, and transferable across frameworks.

## Extrapolating past the measured range

Holding the calibrated 4.23 s wait fixed:

| context | payload | cpu/step | ratio |
|---|---|---|---|
| first step | 2 KB | 0.61 ms | 6926:1 |
| measured ceiling | 20 KB | 2.10 ms | 2011:1 |
| ~25k tokens | 100 KB | 8.77 ms | 482:1 |
| ~100k tokens | 400 KB | 33.66 ms | 126:1 |
| ~250k tokens | 1 MB | 83.42 ms | 51:1 |

**Even at a 250k-token context the ratio is ~50:1.** State this explicitly in the
paper: it pre-empts the "your toy agent is unrepresentative" objection, which is
otherwise the most obvious line of attack.

## The non-obvious result — this is the quotable one

Per-step CPU grows **linearly** with conversation bytes. Per-step *wait* is
dominated by output tokens and grows only weakly with history (prefill is
parallel; decode is not). Therefore:

> **The serverless billing penalty is worst for short agent turns and least bad
> for long ones.**

That inverts the received wisdom. Everyone assumes long-running agents are FaaS's
problem — because of the 15-minute cap — but for the *billing* argument the cap
is a separate axis, and short turns are the pathological case. Two mechanisms
pointing opposite directions on the same substrate is a genuinely interesting
result and deserves its own figure.

## A methodological problem introduced by this round

`DEFAULT_SHAPES` now carries a per-shape scalar `compute_ms_per_step`
(0.734 / 0.806 / 1.155 ms). **That scalar discards the finding.** CPU is not a
constant per step; it is a function of accumulated bytes, and the whole point of
this round was establishing that. Baking in a mean drops the growth term.

Fix: have the synthetic model compute per-step CPU from the fitted linear model
against the state size the tracer already records, rather than from a constant.
The harness should express its own finding.

## Caveats that stay in the methodology section

- One framework (LangGraph 0.6.11), one client (ChatBedrockConverse), one agent
  shape. The 81 ns/byte coefficient is specific to LangChain's serialisation path.
- Measured on a laptop; absolute CPU figures scale with core speed. The *slope*
  is the transferable claim, not the intercept.
- `long_research` compute is a floor: measured at 1 tool/step, the shape uses 2.

## Status: measurement is DONE

The finding is established and well evidenced. The remaining items — the
2-tools-per-step re-measurement, a second framework — are refinements that will
not change the argument. With the deadline on 17 September, further measurement
is now the enemy of the paper. Start writing.

---

# Note on `profiles.json` tool median — 3 Sep 2026

`profiles.json` records a fitted `tool` median of **0.57 s**. Do not use it.

The 30 tool samples are bimodal by construction: GitHub `/zen` at a 0.20 s median and
`httpbin.org/get` at 1.01 s, 15 samples each. Fitting a single lognormal across both
returns the geometric mean, which lands at 0.57 s — in the empty region between the two
modes, describing neither endpoint and matching no observation. The σ of 0.89 was the
symptom.

The empirical pooled median is **0.97 s**. The paper reports the two endpoints separately
and does not use the pooled fit. This note exists so that anyone reading `profiles.json`
alongside the paper is not confused by the discrepancy; the paper itself does not spend
words on it.

Impact on conclusions: none. Turn wall-clock moves ~8%; the wait-versus-compute argument
is unaffected.

---

# Prior work on the capacity argument — 3 Sep 2026, UNVERIFIED LEADS

The §4.1 capacity claim is not novel as a general serverless observation. Before
submitting, verify and cite:

- **MXFaaS: Resource Sharing in Serverless Environments** — ISCA 2023, UIUC
  (iacoma.cs.uiuc.edu/iacoma-papers/isca23_2.pdf). Directly about sharing resources
  across concurrent invocations. Likely the closest prior work to the capacity argument.
- **Concurrency-Informed Orchestration for Serverless Functions** — ASPLOS 2025
  (dl.acm.org/doi/10.1145/3676641.3716253, ds2-lab.github.io/pdfs/asplos25-cidre.pdf).
- **Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a
  Large Cloud Provider** — Shahrad et al., USENIX ATC 2020. The canonical workload
  characterisation; cite for methodology framing.
- **XFaaS: Hyperscale and Low Cost Serverless Functions at Meta** — SOSP 2023.

**None of these has been verified to the standard used for refs.bib.** Check venue, year
and author list at the publisher before citing.

**What this means for the claim of novelty.** "Blocked invocations waste reserved
capacity" is established. What is ours: the agent-specific characterisation, the linear
model of framework CPU in payload bytes, and the short-turn inversion. State the prior
work plainly in §5 and claim only what is actually new.

## Verified platform facts (safe to cite)

- AWS Lambda: "During this entire process, this execution environment is busy and cannot
  process other requests." — docs.aws.amazon.com/lambda/latest/dg/lambda-concurrency.html
- AWS Lambda: "By default, Lambda provides your account with a total concurrency limit of
  1,000 concurrent executions across all functions in an AWS Region." — same page
- Google Cloud Run: default max concurrency 80 per instance (80 x vCPU via CLI/Terraform),
  configurable to 1,000. — docs.cloud.google.com/run/docs/about-concurrency

The Cloud Run figure matters: it shows the one-request-per-instance constraint is a design
choice, not a property of serverless. Use it as an existence proof, not a criticism.

---

# Correction: LangGraph checkpointers ARE incremental — 3 Sep 2026

An earlier §5.2 draft claimed production agent runtimes "persist state through
general-purpose checkpointers writing whole snapshots." **That is false**, and checkable
in thirty seconds from the installed source.

`InMemorySaver.put` (langgraph 0.6.11) pops `channel_values` off the checkpoint, iterates
`new_versions` — only the channels whose versions changed — and serialises each into its
own blob. The checkpoint skeleton is stored separately. Writes are already incremental at
channel granularity. `Checkpoint` even carries `updated_channels` for this purpose.

**Why the 8.8x growth happens anyway:** the channel that changes on every agent step is
the message history, and that channel is an unbounded append-only log. Channel-level
versioning cannot help when the always-changing channel is the one that grows.

**The corrected claim, which is stronger:** the granularity is wrong, not the mechanism.
The workload wants versioning *within* a channel — append-only or delta-encoded writes
over a growing sequence — which is what shared-log systems (Boki) already provide, and
nobody has applied them to agent state.

Lesson: this was a universal negative ("none of this appears in production agent
runtimes") asserted from a sample of one framework, none of whose source had been read.

# Environment capture — 10 Sep 2026

The framework-CPU sweep recorded `steps`, `replicates` and `all_synchronous`
and nothing about the machine. CPU-time constants are not interpretable
without the hardware and interpreter that produced them — CPython 3.11 alone
is materially faster than 3.9, which shifts every ms/KB figure — so the
numbers were, strictly, not reproducible even by us.

`scripts/measure_framework_cpu.py` now calls `capture_environment()` and
writes the result as `environment` at the top of `framework_cpu_raw.json`:
CPU brand, core counts, memory, OS, Python version and implementation, and
the resolved versions of langgraph, langgraph-checkpoint(-sqlite),
langgraph-prebuilt, langchain-core, langchain-aws and boto3. It prints a
one-line summary at startup so a wrong interpreter is visible immediately.

The 3 Sep run predates this, so its `environment` block was backfilled by
hand and is marked as such: Apple M5, macOS 26.3.1, CPython 3.9.6,
langgraph 0.6.11 with langgraph-checkpoint 2.1.2.

Note for the paper: langgraph and langgraph-checkpoint are versioned
separately. "LangGraph 0.6.11" does not pin the checkpointer whose
`InMemorySaver.put` the paper cites by line number; that is 2.1.2.
