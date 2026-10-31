# What each platform bills, reserves and limits while an agent is blocked (E3)

Researched 2026-10-27 from official pages, accessed the same day. Cite each
with that access date: these pages change, and several changed in 2026.

Verification status:
- **[V]** I re-fetched the official page and confirmed the quote myself.
- **[R]** Taken from the research agent's report with its official URL; not
  independently re-checked.
- **[U]** Unconfirmed by any official page.

## Summary

| Platform | Billed while blocked | Held while blocked | Default limit | Requests per environment |
|---|---|---|---|---|
| AgentCore Runtime | CPU no [V]; memory yes, whole session [V] | A dedicated microVM per session [R] | 5,000 active sessions (us-east-1, us-west-2), 2,500 elsewhere, adjustable; 25 new/s [R] | One session per microVM [R] |
| Lambda (event functions) | Wall-clock GB-s | One environment per request [R] | 1,000 per region [R] | 1 |
| Lambda MicroVMs (Jun 2026) | Running: compute; suspended: storage only [R] | Running *or suspended*, counts against the memory quota [R] | 400 GB memory (1,024 GB in 4 regions) [R] | One user/session per MicroVM [R] |
| Lambda durable functions | No compute during waits; steps billed [R] | Waits release the concurrency slot [R]; a model call inside a step holds it [U] | 1,000 concurrency pool [R] | 1 |
| **Lambda Web Functions (Oct 2026)** | Active-CPU billing [R, third-party post] | Environment shared by many requests [V] | Bounded by account vCPU quota; 2 vCPU per environment [V] | Many; dynamic, capped by `maxConcurrencyPerEnvironment` [V] |
| Lambda Managed Instances (Nov 2025) | EC2 instance-hours + 15% fee [R] | Your EC2 fleet | EC2 quotas | Many [R] |
| Azure Foundry hosted agents | CPU and memory "during active sessions" [R] | A sandbox per session until idle timeout (default 15 min) [R] | 1,000–2,000 concurrent sessions per subscription per region [R] | One session per sandbox [R] |
| Vertex AI Agent Engine | Allocated vCPU/GiB; idle time *between* turns not billed [R]; within-turn I/O wait [U] | Instances (min 1, max 100 by default) [R] | max_instances up to 1,000; 90 queries/min [R] | container_concurrency 9 by default; sync code serves 1 at a time [R] |
| Cloud Run | Request-based: billed while ≥1 request in flight [R] | Instance | 100 instances per region [R] | 80 default, 1,000 max [R] |
| Cloudflare Workers | CPU time only; network wait not billed [R] | Shared isolate | No general RPS limit [R] | Many [R] |

## The finding for the paper

**Per-second billing and suspension are real, and they don't remove the constraint.**

- **AgentCore** stopped billing CPU while an agent waits on I/O. Its own
  pricing example bills CPU for 60 s and memory for all 600 s of a session
  [V]. Each session still holds a dedicated microVM and a quota slot, and
  idle sessions count against the quota.
- **Lambda MicroVMs** can suspend, but suspension cuts cost, not capacity:
  a suspended MicroVM still counts against the memory quota [R]. Suspension
  also never happens during an in-flight request, because the idle policy
  triggers only after traffic stops.
- **Durable functions** release the concurrency slot during waits, but a
  model call is a step, not a wait [U].

**Multiplexing is the fix, and it already ships in general-purpose
runtimes:** Lambda Web Functions, Lambda Managed Instances, Cloud Run, and
Cloudflare Workers. It does **not** ship in the per-session agent sandboxes
(AgentCore, Foundry hosted agents). Microsoft's own multiplexing guide
concedes that "one session per user doesn't scale" [R].

**The trade-off to name: isolation against capacity.** Agent runtimes
dedicate a microVM per session for isolation, which is a legitimate reason:
agents run model-chosen tools and code. Our measurements put a price on that
choice: 200–600 environments held per core of work. The research question
becomes whether isolation and multiplexing can coexist. That is the paper's
agenda item, and it is sharper than "scale on blocked count".

Lambda Web Functions launched two weeks before this was written. It is the
existence proof the submission attributed to Cloud Run, and it is now
inside Lambda itself. The paper must discuss it.

## Sources

Re-checked myself:
- AgentCore pricing: https://aws.amazon.com/bedrock/agentcore/pricing/
  - "CPU scales to zero during I/O wait (waiting for LLM responses, tool /
    API calls, or database queries)"
  - The session "spans from microVM ready, initialization, active
    processing, idle periods, until session termination"
  - "Idle memory is reclaimed automatically in Runtime v2 after 120 seconds"
  - Worked example: CPU for 60 s, memory for 600 s
- Lambda Web Functions scaling:
  https://docs.aws.amazon.com/lambda/latest/dg/web-functions-scaling.html
  - Two-dimensional scaling; `maxConcurrencyPerEnvironment`; per-endpoint
    10,000 RPS; 2 vCPU per environment against the account vCPU quota
- Launch date (third-party):
  https://builder.aws.com/content/3KhH7ypXpyictGJ84VrNB0ZpU3A/aws-lambda-web-functions-are-out-heres-what-changes-for-hono
  — I could not find the AWS announcement itself.

From the research agent's report:
- AgentCore sessions:
  https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html
- AgentCore quotas:
  https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html
- Lambda concurrency:
  https://docs.aws.amazon.com/lambda/latest/dg/lambda-concurrency.html
- Lambda MicroVMs:
  https://docs.aws.amazon.com/lambda/latest/dg/microvms-launching.html
- Lambda Managed Instances:
  https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html
- Lambda FAQ (durable functions): https://aws.amazon.com/lambda/faqs/
- Foundry hosted agents:
  https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/hosted-agents
- Foundry limits:
  https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/limits-quotas-regions
- Foundry multiplexing guide:
  https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/multiplex-session-users
- Vertex pricing:
  https://cloud.google.com/products/gemini-enterprise-agent-platform/pricing
- Cloud Run concurrency:
  https://docs.cloud.google.com/run/docs/about-concurrency

**Before submission:** re-fetch every [R] source, and check the Foundry
idle-timeout range (the docs disagree: 2–240 vs 2–60 minutes).
