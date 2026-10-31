"""E5: evaluate the multiplexing remedy against per-session isolation, by
discrete-event simulation driven by the measured distributions.

Evaluates the sharing remedy, and the cost of leaving sessions to an idle
timeout, by simulation driven by the deployed workload.

Workload, all from data/deployed_turns.csv (3,549 warm turns of a deployed
tau2-bench retail agent):
  * a session is n steps, n drawn from the deployed sessions' step counts;
  * each step computes for c (framework CPU) and then blocks for w (model
    call), w drawn from the deployed LangGraph model-call times;
  * sessions arrive as a Poisson process sized for a target mean number of
    concurrent sessions.
c is not drawn from data: deployed CPU per step is bracketed (4.5 to ~12 ms,
results/SENSITIVITY-SUMMARY.md), so it is a parameter swept over that bracket.

Policies:
  A. Per-session isolation, as agent runtimes do it. Each session holds its own
     environment (one vCPU of its own, so no CPU contention) from start to end,
     plus an idle tail: 0 if the client terminates the session explicitly, or
     the platform idle timeout (900 s) if it does not.
  B. Multiplexing, as Cloud Run / Lambda Web Functions do it. Environments of V
     vCPUs each host up to k sessions; a new session goes to the first
     environment with a free slot, and a new environment starts only when none
     has one. CPU bursts queue FIFO per environment, so multiplexing shows up
     as added step latency. Environments are released when empty.

Memory model (stated, illustrative): every environment is one process reserving
BASE plus M_SESS per resident session. An isolated environment is that process with
one session, so the comparison isolates the effect of sharing alone; a microVM's
guest-OS overhead would only widen the gap.

    .venv/bin/python scripts/simulate_remedy.py
Outputs: results/remedy_sim.json, paper/figures/fig5_remedy.{pdf,png}
"""
import csv, json, os, statistics as st
import numpy as np
import simpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
SEED = 20261027

TARGET_CONCURRENCY = 1000        # mean concurrent sessions
WARMUP_S, MEASURE_S = 1200, 1200 # simulated seconds; warm-up must exceed the 900 s idle tail
V_CPUS = 2                       # vCPUs per multiplexed environment
BASE_GB, M_SESS_GB = 0.5, 0.010  # process base + per-session state (illustrative)
M_ISO_GB = BASE_GB + M_SESS_GB   # an isolated environment is one process with one session,
                                 # so k=1 multiplexing and isolation reserve the same memory
IDLE_TIMEOUT_S = 900             # AgentCore / Foundry default idle timeout
CPU_MS_GRID = (4.5, 11.7, 20.3)  # deployed bracket (AgentCore-measured): low / LangGraph mid / Strands high
K_GRID = (1, 4, 16, 64, 256)


def load_workload():
    rows = [l for l in open(P("data", "deployed_turns.csv")) if not l.startswith("#")]
    r = list(csv.DictReader(rows))
    lg = [x for x in r if x["config"] == "c2-langgraph-agentcore"]
    waits = np.array([float(x["model_ms"]) / 1000 for x in lg])
    per_session = {}
    for x in r:
        # Key on run_id, not (config, task, trial): fault-labelled runs reuse trial
        # numbers. Steps per session = highest turn index + 1, which counts the cold
        # first turn excluded from the warm-turn extract.
        per_session[x["run_id"]] = max(per_session.get(x["run_id"], 0), int(x["turn_index"]) + 1)
    steps = np.array(list(per_session.values()))
    return waits, steps


def session_steps(rng, waits, steps, cpu_s):
    n = int(rng.choice(steps))
    return [(cpu_s, float(rng.choice(waits))) for _ in range(n)]


def arrival_rate(waits, steps, cpu_s, tail_s=0.0):
    mean_session_s = steps.mean() * (waits.mean() + cpu_s)
    return TARGET_CONCURRENCY / mean_session_s, mean_session_s


def run_isolated(waits, steps, cpu_s, tail_s):
    rng = np.random.default_rng(SEED)
    env = simpy.Environment()
    lam, mean_s = arrival_rate(waits, steps, cpu_s)
    held = {"n": 0}
    samples = []

    def session(plan):
        held["n"] += 1
        for c, w in plan:
            yield env.timeout(c + w)          # own vCPU: no queueing
        yield env.timeout(tail_s)
        held["n"] -= 1

    def arrivals():
        while True:
            yield env.timeout(rng.exponential(1 / lam))
            env.process(session(session_steps(rng, waits, steps, cpu_s)))

    def sampler():
        yield env.timeout(WARMUP_S)
        while True:
            samples.append(held["n"]); yield env.timeout(1)

    env.process(arrivals()); env.process(sampler())
    env.run(until=WARMUP_S + MEASURE_S)
    s = np.array(samples)
    return {"policy": "isolated", "cpu_ms": cpu_s * 1000, "tail_s": tail_s,
            "envs_mean": float(s.mean()), "envs_p99": float(np.percentile(s, 99)),
            "mem_gb_mean": float(s.mean() * M_ISO_GB), "arrival_per_s": lam,
            "mean_session_s": mean_s, "added_latency_ms": {"p50": 0.0, "p99": 0.0, "p999": 0.0},
            "cpu_util": cpu_s / (cpu_s + waits.mean())}


def run_multiplexed(waits, steps, cpu_s, k):
    rng = np.random.default_rng(SEED)
    env = simpy.Environment()
    lam, mean_s = arrival_rate(waits, steps, cpu_s)
    pool = []          # each: {"cpu": Resource, "sessions": int}
    delays, samples, busy = [], [], {"cpu_s": 0.0}

    def place():
        for e in pool:
            if e["sessions"] < k:
                e["sessions"] += 1; return e
        e = {"cpu": simpy.Resource(env, capacity=V_CPUS), "sessions": 1}
        pool.append(e); return e

    def session(plan):
        e = place()
        for c, w in plan:
            t0 = env.now
            with e["cpu"].request() as req:
                yield req
                if env.now >= WARMUP_S:
                    delays.append(env.now - t0)
                yield env.timeout(c)
            yield env.timeout(w)
        e["sessions"] -= 1
        if e["sessions"] == 0:
            pool.remove(e)

    def arrivals():
        while True:
            yield env.timeout(rng.exponential(1 / lam))
            env.process(session(session_steps(rng, waits, steps, cpu_s)))

    def sampler():
        yield env.timeout(WARMUP_S)
        while True:
            samples.append((len(pool), sum(e["sessions"] for e in pool)))
            yield env.timeout(1)

    env.process(arrivals()); env.process(sampler())
    env.run(until=WARMUP_S + MEASURE_S)
    s = np.array(samples)
    d = np.array(delays) * 1000
    envs = s[:, 0]
    mem = envs * BASE_GB + s[:, 1] * M_SESS_GB
    return {"policy": "multiplexed", "k": k, "cpu_ms": cpu_s * 1000, "tail_s": 0.0,
            "envs_mean": float(envs.mean()), "envs_p99": float(np.percentile(envs, 99)),
            "mem_gb_mean": float(mem.mean()), "arrival_per_s": lam, "mean_session_s": mean_s,
            "sessions_mean": float(s[:, 1].mean()),
            "added_latency_ms": {"p50": float(np.percentile(d, 50)), "p99": float(np.percentile(d, 99)),
                                 "p999": float(np.percentile(d, 99.9))},
            "cpu_util_per_env": float(s[:, 1].mean() * cpu_s / (cpu_s + waits.mean()) / envs.mean() / V_CPUS)}


def main():
    waits, steps = load_workload()
    print(f"workload: {len(waits)} model calls, median wait {np.median(waits):.2f} s; "
          f"{len(steps)} sessions, median {int(np.median(steps))} steps (range {steps.min()}-{steps.max()})")
    out = {"params": {"target_concurrency": TARGET_CONCURRENCY, "warmup_s": WARMUP_S,
                      "measure_s": MEASURE_S, "v_cpus": V_CPUS, "m_iso_gb": M_ISO_GB,
                      "base_gb": BASE_GB, "m_sess_gb": M_SESS_GB, "idle_timeout_s": IDLE_TIMEOUT_S,
                      "seed": SEED},
           "runs": []}
    for cpu_ms in CPU_MS_GRID:
        c = cpu_ms / 1000
        for tail in (0.0, float(IDLE_TIMEOUT_S)):
            r = run_isolated(waits, steps, c, tail); out["runs"].append(r)
            print(f"isolated  cpu {cpu_ms:5.1f} ms tail {tail:4.0f} s: envs {r['envs_mean']:7.0f} "
                  f"(p99 {r['envs_p99']:6.0f})  mem {r['mem_gb_mean']:7.0f} GB")
        for k in K_GRID:
            r = run_multiplexed(waits, steps, c, k); out["runs"].append(r)
            a = r["added_latency_ms"]
            print(f"mux k={k:3d} cpu {cpu_ms:5.1f} ms:        envs {r['envs_mean']:7.0f} "
                  f"(p99 {r['envs_p99']:6.0f})  mem {r['mem_gb_mean']:7.1f} GB  "
                  f"added step latency p50/p99/p99.9 {a['p50']:.2f}/{a['p99']:.2f}/{a['p999']:.2f} ms  "
                  f"vCPU util {100 * r['cpu_util_per_env']:.1f}%")
    os.makedirs(P("results"), exist_ok=True)
    json.dump(out, open(P("results", "remedy_sim.json"), "w"), indent=1)
    print("wrote results/remedy_sim.json")
    figure(out, float(np.median(waits)) * 1000)


def figure(out, median_wait_ms):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GREEN = "#4a7fd4", "#d95f02", "#1b9e77"
    INK, MUTED, GRID = "#1a211d", "#5e6862", "#dce1d9"
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 6.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
        "grid.color": GRID, "grid.linewidth": 0.5, "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    runs = out["runs"]
    mid = CPU_MS_GRID[1]
    mux = sorted([r for r in runs if r["policy"] == "multiplexed" and r["cpu_ms"] == mid], key=lambda r: r["k"])
    iso = {r["tail_s"]: r for r in runs if r["policy"] == "isolated" and r["cpu_ms"] == mid}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.4))

    ks = [r["k"] for r in mux]
    ax.plot(ks, [r["envs_mean"] for r in mux], "o-", color=BLUE, ms=3.5, lw=1, label="environments")
    ax.plot(ks, [r["mem_gb_mean"] for r in mux], "s--", color=GREEN, ms=3.2, lw=1, label="memory reserved (GB)")
    for tail, lab, ls in [(0.0, "isolated, session closed", ":"), (float(IDLE_TIMEOUT_S), "isolated, 15-min idle timeout", "-.")]:
        y = iso[tail]["envs_mean"]
        ax.axhline(y, color=ORANGE, lw=0.9, ls=ls)
        ax.text(1.05, y * 1.12, f"{lab}: {y:,.0f} environments", fontsize=6, color=ORANGE)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(2, 60000)
    ax.set_xlabel("sessions per multiplexed environment (k)")
    ax.set_ylabel("per 1,000 concurrent sessions")
    ax.set_title("(a) What 1,000 concurrent sessions hold", loc="left")
    ax.legend(loc="lower left", frameon=False)

    for cpu_ms, col, mk in zip(CPU_MS_GRID, [MUTED, BLUE, GREEN], ["^", "o", "s"]):
        rr = sorted([r for r in runs if r["policy"] == "multiplexed" and r["cpu_ms"] == cpu_ms], key=lambda r: r["k"])
        bx.plot([r["k"] for r in rr], [r["added_latency_ms"]["p99"] for r in rr], mk + "-", color=col, ms=3.2, lw=1,
                label=f"p99, CPU {cpu_ms:g} ms/step")
        bx.plot([r["k"] for r in rr], [r["added_latency_ms"]["p999"] for r in rr], mk + ":", color=col, ms=2.5, lw=0.8,
                alpha=0.7)
    bx.set_xscale("log", base=2); bx.set_xticks(ks); bx.set_xticklabels([str(k) for k in ks])
    bx.set_xlabel("sessions per multiplexed environment (k)")
    bx.set_ylabel("added step latency (ms)")
    bx.set_title("(b) What multiplexing costs in latency", loc="left")
    bx.text(1.05, bx.get_ylim()[1] * 0.93 if bx.get_ylim()[1] > 0 else 1,
            f"dotted: p99.9. A median model call takes {median_wait_ms:,.0f} ms.", fontsize=6, color=MUTED, va="top")
    bx.legend(loc="center left", frameon=False)
    for a in (ax, bx):
        a.grid(True, which="major")
    os.makedirs(P("paper", "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(P("paper", "figures", f"fig5_remedy.{ext}"))
    print("wrote paper/figures/fig5_remedy.{pdf,png}")


if __name__ == "__main__":
    main()
