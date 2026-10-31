"""Sensitivity of the capacity claim, and validation against a real deployment.

Shows how the capacity result holds up across model response times and CPU
per step, and checks the laptop-plus-calibration combination against agents
measured in deployment.

The capacity claim is Little's law. With N concurrent execution environments,
each turn step blocked for w seconds and computing for c seconds,

    busy cores at the concurrency limit   K = N * c / (w + c)
    environments held per busy core       N / K = (w + c) / c

The second quantity does not depend on N or on the number of steps, so it is
the one we sweep. The submission reported one point on this surface (Apple M5,
mocked socket, calibrated latencies). This script adds:

  1. the same surface over realistic ranges of w and c;
  2. an independent anchor from a deployed agent: 3,549 warm turns of a
     LangGraph and a Strands agent on Amazon Bedrock AgentCore and of the same
     LangGraph agent self-hosted, running tau2-bench retail against Claude
     Sonnet 4.5 (collected Sep 2026 for a separate study);
  3. a check that the submission's linear CPU-in-payload model holds in that
     deployment, and that model wait tracks output length, not input.

Usage:
    .venv/bin/python scripts/sensitivity.py --extract <full-run-v2-runs.jsonl>
    .venv/bin/python scripts/sensitivity.py            # uses data/deployed_turns.csv
Outputs: data/deployed_turns.csv (with --extract), results/sensitivity.json,
paper/figures/fig4_sensitivity.{pdf,png}.
"""
import argparse, csv, json, os, statistics as st
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)

N_LIMIT = 1000                 # AWS Lambda default regional concurrency, as cited
BYTES_PER_TOKEN = (3.0, 4.0, 5.0)   # assumed range for converting KB <-> tokens
SEED = 20261027
CONFIG_LABEL = {
    "c1-langgraph-plain": "LangGraph, self-hosted",
    "c2-langgraph-agentcore": "LangGraph, AgentCore",
    "c3-strands-agentcore": "Strands, AgentCore",
}
FIELDS = ["run_id", "config", "task", "trial", "turn_index", "tokens_in", "tokens_out",
          "model_ms", "overhead_ms", "server_total_ms"]


# ---------------------------------------------------------------- inputs
def extract(jsonl_path):
    """Slim, account-free per-turn extract of the deployed study. Warm turns only:
    a cold turn's server time includes process start-up, a different quantity."""
    out = []
    for line in open(jsonl_path):
        r = json.loads(line)
        for t in r["turns"]:
            if t["cold_start"]:
                continue
            m = t["model"]["ms"] if isinstance(t["model"], dict) else t["model"]
            out.append([r["run_id"], r["config_id"], r["task_id"], r["trial"], t["turn_index"],
                        t["tokens_in"], t["tokens_out"], round(m, 3),
                        round(t["overhead_ms"], 4), round(t["server_total_ms"], 3)])
    path = P("data", "deployed_turns.csv")
    with open(path, "w", newline="") as f:
        f.write("# Warm turns from a deployed tau2-bench retail study, Sep 2026.\n"
                "# Model us.anthropic.claude-sonnet-4-5-20250929-v1:0, us-east-1.\n"
                "# overhead_ms = server_total_ms - model_ms: framework + runtime time\n"
                "# inside the deployed process (wall clock, an upper bound on CPU).\n"
                "# Tools are executed by the benchmark, outside the agent: no tool wait.\n")
        w = csv.writer(f); w.writerow(FIELDS); w.writerows(out)
    print(f"wrote {len(out)} turns to {path}")


def load_deployed():
    rows = list(csv.reader(l for l in open(P("data", "deployed_turns.csv"))
                           if not l.startswith("#")))
    hdr, body = rows[0], rows[1:]
    by = {}
    for r in body:
        d = dict(zip(hdr, r))
        by.setdefault(d["config"], []).append(
            (float(d["tokens_in"]), float(d["tokens_out"]),
             float(d["model_ms"]), float(d["overhead_ms"])))
    return {k: np.array(v) for k, v in by.items()}


CHIP = {"0xd40": "Graviton3", "0xd0c": "Graviton2", "0xd4f": "Graviton4"}


def chip_name(brand):
    """Readable processor name; arm64 /proc/cpuinfo gives only a CPU part number."""
    for part, name in CHIP.items():
        if brand and part in brand:
            return name
    return brand


def raw_fit(name, cpu=None):
    """Fit model-path CPU (ms) against request payload exactly as the submission
    and make_figures.py do: per-iteration medians across every run in the file,
    then OLS. Platform files record one environment per invocation; `cpu`
    restricts the fit to runs on that processor. Returns None if absent."""
    path = P("data", name)
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    envs = d.get("environments")
    byit, tool = {}, []
    for rl in d["runs"].values():
        for r in rl:
            if cpu and envs and chip_name(envs[r["invocation"]]["cpu_brand"]) != cpu:
                continue
            for it in r["iterations"]:
                byit.setdefault(it["iteration"], []).append(
                    (it["request_body_bytes"], it["model_path_cpu_s"] * 1000))
                tool.append(it["tool_dispatch_cpu_s"] * 1000)
    its = sorted(byit)
    bx = np.array([st.median(v[0] for v in byit[i]) for i in its])
    by = np.array([st.median(v[1] for v in byit[i]) for i in its])
    slope, icept = np.polyfit(bx, by, 1)
    r2 = 1 - np.sum((by - (icept + slope * bx)) ** 2) / np.sum((by - by.mean()) ** 2)
    env = d.get("environment", {})
    return {"file": name, "c_fixed_ms": icept, "ms_per_kb": slope * 1024, "r2": r2,
            "tool_dispatch_ms": st.median(tool),
            "payload_kb_range": (bx.min() / 1024, bx.max() / 1024),
            "machine": cpu or (sorted({chip_name(e["cpu_brand"]) for e in envs}) if envs else None)
                       or env.get("cpu_brand") or env.get("processor") or env.get("machine"),
            "n_runs": sum(len(v) for v in byit.values()) // max(len(byit), 1),
            "python": env.get("python_version"),
            "points": [(bx / 1024).tolist(), by.tolist()]}


def predict_cpu(fit, kb):
    return fit["c_fixed_ms"] + fit["ms_per_kb"] * kb


def m5_latency():
    d = json.load(open(P("data", "latency_raw.json")))
    llm = [x["latency_s"] for x in d["llm"]]
    tool = [x["latency_s"] for x in d["tool"]]
    out = [x["completion_tokens"] for x in d["llm"]]
    return {"llm_median_s": st.median(llm), "tool_median_s": st.median(tool),
            "llm_completion_tokens_median": st.median(out)}


# ---------------------------------------------------------------- analysis
def binned_fit(tin, y, edges):
    """OLS through per-bin medians (bins with >= 20 turns), as the submission
    does for iterations. Returns slope per 1k tokens, intercept, R^2, points."""
    idx = np.digitize(tin, edges)
    px, py = [], []
    for k in range(1, len(edges)):
        sel = idx == k
        if sel.sum() >= 20:
            px.append(np.median(tin[sel]) / 1000); py.append(np.median(y[sel]))
    px, py = np.array(px), np.array(py)
    s, i = np.polyfit(px, py, 1)
    r2 = 1 - np.sum((py - (i + s * px)) ** 2) / np.sum((py - py.mean()) ** 2)
    return s, i, r2, px, py


def bootstrap_slope(tin, y, edges, n=2000):
    rng = np.random.default_rng(SEED)
    sl = []
    for _ in range(n):
        j = rng.integers(0, len(tin), len(tin))
        try:
            sl.append(binned_fit(tin[j], y[j], edges)[0])
        except (np.linalg.LinAlgError, ValueError, TypeError):
            pass
    return np.percentile(sl, [2.5, 97.5])


def envs_per_core(wait_s, cpu_ms):
    c = cpu_ms / 1000
    return (wait_s + c) / c


def cores_at_limit(wait_s, cpu_ms, n=N_LIMIT):
    return n / envs_per_core(wait_s, cpu_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", metavar="RUNS_JSONL")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract(a.extract)

    lat, dep = m5_latency(), load_deployed()
    fits = {"langgraph_m5": raw_fit("framework_cpu_raw.json"),
            "langgraph_x86": raw_fit("framework_cpu_raw_x86.json"),
            "strands_x86": raw_fit("strands_cpu_raw_x86.json")}
    fits.update({"langgraph_agentcore": raw_fit("platform/agentcore-langgraph.json"),
                 "strands_agentcore": raw_fit("platform/agentcore-strands.json"),
                 "strands_agentcore_graviton3": raw_fit("platform/agentcore-strands.json", "Graviton3"),
                 "strands_agentcore_graviton2": raw_fit("platform/agentcore-strands.json", "Graviton2")})
    fits = {k: v for k, v in fits.items() if v}
    fit = fits["langgraph_m5"]
    edges = np.arange(4000, 21000, 1000)
    res = {"n_limit": N_LIMIT, "fits": fits, "m5_latency": lat, "deployed": {}, "scenarios": []}

    # --- 1. the deployed anchor
    # IMPORTANT: in the deployed study, model time was timed around the model
    # client call (litellm.completion for LangGraph; Strands' Before/After
    # ModelCallEvent hooks). That window contains the client's own request
    # serialisation and response parsing -- the work the transport mock measures
    # as model-path CPU. So deployed overhead_ms is framework/server time OUTSIDE
    # the model client. It omits client serialisation and may include non-CPU
    # wall time: it is not a bound on CPU in either direction. We therefore
    # estimate deployed per-step CPU as overhead + client serialisation, the
    # latter predicted from the transport-mock fit for the same framework on the
    # x86 machine at the deployed payload size.
    # Client serialisation on AgentCore is predicted from the transport mock run
    # ON AgentCore (its own vCPUs, all processors it placed us on); the self-hosted
    # configuration ran on an unrecorded EC2 instance, so it keeps the x86 desktop fit.
    client_fit = {"c1-langgraph-plain": "langgraph_x86",
                  "c2-langgraph-agentcore": "langgraph_agentcore" if "langgraph_agentcore" in fits else "langgraph_x86",
                  "c3-strands-agentcore": "strands_agentcore" if "strands_agentcore" in fits else "strands_x86"}
    for cfg, a_ in dep.items():
        tin, tout, m, o = a_.T
        s_, i_, r2, px, py = binned_fit(tin, o, edges)
        ci = bootstrap_slope(tin, o, edges)
        X = np.c_[np.ones_like(tin), tin / 1000, tout / 100]
        coef, *_ = np.linalg.lstsq(X, m, rcond=None)
        cf = fits.get(client_fit[cfg])
        kb = [np.median(tin) * b / 1024 for b in BYTES_PER_TOKEN]
        ser = [predict_cpu(cf, k) for k in kb] if cf else [float("nan")] * 3
        o50 = float(np.median(o))
        res["deployed"][cfg] = {
            "n_turns": int(len(a_)),
            "model_ms_pct": dict(zip(["p5", "p25", "p50", "p75", "p95"],
                                     np.percentile(m, [5, 25, 50, 75, 95]).round(0).tolist())),
            "overhead_ms_pct": dict(zip(["p25", "p50", "p75", "p95"],
                                        np.percentile(o, [25, 50, 75, 95]).round(2).tolist())),
            "tokens_in_pct": dict(zip(["p5", "p50", "p95"], np.percentile(tin, [5, 50, 95]).round(0).tolist())),
            "tokens_out_pct": dict(zip(["p5", "p50", "p95"], np.percentile(tout, [5, 50, 95]).round(0).tolist())),
            "outside_client_vs_context": {"ms_per_1k_tokens": s_, "ci95": ci.tolist(),
                                          "intercept_ms": i_, "r2_binned": r2,
                                          "bins": [px.tolist(), py.tolist()]},
            "model_wait_model": {"intercept_ms": coef[0], "ms_per_1k_input_tokens": coef[1],
                                 "ms_per_output_token": coef[2] / 100},
            "client_serialisation_est_ms": dict(zip(["lo", "mid", "hi"], ser)),
            "cpu_per_step_est_ms": dict(zip(["lo", "mid", "hi"], [o50 + x for x in ser])),
            "payload_kb_est": dict(zip(["lo", "mid", "hi"], kb)),
        }

    # --- 2. scenarios, each one point on the surface
    lg, sd = res["deployed"]["c2-langgraph-agentcore"], res["deployed"]["c3-strands-agentcore"]
    m5_wait = lat["llm_median_s"] + lat["tool_median_s"]
    w50, w5 = lg["model_ms_pct"]["p50"] / 1000, lg["model_ms_pct"]["p5"] / 1000
    o_s, o_i = lg["outside_client_vs_context"]["ms_per_1k_tokens"], lg["outside_client_vs_context"]["intercept_ms"]
    lx = fits.get("langgraph_agentcore") or fits.get("langgraph_x86")   # extrapolate with the platform-measured fit

    def big_context(ktok, b=4.0):
        """Per-step CPU at a large context: outside-client slope from the deployment
        plus client serialisation from the x86 transport-mock fit."""
        return o_i + o_s * ktok + predict_cpu(lx, ktok * 1000 * b / 1024)

    scen = [("As submitted: M5, mocked socket, calibrated waits", m5_wait,
             predict_cpu(fit, 1.1) + fit["tool_dispatch_ms"], None)]
    for key, label in (("langgraph_x86", "Same, measured on the x86 desktop"),
                       ("langgraph_agentcore", "Same, measured on AgentCore (Graviton3)")):
        f_ = fits.get(key)
        if f_:
            scen.append((label, m5_wait, predict_cpu(f_, 1.1) + f_["tool_dispatch_ms"], None))
    # Range: low = overhead measured outside the model client (omits client
    # serialisation); high = that plus estimated client serialisation at
    # 5 bytes/token. Point estimate uses 4 bytes/token.
    for name, d in [("LangGraph", lg), ("Strands", sd)]:
        c = d["cpu_per_step_est_ms"]
        scen.append((f"Deployed {name} on AgentCore, medians",
                     d["model_ms_pct"]["p50"] / 1000, c["mid"], (d["overhead_ms_pct"]["p50"], c["hi"])))
    scen.append(("Deployed LangGraph, fast step (p5 wait), high CPU estimate", w5,
                 lg["overhead_ms_pct"]["p95"] + lg["client_serialisation_est_ms"]["hi"], None,
                 (lg["overhead_ms_pct"]["p95"], lg["client_serialisation_est_ms"]["hi"])))
    if lx:
        scen += [("Extrapolated to 50k-token context, median wait", w50, big_context(50), None),
                 ("Extrapolated to 250k-token context, median wait", w50, big_context(250), None),
                 ("250k-token context and p5 wait (worst case, outside measured range)", w5, big_context(250), None)]
    for name, w, c, rng, *parts in scen:
        e = {"name": name, "wait_s_per_step": w, "cpu_ms_per_step": c,
             "envs_per_busy_core": envs_per_core(w, c), "cores_at_limit": cores_at_limit(w, c)}
        if parts:
            e["cpu_parts_ms"] = parts[0]          # (measured in place, benchmark-derived)
        if rng:
            e["cpu_range_ms"] = rng
            e["envs_per_busy_core_range"] = (envs_per_core(w, rng[1]), envs_per_core(w, rng[0]))
            e["cores_at_limit_range"] = (cores_at_limit(w, rng[0]), cores_at_limit(w, rng[1]))
        res["scenarios"].append(e)

    vpath = P("results", "capacity_validation.json")
    if os.path.exists(vpath):
        v = json.load(open(vpath))
        per1000 = [r["measured_cores"] / r["sessions"] * 1000 for r in v["runs"]]
        res["validation"] = {"cores_per_1000_measured": (min(per1000), max(per1000)),
                             "measured_over_predicted": (min(r["measured_over_predicted"] for r in v["runs"]),
                                                         max(r["measured_over_predicted"] for r in v["runs"])),
                             "sessions": [r["sessions"] for r in v["runs"]],
                             "w_mean_s": st.mean(r["w_mean_s"] for r in v["runs"]),
                             "cpu_per_call_ms": [r["measured_cores"] * v["window_s"] / r["calls"] * 1000 for r in v["runs"]]}
    os.makedirs(P("results"), exist_ok=True)
    json.dump(res, open(P("results", "sensitivity.json"), "w"), indent=1, default=float)
    report(res)
    if not a.no_figure:
        figure(res, dep, edges)


def report(res):
    print("\nTransport-mock fits (model-path CPU vs request payload):")
    for k, f in res["fits"].items():
        print(f"  {k:14s} {f['c_fixed_ms']:.3f} ms + {f['ms_per_kb']:.4f} ms/KB  R2 {f['r2']:.3f}  "
              f"tool dispatch {f['tool_dispatch_ms']:.3f} ms  [{f['machine']}, Python {f['python']}]")
    for cfg, d in res["deployed"].items():
        oc, mw = d["outside_client_vs_context"], d["model_wait_model"]
        c, ser = d["cpu_per_step_est_ms"], d["client_serialisation_est_ms"]
        print(f"\n{CONFIG_LABEL[cfg]}  (n={d['n_turns']} warm turns)")
        print(f"  model wait p5/p50/p95 ms        {d['model_ms_pct']['p5']:.0f} / {d['model_ms_pct']['p50']:.0f} / {d['model_ms_pct']['p95']:.0f}")
        print(f"  output tokens p50 (p95)         {d['tokens_out_pct']['p50']:.0f} ({d['tokens_out_pct']['p95']:.0f})")
        print(f"  model wait                      {mw['intercept_ms']:.0f} ms + {mw['ms_per_output_token']:.1f} ms/output tok + {mw['ms_per_1k_input_tokens']:.1f} ms/1k input tok")
        print(f"  overhead outside client p50     {d['overhead_ms_pct']['p50']:.2f} ms; vs context {oc['ms_per_1k_tokens']:.3f} ms/1k tok "
              f"[{oc['ci95'][0]:.3f}, {oc['ci95'][1]:.3f}], R2 binned {oc['r2_binned']:.2f}")
        print(f"  + client serialisation (est.)   {ser['lo']:.2f}-{ser['hi']:.2f} ms at {d['payload_kb_est']['lo']:.0f}-{d['payload_kb_est']['hi']:.0f} KB")
        print(f"  = CPU per step (est.)           {c['lo']:.2f}-{c['hi']:.2f} ms (mid {c['mid']:.2f})")
    print(f"\n{'scenario':70s} {'wait s':>7s} {'cpu ms':>7s} {'envs/core':>14s} {'cores@'+str(res['n_limit']):>12s}")
    for s in res["scenarios"]:
        if "cpu_range_ms" in s:
            ep = f"{s['envs_per_busy_core_range'][0]:.0f}-{s['envs_per_busy_core_range'][1]:.0f}"
            cr = f"{s['cores_at_limit_range'][0]:.1f}-{s['cores_at_limit_range'][1]:.1f}"
        else:
            ep, cr = f"{s['envs_per_busy_core']:.0f}", f"{s['cores_at_limit']:.2f}"
        print(f"{s['name']:70s} {s['wait_s_per_step']:7.2f} {s['cpu_ms_per_step']:7.2f} {ep:>14s} {cr:>12s}")


# ---------------------------------------------------------------- figure
def figure(res, dep, edges):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    BLUE, ORANGE, GREEN = "#4a7fd4", "#d95f02", "#1b9e77"
    INK, MUTED, GRID = "#1a211d", "#5e6862", "#dce1d9"
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "legend.fontsize": 6.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "grid.color": GRID, "grid.linewidth": 0.5,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    def draw_a(ax):
        # (a) the surface: environments held per busy core
        W = np.logspace(np.log10(0.3), np.log10(30), 200)
        C = np.logspace(np.log10(0.1), np.log10(600), 200)
        WW, CC = np.meshgrid(W, C)
        Z = envs_per_core(WW, CC)
        levels = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
        ax.contourf(WW, CC, Z, levels=[1] + levels + [1e6], norm=LogNorm(), cmap="Greys_r", alpha=0.35)
        cs = ax.contour(WW, CC, Z, levels=levels, colors=MUTED,
                        linewidths=[1.4 if v == N_LIMIT else 0.5 for v in levels])
        ax.clabel(cs, levels=[v for v in levels if v != N_LIMIT and v <= 10000],
                  fmt=lambda v: f"{v:,.0f}", fontsize=6, inline_spacing=2)
        ax.text(13, 8.0, f"{N_LIMIT:,}: one busy core at a\n{N_LIMIT:,}-environment limit",
                fontsize=5.8, color=INK, rotation=31, ha="center", va="bottom")
        sc = {x["name"]: x for x in res["scenarios"]}
        s0 = res["scenarios"][0]
        ax.plot(s0["wait_s_per_step"], s0["cpu_ms_per_step"], "s", color=ORANGE, ms=5, zorder=5)
        ax.annotate("laboratory\n(M5, mocked socket)", (s0["wait_s_per_step"], s0["cpu_ms_per_step"]),
                    xytext=(6, -4), textcoords="offset points", fontsize=6, color=ORANGE, va="top")
        for cfg, col, mk in [("c2-langgraph-agentcore", BLUE, "o"), ("c3-strands-agentcore", GREEN, "^")]:
            d = res["deployed"][cfg]
            x = d["model_ms_pct"]["p50"] / 1000
            c = d["cpu_per_step_est_ms"]
            ax.errorbar(x, c["mid"], xerr=[[x - d["model_ms_pct"]["p25"] / 1000], [d["model_ms_pct"]["p75"] / 1000 - x]],
                        yerr=[[c["mid"] - d["overhead_ms_pct"]["p50"]], [c["hi"] - c["mid"]]],
                        fmt=mk, color=col, ms=4.5, lw=0.8, capsize=1.5, zorder=5)
        lgm = res["deployed"]["c2-langgraph-agentcore"]
        ax.annotate("deployed agents\n(LangGraph \u25cf, Strands \u25b2)",
                    (lgm["model_ms_pct"]["p25"] / 1000, lgm["cpu_per_step_est_ms"]["mid"]),
                    xytext=(-8, -9), textcoords="offset points", fontsize=6, color=BLUE, ha="right", va="center")
        for key, lab in [("Extrapolated to 50k-token context, median wait", "50k tok"),
                         ("Extrapolated to 250k-token context, median wait", "250k tok")]:
            if key in sc:
                x_ = sc[key]
                ax.plot(x_["wait_s_per_step"], x_["cpu_ms_per_step"], "o", mfc="white", mec=BLUE, ms=4, zorder=5)
                ax.annotate(lab, (x_["wait_s_per_step"], x_["cpu_ms_per_step"]), xytext=(4, -2),
                            textcoords="offset points", fontsize=6, color=BLUE)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("blocked time per step (s)"); ax.set_ylabel("framework CPU per step (ms)")
        ax.set_ylim(0.1, 600)


    def draw_b(bx):
        # (b) the CPU model, per framework and machine, under the transport mock
        # colour = framework, line style = machine
        styles = {"langgraph_m5": (BLUE, ":", "s", "LangGraph, M5 laptop"),
                  "langgraph_x86": (BLUE, "--", "o", "LangGraph, x86 desktop"),
                  "langgraph_agentcore": (BLUE, "-", "D", "LangGraph, AgentCore (Graviton3)"),
                  "strands_x86": (GREEN, "--", "^", "Strands, x86 desktop"),
                  "strands_agentcore_graviton3": (GREEN, "-", "v", "Strands, AgentCore (Graviton3)"),
                  "strands_agentcore_graviton2": (GREEN, "-.", "<", "Strands, AgentCore (Graviton2)")}
        for k, (col, ls, mk, lab) in styles.items():
            f = res["fits"].get(k)
            if not f:
                continue
            px, py = np.array(f["points"][0]), np.array(f["points"][1])
            bx.plot(px[1::3], py[1::3], mk, color=col, ms=2.2, alpha=0.6, lw=0)
            xx = np.linspace(px.min(), px.max(), 10)
            bx.plot(xx, f["c_fixed_ms"] + f["ms_per_kb"] * xx, color=col, ls=ls, lw=1.0,
                    label=f"{lab}: {f['ms_per_kb']:.2f}")
        bx.set_xlabel("serialised request body (KB)"); bx.set_ylabel("model-path CPU per step (ms)")
        bx.set_ylim(0, 17)   # headroom so the legend sits above the lines
        bx.legend(loc="upper left", frameon=False, fontsize=5.6, title="ms per KB", title_fontsize=5.6)

    os.makedirs(P("paper", "figures"), exist_ok=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    draw_a(ax); draw_b(bx)
    ax.set_title("(a) Environments held per busy core", loc="left")
    bx.set_title("(b) CPU is linear in payload", loc="left")
    for a_ in (ax, bx):
        a_.grid(True, which="major")
    for ext in ("pdf", "png"):
        fig.savefig(P("paper", "figures", f"fig4_sensitivity.{ext}"))
    # single-column versions for the paper
    def corrected_cores(x):
        """Upper end after the idle-wait effect: scale only the benchmark-derived
        part of the deployed CPU (client serialisation) by the factor measured in
        results/capacity_validation.json; the in-place overhead already includes it."""
        val = res.get("validation")
        if not val or x.get("cpu_ms_per_step") is None or not ("cpu_range_ms" in x or "cpu_parts_ms" in x):
            return None
        f = sum(val["measured_over_predicted"]) / 2
        if "cpu_parts_ms" in x:
            o, ser = x["cpu_parts_ms"]
        else:
            o, hi = x["cpu_range_ms"]; ser = hi - o
        c = o + f * ser
        return N_LIMIT * c / 1000 / (x["wait_s_per_step"] + c / 1000)

    def draw_bars(ax):
        """Plain-language version of the surface: busy CPU cores while 1,000
        sessions are held, one bar per scenario."""
        sc = {x["name"]: x for x in res["scenarios"]}
        rows = [  # (scenario key, label, colour, extrapolated?)
            ("As submitted: M5, mocked socket, calibrated waits", "Laboratory: laptop CPU", ORANGE, False),
            ("Same, measured on AgentCore (Graviton3)", "Laboratory: AgentCore CPU", ORANGE, False),
            ("Deployed LangGraph on AgentCore, medians", "Deployed LangGraph", BLUE, False),
            ("Deployed Strands on AgentCore, medians", "Deployed Strands", GREEN, False),
            ("Deployed LangGraph, fast step (p5 wait), high CPU estimate", "Deployed: fastest steps", BLUE, False),
            ("Extrapolated to 50k-token context, median wait", "Extrapolated: 50k tokens", MUTED, True),
            ("Extrapolated to 250k-token context, median wait", "Extrapolated: 250k tokens", MUTED, True),
        ]
        rows = [r for r in rows if r[0] in sc]
        val = res.get("validation")
        if val:   # directly measured, inserted after the deployed rows
            lo_v, hi_v = val["cores_per_1000_measured"]
            sc["__measured__"] = {"cores_at_limit": (lo_v + hi_v) / 2, "cores_at_limit_range": (lo_v, hi_v),
                                  "wait_s_per_step": val["w_mean_s"], "cpu_ms_per_step": None,
                                  "cpu_range_ms": (min(val["cpu_per_call_ms"]), max(val["cpu_per_call_ms"]))}
            idx = next(i for i, r in enumerate(rows) if r[1].startswith("Deployed Strands")) + 1
            rows.insert(idx, ("__measured__", "Measured: 50–400 sessions", INK, False))
        y = np.arange(len(rows))[::-1]
        for yi, (key, label, col, extrap) in zip(y, rows):
            x = sc[key]
            v = x["cores_at_limit"]
            ax.barh(yi + 0.12, v, color=col, alpha=0.35 if extrap else 0.85, height=0.46,
                    hatch="///" if extrap else None, edgecolor=col, lw=0.6)
            if "cores_at_limit_range" in x:
                lo, hi = x["cores_at_limit_range"]
                ax.errorbar(v, yi + 0.12, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, lw=0.7, capsize=2)
                txt = f"{lo:.1f}–{hi:.1f} cores"
                xt = hi
                corr = corrected_cores(x)
                if corr:
                    ax.plot([hi, corr], [yi + 0.12] * 2, color=INK, lw=0.7, ls=(0, (2, 1.5)))
                    ax.plot([corr], [yi + 0.12], marker="|", color=INK, ms=5, mew=0.7)
                    txt = f"{lo:.1f}–{hi:.1f} (to {corr:.0f}) cores"
                    xt = corr
            else:
                txt = f"{v:.2g} cores" if v < 10 else f"{v:.0f} cores"
                xt = v
                corr = corrected_cores(x)
                if corr:
                    ax.plot([v, corr], [yi + 0.12] * 2, color=INK, lw=0.7, ls=(0, (2, 1.5)))
                    ax.plot([corr], [yi + 0.12], marker="|", color=INK, ms=5, mew=0.7)
                    txt = f"{v:.1f} (to {corr:.0f}) cores"
                    xt = corr
            ax.text(xt * 1.25, yi + 0.12, txt, va="center", fontsize=6.3, color=INK)
            w, c = x["wait_s_per_step"], x["cpu_ms_per_step"]
            crange = x.get("cpu_range_ms")
            cdesc = f"{crange[0]:.1f}–{crange[1]:.0f} ms" if crange else (f"{c:.1f} ms" if c < 10 else f"{c:.0f} ms")
            ax.text(0.012, yi - 0.30, f"waits {w:.1f} s, computes {cdesc} per step", fontsize=5.4,
                    color=MUTED, va="center", transform=ax.get_yaxis_transform())
        ax.set_yticks(y + 0.12)
        ax.set_yticklabels([r[1] for r in rows])
        ax.axvline(1000, color=INK, lw=0.9, ls="--")
        ax.text(1000 * 0.85, y.max() + 0.6, "all 1,000\nbusy", ha="right", va="top", fontsize=6, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(0.05, 2000)
        ax.set_ylim(-0.7, len(rows) - 0.1)
        ax.set_xticks([0.1, 1, 10, 100, 1000])
        ax.set_xticklabels(["0.1", "1", "10", "100", "1,000"])
        ax.set_xlabel("CPU cores in use while 1,000 sessions are held")
        ax.grid(True, axis="x", which="major")
        ax.tick_params(axis="y", length=0)

    def draw_ranges(ax):
        """One band per scenario: the range of CPU cores busy while 1,000 sessions
        are held, with the slower-steps-after-waiting correction already folded
        into the deployed upper ends. Designed to read without the text."""
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        sc = {x["name"]: x for x in res["scenarios"]}
        val = res.get("validation")
        f = sum(val["measured_over_predicted"]) / 2 if val else 1.0

        def corrected_c(x):                       # CPU per step after the correction
            o, hi = x["cpu_range_ms"]; return o + f * (hi - o)

        lab_lo = sc["As submitted: M5, mocked socket, calibrated waits"]["cores_at_limit"]
        lab_hi = sc["Same, measured on AgentCore (Graviton3)"]["cores_at_limit"]
        lg = sc["Deployed LangGraph on AgentCore, medians"]; sd = sc["Deployed Strands on AgentCore, medians"]
        e50 = sc["Extrapolated to 50k-token context, median wait"]; e250 = sc["Extrapolated to 250k-token context, median wait"]
        rows = [
            ("Benchmark only", "benchmark CPU, steps run back to back", lab_lo, lab_hi, ORANGE, "est"),
            ("Deployed LangGraph", f"real waits ({lg['wait_s_per_step']:.1f} s); CPU {lg['cpu_range_ms'][0]:.1f}–{corrected_c(lg):.0f} ms per step",
             lg["cores_at_limit_range"][0], corrected_cores(lg), BLUE, "est"),
            ("Deployed Strands", f"real waits ({sd['wait_s_per_step']:.1f} s); CPU {sd['cpu_range_ms'][0]:.1f}–{corrected_c(sd):.0f} ms per step",
             sd["cores_at_limit_range"][0], corrected_cores(sd), GREEN, "est"),
        ]
        if val:
            lo_v, hi_v = val["cores_per_1000_measured"]
            rows.append(("Measured directly", "50–400 real LangGraph sessions, scaled to 1,000", lo_v, hi_v, INK, "meas"))
        ext = lambda x: x["cores_at_limit"] * 1.0
        rows += [("50k-token conversations", "estimate: 5x longer than we measured", e50["cores_at_limit"],
                  e50["cores_at_limit"] * 1.84, MUTED, "extrap"),
                 ("250k-token conversations", "estimate: 25x longer than we measured", e250["cores_at_limit"],
                  e250["cores_at_limit"] * 1.75, MUTED, "extrap")]
        y = np.arange(len(rows))[::-1].astype(float)
        for yi, (label, sub, lo, hi, col, kind) in zip(y, rows):
            ax.barh(yi + 0.12, hi - lo, left=lo, height=0.42, color=col,
                    alpha={"est": 0.85, "meas": 0.9, "extrap": 0.35}[kind],
                    hatch="////" if kind == "extrap" else None, edgecolor=col, lw=0.6)
            fmt = lambda v: f"{v:.2g}" if v < 10 else f"{v:.0f}"
            ax.text(hi * 1.3, yi + 0.12, f"{fmt(lo)}–{fmt(hi)}", va="center", fontsize=6.4, color=INK)
            ax.text(0.012, yi - 0.30, sub, fontsize=5.4, color=MUTED, va="center",
                    transform=ax.get_yaxis_transform())
        ax.set_yticks(y + 0.12)
        ax.set_yticklabels([r[0] for r in rows])
        ax.axvline(1000, color=INK, lw=0.9, ls="--")
        ax.text(1000 * 0.85, y.max() + 0.55, "if all 1,000\nwere busy", ha="right", va="top", fontsize=6, color=INK)
        ax.set_xscale("log"); ax.set_xlim(0.05, 2000); ax.set_ylim(-0.75, len(rows) - 0.05)
        ax.set_xticks([0.1, 1, 10, 100, 1000]); ax.set_xticklabels(["0.1", "1", "10", "100", "1,000"])
        ax.set_xlabel("CPU cores busy while a platform holds 1,000 agent sessions")
        ax.grid(True, axis="x", which="major"); ax.tick_params(axis="y", length=0)
        ax.legend(handles=[Patch(facecolor="#9aa7b5", edgecolor="#9aa7b5",
                                 label="range estimated from measured wait and CPU per step"),
                           Patch(facecolor=INK, alpha=0.9, label="measured directly"),
                           Patch(facecolor=MUTED, alpha=0.35, hatch="////", edgecolor=MUTED,
                                 label="estimate beyond the conversation lengths measured")],
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False, fontsize=6)

    def draw_waffle():
        """Unit chart: each panel is 1,000 squares, one per environment held; the
        squares doing work are coloured. Busy = low estimate; possibly busy = up to
        the high estimate (after-wait CPU included); the rest are held and idle."""
        from matplotlib.patches import Patch
        from matplotlib.colors import to_rgb
        sc = {x["name"]: x for x in res["scenarios"]}
        val = res.get("validation")
        lg = sc["Deployed LangGraph on AgentCore, medians"]; sd = sc["Deployed Strands on AgentCore, medians"]
        e50 = sc["Extrapolated to 50k-token context, median wait"]; e250 = sc["Extrapolated to 250k-token context, median wait"]
        panels = [
            ("Benchmark only", sc["As submitted: M5, mocked socket, calibrated waits"]["cores_at_limit"],
             sc["Same, measured on AgentCore (Graviton3)"]["cores_at_limit"]),
            ("Deployed agents", min(lg["cores_at_limit_range"][0], sd["cores_at_limit_range"][0]),
             max(corrected_cores(lg), corrected_cores(sd))),
        ]
        if val:
            panels.append(("Measured directly", *val["cores_per_1000_measured"]))
        panels += [("50k tokens*", e50["cores_at_limit"], e50["cores_at_limit"] * 1.84),
                   ("250k tokens*", e250["cores_at_limit"], e250["cores_at_limit"] * 1.75)]
        cols, rows_, px = 40, 25, 6                       # 6 px per cell, last px is the gap
        BUSY, MAYBE, IDLE, GAP = INK, "#8fa3b8", "#e3e7eb", "#ffffff"
        fig = plt.figure(figsize=(7.0, 1.72))
        w = 0.188; xs = [0.005 + i * 0.2 for i in range(len(panels))]
        for x0, (title, lo, hi) in zip(xs, panels):
            n_lo, n_hi = int(round(lo)), int(round(hi))
            img = np.ones((rows_ * px, cols * px, 3)) * np.array(to_rgb(GAP))
            for k in range(cols * rows_):
                r, c = divmod(k, cols)
                colr = BUSY if k < n_lo else MAYBE if k < n_hi else IDLE
                img[r * px:(r + 1) * px - 1, c * px:(c + 1) * px - 1] = to_rgb(colr)
            ax = fig.add_axes([x0, 0.17, w, 0.55])
            ax.imshow(img, interpolation="nearest"); ax.axis("off")
            fmt = lambda v: "<1" if v < 1 else f"{v:.0f}"
            busy = f"{fmt(lo)}–{fmt(hi)}" if fmt(lo) != fmt(hi) else fmt(lo)
            ax.text(0, 1.20, title, transform=ax.transAxes, fontsize=7.2, va="bottom", ha="left",
                    color=INK, fontweight="bold")
            ax.text(0, 1.04, f"{busy} of 1,000 busy", transform=ax.transAxes, fontsize=6.6,
                    va="bottom", ha="left", color=INK)
        fig.legend(handles=[Patch(color=BUSY, label="busy (low estimate)"),
                            Patch(color=MAYBE, label="busy at the high estimate"),
                            Patch(color=IDLE, label="held, waiting on the model")],
                   loc="lower left", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.0, -0.02))
        fig.text(0.995, 0.03, "* conversations longer than any we measured", ha="right", fontsize=6, color=MUTED)
        for ext in ("pdf", "png"):
            fig.savefig(P("paper", "figures", f"fig_capacity_waffle.{ext}"), dpi=300)
        plt.close(fig)

    draw_waffle()

    f1, a1 = plt.subplots(figsize=(3.4, 3.3))
    draw_ranges(a1)
    for ext in ("pdf", "png"):
        f1.savefig(P("paper", "figures", f"fig_capacity_ranges.{ext}"))
    plt.close(f1)

    f1, a1 = plt.subplots(figsize=(3.4, 3.0))
    draw_bars(a1)
    for ext in ("pdf", "png"):
        f1.savefig(P("paper", "figures", f"fig_capacity_bars.{ext}"))
    plt.close(f1)
    for name, fn in (("fig_surface", draw_a), ("fig_cpu_payload", draw_b)):
        f1, a1 = plt.subplots(figsize=(3.4, 2.5))
        fn(a1); a1.grid(True, which="major")
        for ext in ("pdf", "png"):
            f1.savefig(P("paper", "figures", f"{name}.{ext}"))
        plt.close(f1)
    print("\nwrote paper/figures/fig4_sensitivity, fig_surface, fig_cpu_payload {pdf,png}")


if __name__ == "__main__":
    main()
