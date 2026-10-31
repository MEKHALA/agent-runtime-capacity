"""Two candidate designs for the paper's capacity figure (section 5).

  fig_capacity_waffle3   three unit charts: of every 1,000 sessions held, how
                         many are computing (deployed, measured, 250k worst case)
  fig_session_timeline   one real deployed session drawn to scale: the
                         environment is held throughout; CPU is used in slivers

Reads results/sensitivity.json (run scripts/sensitivity.py first) and
data/deployed_turns.csv.

    .venv/bin/python scripts/fig_capacity_alt.py
"""
import csv, json, os, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, ConnectionPatch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
BLUE, GREEN, INK, MUTED = "#4a7fd4", "#1b9e77", "#1a211d", "#5e6862"
BUSY, MAYBE, IDLE = INK, "#8fa3b8", "#e3e7eb"
WAIT = "#9ecae1"                      # timeline: waiting on the model (light blue)
CPU_LO, CPU_HI = "#d95f02", "#fdae6b"   # timeline: computing, low / high estimate (dark / light orange)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
                     "ytick.color": MUTED, "xtick.labelsize": 6.8, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})

res = json.load(open(P("results", "sensitivity.json")))
sc = {x["name"]: x for x in res["scenarios"]}
FACTOR = sum(res["validation"]["measured_over_predicted"]) / 2   # extra CPU of a step after a wait


def corrected(x):
    o, hi = x["cpu_range_ms"]
    c = o + FACTOR * (hi - o)
    return 1000 * (c / 1000) / (x["wait_s_per_step"] + c / 1000)


def waffle3():
    lg, sd = sc["Deployed LangGraph on AgentCore, medians"], sc["Deployed Strands on AgentCore, medians"]
    e250 = sc["Extrapolated to 250k-token context, median wait"]
    lo_m, hi_m = res["validation"]["cores_per_1000_measured"]
    panels = [
        ("Deployed agents", "estimated from 3,549 real steps",
         min(lg["cores_at_limit_range"][0], sd["cores_at_limit_range"][0]), max(corrected(lg), corrected(sd))),
        ("Measured directly", "50–400 concurrent sessions, scaled",
         lo_m, hi_m),
        ("Worst case: 250k-token context", "estimate, 25x longer than measured",
         e250["cores_at_limit"], e250["cores_at_limit"] * 1.75),
    ]
    cols, rows_, px = 40, 25, 6
    fig = plt.figure(figsize=(7.0, 1.95))
    for i, (title, sub, lo, hi) in enumerate(panels):
        n_lo, n_hi = int(round(lo)), int(round(hi))
        img = np.ones((rows_ * px, cols * px, 3))
        for k in range(cols * rows_):
            r, c = divmod(k, cols)
            img[r * px:(r + 1) * px - 1, c * px:(c + 1) * px - 1] = to_rgb(
                BUSY if k < n_lo else MAYBE if k < n_hi else IDLE)
        ax = fig.add_axes([0.005 + i * 0.335, 0.16, 0.30, 0.54])
        ax.imshow(img, interpolation="nearest"); ax.axis("off")
        ax.text(0, 1.30, title, transform=ax.transAxes, fontsize=7.6, fontweight="bold", va="bottom")
        ax.text(0, 1.15, f"{n_lo}–{n_hi} of every 1,000 sessions computing", transform=ax.transAxes,
                fontsize=7, va="bottom")
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=6.2, color=MUTED, va="bottom")
    fig.legend(handles=[Patch(color=BUSY, label="computing (low estimate)"),
                        Patch(color=MAYBE, label="computing at the high estimate"),
                        Patch(color=IDLE, label="held, waiting on the model")],
               loc="lower left", ncol=3, frameon=False, fontsize=6.6, bbox_to_anchor=(0.0, -0.03))
    for ext in ("pdf", "png"):
        fig.savefig(P("paper", "figures", f"fig_capacity_waffle3.{ext}"), dpi=300)
    plt.close(fig)


def timeline():
    """A real deployed LangGraph session (closest to the median step count and
    median total wait), with each step's compute drawn at the high CPU estimate."""
    rows = [r for r in csv.DictReader(l for l in open(P("data", "deployed_turns.csv")) if not l.startswith("#"))
            if r["config"] == "c2-langgraph-agentcore"]
    by = {}
    for r in rows:
        by.setdefault(r["run_id"], []).append((int(r["turn_index"]), float(r["model_ms"]) / 1000))
    sess = [sorted(v) for v in by.values()]
    med_n, med_t = st.median(len(v) for v in sess), st.median(sum(w for _, w in v) for v in sess)
    waits = [w for _, w in min(sess, key=lambda v: (abs(len(v) - med_n), abs(sum(w for _, w in v) - med_t)))]
    lg = sc["Deployed LangGraph on AgentCore, medians"]
    c_lo = lg["cpu_range_ms"][0] / 1000
    c_hi = (lg["cpu_range_ms"][0] + FACTOR * (lg["cpu_range_ms"][1] - lg["cpu_range_ms"][0])) / 1000

    fig = plt.figure(figsize=(7.0, 2.4))
    a1 = fig.add_axes([0.02, 0.70, 0.96, 0.12])   # whole session
    a2 = fig.add_axes([0.02, 0.36, 0.96, 0.12])   # one step
    a3 = fig.add_axes([0.02, 0.05, 0.30, 0.12])   # first 60 ms of that step

    def draw(ax, segs, t0, t1, h=1.0):
        for kind, a, b in segs:
            if b < t0 or a > t1:
                continue
            ax.add_patch(plt.Rectangle((a, 0), b - a, h, color={"wait": WAIT, "hi": CPU_HI, "lo": CPU_LO}[kind], lw=0))
        ax.set_xlim(t0, t1); ax.set_ylim(0, h); ax.set_yticks([])
        for sp in ("left", "right", "top"):
            ax.spines[sp].set_visible(False)

    segs, t = [], 0.0
    for w in waits:
        segs += [("lo", t, t + c_lo), ("hi", t + c_lo, t + c_hi), ("wait", t + c_hi, t + c_hi + w)]
        t += c_hi + w
    total = t
    draw(a1, segs, 0, total)
    a1.set_xticks(range(0, int(total) + 1, 5)); a1.set_xticklabels([f"{x} s" for x in range(0, int(total) + 1, 5)])
    # pointers above each computing sliver: the slivers are drawn to scale, so they are thin
    starts = [kind_a for kind, kind_a, _ in segs if kind == "lo"]
    a1.plot(starts, [1.12] * len(starts), "v", color=CPU_LO, ms=4.2, clip_on=False)
    a1.text(0, 2.05, f"One real deployed session: {len(waits)} steps, {total:.0f} s. The platform holds an "
            f"isolated environment for all of it.", transform=a1.transAxes, fontsize=7.4, fontweight="bold")
    busy_lo, busy_hi = len(waits) * c_lo / total * 100, len(waits) * c_hi / total * 100
    a1.text(0, 1.62, f"Computing: {busy_lo:.1f}–{busy_hi:.1f}% of the session (orange slivers, marked ▼). "
            f"Waiting on the model: the rest.", transform=a1.transAxes, fontsize=6.8, color=MUTED)

    k = 2                                           # zoom on the third step
    s0 = sum(c_hi + w for w in waits[:k]); s1 = s0 + c_hi + waits[k]
    seg_k = [(kind, a - s0, b - s0) for kind, a, b in segs if a >= s0 - 1e-9 and b <= s1 + 1e-9]
    draw(a2, seg_k, 0, s1 - s0)
    a2.text((s1 - s0) * 0.5, 0.5, f"waiting on the model: {waits[k]:.1f} s", ha="center", va="center",
            fontsize=7, color=INK)
    a2.set_xticks(np.arange(0, s1 - s0 + 0.01, 0.5)); a2.set_xticklabels([f"{x:.1f} s" for x in np.arange(0, s1 - s0 + 0.01, 0.5)])
    a2.text(0, 1.18, f"One step, zoomed: {waits[k]:.1f} s waiting on the model, {c_lo*1000:.1f}–{c_hi*1000:.0f} ms "
            f"computing (the orange sliver at the left edge).", transform=a2.transAxes, fontsize=6.8, color=MUTED)
    for x in (s0, s1):                              # zoom guides from the session bar
        fig.add_artist(ConnectionPatch(xyA=(x, 0), coordsA=a1.transData,
                                       xyB=((x - s0), 1), coordsB=a2.transData, color=MUTED, lw=0.5, ls=":"))

    draw(a3, seg_k, 0, 0.06)
    a3.text(c_hi / 2, 0.5, "computing", ha="center", va="center", fontsize=6.4, color=INK)
    a3.text(c_hi + (0.06 - c_hi) / 2, 0.5, "waiting", ha="center", va="center", fontsize=6.4, color=INK)
    a3.set_xticks([0, 0.02, 0.04, 0.06]); a3.set_xticklabels(["0", "20 ms", "40 ms", "60 ms"])
    a3.text(0, 1.18, "Its first 60 ms, zoomed again.", transform=a3.transAxes, fontsize=6.8, color=MUTED)
    fig.add_artist(ConnectionPatch(xyA=(0, 0), coordsA=a2.transData, xyB=(0, 1), coordsB=a3.transData,
                                   color=MUTED, lw=0.5, ls=":"))
    fig.add_artist(ConnectionPatch(xyA=(0.06, 0), coordsA=a2.transData, xyB=(0.06, 1), coordsB=a3.transData,
                                   color=MUTED, lw=0.5, ls=":"))
    fig.legend(handles=[Patch(color=CPU_LO, label=f"computing, low estimate ({c_lo*1000:.1f} ms/step)"),
                        Patch(color=CPU_HI, label=f"computing, high estimate (to {c_hi*1000:.0f} ms/step)"),
                        Patch(color=WAIT, label="waiting on the model (environment held, CPU idle)"),
                        Line2D([], [], color=CPU_LO, marker="v", ls="none", ms=4.2,
                               label="marks each step's computing (too thin to see at full-session scale)")],
               loc="lower right", bbox_to_anchor=(0.99, 0.02), frameon=False, fontsize=6.6, ncol=1)
    for ext in ("pdf", "png"):
        fig.savefig(P("paper", "figures", f"fig_session_timeline.{ext}"), dpi=300)
    plt.close(fig)
    print(f"timeline: {len(waits)} steps, {total:.1f} s; busy {busy_lo:.2f}-{busy_hi:.2f}%")


if __name__ == "__main__":
    waffle3()
    timeline()
    print("wrote paper/figures/fig_capacity_waffle3.{pdf,png}, fig_session_timeline.{pdf,png}")
