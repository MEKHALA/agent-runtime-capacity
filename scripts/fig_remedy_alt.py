"""Single-panel redesign of the section 7 simulation figure.

One row per hosting choice, ordered from most to fewest environments: the bar is
how many environments 1,000 concurrent sessions need; the text beside it is the
extra delay sharing adds to a step (slowest 1% of steps, across the CPU range).

Reads results/remedy_sim.json (scripts/simulate_remedy.py).

    .venv/bin/python scripts/fig_remedy_alt.py
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
INK, MUTED = "#1a211d", "#5e6862"
ISO, SHARED = "#d95f02", "#4a7fd4"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
                     "ytick.color": INK, "xtick.labelsize": 6.8, "ytick.labelsize": 7.2,
                     "grid.color": "#dce1d9", "grid.linewidth": 0.5,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

d = json.load(open(P("results", "remedy_sim.json")))
runs = d["runs"]
cpu_levels = sorted({r["cpu_ms"] for r in runs})
mid = cpu_levels[len(cpu_levels) // 2]


def envs(policy, **kw):
    r = [x for x in runs if x["policy"] == policy and x["cpu_ms"] == mid
         and all(x.get(k) == v for k, v in kw.items())][0]
    return r["envs_mean"]


def p99_range(k):
    v = [x["added_latency_ms"]["p99"] for x in runs if x["policy"] == "multiplexed" and x["k"] == k]
    return min(v), max(v)


def delay_text(k):
    lo, hi = p99_range(k)
    if hi < 0.5:
        return "no extra delay"
    return f"+{lo:.0f}–{hi:.0f} ms per step" if lo >= 0.5 else f"up to +{hi:.0f} ms per step"


rows = [  # (label, environments, colour, hatch, delay text)
    ("One per session,\nleft to the 15-min idle timeout", envs("isolated", tail_s=900.0), ISO, "light", "no extra delay"),
    ("One per session,\nclosed when the conversation ends", envs("isolated", tail_s=0.0), ISO, None, "no extra delay"),
    ("Shared: 16 sessions\nper environment", envs("multiplexed", k=16), SHARED, None, delay_text(16)),
    ("Shared: 64 sessions\nper environment", envs("multiplexed", k=64), SHARED, None, delay_text(64)),
    ("Shared: 256 sessions\nper environment", envs("multiplexed", k=256), SHARED, None, delay_text(256)),
]

fig, ax = plt.subplots(figsize=(3.4, 2.75))
y = np.arange(len(rows))[::-1]
for yi, (label, n, col, hatch, delay) in zip(y, rows):
    ax.barh(yi, n, color=col, alpha=0.45 if hatch else 0.85, edgecolor=col, lw=0.6, height=0.58)
    ax.text(n * 1.25, yi + 0.13, f"{n:,.0f}", va="center", fontsize=7.2, fontweight="bold")
    ax.text(n * 1.25, yi - 0.17, delay, va="center", fontsize=6.3, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.6, linespacing=1.1)
ax.set_xscale("log"); ax.set_xlim(1, 4e5)
from matplotlib.ticker import NullLocator
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticks([1, 10, 100, 1000, 10000]); ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
ax.set_xlabel("environments needed for 1,000 concurrent sessions")
ax.grid(True, axis="x"); ax.tick_params(axis="y", length=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(handles=[Patch(color=ISO, alpha=0.85, label="one environment per session (isolated)"),
                   Patch(color=SHARED, alpha=0.85, label="sessions share an environment")],
          loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, frameon=False, fontsize=6.4)
fig.text(0.0, -0.06, "Delay: extra time a step waits for CPU when sharing, in the slowest 1% of steps,\n"
         f"across the measured CPU range. A typical model call takes {2741:,} ms.",
         fontsize=5.9, color=MUTED, va="top")
for ext in ("pdf", "png"):
    fig.savefig(P("paper", "figures", f"fig_remedy_bars.{ext}"), dpi=300)
print("wrote paper/figures/fig_remedy_bars.{pdf,png}")
for r in rows:
    print(f"  {r[0].replace(chr(10), ' '):55s} {r[1]:8,.0f}  {r[4]}")
