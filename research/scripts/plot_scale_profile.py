"""Plot the ScaleRank scale profile p(K) ∝ K^-gamma for several gamma.

Writes results/figures/scale_profile.png (N = 1797, the size of digits).

    python scripts/plot_scale_profile.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scalerank_research.scalerank import scale_profile

N = 1797  # digits
K = np.arange(1, N - 1)
gammas = [0.0, 0.5, 1.0, 1.5, 2.0]
# ordinal parameter -> one-hue sequential ramp, light to dark (no lighter than step 250)
colors = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
for g, c in zip(gammas, colors):
    p = scale_profile(N, gamma=g)
    cum = np.cumsum(p)
    axes[0].plot(K, p, color=c, lw=2, label=f"γ = {g:g}")
    axes[1].plot(K, cum, color=c, lw=2, label=f"γ = {g:g}")
    axes[0].annotate(f"γ={g:g}", (K[-1], p[-1]), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=8, color=MUTED)
    print(f"gamma={g:<4} p(K=1)={p[0]:6.1%}  K<=10:{cum[9]:6.1%}  K<=100:{cum[99]:6.1%}  K>100:{1-cum[99]:6.1%}  median K={K[np.searchsorted(cum, 0.5)]}")

ax = axes[0]
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("p(K) ∝ K^(−γ)（両対数、傾きが −γ）", loc="left", fontsize=10, color=TEXT)
ax.set_ylabel("p(K)", color=TEXT, fontsize=9)
ax = axes[1]
ax.set_xscale("log")
ax.set_ylim(0, 1.02)
ax.axvline(10, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(10, 0.03, " K=10", color=MUTED, fontsize=8)
ax.set_title("累積: K 以下に割り当てられる重みの割合", loc="left", fontsize=10, color=TEXT)
ax.set_ylabel("Σ p(k), k ≤ K", color=TEXT, fontsize=9)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
for ax in axes:
    ax.set_xlabel("K（近傍サイズ）", color=TEXT, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, which="major")
    ax.set_axisbelow(True)
fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="lower center", ncol=5, labelcolor=TEXT)
fig.suptitle(f"スケール分布 p(K)（N = {N}、digits と同じサイズ）", x=0.01, ha="left", fontsize=11, color=TEXT)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig("results/figures/scale_profile.png")
