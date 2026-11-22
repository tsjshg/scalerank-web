"""Can tuned t-SNE / UMAP reach where ScaleRank sits on the local-global plane?

Reads cached results for the t-SNE grid (perplexity x exaggeration, plus
random init) and the UMAP grid (n_neighbors x min_dist, plus random init)
and compares them with ScaleRank (current default at gamma 1/0.5/0) on
local R_NX@10 vs global distance correlation.

    # the sweep (takes a while; 30 specs x 8 datasets x 3 seeds)
    drbench run -m tsne:perplexity=5,exaggeration=1 ... umap:init=random
    python scripts/compare_tuned_baselines.py

Prints, per dataset, whether any tuned setting dominates each ScaleRank
point, how far the tuned front reaches, per-parameter effects averaged over
datasets, and writes results/figures/tuned_tradeoff.png.
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drbench.metrics import local_radius
from drbench.runner import RESULTS_DIR, _paths, load_results, summarize

plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ORDER = ["s_curve", "hier_blobs", "density_blobs", "digits", "pollen", "usoskin", "kolodziejczyk", "buettner"]
SR = {"scalerank:gamma=1.0": "γ=1", "scalerank:gamma=0.5": "γ=0.5", "scalerank:gamma=0.0": "γ=0"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_TSNE, C_UMAP, C_SR, C_MDS, C_PCA = "#eb6834", "#1baf7a", "#e34948", "#008300", "#2a78d6"


def family(spec):
    if spec.startswith("tsne:"):
        return "tsne"
    if spec.startswith("umap:"):
        return "umap"
    return None


def short(spec):
    s = spec.split(":", 1)[1]
    s = s.replace("perplexity=", "p").replace(",exaggeration=", " e").replace("initialization=random", "random init")
    s = s.replace("n_neighbors=", "k").replace(",min_dist=", " md").replace("init=random", "random init")
    return ("t-SNE " if spec.startswith("tsne") else "UMAP ") + s


def outlier_frac(ds, spec):
    fr = []
    for s in (0, 1, 2):
        p = _paths(ds, spec, s)["emb"]
        if p.exists():
            r = local_radius(np.load(p), 10)
            fr.append(np.mean(r / np.median(r) > 3))
    return float(np.mean(fr)) if fr else np.nan


def pareto(df):
    """Rows not dominated on (rnx@10, dist_spearman)."""
    keep = []
    for i, r in df.iterrows():
        dom = df[(df["rnx@10"] >= r["rnx@10"]) & (df.dist_spearman >= r.dist_spearman)
                 & ((df["rnx@10"] > r["rnx@10"]) | (df.dist_spearman > r.dist_spearman))]
        if dom.empty:
            keep.append(i)
    return df.loc[keep].sort_values("dist_spearman")


def main():
    t = summarize(load_results()).reset_index()
    t["family"] = t.method.map(family)
    tuned = t[t.family.notna()].copy()
    if tuned.empty:
        raise SystemExit("no tuned t-SNE/UMAP results cached; run the sweep first")
    n_cfg = tuned.groupby("family").method.nunique().to_dict()
    print(f"tuned settings: {n_cfg}")

    # ---- 1. per dataset: is ScaleRank dominated by any tuned setting?
    print("\n== per dataset: tuned settings that beat ScaleRank on both local R_NX@10 and global distance corr.")
    rows = []
    for ds in ORDER:
        tu = tuned[tuned.dataset == ds]
        for spec, g in SR.items():
            r = t[(t.dataset == ds) & (t.method == spec)]
            if r.empty:
                continue
            r = r.iloc[0]
            dom = tu[(tu["rnx@10"] >= r["rnx@10"]) & (tu.dist_spearman >= r.dist_spearman)]
            # best global among tuned settings at least as local as ScaleRank, and vice versa
            g_at = tu[tu["rnx@10"] >= r["rnx@10"]].dist_spearman.max()
            l_at = tu[tu.dist_spearman >= r.dist_spearman]["rnx@10"].max()
            rows.append(dict(dataset=ds, gamma=g, sr_local=r["rnx@10"], sr_global=r.dist_spearman,
                             tuned_global_at_same_local=g_at, tuned_local_at_same_global=l_at,
                             dominated_by=", ".join(short(s) for s in dom.method) or "-"))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(res.round(3).to_string(index=False))

    # ---- 2. per-setting means over datasets
    print("\n== mean over 8 datasets for each setting")
    means = tuned.groupby(["family", "method"])[["rnx@10", "auc_rnx", "dist_spearman", "density_corr", "time_s"]].mean()
    means["outliers"] = [np.mean([outlier_frac(ds, m) for ds in ORDER]) for _, m in means.index]
    means.index = [short(m) for _, m in means.index]
    ref = {}
    for spec, name in [*SR.items(), ("tsne", "t-SNE default"), ("umap", "UMAP default"), ("mds", "MDS"), ("pca", "PCA")]:
        sub = t[t.method == spec]
        if len(sub):
            ref[name] = {**sub[["rnx@10", "auc_rnx", "dist_spearman", "density_corr", "time_s"]].mean().to_dict(),
                         "outliers": np.mean([outlier_frac(ds, spec) for ds in ORDER])}
    means = pd.concat([means, pd.DataFrame(ref).T.rename(index=lambda s: f"[{s}]")])
    print(means.sort_values("dist_spearman").round(3).to_string())

    # ---- 3. parameter effects
    def params(spec):
        return dict(re.findall(r"(\w+)=([\w.]+)", spec.split(":", 1)[1]))

    print("\n== t-SNE: mean over datasets by perplexity x exaggeration  (R_NX@10 / distance corr.)")
    ts = tuned[(tuned.family == "tsne") & tuned.method.str.contains("perplexity")].copy()
    ts["perplexity"] = ts.method.map(lambda s: int(params(s)["perplexity"]))
    ts["exaggeration"] = ts.method.map(lambda s: float(params(s)["exaggeration"]))
    print(ts.pivot_table(index="perplexity", columns="exaggeration", values="rnx@10").round(3).to_string())
    print(ts.pivot_table(index="perplexity", columns="exaggeration", values="dist_spearman").round(3).to_string())
    print("\n== UMAP: mean over datasets by n_neighbors x min_dist  (R_NX@10 / distance corr.)")
    um = tuned[(tuned.family == "umap") & tuned.method.str.contains("n_neighbors")].copy()
    um["n_neighbors"] = um.method.map(lambda s: int(params(s)["n_neighbors"]))
    um["min_dist"] = um.method.map(lambda s: float(params(s)["min_dist"]))
    print(um.pivot_table(index="n_neighbors", columns="min_dist", values="rnx@10").round(3).to_string())
    print(um.pivot_table(index="n_neighbors", columns="min_dist", values="dist_spearman").round(3).to_string())

    # ---- 4. figure: 8 datasets + mean panel
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), dpi=140)
    panels = [(ds, t[t.dataset == ds]) for ds in ORDER]
    mean_t = t.groupby("method")[["rnx@10", "dist_spearman"]].mean().reset_index()
    mean_t["family"] = mean_t.method.map(family)
    panels.append(("8 データ平均", mean_t))
    for ax, (name, sub) in zip(axes.flat, panels):
        for fam, c, lab in (("tsne", C_TSNE, "t-SNE（調整）"), ("umap", C_UMAP, "UMAP（調整）")):
            f = sub[sub.method.map(family) == fam]
            ax.scatter(f.dist_spearman, f["rnx@10"], s=22, color=c, alpha=0.55, linewidths=0, label=lab, zorder=2)
            front = pareto(f)
            ax.plot(front.dist_spearman, front["rnx@10"], color=c, lw=1.2, alpha=0.9, zorder=2)
        srp = sub[sub.method.isin(SR)].set_index("method").loc[[s for s in SR if s in set(sub.method)]]
        ax.plot(srp.dist_spearman, srp["rnx@10"], color=C_SR, lw=2.2, zorder=3)
        ax.scatter(srp.dist_spearman, srp["rnx@10"], s=46, color=C_SR, edgecolors="white", linewidths=1.5, zorder=4, label="ScaleRank")
        for spec, row in srp.iterrows():
            ax.annotate(SR[spec], (row.dist_spearman, row["rnx@10"]), xytext=(5, 4), textcoords="offset points", fontsize=7.5, color=MUTED)
        for spec, c, lab in (("mds", C_MDS, "MDS"), ("pca", C_PCA, "PCA")):
            r = sub[sub.method == spec]
            if len(r):
                ax.scatter(r.dist_spearman, r["rnx@10"], s=30, color=c, marker="s", zorder=3, label=lab)
        if name == "8 データ平均":
            for _, r in pd.concat([pareto(mean_t[mean_t.family == "tsne"]), pareto(mean_t[mean_t.family == "umap"])]).iterrows():
                ax.annotate(short(r.method).replace("t-SNE ", "").replace("UMAP ", ""), (r.dist_spearman, r["rnx@10"]),
                            xytext=(4, -9), textcoords="offset points", fontsize=6.5, color=MUTED)
        ax.set_title(name, loc="left", fontsize=10, color=TEXT)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel("大域: 距離相関 →", fontsize=8.5, color=TEXT)
    for ax in axes[:, 0]:
        ax.set_ylabel("局所: R_NX@10 →", fontsize=8.5, color=TEXT)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle("t-SNE / UMAP のパラメータ調整 vs ScaleRank（線は各手法のパレート前線、右上ほど良い）",
                 x=0.01, ha="left", fontsize=11.5, color=TEXT)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = RESULTS_DIR / "figures" / "tuned_tradeoff.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
