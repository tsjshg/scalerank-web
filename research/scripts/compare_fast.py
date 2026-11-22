"""Does ScaleRankFast (no N x N tables) reproduce ScaleRank?

Compares cached ``scalerank:gamma=g`` and ``<fast>:gamma=g`` runs (mean over
seeds) on the 8 benchmark datasets and prints the per-dataset differences
(fast - exact) and their means per gamma.  ``<fast>`` is ``scalerank_fast``
(Python) or, given as the first argument, e.g. ``scalerank_rs`` (Rust port).

    drbench run -m scalerank_fast:gamma=1.0 scalerank_fast:gamma=0.5 scalerank_fast:gamma=0.0
    python scripts/compare_fast.py [scalerank_rs]
"""

import sys

import pandas as pd

from drbench.runner import load_results, summarize

ORDER = ["s_curve", "hier_blobs", "density_blobs", "digits", "pollen", "usoskin", "kolodziejczyk", "buettner"]
COLS = ["rnx@10", "auc_rnx", "dist_spearman", "density_corr", "knn_acc"]


def main(fast="scalerank_fast"):
    t = summarize(load_results()).reset_index()
    rows = []
    for g in ("1.0", "0.5", "0.0"):
        for ds in ORDER:
            e = t[(t.dataset == ds) & (t.method == f"scalerank:gamma={g}")]
            f = t[(t.dataset == ds) & (t.method == f"{fast}:gamma={g}")]
            if e.empty or f.empty:
                continue
            e, f = e.iloc[0], f.iloc[0]
            rows.append(dict(gamma=g, dataset=ds, **{c: f[c] - e[c] for c in COLS}, time_exact=e.time_s, time_fast=f.time_s))
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"{fast} - scalerank (mean over seeds)")
    print(r.round(3).to_string(index=False))
    print("\nmean over datasets")
    print(r.groupby("gamma", sort=False)[COLS].mean().round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
