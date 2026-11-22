"""Do the Rust implementations (scalerank-web, via scalerank_rs) match the references?

Mean over seeds, then over the 8 benchmark datasets, of (Rust - reference)
for each pair below, plus the mean run time of each.

    drbench run -m tsne_rs umap_rs umap:init=pca scalerank_rs:gamma=1.0
    python scripts/compare_rs.py
"""

import pandas as pd

from drbench.runner import load_results, summarize

ORDER = ["s_curve", "hier_blobs", "density_blobs", "digits", "pollen", "usoskin", "kolodziejczyk", "buettner"]
COLS = ["rnx@10", "auc_rnx", "dist_spearman", "density_corr", "knn_acc"]
PAIRS = [
    ("tsne_rs", "tsne"),
    ("umap_rs", "umap:init=pca"),
    ("scalerank_rs:gamma=1.0", "scalerank_fast:gamma=1.0"),
    ("scalerank_rs:gamma=0.0", "scalerank_fast:gamma=0.0"),
]


def main():
    t = summarize(load_results()).reset_index()
    t = t[t.dataset.isin(ORDER)]
    rows = []
    for rs, ref in PAIRS:
        a = t[t.method == rs].set_index("dataset")
        b = t[t.method == ref].set_index("dataset")
        common = a.index.intersection(b.index)
        if common.empty:
            continue
        d = (a.loc[common, COLS] - b.loc[common, COLS]).mean()
        rows.append(dict(rust=rs, reference=ref, n_datasets=len(common), **d.round(3),
                         time_rust=a.loc[common, "time_s"].mean().round(2), time_ref=b.loc[common, "time_s"].mean().round(2)))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
