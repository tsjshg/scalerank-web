"""Who ends up as a visual outlier, and is it the data or the method?

A point is an *outlier* in an embedding when its 10-NN radius there is more
than 3x the median.  For each embedding this reports how many there are and
two properties of those points in the original space:

- relative 10-NN radius: >> 1 means the point really is isolated (faithful);
- k-occurrence N10 (how many points list it among their 10 NN): low values
  are *anti-hubs*, which high-dimensional data produces in large numbers
  (hubness).  ScaleRank's hard negatives strand anti-hubs (the research notes).

Embeddings come from the drbench cache (method specs as given to
``drbench run``) or, with ``--fit``, are computed with ScaleRank kwargs.

    python scripts/diagnose_outliers.py hier_blobs -m tsne "scalerank:gamma=1.0"
    python scripts/diagnose_outliers.py hier_blobs --fit hard=0.0 --fit hard=0.3
"""

import argparse

import numpy as np
from scipy.stats import skew, spearmanr
from sklearn.neighbors import NearestNeighbors

from drbench import datasets, methods
from drbench.metrics import local_radius, rank_metrics
from drbench.runner import _paths


def k_occurrence(X, k=10):
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    return np.bincount(idx.ravel(), minlength=len(X))


def outliers(Y, k=10, factor=3.0):
    r = local_radius(Y, k)
    return r / np.median(r) > factor, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("-m", "--methods", nargs="*", default=[], help="cached method specs")
    ap.add_argument("--fit", action="append", default=[], help="ScaleRank kwargs, e.g. hard=0.0,gamma=1.5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = datasets.load(args.dataset)
    nk = k_occurrence(ds.X)
    rx = local_radius(ds.X, 10)
    rx_rel = rx / np.median(rx)
    anti = nk <= 2
    print(f"{ds.name}: hubness skew(N10)={skew(nk):.2f}, anti-hubs (N10<=2)={anti.mean():.1%}")

    runs = [(spec, lambda s=spec: np.load(_paths(ds.name, s, args.seed)["emb"])) for spec in args.methods]
    for kw in args.fit:
        spec = f"scalerank:{kw}"
        runs.append((spec, lambda s=spec: methods.run(s, ds.X, seed=args.seed)))

    print(f"  {'embedding':48s} {'outl':>6s} {'P(out|anti)':>11s} {'P(out|N10>=10)':>14s} "
          f"{'radius_X out/in':>15s} {'overlap@10 out/in':>17s}")
    for spec, get in runs:
        Y = get()
        out, r = outliers(Y)
        ov = rank_metrics(ds.X, Y, ks=(10,)).pointwise_overlap[10]

        def ratio(v):
            return f"{v[out].mean():.2f}/{v[~out].mean():.2f}" if out.any() else f"  -/{v.mean():.2f}"

        p_anti = f"{out[anti].mean():.2f}" if anti.any() else "-"
        print(f"  {spec:48s} {out.mean():6.1%} {p_anti:>11s} {out[nk >= 10].mean():14.2f} "
              f"{ratio(rx_rel):>15s} {ratio(ov):>17s}   corr(outlier score, N10)={spearmanr(r, nk).statistic:+.2f}")


if __name__ == "__main__":
    main()
