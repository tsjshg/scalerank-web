"""Score cached embeddings under ScaleRank's own objective.

Tells an optimisation failure apart from a surrogate mismatch: if another
method's embedding gets a *lower* ScaleRank loss than ScaleRank itself, the
optimiser is stuck; if ScaleRank is lowest but still loses on a metric, the
objective prefers a different trade-off.  This is how the need for hard
negatives was found (the research notes).

The loss depends on the overall scale of the layout (Student-t kernel), so
each embedding is standardised and the best global scale factor is searched.

    drbench run -d digits             # embeddings must be cached first
    python scripts/score_under_objective.py digits --gammas 1 0
    python scripts/score_under_objective.py hier_blobs --k0 10 --width 0.7   # band profile
"""

import argparse

import numpy as np

from drbench import datasets
from drbench.metrics import rank_metrics
from scalerank_research.scalerank import neighbour_order, sample_triplets, scale_profile, triplet_loss_and_grad
from drbench.runner import RESULTS_DIR, _paths


def best_scaled_loss(Y, triplets, scales):
    Y = Y / Y.std()
    losses = [triplet_loss_and_grad(Y * s, *triplets)[0] for s in scales]
    k = int(np.argmin(losses))
    return losses[k], scales[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--gammas", nargs="+", type=float, default=[1.0, 0.0])
    ap.add_argument("--k0", type=float, default=None, help="use a log-normal band around k0 instead of K^-gamma")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="which cached embeddings to score")
    ap.add_argument("--per-point", type=int, default=200, help="triplets per point for the estimate")
    args = ap.parse_args()

    ds = datasets.load(args.dataset)
    order, rank = neighbour_order(ds.X)
    specs = sorted(p.name.split("__s")[0] for p in (RESULTS_DIR / "embeddings" / ds.name).glob(f"*__s{args.seed}.npy"))
    if not specs:
        raise SystemExit(f"no cached embeddings for {ds.name}; run `drbench run -d {ds.name}` first")
    scales = np.geomspace(0.1, 1000, 50)
    rng = np.random.default_rng(123)

    profiles = [(f"k0={args.k0},width={args.width}", dict(k0=args.k0, width=args.width))] if args.k0 else \
        [(f"gamma={g}", dict(gamma=g)) for g in args.gammas]
    for label, kw in profiles:
        p = scale_profile(ds.n, **kw)
        triplets = sample_triplets(order, p, args.per_point, rng)[:3]  # uniform negatives: the plain objective
        print(f"\n{ds.name}  {label}  (ScaleRank loss, lower is better)")
        print(f"  {'method':26s} {'loss':>7s} {'scale':>7s} {'rnx@10':>7s} {'auc':>6s}")
        rows = []
        for spec in specs:
            Y = np.load(_paths(ds.name, spec, args.seed)["emb"])
            loss, s = best_scaled_loss(Y, triplets, scales)
            rm = rank_metrics(ds.X, Y, ks=())
            edge = " (scale at search edge)" if s in (scales[0], scales[-1]) else ""
            rows.append((loss, spec, s, rm.r_nx[10], rm.auc_r_nx, edge))
        for loss, spec, s, rnx, auc, edge in sorted(rows):
            print(f"  {spec:26s} {loss:7.4f} {s:7.1f} {rnx:7.3f} {auc:6.3f}{edge}")


if __name__ == "__main__":
    main()
