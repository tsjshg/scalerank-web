"""Attribute the outward force on points to the terms of ScaleRank's gradient.

Trains ScaleRank on a labelled dataset with snapshots, marks the points that
end up as outliers (10-NN radius > 3x median), and at selected epochs splits
the displacement each point receives into six sources, projected on the
direction away from its own cluster's centre (positive = outward):

  as anchor: pull toward own in-neighbour j / push from uniform m / from hard m
  as j: pulled by other anchors;  as m: pushed by other anchors (uniform / hard)

This is how the anti-hub mechanism behind ScaleRank's outliers was found
(the research notes): outliers receive ~1/3 of the "as j" pull and ~2x the
hard-negative push of other points, and are left behind as cluster cores
contract.

The decomposition is written for the triplet loss, so it runs the triplet
configuration of the first prototype (n_neg=1, per_point=32, hard=0.3)
unless --params overrides those; n_neg > 1 is rejected.

    python scripts/force_decomposition.py hier_blobs --epochs 100 300 500 700
    python scripts/force_decomposition.py hier_blobs --params hard=0.0
"""

import argparse

import numpy as np
from sklearn.neighbors import NearestNeighbors

from drbench import datasets
from drbench.methods import parse_spec
from drbench.metrics import local_radius
from scalerank_research.scalerank import ScaleRank, neighbour_order, sample_triplets, scale_profile

TERMS = [
    "anchor: pull to own nbr j",
    "anchor: push from uniform m",
    "anchor: push from hard m",
    "as j: pulled by anchors",
    "as m: pushed by anchors (unif)",
    "as m: pushed by anchors (hard)",
]


def force_terms(Y, order, rank, p, per_point, hard, hard_k, rng):
    n = len(Y)
    nb = NearestNeighbors(n_neighbors=hard_k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    i, j, m, _, is_hard = sample_triplets(order, p, per_point, rng, rank, nb, hard)
    dij, dim = Y[i] - Y[j], Y[i] - Y[m]
    sij, sim = (dij**2).sum(1), (dim**2).sum(1)
    s = 1 / (1 + np.exp(-(np.log1p(sij) - np.log1p(sim))))
    gij = (2 * s / (1 + sij))[:, None] * dij
    gim = (-2 * s / (1 + sim))[:, None] * dim
    every = np.ones(len(i), bool)
    # displacement = -gradient, split by which role the point plays
    parts = [(i, -gij, every), (i, -gim, ~is_hard), (i, -gim, is_hard),
             (j, gij, every), (m, gim, ~is_hard), (m, gim, is_hard)]
    out = []
    for idx, vec, mask in parts:
        F = np.zeros((n, 2))
        for c in range(2):
            F[:, c] = np.bincount(idx[mask], weights=vec[mask, c], minlength=n)
        out.append(F / per_point)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="needs categorical labels (cluster centres)")
    ap.add_argument("--params", default="gamma=1.0", help="ScaleRank kwargs, e.g. gamma=1.0,hard=0.0")
    ap.add_argument("--epochs", nargs="+", type=int, default=[100, 300, 500, 700])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = datasets.load(args.dataset)
    if ds.labels is None:
        raise SystemExit(f"{ds.name} has no categorical labels")
    _, given = parse_spec(f"x:{args.params}")
    kw = {"n_neg": 1, "per_point": 32, "hard": 0.3, **given}
    if kw["n_neg"] != 1:
        raise SystemExit("force decomposition is only implemented for the triplet loss (n_neg=1)")
    model = ScaleRank(random_state=args.seed, snapshot_every=50, **kw)
    Y_final = model.fit_transform(ds.X)
    r = local_radius(Y_final, 10)
    outl = r / np.median(r) > 3
    print(f"{ds.name} {args.params}: {outl.mean():.1%} outliers at the end")
    if model.local_scaling:
        print("  note: forces below use raw ranks; local_scaling is not reproduced here")

    order, rank = neighbour_order(ds.X)
    p = scale_profile(ds.n, model.gamma, model.k0, model.width)
    rng = np.random.default_rng(args.seed)
    lab = ds.labels
    for t, Y in model.snapshots_:
        if t not in args.epochs:
            continue
        centre = np.array([Y[lab == c].mean(0) for c in np.unique(lab)])[np.searchsorted(np.unique(lab), lab)]
        u = Y - centre
        dist = np.linalg.norm(u, axis=1)
        u /= dist[:, None] + 1e-12
        hard = model.hard * (1 - (t - 1) / model.n_epochs) if model.hard_decay else model.hard
        terms = force_terms(Y, order, rank, p, model.per_point, hard, min(model.hard_k, ds.n - 1), rng)
        print(f"\nepoch {t}: distance to own cluster centre  outliers {dist[outl].mean():.1f}  others {dist[~outl].mean():.1f}")
        total = np.zeros(ds.n)
        for name, F in zip(TERMS, terms):
            rad = (F * u).sum(1)
            total += rad
            print(f"  {name:32s} outward: outliers {rad[outl].mean():+.4f}  others {rad[~outl].mean():+.4f}")
        print(f"  {'TOTAL':32s} outward: outliers {total[outl].mean():+.4f}  others {total[~outl].mean():+.4f}")


if __name__ == "__main__":
    main()
