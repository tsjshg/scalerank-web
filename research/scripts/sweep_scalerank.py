"""Grid search over ScaleRank parameters on one or more datasets.

Each ``--grid`` item is ``name=v1,v2,...``; every combination is run and
scored with the headline metrics.  Results are printed and, with ``--out``,
written to CSV.  Not cached: this is for quick exploration, the benchmark
proper goes through ``drbench run``.

``final_loss`` is measured on the triplets the run itself samples, so it is
not comparable across runs with different ``hard`` (harder negatives give a
higher loss for a better embedding); use scripts/score_under_objective.py for
a common yardstick.

    python scripts/sweep_scalerank.py -d digits pollen --grid hard=0,0.3,0.6 gamma=1
    python scripts/sweep_scalerank.py -d digits --grid lr=0.3,1,3 refresh_every=1,5,20 --out sweep.csv
"""

import argparse
import ast
import itertools
import time

import numpy as np
import pandas as pd

from drbench import datasets
from drbench.metrics import distance_spearman, rank_metrics
from scalerank_research.scalerank import ScaleRank


def parse_grid(items):
    grid = {}
    for item in items:
        key, _, vals = item.partition("=")
        grid[key] = [ast.literal_eval(v) if v not in ("None",) else None for v in vals.split(",")]
    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--datasets", nargs="+", default=["digits"])
    ap.add_argument("--grid", nargs="+", default=["gamma=1.0"], help="name=v1,v2 ... (ScaleRank kwargs)")
    ap.add_argument("-s", "--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default=None, help="optional CSV path")
    args = ap.parse_args()

    grid = parse_grid(args.grid)
    keys = list(grid)
    rows = []
    for name in args.datasets:
        X = datasets.load(name).X
        for values in itertools.product(*grid.values()):
            params = dict(zip(keys, values))
            for seed in args.seeds:
                t0 = time.perf_counter()
                model = ScaleRank(random_state=seed, **params)
                Y = model.fit_transform(X)
                rm = rank_metrics(X, Y, ks=(10,))
                row = {"dataset": name, **params, "seed": seed,
                       "rnx@10": rm.r_nx[10], "auc_rnx": rm.auc_r_nx, "trust@10": rm.trustworthiness[10],
                       "dist_spearman": distance_spearman(X, Y),
                       "final_loss": float(np.mean(model.loss_history_[-20:])),
                       "time_s": time.perf_counter() - t0}
                rows.append(row)
                shown = " ".join(f"{k}={v}" for k, v in params.items())
                print(f"{name:10s} {shown:30s} s{seed}  rnx@10={row['rnx@10']:.3f} auc={row['auc_rnx']:.3f} "
                      f"sp={row['dist_spearman']:.3f} loss={row['final_loss']:.4f} {row['time_s']:.1f}s", flush=True)
    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
