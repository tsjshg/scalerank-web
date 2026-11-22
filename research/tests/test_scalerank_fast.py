import numpy as np
import pytest

from scalerank_research.scalerank import neighbour_order
from scalerank_research.scalerank_fast import ScaleRankFast, _row_search, rank_from_dist, rank_structures


def test_row_search_matches_searchsorted_per_row():
    rng = np.random.default_rng(0)
    table = np.sort(rng.integers(1, 20, size=(30, 12)).astype(float), axis=1)
    rows = rng.integers(0, 30, 500)
    # include values below every row's minimum and above its maximum
    vals = rng.uniform(-5, 25, 500)
    for side in ("left", "right"):
        got = _row_search(table, rows, vals, side)
        want = np.array([np.searchsorted(table[r], v, side) for r, v in zip(rows, vals)])
        assert (got == want).all()


def test_rank_estimates():
    X = np.random.default_rng(1).normal(size=(1500, 10))
    n = len(X)
    _, rank = neighbour_order(X)
    rng = np.random.default_rng(0)
    ids, d, r, w, ok = rank_structures(X, kmax=50, r_pool=200, ratio=4.0, per_level=16, rng=rng)
    assert (np.diff(d, axis=1) >= 0).all() and (np.diff(r, axis=1) >= 0).all()
    assert (r[:, :50] == np.arange(1, 51)).all()  # exact kNN ranks first
    assert (r[:, -1] == n - 1).all()
    assert (r[ok] <= 200).all() and (w[ok] > 0).all()
    # points inside the kNN range get their exact rank back
    a = np.repeat(np.arange(n), 50)
    assert np.allclose(rank_from_dist(d, r, a, d[:, :50].ravel(), n), np.tile(np.arange(1, 51), n))
    # random pairs: estimate within ~25% of the true rank for most pairs
    i, j = rng.integers(0, n, (2, 5000))
    keep = i != j
    i, j = i[keep], j[keep]
    est = rank_from_dist(d, r, i, np.linalg.norm(X[i] - X[j], axis=1), n)
    assert np.median(np.abs(np.log(est / rank[i, j]))) < 0.25


@pytest.mark.parametrize("gamma", [1.0, 0.0])
def test_recovers_a_2d_layout(gamma):
    from drbench.metrics import rank_metrics

    X = np.random.default_rng(2).uniform(size=(300, 2))
    Y = ScaleRankFast(gamma=gamma, init="random", n_epochs=300, random_state=0).fit_transform(X)
    assert Y.shape == (300, 2) and np.isfinite(Y).all()
    assert rank_metrics(X, Y).r_nx[10] > 0.7
