import numpy as np
import pytest
from scipy.spatial.distance import cdist

from scalerank_research.scalerank import (
    ScaleRank,
    neighbour_order,
    sample_triplets,
    scale_profile,
    triplet_loss_and_grad,
)


def test_scale_profile():
    p = scale_profile(100, gamma=1.0)
    assert p.shape == (98,) and p.sum() == pytest.approx(1.0)
    assert np.allclose(p[0] / p[9], 10.0)  # 1/K weighting
    assert np.allclose(scale_profile(100, gamma=0.0), 1 / 98)
    band = scale_profile(1000, k0=30, width=0.5)
    assert 15 < np.argmax(band * np.arange(1, 999)) + 1 < 45  # mass in log K peaks near k0


def test_triplets_respect_ranks():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    order, rank = neighbour_order(X)
    for a in range(80):
        assert (rank[a, order[a]] == np.arange(1, 80)).all()
    p = scale_profile(80, gamma=0.5)
    low_nbrs = np.argsort(cdist(X[:, :2], X[:, :2]), axis=1)[:, 1:31]  # any low-dim layout
    for hard in (0.0, 1.0):
        i, j, m, w, is_hard = sample_triplets(order, p, 50, rng, rank, low_nbrs, hard, hard_c=1.0)
        assert is_hard.any() == (hard > 0)
        assert ((w > 0) & (w <= 1)).all()
        assert (j != i).all() and (m != i).all()
        assert (rank[i, j] < rank[i, m]).all()


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(12, 2))
    i, j, m = rng.integers(0, 12, (3, 40))
    keep = (i != j) & (i != m) & (j != m)
    i, j, m = i[keep], j[keep], m[keep]
    _, G = triplet_loss_and_grad(Y, i, j, m)
    eps = 1e-6
    num = np.zeros_like(Y)
    for a in range(12):
        for c in range(2):
            Yp, Ym = Y.copy(), Y.copy()
            Yp[a, c] += eps
            Ym[a, c] -= eps
            num[a, c] = (triplet_loss_and_grad(Yp, i, j, m)[0] - triplet_loss_and_grad(Ym, i, j, m)[0]) / (2 * eps)
    assert np.allclose(G / len(i), num, atol=1e-6)


@pytest.mark.parametrize(
    "kw, floor",
    [
        # Default softmax loss: on intrinsically 2-D data it levels off near
        # R_NX@10 = 0.80 (also with 800 epochs or PCA init); see the
        # s_curve regression in the research notes.
        (dict(), 0.75),
        # Triplet configuration of the first prototype keeps its earlier level.
        (dict(n_neg=1, per_point=32, hard=0.3), 0.85),
    ],
)
def test_recovers_a_2d_layout(kw, floor):
    from drbench.metrics import rank_metrics

    X = np.random.default_rng(2).uniform(size=(300, 2))
    Y = ScaleRank(gamma=1.0, init="random", n_epochs=300, random_state=0, **kw).fit_transform(X)
    assert rank_metrics(X, Y).r_nx[10] > floor


def test_local_scaling_rank_tables_are_permutations():
    X = np.random.default_rng(3).normal(size=(60, 20))
    order, rank = neighbour_order(X, local_scaling=5)
    for a in range(60):
        assert sorted(order[a]) == sorted(set(range(60)) - {a})
        assert (rank[a, order[a]] == np.arange(1, 60)).all()


def test_frozen_out_neighbours_receive_no_gradient():
    rng = np.random.default_rng(4)
    Y = rng.normal(size=(10, 2))
    i, j, m = np.array([0, 1]), np.array([2, 3]), np.array([4, 5])
    _, G = triplet_loss_and_grad(Y, i, j, m, freeze_m=np.array([True, False]))
    assert np.allclose(G[4], 0) and not np.allclose(G[5], 0)


@pytest.mark.parametrize(
    "kw",
    [
        dict(local_scaling=10, local_scaling_kmax=30),
        dict(n_neg=1, hard=0.3, hard_one_sided=True, hard_c=1.0),
        dict(n_neg=5, hard=0.3, hard_one_sided=True),
    ],
)
def test_options_run(kw):
    X = np.random.default_rng(5).normal(size=(150, 8))
    Y = ScaleRank(n_epochs=20, random_state=0, **kw).fit_transform(X)
    assert Y.shape == (150, 2) and np.isfinite(Y).all()


def test_softmax_with_one_negative_equals_triplet():
    from scalerank_research.scalerank import softmax_loss_and_grad

    rng = np.random.default_rng(6)
    Y = rng.normal(size=(15, 2))
    i, j, m = np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11])
    l1, G1 = triplet_loss_and_grad(Y, i, j, m)
    l2, G2 = softmax_loss_and_grad(Y, i, j, m[:, None])
    assert l1 == pytest.approx(l2)
    assert np.allclose(G1, G2)


def test_softmax_gradient_matches_finite_differences():
    from scalerank_research.scalerank import softmax_loss_and_grad

    rng = np.random.default_rng(7)
    Y = rng.normal(size=(12, 2))
    i = np.array([0, 1, 2, 3, 0])
    j = np.array([4, 5, 6, 7, 8])
    M = rng.integers(0, 12, size=(5, 6))
    _, G = softmax_loss_and_grad(Y, i, j, M)
    eps = 1e-6
    num = np.zeros_like(Y)
    for a in range(12):
        for c in range(2):
            Yp, Ym = Y.copy(), Y.copy()
            Yp[a, c] += eps
            Ym[a, c] -= eps
            num[a, c] = (softmax_loss_and_grad(Yp, i, j, M)[0] - softmax_loss_and_grad(Ym, i, j, M)[0]) / (2 * eps)
    assert np.allclose(G / len(i), num, atol=1e-6)


def test_groups_respect_ranks():
    from scalerank_research.scalerank import sample_groups

    rng = np.random.default_rng(8)
    X = rng.normal(size=(70, 5))
    order, rank = neighbour_order(X)
    low_nbrs = np.argsort(cdist(X[:, :2], X[:, :2]), axis=1)[:, 1:31]
    for hard in (0.0, 0.5):
        i, j, M, is_hard = sample_groups(order, scale_profile(70, 1.0), 20, rng, 8, rank, low_nbrs, hard)
        assert M.shape == (70 * 20, 8)
        assert (rank[i[:, None], M] > rank[i, j][:, None]).all()
        assert (M != i[:, None]).all()


def test_hard_c_requires_triplets():
    with pytest.raises(ValueError):
        ScaleRank(n_neg=20, hard_c=1.0)


def test_weighted_negatives_gradient_matches_finite_differences():
    from scalerank_research.scalerank import softmax_loss_and_grad

    rng = np.random.default_rng(9)
    Y = rng.normal(size=(12, 2))
    i = np.array([0, 1, 2, 3, 0])
    j = np.array([4, 5, 6, 7, 8])
    M = rng.integers(0, 12, size=(5, 6))
    w = rng.uniform(1, 50, 5)
    _, G = softmax_loss_and_grad(Y, i, j, M, neg_w=w)
    eps = 1e-6
    num = np.zeros_like(Y)
    for a in range(12):
        for c in range(2):
            Yp, Ym = Y.copy(), Y.copy()
            Yp[a, c] += eps
            Ym[a, c] -= eps
            num[a, c] = (softmax_loss_and_grad(Yp, i, j, M, neg_w=w)[0] - softmax_loss_and_grad(Ym, i, j, M, neg_w=w)[0]) / (2 * eps)
    assert np.allclose(G / len(i), num, atol=1e-6)
