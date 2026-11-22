"""ScaleRank without the N x N rank tables (the plan for the browser app).

Same objective and loss as :mod:`.scalerank` (softmax over one in-neighbour
j of rank <= K and ``n_neg`` out-neighbours of rank > K, K ~ p), but every
rank comes from O(N log N)-sized structures:

* **kNN list** (``kmax`` neighbours): exact ranks 1..kmax.
* **Multi-scale pool**: for rank strata (B, cB], B = kmax, c kmax, ...,
  a random subsample with rate f = m / ((c-1) B) and each point's nearest
  subsample members.  The a-th of them has rank ~ a N / (n_s + 1), and the
  members falling in the stratum are a uniform sample of it, each standing
  for 1/f points.  Together with the kNN list this gives every point a
  distance -> rank table (``prof_d``, ``prof_r``) up to rank N, and a fixed
  set of in-neighbour candidates.
* **Fresh candidates**: every epoch, ``n_cand`` uniform random points per
  anchor (shared within blocks of anchors so distances are one matmul),
  ranked through the table.  Negatives are drawn from those with rank > K,
  i.e. uniformly from outside the top-K.  In-neighbours of rank above
  ``r_pool`` (where the pool stops) come from them too, so far
  in-neighbours are not always the same few landmarks.

Pool members other than kNN neighbours are shared landmarks that many
points pull on; with ``pool_stopgrad`` the anchor moves towards them but
they do not feel the pull (otherwise a landmark would be dragged by a
large fraction of the data).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .scalerank import scale_profile, softmax_loss_and_grad


def _row_search(table: np.ndarray, rows: np.ndarray, vals: np.ndarray, side: str = "left") -> np.ndarray:
    """searchsorted of vals[k] in the sorted row table[rows[k]], vectorised.

    Rows are laid end to end with offsets so one np.searchsorted does all.
    Returns positions in 0..P.
    """
    n, P = table.shape
    lo, hi = float(table.min()), float(table.max())
    # row k occupies (k span, (k+1) span); queries are clipped half a unit
    # beyond the table's range so they never spill into a neighbouring row
    span = hi - lo + 2.0
    flat = (table - lo + 1.0 + span * np.arange(n)[:, None]).ravel()
    q = np.clip(vals, lo - 0.5, hi + 0.5) - lo + 1.0 + span * rows
    return np.searchsorted(flat, q, side=side) - rows * P


def _monotone(r: np.ndarray) -> np.ndarray:
    """Cheap non-decreasing fit along axis 1: mean of running max and reverse running min."""
    up = np.maximum.accumulate(r, axis=1)
    down = np.minimum.accumulate(r[:, ::-1], axis=1)[:, ::-1]
    return 0.5 * (up + down)


def rank_structures(X: np.ndarray, kmax: int, r_pool: float, ratio: float, per_level: int, rng: np.random.Generator):
    """kNN list + multi-scale pool, merged into per-point tables sorted by distance.

    Returns (ids, dist, rank, weight, pool_ok): (n, P) arrays.  ``weight`` is
    how many points an entry stands for (1 for kNN neighbours, 1/f for pool
    members, 0 for padding); ``pool_ok`` marks entries usable as
    in-neighbour candidates (rank <= r_pool).
    """
    n = X.shape[0]
    dist, ids = NearestNeighbors(n_neighbors=kmax + 1, algorithm="brute", n_jobs=-1).fit(X).kneighbors(X)
    dist, ids = dist[:, 1:], ids[:, 1:]  # drop self (column 0)
    cols_i = [ids]
    cols_d = [dist]
    cols_r = [np.broadcast_to(np.arange(1, kmax + 1, dtype=np.float64), (n, kmax))]
    cols_w = [np.ones((n, kmax))]
    lo = float(kmax)
    while lo < n - 1:
        hi = min(ratio * lo, n - 1.0)
        f = per_level / ((ratio - 1) * lo)
        n_s = int(min(n, max(2, round(f * n))))
        sub = rng.choice(n, n_s, replace=False)
        a_max = min(n_s, int(np.ceil(hi * (n_s + 1) / n)) + 1)
        d_s, j_s = NearestNeighbors(n_neighbors=a_max, algorithm="brute", n_jobs=-1).fit(X[sub]).kneighbors(X)
        j_s = sub[j_s]
        is_self = j_s == np.arange(n)[:, None]
        # a = 1-based order among the subsample members other than i itself
        a = np.cumsum(~is_self, axis=1).astype(np.float64)
        r = a * n / (n_s + 1)
        keep = (~is_self) & (r > lo) & (r <= hi) & (d_s > dist[:, -1:])  # d <= d_kmax: already a kNN entry
        w = np.where(keep, n / (n_s + 1), 0.0)
        cols_i.append(np.where(keep, j_s, 0))
        cols_d.append(np.where(keep, d_s, np.inf))
        cols_r.append(np.where(keep, r, np.inf))
        cols_w.append(w)
        lo = hi
    ids = np.concatenate(cols_i, axis=1)
    d = np.concatenate(cols_d, axis=1)
    r = np.concatenate(cols_r, axis=1)
    w = np.concatenate(cols_w, axis=1)
    o = np.argsort(d, axis=1, kind="stable")
    ids, d, r, w = (np.take_along_axis(a, o, axis=1) for a in (ids, d, r, w))
    # padding (inf) sits at the end of each row; give it the last real values so
    # rows stay sorted and finite, with zero weight.
    pad = ~np.isfinite(d)
    last = (~pad).sum(axis=1) - 1
    rows = np.arange(n)
    d = np.where(pad, d[rows, last][:, None], d)
    r = np.where(pad, r[rows, last][:, None], r)
    r[:, kmax:] = np.maximum(_monotone(r)[:, kmax:], kmax)  # noisy pool ranks -> non-decreasing in distance
    r = np.maximum.accumulate(r, axis=1)
    # Last entry: rank n-1 at (an estimate of) the farthest distance, so that
    # candidates beyond the last pool member are ranked up to n-1 instead of
    # all sharing its rank.  The farthest point from anyone lies on the convex
    # hull; points extreme along random directions stand in for it.
    proj = X @ rng.normal(size=(X.shape[1], 32))
    ext = np.unique(np.r_[proj.argmax(axis=0), proj.argmin(axis=0)])
    far = np.sqrt(((X[:, None, :] - X[ext][None, :, :]) ** 2).sum(axis=2)).max(axis=1)
    far = np.maximum(far, d[:, -1] * (1 + 1e-9) + 1e-12)
    ids = np.c_[ids, np.zeros(n, dtype=ids.dtype)]
    d = np.c_[d, far]
    r = np.c_[r, np.full(n, n - 1.0)]
    w = np.c_[w, np.zeros(n)]
    pool_ok = (w > 0) & (r <= r_pool)
    return ids, d, r, w, pool_ok


def rank_from_dist(prof_d, prof_r, rows, d, n):
    """Rank estimate of a point at distance d from rows[k], by interpolating the table."""
    P = prof_d.shape[1]
    pos = _row_search(prof_d, rows, d, "left")
    k1 = np.clip(pos, 1, P - 1)
    d0, d1 = prof_d[rows, k1 - 1], prof_d[rows, k1]
    r0, r1 = prof_r[rows, k1 - 1], prof_r[rows, k1]
    t = (d - d0) / np.maximum(d1 - d0, 1e-12)
    # beyond the last entry keep the last segment's slope, capped at n - 1
    t = np.where(pos >= P, t, np.clip(t, 0.0, 1.0))
    r = r0 + t * (r1 - r0)
    r = np.where(pos == 0, prof_r[rows, 0], r)
    return np.clip(r, 1.0, n - 1.0)


class ScaleRankFast:
    def __init__(
        self,
        gamma: float = 1.0,
        k0: float | None = None,
        width: float = 1.0,
        n_components: int = 2,
        n_epochs: int = 800,
        per_point: int = 8,
        n_neg: int = 20,
        lr: float = 1.0,
        init: str = "pca",
        init_scale: float = 1.0,
        kmax: int = 100,
        n_cand: int = 64,
        pool_ratio: float = 4.0,
        pool_per_level: int = 16,
        fresh_min: float = 4.0,
        pool_stopgrad: bool = True,
        neg_scale: float = 0.0,
        block: int = 16,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.gamma, self.k0, self.width = gamma, k0, width
        self.n_components = n_components
        self.n_epochs, self.per_point, self.n_neg, self.lr = n_epochs, per_point, n_neg, lr
        self.init, self.init_scale = init, init_scale
        self.kmax, self.n_cand = kmax, n_cand
        self.pool_ratio, self.pool_per_level = pool_ratio, pool_per_level
        # fresh candidates take over from the pool above the rank where
        # ~fresh_min of them are expected per stratum of that size
        self.fresh_min = fresh_min
        self.pool_stopgrad = pool_stopgrad
        # Z = q_ij + w sum_m q_im with w = ((N - 1 - K) / Q) ** neg_scale: 0 is
        # the plain sampled softmax, whose repulsion weakens as N grows (with
        # Q fixed it drifts from t-SNE-like towards UMAP-like, Damrich et al.
        # 2023); 1 makes the sum an estimate over the whole complement.
        self.neg_scale = neg_scale
        self.block = block
        self.random_state = random_state
        self.verbose = verbose
        self.loss_history_: list[float] = []

    def _init(self, X, rng):
        if self.init == "pca":
            Y = PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(X)
        else:
            Y = rng.normal(size=(X.shape[0], self.n_components))
        return Y / Y[:, 0].std() * self.init_scale

    def _fresh(self, X, sqn, rng):
        """n_cand random candidates per anchor, shared within random blocks of anchors."""
        n, L, B = X.shape[0], self.n_cand, self.block
        perm = rng.permutation(n)
        nb = -(-n // B)
        anchors = np.r_[perm, perm[: nb * B - n]].reshape(nb, B)
        cand = rng.integers(0, n, size=(nb, L))
        g = np.einsum("bkd,bld->bkl", X[anchors], X[cand])
        d2 = sqn[anchors][:, :, None] + sqn[cand][:, None, :] - 2 * g
        ids = np.empty((n, L), dtype=np.int64)
        d = np.empty((n, L))
        a = anchors.ravel()
        ids[a] = np.repeat(cand, B, axis=0)  # duplicate anchors (padding) just overwrite
        d[a] = np.sqrt(np.maximum(d2, 0.0)).reshape(-1, L)
        return ids, d

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        kmax = min(self.kmax, n - 2)
        p = scale_profile(n, self.gamma, self.k0, self.width)
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        r_pool = max(float(kmax), self.fresh_min * n / self.n_cand)
        ids, prof_d, prof_r, w, pool_ok = rank_structures(X, kmax, r_pool, self.pool_ratio, self.pool_per_level, rng)
        cw = np.cumsum(np.where(pool_ok, w, 0.0), axis=1)
        is_landmark = w > 1
        self.n_pool_ = int(pool_ok.sum(axis=1).mean())
        sqn = (X**2).sum(axis=1)
        L, Q = self.n_cand, self.n_neg
        fresh_w = (n - 1) / L
        Y = self._init(X, rng)

        b1, b2, eps = 0.9, 0.999, 1e-8
        mom = np.zeros_like(Y)
        vel = np.zeros_like(Y)
        self.loss_history_ = []
        i_all = np.repeat(np.arange(n), self.per_point)
        G_n = i_all.size
        for t in range(1, self.n_epochs + 1):
            # fresh candidates, ranked and sorted per anchor
            f_ids, f_d = self._fresh(X, sqn, rng)
            f_r = rank_from_dist(prof_d, prof_r, np.repeat(np.arange(n), L), f_d.ravel(), n).reshape(n, L)
            f_r[f_ids == np.arange(n)[:, None]] = 0.0  # self: never a negative or a fresh positive
            o = np.argsort(f_r, axis=1)
            f_ids = np.take_along_axis(f_ids, o, axis=1)
            f_r = np.take_along_axis(f_r, o, axis=1)

            i = i_all
            K = (np.searchsorted(cdf, rng.random(G_n)) + 1).astype(np.float64)  # K = 1..n-2
            # negatives: uniform over fresh candidates with rank > K
            c_K = _row_search(f_r, i, K, "right")
            ok = c_K < L
            # in-neighbour: pool entries of rank <= min(K, r_pool), weighted,
            # plus fresh candidates of rank in (r_pool, K]
            c_pool = _row_search(prof_r, i, np.minimum(K, r_pool), "right")
            w_pool = cw[i, np.maximum(c_pool - 1, 0)] * (c_pool > 0)
            c_lo = _row_search(f_r, i, np.full(G_n, r_pool), "right")
            n_fr = np.maximum(c_K - c_lo, 0)
            w_fr = n_fr * fresh_w
            ok &= (w_pool + w_fr) > 0
            i, K, c_K, c_pool, w_pool, c_lo, n_fr, w_fr = (
                a[ok] for a in (i, K, c_K, c_pool, w_pool, c_lo, n_fr, w_fr)
            )
            u = rng.random(i.size) * (w_pool + w_fr)
            from_pool = u < w_pool
            k_pool = np.minimum(_row_search(cw, i, u, "right"), cw.shape[1] - 1)
            k_fr = np.minimum(c_lo + (rng.random(i.size) * np.maximum(n_fr, 1)).astype(np.int64), L - 1)
            j = np.where(from_pool, ids[i, k_pool], f_ids[i, k_fr])
            freeze_j = from_pool & is_landmark[i, k_pool] if self.pool_stopgrad else None
            pos = c_K[:, None] + (rng.random((i.size, Q)) * (L - c_K)[:, None]).astype(np.int64)
            M = f_ids[i[:, None], pos]

            neg_w = ((n - 1 - K) / Q) ** self.neg_scale if self.neg_scale else None
            loss, Gr = softmax_loss_and_grad(Y, i, j, M, freeze_j=freeze_j, neg_w=neg_w)
            Gr /= self.per_point
            step = self.lr * (1 - (t - 1) / self.n_epochs)
            mom = b1 * mom + (1 - b1) * Gr
            vel = b2 * vel + (1 - b2) * Gr**2
            Y -= step * (mom / (1 - b1**t)) / (np.sqrt(vel / (1 - b2**t)) + eps)
            self.loss_history_.append(loss)
            if self.verbose and (t % 50 == 0 or t == 1):
                print(f"epoch {t:4d}  loss {loss:.4f}  groups {i.size}/{G_n}")
        self.embedding_ = Y
        return Y
