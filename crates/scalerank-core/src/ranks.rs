//! Per-point distance -> rank tables without N x N storage.
//!
//! * kNN list (`kmax` neighbours): exact ranks 1..kmax.
//! * Multi-scale pool: for rank strata (B, cB], B = kmax, c kmax, ..., a
//!   random subsample with rate f = m / ((c-1) B) and each point's nearest
//!   subsample members.  The a-th of them has rank ~ a N / (n_s + 1), and the
//!   members falling in the stratum are a uniform sample of it, each standing
//!   for 1/f points.
//! * A last knot at rank N-1 and (an estimate of) the farthest distance, so
//!   points beyond the last pool member are ranked up to N-1.
//!
//! Pool entries of rank <= `r_pool` double as in-neighbour candidates, picked
//! in proportion to the number of points they stand for (`cw`, cumulative).

use crate::knn::{Knn, knn_exact};
use crate::par;
use crate::points::{Points, dist};
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct RankTable {
    pub n: usize,
    pub kmax: usize,
    pub r_pool: f32,
    offsets: Vec<usize>,
    /// entries of each row, sorted by distance; the first `kmax` are the kNN list
    pub ids: Vec<u32>,
    pub dist: Vec<f32>,
    pub rank: Vec<f32>,
    /// cumulative weight of the in-neighbour candidates (rank <= r_pool) along the row
    pub cw: Vec<f32>,
}

struct Level {
    lo: f64,
    hi: f64,
    n_s: usize,
    knn: Knn,
}

impl RankTable {
    /// `knn` must hold at least `kmax` exact (or nearly exact) neighbours per point.
    pub fn build(p: &Points, knn: &Knn, kmax: usize, r_pool: f64, ratio: f64, per_level: usize, rng: &mut Rng) -> Self {
        let n = p.n;
        assert!(n >= 3 && kmax >= 1 && kmax <= n - 2, "need 1 <= kmax <= n - 2");
        assert!(knn.k >= kmax, "kNN list shorter than kmax");
        let mut levels = Vec::new();
        let mut lo = kmax as f64;
        let n1 = (n - 1) as f64;
        while lo < n1 {
            let hi = (ratio * lo).min(n1);
            let f = per_level as f64 / ((ratio - 1.0) * lo);
            let n_s = ((f * n as f64).round() as usize).clamp(2, n);
            let sub = rng.sample(n, n_s);
            let a_max = n_s.min((hi * (n_s + 1) as f64 / n as f64).ceil() as usize + 1);
            levels.push(Level { lo, hi, n_s, knn: knn_exact(p, Some(&sub), a_max) });
            lo = hi;
        }
        let far = farthest(p, rng);

        let rows = par::map_collect(n, |i| row_entries(i, n, knn, kmax, &levels, far[i], r_pool));
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        let total: usize = rows.iter().map(|r| r.0.len()).sum();
        let (mut ids, mut dist, mut rank, mut cw) =
            (Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total));
        for (ri, rd, rr, rc) in rows {
            ids.extend(ri);
            dist.extend(rd);
            rank.extend(rr);
            cw.extend(rc);
            offsets.push(ids.len());
        }
        RankTable { n, kmax, r_pool: r_pool as f32, offsets, ids, dist, rank, cw }
    }

    #[inline]
    pub fn row(&self, i: usize) -> std::ops::Range<usize> {
        self.offsets[i]..self.offsets[i + 1]
    }

    /// Mean number of entries per row.
    pub fn mean_len(&self) -> f64 {
        self.ids.len() as f64 / self.n as f64
    }

    /// Rank estimate of a point at distance `d` from point `i`: linear
    /// interpolation in the table, extrapolated past its end, clipped to
    /// [1, n-1].
    pub fn rank_of(&self, i: usize, d: f32) -> f32 {
        let r = self.row(i);
        let (pd, pr) = (&self.dist[r.clone()], &self.rank[r]);
        let len = pd.len();
        let pos = pd.partition_point(|&x| x < d);
        let est = if pos == 0 {
            pr[0]
        } else {
            let k1 = pos.min(len - 1);
            let (d0, d1, r0, r1) = (pd[k1 - 1], pd[k1], pr[k1 - 1], pr[k1]);
            let t = (d - d0) / (d1 - d0).max(1e-12);
            let t = if pos >= len { t } else { t.clamp(0.0, 1.0) };
            r0 + t * (r1 - r0)
        };
        est.clamp(1.0, (self.n - 1) as f32)
    }
}

type Row = (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn row_entries(i: usize, n: usize, knn: &Knn, kmax: usize, levels: &[Level], far: f32, r_pool: f64) -> Row {
    let d_kmax = knn.dist(i)[kmax - 1];
    // (dist, rank, weight, id); kNN first, then levels in order (stable sort keeps ties so)
    let mut e: Vec<(f32, f64, f64, u32)> =
        (0..kmax).map(|c| (knn.dist(i)[c], (c + 1) as f64, 1.0, knn.ids(i)[c])).collect();
    for lv in levels {
        let w = n as f64 / (lv.n_s + 1) as f64;
        for (a, (&j, &d)) in lv.knn.ids(i).iter().zip(lv.knn.dist(i)).enumerate() {
            if j == u32::MAX {
                break;
            }
            let r = (a + 1) as f64 * w;
            if r > lv.lo && r <= lv.hi && d > d_kmax {
                e.push((d, r, w, j));
            }
        }
    }
    e.sort_by(|a, b| a.0.total_cmp(&b.0));
    let len = e.len();
    // noisy pool ranks -> non-decreasing in distance: mean of running max and reverse running min
    let mut up = vec![0f64; len];
    let mut down = vec![0f64; len];
    let mut m = f64::NEG_INFINITY;
    for k in 0..len {
        m = m.max(e[k].1);
        up[k] = m;
    }
    let mut m = f64::INFINITY;
    for k in (0..len).rev() {
        m = m.min(e[k].1);
        down[k] = m;
    }
    let mut rank: Vec<f32> = Vec::with_capacity(len + 1);
    let mut run = f64::NEG_INFINITY;
    for k in 0..len {
        let r = if k < kmax { e[k].1 } else { (0.5 * (up[k] + down[k])).max(kmax as f64) };
        run = run.max(r);
        rank.push(run as f32);
    }
    let mut ids: Vec<u32> = e.iter().map(|x| x.3).collect();
    let mut dist: Vec<f32> = e.iter().map(|x| x.0).collect();
    let mut cw = Vec::with_capacity(len + 1);
    let mut acc = 0f64;
    for k in 0..len {
        if rank[k] as f64 <= r_pool {
            acc += e[k].2;
        }
        cw.push(acc as f32);
    }
    ids.push(0);
    dist.push(far.max(dist[len - 1].next_up()));
    rank.push((n - 1) as f32);
    cw.push(acc as f32);
    (ids, dist, rank, cw)
}

/// Distance from each point to the farthest of the points that are extreme
/// along 32 random directions (the farthest point from anyone lies on the
/// convex hull; these stand in for it).
fn farthest(p: &Points, rng: &mut Rng) -> Vec<f32> {
    let dirs: Vec<Vec<f64>> = (0..32).map(|_| (0..p.d).map(|_| rng.normal()).collect()).collect();
    let mut ext = Vec::new();
    for g in &dirs {
        let proj = |i: usize| p.row(i).iter().zip(g).map(|(&x, &w)| x as f64 * w).sum::<f64>();
        let (mut amin, mut amax) = (0, 0);
        let (mut vmin, mut vmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..p.n {
            let v = proj(i);
            if v < vmin {
                (vmin, amin) = (v, i);
            }
            if v > vmax {
                (vmax, amax) = (v, i);
            }
        }
        ext.push(amin);
        ext.push(amax);
    }
    ext.sort_unstable();
    ext.dedup();
    par::map_collect(p.n, |i| ext.iter().map(|&e| dist(p.row(i), p.row(e))).fold(0f32, f32::max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::points::sqdist;

    #[test]
    fn rank_estimates_track_true_ranks() {
        let mut r = Rng::new(11);
        let (n, d) = (1500, 5);
        let x: Vec<f32> = (0..n * d).map(|_| r.normal() as f32).collect();
        let p = Points::new(&x, n, d);
        let t = RankTable::build(&p, &knn_exact(&p, None, 40), 30, 200.0, 4.0, 16, &mut r);
        let mut err = Vec::new();
        for i in (0..n).step_by(37) {
            let mut ds: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| sqdist(p.row(i), p.row(j)).sqrt()).collect();
            ds.sort_by(f32::total_cmp);
            for (k, &dk) in ds.iter().enumerate() {
                let est = t.rank_of(i, dk) as f64;
                let truth = (k + 1) as f64;
                if k < 30 {
                    assert!((est - truth).abs() < 1e-3, "kNN ranks are exact");
                }
                err.push(((est.max(1.0)).ln() - truth.ln()).abs());
            }
        }
        err.sort_by(f64::total_cmp);
        let med = err[err.len() / 2];
        let p90 = err[err.len() * 9 / 10];
        assert!(med < 0.15 && p90 < 0.4, "log-rank error median {med:.3} p90 {p90:.3}");
        // tables end at rank n-1, and in-neighbour weight stops at r_pool
        for i in [0, 700] {
            let r = t.row(i);
            assert_eq!(t.rank[r.end - 1], (n - 1) as f32);
            assert!(t.rank[r.clone()].windows(2).all(|w| w[0] <= w[1]));
            assert!(t.dist[r].windows(2).all(|w| w[0] <= w[1]));
        }
    }
}
