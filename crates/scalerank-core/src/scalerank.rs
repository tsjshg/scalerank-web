//! ScaleRank optimiser on the N^2-free rank tables (port of drbench's
//! `ScaleRankFast`).
//!
//! Each group is an anchor i, a scale K ~ p(K) (K^-gamma, or a log-normal
//! band), one in-neighbour j of rank <= K and `n_neg` out-neighbours of rank
//! > K; the loss is -log(q_ij / (q_ij + w sum_m q_im)), q = 1 / (1 + d^2).
//!
//! * Negatives come from `n_cand` fresh uniform candidates per anchor and
//!   epoch (shared within blocks of `block` anchors), ranked via the table.
//! * In-neighbours of rank <= `r_pool` come from the table (kNN list and pool,
//!   weighted by the points they stand for), above it from the fresh
//!   candidates.  Pool landmarks get no gradient (`pool_stopgrad`), otherwise
//!   a few landmarks would be dragged by a large share of the data.

use crate::knn::{Knn, knn_exact};
use crate::par;
use crate::pca::pca;
use crate::points::{Points, dist};
use crate::ranks::RankTable;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Init {
    Pca,
    Random,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub gamma: f64,
    /// with `k0`, a log-normal band around k0 of log-width `width` instead of K^-gamma
    pub k0: Option<f64>,
    pub width: f64,
    pub n_components: usize,
    pub n_epochs: usize,
    pub per_point: usize,
    pub n_neg: usize,
    pub lr: f64,
    pub init: Init,
    pub init_scale: f64,
    pub kmax: usize,
    pub n_cand: usize,
    pub pool_ratio: f64,
    pub pool_per_level: usize,
    /// fresh candidates take over from the pool above the rank where
    /// ~fresh_min of them are expected per stratum of that size
    pub fresh_min: f64,
    pub pool_stopgrad: bool,
    /// Z = q_ij + w sum_m q_im with w = ((N - 1 - K) / Q)^neg_scale.  0 is the
    /// plain sampled softmax, whose repulsion weakens as N grows (with Q fixed
    /// it drifts from t-SNE-like towards UMAP-like, Damrich et al. 2023).
    pub neg_scale: f64,
    pub block: usize,
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            gamma: 1.0,
            k0: None,
            width: 1.0,
            n_components: 2,
            n_epochs: 800,
            per_point: 8,
            n_neg: 20,
            lr: 1.0,
            init: Init::Pca,
            init_scale: 1.0,
            kmax: 100,
            n_cand: 64,
            pool_ratio: 4.0,
            pool_per_level: 16,
            fresh_min: 4.0,
            pool_stopgrad: true,
            neg_scale: 0.0,
            block: 16,
            seed: 0,
        }
    }
}

/// p(K) for K = 1..n-2 (index 0 is K = 1).
pub fn scale_profile(n: usize, gamma: f64, k0: Option<f64>, width: f64) -> Vec<f64> {
    let p: Vec<f64> = (1..n - 1)
        .map(|k| {
            let k = k as f64;
            match k0 {
                None => k.powf(-gamma),
                Some(k0) => (-(k.ln() - k0.ln()).powi(2) / (2.0 * width * width)).exp() / k,
            }
        })
        .collect();
    let s: f64 = p.iter().sum();
    p.into_iter().map(|x| x / s).collect()
}

/// Progress report handed to the callback after every epoch.
pub struct Progress<'a> {
    pub epoch: usize,
    pub n_epochs: usize,
    pub loss: f64,
    pub embedding: &'a [f64],
}

pub struct Fit {
    /// n x n_components, row-major
    pub embedding: Vec<f64>,
    pub loss_history: Vec<f64>,
    /// mean number of rank-table entries per point
    pub table_len: f64,
}

const MAX_COMPONENTS: usize = 8;

/// Initial layout: PCA (or Gaussian) scaled so the first axis has standard
/// deviation `init_scale`.
pub fn initial_layout(p: &Points, params: &Params, rng: &mut Rng) -> Vec<f64> {
    let nc = params.n_components;
    let mut y = match params.init {
        Init::Pca => pca(p, nc, rng),
        Init::Random => (0..p.n * nc).map(|_| rng.normal()).collect(),
    };
    let n = p.n as f64;
    let mean = (0..p.n).map(|i| y[i * nc]).sum::<f64>() / n;
    let sd = ((0..p.n).map(|i| (y[i * nc] - mean).powi(2)).sum::<f64>() / n).sqrt();
    let s = params.init_scale / sd.max(1e-12);
    y.iter_mut().for_each(|v| *v *= s);
    y
}

/// Runs ScaleRank.  `on_epoch` is called after every epoch; returning false
/// stops early (the embedding so far is returned).
pub fn fit(p: &Points, params: &Params, on_epoch: impl FnMut(&Progress) -> bool) -> Fit {
    assert!(p.n >= 4, "need at least 4 points");
    let knn = knn_exact(p, None, params.kmax.min(p.n - 2));
    fit_with_knn(p, &knn, params, on_epoch)
}

/// [`fit`] with a precomputed kNN list (at least `min(kmax, n-2)` neighbours).
pub fn fit_with_knn(p: &Points, knn: &Knn, params: &Params, mut on_epoch: impl FnMut(&Progress) -> bool) -> Fit {
    let n = p.n;
    let nc = params.n_components;
    assert!(n >= 4, "need at least 4 points");
    assert!((1..=MAX_COMPONENTS).contains(&nc));
    let mut rng = Rng::new(params.seed);
    let kmax = params.kmax.min(n - 2);
    let profile = scale_profile(n, params.gamma, params.k0, params.width);
    let mut cdf = Vec::with_capacity(profile.len());
    let mut acc = 0.0;
    for v in &profile {
        acc += v;
        cdf.push(acc);
    }
    cdf.iter_mut().for_each(|c| *c /= acc);
    let r_pool = (kmax as f64).max(params.fresh_min * n as f64 / params.n_cand as f64);
    let table = RankTable::build(p, knn, kmax, r_pool, params.pool_ratio, params.pool_per_level, &mut rng);
    let mut y = initial_layout(p, params, &mut rng);

    let ctx = Ctx { p, table: &table, cdf: &cdf, params, kmax, r_pool: r_pool as f32 };
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8);
    let mut mom = vec![0f64; n * nc];
    let mut vel = vec![0f64; n * nc];
    let mut loss_history = Vec::with_capacity(params.n_epochs);
    let block = params.block.max(1);
    for t in 1..=params.n_epochs {
        let epoch_seed = rng.next_u64();
        let perm = rng.permutation(n);
        let blocks: Vec<&[u32]> = perm.chunks(block).collect();
        // gradient: per range of blocks, then summed in a fixed order
        let ranges = par::ranges(blocks.len());
        let parts = par::map_collect(ranges.len(), |c| {
            let mut g = vec![0f64; n * nc];
            let (mut loss, mut groups) = (0f64, 0usize);
            for b in ranges[c].clone() {
                let mut brng = Rng::stream(epoch_seed, b as u64);
                let (l, k) = ctx.block_grad(blocks[b], &y, &mut g, &mut brng);
                loss += l;
                groups += k;
            }
            (g, loss, groups)
        });
        let mut g = vec![0f64; n * nc];
        let (mut loss, mut groups) = (0f64, 0usize);
        for (pg, l, k) in parts {
            g.iter_mut().zip(pg).for_each(|(a, b)| *a += b);
            loss += l;
            groups += k;
        }
        let step = params.lr * (1.0 - (t - 1) as f64 / params.n_epochs as f64);
        let (c1, c2) = (1.0 - b1.powi(t as i32), 1.0 - b2.powi(t as i32));
        let inv = 1.0 / params.per_point as f64;
        for k in 0..n * nc {
            let gk = g[k] * inv;
            mom[k] = b1 * mom[k] + (1.0 - b1) * gk;
            vel[k] = b2 * vel[k] + (1.0 - b2) * gk * gk;
            y[k] -= step * (mom[k] / c1) / ((vel[k] / c2).sqrt() + eps);
        }
        let loss = loss / groups.max(1) as f64;
        loss_history.push(loss);
        if !on_epoch(&Progress { epoch: t, n_epochs: params.n_epochs, loss, embedding: &y }) {
            break;
        }
    }
    Fit { embedding: y, loss_history, table_len: table.mean_len() }
}

struct Ctx<'a> {
    p: &'a Points<'a>,
    table: &'a RankTable,
    cdf: &'a [f64],
    params: &'a Params,
    kmax: usize,
    r_pool: f32,
}

impl Ctx<'_> {
    /// Adds the gradient of all groups anchored in `anchors` to `g`; returns
    /// (summed loss, number of groups).
    fn block_grad(&self, anchors: &[u32], y: &[f64], g: &mut [f64], rng: &mut Rng) -> (f64, usize) {
        let (n, nc) = (self.p.n, self.params.n_components);
        let (l_cand, q_neg) = (self.params.n_cand, self.params.n_neg);
        let t = self.table;
        let fresh_w = (n - 1) as f64 / l_cand as f64;
        let cand: Vec<u32> = (0..l_cand).map(|_| rng.below(n) as u32).collect();
        let mut fresh: Vec<(f32, u32)> = Vec::with_capacity(l_cand);
        let mut neg: Vec<u32> = vec![0; q_neg];
        let mut qim: Vec<f64> = vec![0.0; q_neg];
        let (mut loss, mut groups) = (0f64, 0usize);
        for &a in anchors {
            let i = a as usize;
            let xi = self.p.row(i);
            // fresh candidates, ranked and sorted; self gets rank 0
            fresh.clear();
            fresh.extend(cand.iter().map(|&c| {
                let r = if c == a { 0.0 } else { t.rank_of(i, dist(xi, self.p.row(c as usize))) };
                (r, c)
            }));
            fresh.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
            let row = t.row(i);
            let (prank, pcw, pids) = (&t.rank[row.clone()], &t.cw[row.clone()], &t.ids[row]);
            let c_lo = fresh.partition_point(|f| f.0 <= self.r_pool);
            for _ in 0..self.params.per_point {
                let uk = rng.uniform();
                let k = (self.cdf.partition_point(|&c| c < uk) + 1) as f64;
                let kf = k as f32;
                // negatives: uniform over fresh candidates with rank > K
                let c_k = fresh.partition_point(|f| f.0 <= kf);
                // in-neighbour: pool entries of rank <= min(K, r_pool), weighted,
                // plus fresh candidates of rank in (r_pool, K]
                let c_pool = prank.partition_point(|&r| r <= kf.min(self.r_pool));
                let w_pool = if c_pool > 0 { pcw[c_pool - 1] as f64 } else { 0.0 };
                let n_fr = c_k.saturating_sub(c_lo);
                let w_fr = n_fr as f64 * fresh_w;
                let (u, u2) = (rng.uniform(), rng.uniform());
                if c_k >= l_cand || w_pool + w_fr <= 0.0 {
                    continue;
                }
                let u = u * (w_pool + w_fr);
                let (j, freeze_j) = if u < w_pool {
                    let kp = pcw.partition_point(|&c| c as f64 <= u).min(pcw.len() - 1);
                    let w_k = pcw[kp] - if kp > 0 { pcw[kp - 1] } else { 0.0 };
                    (pids[kp], self.params.pool_stopgrad && kp >= self.kmax && w_k > 1.0)
                } else {
                    let kf = (c_lo + (u2 * n_fr.max(1) as f64) as usize).min(l_cand - 1);
                    (fresh[kf].1, false)
                };
                for m in neg.iter_mut() {
                    *m = fresh[c_k + rng.below(l_cand - c_k)].1;
                }
                let w = if self.params.neg_scale != 0.0 {
                    ((n as f64 - 1.0 - k) / q_neg as f64).powf(self.params.neg_scale)
                } else {
                    1.0
                };
                loss += group_grad(y, g, nc, i, j as usize, freeze_j, &neg, w, &mut qim);
                groups += 1;
            }
        }
        (loss, groups)
    }
}

/// Loss -log(q_ij / (q_ij + w sum_m q_im)) of one group; adds its gradient to `g`.
#[allow(clippy::too_many_arguments)]
pub fn group_grad(
    y: &[f64],
    g: &mut [f64],
    nc: usize,
    i: usize,
    j: usize,
    freeze_j: bool,
    neg: &[u32],
    w: f64,
    qim: &mut [f64],
) -> f64 {
    let yi = &y[i * nc..(i + 1) * nc];
    let mut dij = [0f64; MAX_COMPONENTS];
    let mut s = 0.0;
    for c in 0..nc {
        dij[c] = yi[c] - y[j * nc + c];
        s += dij[c] * dij[c];
    }
    let qij = 1.0 / (1.0 + s);
    let mut z = 0.0;
    for (q, &m) in qim.iter_mut().zip(neg) {
        let m = m as usize;
        let s: f64 = (0..nc).map(|c| (yi[c] - y[m * nc + c]).powi(2)).sum();
        *q = 1.0 / (1.0 + s);
        z += *q;
    }
    let z = qij + w * z;
    // d loss / d y_i through d_ij: 2 (1 - q_ij / Z) q_ij (y_i - y_j)
    let cj = 2.0 * (1.0 - qij / z) * qij;
    let mut gi = [0f64; MAX_COMPONENTS];
    for c in 0..nc {
        gi[c] += cj * dij[c];
        if !freeze_j {
            g[j * nc + c] -= cj * dij[c];
        }
    }
    // through each d_im: -2 (w q_im / Z) q_im (y_i - y_m)
    for (&q, &m) in qim.iter().zip(neg) {
        let m = m as usize;
        let cm = -2.0 * w * q * q / z;
        for c in 0..nc {
            let d = cm * (yi[c] - y[m * nc + c]);
            gi[c] += d;
            g[m * nc + c] -= d;
        }
    }
    for c in 0..nc {
        g[i * nc + c] += gi[c];
    }
    z.ln() - qij.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_shapes() {
        let p = scale_profile(100, 1.0, None, 1.0);
        assert_eq!(p.len(), 98);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((p[0] / p[9] - 10.0).abs() < 1e-9);
        assert!(scale_profile(100, 0.0, None, 1.0).iter().all(|&x| (x - 1.0 / 98.0).abs() < 1e-12));
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let mut r = Rng::new(7);
        let (n, nc) = (12, 2);
        let y: Vec<f64> = (0..n * nc).map(|_| r.normal()).collect();
        let groups = [(0usize, 4usize, false, [8u32, 9, 1], 1.0), (1, 5, true, [2, 3, 11], 7.5), (2, 6, false, [0, 10, 4], 30.0)];
        let total = |y: &[f64], g: &mut [f64]| {
            let mut q = [0f64; 3];
            groups.iter().map(|&(i, j, f, neg, w)| group_grad(y, g, nc, i, j, f, &neg, w, &mut q)).sum::<f64>()
        };
        let mut g = vec![0f64; n * nc];
        total(&y, &mut g);
        let eps = 1e-6;
        for k in 0..n * nc {
            let (mut yp, mut ym) = (y.clone(), y.clone());
            yp[k] += eps;
            ym[k] -= eps;
            let mut dump = vec![0f64; n * nc];
            let num = (total(&yp, &mut dump) - total(&ym, &mut dump)) / (2.0 * eps);
            // frozen j (point 5) feels no pull from its group
            if k / nc == 5 {
                continue;
            }
            assert!((g[k] - num).abs() < 1e-6, "coordinate {k}: {} vs {num}", g[k]);
        }
    }
}
