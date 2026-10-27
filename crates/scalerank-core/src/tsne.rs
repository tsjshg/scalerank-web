//! t-SNE with Barnes-Hut repulsion (van der Maaten 2014), following
//! openTSNE's defaults and update rule: perplexity-calibrated Gaussian
//! affinities on 3 x perplexity neighbours, symmetrised and normalised;
//! 250 iterations of early exaggeration 12, then 500 at 1; learning rate
//! n / exaggeration; momentum 0.8 with delta-bar-delta gains; per-point
//! step norm capped at 5; the embedding is re-centred after each step.
//!
//! Only 2-D output (the quadtree).  openTSNE switches to FFT-accelerated
//! interpolation above 10,000 points; that is not implemented here yet.

use crate::knn::{Knn, knn_exact};
use crate::par;
use crate::pca::pca;
use crate::points::Points;
use crate::rng::Rng;
use crate::scalerank::{Init, Progress};

#[derive(Clone, Debug)]
pub struct TsneParams {
    pub perplexity: f64,
    pub early_exaggeration_iter: usize,
    pub early_exaggeration: f64,
    pub n_iter: usize,
    pub exaggeration: f64,
    /// default n / exaggeration in each phase
    pub learning_rate: Option<f64>,
    pub momentum: f64,
    pub max_step_norm: f64,
    pub theta: f64,
    pub init: Init,
    pub seed: u64,
}

impl Default for TsneParams {
    fn default() -> Self {
        TsneParams {
            perplexity: 30.0,
            early_exaggeration_iter: 250,
            early_exaggeration: 12.0,
            n_iter: 500,
            exaggeration: 1.0,
            learning_rate: None,
            momentum: 0.8,
            max_step_norm: 5.0,
            theta: 0.5,
            init: Init::Pca,
            seed: 0,
        }
    }
}

/// Neighbours needed for a perplexity: min(n - 1, floor(3 perplexity)).
pub fn n_neighbors(n: usize, perplexity: f64) -> usize {
    ((3.0 * perplexity) as usize).min(n - 1).max(1)
}

/// Symmetric affinities in CSR form, summing to 1.
pub struct Affinities {
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub data: Vec<f64>,
}

/// Conditional Gaussian affinities on the first `k` neighbours, with the
/// precision per point found by bisection so the entropy is ln(perplexity),
/// then P = (P + P^T) / 2 normalised to sum 1 (openTSNE's
/// `joint_probabilities_nn`).
pub fn affinities(knn: &Knn, n: usize, k: usize, perplexity: f64) -> Affinities {
    assert!(knn.k >= k);
    let target = perplexity.ln();
    let rows = par::map_collect(n, |i| {
        let d2: Vec<f64> = knn.dist(i)[..k].iter().map(|&d| (d as f64).powi(2)).collect();
        let (mut beta, mut lo, mut hi) = (1.0f64, f64::NEG_INFINITY, f64::INFINITY);
        let mut p = vec![0f64; k];
        for _ in 0..200 {
            let mut sum = 0.0;
            for (pj, &d) in p.iter_mut().zip(&d2) {
                *pj = (-d * beta).exp();
                sum += *pj;
            }
            let sum = sum.max(f64::MIN_POSITIVE);
            let mut sum_dp = 0.0;
            for (pj, &d) in p.iter_mut().zip(&d2) {
                *pj /= sum;
                sum_dp += d * *pj;
            }
            let diff = sum_dp * beta + sum.ln() - target;
            if diff.abs() <= 1e-8 {
                break;
            }
            if diff > 0.0 {
                lo = beta;
                beta = if hi.is_infinite() { beta * 2.0 } else { (beta + hi) / 2.0 };
            } else {
                hi = beta;
                beta = if lo.is_infinite() { beta / 2.0 } else { (beta + lo) / 2.0 };
            }
        }
        p
    });
    let mut trip: Vec<(u32, u32, f64)> = Vec::with_capacity(2 * n * k);
    for (i, p) in rows.iter().enumerate() {
        for (c, &pij) in p.iter().enumerate() {
            let j = knn.ids(i)[c];
            if j == u32::MAX {
                continue;
            }
            trip.push((i as u32, j, pij / 2.0));
            trip.push((j, i as u32, pij / 2.0));
        }
    }
    trip.sort_unstable_by_key(|t| (t.0, t.1));
    let mut indptr = vec![0usize; n + 1];
    let mut indices = Vec::with_capacity(trip.len());
    let mut data: Vec<f64> = Vec::with_capacity(trip.len());
    let mut s = 0;
    while s < trip.len() {
        let (i, j) = (trip[s].0, trip[s].1);
        let mut v = 0.0;
        while s < trip.len() && trip[s].0 == i && trip[s].1 == j {
            v += trip[s].2;
            s += 1;
        }
        indices.push(j);
        data.push(v);
        indptr[i as usize + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let total: f64 = data.iter().sum();
    data.iter_mut().for_each(|v| *v /= total);
    Affinities { indptr, indices, data }
}

/// Quadtree node: cell centre and half-width, centre of mass, point count,
/// index of the first of four children (0 for a leaf).
#[derive(Clone, Copy)]
struct Node {
    cx: f64,
    cy: f64,
    hw: f64,
    mx: f64,
    my: f64,
    count: u32,
    child: u32,
}

struct QuadTree {
    nodes: Vec<Node>,
}

impl QuadTree {
    fn new(y: &[f64]) -> Self {
        let n = y.len() / 2;
        let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..n {
            x0 = x0.min(y[2 * i]);
            x1 = x1.max(y[2 * i]);
            y0 = y0.min(y[2 * i + 1]);
            y1 = y1.max(y[2 * i + 1]);
        }
        let hw = 0.5 * (x1 - x0).max(y1 - y0) * (1.0 + 1e-9) + 1e-12;
        let root = Node { cx: 0.5 * (x0 + x1), cy: 0.5 * (y0 + y1), hw, mx: 0.0, my: 0.0, count: 0, child: 0 };
        let mut t = QuadTree { nodes: vec![root] };
        let mut idx: Vec<u32> = (0..n as u32).collect();
        t.build(0, &mut idx, y, 0);
        t
    }

    fn build(&mut self, at: usize, idx: &mut [u32], y: &[f64], depth: usize) {
        let (sx, sy) = idx.iter().fold((0.0, 0.0), |(a, b), &i| (a + y[2 * i as usize], b + y[2 * i as usize + 1]));
        let c = idx.len() as f64;
        let node = &mut self.nodes[at];
        (node.mx, node.my, node.count) = (sx / c, sy / c, idx.len() as u32);
        if idx.len() <= 1 || depth >= 48 {
            return;
        }
        let (cx, cy, hw) = (node.cx, node.cy, node.hw);
        // partition into quadrants: 0 = (-,-), 1 = (+,-), 2 = (-,+), 3 = (+,+)
        let quad = |i: u32| (y[2 * i as usize] >= cx) as usize + 2 * (y[2 * i as usize + 1] >= cy) as usize;
        idx.sort_unstable_by_key(|&i| quad(i));
        let first = self.nodes.len() as u32;
        self.nodes[at].child = first;
        let h = 0.5 * hw;
        for q in 0..4 {
            let ccx = cx + if q & 1 == 1 { h } else { -h };
            let ccy = cy + if q & 2 == 2 { h } else { -h };
            self.nodes.push(Node { cx: ccx, cy: ccy, hw: h, mx: 0.0, my: 0.0, count: 0, child: 0 });
        }
        let mut start = 0;
        for q in 0..4 {
            let end = start + idx[start..].iter().take_while(|&&i| quad(i) == q).count();
            if end > start {
                self.build(first as usize + q, &mut idx[start..end], y, depth + 1);
            }
            start = end;
        }
    }

    /// (sum_j w_ij^2 (y_i - y_j), sum_j w_ij) over all other points, w = 1 / (1 + d^2).
    fn repulsion(&self, px: f64, py: f64, theta: f64) -> (f64, f64, f64) {
        let (mut gx, mut gy, mut sum_q) = (0.0, 0.0, 0.0);
        let mut stack = vec![0u32];
        while let Some(at) = stack.pop() {
            let nd = &self.nodes[at as usize];
            if nd.count == 0 {
                continue;
            }
            let (dx, dy) = (px - nd.mx, py - nd.my);
            let d2 = dx * dx + dy * dy;
            if nd.child == 0 || (2.0 * nd.hw) * (2.0 * nd.hw) < theta * theta * d2 {
                if d2 < 1e-24 {
                    continue; // the point itself (or exact duplicates)
                }
                let w = 1.0 / (1.0 + d2);
                let c = nd.count as f64;
                sum_q += c * w;
                gx += c * w * w * dx;
                gy += c * w * w * dy;
            } else {
                (0..4).for_each(|q| stack.push(nd.child + q));
            }
        }
        (gx, gy, sum_q)
    }
}

/// KL gradient (without openTSNE's absent factor 4): sum_j (p_ij - q_ij) w_ij (y_i - y_j).
fn gradient(y: &[f64], p: &Affinities, exaggeration: f64, theta: f64) -> Vec<f64> {
    let n = y.len() / 2;
    let tree = QuadTree::new(y);
    let rows = par::map_collect(n, |i| {
        let (yx, yy) = (y[2 * i], y[2 * i + 1]);
        let (mut ax, mut ay) = (0.0, 0.0);
        for s in p.indptr[i]..p.indptr[i + 1] {
            let j = p.indices[s] as usize;
            let (dx, dy) = (yx - y[2 * j], yy - y[2 * j + 1]);
            let w = 1.0 / (1.0 + dx * dx + dy * dy);
            ax += exaggeration * p.data[s] * w * dx;
            ay += exaggeration * p.data[s] * w * dy;
        }
        let (rx, ry, sq) = tree.repulsion(yx, yy, theta);
        (ax, ay, rx, ry, sq)
    });
    let z: f64 = rows.iter().map(|r| r.4).sum();
    rows.iter().flat_map(|r| [r.0 - r.2 / z, r.1 - r.3 / z]).collect()
}

pub fn fit(p: &Points, params: &TsneParams, on_iter: impl FnMut(&Progress) -> bool) -> Vec<f64> {
    let knn = knn_exact(p, None, n_neighbors(p.n, params.perplexity));
    fit_with_knn(p, &knn, params, on_iter)
}

/// t-SNE from a precomputed kNN list (at least `n_neighbors(n, perplexity)` neighbours).
pub fn fit_with_knn(p: &Points, knn: &Knn, params: &TsneParams, mut on_iter: impl FnMut(&Progress) -> bool) -> Vec<f64> {
    let n = p.n;
    assert!(n >= 4);
    let mut rng = Rng::new(params.seed);
    let aff = affinities(knn, n, n_neighbors(n, params.perplexity), params.perplexity);
    let mut y = match params.init {
        Init::Pca => pca(p, 2, &mut rng),
        Init::Random => (0..2 * n).map(|_| rng.normal()).collect(),
    };
    // openTSNE: first axis rescaled to standard deviation 1e-4, plus 1% jitter
    let sd = ((0..n).map(|i| y[2 * i].powi(2)).sum::<f64>() / n as f64 - ((0..n).map(|i| y[2 * i]).sum::<f64>() / n as f64).powi(2)).sqrt();
    let s = 1e-4 / sd.max(1e-300);
    y.iter_mut().for_each(|v| *v = *v * s + 1e-6 * rng.normal());

    let mut update = vec![0f64; 2 * n];
    let mut gains = vec![1f64; 2 * n];
    let total = params.early_exaggeration_iter + params.n_iter;
    let phases = [(params.early_exaggeration_iter, params.early_exaggeration), (params.n_iter, params.exaggeration)];
    let mut it = 0;
    for (iters, ex) in phases {
        let lr = params.learning_rate.unwrap_or(n as f64 / ex);
        for _ in 0..iters {
            let g = gradient(&y, &aff, ex, params.theta);
            for k in 0..2 * n {
                // delta-bar-delta: sign(update) != sign(gradient) counts as flipped (also at 0)
                if update[k].signum() != g[k].signum() || update[k] == 0.0 {
                    gains[k] += 0.2;
                } else {
                    gains[k] = gains[k] * 0.8 + 0.01;
                }
                update[k] = params.momentum * update[k] - lr * gains[k] * g[k];
            }
            for i in 0..n {
                let norm = (update[2 * i].powi(2) + update[2 * i + 1].powi(2)).sqrt();
                if norm > params.max_step_norm {
                    update[2 * i] *= params.max_step_norm / norm;
                    update[2 * i + 1] *= params.max_step_norm / norm;
                }
            }
            let (mut mx, mut my) = (0.0, 0.0);
            for i in 0..n {
                y[2 * i] += update[2 * i];
                y[2 * i + 1] += update[2 * i + 1];
                mx += y[2 * i];
                my += y[2 * i + 1];
            }
            let (mx, my) = (mx / n as f64, my / n as f64);
            for i in 0..n {
                y[2 * i] -= mx;
                y[2 * i + 1] -= my;
            }
            it += 1;
            if !on_iter(&Progress { epoch: it, n_epochs: total, loss: f64::NAN, embedding: &y }) {
                return y;
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affinities_match_perplexity_and_sum_to_one() {
        let mut r = Rng::new(1);
        let (n, d) = (300, 5);
        let x: Vec<f32> = (0..n * d).map(|_| r.normal() as f32).collect();
        let p = Points::new(&x, n, d);
        let k = n_neighbors(n, 10.0);
        let knn = knn_exact(&p, None, k);
        let a = affinities(&knn, n, k, 10.0);
        assert!((a.data.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // symmetric
        for i in [0usize, 17, 299] {
            for s in a.indptr[i]..a.indptr[i + 1] {
                let j = a.indices[s] as usize;
                let back = (a.indptr[j]..a.indptr[j + 1]).find(|&t| a.indices[t] as usize == i).unwrap();
                assert!((a.data[back] - a.data[s]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn barnes_hut_matches_exact_repulsion() {
        let mut r = Rng::new(2);
        let n = 500;
        let y: Vec<f64> = (0..2 * n).map(|_| 3.0 * r.normal()).collect();
        let tree = QuadTree::new(&y);
        for i in [0usize, 250, 499] {
            let (mut gx, mut gy, mut sq) = (0.0, 0.0, 0.0);
            for j in (0..n).filter(|&j| j != i) {
                let (dx, dy) = (y[2 * i] - y[2 * j], y[2 * i + 1] - y[2 * j + 1]);
                let w = 1.0 / (1.0 + dx * dx + dy * dy);
                sq += w;
                gx += w * w * dx;
                gy += w * w * dy;
            }
            let exact = tree.repulsion(y[2 * i], y[2 * i + 1], 0.0);
            assert!((exact.2 - sq).abs() < 1e-9 * sq && (exact.0 - gx).abs() < 1e-9 && (exact.1 - gy).abs() < 1e-9);
            let bh = tree.repulsion(y[2 * i], y[2 * i + 1], 0.5);
            assert!((bh.2 - sq).abs() < 0.02 * sq, "sum_q {} vs {sq}", bh.2);
        }
    }
}
