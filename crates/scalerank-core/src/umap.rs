//! UMAP (McInnes et al. 2018), following umap-learn's defaults and update
//! rules: fuzzy simplicial set from the kNN graph (smooth_knn_dist,
//! fuzzy union), edges sampled in proportion to their weight, negative
//! sampling, clipped SGD with a linearly decaying learning rate.
//!
//! Differences from umap-learn: initialisation is PCA (or random), not
//! spectral, and the SGD runs serially with one random stream, so results
//! are reproducible but not identical to umap-learn's.

use crate::knn::{Knn, knn_exact};
use crate::pca::pca;
use crate::points::Points;
use crate::rng::Rng;
use crate::scalerank::{Init, Progress};

#[derive(Clone, Debug)]
pub struct UmapParams {
    pub n_neighbors: usize,
    pub min_dist: f64,
    pub spread: f64,
    pub n_components: usize,
    /// default: 500 up to 10,000 points, 200 above
    pub n_epochs: Option<usize>,
    pub learning_rate: f64,
    pub negative_sample_rate: usize,
    pub repulsion_strength: f64,
    pub init: Init,
    pub seed: u64,
}

impl Default for UmapParams {
    fn default() -> Self {
        UmapParams {
            n_neighbors: 15,
            min_dist: 0.1,
            spread: 1.0,
            n_components: 2,
            n_epochs: None,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
            init: Init::Pca,
            seed: 0,
        }
    }
}

pub struct UmapFit {
    /// n x n_components, row-major
    pub embedding: Vec<f64>,
    pub a: f64,
    pub b: f64,
    pub n_edges: usize,
}

/// Directed edges (head, tail, weight) of the symmetrised fuzzy graph, both
/// directions, sorted by (head, tail).
pub type Graph = Vec<(u32, u32, f64)>;

const SMOOTH_K_TOLERANCE: f64 = 1e-5;
const MIN_K_DIST_SCALE: f64 = 1e-3;

/// (sigma, rho) per point from sorted kNN distances whose column 0 is the
/// point itself (distance 0), as umap-learn's `smooth_knn_dist` with
/// local_connectivity = 1 and bandwidth = 1.
pub fn smooth_knn_dist(dist: &[f64], n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let target = (k as f64).log2();
    let mean_all = dist.iter().sum::<f64>() / dist.len() as f64;
    let mut sigma = vec![0f64; n];
    let mut rho = vec![0f64; n];
    for i in 0..n {
        let row = &dist[i * k..(i + 1) * k];
        rho[i] = row.iter().copied().find(|&d| d > 0.0).unwrap_or(0.0);
        let (mut lo, mut hi, mut mid) = (0f64, f64::INFINITY, 1f64);
        for _ in 0..64 {
            let psum: f64 = row[1..].iter().map(|&d| if d - rho[i] > 0.0 { (-(d - rho[i]) / mid).exp() } else { 1.0 }).sum();
            if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                break;
            }
            if psum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                mid = if hi.is_infinite() { mid * 2.0 } else { (lo + hi) / 2.0 };
            }
        }
        let floor = MIN_K_DIST_SCALE * if rho[i] > 0.0 { row.iter().sum::<f64>() / k as f64 } else { mean_all };
        sigma[i] = mid.max(floor);
    }
    (sigma, rho)
}

/// Fuzzy simplicial set of the `n_neighbors`-NN graph (self counted as the
/// first neighbour, as umap-learn does): w = a + b - ab over both directions.
pub fn fuzzy_graph(knn: &Knn, n: usize, n_neighbors: usize) -> Graph {
    let m = n_neighbors - 1;
    assert!(knn.k >= m, "kNN list shorter than n_neighbors - 1");
    let k = n_neighbors;
    let mut dist = vec![0f64; n * k];
    for i in 0..n {
        for c in 0..m {
            dist[i * k + c + 1] = knn.dist(i)[c] as f64;
        }
    }
    let (sigma, rho) = smooth_knn_dist(&dist, n, k);
    // (lo, hi, value of lo->hi, value of hi->lo)
    let mut half: Vec<(u32, u32, f64, f64)> = Vec::with_capacity(n * m);
    for i in 0..n {
        for c in 0..m {
            let j = knn.ids(i)[c];
            if j == u32::MAX {
                continue;
            }
            let d = dist[i * k + c + 1] - rho[i];
            let v = if d <= 0.0 || sigma[i] == 0.0 { 1.0 } else { (-d / sigma[i]).exp() };
            if v == 0.0 {
                continue;
            }
            let i = i as u32;
            half.push(if i < j { (i, j, v, 0.0) } else { (j, i, 0.0, v) });
        }
    }
    half.sort_unstable_by_key(|e| (e.0, e.1));
    let mut graph = Vec::with_capacity(2 * half.len());
    let mut s = 0;
    while s < half.len() {
        let (lo, hi) = (half[s].0, half[s].1);
        let (mut a, mut b) = (0.0, 0.0);
        while s < half.len() && half[s].0 == lo && half[s].1 == hi {
            a += half[s].2;
            b += half[s].3;
            s += 1;
        }
        let w = a + b - a * b;
        graph.push((lo, hi, w));
        graph.push((hi, lo, w));
    }
    graph.sort_unstable_by_key(|e| (e.0, e.1));
    graph
}

/// (a, b) of the low-dimensional similarity 1 / (1 + a d^(2b)), fitted by
/// least squares to 1 below min_dist and exp(-(d - min_dist) / spread)
/// above, on 300 points of [0, 3 spread] (umap-learn's `find_ab_params`).
pub fn find_ab_params(spread: f64, min_dist: f64) -> (f64, f64) {
    let xs: Vec<f64> = (0..300).map(|t| 3.0 * spread * t as f64 / 299.0).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| if x < min_dist { 1.0 } else { (-(x - min_dist) / spread).exp() }).collect();
    let sse = |a: f64, b: f64| xs.iter().zip(&ys).map(|(&x, &y)| (1.0 / (1.0 + a * x.powf(2.0 * b)) - y).powi(2)).sum::<f64>();
    // Levenberg-Marquardt on (a, b)
    let (mut a, mut b, mut lambda) = (1.0f64, 1.0f64, 1e-3);
    let mut cur = sse(a, b);
    for _ in 0..500 {
        let (mut jtj, mut jtr) = ([[0f64; 2]; 2], [0f64; 2]);
        for (&x, &y) in xs.iter().zip(&ys) {
            let u = if x > 0.0 { x.powf(2.0 * b) } else { 0.0 };
            let f = 1.0 / (1.0 + a * u);
            let da = -u * f * f;
            let db = if x > 0.0 { -a * u * 2.0 * x.ln() * f * f } else { 0.0 };
            let r = f - y;
            let g = [da, db];
            for s in 0..2 {
                jtr[s] += g[s] * r;
                for t in 0..2 {
                    jtj[s][t] += g[s] * g[t];
                }
            }
        }
        let m00 = jtj[0][0] * (1.0 + lambda);
        let m11 = jtj[1][1] * (1.0 + lambda);
        let det = m00 * m11 - jtj[0][1] * jtj[1][0];
        if det.abs() < 1e-300 {
            break;
        }
        let step_a = (m11 * jtr[0] - jtj[0][1] * jtr[1]) / det;
        let step_b = (m00 * jtr[1] - jtj[1][0] * jtr[0]) / det;
        let (na, nb) = (a - step_a, b - step_b);
        let new = if na > 0.0 && nb > 0.0 { sse(na, nb) } else { f64::INFINITY };
        if new < cur {
            let done = (cur - new) < 1e-15 * cur.max(1e-300);
            (a, b, cur) = (na, nb, new);
            lambda *= 0.3;
            if done {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }
    }
    (a, b)
}

pub fn fit(p: &Points, params: &UmapParams, on_epoch: impl FnMut(&Progress) -> bool) -> UmapFit {
    let knn = knn_exact(p, None, params.n_neighbors - 1);
    fit_with_knn(p, &knn, params, on_epoch)
}

/// UMAP from a precomputed kNN list (at least n_neighbors - 1 neighbours).
pub fn fit_with_knn(p: &Points, knn: &Knn, params: &UmapParams, mut on_epoch: impl FnMut(&Progress) -> bool) -> UmapFit {
    let n = p.n;
    let nc = params.n_components;
    assert!(n > params.n_neighbors && params.n_neighbors >= 2);
    let mut rng = Rng::new(params.seed);
    let mut graph = fuzzy_graph(knn, n, params.n_neighbors);
    let n_epochs = params.n_epochs.unwrap_or(if n <= 10_000 { 500 } else { 200 });
    let w_max = graph.iter().map(|e| e.2).fold(0f64, f64::max);
    graph.retain(|e| e.2 >= w_max / n_epochs as f64);
    let eps: Vec<f64> = graph.iter().map(|e| n_epochs as f64 / (n_epochs as f64 * e.2 / w_max)).collect();
    let eps_neg: Vec<f64> = eps.iter().map(|e| e / params.negative_sample_rate as f64).collect();
    let mut next = eps.clone();
    let mut next_neg = eps_neg.clone();
    let (a, b) = find_ab_params(params.spread, params.min_dist);
    let gamma = params.repulsion_strength;

    let mut y = initial_layout(p, params, &mut rng);
    let clip = |v: f64| v.clamp(-4.0, 4.0);
    let mut alpha = params.learning_rate;
    for epoch in 0..n_epochs {
        let ep = epoch as f64;
        for (e, &(head, tail, _)) in graph.iter().enumerate() {
            if next[e] > ep {
                continue;
            }
            let (j, k) = (head as usize, tail as usize);
            let d2: f64 = (0..nc).map(|c| (y[j * nc + c] - y[k * nc + c]).powi(2)).sum();
            let coeff = if d2 > 0.0 { -2.0 * a * b * d2.powf(b - 1.0) / (a * d2.powf(b) + 1.0) } else { 0.0 };
            for c in 0..nc {
                let g = clip(coeff * (y[j * nc + c] - y[k * nc + c]));
                y[j * nc + c] += g * alpha;
                y[k * nc + c] -= g * alpha;
            }
            next[e] += eps[e];
            let n_neg = ((ep - next_neg[e]) / eps_neg[e]) as i64;
            for _ in 0..n_neg.max(0) {
                let k = rng.below(n);
                let d2: f64 = (0..nc).map(|c| (y[j * nc + c] - y[k * nc + c]).powi(2)).sum();
                if d2 <= 0.0 {
                    continue;
                }
                let coeff = 2.0 * gamma * b / ((0.001 + d2) * (a * d2.powf(b) + 1.0));
                for c in 0..nc {
                    y[j * nc + c] += clip(coeff * (y[j * nc + c] - y[k * nc + c])) * alpha;
                }
            }
            next_neg[e] += n_neg.max(0) as f64 * eps_neg[e];
        }
        alpha = params.learning_rate * (1.0 - ep / n_epochs as f64);
        if !on_epoch(&Progress { epoch: epoch + 1, n_epochs, loss: f64::NAN, embedding: &y }) {
            break;
        }
    }
    UmapFit { embedding: y, a, b, n_edges: graph.len() }
}

/// PCA (or uniform in [-10, 10]) scaled to a largest coordinate of 10 with
/// N(0, 1e-4) noise, then each axis mapped to [0, 10], as umap-learn does.
fn initial_layout(p: &Points, params: &UmapParams, rng: &mut Rng) -> Vec<f64> {
    let (n, nc) = (p.n, params.n_components);
    let mut y = match params.init {
        Init::Pca => {
            let mut y = pca(p, nc, rng);
            let m = y.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-300);
            y.iter_mut().for_each(|v| *v = *v * 10.0 / m + 1e-4 * rng.normal());
            y
        }
        Init::Random => (0..n * nc).map(|_| 20.0 * rng.uniform() - 10.0).collect(),
    };
    for c in 0..nc {
        let (lo, hi) = (0..n).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), i| (lo.min(y[i * nc + c]), hi.max(y[i * nc + c])));
        let span = (hi - lo).max(1e-300);
        (0..n).for_each(|i| y[i * nc + c] = 10.0 * (y[i * nc + c] - lo) / span);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ab_params_match_umap_learn() {
        // umap-learn: find_ab_params(1.0, 0.1) = (1.5769..., 0.8950...)
        let (a, b) = find_ab_params(1.0, 0.1);
        assert!((a - 1.5769).abs() < 2e-3 && (b - 0.8951).abs() < 2e-3, "a {a} b {b}");
    }

    #[test]
    fn smooth_knn_dist_hits_target() {
        let (n, k) = (3, 6);
        let dist: Vec<f64> = (0..n).flat_map(|i| (0..k).map(move |c| if c == 0 { 0.0 } else { (i + c) as f64 * 0.7 })).collect();
        let (sigma, rho) = smooth_knn_dist(&dist, n, k);
        for i in 0..n {
            assert_eq!(rho[i], dist[i * k + 1]);
            let psum: f64 = dist[i * k + 1..(i + 1) * k].iter().map(|&d| (-(d - rho[i]).max(0.0) / sigma[i]).exp()).sum();
            assert!((psum - (k as f64).log2()).abs() < 1e-4);
        }
    }
}
