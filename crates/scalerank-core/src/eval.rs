//! Embedding quality, as in drbench's `metrics` module.
//!
//! Rank metrics follow Lee & Verleysen (2009): co-ranking based Q_NX / R_NX,
//! the log-K weighted AUC of R_NX (Lee et al. 2015), trustworthiness and
//! continuity.  Each is a mean over rows, so with `anchors` only those rows
//! are ranked (O(anchors x N)) and the result is an unbiased estimate.
//! Ranks break distance ties by index; self gets rank 0 and is left out.

use crate::knn::Knn;
use crate::par;
use crate::points::Points;
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct RankMetrics {
    pub n: usize,
    /// rows the estimates are based on (n when exact)
    pub n_anchors: usize,
    /// index K -> R_NX(K), K = 1..n-2 (index 0 unused)
    pub r_nx: Vec<f64>,
    pub auc_r_nx: f64,
    /// (K, trustworthiness, continuity)
    pub trust_cont: Vec<(usize, f64, f64)>,
    /// mean |rank_h - rank_l| / (n-1)^2 over pairs; about 1/3 for a random layout
    pub footrule: f64,
}

/// Ranks of all points as seen from row i (self 0, others 1..n-1).
fn row_ranks(dist: &mut Vec<(f64, u32)>, i: usize, ranks: &mut [u32], d_of: impl Fn(usize) -> f64) {
    let n = ranks.len();
    dist.clear();
    dist.extend((0..n).map(|j| (if j == i { f64::NEG_INFINITY } else { d_of(j) }, j as u32)));
    dist.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    for (r, &(_, j)) in dist.iter().enumerate() {
        ranks[j as usize] = r as u32;
    }
}

pub fn rank_metrics(x: &Points, y: &[f64], dy: usize, anchors: Option<&[u32]>, ks: &[usize]) -> RankMetrics {
    let n = x.n;
    assert_eq!(y.len(), n * dy);
    assert!(n >= 4);
    let all: Vec<u32>;
    let anchors = match anchors {
        Some(a) => a,
        None => {
            all = (0..n as u32).collect();
            &all
        }
    };
    let s = anchors.len();
    let ks: Vec<usize> = ks.iter().copied().filter(|&k| k >= 1 && 2 * k < n).collect();
    let ranges = par::ranges(s);
    // per part: histogram of max(rank_h, rank_l), trust/cont penalties, sum |rank_h - rank_l|
    let parts = par::map_collect(ranges.len(), |c| {
        let mut hist = vec![0u64; n];
        let mut trust = vec![0f64; ks.len()];
        let mut cont = vec![0f64; ks.len()];
        let mut abs_diff = 0f64;
        let (mut rh, mut rl) = (vec![0u32; n], vec![0u32; n]);
        let mut buf = Vec::with_capacity(n);
        for a in ranges[c].clone() {
            let i = anchors[a] as usize;
            let xi = x.row(i);
            row_ranks(&mut buf, i, &mut rh, |j| crate::points::sqdist(xi, x.row(j)) as f64);
            let yi = &y[i * dy..(i + 1) * dy];
            row_ranks(&mut buf, i, &mut rl, |j| (0..dy).map(|t| (yi[t] - y[j * dy + t]).powi(2)).sum());
            for j in 0..n {
                let (h, l) = (rh[j], rl[j]);
                hist[h.max(l) as usize] += 1;
                abs_diff += (h as f64 - l as f64).abs();
                if j == i {
                    continue;
                }
                for (t, &k) in ks.iter().enumerate() {
                    let (in_h, in_l) = (h as usize <= k, l as usize <= k);
                    if in_l && !in_h {
                        trust[t] += (h as usize - k) as f64; // intrusion
                    } else if in_h && !in_l {
                        cont[t] += (l as usize - k) as f64; // extrusion
                    }
                }
            }
        }
        (hist, trust, cont, abs_diff)
    });
    let mut hist = vec![0u64; n];
    let mut trust = vec![0f64; ks.len()];
    let mut cont = vec![0f64; ks.len()];
    let mut abs_diff = 0f64;
    for (h, t, c, a) in parts {
        hist.iter_mut().zip(h).for_each(|(x, y)| *x += y);
        trust.iter_mut().zip(t).for_each(|(x, y)| *x += y);
        cont.iter_mut().zip(c).for_each(|(x, y)| *x += y);
        abs_diff += a;
    }
    // Q_NX(K) = #{(i, j): max rank <= K} / (K s); the self pair has max rank 0
    let mut r_nx = vec![0f64; n - 1];
    let mut cum = 0u64;
    let (mut num, mut den) = (0f64, 0f64);
    for k in 1..n - 1 {
        cum += hist[k];
        let q = cum as f64 / (k * s) as f64;
        r_nx[k] = ((n - 1) as f64 * q - k as f64) / (n - 1 - k) as f64;
        num += r_nx[k] / k as f64;
        den += 1.0 / k as f64;
    }
    let norm = |k: usize| 2.0 / (s as f64 * k as f64 * (2 * n - 3 * k - 1) as f64);
    RankMetrics {
        n,
        n_anchors: s,
        r_nx,
        auc_r_nx: num / den,
        trust_cont: ks.iter().enumerate().map(|(t, &k)| (k, 1.0 - norm(k) * trust[t], 1.0 - norm(k) * cont[t])).collect(),
        footrule: abs_diff / (s as f64 * ((n - 1) as f64).powi(2)),
    }
}

/// `s` distinct random rows, sorted.
pub fn sample_anchors(n: usize, s: usize, seed: u64) -> Vec<u32> {
    let mut a = Rng::new(seed).sample(n, s.min(n));
    a.sort_unstable();
    a
}

/// Per point, the share of its k nearest neighbours in X that are also among
/// its k nearest in Y (both lists must hold at least k neighbours).
pub fn knn_overlap(knn_x: &Knn, knn_y: &Knn, k: usize) -> Vec<f64> {
    assert!(k <= knn_x.k && k <= knn_y.k);
    let n = knn_x.ids.len() / knn_x.k;
    (0..n)
        .map(|i| {
            let mut a = knn_x.ids(i)[..k].to_vec();
            a.sort_unstable();
            knn_y.ids(i)[..k].iter().filter(|j| a.binary_search(j).is_ok()).count() as f64 / k as f64
        })
        .collect()
}

/// Spearman correlation of pairwise distances (all pairs, or `max_pairs` random ones).
pub fn distance_spearman(x: &Points, y: &[f64], dy: usize, max_pairs: usize, seed: u64) -> f64 {
    let n = x.n;
    let mut pairs = Vec::new();
    if n * (n - 1) / 2 <= max_pairs {
        for i in 0..n {
            for j in i + 1..n {
                pairs.push((i, j));
            }
        }
    } else {
        let mut rng = Rng::new(seed);
        while pairs.len() < max_pairs {
            let (i, j) = (rng.below(n), rng.below(n));
            if i != j {
                pairs.push((i, j));
            }
        }
    }
    let dh: Vec<f64> = pairs.iter().map(|&(i, j)| crate::points::dist(x.row(i), x.row(j)) as f64).collect();
    let dl: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| (0..dy).map(|t| (y[i * dy + t] - y[j * dy + t]).powi(2)).sum::<f64>().sqrt())
        .collect();
    spearman(&dh, &dl)
}

/// Spearman correlation of log mean distances to the k nearest neighbours
/// (a local density proxy, as used to evaluate densMAP).
pub fn density_correlation(knn_x: &Knn, knn_y: &Knn, k: usize) -> f64 {
    let n = knn_x.ids.len() / knn_x.k;
    let radius = |knn: &Knn, i: usize| {
        (knn.dist(i)[..k].iter().map(|&d| d as f64).sum::<f64>() / k as f64 + 1e-12).ln()
    };
    let rx: Vec<f64> = (0..n).map(|i| radius(knn_x, i)).collect();
    let ry: Vec<f64> = (0..n).map(|i| radius(knn_y, i)).collect();
    spearman(&rx, &ry)
}

/// Leave-one-out k-NN classification accuracy (majority vote, ties to the
/// nearest neighbour's label) from a kNN list.
pub fn knn_accuracy(knn: &Knn, labels: &[u32], k: usize) -> f64 {
    let n = labels.len();
    let n_lab = labels.iter().max().map_or(0, |&m| m as usize + 1);
    let mut votes = vec![0usize; n_lab];
    let mut hit = 0;
    for i in 0..n {
        votes.iter_mut().for_each(|v| *v = 0);
        let ids = &knn.ids(i)[..k];
        for &j in ids {
            votes[labels[j as usize] as usize] += 1;
        }
        let best = *votes.iter().max().unwrap();
        // among tied labels, the one of the nearest neighbour wins
        let pred = ids.iter().map(|&j| labels[j as usize]).find(|&l| votes[l as usize] == best).unwrap();
        hit += (pred == labels[i]) as usize;
    }
    hit as f64 / n as f64
}

/// Average ranks (ties share their mean rank), 1-based.
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut o: Vec<usize> = (0..v.len()).collect();
    o.sort_unstable_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0f64; v.len()];
    let mut s = 0;
    while s < o.len() {
        let mut e = s + 1;
        while e < o.len() && v[o[e]] == v[o[s]] {
            e += 1;
        }
        let mean = (s + e + 1) as f64 / 2.0;
        o[s..e].iter().for_each(|&t| r[t] = mean);
        s = e;
    }
    r
}

pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in ra.iter().zip(&rb) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_embedding_is_perfect_and_anchors_agree() {
        let mut r = Rng::new(9);
        let n = 400;
        let x: Vec<f32> = (0..n * 2).map(|_| r.normal() as f32).collect();
        let y: Vec<f64> = x.iter().map(|&v| 3.0 * v as f64).collect();
        let p = Points::new(&x, n, 2);
        let m = rank_metrics(&p, &y, 2, None, &[5, 10]);
        assert!(m.r_nx[1..].iter().all(|&v| (v - 1.0).abs() < 1e-12));
        assert!(m.trust_cont.iter().all(|&(_, t, c)| t == 1.0 && c == 1.0));
        assert_eq!(m.footrule, 0.0);
        assert!((distance_spearman(&p, &y, 2, 1_000_000, 0) - 1.0).abs() < 1e-6);
        // a noisy embedding: anchors estimate the full metrics
        let y: Vec<f64> = x.iter().map(|&v| v as f64 + 0.5 * r.normal()).collect();
        let full = rank_metrics(&p, &y, 2, None, &[10]);
        let est = rank_metrics(&p, &y, 2, Some(&sample_anchors(n, 200, 1)), &[10]);
        assert!((full.r_nx[10] - est.r_nx[10]).abs() < 0.05);
        assert!((full.auc_r_nx - est.auc_r_nx).abs() < 0.03);
    }

    #[test]
    fn spearman_handles_ties() {
        assert!((spearman(&[1.0, 2.0, 2.0, 3.0], &[1.0, 2.0, 2.0, 3.0]) - 1.0).abs() < 1e-12);
        assert!((spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
    }
}
