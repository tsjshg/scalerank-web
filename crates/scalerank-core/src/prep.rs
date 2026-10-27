//! Preprocessing of a raw table: missing values, standardisation, PCA.

use crate::pca::pca;
use crate::points::Points;
use crate::rng::Rng;

pub struct Prepared {
    /// row-major n x d
    pub data: Vec<f32>,
    pub n: usize,
    pub d: usize,
    /// columns dropped because they were constant
    pub dropped: usize,
    /// share of the variance kept by PCA (1 without PCA)
    pub explained: f64,
}

/// Fills missing values (NaN) with the column mean, drops constant columns,
/// optionally standardises each column, and reduces to `max_dim` principal
/// components when there are more columns than that.
pub fn prepare(raw: &[f32], n: usize, d: usize, standardize: bool, max_dim: usize, seed: u64) -> Prepared {
    assert_eq!(raw.len(), n * d);
    let mut mean = vec![0f64; d];
    let mut count = vec![0usize; d];
    for i in 0..n {
        for j in 0..d {
            let v = raw[i * d + j];
            if v.is_finite() {
                mean[j] += v as f64;
                count[j] += 1;
            }
        }
    }
    for j in 0..d {
        mean[j] = if count[j] > 0 { mean[j] / count[j] as f64 } else { 0.0 };
    }
    let mut var = vec![0f64; d];
    for i in 0..n {
        for j in 0..d {
            let v = raw[i * d + j];
            let v = if v.is_finite() { v as f64 } else { mean[j] };
            var[j] += (v - mean[j]).powi(2);
        }
    }
    let sd: Vec<f64> = var.iter().map(|v| (v / n as f64).sqrt()).collect();
    let keep: Vec<usize> = (0..d).filter(|&j| sd[j] > 1e-12 * (1.0 + mean[j].abs())).collect();
    let dk = keep.len().max(1);
    let mut data = vec![0f32; n * dk];
    for i in 0..n {
        for (c, &j) in keep.iter().enumerate() {
            let v = raw[i * d + j];
            let v = if v.is_finite() { v as f64 } else { mean[j] };
            let v = if standardize { (v - mean[j]) / sd[j] } else { v - mean[j] };
            data[i * dk + c] = v as f32;
        }
    }
    let dropped = d - keep.len();
    if dk <= max_dim {
        return Prepared { data, n, d: dk, dropped, explained: 1.0 };
    }
    let total: f64 = keep.iter().map(|&j| if standardize { 1.0 } else { sd[j] * sd[j] }).sum::<f64>() * n as f64;
    let scores = pca(&Points::new(&data, n, dk), max_dim, &mut Rng::new(seed));
    let kept: f64 = scores.iter().map(|v| v * v).sum();
    Prepared { data: scores.iter().map(|&v| v as f32).collect(), n, d: max_dim, dropped, explained: kept / total }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_standardises_and_reduces() {
        let mut r = Rng::new(0);
        let (n, d) = (200, 6);
        let mut raw: Vec<f32> = (0..n * d).map(|_| (10.0 + 5.0 * r.normal()) as f32).collect();
        raw[3] = f32::NAN;
        (0..n).for_each(|i| raw[i * d + 5] = 7.0); // constant column
        let p = prepare(&raw, n, d, true, 50, 0);
        assert_eq!((p.d, p.dropped), (5, 1));
        assert!(p.data.iter().all(|v| v.is_finite()));
        let m: f64 = (0..n).map(|i| p.data[i * 5] as f64).sum::<f64>() / n as f64;
        let v: f64 = (0..n).map(|i| (p.data[i * 5] as f64).powi(2)).sum::<f64>() / n as f64;
        assert!(m.abs() < 1e-5 && (v - 1.0).abs() < 1e-4);
        let q = prepare(&raw, n, d, true, 3, 0);
        assert_eq!(q.d, 3);
        assert!(q.explained > 0.5 && q.explained < 1.0);
    }
}
