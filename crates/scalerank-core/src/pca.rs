//! PCA by randomised subspace iteration (Halko et al. 2011) with a final
//! Rayleigh-Ritz step, so only O(n d q) work and no d x d matrix.

use crate::par;
use crate::points::Points;
use crate::rng::Rng;

/// Scores (n x k, row-major) on the top-k principal axes, by decreasing
/// variance.  Each column's sign is fixed so that its largest-magnitude score
/// is positive.  Axes beyond the data dimension are zero.
pub fn pca(p: &Points, k: usize, rng: &mut Rng) -> Vec<f64> {
    let (n, d) = (p.n, p.d);
    let q = (k + 10).min(d);
    let mut mean = vec![0f64; d];
    for i in 0..n {
        for (m, &v) in mean.iter_mut().zip(p.row(i)) {
            *m += v as f64;
        }
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);

    // omega: d x q, row-major
    let mut omega: Vec<f64> = (0..d * q).map(|_| rng.normal()).collect();
    orthonormalise(&mut omega, d, q);
    for _ in 0..7 {
        let a = project(p, &mean, &omega, q);
        omega = back_project(p, &mean, &a, q);
        orthonormalise(&mut omega, d, q);
    }
    let a = project(p, &mean, &omega, q);
    // Rayleigh-Ritz: eigenvectors of A^T A order and rotate the q directions
    let mut c = vec![0f64; q * q];
    for i in 0..n {
        let r = &a[i * q..(i + 1) * q];
        for s in 0..q {
            for t in 0..q {
                c[s * q + t] += r[s] * r[t];
            }
        }
    }
    let v = jacobi_eigenvectors(&mut c, q);
    let mut out = vec![0f64; n * k];
    for col in 0..k.min(q) {
        for i in 0..n {
            out[i * k + col] = (0..q).map(|s| a[i * q + s] * v[s * q + col]).sum();
        }
        let big = (0..n).map(|i| out[i * k + col]).fold(0f64, |m, x| if x.abs() > m.abs() { x } else { m });
        if big < 0.0 {
            (0..n).for_each(|i| out[i * k + col] = -out[i * k + col]);
        }
    }
    out
}

/// (X - mean) omega: n x q.
fn project(p: &Points, mean: &[f64], omega: &[f64], q: usize) -> Vec<f64> {
    let rows = par::map_collect(p.n, |i| {
        let mut r = vec![0f64; q];
        for (j, &x) in p.row(i).iter().enumerate() {
            let xc = x as f64 - mean[j];
            for (s, o) in r.iter_mut().zip(&omega[j * q..(j + 1) * q]) {
                *s += xc * o;
            }
        }
        r
    });
    rows.concat()
}

/// (X - mean)^T a: d x q.
fn back_project(p: &Points, mean: &[f64], a: &[f64], q: usize) -> Vec<f64> {
    let d = p.d;
    let parts = par::map_collect(par::ranges(p.n).len(), |c| {
        let range = par::ranges(p.n)[c].clone();
        let mut b = vec![0f64; d * q];
        for i in range {
            let ar = &a[i * q..(i + 1) * q];
            for (j, &x) in p.row(i).iter().enumerate() {
                let xc = x as f64 - mean[j];
                for (s, v) in b[j * q..(j + 1) * q].iter_mut().zip(ar) {
                    *s += xc * v;
                }
            }
        }
        b
    });
    let mut b = vec![0f64; d * q];
    for part in parts {
        b.iter_mut().zip(part).for_each(|(s, v)| *s += v);
    }
    b
}

/// Modified Gram-Schmidt on the q columns of a row-major d x q matrix.
fn orthonormalise(m: &mut [f64], d: usize, q: usize) {
    for c in 0..q {
        for prev in 0..c {
            let dot: f64 = (0..d).map(|j| m[j * q + c] * m[j * q + prev]).sum();
            (0..d).for_each(|j| m[j * q + c] -= dot * m[j * q + prev]);
        }
        let norm = (0..d).map(|j| m[j * q + c].powi(2)).sum::<f64>().sqrt();
        if norm > 1e-12 {
            (0..d).for_each(|j| m[j * q + c] /= norm);
        } else {
            (0..d).for_each(|j| m[j * q + c] = 0.0);
        }
    }
}

/// Eigenvectors of a symmetric q x q matrix (cyclic Jacobi), as the columns
/// of the returned row-major matrix, ordered by decreasing eigenvalue.
fn jacobi_eigenvectors(a: &mut [f64], q: usize) -> Vec<f64> {
    let mut v = vec![0f64; q * q];
    (0..q).for_each(|s| v[s * q + s] = 1.0);
    for _sweep in 0..100 {
        let off: f64 = (0..q).flat_map(|s| (0..q).filter(move |&t| t != s).map(move |t| (s, t))).map(|(s, t)| a[s * q + t].powi(2)).sum();
        let scale: f64 = (0..q).map(|s| a[s * q + s].powi(2)).sum();
        if off <= 1e-24 * scale.max(1e-300) {
            break;
        }
        for s in 0..q {
            for t in s + 1..q {
                let apq = a[s * q + t];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[t * q + t] - a[s * q + s]) / (2.0 * apq);
                let tt = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let tt = if theta == 0.0 { 1.0 } else { tt };
                let c = 1.0 / (tt * tt + 1.0).sqrt();
                let sn = tt * c;
                for k in 0..q {
                    let (aks, akt) = (a[k * q + s], a[k * q + t]);
                    a[k * q + s] = c * aks - sn * akt;
                    a[k * q + t] = sn * aks + c * akt;
                }
                for k in 0..q {
                    let (ask, atk) = (a[s * q + k], a[t * q + k]);
                    a[s * q + k] = c * ask - sn * atk;
                    a[t * q + k] = sn * ask + c * atk;
                }
                for k in 0..q {
                    let (vks, vkt) = (v[k * q + s], v[k * q + t]);
                    v[k * q + s] = c * vks - sn * vkt;
                    v[k * q + t] = sn * vks + c * vkt;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..q).collect();
    order.sort_by(|&x, &y| a[y * q + y].total_cmp(&a[x * q + x]));
    let mut out = vec![0f64; q * q];
    for (col, &src) in order.iter().enumerate() {
        (0..q).for_each(|r| out[r * q + col] = v[r * q + src]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_axes() {
        // variance 9, 4, 1, 0.01 along a rotated basis in 6-D
        let mut r = Rng::new(5);
        let (n, d) = (2000, 6);
        let sd = [3.0, 2.0, 1.0, 0.1, 0.1, 0.1];
        let mut basis: Vec<f64> = (0..d * d).map(|_| r.normal()).collect();
        orthonormalise(&mut basis, d, d);
        let mut x = vec![0f32; n * d];
        for i in 0..n {
            let z: Vec<f64> = sd.iter().map(|s| s * r.normal()).collect();
            for j in 0..d {
                x[i * d + j] = (5.0 + (0..d).map(|c| basis[j * d + c] * z[c]).sum::<f64>()) as f32;
            }
        }
        let p = Points::new(&x, n, d);
        let s = pca(&p, 3, &mut Rng::new(0));
        let var = |c: usize| (0..n).map(|i| s[i * 3 + c].powi(2)).sum::<f64>() / n as f64;
        assert!((var(0) / 9.0 - 1.0).abs() < 0.1);
        assert!((var(1) / 4.0 - 1.0).abs() < 0.1);
        assert!((var(2) - 1.0).abs() < 0.1);
        let cov01: f64 = (0..n).map(|i| s[i * 3] * s[i * 3 + 1]).sum::<f64>() / n as f64;
        assert!(cov01.abs() < 1e-6);
    }
}
