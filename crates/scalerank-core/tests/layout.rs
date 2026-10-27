use scalerank_core::points::sqdist;
use scalerank_core::{Init, Params, Points, Rng, fit};

/// R_NX(k) by brute force: rescaled mean overlap of the k-NN sets.
fn rnx(x: &[f32], dx: usize, y: &[f64], dy: usize, n: usize, k: usize) -> f64 {
    let yf: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let nbrs = |data: &[f32], d: usize, i: usize| {
        let p = Points::new(data, n, d);
        let mut o: Vec<(f32, usize)> = (0..n).filter(|&j| j != i).map(|j| (sqdist(p.row(i), p.row(j)), j)).collect();
        o.sort_by(|a, b| a.0.total_cmp(&b.0));
        o[..k].iter().map(|a| a.1).collect::<Vec<_>>()
    };
    let mut overlap = 0usize;
    for i in 0..n {
        let a = nbrs(x, dx, i);
        let b = nbrs(&yf, dy, i);
        overlap += a.iter().filter(|j| b.contains(j)).count();
    }
    let q = overlap as f64 / (k * n) as f64;
    ((n - 1) as f64 * q - k as f64) / (n - 1 - k) as f64
}

#[test]
fn recovers_a_2d_layout() {
    let mut r = Rng::new(2);
    let n = 300;
    let x: Vec<f32> = (0..n * 2).map(|_| r.uniform() as f32).collect();
    for gamma in [1.0, 0.0] {
        let params = Params { gamma, init: Init::Random, n_epochs: 300, seed: 0, ..Params::default() };
        let f = fit(&Points::new(&x, n, 2), &params, |_| true);
        assert!(f.embedding.iter().all(|v| v.is_finite()));
        let q = rnx(&x, 2, &f.embedding, 2, n, 10);
        assert!(q > 0.7, "gamma {gamma}: R_NX@10 = {q:.3}");
    }
}

#[test]
fn callback_can_stop_early() {
    let mut r = Rng::new(3);
    let n = 80;
    let x: Vec<f32> = (0..n * 4).map(|_| r.normal() as f32).collect();
    let params = Params { n_epochs: 100, kmax: 20, ..Params::default() };
    let f = fit(&Points::new(&x, n, 4), &params, |pr| pr.epoch < 5);
    assert_eq!(f.loss_history.len(), 5);
}
