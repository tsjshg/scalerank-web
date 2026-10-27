//! WebAssembly bindings of scalerank-core.  A `Session` holds one prepared
//! table and its kNN list, shared by ScaleRank, UMAP and t-SNE.  Runs report
//! progress through a JS callback `(step, total, embedding) => boolean`;
//! returning false stops the run early.

use js_sys::{Float32Array, Function, Object, Reflect};
use scalerank_core::eval;
use scalerank_core::knn::{Knn, knn_exact};
use scalerank_core::prep::prepare;
use scalerank_core::scalerank::{self, Params};
use scalerank_core::tsne::{self, TsneParams};
use scalerank_core::umap::{self, UmapParams};
use scalerank_core::{Points, Progress};
use wasm_bindgen::prelude::*;

/// Neighbours kept for all methods: ScaleRank's kmax (100) covers t-SNE's
/// 3 x perplexity and UMAP's n_neighbors - 1 at their defaults.
const K_SHARED: usize = 100;

#[wasm_bindgen]
pub struct Session {
    data: Vec<f32>,
    n: usize,
    d: usize,
    dropped: usize,
    explained: f64,
    knn: Option<Knn>,
}

fn set(o: &Object, key: &str, v: impl Into<JsValue>) {
    Reflect::set(o, &JsValue::from_str(key), &v.into()).expect("set property");
}

/// Calls `cb(step, total, embedding)` every `every` steps and at the end.
fn reporter<'a>(cb: &'a Function, every: usize) -> impl FnMut(&Progress) -> bool + 'a {
    let every = every.max(1);
    move |pr: &Progress| {
        if !pr.epoch.is_multiple_of(every) && pr.epoch != pr.n_epochs {
            return true;
        }
        let y: Vec<f32> = pr.embedding.iter().map(|&v| v as f32).collect();
        let r = cb.call3(&JsValue::NULL, &JsValue::from(pr.epoch as u32), &JsValue::from(pr.n_epochs as u32), &Float32Array::from(&y[..]));
        !matches!(r, Ok(v) if v == JsValue::FALSE)
    }
}

#[wasm_bindgen]
impl Session {
    /// `raw` is a row-major n x d table (NaN for missing values).
    #[wasm_bindgen(constructor)]
    pub fn new(raw: Vec<f32>, n: usize, d: usize, standardize: bool, max_dim: usize) -> Result<Session, JsError> {
        if n < 10 || d == 0 || raw.len() != n * d {
            return Err(JsError::new("need at least 10 rows and 1 numeric column"));
        }
        let p = prepare(&raw, n, d, standardize, max_dim, 0);
        Ok(Session { data: p.data, n, d: p.d, dropped: p.dropped, explained: p.explained, knn: None })
    }

    #[wasm_bindgen(getter)]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Dimension after preprocessing.
    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> usize {
        self.d
    }

    #[wasm_bindgen(getter)]
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    #[wasm_bindgen(getter)]
    pub fn explained(&self) -> f64 {
        self.explained
    }

    fn points(&self) -> Points<'_> {
        Points::new(&self.data, self.n, self.d)
    }

    /// Exact kNN list shared by all methods (computed once).
    pub fn compute_knn(&mut self) {
        if self.knn.is_none() {
            let k = K_SHARED.min(self.n - 2);
            self.knn = Some(knn_exact(&self.points(), None, k));
        }
    }

    fn knn(&mut self) -> &Knn {
        self.compute_knn();
        self.knn.as_ref().unwrap()
    }

    pub fn scalerank(&mut self, gamma: f64, n_epochs: usize, neg_scale: f64, seed: u32, every: usize, cb: &Function) -> Vec<f32> {
        self.compute_knn();
        let params = Params { gamma, n_epochs, neg_scale, seed: seed as u64, kmax: K_SHARED.min(self.n - 2), ..Params::default() };
        let fit = scalerank::fit_with_knn(&self.points(), self.knn.as_ref().unwrap(), &params, reporter(cb, every));
        fit.embedding.iter().map(|&v| v as f32).collect()
    }

    pub fn umap(&mut self, n_neighbors: usize, min_dist: f64, seed: u32, every: usize, cb: &Function) -> Vec<f32> {
        let n_neighbors = n_neighbors.clamp(2, (K_SHARED + 1).min(self.n - 1));
        self.compute_knn();
        let params = UmapParams { n_neighbors, min_dist, seed: seed as u64, ..UmapParams::default() };
        let fit = umap::fit_with_knn(&self.points(), self.knn.as_ref().unwrap(), &params, reporter(cb, every));
        fit.embedding.iter().map(|&v| v as f32).collect()
    }

    pub fn tsne(&mut self, perplexity: f64, seed: u32, every: usize, cb: &Function) -> Vec<f32> {
        let k_max = self.knn().k;
        // perplexity is capped so that 3 x perplexity neighbours are available
        let perplexity = perplexity.min(k_max as f64 / 3.0).max(1.0);
        let params = TsneParams { perplexity, seed: seed as u64, ..TsneParams::default() };
        let y = tsne::fit_with_knn(&self.points(), self.knn.as_ref().unwrap(), &params, reporter(cb, every));
        y.iter().map(|&v| v as f32).collect()
    }

    /// Quality of a 2-D embedding: { rnx10, auc, trust10, cont10,
    /// distSpearman, densityCorr, knnAcc?, nAnchors, rnx (curve), overlap10
    /// (per point) }.  Rank metrics use `n_anchors` random rows when that is
    /// below n.  `labels` (one per row, u32::MAX for none) enable knnAcc.
    pub fn quality(&mut self, y: Vec<f32>, labels: Option<Vec<u32>>, n_anchors: usize) -> Result<Object, JsError> {
        let n = self.n;
        if y.len() != 2 * n {
            return Err(JsError::new("embedding must be n x 2"));
        }
        self.compute_knn();
        let yd: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        let p = self.points();
        let anchors = (n_anchors < n).then(|| eval::sample_anchors(n, n_anchors, 0));
        let rm = eval::rank_metrics(&p, &yd, 2, anchors.as_deref(), &[10]);
        let kx = self.knn.as_ref().unwrap();
        let k = 10.min(kx.k);
        let ky = knn_exact(&Points::new(&y, n, 2), None, k);
        let o = Object::new();
        set(&o, "rnx10", rm.r_nx.get(10).copied().unwrap_or(f64::NAN));
        set(&o, "auc", rm.auc_r_nx);
        if let Some(&(_, t, c)) = rm.trust_cont.first() {
            set(&o, "trust10", t);
            set(&o, "cont10", c);
        }
        set(&o, "distSpearman", eval::distance_spearman(&p, &yd, 2, 100_000, 0));
        set(&o, "densityCorr", eval::density_correlation(kx, &ky, k));
        if let Some(l) = labels.filter(|l| l.len() == n && l.iter().all(|&v| v != u32::MAX)) {
            set(&o, "knnAcc", eval::knn_accuracy(&ky, &l, k));
        }
        set(&o, "nAnchors", rm.n_anchors as u32);
        let curve: Vec<f32> = rm.r_nx.iter().map(|&v| v as f32).collect();
        set(&o, "rnx", Float32Array::from(&curve[..]));
        let ov: Vec<f32> = eval::knn_overlap(kx, &ky, k).iter().map(|&v| v as f32).collect();
        set(&o, "overlap10", Float32Array::from(&ov[..]));
        Ok(o)
    }
}
