//! Python bindings of scalerank-core, so drbench can run and score the Rust
//! implementation next to its Python prototype.

#![allow(clippy::type_complexity)]

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scalerank_core::eval;
use scalerank_core::knn::knn_exact;
use scalerank_core::nndescent::{NnDescentParams, knn_nndescent};
use scalerank_core::ranks::RankTable;
use scalerank_core::{Init, Params, Points, Rng};

fn to_f32(x: &PyReadonlyArray2<f64>) -> (Vec<f32>, usize, usize) {
    let a = x.as_array();
    let (n, d) = a.dim();
    (a.iter().map(|&v| v as f32).collect(), n, d)
}

fn seed_or_clock(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
    })
}

fn parse_init(init: &str) -> PyResult<Init> {
    match init {
        "pca" => Ok(Init::Pca),
        "random" => Ok(Init::Random),
        other => Err(PyValueError::new_err(format!("unknown init {other:?}"))),
    }
}

/// ScaleRank with N^2-free rank tables; same parameters as drbench's ScaleRankFast.
#[pyclass(module = "scalerank_rs")]
struct ScaleRankFast {
    params: Params,
    verbose: bool,
    #[pyo3(get)]
    loss_history_: Vec<f64>,
    #[pyo3(get)]
    table_len_: f64,
}

#[pymethods]
impl ScaleRankFast {
    #[new]
    #[pyo3(signature = (
        gamma=1.0, k0=None, width=1.0, n_components=2, n_epochs=800, per_point=8, n_neg=20,
        lr=1.0, init="pca", init_scale=1.0, kmax=100, n_cand=64, pool_ratio=4.0,
        pool_per_level=16, fresh_min=4.0, pool_stopgrad=true, neg_scale=0.0, block=16,
        random_state=None, verbose=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        gamma: f64,
        k0: Option<f64>,
        width: f64,
        n_components: usize,
        n_epochs: usize,
        per_point: usize,
        n_neg: usize,
        lr: f64,
        init: &str,
        init_scale: f64,
        kmax: usize,
        n_cand: usize,
        pool_ratio: f64,
        pool_per_level: usize,
        fresh_min: f64,
        pool_stopgrad: bool,
        neg_scale: f64,
        block: usize,
        random_state: Option<u64>,
        verbose: bool,
    ) -> PyResult<Self> {
        let init = parse_init(init)?;
        let params = Params {
            gamma,
            k0,
            width,
            n_components,
            n_epochs,
            per_point,
            n_neg,
            lr,
            init,
            init_scale,
            kmax,
            n_cand,
            pool_ratio,
            pool_per_level,
            fresh_min,
            pool_stopgrad,
            neg_scale,
            block,
            seed: seed_or_clock(random_state),
        };
        Ok(ScaleRankFast { params, verbose, loss_history_: Vec::new(), table_len_: 0.0 })
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (data, n, d) = to_f32(&x);
        if n < 4 {
            return Err(PyValueError::new_err("need at least 4 points"));
        }
        let params = self.params.clone();
        let verbose = self.verbose;
        let fit = py.detach(|| {
            scalerank_core::fit(&Points::new(&data, n, d), &params, |pr| {
                if verbose && (pr.epoch == 1 || pr.epoch % 50 == 0) {
                    eprintln!("epoch {:4}  loss {:.4}", pr.epoch, pr.loss);
                }
                true
            })
        });
        self.loss_history_ = fit.loss_history;
        self.table_len_ = fit.table_len;
        let nc = params.n_components;
        let y = Array2::from_shape_vec((n, nc), fit.embedding).expect("shape");
        Ok(y.into_pyarray(py))
    }
}

/// UMAP (umap-learn's defaults, PCA initialisation).
#[pyfunction]
#[pyo3(signature = (x, n_neighbors=15, min_dist=0.1, spread=1.0, n_components=2, n_epochs=None, learning_rate=1.0, negative_sample_rate=5, init="pca", random_state=None))]
#[allow(clippy::too_many_arguments)]
fn umap<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    n_neighbors: usize,
    min_dist: f64,
    spread: f64,
    n_components: usize,
    n_epochs: Option<usize>,
    learning_rate: f64,
    negative_sample_rate: usize,
    init: &str,
    random_state: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (data, n, d) = to_f32(&x);
    let init = parse_init(init)?;
    let params = scalerank_core::umap::UmapParams {
        n_neighbors,
        min_dist,
        spread,
        n_components,
        n_epochs,
        learning_rate,
        negative_sample_rate,
        repulsion_strength: 1.0,
        init,
        seed: seed_or_clock(random_state),
    };
    let fit = py.detach(|| scalerank_core::umap::fit(&Points::new(&data, n, d), &params, |_| true));
    Ok(Array2::from_shape_vec((n, n_components), fit.embedding).expect("shape").into_pyarray(py))
}

/// t-SNE (openTSNE's defaults, Barnes-Hut repulsion), 2-D.
#[pyfunction]
#[pyo3(signature = (x, perplexity=30.0, early_exaggeration_iter=250, early_exaggeration=12.0, n_iter=500, exaggeration=1.0, learning_rate=None, theta=0.5, init="pca", random_state=None))]
#[allow(clippy::too_many_arguments)]
fn tsne<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    perplexity: f64,
    early_exaggeration_iter: usize,
    early_exaggeration: f64,
    n_iter: usize,
    exaggeration: f64,
    learning_rate: Option<f64>,
    theta: f64,
    init: &str,
    random_state: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (data, n, d) = to_f32(&x);
    let params = scalerank_core::tsne::TsneParams {
        perplexity,
        early_exaggeration_iter,
        early_exaggeration,
        n_iter,
        exaggeration,
        learning_rate,
        theta,
        init: parse_init(init)?,
        seed: seed_or_clock(random_state),
        ..Default::default()
    };
    let y = py.detach(|| scalerank_core::tsne::fit(&Points::new(&data, n, d), &params, |_| true));
    Ok(Array2::from_shape_vec((n, 2), y).expect("shape").into_pyarray(py))
}

/// UMAP's fuzzy graph as COO arrays (rows, cols, weights), both directions.
#[pyfunction]
#[pyo3(signature = (x, n_neighbors=15))]
fn umap_graph<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    n_neighbors: usize,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
    let (data, n, d) = to_f32(&x);
    let g = py.detach(|| {
        let p = Points::new(&data, n, d);
        scalerank_core::umap::fuzzy_graph(&knn_exact(&p, None, n_neighbors - 1), n, n_neighbors)
    });
    let rows: Vec<i64> = g.iter().map(|e| e.0 as i64).collect();
    let cols: Vec<i64> = g.iter().map(|e| e.1 as i64).collect();
    let vals: Vec<f64> = g.iter().map(|e| e.2).collect();
    (rows.into_pyarray(py), cols.into_pyarray(py), vals.into_pyarray(py))
}

/// kNN (self excluded), exact or by NN-descent: (ids, distances), each n x k.
#[pyfunction]
#[pyo3(signature = (x, k, method="exact", seed=0, n_trees=8, leaf_size=None, max_cand=30, max_iter=10, delta=0.01))]
#[allow(clippy::too_many_arguments)]
fn knn<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    k: usize,
    method: &str,
    seed: u64,
    n_trees: Option<usize>,
    leaf_size: Option<usize>,
    max_cand: usize,
    max_iter: usize,
    delta: f64,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
    let (data, n, d) = to_f32(&x);
    let exact = match method {
        "exact" => true,
        "nndescent" => false,
        other => return Err(PyValueError::new_err(format!("unknown method {other:?}"))),
    };
    let r = py.detach(|| {
        let p = Points::new(&data, n, d);
        if exact { knn_exact(&p, None, k) } else { knn_nndescent(&p, k, &NnDescentParams { seed, n_trees, leaf_size, max_cand, max_iter, delta }) }
    });
    let ids = Array2::from_shape_vec((n, k), r.ids.iter().map(|&v| if v == u32::MAX { -1 } else { v as i64 }).collect()).unwrap();
    let dist = Array2::from_shape_vec((n, k), r.dist.iter().map(|&v| v as f64).collect()).unwrap();
    Ok((ids.into_pyarray(py), dist.into_pyarray(py)))
}

/// Rank of j[k] among the neighbours of i[k], estimated from the rank table.
#[pyfunction]
#[pyo3(signature = (x, i, j, kmax=100, r_pool=100.0, pool_ratio=4.0, pool_per_level=16, seed=0))]
#[allow(clippy::too_many_arguments)]
fn rank_estimates<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    i: PyReadonlyArray1<'py, i64>,
    j: PyReadonlyArray1<'py, i64>,
    kmax: usize,
    r_pool: f64,
    pool_ratio: f64,
    pool_per_level: usize,
    seed: u64,
) -> Bound<'py, PyArray1<f64>> {
    let (data, n, d) = to_f32(&x);
    let (i, j) = (i.as_array().to_vec(), j.as_array().to_vec());
    let est = py.detach(|| {
        let p = Points::new(&data, n, d);
        let t = RankTable::build(&p, &knn_exact(&p, None, kmax), kmax, r_pool, pool_ratio, pool_per_level, &mut Rng::new(seed));
        i.iter()
            .zip(&j)
            .map(|(&a, &b)| t.rank_of(a as usize, scalerank_core::points::dist(p.row(a as usize), p.row(b as usize))) as f64)
            .collect::<Vec<f64>>()
    });
    est.into_pyarray(py)
}

/// Quality metrics with drbench's names: (scalars, arrays).  Above
/// `exact_max_n` points the rank metrics use `n_anchors` random rows.
#[pyfunction]
#[pyo3(signature = (x, y, labels=None, ks=vec![5, 10, 30, 100], n_anchors=2000, exact_max_n=10000, seed=0))]
#[allow(clippy::too_many_arguments)]
fn evaluate<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    labels: Option<PyReadonlyArray1<'py, i64>>,
    ks: Vec<usize>,
    n_anchors: usize,
    exact_max_n: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let (data, n, d) = to_f32(&x);
    let ya = y.as_array();
    let dy = ya.dim().1;
    if ya.dim().0 != n {
        return Err(PyValueError::new_err("x and y have different numbers of rows"));
    }
    let yv: Vec<f64> = ya.iter().copied().collect();
    let labels: Option<Vec<u32>> = labels.map(|l| l.as_array().iter().map(|&v| v as u32).collect());
    let ks_ok: Vec<usize> = ks.iter().copied().filter(|&k| k >= 1 && 2 * k < n).collect();
    let (rm, overlap, dsp, dens, acc) = py.detach(|| {
        let p = Points::new(&data, n, d);
        let anchors = (n > exact_max_n).then(|| eval::sample_anchors(n, n_anchors, seed));
        let rm = eval::rank_metrics(&p, &yv, dy, anchors.as_deref(), &ks_ok);
        let kk = ks_ok.iter().copied().max().unwrap_or(1).max(10).min(n - 1);
        let yf: Vec<f32> = yv.iter().map(|&v| v as f32).collect();
        let kx = knn_exact(&p, None, kk);
        let ky = knn_exact(&Points::new(&yf, n, dy), None, kk);
        let overlap: Vec<(usize, Vec<f64>)> = ks_ok.iter().map(|&k| (k, eval::knn_overlap(&kx, &ky, k))).collect();
        let dsp = eval::distance_spearman(&p, &yv, dy, 200_000, seed);
        let dens = eval::density_correlation(&kx, &ky, 10.min(kk));
        let acc = labels.as_ref().map(|l| eval::knn_accuracy(&ky, l, 10.min(kk)));
        (rm, overlap, dsp, dens, acc)
    });
    let scalars = PyDict::new(py);
    scalars.set_item("auc_rnx", rm.auc_r_nx)?;
    scalars.set_item("footrule", rm.footrule)?;
    scalars.set_item("dist_spearman", dsp)?;
    scalars.set_item("density_corr", dens)?;
    for &(k, t, c) in &rm.trust_cont {
        scalars.set_item(format!("trust@{k}"), t)?;
        scalars.set_item(format!("cont@{k}"), c)?;
        scalars.set_item(format!("rnx@{k}"), rm.r_nx[k])?;
    }
    if let Some(a) = acc {
        scalars.set_item("knn_acc_loo", a)?;
    }
    if rm.n_anchors < n {
        scalars.set_item("n_anchors", rm.n_anchors)?;
    }
    let arrays = PyDict::new(py);
    arrays.set_item("r_nx", rm.r_nx.into_pyarray(py))?;
    for (k, v) in overlap {
        arrays.set_item(format!("overlap@{k}"), v.into_pyarray(py))?;
    }
    Ok((scalars, arrays))
}

#[pymodule]
fn scalerank_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ScaleRankFast>()?;
    m.add_function(wrap_pyfunction!(knn, m)?)?;
    m.add_function(wrap_pyfunction!(rank_estimates, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(umap, m)?)?;
    m.add_function(wrap_pyfunction!(umap_graph, m)?)?;
    m.add_function(wrap_pyfunction!(tsne, m)?)?;
    Ok(())
}
