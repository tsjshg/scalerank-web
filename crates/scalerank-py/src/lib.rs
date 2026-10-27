//! Python bindings of scalerank-core, so drbench can run and score the Rust
//! implementation next to its Python prototype.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scalerank_core::knn::knn_exact;
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
        let init = match init {
            "pca" => Init::Pca,
            "random" => Init::Random,
            other => return Err(PyValueError::new_err(format!("unknown init {other:?}"))),
        };
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

/// Exact kNN (self excluded): (ids, distances), each n x k.
#[pyfunction]
fn knn<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, k: usize) -> (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>) {
    let (data, n, d) = to_f32(&x);
    let r = py.detach(|| knn_exact(&Points::new(&data, n, d), None, k));
    let ids = Array2::from_shape_vec((n, k), r.ids.iter().map(|&v| if v == u32::MAX { -1 } else { v as i64 }).collect()).unwrap();
    let dist = Array2::from_shape_vec((n, k), r.dist.iter().map(|&v| v as f64).collect()).unwrap();
    (ids.into_pyarray(py), dist.into_pyarray(py))
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
        let t = RankTable::build(&p, kmax, r_pool, pool_ratio, pool_per_level, &mut Rng::new(seed));
        i.iter()
            .zip(&j)
            .map(|(&a, &b)| t.rank_of(a as usize, scalerank_core::points::dist(p.row(a as usize), p.row(b as usize))) as f64)
            .collect::<Vec<f64>>()
    });
    est.into_pyarray(py)
}

#[pymodule]
fn scalerank_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ScaleRankFast>()?;
    m.add_function(wrap_pyfunction!(knn, m)?)?;
    m.add_function(wrap_pyfunction!(rank_estimates, m)?)?;
    Ok(())
}
