//! Core of ScaleRank: dimensionality reduction that preserves neighbour
//! ranks over a chosen range of scales, without N x N structures.
//!
//! The crate has no dependencies beyond the optional `rayon` (feature
//! `parallel`), so it builds for WASM as is.

pub mod eval;
pub mod knn;
pub mod nndescent;
pub mod par;
pub mod pca;
pub mod points;
pub mod prep;
pub mod ranks;
pub mod rng;
pub mod scalerank;
pub mod tsne;
pub mod umap;

pub use points::Points;
pub use rng::Rng;
pub use scalerank::{Fit, Init, Params, Progress, fit};
