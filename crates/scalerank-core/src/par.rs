//! The few parallel patterns the crate needs, with a serial fallback when the
//! `parallel` feature (rayon) is off, e.g. for single-threaded WASM.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn num_threads() -> usize {
    #[cfg(feature = "parallel")]
    return rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    return 1;
}

/// `f(0), ..., f(n-1)` in order, computed in parallel.
pub fn map_collect<R: Send>(n: usize, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    #[cfg(feature = "parallel")]
    return (0..n).into_par_iter().map(f).collect();
    #[cfg(not(feature = "parallel"))]
    return (0..n).map(f).collect();
}

/// Calls `f(chunk_index, chunk)` on consecutive chunks of `data`.
pub fn chunks_mut<T: Send>(data: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    #[cfg(feature = "parallel")]
    data.par_chunks_mut(chunk).enumerate().for_each(|(c, x)| f(c, x));
    #[cfg(not(feature = "parallel"))]
    data.chunks_mut(chunk).enumerate().for_each(|(c, x)| f(c, x));
}

/// Splits 0..n into consecutive ranges, roughly two per thread.
pub fn ranges(n: usize) -> Vec<std::ops::Range<usize>> {
    let parts = (2 * num_threads()).clamp(1, n.max(1));
    (0..parts).map(|p| p * n / parts..(p + 1) * n / parts).collect()
}
