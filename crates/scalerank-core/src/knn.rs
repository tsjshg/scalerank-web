//! Exact k-nearest neighbours by brute force.
//!
//! Rows are processed in tiles so that each candidate point is read once per
//! tile rather than once per row; each row keeps a bounded max-heap.

use crate::par;
use crate::points::{Points, sqdist};

/// Row-major n x k neighbour lists, nearest first.  Rows with fewer than k
/// candidates are padded with `u32::MAX` / infinity.
#[derive(Clone, Debug)]
pub struct Knn {
    pub k: usize,
    pub ids: Vec<u32>,
    pub dist: Vec<f32>,
}

impl Knn {
    pub fn ids(&self, i: usize) -> &[u32] {
        &self.ids[i * self.k..(i + 1) * self.k]
    }
    pub fn dist(&self, i: usize) -> &[f32] {
        &self.dist[i * self.k..(i + 1) * self.k]
    }
}

/// Bounded max-heap of (squared distance, id): the k smallest seen so far.
struct TopK {
    k: usize,
    d: Vec<f32>,
    id: Vec<u32>,
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK { k, d: Vec::with_capacity(k), id: Vec::with_capacity(k) }
    }

    #[inline]
    fn worst(&self) -> f32 {
        if self.d.len() < self.k { f32::INFINITY } else { self.d[0] }
    }

    fn push(&mut self, d: f32, id: u32) {
        if self.d.len() < self.k {
            self.d.push(d);
            self.id.push(id);
            let mut c = self.d.len() - 1;
            while c > 0 {
                let p = (c - 1) / 2;
                if self.d[p] >= self.d[c] {
                    break;
                }
                self.d.swap(p, c);
                self.id.swap(p, c);
                c = p;
            }
        } else if d < self.d[0] {
            self.d[0] = d;
            self.id[0] = id;
            let n = self.k;
            let mut p = 0;
            loop {
                let (l, r) = (2 * p + 1, 2 * p + 2);
                let mut m = p;
                if l < n && self.d[l] > self.d[m] {
                    m = l;
                }
                if r < n && self.d[r] > self.d[m] {
                    m = r;
                }
                if m == p {
                    break;
                }
                self.d.swap(p, m);
                self.id.swap(p, m);
                p = m;
            }
        }
    }

    /// Sorted ascending (ties by id), as Euclidean distances, padded to k.
    fn finish(self, ids: &mut [u32], dist: &mut [f32]) {
        let mut pairs: Vec<(f32, u32)> = self.d.into_iter().zip(self.id).collect();
        pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (c, slot) in ids.iter_mut().enumerate() {
            match pairs.get(c) {
                Some(&(d, id)) => {
                    *slot = id;
                    dist[c] = d.sqrt();
                }
                None => {
                    *slot = u32::MAX;
                    dist[c] = f32::INFINITY;
                }
            }
        }
    }
}

const TILE: usize = 32;

/// The k nearest neighbours of every point among `cands` (all points when
/// `None`), excluding the point itself.  Returned ids are point indices.
pub fn knn_exact(p: &Points, cands: Option<&[u32]>, k: usize) -> Knn {
    let n = p.n;
    let all: Vec<u32>;
    let cands = match cands {
        Some(c) => c,
        None => {
            all = (0..n as u32).collect();
            &all
        }
    };
    let mut ids = vec![0u32; n * k];
    let mut dist = vec![0f32; n * k];
    if k == 0 {
        return Knn { k, ids, dist };
    }
    // tiles of TILE rows: compute into per-tile buffers, then copy out
    let tiles = par::map_collect(n.div_ceil(TILE), |t| {
        let rows = t * TILE..((t + 1) * TILE).min(n);
        let mut heaps: Vec<TopK> = rows.clone().map(|_| TopK::new(k)).collect();
        for &c in cands {
            let xc = p.row(c as usize);
            for (h, i) in heaps.iter_mut().zip(rows.clone()) {
                if i == c as usize {
                    continue;
                }
                let d = sqdist(p.row(i), xc);
                if d < h.worst() {
                    h.push(d, c);
                }
            }
        }
        let mut ti = vec![0u32; rows.len() * k];
        let mut td = vec![0f32; rows.len() * k];
        for (r, h) in heaps.into_iter().enumerate() {
            h.finish(&mut ti[r * k..(r + 1) * k], &mut td[r * k..(r + 1) * k]);
        }
        (ti, td)
    });
    for (t, (ti, td)) in tiles.into_iter().enumerate() {
        let s = t * TILE * k;
        ids[s..s + ti.len()].copy_from_slice(&ti);
        dist[s..s + td.len()].copy_from_slice(&td);
    }
    Knn { k, ids, dist }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn matches_full_sort() {
        let mut r = Rng::new(3);
        let (n, d, k) = (157, 7, 9);
        let x: Vec<f32> = (0..n * d).map(|_| r.normal() as f32).collect();
        let p = Points::new(&x, n, d);
        let knn = knn_exact(&p, None, k);
        for i in [0, 31, 32, 100, 156] {
            let mut all: Vec<(f32, u32)> =
                (0..n).filter(|&j| j != i).map(|j| (sqdist(p.row(i), p.row(j)), j as u32)).collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            let want: Vec<u32> = all[..k].iter().map(|a| a.1).collect();
            assert_eq!(knn.ids(i), &want[..]);
        }
        // a candidate subset smaller than k pads the rows
        let sub = [3u32, 10, 50];
        let small = knn_exact(&p, Some(&sub), 4);
        assert!(small.ids(3).contains(&u32::MAX) && !small.ids(3).contains(&3));
        assert_eq!(small.ids(3)[2], u32::MAX);
        assert!(small.ids(0)[..3].iter().all(|i| sub.contains(i)));
    }
}
