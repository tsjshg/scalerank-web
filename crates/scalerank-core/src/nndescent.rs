//! Approximate kNN: random-projection-tree initialisation followed by
//! NN-descent (Dong et al. 2011), in the style of PyNNDescent.
//!
//! Each point keeps a bounded max-heap of its current neighbours with a
//! "new" flag.  An iteration samples up to `max_cand` new and old candidates
//! per point (forward and reverse neighbours, by random priority) and
//! compares new-new and new-old pairs among them ("local join"); a pair closer
//! than a point's current worst neighbour updates its heap.  Updates are
//! generated for blocks of points in parallel and applied per target point.

use crate::knn::Knn;
use crate::par;
use crate::points::{Points, sqdist};
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct NnDescentParams {
    pub n_trees: Option<usize>,
    pub leaf_size: Option<usize>,
    pub max_cand: usize,
    pub max_iter: usize,
    /// stop when fewer than delta * n * k heap updates happen in an iteration
    pub delta: f64,
    pub seed: u64,
}

impl Default for NnDescentParams {
    fn default() -> Self {
        NnDescentParams { n_trees: Some(8), leaf_size: None, max_cand: 30, max_iter: 10, delta: 0.01, seed: 0 }
    }
}

/// One point's neighbour heap (max-heap on squared distance).
#[derive(Clone)]
struct Heap {
    d: Vec<f32>,
    id: Vec<u32>,
    new: Vec<bool>,
}

impl Heap {
    fn with_capacity(k: usize) -> Self {
        Heap { d: Vec::with_capacity(k), id: Vec::with_capacity(k), new: Vec::with_capacity(k) }
    }

    #[inline]
    fn worst(&self, k: usize) -> f32 {
        if self.d.len() < k { f32::INFINITY } else { self.d[0] }
    }

    /// Inserts (d, id) if it beats the worst entry and is not present yet.
    fn push(&mut self, k: usize, d: f32, id: u32) -> bool {
        if d >= self.worst(k) || self.id.contains(&id) {
            return false;
        }
        if self.d.len() < k {
            self.d.push(d);
            self.id.push(id);
            self.new.push(true);
            let mut c = self.d.len() - 1;
            while c > 0 {
                let p = (c - 1) / 2;
                if self.d[p] >= self.d[c] {
                    break;
                }
                self.swap(p, c);
                c = p;
            }
        } else {
            self.d[0] = d;
            self.id[0] = id;
            self.new[0] = true;
            let mut p = 0;
            loop {
                let (l, r) = (2 * p + 1, 2 * p + 2);
                let mut m = p;
                if l < k && self.d[l] > self.d[m] {
                    m = l;
                }
                if r < k && self.d[r] > self.d[m] {
                    m = r;
                }
                if m == p {
                    break;
                }
                self.swap(p, m);
                p = m;
            }
        }
        true
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.d.swap(a, b);
        self.id.swap(a, b);
        self.new.swap(a, b);
    }
}

/// Up to `cap` candidates with the smallest random priority (max-heap on priority).
struct Sample {
    pr: Vec<f32>,
    id: Vec<u32>,
}

impl Sample {
    fn push(&mut self, cap: usize, pr: f32, id: u32) {
        if self.id.contains(&id) {
            return;
        }
        if self.pr.len() < cap {
            self.pr.push(pr);
            self.id.push(id);
            let mut c = self.pr.len() - 1;
            while c > 0 {
                let p = (c - 1) / 2;
                if self.pr[p] >= self.pr[c] {
                    break;
                }
                self.pr.swap(p, c);
                self.id.swap(p, c);
                c = p;
            }
        } else if pr < self.pr[0] {
            self.pr[0] = pr;
            self.id[0] = id;
            let mut p = 0;
            loop {
                let (l, r) = (2 * p + 1, 2 * p + 2);
                let mut m = p;
                if l < cap && self.pr[l] > self.pr[m] {
                    m = l;
                }
                if r < cap && self.pr[r] > self.pr[m] {
                    m = r;
                }
                if m == p {
                    break;
                }
                self.pr.swap(p, m);
                self.id.swap(p, m);
                p = m;
            }
        }
    }
}

/// Approximate k nearest neighbours of every point (self excluded).
pub fn knn_nndescent(p: &Points, k: usize, params: &NnDescentParams) -> Knn {
    let n = p.n;
    assert!(k >= 1 && k < n);
    let mut rng = Rng::new(params.seed);
    let mut graph: Vec<Heap> = (0..n).map(|_| Heap::with_capacity(k)).collect();

    // random projection trees: all pairs within each leaf
    let n_trees = params.n_trees.unwrap_or_else(|| (5 + (n as f64).powf(0.25).round() as usize).min(32));
    let leaf_size = params.leaf_size.unwrap_or(k.max(10));
    let tree_seed = rng.next_u64();
    let trees = par::map_collect(n_trees, |t| rp_leaves(p, leaf_size, &mut Rng::stream(tree_seed, t as u64)));
    // one tree at a time, so the updates stay O(n leaf_size)
    for leaves in &trees {
        let worst: Vec<f32> = graph.iter().map(|h| h.worst(k)).collect();
        let mut updates = par::map_collect(leaves.len(), |l| {
            let leaf = &leaves[l];
            let mut u = Vec::new();
            for (a, &x) in leaf.iter().enumerate() {
                for &y in &leaf[a + 1..] {
                    let d = sqdist(p.row(x as usize), p.row(y as usize));
                    if d < worst[x as usize] {
                        u.push((x, y, d));
                    }
                    if d < worst[y as usize] {
                        u.push((y, x, d));
                    }
                }
            }
            u
        })
        .concat();
        apply(&mut graph, k, &mut updates);
    }
    // rows still short of k (tiny leaves) get random neighbours
    for (i, h) in graph.iter_mut().enumerate() {
        while h.d.len() < k {
            let j = rng.below(n);
            if j != i {
                h.push(k, sqdist(p.row(i), p.row(j)), j as u32);
            }
        }
    }

    let cap = params.max_cand.min(k);
    for _it in 0..params.max_iter {
        // candidates: forward and reverse neighbours, new and old separately
        let mut new_c: Vec<Sample> = (0..n).map(|_| Sample { pr: Vec::new(), id: Vec::new() }).collect();
        let mut old_c: Vec<Sample> = (0..n).map(|_| Sample { pr: Vec::new(), id: Vec::new() }).collect();
        for i in 0..n {
            for s in 0..graph[i].id.len() {
                let (j, is_new) = (graph[i].id[s], graph[i].new[s]);
                let pr = rng.uniform() as f32;
                let c = if is_new { &mut new_c } else { &mut old_c };
                c[i].push(cap, pr, j);
                c[j as usize].push(cap, pr, i as u32);
            }
        }
        // sampled new neighbours become old
        for i in 0..n {
            let mut sampled = new_c[i].id.clone();
            sampled.sort_unstable();
            let h = &mut graph[i];
            for s in 0..h.id.len() {
                if h.new[s] && sampled.binary_search(&h.id[s]).is_ok() {
                    h.new[s] = false;
                }
            }
        }
        // local join in blocks; thresholds are refreshed after each block
        let mut changes = 0usize;
        const BLOCK: usize = 4096;
        for start in (0..n).step_by(BLOCK) {
            let end = (start + BLOCK).min(n);
            let worst: Vec<f32> = graph.iter().map(|h| h.worst(k)).collect();
            let ranges = par::ranges(end - start);
            let mut updates = par::map_collect(ranges.len(), |c| {
                let mut u = Vec::new();
                for i in ranges[c].clone() {
                    let i = start + i;
                    let (nw, old) = (&new_c[i].id, &old_c[i].id);
                    for (a, &x) in nw.iter().enumerate() {
                        let xr = p.row(x as usize);
                        for &y in nw[a + 1..].iter().chain(old.iter()) {
                            if x == y {
                                continue;
                            }
                            let d = sqdist(xr, p.row(y as usize));
                            if d < worst[x as usize] {
                                u.push((x, y, d));
                            }
                            if d < worst[y as usize] {
                                u.push((y, x, d));
                            }
                        }
                    }
                }
                u
            })
            .concat();
            changes += apply(&mut graph, k, &mut updates);
        }
        if (changes as f64) < params.delta * (n * k) as f64 {
            break;
        }
    }

    let mut ids = vec![0u32; n * k];
    let mut dist = vec![0f32; n * k];
    for (i, h) in graph.into_iter().enumerate() {
        let mut pairs: Vec<(f32, u32)> = h.d.into_iter().zip(h.id).collect();
        pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (c, (d, j)) in pairs.into_iter().enumerate() {
            ids[i * k + c] = j;
            dist[i * k + c] = d.sqrt();
        }
    }
    Knn { k, ids, dist }
}

/// Applies (target, other, d) updates; returns how many changed a heap.
fn apply(graph: &mut [Heap], k: usize, updates: &mut [(u32, u32, f32)]) -> usize {
    if updates.is_empty() {
        return 0;
    }
    updates.sort_unstable_by_key(|u| u.0);
    let n = graph.len();
    // offsets of each target's updates
    let mut start = vec![0usize; n + 1];
    for u in updates.iter() {
        start[u.0 as usize + 1] += 1;
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    const ROWS: usize = 256;
    let counts = std::sync::Mutex::new(0usize);
    let updates: &[(u32, u32, f32)] = updates;
    par::chunks_mut(graph, ROWS, |c, rows| {
        let mut changed = 0;
        for (r, h) in rows.iter_mut().enumerate() {
            let i = c * ROWS + r;
            for &(_, j, d) in &updates[start[i]..start[i + 1]] {
                changed += h.push(k, d, j) as usize;
            }
        }
        *counts.lock().unwrap() += changed;
    });
    counts.into_inner().unwrap()
}

/// Leaves of one random projection tree: each split is the hyperplane
/// halfway between two random points of the node.
fn rp_leaves(p: &Points, leaf_size: usize, rng: &mut Rng) -> Vec<Vec<u32>> {
    let mut leaves = Vec::new();
    let mut stack = vec![(0..p.n as u32).collect::<Vec<u32>>()];
    while let Some(idx) = stack.pop() {
        if idx.len() <= leaf_size {
            leaves.push(idx);
            continue;
        }
        let a = idx[rng.below(idx.len())] as usize;
        let mut b = idx[rng.below(idx.len())] as usize;
        for _ in 0..8 {
            if b != a {
                break;
            }
            b = idx[rng.below(idx.len())] as usize;
        }
        let (xa, xb) = (p.row(a), p.row(b));
        let normal: Vec<f32> = xa.iter().zip(xb).map(|(u, v)| u - v).collect();
        let offset: f32 = normal.iter().zip(xa.iter().zip(xb)).map(|(w, (u, v))| w * 0.5 * (u + v)).sum();
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for &i in &idx {
            let s: f32 = normal.iter().zip(p.row(i as usize)).map(|(w, x)| w * x).sum::<f32>() - offset;
            if s > 0.0 || (s == 0.0 && rng.below(2) == 0) {
                left.push(i);
            } else {
                right.push(i);
            }
        }
        // degenerate split (duplicates): halve at random
        if left.is_empty() || right.is_empty() {
            let mut idx = idx;
            for t in (1..idx.len()).rev() {
                idx.swap(t, rng.below(t + 1));
            }
            right = idx.split_off(idx.len() / 2);
            left = idx;
        }
        stack.push(left);
        stack.push(right);
    }
    leaves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn::knn_exact;

    #[test]
    fn high_recall_on_clustered_data() {
        let mut r = Rng::new(4);
        let (n, d, k) = (4000, 20, 30);
        let centres: Vec<Vec<f64>> = (0..10).map(|_| (0..d).map(|_| 4.0 * r.normal()).collect()).collect();
        let x: Vec<f32> =
            (0..n).flat_map(|i| (0..d).map(|j| (centres[i % 10][j] + r.normal()) as f32).collect::<Vec<_>>()).collect();
        let p = Points::new(&x, n, d);
        let exact = knn_exact(&p, None, k);
        let approx = knn_nndescent(&p, k, &NnDescentParams::default());
        let mut hit = 0;
        for i in 0..n {
            let e = exact.ids(i);
            hit += approx.ids(i).iter().filter(|j| e.contains(j)).count();
            assert!(!approx.ids(i).contains(&(i as u32)));
            assert!(approx.dist(i).windows(2).all(|w| w[0] <= w[1]));
        }
        let recall = hit as f64 / (n * k) as f64;
        assert!(recall > 0.97, "recall {recall:.3}");
    }
}
