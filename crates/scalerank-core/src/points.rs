/// A row-major n x d matrix of input points.
#[derive(Clone, Copy, Debug)]
pub struct Points<'a> {
    pub data: &'a [f32],
    pub n: usize,
    pub d: usize,
}

impl<'a> Points<'a> {
    pub fn new(data: &'a [f32], n: usize, d: usize) -> Self {
        assert_eq!(data.len(), n * d, "data length must be n * d");
        Points { data, n, d }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &'a [f32] {
        &self.data[i * self.d..(i + 1) * self.d]
    }
}

/// Squared Euclidean distance; eight accumulators so it vectorises.
#[inline]
pub fn sqdist(a: &[f32], b: &[f32]) -> f32 {
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    let mut acc = [0f32; 8];
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..8 {
            let t = x[k] - y[k];
            acc[k] += t * t;
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        let t = x - y;
        s += t * t;
    }
    s
}

#[inline]
pub fn dist(a: &[f32], b: &[f32]) -> f32 {
    sqdist(a, b).sqrt()
}
