//! Small seedable PRNG (xoshiro256++), so results do not depend on an external
//! crate's version and the same stream comes out natively and in WASM.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        Rng { s: [splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x)] }
    }

    /// Independent stream number `idx` of `seed`, e.g. one per block of work,
    /// so results do not depend on how the blocks are spread over threads.
    pub fn stream(seed: u64, idx: u64) -> Self {
        Rng::new(seed ^ idx.wrapping_mul(0xD1B5_4A32_D192_ED03))
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in 0..n (multiply-shift; the bias is below 2^-32 for n < 2^32).
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    pub fn permutation(&mut self, n: usize) -> Vec<u32> {
        let mut p: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            p.swap(i, self.below(i + 1));
        }
        p
    }

    /// k distinct values from 0..n (partial Fisher-Yates), in random order.
    pub fn sample(&mut self, n: usize, k: usize) -> Vec<u32> {
        assert!(k <= n);
        let mut p: Vec<u32> = (0..n as u32).collect();
        for i in 0..k {
            p.swap(i, i + self.below(n - i));
        }
        p.truncate(k);
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_moments_and_sample_distinct() {
        let mut r = Rng::new(1);
        let u: Vec<f64> = (0..100_000).map(|_| r.uniform()).collect();
        let m = u.iter().sum::<f64>() / u.len() as f64;
        assert!((m - 0.5).abs() < 0.01);
        let z: Vec<f64> = (0..100_000).map(|_| r.normal()).collect();
        let v = z.iter().map(|x| x * x).sum::<f64>() / z.len() as f64;
        assert!((v - 1.0).abs() < 0.02);
        let mut s = r.sample(1000, 300);
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 300);
        let mut p = r.permutation(50);
        p.sort();
        assert_eq!(p, (0..50).collect::<Vec<u32>>());
    }
}
