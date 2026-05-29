//! Self-contained, seedable gradient noise.
//!
//! We avoid pulling in an external noise crate so world generation stays fully
//! deterministic and under our control. This is a Perlin-style improved-noise
//! implementation with a seeded permutation table, plus fractal Brownian motion
//! (fBm) helpers used to layer octaves for natural-looking terrain — the same
//! technique Minecraft and Terraria use for their height fields.

/// Permutation-table gradient noise. Cheap, deterministic, seedable.
#[derive(Clone)]
pub struct Perlin {
    /// Doubled permutation table to avoid index wrapping in the hot loop.
    perm: [u8; 512],
}

impl Perlin {
    pub fn new(seed: u64) -> Self {
        // Fisher–Yates shuffle of 0..256 driven by a SplitMix64 PRNG so the
        // table is a deterministic function of the seed.
        let mut p: [u8; 256] = [0; 256];
        for (i, slot) in p.iter_mut().enumerate() {
            *slot = i as u8;
        }
        let mut rng = SplitMix64::new(seed);
        for i in (1..256).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Self { perm }
    }

    #[inline]
    fn grad2(hash: u8, x: f32, y: f32) -> f32 {
        // 8 gradient directions on the unit circle (diagonals + axes).
        match hash & 7 {
            0 => x + y,
            1 => x - y,
            2 => -x + y,
            3 => -x - y,
            4 => x,
            5 => -x,
            6 => y,
            _ => -y,
        }
    }

    /// 2D noise in roughly `[-1, 1]`.
    pub fn noise2(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor() as i32 & 255;
        let yi = y.floor() as i32 & 255;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let u = fade(xf);
        let v = fade(yf);
        let p = &self.perm;
        let aa = p[(p[xi as usize] as usize + yi as usize) & 511];
        let ab = p[(p[xi as usize] as usize + yi as usize + 1) & 511];
        let ba = p[(p[(xi as usize + 1) & 511] as usize + yi as usize) & 511];
        let bb = p[(p[(xi as usize + 1) & 511] as usize + yi as usize + 1) & 511];
        let x1 = lerp(Self::grad2(aa, xf, yf), Self::grad2(ba, xf - 1.0, yf), u);
        let x2 = lerp(
            Self::grad2(ab, xf, yf - 1.0),
            Self::grad2(bb, xf - 1.0, yf - 1.0),
            u,
        );
        lerp(x1, x2, v)
    }

    #[inline]
    fn grad3(hash: u8, x: f32, y: f32, z: f32) -> f32 {
        match hash & 15 {
            0 | 12 => x + y,
            1 | 14 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x + z,
            5 => -x + z,
            6 => x - z,
            7 => -x - z,
            8 => y + z,
            9 | 13 => -y + z,
            10 => y - z,
            _ => -y - z,
        }
    }

    /// 3D noise in roughly `[-1, 1]` (used for caves and overhangs).
    pub fn noise3(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = (x.floor() as i32 & 255) as usize;
        let yi = (y.floor() as i32 & 255) as usize;
        let zi = (z.floor() as i32 & 255) as usize;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();
        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);
        let p = &self.perm;
        let a = p[xi] as usize + yi;
        let aa = p[a & 511] as usize + zi;
        let ab = p[(a + 1) & 511] as usize + zi;
        let b = p[(xi + 1) & 511] as usize + yi;
        let ba = p[b & 511] as usize + zi;
        let bb = p[(b + 1) & 511] as usize + zi;
        let g = Self::grad3;
        let x1 = lerp(
            g(p[aa & 511], xf, yf, zf),
            g(p[ba & 511], xf - 1.0, yf, zf),
            u,
        );
        let x2 = lerp(
            g(p[ab & 511], xf, yf - 1.0, zf),
            g(p[bb & 511], xf - 1.0, yf - 1.0, zf),
            u,
        );
        let y1 = lerp(x1, x2, v);
        let x3 = lerp(
            g(p[(aa + 1) & 511], xf, yf, zf - 1.0),
            g(p[(ba + 1) & 511], xf - 1.0, yf, zf - 1.0),
            u,
        );
        let x4 = lerp(
            g(p[(ab + 1) & 511], xf, yf - 1.0, zf - 1.0),
            g(p[(bb + 1) & 511], xf - 1.0, yf - 1.0, zf - 1.0),
            u,
        );
        let y2 = lerp(x3, x4, v);
        lerp(y1, y2, w)
    }

    /// Fractal Brownian motion: sum of `octaves` noise layers at doubling
    /// frequency and halving amplitude. Returns roughly `[-1, 1]`.
    pub fn fbm2(&self, x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let mut freq = 1.0;
        let mut amp = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for _ in 0..octaves {
            sum += self.noise2(x * freq, y * freq) * amp;
            norm += amp;
            freq *= lacunarity;
            amp *= gain;
        }
        if norm > 0.0 {
            sum / norm
        } else {
            0.0
        }
    }

    pub fn fbm3(&self, x: f32, y: f32, z: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let mut freq = 1.0;
        let mut amp = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for _ in 0..octaves {
            sum += self.noise3(x * freq, y * freq, z * freq) * amp;
            norm += amp;
            freq *= lacunarity;
            amp *= gain;
        }
        if norm > 0.0 {
            sum / norm
        } else {
            0.0
        }
    }
}

#[inline]
fn fade(t: f32) -> f32 {
    // Ken Perlin's quintic smoothstep 6t^5 - 15t^4 + 10t^3.
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Tiny fast PRNG used to seed the permutation table and for scatter decisions
/// (tree/flower placement). Deterministic given a seed.
#[derive(Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Use the top 24 bits for a uniform mantissa.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Deterministically derive a per-coordinate seed, so decoration decisions at a
/// given world column are reproducible regardless of generation order.
#[inline]
pub fn hash_coords(seed: u64, x: i32, z: i32) -> u64 {
    let mut h = seed;
    h ^= (x as u32 as u64).wrapping_mul(0x9E37_79B1);
    h = h.rotate_left(31);
    h ^= (z as u32 as u64).wrapping_mul(0x85EB_CA77);
    let mut m = SplitMix64::new(h);
    m.next_u64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_deterministic_for_seed() {
        let a = Perlin::new(42);
        let b = Perlin::new(42);
        for i in 0..50 {
            let x = i as f32 * 0.137;
            assert_eq!(a.noise2(x, -x), b.noise2(x, -x));
            assert_eq!(a.noise3(x, x, -x), b.noise3(x, x, -x));
        }
    }

    #[test]
    fn different_seeds_differ() {
        let a = Perlin::new(1);
        let b = Perlin::new(2);
        let mut diff = false;
        for i in 0..100 {
            let x = i as f32 * 0.3;
            if (a.noise2(x, x) - b.noise2(x, x)).abs() > 1e-6 {
                diff = true;
                break;
            }
        }
        assert!(diff, "distinct seeds produced identical noise");
    }

    #[test]
    fn noise_stays_in_expected_range() {
        let p = Perlin::new(7);
        for i in 0..1000 {
            let x = i as f32 * 0.05;
            let y = (i as f32 * 0.017).sin() * 20.0;
            let n = p.noise2(x, y);
            assert!(n >= -1.5 && n <= 1.5, "noise2 out of range: {n}");
            let f = p.fbm2(x, y, 5, 2.0, 0.5);
            assert!(f >= -1.1 && f <= 1.1, "fbm2 out of range: {f}");
        }
    }

    #[test]
    fn splitmix_uniform_floatish() {
        let mut r = SplitMix64::new(99);
        let mut sum = 0.0;
        const N: usize = 10_000;
        for _ in 0..N {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v));
            sum += v;
        }
        let mean = sum / N as f32;
        assert!((mean - 0.5).abs() < 0.05, "mean {mean} not near 0.5");
    }
}
