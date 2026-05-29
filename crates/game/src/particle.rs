//! A small particle system for cosy flourishes: cherry petals drifting on the
//! breeze and little bursts when blocks are mined. Particles are short-lived
//! billposted cubes drawn in the transparent layer, fading out over their life.

use glam::Vec3;
use pixelcraft_core::block::Color;
use pixelcraft_mesh::Vertex;
use pixelcraft_worldgen::SplitMix64;

#[derive(Clone)]
struct Particle {
    pos: Vec3,
    vel: Vec3,
    life: f32,
    max_life: f32,
    size: f32,
    color: Color,
    /// Sway phase for fluttering petals (0 = no sway, e.g. debris).
    flutter: f32,
}

pub struct ParticleSystem {
    particles: Vec<Particle>,
    rng: SplitMix64,
    max: usize,
}

impl ParticleSystem {
    pub fn new(seed: u64) -> Self {
        Self {
            particles: Vec::new(),
            rng: SplitMix64::new(seed ^ 0x9A7C_1EE),
            max: 600,
        }
    }

    pub fn count(&self) -> usize {
        self.particles.len()
    }

    /// Drift petals down from above the player while in a cherry grove.
    pub fn emit_petals(&mut self, around: Vec3, dt: f32) {
        // Rate-limited Poisson-ish spawning.
        let expected = 14.0 * dt; // particles/sec
        let mut n = expected as i32;
        if self.rng.next_f32() < expected.fract() {
            n += 1;
        }
        for _ in 0..n {
            if self.particles.len() >= self.max {
                break;
            }
            let r = 18.0;
            let pos = around
                + Vec3::new(
                    (self.rng.next_f32() - 0.5) * r,
                    6.0 + self.rng.next_f32() * 8.0,
                    (self.rng.next_f32() - 0.5) * r,
                );
            self.particles.push(Particle {
                pos,
                vel: Vec3::new(0.0, -0.6 - self.rng.next_f32() * 0.4, 0.0),
                life: 0.0,
                max_life: 6.0 + self.rng.next_f32() * 3.0,
                size: 0.12,
                color: Color::rgb(246, 178, 212),
                flutter: 1.0 + self.rng.next_f32(),
            });
        }
    }

    /// Emit weather precipitation around the player: drifting snow or quick
    /// rain streaks falling from above.
    pub fn emit_weather(&mut self, around: Vec3, snow: bool, dt: f32) {
        let rate = if snow { 26.0 } else { 60.0 };
        let expected = rate * dt;
        let mut n = expected as i32;
        if self.rng.next_f32() < expected.fract() {
            n += 1;
        }
        for _ in 0..n {
            if self.particles.len() >= self.max {
                break;
            }
            let r = 22.0;
            let pos = around
                + Vec3::new(
                    (self.rng.next_f32() - 0.5) * r,
                    8.0 + self.rng.next_f32() * 8.0,
                    (self.rng.next_f32() - 0.5) * r,
                );
            if snow {
                self.particles.push(Particle {
                    pos,
                    vel: Vec3::new(0.0, -1.1 - self.rng.next_f32() * 0.5, 0.0),
                    life: 0.0,
                    max_life: 5.0 + self.rng.next_f32() * 2.0,
                    size: 0.08,
                    color: Color::rgb(245, 248, 255),
                    flutter: 1.0 + self.rng.next_f32(),
                });
            } else {
                self.particles.push(Particle {
                    pos,
                    vel: Vec3::new(0.0, -16.0 - self.rng.next_f32() * 4.0, 0.0),
                    life: 0.0,
                    max_life: 1.2,
                    size: 0.05,
                    color: Color::rgb(170, 190, 220),
                    flutter: 0.0,
                });
            }
        }
    }

    /// A small debris burst (e.g. when a block is broken).
    pub fn burst(&mut self, at: Vec3, color: Color, count: u32) {
        for _ in 0..count {
            if self.particles.len() >= self.max {
                break;
            }
            let dir = Vec3::new(
                self.rng.next_f32() - 0.5,
                self.rng.next_f32() * 0.8,
                self.rng.next_f32() - 0.5,
            )
            .normalize_or_zero();
            self.particles.push(Particle {
                pos: at,
                vel: dir * (2.0 + self.rng.next_f32() * 2.0),
                life: 0.0,
                max_life: 0.6 + self.rng.next_f32() * 0.4,
                size: 0.1,
                color,
                flutter: 0.0,
            });
        }
    }

    pub fn update(&mut self, dt: f32) {
        for p in &mut self.particles {
            p.life += dt;
            // Debris falls under gravity; petals flutter and fall gently.
            if p.flutter > 0.0 {
                p.pos.x += (p.life * p.flutter * 2.0).sin() * 0.4 * dt;
                p.pos.z += (p.life * p.flutter * 1.7).cos() * 0.4 * dt;
            } else {
                p.vel.y -= 9.0 * dt;
            }
            p.pos += p.vel * dt;
        }
        self.particles.retain(|p| p.life < p.max_life);
    }

    /// Build transparent billboard-ish cube geometry for all particles.
    pub fn build_geometry(&self, white_layer: u32) -> (Vec<Vertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        for p in &self.particles {
            let fade = (1.0 - p.life / p.max_life).clamp(0.0, 1.0);
            let s = p.size;
            let c = [
                p.color.r as f32 / 255.0,
                p.color.g as f32 / 255.0,
                p.color.b as f32 / 255.0,
                fade,
            ];
            append_quad_cube(&mut verts, &mut indices, p.pos, s, c, white_layer);
        }
        (verts, indices)
    }
}

/// A minimal 6-face cube centred at `pos`, all faces the same colour.
fn append_quad_cube(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    pos: Vec3,
    s: f32,
    color: [f32; 4],
    layer: u32,
) {
    let faces: [([f32; 3], [Vec3; 4]); 6] = [
        ([1.0, 0.0, 0.0], [Vec3::new(s, -s, -s), Vec3::new(s, -s, s), Vec3::new(s, s, s), Vec3::new(s, s, -s)]),
        ([-1.0, 0.0, 0.0], [Vec3::new(-s, -s, s), Vec3::new(-s, -s, -s), Vec3::new(-s, s, -s), Vec3::new(-s, s, s)]),
        ([0.0, 1.0, 0.0], [Vec3::new(-s, s, -s), Vec3::new(s, s, -s), Vec3::new(s, s, s), Vec3::new(-s, s, s)]),
        ([0.0, -1.0, 0.0], [Vec3::new(-s, -s, s), Vec3::new(s, -s, s), Vec3::new(s, -s, -s), Vec3::new(-s, -s, -s)]),
        ([0.0, 0.0, 1.0], [Vec3::new(s, -s, s), Vec3::new(-s, -s, s), Vec3::new(-s, s, s), Vec3::new(s, s, s)]),
        ([0.0, 0.0, -1.0], [Vec3::new(-s, -s, -s), Vec3::new(s, -s, -s), Vec3::new(s, s, -s), Vec3::new(-s, s, -s)]),
    ];
    for (normal, corners) in faces {
        let base = verts.len() as u32;
        for c in corners {
            verts.push(Vertex {
                position: [pos.x + c.x, pos.y + c.y, pos.z + c.z],
                normal,
                color,
                uv: [0.0, 0.0],
                layer,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn petals_spawn_and_expire() {
        let mut ps = ParticleSystem::new(1);
        for _ in 0..20 {
            ps.emit_petals(Vec3::new(0.0, 64.0, 0.0), 1.0 / 30.0);
            ps.update(1.0 / 30.0);
        }
        assert!(ps.count() > 0, "petals should be alive");
        // Run well past their lifetime; all should expire.
        for _ in 0..400 {
            ps.update(1.0 / 30.0);
        }
        assert_eq!(ps.count(), 0, "all particles should have expired");
    }

    #[test]
    fn burst_respects_cap() {
        let mut ps = ParticleSystem::new(2);
        for _ in 0..500 {
            ps.burst(Vec3::ZERO, Color::rgb(200, 100, 100), 20);
        }
        assert!(ps.count() <= 600, "particle cap exceeded: {}", ps.count());
    }

    #[test]
    fn geometry_indices_are_valid() {
        let mut ps = ParticleSystem::new(3);
        ps.burst(Vec3::ZERO, Color::rgb(120, 200, 120), 10);
        let (v, i) = ps.build_geometry(7);
        assert!(!v.is_empty());
        assert!(i.iter().all(|&idx| (idx as usize) < v.len()));
    }
}
