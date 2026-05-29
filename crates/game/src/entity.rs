//! Ambient critters that bring the world to life: ground animals that wander
//! and hop, butterflies that flutter over meadows by day, and fireflies that
//! drift and glow at night.
//!
//! Entities spawn in a ring around the player and despawn when left behind,
//! keeping the active population small and the simulation cheap regardless of
//! how far the world streams. AI is intentionally simple (timed random
//! re-targeting + bobbing) but reads as lively.

use glam::Vec3;
use pixelcraft_core::block::BlockId;
use pixelcraft_core::block::Color;
use pixelcraft_mesh::Vertex;
use pixelcraft_physics::{move_and_collide, Aabb, SolidQuery};
use pixelcraft_worldgen::SplitMix64;

use crate::sampler::WorldSolid;
use crate::streaming::ChunkManager;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EntityKind {
    /// A small round ground critter ("puffkit") that wanders and hops.
    Critter,
    /// A daytime butterfly that flutters above the meadow.
    Butterfly,
    /// A nighttime firefly that drifts and glows.
    Firefly,
}

#[derive(Clone)]
pub struct Entity {
    pub kind: EntityKind,
    pub position: Vec3,
    pub velocity: Vec3,
    pub yaw: f32,
    pub on_ground: bool,
    /// Countdown to the next AI decision (seconds).
    decision_timer: f32,
    /// Current wander heading (unit, horizontal) for ground critters.
    heading: Vec3,
    /// Animation phase for bobbing/fluttering.
    phase: f32,
    /// Anchor point flyers loosely orbit.
    home: Vec3,
}

impl Entity {
    fn half_extents(&self) -> (f32, f32) {
        match self.kind {
            EntityKind::Critter => (0.32, 0.5),
            EntityKind::Butterfly => (0.18, 0.2),
            EntityKind::Firefly => (0.12, 0.12),
        }
    }

    fn aabb(&self) -> Aabb {
        let (hw, h) = self.half_extents();
        Aabb::from_feet(self.position, hw, h)
    }
}

/// Tunables for ambient critter population.
#[derive(Clone, Copy)]
pub struct EntityConfig {
    pub max_entities: usize,
    pub spawn_min_radius: f32,
    pub spawn_max_radius: f32,
    pub despawn_radius: f32,
    /// Spawn attempts per update (rate-limits population growth).
    pub spawn_attempts_per_update: usize,
}

impl Default for EntityConfig {
    fn default() -> Self {
        Self {
            max_entities: 48,
            spawn_min_radius: 12.0,
            spawn_max_radius: 40.0,
            despawn_radius: 64.0,
            spawn_attempts_per_update: 2,
        }
    }
}

pub struct EntityManager {
    pub entities: Vec<Entity>,
    config: EntityConfig,
    rng: SplitMix64,
}

impl EntityManager {
    pub fn new(seed: u64, config: EntityConfig) -> Self {
        Self {
            entities: Vec::new(),
            config,
            rng: SplitMix64::new(seed ^ 0xE471_7900),
        }
    }

    pub fn count(&self) -> usize {
        self.entities.len()
    }

    /// Step all critters, then handle spawning/despawning around the player.
    /// `daylight` in `[0,1]` selects day vs night species.
    pub fn update(&mut self, dt: f32, player: Vec3, manager: &ChunkManager, daylight: f32) {
        let solid = WorldSolid {
            world: &manager.world,
            registry: &manager.registry,
        };

        // Despawn anything that wandered (or got left) too far away.
        let despawn_sq = self.config.despawn_radius * self.config.despawn_radius;
        self.entities
            .retain(|e| e.position.distance_squared(player) <= despawn_sq);

        // Advance AI.
        for e in &mut self.entities {
            update_entity(e, dt, &solid, &mut self.rng, manager);
        }

        // Spawn new ambient life up to the cap.
        let is_night = daylight < 0.35;
        for _ in 0..self.config.spawn_attempts_per_update {
            if self.entities.len() >= self.config.max_entities {
                break;
            }
            if let Some(e) = self.try_spawn(player, manager, is_night) {
                self.entities.push(e);
            }
        }
    }

    fn try_spawn(&mut self, player: Vec3, manager: &ChunkManager, is_night: bool) -> Option<Entity> {
        // Random point in the spawn ring around the player.
        let theta = self.rng.next_f32() * std::f32::consts::TAU;
        let r = self.config.spawn_min_radius
            + self.rng.next_f32() * (self.config.spawn_max_radius - self.config.spawn_min_radius);
        let x = player.x + r * theta.cos();
        let z = player.z + r * theta.sin();
        let xi = x.floor() as i32;
        let zi = z.floor() as i32;
        let surface = manager.surface_height(xi, zi);

        // Skip water and unloaded ground.
        if surface < pixelcraft_worldgen::SEA_LEVEL {
            return None;
        }
        let solid = WorldSolid {
            world: &manager.world,
            registry: &manager.registry,
        };
        if !solid.is_solid(pixelcraft_core::coords::BlockPos::new(xi, surface, zi)) {
            return None; // chunk not loaded here yet
        }

        // Choose a species appropriate to the time of day.
        let roll = self.rng.next_f32();
        let kind = if is_night {
            if roll < 0.6 {
                EntityKind::Firefly
            } else {
                EntityKind::Critter
            }
        } else if roll < 0.5 {
            EntityKind::Butterfly
        } else {
            EntityKind::Critter
        };

        let base = Vec3::new(x, surface as f32 + 1.0, z);
        let home = match kind {
            EntityKind::Critter => base,
            // Flyers hover a few blocks above the ground.
            _ => base + Vec3::new(0.0, 2.0 + self.rng.next_f32() * 3.0, 0.0),
        };
        Some(Entity {
            kind,
            position: home,
            velocity: Vec3::ZERO,
            yaw: theta,
            on_ground: false,
            decision_timer: self.rng.next_f32() * 2.0,
            heading: Vec3::new(theta.cos(), 0.0, theta.sin()),
            phase: self.rng.next_f32() * std::f32::consts::TAU,
            home,
        })
    }

    /// Build renderable cube geometry for all entities (world-space).
    pub fn build_geometry(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        for e in &self.entities {
            append_entity(&mut verts, &mut indices, e);
        }
        (verts, indices)
    }
}

const GRAVITY: f32 = 22.0;

fn update_entity<Q: SolidQuery>(
    e: &mut Entity,
    dt: f32,
    solid: &Q,
    rng: &mut SplitMix64,
    manager: &ChunkManager,
) {
    e.phase += dt;
    e.decision_timer -= dt;

    match e.kind {
        EntityKind::Critter => {
            if e.decision_timer <= 0.0 {
                // Pick a new wander heading and schedule the next decision.
                let theta = rng.next_f32() * std::f32::consts::TAU;
                e.heading = Vec3::new(theta.cos(), 0.0, theta.sin());
                e.yaw = theta;
                e.decision_timer = 1.5 + rng.next_f32() * 2.5;
                // Occasionally hop for a touch of personality.
                if e.on_ground && rng.next_f32() < 0.4 {
                    e.velocity.y = 6.0;
                    e.on_ground = false;
                }
            }
            // Amble in the current heading.
            let speed = 1.6;
            e.velocity.x = e.heading.x * speed;
            e.velocity.z = e.heading.z * speed;
            e.velocity.y -= GRAVITY * dt;

            let (resolved, flags) = move_and_collide(e.aabb(), e.velocity * dt, solid);
            let (hw, _) = e.half_extents();
            e.position = Vec3::new(
                resolved.min.x + hw,
                resolved.min.y,
                resolved.min.z + hw,
            );
            if flags.neg_y || flags.pos_y {
                e.velocity.y = 0.0;
            }
            // Turn around when bumping a wall so critters don't grind into cliffs.
            if flags.hit_wall() {
                e.heading = -e.heading;
                e.yaw += std::f32::consts::PI;
            }
            e.on_ground = flags.on_ground();
        }
        EntityKind::Butterfly | EntityKind::Firefly => {
            // Gentle wandering orbit around `home`, with a bobbing vertical sine.
            if e.decision_timer <= 0.0 {
                let theta = rng.next_f32() * std::f32::consts::TAU;
                let reach = if e.kind == EntityKind::Firefly { 3.0 } else { 5.0 };
                e.home += Vec3::new(theta.cos(), 0.0, theta.sin()) * reach * (rng.next_f32());
                e.decision_timer = 1.0 + rng.next_f32() * 2.0;
                // Keep the home anchored a sensible height above the terrain.
                let surface = manager.surface_height(e.home.x.floor() as i32, e.home.z.floor() as i32);
                let min_y = surface as f32 + 2.0;
                if e.home.y < min_y {
                    e.home.y = min_y;
                }
            }
            let speed = if e.kind == EntityKind::Firefly { 0.6 } else { 1.1 };
            let to_home = e.home - e.position;
            let bob = (e.phase * 2.0).sin() * 0.35;
            let drift = to_home * speed * dt;
            e.position += drift;
            e.position.y += bob * dt;
            e.yaw += dt * speed;
        }
    }
}

// --- Rendering geometry --------------------------------------------------

/// Append an axis-aligned box with simple top-bright face shading.
fn append_box(verts: &mut Vec<Vertex>, indices: &mut Vec<u32>, min: Vec3, max: Vec3, color: Color) {
    // 6 faces: (normal, 4 corners CCW from outside).
    let faces: [([f32; 3], [Vec3; 4]); 6] = [
        // +X
        ([1.0, 0.0, 0.0], [Vec3::new(max.x, min.y, min.z), Vec3::new(max.x, min.y, max.z), Vec3::new(max.x, max.y, max.z), Vec3::new(max.x, max.y, min.z)]),
        // -X
        ([-1.0, 0.0, 0.0], [Vec3::new(min.x, min.y, max.z), Vec3::new(min.x, min.y, min.z), Vec3::new(min.x, max.y, min.z), Vec3::new(min.x, max.y, max.z)]),
        // +Y (top)
        ([0.0, 1.0, 0.0], [Vec3::new(min.x, max.y, min.z), Vec3::new(max.x, max.y, min.z), Vec3::new(max.x, max.y, max.z), Vec3::new(min.x, max.y, max.z)]),
        // -Y (bottom)
        ([0.0, -1.0, 0.0], [Vec3::new(min.x, min.y, max.z), Vec3::new(max.x, min.y, max.z), Vec3::new(max.x, min.y, min.z), Vec3::new(min.x, min.y, min.z)]),
        // +Z
        ([0.0, 0.0, 1.0], [Vec3::new(max.x, min.y, max.z), Vec3::new(min.x, min.y, max.z), Vec3::new(min.x, max.y, max.z), Vec3::new(max.x, max.y, max.z)]),
        // -Z
        ([0.0, 0.0, -1.0], [Vec3::new(min.x, min.y, min.z), Vec3::new(max.x, min.y, min.z), Vec3::new(max.x, max.y, min.z), Vec3::new(min.x, max.y, min.z)]),
    ];
    for (normal, corners) in faces {
        let shade = match normal {
            [_, y, _] if y > 0.5 => 1.0,
            [_, y, _] if y < -0.5 => 0.55,
            [x, _, _] if x.abs() > 0.5 => 0.82,
            _ => 0.7,
        };
        let col = [
            (color.r as f32 / 255.0) * shade,
            (color.g as f32 / 255.0) * shade,
            (color.b as f32 / 255.0) * shade,
            color.a as f32 / 255.0,
        ];
        let base = verts.len() as u32;
        for c in corners {
            verts.push(Vertex {
                position: [c.x, c.y, c.z],
                normal,
                color: col,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Build the little body of an entity from a few boxes. Kept blocky and cute.
fn append_entity(verts: &mut Vec<Vertex>, indices: &mut Vec<u32>, e: &Entity) {
    let p = e.position;
    match e.kind {
        EntityKind::Critter => {
            // Cream body + two ears + a pink nose dot.
            let body_col = Color::rgb(244, 226, 198);
            let ear_col = Color::rgb(232, 196, 168);
            let body_min = p + Vec3::new(-0.3, 0.0, -0.3);
            let body_max = p + Vec3::new(0.3, 0.55, 0.3);
            append_box(verts, indices, body_min, body_max, body_col);
            // Ears (front, based on yaw — approximate with +Z for simplicity).
            let (sy, cy) = e.yaw.sin_cos();
            let fwd = Vec3::new(sy, 0.0, cy) * 0.18;
            for side in [-0.16f32, 0.16] {
                let right = Vec3::new(cy, 0.0, -sy) * side;
                let ear = p + Vec3::new(0.0, 0.55, 0.0) + fwd + right;
                append_box(
                    verts,
                    indices,
                    ear + Vec3::new(-0.06, 0.0, -0.06),
                    ear + Vec3::new(0.06, 0.16, 0.06),
                    ear_col,
                );
            }
        }
        EntityKind::Butterfly => {
            // Two pastel wings flapping with the animation phase.
            let flap = (e.phase * 8.0).sin() * 0.12;
            let wing_col = Color::rgb(246, 188, 222);
            let body_col = Color::rgb(120, 96, 110);
            append_box(verts, indices, p + Vec3::new(-0.04, 0.0, -0.04), p + Vec3::new(0.04, 0.16, 0.04), body_col);
            append_box(verts, indices, p + Vec3::new(-0.26, 0.05 + flap, -0.02), p + Vec3::new(-0.04, 0.2 + flap, 0.02), wing_col);
            append_box(verts, indices, p + Vec3::new(0.04, 0.05 + flap, -0.02), p + Vec3::new(0.26, 0.2 + flap, 0.02), wing_col);
        }
        EntityKind::Firefly => {
            // Tiny warm glowing mote (bright colour stands in for emission).
            let glow = Color::rgb(255, 244, 170);
            append_box(verts, indices, p + Vec3::new(-0.09, 0.0, -0.09), p + Vec3::new(0.09, 0.18, 0.09), glow);
        }
    }
}

/// Air block id, re-exported for clarity in spawn checks.
pub const AIR: BlockId = BlockId::AIR;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::StreamConfig;

    fn settled(seed: u64) -> ChunkManager {
        let cfg = StreamConfig {
            view_distance: 4,
            min_chunk_y: 0,
            max_chunk_y: 4,
            unload_margin: 1,
            max_uploads_per_update: 4096,
            max_meshes_per_update: 4096,
            max_jobs_per_update: 4096,
        };
        let mut m = ChunkManager::new(seed, cfg, 0);
        for _ in 0..12 {
            m.update(Vec3::new(8.0, 80.0, 8.0));
            if m.is_settled() {
                break;
            }
        }
        m
    }

    #[test]
    fn population_respects_cap() {
        let manager = settled(2024);
        let mut em = EntityManager::new(1, EntityConfig { max_entities: 10, spawn_attempts_per_update: 8, ..Default::default() });
        for _ in 0..50 {
            em.update(1.0 / 30.0, Vec3::new(8.0, 64.0, 8.0), &manager, 1.0);
        }
        assert!(em.count() <= 10, "exceeded cap: {}", em.count());
        assert!(em.count() > 0, "nothing spawned on settled land");
    }

    #[test]
    fn critters_stay_above_ground() {
        let manager = settled(7);
        let mut em = EntityManager::new(3, EntityConfig::default());
        let player = Vec3::new(8.0, 64.0, 8.0);
        for _ in 0..200 {
            em.update(1.0 / 30.0, player, &manager, 1.0);
        }
        for e in &em.entities {
            if e.kind == EntityKind::Critter {
                let surface = manager.surface_height(e.position.x.floor() as i32, e.position.z.floor() as i32);
                // Allow a little slack but critters must not sink far underground.
                assert!(
                    e.position.y > surface as f32 - 2.0,
                    "critter sank below ground: y {} surface {surface}",
                    e.position.y
                );
            }
        }
    }

    #[test]
    fn distant_entities_despawn_when_player_leaves() {
        let manager = settled(11);
        let mut em = EntityManager::new(5, EntityConfig::default());
        let player = Vec3::new(8.0, 64.0, 8.0);
        for _ in 0..40 {
            em.update(1.0 / 30.0, player, &manager, 1.0);
        }
        assert!(em.count() > 0);
        // Teleport the player far away; entities should be culled.
        let far = Vec3::new(8.0 + 500.0, 64.0, 8.0);
        em.update(1.0 / 30.0, far, &manager, 1.0);
        for e in &em.entities {
            assert!(e.position.distance(far) <= em.config.despawn_radius + 1.0);
        }
    }

    #[test]
    fn geometry_is_nonempty_with_entities() {
        let manager = settled(2024);
        let mut em = EntityManager::new(1, EntityConfig { spawn_attempts_per_update: 8, ..Default::default() });
        for _ in 0..20 {
            em.update(1.0 / 30.0, Vec3::new(8.0, 64.0, 8.0), &manager, 1.0);
        }
        let (v, i) = em.build_geometry();
        assert!(!v.is_empty() && !i.is_empty());
        // Indices must reference valid vertices.
        assert!(i.iter().all(|&idx| (idx as usize) < v.len()));
    }
}
