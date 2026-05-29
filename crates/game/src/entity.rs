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
    /// A friendly sentient cat NPC you can talk to.
    Friend,
}

/// Dialogue data for a friendly cat NPC.
#[derive(Clone)]
pub struct Npc {
    pub name: &'static str,
    pub lines: &'static [&'static str],
    pub line: usize,
}

/// The cast of friendly cats and their cosy chatter.
const FRIENDS: &[(&str, &[&str])] = &[
    (
        "MITTENS",
        &[
            "OH, HELLO TRAVELLER!",
            "THE BLOSSOMS ARE LOVELY TODAY.",
            "MIND THE PUDDLES, HEE HEE!",
        ],
    ),
    (
        "BISCUIT",
        &[
            "PURR... WELCOME TO THE VALLEY.",
            "I AM BUILDING A COSY HOME NEARBY.",
            "HAVE YOU TRIED THE SWEET BERRIES?",
        ],
    ),
    (
        "CLOVER",
        &[
            "MEOW! LOST AGAIN, FRIEND?",
            "LANTERNS KEEP THE NIGHT FRIENDLY.",
            "SAFE TRAVELS, LITTLE ONE.",
        ],
    ),
];

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
    /// Dialogue, present only for [`EntityKind::Friend`].
    pub npc: Option<Npc>,
}

impl Entity {
    fn half_extents(&self) -> (f32, f32) {
        match self.kind {
            EntityKind::Critter => (0.32, 0.5),
            EntityKind::Friend => (0.3, 0.5),
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

    fn try_spawn(
        &mut self,
        player: Vec3,
        manager: &ChunkManager,
        is_night: bool,
    ) -> Option<Entity> {
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

        // Choose a species appropriate to the time of day. A friendly cat NPC
        // appears occasionally regardless of the hour.
        let roll = self.rng.next_f32();
        let kind = if roll < 0.1 {
            EntityKind::Friend
        } else if is_night {
            if roll < 0.6 {
                EntityKind::Firefly
            } else {
                EntityKind::Critter
            }
        } else if roll < 0.55 {
            EntityKind::Butterfly
        } else {
            EntityKind::Critter
        };

        let base = Vec3::new(x, surface as f32 + 1.0, z);
        let home = match kind {
            EntityKind::Critter | EntityKind::Friend => base,
            // Flyers hover a few blocks above the ground.
            _ => base + Vec3::new(0.0, 2.0 + self.rng.next_f32() * 3.0, 0.0),
        };
        let npc = if kind == EntityKind::Friend {
            let (name, lines) = FRIENDS[(self.rng.next_u64() as usize) % FRIENDS.len()];
            Some(Npc {
                name,
                lines,
                line: 0,
            })
        } else {
            None
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
            npc,
        })
    }

    /// Remove all entities of `kind` within `radius` of `pos`, returning how
    /// many were collected (used for the firefly-catching minigame).
    pub fn collect_near(&mut self, pos: Vec3, radius: f32, kind: EntityKind) -> u32 {
        let r2 = radius * radius;
        let before = self.entities.len();
        self.entities
            .retain(|e| !(e.kind == kind && e.position.distance_squared(pos) <= r2));
        (before - self.entities.len()) as u32
    }

    /// Find the friendly cat the player is looking at (within reach) and return
    /// its current dialogue line, advancing to the next line for the next chat.
    pub fn talk_to(&mut self, eye: Vec3, look: Vec3) -> Option<(String, String)> {
        let mut best: Option<(usize, f32)> = None;
        for (idx, e) in self.entities.iter().enumerate() {
            if e.kind != EntityKind::Friend {
                continue;
            }
            let to = (e.position + Vec3::new(0.0, 0.4, 0.0)) - eye;
            let dist = to.length();
            if !(0.01..=5.0).contains(&dist) {
                continue;
            }
            let align = to.normalize().dot(look);
            if align < 0.9 {
                continue;
            }
            // Prefer the most centred (best-aligned) friend.
            if best.map(|(_, a)| align > a).unwrap_or(true) {
                best = Some((idx, align));
            }
        }
        let (idx, _) = best?;
        let e = &mut self.entities[idx];
        let npc = e.npc.as_mut()?;
        let line = npc.lines[npc.line % npc.lines.len()].to_string();
        let name = npc.name.to_string();
        npc.line = npc.line.wrapping_add(1);
        Some((name, line))
    }

    /// Build renderable cube geometry for all entities (world-space). Entities
    /// are untextured, so they sample the solid `white_layer` tile and rely on
    /// their baked vertex colour.
    pub fn build_geometry(&self, white_layer: u32) -> (Vec<Vertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        for e in &self.entities {
            append_entity(&mut verts, &mut indices, e, white_layer);
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
        EntityKind::Critter | EntityKind::Friend => {
            let is_friend = e.kind == EntityKind::Friend;
            if e.decision_timer <= 0.0 {
                // Pick a new wander heading and schedule the next decision.
                let theta = rng.next_f32() * std::f32::consts::TAU;
                e.heading = Vec3::new(theta.cos(), 0.0, theta.sin());
                e.yaw = theta;
                e.decision_timer = 1.5 + rng.next_f32() * 2.5;
                // Critters occasionally hop; the dignified cats do not.
                if !is_friend && e.on_ground && rng.next_f32() < 0.4 {
                    e.velocity.y = 6.0;
                    e.on_ground = false;
                }
            }
            // Amble in the current heading (friends stroll more slowly).
            let speed = if is_friend { 0.9 } else { 1.6 };
            e.velocity.x = e.heading.x * speed;
            e.velocity.z = e.heading.z * speed;
            e.velocity.y -= GRAVITY * dt;

            let (resolved, flags) = move_and_collide(e.aabb(), e.velocity * dt, solid);
            let (hw, _) = e.half_extents();
            e.position = Vec3::new(resolved.min.x + hw, resolved.min.y, resolved.min.z + hw);
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
                let reach = if e.kind == EntityKind::Firefly {
                    3.0
                } else {
                    5.0
                };
                e.home += Vec3::new(theta.cos(), 0.0, theta.sin()) * reach * (rng.next_f32());
                e.decision_timer = 1.0 + rng.next_f32() * 2.0;
                // Keep the home anchored a sensible height above the terrain.
                let surface =
                    manager.surface_height(e.home.x.floor() as i32, e.home.z.floor() as i32);
                let min_y = surface as f32 + 2.0;
                if e.home.y < min_y {
                    e.home.y = min_y;
                }
            }
            let speed = if e.kind == EntityKind::Firefly {
                0.6
            } else {
                1.1
            };
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
fn append_box(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    min: Vec3,
    max: Vec3,
    color: Color,
    layer: u32,
) {
    // 6 faces: (normal, 4 corners CCW from outside).
    let faces: [([f32; 3], [Vec3; 4]); 6] = [
        // +X
        (
            [1.0, 0.0, 0.0],
            [
                Vec3::new(max.x, min.y, min.z),
                Vec3::new(max.x, min.y, max.z),
                Vec3::new(max.x, max.y, max.z),
                Vec3::new(max.x, max.y, min.z),
            ],
        ),
        // -X
        (
            [-1.0, 0.0, 0.0],
            [
                Vec3::new(min.x, min.y, max.z),
                Vec3::new(min.x, min.y, min.z),
                Vec3::new(min.x, max.y, min.z),
                Vec3::new(min.x, max.y, max.z),
            ],
        ),
        // +Y (top)
        (
            [0.0, 1.0, 0.0],
            [
                Vec3::new(min.x, max.y, min.z),
                Vec3::new(max.x, max.y, min.z),
                Vec3::new(max.x, max.y, max.z),
                Vec3::new(min.x, max.y, max.z),
            ],
        ),
        // -Y (bottom)
        (
            [0.0, -1.0, 0.0],
            [
                Vec3::new(min.x, min.y, max.z),
                Vec3::new(max.x, min.y, max.z),
                Vec3::new(max.x, min.y, min.z),
                Vec3::new(min.x, min.y, min.z),
            ],
        ),
        // +Z
        (
            [0.0, 0.0, 1.0],
            [
                Vec3::new(max.x, min.y, max.z),
                Vec3::new(min.x, min.y, max.z),
                Vec3::new(min.x, max.y, max.z),
                Vec3::new(max.x, max.y, max.z),
            ],
        ),
        // -Z
        (
            [0.0, 0.0, -1.0],
            [
                Vec3::new(min.x, min.y, min.z),
                Vec3::new(max.x, min.y, min.z),
                Vec3::new(max.x, max.y, min.z),
                Vec3::new(min.x, max.y, min.z),
            ],
        ),
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
                uv: [0.0, 0.0],
                layer,
                // Critters are lit by daylight with a small floor so they stay
                // visible at night.
                light: [1.0, 0.25],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Build the little body of an entity from a few boxes. Kept blocky and cute.
fn append_entity(verts: &mut Vec<Vertex>, indices: &mut Vec<u32>, e: &Entity, layer: u32) {
    let p = e.position;
    match e.kind {
        EntityKind::Critter => {
            // Cream body + two ears + a pink nose dot.
            let body_col = Color::rgb(244, 226, 198);
            let ear_col = Color::rgb(232, 196, 168);
            let body_min = p + Vec3::new(-0.3, 0.0, -0.3);
            let body_max = p + Vec3::new(0.3, 0.55, 0.3);
            append_box(verts, indices, body_min, body_max, body_col, layer);
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
                    layer,
                );
            }
        }
        EntityKind::Butterfly => {
            // Two pastel wings flapping with the animation phase.
            let flap = (e.phase * 8.0).sin() * 0.12;
            let wing_col = Color::rgb(246, 188, 222);
            let body_col = Color::rgb(120, 96, 110);
            append_box(
                verts,
                indices,
                p + Vec3::new(-0.04, 0.0, -0.04),
                p + Vec3::new(0.04, 0.16, 0.04),
                body_col,
                layer,
            );
            append_box(
                verts,
                indices,
                p + Vec3::new(-0.26, 0.05 + flap, -0.02),
                p + Vec3::new(-0.04, 0.2 + flap, 0.02),
                wing_col,
                layer,
            );
            append_box(
                verts,
                indices,
                p + Vec3::new(0.04, 0.05 + flap, -0.02),
                p + Vec3::new(0.26, 0.2 + flap, 0.02),
                wing_col,
                layer,
            );
        }
        EntityKind::Firefly => {
            // Tiny warm glowing mote (bright colour stands in for emission).
            let glow = Color::rgb(255, 244, 170);
            append_box(
                verts,
                indices,
                p + Vec3::new(-0.09, 0.0, -0.09),
                p + Vec3::new(0.09, 0.18, 0.09),
                glow,
                layer,
            );
        }
        EntityKind::Friend => {
            // A soft lilac-grey tabby.
            append_cat(
                verts,
                indices,
                p,
                e.yaw,
                Color::rgb(176, 168, 196),
                Color::rgb(150, 142, 172),
                layer,
            );
        }
    }
}

/// Build a sit-up cat: body, head, two ears and a curled tail, facing `yaw`.
/// Shared by friendly NPCs and the player's own hero cat.
pub fn append_cat(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    p: Vec3,
    yaw: f32,
    fur: Color,
    ear: Color,
    layer: u32,
) {
    let (sy, cy) = yaw.sin_cos();
    let fwd = Vec3::new(sy, 0.0, cy);
    let right = Vec3::new(cy, 0.0, -sy);
    // Body.
    append_box(
        verts,
        indices,
        p + Vec3::new(-0.22, 0.0, -0.22),
        p + Vec3::new(0.22, 0.42, 0.22),
        fur,
        layer,
    );
    // Head, forward and up.
    let head = p + Vec3::new(0.0, 0.42, 0.0) + fwd * 0.06;
    append_box(
        verts,
        indices,
        head + Vec3::new(-0.18, 0.0, -0.18),
        head + Vec3::new(0.18, 0.34, 0.18),
        fur,
        layer,
    );
    // Ears.
    for side in [-0.12f32, 0.12] {
        let base = head + Vec3::new(0.0, 0.34, 0.0) + right * side + fwd * 0.02;
        append_box(
            verts,
            indices,
            base + Vec3::new(-0.06, 0.0, -0.06),
            base + Vec3::new(0.06, 0.12, 0.06),
            ear,
            layer,
        );
    }
    // Curled tail at the back.
    let tail = p - fwd * 0.24 + Vec3::new(0.0, 0.1, 0.0);
    append_box(
        verts,
        indices,
        tail + Vec3::new(-0.06, 0.0, -0.06),
        tail + Vec3::new(0.06, 0.3, 0.06),
        ear,
        layer,
    );
}

/// Geometry for the player's own cat (a warm ginger tabby) at `pos`/`yaw`.
pub fn player_cat_geometry(pos: Vec3, yaw: f32, layer: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut v = Vec::new();
    let mut i = Vec::new();
    append_cat(
        &mut v,
        &mut i,
        pos,
        yaw,
        Color::rgb(236, 158, 92),
        Color::rgb(210, 130, 70),
        layer,
    );
    (v, i)
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
        let mut em = EntityManager::new(
            1,
            EntityConfig {
                max_entities: 10,
                spawn_attempts_per_update: 8,
                ..Default::default()
            },
        );
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
                let surface = manager
                    .surface_height(e.position.x.floor() as i32, e.position.z.floor() as i32);
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
    fn collect_near_removes_matching_in_radius() {
        let manager = settled(2024);
        let mut em = EntityManager::new(9, EntityConfig::default());
        // Hand-place a few fireflies and a critter near the origin.
        let here = Vec3::new(0.0, 64.0, 0.0);
        for _ in 0..3 {
            em.entities.push(Entity {
                kind: EntityKind::Firefly,
                position: here,
                velocity: Vec3::ZERO,
                yaw: 0.0,
                on_ground: false,
                decision_timer: 0.0,
                heading: Vec3::X,
                phase: 0.0,
                home: here,
                npc: None,
            });
        }
        let _ = &manager;
        let before = em.count();
        let caught = em.collect_near(here, 1.5, EntityKind::Firefly);
        assert_eq!(caught, 3);
        assert_eq!(em.count(), before - 3);
        // A firefly far away is untouched.
        em.entities.push(Entity {
            kind: EntityKind::Firefly,
            position: Vec3::new(100.0, 64.0, 0.0),
            velocity: Vec3::ZERO,
            yaw: 0.0,
            on_ground: false,
            decision_timer: 0.0,
            heading: Vec3::X,
            phase: 0.0,
            home: here,
            npc: None,
        });
        assert_eq!(em.collect_near(here, 1.5, EntityKind::Firefly), 0);
    }

    #[test]
    fn geometry_is_nonempty_with_entities() {
        let manager = settled(2024);
        let mut em = EntityManager::new(
            1,
            EntityConfig {
                spawn_attempts_per_update: 8,
                ..Default::default()
            },
        );
        for _ in 0..20 {
            em.update(1.0 / 30.0, Vec3::new(8.0, 64.0, 8.0), &manager, 1.0);
        }
        let (v, i) = em.build_geometry(99);
        assert!(!v.is_empty() && !i.is_empty());
        // Indices must reference valid vertices.
        assert!(i.iter().all(|&idx| (idx as usize) < v.len()));
    }
}
