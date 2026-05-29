//! PixelCraft game logic: the platform-independent glue binding world
//! generation, meshing, physics and player interaction together.
//!
//! Everything here is headless and testable; the rendering front-end lives in
//! the binary (`main.rs`) and consumes these systems.

pub mod camera;
pub mod entity;
pub mod environment;
pub mod interaction;
pub mod inventory;
pub mod particle;
pub mod persistence;
pub mod quest;
pub mod raycast;
pub mod sampler;
pub mod streaming;

#[cfg(any(feature = "render", feature = "capture"))]
pub mod render;

pub use camera::{Camera, Frustum};
pub use entity::{EntityConfig, EntityKind, EntityManager};
pub use environment::Environment;
pub use interaction::{mine, place, target, Interaction, REACH};
pub use inventory::{Inventory, HOTBAR};
pub use quest::{Objective, Quest, QuestLog};
pub use raycast::{cast, RayHit};
pub use streaming::{ChunkManager, StreamConfig, StreamStats};

use glam::Vec3;
use pixelcraft_physics::{MovementInput, Player, SolidQuery};

/// Current weather. Snowy peaks always snow; elsewhere it's mostly clear with
/// occasional gentle rain.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Weather {
    Clear,
    Rain,
    Snow,
}

impl Weather {
    pub fn label(self) -> &'static str {
        match self {
            Weather::Clear => "CLEAR",
            Weather::Rain => "RAIN",
            Weather::Snow => "SNOW",
        }
    }
}

/// Decide the weather for a moment in time at a biome. Rain comes in slow
/// ~45-second spells; cold peaks always snow.
fn weather_at(time: f32, biome: pixelcraft_worldgen::Biome) -> Weather {
    if biome == pixelcraft_worldgen::Biome::SnowyPeaks {
        return Weather::Snow;
    }
    let period = (time / 45.0).max(0.0) as u64;
    // ~25% of spells are rainy.
    if pixelcraft_worldgen::SplitMix64::new(period ^ 0xBEEF_CAFE).next_f32() < 0.25 {
        Weather::Rain
    } else {
        Weather::Clear
    }
}

/// A complete, render-agnostic game session: the streamed world, the cat, its
/// inventory, ambient critters and the day/night environment. The renderer
/// drives this each frame and reads back what it needs to draw.
pub struct Game {
    pub manager: ChunkManager,
    pub player: Player,
    pub inventory: Inventory,
    pub entities: EntityManager,
    pub environment: Environment,
    pub quests: QuestLog,
    pub particles: particle::ParticleSystem,
    pub weather: Weather,
    /// Active NPC dialogue (speaker, line), shown for a few seconds.
    pub active_dialogue: Option<(String, String)>,
    dialogue_timer: f32,
    /// Seconds of accumulated simulation time.
    pub time: f32,
}

impl Game {
    pub fn new(seed: u64, config: StreamConfig, worker_count: usize) -> Self {
        let manager = ChunkManager::new(seed, config, worker_count);
        let inventory = Inventory::new(manager.registry.len());
        // Find dry land near the origin so the cat doesn't spawn in the ocean,
        // then drop in from just above that surface.
        let (sx, sz, sh) = find_land_spawn(&manager);
        let spawn = Vec3::new(sx as f32 + 0.5, sh as f32 + 3.0, sz as f32 + 0.5);
        let player = Player::new(spawn);
        Self {
            manager,
            player,
            inventory,
            entities: EntityManager::new(seed, EntityConfig::default()),
            environment: Environment::default(),
            quests: QuestLog::cosy_chain(),
            particles: particle::ParticleSystem::new(seed),
            weather: Weather::Clear,
            active_dialogue: None,
            dialogue_timer: 0.0,
            time: 0.0,
        }
    }

    /// Talk to the friendly cat the player is looking at, if any.
    pub fn talk(&mut self) {
        let eye = self.player.eye();
        let look = self.player.look_dir();
        if let Some(line) = self.entities.talk_to(eye, look) {
            self.active_dialogue = Some(line);
            self.dialogue_timer = 6.0;
        }
    }

    /// Mine the block the cat is looking at, crediting the inventory and quest
    /// log (and applying any quest reward). Returns the interaction outcome.
    pub fn do_mine(&mut self, eye: Vec3, look: Vec3) -> Interaction {
        // Capture the target before mining so we can spawn debris at it.
        let hit = interaction::target(&self.manager, eye, look);
        let result = interaction::mine(&mut self.manager, &mut self.inventory, eye, look);
        if let Interaction::Mined(id) = result {
            if let Some(h) = hit {
                let color = self.manager.registry.get(id).color;
                let center = Vec3::new(
                    h.block.x as f32 + 0.5,
                    h.block.y as f32 + 0.5,
                    h.block.z as f32 + 0.5,
                );
                self.particles.burst(center, color, 14);
            }
            let reward = self.quests.on_collect(id, 1);
            self.grant(&reward);
        }
        result
    }

    /// Place the selected block, crediting the quest log on success.
    pub fn do_place(&mut self, eye: Vec3, look: Vec3) -> Interaction {
        let body = self.player.aabb();
        let result = interaction::place(&mut self.manager, &mut self.inventory, eye, look, body);
        if let Interaction::Placed(id) = result {
            let reward = self.quests.on_place(id);
            self.grant(&reward);
        }
        result
    }

    fn grant(&mut self, reward: &[(pixelcraft_core::block::BlockId, u32)]) {
        for &(id, n) in reward {
            self.inventory.add(id, n);
        }
    }

    /// Step the whole game forward by `dt` seconds with the given input.
    pub fn update(&mut self, input: MovementInput, dt: f32) {
        self.time += dt;
        self.environment.advance(dt);
        // Stream chunks around the player first so the ground exists before we
        // simulate physics against it.
        self.manager.update(self.player.position);
        let solid = crate::sampler::WorldSolid {
            world: &self.manager.world,
            registry: &self.manager.registry,
        };
        self.player.update(input, dt, &solid);
        // Ambient critters spawn/wander around the player.
        self.entities.update(
            dt,
            self.player.position,
            &self.manager,
            self.environment.daylight(),
        );
        // Weather + cherry-blossom petals, by biome.
        let biome = self.manager.biome_at(
            self.player.position.x.floor() as i32,
            self.player.position.z.floor() as i32,
        );
        self.weather = weather_at(self.time, biome);
        match self.weather {
            Weather::Snow => self.particles.emit_weather(self.player.position, true, dt),
            Weather::Rain => self.particles.emit_weather(self.player.position, false, dt),
            Weather::Clear => {}
        }
        if biome == pixelcraft_worldgen::Biome::CherryGrove {
            self.particles.emit_petals(self.player.position, dt);
        }
        self.particles.update(dt);

        // Fade out any active dialogue.
        if self.dialogue_timer > 0.0 {
            self.dialogue_timer -= dt;
            if self.dialogue_timer <= 0.0 {
                self.active_dialogue = None;
            }
        }
    }

    /// Is the spawn area fully streamed in (so the player won't fall through
    /// not-yet-generated ground)?
    pub fn world_ready_under_player(&self) -> bool {
        let solid = crate::sampler::WorldSolid {
            world: &self.manager.world,
            registry: &self.manager.registry,
        };
        // Some solid ground exists somewhere below the player.
        let px = self.player.position.x.floor() as i32;
        let pz = self.player.position.z.floor() as i32;
        (0..=self.player.position.y.ceil() as i32)
            .rev()
            .any(|y| solid.is_solid(pixelcraft_core::coords::BlockPos::new(px, y, pz)))
    }
}

/// Spiral outward from the origin to find a column whose surface is above sea
/// level (i.e. dry land), returning `(x, z, surface_height)`. Falls back to the
/// origin if nothing is found within the search radius.
fn find_land_spawn(manager: &ChunkManager) -> (i32, i32, i32) {
    use pixelcraft_worldgen::SEA_LEVEL;
    let mut best = (0, 0, manager.surface_height(0, 0));
    // Sample on a grid spiralling out; we want land a little above the waterline
    // so the spawn feels like a proper beach/meadow, not a puddle. Ocean-heavy
    // worlds can push land far from the origin, so search a wide radius — it's
    // only a few thousand cheap noise evaluations.
    for radius in (0..=8000).step_by(12) {
        // More angular samples as the ring grows, to keep spatial coverage even.
        let angles = (16 + radius / 24).min(96);
        for angle_step in 0..angles {
            let theta = angle_step as f32 / angles as f32 * std::f32::consts::TAU;
            let x = (radius as f32 * theta.cos()) as i32;
            let z = (radius as f32 * theta.sin()) as i32;
            let h = manager.surface_height(x, z);
            if h >= SEA_LEVEL + 2 {
                return (x, z, h);
            }
            if h > best.2 {
                best = (x, z, h);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> StreamConfig {
        StreamConfig {
            view_distance: 2,
            min_chunk_y: 0,
            max_chunk_y: 5,
            unload_margin: 1,
            max_uploads_per_update: 4096,
            max_meshes_per_update: 4096,
            max_jobs_per_update: 4096,
        }
    }

    #[test]
    fn player_settles_onto_streamed_terrain() {
        let mut game = Game::new(2024, test_config(), 0);
        // Run a few seconds of simulation; the cat should fall and land.
        for _ in 0..600 {
            game.update(MovementInput::default(), 1.0 / 60.0);
            if game.player.on_ground {
                break;
            }
        }
        assert!(game.player.on_ground, "cat never landed on terrain");
        // It should have come to rest above sea level-ish, not at the void.
        assert!(game.player.position.y > 1.0);
        assert!(game.world_ready_under_player());
    }

    #[test]
    fn spawn_is_on_dry_land() {
        use pixelcraft_worldgen::SEA_LEVEL;
        // Across several seeds, the chosen spawn column should be above water.
        for seed in [1u64, 2, 3, 42, 777, 0xCA75_C0DE] {
            let game = Game::new(seed, test_config(), 0);
            let h = game
                .manager
                .surface_height(game.player.position.x as i32, game.player.position.z as i32);
            assert!(
                h >= SEA_LEVEL,
                "seed {seed}: spawned underwater (surface {h} < sea {SEA_LEVEL})"
            );
        }
    }

    #[test]
    fn game_streams_and_settles() {
        let mut game = Game::new(7, test_config(), 0);
        for _ in 0..200 {
            game.update(MovementInput::default(), 1.0 / 60.0);
            if game.manager.is_settled() {
                break;
            }
        }
        assert!(game.manager.is_settled());
        assert!(game.manager.mesh_count() > 0);
    }
}
