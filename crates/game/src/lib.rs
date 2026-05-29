//! PixelCraft game logic: the platform-independent glue binding world
//! generation, meshing, physics and player interaction together.
//!
//! Everything here is headless and testable; the rendering front-end lives in
//! the binary (`main.rs`) and consumes these systems.

pub mod camera;
pub mod interaction;
pub mod inventory;
pub mod raycast;
pub mod sampler;
pub mod streaming;

pub use camera::{Camera, Frustum};
pub use interaction::{mine, place, target, Interaction, REACH};
pub use inventory::{Inventory, HOTBAR};
pub use raycast::{cast, RayHit};
pub use streaming::{ChunkManager, StreamConfig, StreamStats};

use glam::Vec3;
use pixelcraft_physics::{MovementInput, Player, SolidQuery};

/// A complete, render-agnostic game session: the streamed world, the cat, and
/// its inventory. The renderer drives this each frame and reads back what it
/// needs to draw.
pub struct Game {
    pub manager: ChunkManager,
    pub player: Player,
    pub inventory: Inventory,
    /// Seconds of accumulated simulation time, for animation/day-night later.
    pub time: f32,
}

impl Game {
    pub fn new(seed: u64, config: StreamConfig, worker_count: usize) -> Self {
        let manager = ChunkManager::new(seed, config, worker_count);
        let inventory = Inventory::new(manager.registry.len());
        // Spawn high in the sky; the player settles onto terrain once the
        // spawn column streams in.
        let spawn = Vec3::new(0.5, 160.0, 0.5);
        let player = Player::new(spawn);
        Self {
            manager,
            player,
            inventory,
            time: 0.0,
        }
    }

    /// Step the whole game forward by `dt` seconds with the given input.
    pub fn update(&mut self, input: MovementInput, dt: f32) {
        self.time += dt;
        // Stream chunks around the player first so the ground exists before we
        // simulate physics against it.
        self.manager.update(self.player.position);
        let solid = crate::sampler::WorldSolid {
            world: &self.manager.world,
            registry: &self.manager.registry,
        };
        self.player.update(input, dt, &solid);
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
