//! Mining and building: turning a look ray into block edits, mediated by the
//! inventory so the cat collects what it breaks and spends what it places.

use crate::inventory::Inventory;
use crate::raycast::{self, RayHit};
use crate::streaming::ChunkManager;
use glam::Vec3;
use pixelcraft_core::block::BlockId;
use pixelcraft_physics::Aabb;

/// How far the cat can reach to mine or place.
pub const REACH: f32 = 5.0;

/// Outcome of an interaction attempt, for UI feedback / sound triggers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interaction {
    Mined(BlockId),
    Placed(BlockId),
    /// Nothing in reach, or placement was blocked.
    Nothing,
}

/// Find the block currently targeted by the look ray, if any.
pub fn target(
    mgr: &ChunkManager,
    eye: Vec3,
    look: Vec3,
) -> Option<RayHit> {
    raycast::cast(&mgr.world, &mgr.registry, eye, look, REACH)
}

/// Mine the targeted block: replace it with air and, if harvestable, add it to
/// the inventory.
pub fn mine(
    mgr: &mut ChunkManager,
    inv: &mut Inventory,
    eye: Vec3,
    look: Vec3,
) -> Interaction {
    let Some(hit) = target(mgr, eye, look) else {
        return Interaction::Nothing;
    };
    let id = mgr.world.block_at(hit.block);
    if id.is_air() {
        return Interaction::Nothing;
    }
    let harvestable = mgr.registry.get(id).harvestable;
    if !mgr.set_block(hit.block, BlockId::AIR) {
        return Interaction::Nothing;
    }
    if harvestable {
        inv.add(id, 1);
    }
    Interaction::Mined(id)
}

/// Place the currently selected hotbar block against the targeted face, if the
/// inventory has one and the target cell is empty and not overlapping the
/// player's body.
pub fn place(
    mgr: &mut ChunkManager,
    inv: &mut Inventory,
    eye: Vec3,
    look: Vec3,
    player_box: Aabb,
) -> Interaction {
    let Some(hit) = target(mgr, eye, look) else {
        return Interaction::Nothing;
    };
    let cell = hit.place_pos();
    if !mgr.world.block_at(cell).is_air() {
        return Interaction::Nothing;
    }
    // Don't let the cat trap itself: reject placement intersecting its body.
    let block_box = Aabb::new(
        Vec3::new(cell.x as f32, cell.y as f32, cell.z as f32),
        Vec3::new(cell.x as f32 + 1.0, cell.y as f32 + 1.0, cell.z as f32 + 1.0),
    );
    if player_box.intersects(block_box) {
        return Interaction::Nothing;
    }

    let block = inv.selected_block();
    if !inv.take_one(block) {
        return Interaction::Nothing; // nothing of that type to place
    }
    if !mgr.set_block(cell, block) {
        // Edit failed (chunk unloaded) — refund.
        inv.add(block, 1);
        return Interaction::Nothing;
    }
    Interaction::Placed(block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::StreamConfig;
    use pixelcraft_core::block::blocks;
    use pixelcraft_core::coords::BlockPos;

    fn settled_manager() -> ChunkManager {
        let cfg = StreamConfig {
            view_distance: 1,
            min_chunk_y: 0,
            max_chunk_y: 3,
            unload_margin: 1,
            max_uploads_per_update: 4096,
            max_meshes_per_update: 4096,
            max_jobs_per_update: 4096,
        };
        let mut mgr = ChunkManager::new(2024, cfg, 0);
        for _ in 0..10 {
            mgr.update(Vec3::new(8.0, 80.0, 8.0));
            if mgr.is_settled() {
                break;
            }
        }
        mgr
    }

    #[test]
    fn mine_then_place_roundtrip() {
        let mut mgr = settled_manager();
        let mut inv = Inventory::new(mgr.registry.len());

        // Carve a clear horizontal tunnel through whatever terrain generated, so
        // the look ray isn't blocked by hills, then set a known backstop block.
        let eye = Vec3::new(8.5, 80.5, 8.5);
        for x in 8..=13 {
            mgr.set_block(BlockPos::new(x, 80, 8), BlockId::AIR);
        }
        let target_pos = BlockPos::new(11, 80, 8);
        mgr.set_block(target_pos, blocks::STONE);
        let look = Vec3::new(1.0, 0.0, 0.0);

        let result = mine(&mut mgr, &mut inv, eye, look);
        assert_eq!(result, Interaction::Mined(blocks::STONE));
        assert_eq!(inv.count(blocks::STONE), 1);
        assert!(mgr.world.block_at(target_pos).is_air());

        // Place a fresh backstop two cells out and build against its near face,
        // which should drop the stone into the now-empty target cell.
        mgr.set_block(BlockPos::new(12, 80, 8), blocks::STONE);
        inv.select(2); // HOTBAR[2] == STONE
        assert_eq!(inv.selected_block(), blocks::STONE);
        let player = Aabb::from_feet(Vec3::new(8.5, 79.0, 8.5), 0.3, 0.9);
        let result = place(&mut mgr, &mut inv, eye, look, player);
        assert_eq!(result, Interaction::Placed(blocks::STONE));
        assert_eq!(inv.count(blocks::STONE), 0);
        assert_eq!(mgr.world.block_at(target_pos), blocks::STONE);
    }

    #[test]
    fn cannot_place_without_inventory() {
        let mut mgr = settled_manager();
        let mut inv = Inventory::new(mgr.registry.len());
        let eye = Vec3::new(8.5, 80.5, 8.5);
        mgr.set_block(BlockPos::new(11, 80, 8), blocks::STONE);
        let player = Aabb::from_feet(Vec3::new(8.5, 79.0, 8.5), 0.3, 0.9);
        // Empty inventory → placement does nothing.
        let result = place(&mut mgr, &mut inv, eye, Vec3::new(1.0, 0.0, 0.0), player);
        assert_eq!(result, Interaction::Nothing);
    }

    #[test]
    fn mining_air_is_nothing() {
        let mut mgr = settled_manager();
        let mut inv = Inventory::new(mgr.registry.len());
        // Look straight up into open sky.
        let result = mine(&mut mgr, &mut inv, Vec3::new(8.5, 120.0, 8.5), Vec3::Y);
        assert_eq!(result, Interaction::Nothing);
    }
}
