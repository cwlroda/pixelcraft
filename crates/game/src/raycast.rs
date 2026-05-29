//! Voxel ray casting via the Amanatides–Woo grid traversal algorithm.
//!
//! Used to find which block the cat is looking at (for mining) and which empty
//! cell a new block would be placed into (the face the ray entered through).

use glam::{IVec3, Vec3};
use pixelcraft_core::block::BlockRegistry;
use pixelcraft_core::coords::BlockPos;
use pixelcraft_core::world::World;

/// A successful ray hit against a non-air block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RayHit {
    /// The solid block that was hit.
    pub block: BlockPos,
    /// The face normal of the entry face — the adjacent empty cell is
    /// `block + normal`, which is where placement happens.
    pub normal: IVec3,
}

impl RayHit {
    /// Cell adjacent to the hit face (placement target).
    pub fn place_pos(&self) -> BlockPos {
        BlockPos::new(
            self.block.x + self.normal.x,
            self.block.y + self.normal.y,
            self.block.z + self.normal.z,
        )
    }
}

/// Cast a ray from `origin` along `dir` (need not be normalised) up to
/// `max_dist` world units. Returns the first block the registry marks as a hit
/// target (anything that is visible — so the cat can also target flowers and
/// leaves, not just solids).
pub fn cast(
    world: &World,
    registry: &BlockRegistry,
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
) -> Option<RayHit> {
    cast_filtered(world, registry, origin, dir, max_dist, |id| {
        registry.get(id).is_visible()
    })
}

/// Generalised cast: `hit_if` decides which block ids count as a hit.
pub fn cast_filtered(
    world: &World,
    _registry: &BlockRegistry,
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    hit_if: impl Fn(pixelcraft_core::block::BlockId) -> bool,
) -> Option<RayHit> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }

    // Current voxel.
    let mut voxel = IVec3::new(
        origin.x.floor() as i32,
        origin.y.floor() as i32,
        origin.z.floor() as i32,
    );

    // Step direction per axis.
    let step = IVec3::new(
        dir.x.signum() as i32,
        dir.y.signum() as i32,
        dir.z.signum() as i32,
    );

    // Distance (in t) to cross one voxel along each axis.
    let inv = Vec3::new(
        if dir.x != 0.0 { 1.0 / dir.x.abs() } else { f32::INFINITY },
        if dir.y != 0.0 { 1.0 / dir.y.abs() } else { f32::INFINITY },
        if dir.z != 0.0 { 1.0 / dir.z.abs() } else { f32::INFINITY },
    );

    // Distance (in t) to the first voxel boundary on each axis.
    let mut t_max = Vec3::new(
        boundary_dist(origin.x, dir.x),
        boundary_dist(origin.y, dir.y),
        boundary_dist(origin.z, dir.z),
    );

    let mut last_normal = IVec3::ZERO;
    let mut t = 0.0f32;

    // Bound the iteration count to avoid pathological loops.
    let max_steps = (max_dist.ceil() as i32 + 1) * 3;
    for _ in 0..max_steps {
        let id = world.block_at(BlockPos::new(voxel.x, voxel.y, voxel.z));
        if !id.is_air() && hit_if(id) {
            return Some(RayHit {
                block: BlockPos::new(voxel.x, voxel.y, voxel.z),
                normal: last_normal,
            });
        }

        // Advance to the next voxel along whichever axis has the nearest boundary.
        if t_max.x < t_max.y && t_max.x < t_max.z {
            voxel.x += step.x;
            t = t_max.x;
            t_max.x += inv.x;
            last_normal = IVec3::new(-step.x, 0, 0);
        } else if t_max.y < t_max.z {
            voxel.y += step.y;
            t = t_max.y;
            t_max.y += inv.y;
            last_normal = IVec3::new(0, -step.y, 0);
        } else {
            voxel.z += step.z;
            t = t_max.z;
            t_max.z += inv.z;
            last_normal = IVec3::new(0, 0, -step.z);
        }

        if t > max_dist {
            break;
        }
    }
    None
}

/// `t` to reach the next integer boundary from `p` moving with velocity `d`.
fn boundary_dist(p: f32, d: f32) -> f32 {
    if d == 0.0 {
        return f32::INFINITY;
    }
    if d > 0.0 {
        (p.floor() + 1.0 - p) / d
    } else {
        (p - p.floor()) / -d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelcraft_core::block::blocks;
    use pixelcraft_core::chunk::ChunkStorage;
    use pixelcraft_core::coords::ChunkPos;

    fn world_with_block(at: BlockPos) -> World {
        let mut w = World::new();
        // All test rays start and end within chunk (0,0,0).
        w.insert_chunk(ChunkPos::new(0, 0, 0), ChunkStorage::empty());
        assert_eq!(at.chunk(), ChunkPos::new(0, 0, 0));
        w.set_block(at, blocks::STONE);
        w
    }

    #[test]
    fn hits_block_straight_ahead() {
        let target = BlockPos::new(5, 2, 2);
        let w = world_with_block(target);
        let reg = BlockRegistry::with_defaults();
        let hit = cast(&w, &reg, Vec3::new(0.5, 2.5, 2.5), Vec3::new(1.0, 0.0, 0.0), 20.0);
        let hit = hit.expect("ray should hit the block");
        assert_eq!(hit.block, target);
        // Entered through the -X face.
        assert_eq!(hit.normal, IVec3::new(-1, 0, 0));
        assert_eq!(hit.place_pos(), BlockPos::new(4, 2, 2));
    }

    #[test]
    fn misses_when_nothing_in_path() {
        let w = world_with_block(BlockPos::new(5, 2, 2));
        let reg = BlockRegistry::with_defaults();
        // Fire upward where there's nothing.
        let hit = cast(&w, &reg, Vec3::new(0.5, 2.5, 2.5), Vec3::new(0.0, 1.0, 0.0), 20.0);
        assert!(hit.is_none());
    }

    #[test]
    fn respects_max_distance() {
        let target = BlockPos::new(18, 2, 2);
        let w = world_with_block(target);
        let reg = BlockRegistry::with_defaults();
        let hit = cast(&w, &reg, Vec3::new(0.5, 2.5, 2.5), Vec3::new(1.0, 0.0, 0.0), 5.0);
        assert!(hit.is_none(), "block beyond max_dist should not be hit");
    }

    #[test]
    fn diagonal_ray_reports_consistent_face() {
        let target = BlockPos::new(4, 4, 0);
        let w = world_with_block(target);
        let reg = BlockRegistry::with_defaults();
        let hit = cast(&w, &reg, Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 1.0, 0.0), 20.0);
        if let Some(h) = hit {
            // Placement cell must be empty and adjacent to the hit.
            assert_eq!(h.place_pos(), h.block.offset(h.normal.x, h.normal.y, h.normal.z));
        }
    }
}
