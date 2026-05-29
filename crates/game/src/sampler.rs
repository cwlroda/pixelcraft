//! Adapters that let the mesher and physics read from a live [`World`].

use pixelcraft_core::block::{BlockId, BlockRegistry};
use pixelcraft_core::chunk::ChunkStorage;
use pixelcraft_core::coords::{BlockPos, ChunkPos, LocalPos, CHUNK_SIZE};
use pixelcraft_core::world::World;
use pixelcraft_mesh::BlockSampler;
use pixelcraft_physics::SolidQuery;

const N: i32 = CHUNK_SIZE as i32;

/// Mesher sampler for one chunk. Interior reads hit the chunk's own storage
/// directly (the common case); only the one-block border ring falls back to the
/// world hash map to reach neighbouring chunks. This keeps meshing cache-hot.
pub struct ChunkNeighborSampler<'a> {
    world: &'a World,
    origin: BlockPos,
    center: &'a ChunkStorage,
}

impl<'a> ChunkNeighborSampler<'a> {
    pub fn new(world: &'a World, pos: ChunkPos) -> Option<Self> {
        let center = &world.get_chunk(pos)?.storage;
        Some(Self {
            world,
            origin: pos.origin(),
            center,
        })
    }
}

impl BlockSampler for ChunkNeighborSampler<'_> {
    #[inline]
    fn block_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        if (0..N).contains(&x) && (0..N).contains(&y) && (0..N).contains(&z) {
            self.center
                .get(LocalPos::new(x as u8, y as u8, z as u8))
        } else {
            self.world
                .block_at(BlockPos::new(self.origin.x + x, self.origin.y + y, self.origin.z + z))
        }
    }
}

/// Physics collision query backed by the world and block registry. A block is
/// solid iff its registry entry says so.
pub struct WorldSolid<'a> {
    pub world: &'a World,
    pub registry: &'a BlockRegistry,
}

impl SolidQuery for WorldSolid<'_> {
    #[inline]
    fn is_solid(&self, pos: BlockPos) -> bool {
        self.registry.get(self.world.block_at(pos)).solid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelcraft_core::block::blocks;

    #[test]
    fn sampler_reads_neighbour_across_border() {
        let mut world = World::new();
        world.insert_chunk(ChunkPos::new(0, 0, 0), ChunkStorage::empty());
        // Neighbour chunk to -X with a solid block at its far edge (x=31).
        let mut neigh = ChunkStorage::empty();
        neigh.set(LocalPos::new(31, 4, 4), blocks::STONE);
        world.insert_chunk(ChunkPos::new(-1, 0, 0), neigh);

        let s = ChunkNeighborSampler::new(&world, ChunkPos::new(0, 0, 0)).unwrap();
        // Local x = -1 maps into the neighbour's x = 31.
        assert_eq!(s.block_at(-1, 4, 4), blocks::STONE);
        assert_eq!(s.block_at(0, 4, 4), BlockId::AIR);
    }

    #[test]
    fn solid_query_respects_registry() {
        let reg = BlockRegistry::with_defaults();
        let mut world = World::new();
        world.insert_chunk(ChunkPos::new(0, 0, 0), ChunkStorage::empty());
        world.set_block(BlockPos::new(1, 1, 1), blocks::STONE);
        world.set_block(BlockPos::new(2, 1, 1), blocks::WATER);
        let q = WorldSolid { world: &world, registry: &reg };
        assert!(q.is_solid(BlockPos::new(1, 1, 1)));
        assert!(!q.is_solid(BlockPos::new(2, 1, 1))); // water is not solid
        assert!(!q.is_solid(BlockPos::new(5, 5, 5))); // air
    }
}
