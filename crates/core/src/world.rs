//! In-memory world: a sparse map of loaded chunks keyed by [`ChunkPos`].
//!
//! The world is conceptually infinite; only chunks near players are resident.
//! Storage is a hash map with a fast non-cryptographic hasher because chunk
//! lookups happen constantly (physics raycasts, neighbour queries in meshing,
//! block edits) and we never need DoS resistance.

use crate::block::BlockId;
use crate::chunk::ChunkStorage;
use crate::coords::{BlockPos, ChunkPos};
use ahash::AHashMap;

/// A resident chunk plus bookkeeping for the streaming/meshing pipeline.
pub struct Chunk {
    pub pos: ChunkPos,
    pub storage: ChunkStorage,
    /// Set when the voxels changed and the GPU mesh is stale.
    pub dirty: bool,
}

impl Chunk {
    pub fn new(pos: ChunkPos, storage: ChunkStorage) -> Self {
        Self {
            pos,
            storage,
            dirty: true,
        }
    }
}

/// The collection of currently-loaded chunks.
#[derive(Default)]
pub struct World {
    chunks: AHashMap<ChunkPos, Chunk>,
}

impl World {
    pub fn new() -> Self {
        Self {
            chunks: AHashMap::new(),
        }
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn contains_chunk(&self, pos: ChunkPos) -> bool {
        self.chunks.contains_key(&pos)
    }

    pub fn get_chunk(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&pos)
    }

    pub fn get_chunk_mut(&mut self, pos: ChunkPos) -> Option<&mut Chunk> {
        self.chunks.get_mut(&pos)
    }

    /// Insert (or replace) a chunk's storage.
    pub fn insert_chunk(&mut self, pos: ChunkPos, storage: ChunkStorage) {
        self.chunks.insert(pos, Chunk::new(pos, storage));
    }

    /// Remove a chunk (e.g. when it streams out of range), returning it so the
    /// caller can persist it if needed.
    pub fn remove_chunk(&mut self, pos: ChunkPos) -> Option<Chunk> {
        self.chunks.remove(&pos)
    }

    /// Block at a world position. Returns air for unloaded regions so callers
    /// (meshing, physics) can treat the frontier as empty without branching on
    /// load state.
    #[inline]
    pub fn block_at(&self, pos: BlockPos) -> BlockId {
        match self.chunks.get(&pos.chunk()) {
            Some(c) => c.storage.get(pos.local()),
            None => BlockId::AIR,
        }
    }

    /// Set a block, marking its chunk dirty. Returns `false` if the chunk is
    /// not loaded (the edit is dropped). Also marks neighbouring chunks dirty
    /// when the edit touches a chunk border, since their boundary faces may
    /// now need to appear or disappear.
    pub fn set_block(&mut self, pos: BlockPos, block: BlockId) -> bool {
        let cpos = pos.chunk();
        let Some(chunk) = self.chunks.get_mut(&cpos) else {
            return false;
        };
        let local = pos.local();
        if chunk.storage.get(local) == block {
            return true; // no change, no dirtying
        }
        chunk.storage.set(local, block);
        chunk.dirty = true;
        self.mark_border_neighbours_dirty(pos, cpos);
        true
    }

    fn mark_border_neighbours_dirty(&mut self, pos: BlockPos, cpos: ChunkPos) {
        let l = pos.local();
        let edge = crate::coords::CHUNK_SIZE as u8 - 1;
        let mut dirty_neighbour = |dx, dy, dz| {
            if let Some(n) = self.chunks.get_mut(&cpos.offset(dx, dy, dz)) {
                n.dirty = true;
            }
        };
        if l.x == 0 {
            dirty_neighbour(-1, 0, 0);
        }
        if l.x == edge {
            dirty_neighbour(1, 0, 0);
        }
        if l.y == 0 {
            dirty_neighbour(0, -1, 0);
        }
        if l.y == edge {
            dirty_neighbour(0, 1, 0);
        }
        if l.z == 0 {
            dirty_neighbour(0, 0, -1);
        }
        if l.z == edge {
            dirty_neighbour(0, 0, 1);
        }
    }

    pub fn iter_chunks(&self) -> impl Iterator<Item = &Chunk> {
        self.chunks.values()
    }

    pub fn iter_chunks_mut(&mut self) -> impl Iterator<Item = &mut Chunk> {
        self.chunks.values_mut()
    }

    /// Total approximate heap used by resident voxel data.
    pub fn voxel_memory_bytes(&self) -> usize {
        self.chunks.values().map(|c| c.storage.memory_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::blocks;

    #[test]
    fn unloaded_regions_read_as_air() {
        let w = World::new();
        assert_eq!(w.block_at(BlockPos::new(5, 70, -3)), BlockId::AIR);
    }

    #[test]
    fn set_block_requires_loaded_chunk() {
        let mut w = World::new();
        let p = BlockPos::new(1, 2, 3);
        assert!(!w.set_block(p, blocks::STONE));
        w.insert_chunk(p.chunk(), ChunkStorage::empty());
        assert!(w.set_block(p, blocks::STONE));
        assert_eq!(w.block_at(p), blocks::STONE);
    }

    #[test]
    fn edit_marks_chunk_dirty() {
        let mut w = World::new();
        let p = BlockPos::new(8, 8, 8);
        w.insert_chunk(p.chunk(), ChunkStorage::empty());
        // Freshly inserted chunks start dirty; clear then edit.
        w.get_chunk_mut(p.chunk()).unwrap().dirty = false;
        w.set_block(p, blocks::DIRT);
        assert!(w.get_chunk(p.chunk()).unwrap().dirty);
    }

    #[test]
    fn border_edit_dirties_neighbour() {
        let mut w = World::new();
        let center = ChunkPos::new(0, 0, 0);
        let neighbour = ChunkPos::new(-1, 0, 0);
        w.insert_chunk(center, ChunkStorage::empty());
        w.insert_chunk(neighbour, ChunkStorage::empty());
        w.get_chunk_mut(neighbour).unwrap().dirty = false;
        // x == 0 is the -X border of the center chunk.
        w.set_block(BlockPos::new(0, 4, 4), blocks::STONE);
        assert!(w.get_chunk(neighbour).unwrap().dirty);
    }
}
