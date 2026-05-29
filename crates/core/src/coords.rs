//! World and chunk coordinate systems and conversions.
//!
//! The world is partitioned into cubic chunks of [`CHUNK_SIZE`] voxels per
//! edge. We keep the edge length a power of two so that converting between
//! world-space block coordinates and (chunk, local) pairs reduces to cheap
//! bit-shifts and masks instead of integer division — this is on the hottest
//! path in the engine (every block lookup during meshing, physics and
//! world-gen funnels through here).

/// Number of voxels along one edge of a chunk.
pub const CHUNK_SIZE: usize = 32;
/// `log2(CHUNK_SIZE)` — used for shift-based coordinate math.
pub const CHUNK_SIZE_BITS: u32 = 5;
/// Mask isolating the in-chunk component of a world coordinate.
pub const CHUNK_MASK: i32 = (CHUNK_SIZE as i32) - 1;
/// Total voxels in a chunk.
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

const _: () = assert!(CHUNK_SIZE == 1 << CHUNK_SIZE_BITS);

/// Position of a chunk on the chunk grid (one unit = one chunk edge).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkPos {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// World-space block coordinate of this chunk's `(0,0,0)` corner.
    pub const fn origin(self) -> BlockPos {
        BlockPos::new(
            self.x << CHUNK_SIZE_BITS,
            self.y << CHUNK_SIZE_BITS,
            self.z << CHUNK_SIZE_BITS,
        )
    }

    /// Squared distance (in chunk units) to another chunk; avoids a sqrt on
    /// the streaming hot path where only ordering matters.
    pub fn distance_sq(self, other: ChunkPos) -> i64 {
        let dx = (self.x - other.x) as i64;
        let dy = (self.y - other.y) as i64;
        let dz = (self.z - other.z) as i64;
        dx * dx + dy * dy + dz * dz
    }

    pub fn offset(self, dx: i32, dy: i32, dz: i32) -> Self {
        Self::new(self.x + dx, self.y + dy, self.z + dz)
    }
}

/// A voxel position in world space.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct BlockPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl BlockPos {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Which chunk contains this block. Uses arithmetic shift so that
    /// negative coordinates round toward negative infinity (floor division).
    pub const fn chunk(self) -> ChunkPos {
        ChunkPos::new(
            self.x >> CHUNK_SIZE_BITS,
            self.y >> CHUNK_SIZE_BITS,
            self.z >> CHUNK_SIZE_BITS,
        )
    }

    /// Coordinates of this block relative to its containing chunk (`0..CHUNK_SIZE`).
    pub const fn local(self) -> LocalPos {
        LocalPos {
            x: (self.x & CHUNK_MASK) as u8,
            y: (self.y & CHUNK_MASK) as u8,
            z: (self.z & CHUNK_MASK) as u8,
        }
    }

    pub fn offset(self, dx: i32, dy: i32, dz: i32) -> Self {
        Self::new(self.x + dx, self.y + dy, self.z + dz)
    }
}

/// A voxel position within a single chunk; each axis is in `0..CHUNK_SIZE`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LocalPos {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

impl LocalPos {
    /// Build from raw axis values; debug-asserts they are in range.
    pub fn new(x: u8, y: u8, z: u8) -> Self {
        debug_assert!((x as usize) < CHUNK_SIZE);
        debug_assert!((y as usize) < CHUNK_SIZE);
        debug_assert!((z as usize) < CHUNK_SIZE);
        Self { x, y, z }
    }

    /// Flatten to a linear index in `0..CHUNK_VOLUME`.
    ///
    /// Layout is `x + z*S + y*S*S` (Y-major). Keeping all blocks of a
    /// horizontal layer contiguous matches the access pattern of column-based
    /// world-gen and per-Y-slice greedy meshing.
    #[inline]
    pub const fn index(self) -> usize {
        (self.x as usize)
            + (self.z as usize) * CHUNK_SIZE
            + (self.y as usize) * CHUNK_SIZE * CHUNK_SIZE
    }

    /// Inverse of [`LocalPos::index`].
    #[inline]
    pub const fn from_index(i: usize) -> Self {
        let x = i & (CHUNK_SIZE - 1);
        let z = (i >> CHUNK_SIZE_BITS) & (CHUNK_SIZE - 1);
        let y = i >> (CHUNK_SIZE_BITS * 2);
        Self {
            x: x as u8,
            y: y as u8,
            z: z as u8,
        }
    }

    /// World-space block position given the owning chunk.
    pub const fn to_block(self, chunk: ChunkPos) -> BlockPos {
        let o = chunk.origin();
        BlockPos::new(
            o.x + self.x as i32,
            o.y + self.y as i32,
            o.z + self.z as i32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_to_chunk_local_roundtrip() {
        for &(x, y, z) in &[
            (0, 0, 0),
            (31, 31, 31),
            (32, 0, 0),
            (-1, -1, -1),
            (-32, -33, 64),
            (1_000_000, -500_000, 12345),
        ] {
            let b = BlockPos::new(x, y, z);
            let c = b.chunk();
            let l = b.local();
            let rebuilt = l.to_block(c);
            assert_eq!(b, rebuilt, "roundtrip failed for {b:?}");
        }
    }

    #[test]
    fn negative_coords_floor_correctly() {
        // -1 lives in chunk -1 at local index CHUNK_SIZE-1.
        let b = BlockPos::new(-1, -1, -1);
        assert_eq!(b.chunk(), ChunkPos::new(-1, -1, -1));
        assert_eq!(b.local(), LocalPos::new(31, 31, 31));
    }

    #[test]
    fn local_index_roundtrip_full_volume() {
        for i in 0..CHUNK_VOLUME {
            assert_eq!(LocalPos::from_index(i).index(), i);
        }
    }

    #[test]
    fn index_is_within_volume() {
        let l = LocalPos::new(31, 31, 31);
        assert_eq!(l.index(), CHUNK_VOLUME - 1);
    }
}
