//! Palette-compressed chunk storage.
//!
//! A naive chunk would store one [`BlockId`] (2 bytes) per voxel — 64 KiB for a
//! 32³ chunk. The overwhelming majority of chunks contain only a handful of
//! distinct block types (air + a couple of terrain materials), so we follow the
//! same approach Minecraft adopted in 1.13: a *paletted container*.
//!
//! Each chunk keeps a small `palette` of the distinct blocks it actually uses
//! and a tightly bit-packed array of indices into that palette. A chunk made of
//! a single block (extremely common: pure air or pure stone) collapses to
//! `bits == 0` and stores **no** per-voxel data at all. As more block types are
//! introduced the index width grows just enough to address the palette.

use crate::block::BlockId;
use crate::coords::{LocalPos, CHUNK_VOLUME};

/// Storage for the blocks of one chunk.
///
/// Invariants:
/// * `palette` is never empty; `palette[0]` is the "default" / fill block.
/// * `bits == 0` ⇔ `data` is empty ⇔ every voxel is `palette[0]`.
/// * otherwise `data` holds `CHUNK_VOLUME` indices of `bits` width each,
///   packed into `u64` words without straddling word boundaries.
#[derive(Clone)]
pub struct ChunkStorage {
    palette: Vec<BlockId>,
    data: Vec<u64>,
    bits: u32,
}

#[inline]
const fn entries_per_word(bits: u32) -> usize {
    (64 / bits) as usize
}

#[inline]
const fn words_needed(bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        let epw = entries_per_word(bits);
        CHUNK_VOLUME.div_ceil(epw)
    }
}

/// Smallest index width that can address `palette_len` distinct entries.
#[inline]
fn bits_for(palette_len: usize) -> u32 {
    if palette_len <= 1 {
        0
    } else {
        // ceil(log2(palette_len)), minimum 1.
        (usize::BITS - (palette_len - 1).leading_zeros()).max(1)
    }
}

impl ChunkStorage {
    /// A chunk entirely filled with `fill` (no per-voxel allocation).
    pub fn filled(fill: BlockId) -> Self {
        Self {
            palette: vec![fill],
            data: Vec::new(),
            bits: 0,
        }
    }

    /// An all-air chunk.
    pub fn empty() -> Self {
        Self::filled(BlockId::AIR)
    }

    /// True when every voxel is the same block (cheap to test, useful for
    /// skipping meshing of solid-air or solid-stone chunks).
    #[inline]
    pub fn is_uniform(&self) -> bool {
        self.bits == 0
    }

    /// If uniform, the single block; otherwise `None`.
    #[inline]
    pub fn uniform_block(&self) -> Option<BlockId> {
        if self.bits == 0 {
            Some(self.palette[0])
        } else {
            None
        }
    }

    #[inline]
    pub fn get(&self, pos: LocalPos) -> BlockId {
        self.get_index(pos.index())
    }

    #[inline]
    pub fn get_index(&self, i: usize) -> BlockId {
        if self.bits == 0 {
            return self.palette[0];
        }
        let epw = entries_per_word(self.bits);
        let word = self.data[i / epw];
        let shift = (i % epw) as u32 * self.bits;
        let mask = (1u64 << self.bits) - 1;
        let palette_idx = ((word >> shift) & mask) as usize;
        self.palette[palette_idx]
    }

    pub fn set(&mut self, pos: LocalPos, block: BlockId) {
        self.set_index(pos.index(), block)
    }

    pub fn set_index(&mut self, i: usize, block: BlockId) {
        let palette_idx = self.palette_index_of_or_insert(block);
        if self.bits == 0 {
            // Still uniform: either a no-op, or we just grew the palette in
            // `palette_index_of_or_insert`, in which case `bits` was bumped and
            // `data` allocated (all zeros = old fill block) — fall through.
            if self.bits == 0 {
                return;
            }
        }
        self.write_index(i, palette_idx);
    }

    /// Linear search the palette; insert the block if missing, growing the
    /// index width (and materialising `data`) as required.
    fn palette_index_of_or_insert(&mut self, block: BlockId) -> usize {
        if let Some(idx) = self.palette.iter().position(|&b| b == block) {
            return idx;
        }
        let new_idx = self.palette.len();
        self.palette.push(block);
        let needed = bits_for(self.palette.len());
        if needed > self.bits {
            self.grow_to(needed);
        }
        new_idx
    }

    /// Re-pack existing indices into a wider layout. When growing from the
    /// uniform state (`bits == 0`) the new data is simply all-zero, which
    /// already means "every voxel is `palette[0]`".
    fn grow_to(&mut self, new_bits: u32) {
        debug_assert!(new_bits > self.bits);
        let new_words = words_needed(new_bits);
        if self.bits == 0 {
            self.data = vec![0u64; new_words];
            self.bits = new_bits;
            return;
        }
        let old_bits = self.bits;
        let old_epw = entries_per_word(old_bits);
        let old_mask = (1u64 << old_bits) - 1;
        let old_data = std::mem::take(&mut self.data);
        let mut new_data = vec![0u64; new_words];
        let new_epw = entries_per_word(new_bits);
        for i in 0..CHUNK_VOLUME {
            let ow = old_data[i / old_epw];
            let oshift = (i % old_epw) as u32 * old_bits;
            let val = (ow >> oshift) & old_mask;
            let nshift = (i % new_epw) as u32 * new_bits;
            new_data[i / new_epw] |= val << nshift;
        }
        self.data = new_data;
        self.bits = new_bits;
    }

    #[inline]
    fn write_index(&mut self, i: usize, palette_idx: usize) {
        let epw = entries_per_word(self.bits);
        let shift = (i % epw) as u32 * self.bits;
        let mask = (1u64 << self.bits) - 1;
        let word = &mut self.data[i / epw];
        *word = (*word & !(mask << shift)) | ((palette_idx as u64) << shift);
    }

    /// Number of distinct block types currently in the palette.
    pub fn palette_len(&self) -> usize {
        self.palette.len()
    }

    pub fn bits_per_voxel(&self) -> u32 {
        self.bits
    }

    /// Approximate heap footprint of this chunk's voxel data, in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.palette.capacity() * std::mem::size_of::<BlockId>()
            + self.data.capacity() * std::mem::size_of::<u64>()
    }

    /// Serialise to a compact little-endian byte buffer:
    /// `[bits u32][palette_len u32][palette: u16…][data_len u32][data: u64…]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.palette.len() * 2 + self.data.len() * 8);
        out.extend_from_slice(&self.bits.to_le_bytes());
        out.extend_from_slice(&(self.palette.len() as u32).to_le_bytes());
        for b in &self.palette {
            out.extend_from_slice(&b.0.to_le_bytes());
        }
        out.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        for w in &self.data {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Deserialise from [`ChunkStorage::to_bytes`], returning the storage and
    /// the number of bytes consumed, or `None` if the buffer is malformed.
    pub fn from_bytes(bytes: &[u8]) -> Option<(Self, usize)> {
        let mut c = 0usize;
        let read_u32 = |bytes: &[u8], c: &mut usize| -> Option<u32> {
            let v = bytes.get(*c..*c + 4)?;
            *c += 4;
            Some(u32::from_le_bytes(v.try_into().ok()?))
        };
        let bits = read_u32(bytes, &mut c)?;
        let palette_len = read_u32(bytes, &mut c)? as usize;
        let mut palette = Vec::with_capacity(palette_len);
        for _ in 0..palette_len {
            let v = bytes.get(c..c + 2)?;
            c += 2;
            palette.push(BlockId(u16::from_le_bytes(v.try_into().ok()?)));
        }
        let data_len = read_u32(bytes, &mut c)? as usize;
        // Sanity: data length must match the bit width.
        if data_len != words_needed(bits) || palette.is_empty() {
            return None;
        }
        let mut data = Vec::with_capacity(data_len);
        for _ in 0..data_len {
            let v = bytes.get(c..c + 8)?;
            c += 8;
            data.push(u64::from_le_bytes(v.try_into().ok()?));
        }
        Some((
            Self {
                palette,
                data,
                bits,
            },
            c,
        ))
    }

    /// Drop the palette down to only blocks that are still referenced, and
    /// re-collapse to uniform when possible. Worth calling occasionally after
    /// heavy editing so long-lived chunks don't keep a bloated index width.
    pub fn shrink_palette(&mut self) {
        if self.bits == 0 {
            return;
        }
        // Collect which palette slots are actually used.
        let mut used = vec![false; self.palette.len()];
        for i in 0..CHUNK_VOLUME {
            let epw = entries_per_word(self.bits);
            let word = self.data[i / epw];
            let shift = (i % epw) as u32 * self.bits;
            let mask = (1u64 << self.bits) - 1;
            used[((word >> shift) & mask) as usize] = true;
        }
        if used.iter().filter(|&&u| u).count() <= 1 {
            // One block remains — collapse to uniform.
            let only = (0..self.palette.len()).find(|&i| used[i]).unwrap_or(0);
            *self = ChunkStorage::filled(self.palette[only]);
            return;
        }
        // Re-pack with a compacted palette if any slot is now unused.
        if used.iter().all(|&u| u) {
            return;
        }
        let mut remap = vec![0usize; self.palette.len()];
        let mut new_palette = Vec::new();
        for (old, &is_used) in used.iter().enumerate() {
            if is_used {
                remap[old] = new_palette.len();
                new_palette.push(self.palette[old]);
            }
        }
        let new_bits = bits_for(new_palette.len());
        let old_bits = self.bits;
        let old_epw = entries_per_word(old_bits);
        let old_mask = (1u64 << old_bits) - 1;
        let new_words = words_needed(new_bits);
        let new_epw = entries_per_word(new_bits);
        let mut new_data = vec![0u64; new_words];
        for i in 0..CHUNK_VOLUME {
            let ow = self.data[i / old_epw];
            let oshift = (i % old_epw) as u32 * old_bits;
            let old_idx = ((ow >> oshift) & old_mask) as usize;
            let new_idx = remap[old_idx] as u64;
            let nshift = (i % new_epw) as u32 * new_bits;
            new_data[i / new_epw] |= new_idx << nshift;
        }
        self.palette = new_palette;
        self.data = new_data;
        self.bits = new_bits;
    }
}

impl Default for ChunkStorage {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::blocks;

    fn lp(i: usize) -> LocalPos {
        LocalPos::from_index(i)
    }

    #[test]
    fn empty_chunk_is_uniform_air() {
        let c = ChunkStorage::empty();
        assert!(c.is_uniform());
        assert_eq!(c.uniform_block(), Some(BlockId::AIR));
        assert_eq!(c.get_index(0), BlockId::AIR);
        assert_eq!(c.get_index(CHUNK_VOLUME - 1), BlockId::AIR);
        // Uniform chunks must not allocate per-voxel data.
        assert_eq!(c.bits_per_voxel(), 0);
    }

    #[test]
    fn set_then_get_single_block() {
        let mut c = ChunkStorage::empty();
        c.set(lp(100), blocks::STONE);
        assert_eq!(c.get(lp(100)), blocks::STONE);
        assert_eq!(c.get(lp(101)), BlockId::AIR);
        assert!(!c.is_uniform());
    }

    #[test]
    fn many_block_types_grow_bits_and_stay_correct() {
        let mut c = ChunkStorage::empty();
        // Write a different block id every few voxels, forcing palette growth.
        let ids = [
            blocks::GRASS,
            blocks::DIRT,
            blocks::STONE,
            blocks::SAND,
            blocks::WATER,
            blocks::TRUNK,
            blocks::LEAVES,
            blocks::SNOW,
            blocks::LANTERN,
        ];
        for i in 0..CHUNK_VOLUME {
            c.set_index(i, ids[i % ids.len()]);
        }
        for i in 0..CHUNK_VOLUME {
            assert_eq!(c.get_index(i), ids[i % ids.len()], "mismatch at {i}");
        }
        assert!(c.bits_per_voxel() >= 4);
    }

    #[test]
    fn overwrite_same_voxel_repeatedly() {
        let mut c = ChunkStorage::empty();
        let i = 12345 % CHUNK_VOLUME;
        c.set_index(i, blocks::STONE);
        c.set_index(i, blocks::DIRT);
        c.set_index(i, blocks::WATER);
        assert_eq!(c.get_index(i), blocks::WATER);
    }

    #[test]
    fn shrink_collapses_back_to_uniform() {
        let mut c = ChunkStorage::empty();
        c.set_index(5, blocks::STONE);
        c.set_index(5, BlockId::AIR); // undo
        c.shrink_palette();
        assert!(c.is_uniform());
        assert_eq!(c.uniform_block(), Some(BlockId::AIR));
    }

    #[test]
    fn shrink_compacts_unused_palette_entries() {
        let mut c = ChunkStorage::empty();
        // Introduce several types then erase all but one.
        c.set_index(0, blocks::GRASS);
        c.set_index(1, blocks::DIRT);
        c.set_index(2, blocks::STONE);
        c.set_index(1, blocks::GRASS);
        c.set_index(2, blocks::GRASS);
        let before = c.palette_len();
        c.shrink_palette();
        assert!(c.palette_len() <= before);
        assert_eq!(c.get_index(0), blocks::GRASS);
        assert_eq!(c.get_index(1), blocks::GRASS);
        assert_eq!(c.get_index(2), blocks::GRASS);
        assert_eq!(c.get_index(3), BlockId::AIR);
    }

    #[test]
    fn serialization_roundtrips() {
        // Uniform chunk.
        let a = ChunkStorage::filled(blocks::STONE);
        let (a2, _) = ChunkStorage::from_bytes(&a.to_bytes()).unwrap();
        assert_eq!(a2.uniform_block(), Some(blocks::STONE));

        // Mixed chunk.
        let mut b = ChunkStorage::empty();
        for i in 0..CHUNK_VOLUME {
            if i % 3 == 0 {
                b.set_index(i, blocks::DIRT);
            } else if i % 5 == 0 {
                b.set_index(i, blocks::WATER);
            }
        }
        let bytes = b.to_bytes();
        let (b2, consumed) = ChunkStorage::from_bytes(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        for i in 0..CHUNK_VOLUME {
            assert_eq!(b.get_index(i), b2.get_index(i), "mismatch at {i}");
        }
    }

    #[test]
    fn from_bytes_rejects_garbage() {
        assert!(ChunkStorage::from_bytes(&[0, 1, 2]).is_none());
    }

    #[test]
    fn full_volume_checkerboard_roundtrips() {
        let mut c = ChunkStorage::empty();
        for i in 0..CHUNK_VOLUME {
            if i % 2 == 0 {
                c.set_index(i, blocks::STONE);
            }
        }
        for i in 0..CHUNK_VOLUME {
            let expect = if i % 2 == 0 {
                blocks::STONE
            } else {
                BlockId::AIR
            };
            assert_eq!(c.get_index(i), expect);
        }
    }
}
