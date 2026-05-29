//! Infinite, deterministic terrain generation.
//!
//! Generation is split into two passes per chunk, mirroring the structure used
//! by Minecraft/Terraria-style generators:
//!
//! 1. **Terrain** — a per-column height field (layered fBm noise) fills solid
//!    ground, carves a few caves with 3D noise, and floods water up to sea
//!    level. Surface materials come from the column's [`Biome`].
//! 2. **Decoration** — trees, flowers and mushrooms are scattered using a
//!    per-column deterministic PRNG. Decoration iterates a *padded* window of
//!    columns so structures (like trees) that originate just outside the chunk
//!    still contribute their overhanging blocks, giving seamless borders
//!    regardless of the order chunks are generated in.

use crate::biome::Biome;
use crate::noise::{hash_coords, Perlin, SplitMix64};
use pixelcraft_core::block::{blocks, BlockId};
use pixelcraft_core::chunk::ChunkStorage;
use pixelcraft_core::coords::{ChunkPos, LocalPos, CHUNK_SIZE};

pub const SEA_LEVEL: i32 = 62;
/// Largest tree footprint radius; decoration pads columns by this much.
const DECORATION_MARGIN: i32 = 3;

/// Deterministic terrain generator. Cloneable and `Send`/`Sync` so worker
/// threads can each hold one for parallel chunk generation.
#[derive(Clone)]
pub struct WorldGenerator {
    seed: u64,
    height_noise: Perlin,
    detail_noise: Perlin,
    temperature_noise: Perlin,
    humidity_noise: Perlin,
    cave_noise: Perlin,
}

/// Per-column terrain summary, reused by the decoration pass.
#[derive(Clone, Copy)]
struct Column {
    surface_height: i32,
    biome: Biome,
}

impl WorldGenerator {
    pub fn new(seed: u64) -> Self {
        // Decorrelate the sub-fields by offsetting the seed; identical seeds
        // would make temperature and humidity perfectly correlated.
        Self {
            seed,
            height_noise: Perlin::new(seed),
            detail_noise: Perlin::new(seed ^ 0xA1B2_C3D4),
            temperature_noise: Perlin::new(seed ^ 0x1357_9BDF),
            humidity_noise: Perlin::new(seed ^ 0x2468_ACE0),
            cave_noise: Perlin::new(seed ^ 0xDEAD_BEEF),
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Surface height (world Y of the topmost solid terrain block) for a column.
    pub fn surface_height(&self, wx: i32, wz: i32) -> i32 {
        self.column(wx, wz).surface_height
    }

    fn column(&self, wx: i32, wz: i32) -> Column {
        let fx = wx as f32;
        let fz = wz as f32;

        // Continent shape: very low frequency, the broad land/relief signal.
        let continent = self
            .height_noise
            .fbm2(fx * 0.0021, fz * 0.0021, 5, 2.0, 0.5);
        // Hill detail at higher frequency, gentler amplitude.
        let detail = self
            .detail_noise
            .fbm2(fx * 0.012, fz * 0.012, 4, 2.1, 0.45);

        let temperature = self
            .temperature_noise
            .fbm2(fx * 0.0009, fz * 0.0009, 3, 2.0, 0.5);
        let humidity = self
            .humidity_noise
            .fbm2(fx * 0.0011, fz * 0.0011, 3, 2.0, 0.5);

        // Provisional height for biome selection.
        let rough_height = SEA_LEVEL as f32 + continent * 26.0 + detail * 6.0;
        let biome = Biome::classify(temperature, humidity, rough_height as i32, SEA_LEVEL);
        let profile = biome.profile();

        let height = SEA_LEVEL as f32
            + continent * 26.0 * profile.relief
            + detail * 6.0
            + profile.base_offset;

        Column {
            surface_height: height.round() as i32,
            biome,
        }
    }

    /// Whether 3D cave noise carves out the given world position.
    #[inline]
    fn is_cave(&self, wx: i32, wy: i32, wz: i32) -> bool {
        if wy < 5 {
            return false; // keep a solid floor
        }
        let n = self.cave_noise.fbm3(
            wx as f32 * 0.03,
            wy as f32 * 0.045,
            wz as f32 * 0.03,
            3,
            2.0,
            0.5,
        );
        n > 0.62
    }

    /// Generate the storage for one chunk.
    pub fn generate_chunk(&self, pos: ChunkPos) -> ChunkStorage {
        let mut storage = ChunkStorage::empty();
        let origin = pos.origin();
        let chunk_min_y = origin.y;
        let chunk_max_y = origin.y + CHUNK_SIZE as i32 - 1;

        // Fast-skip: a chunk entirely above the highest possible terrain in its
        // footprint is pure air and needs no per-voxel work.
        let mut any_solid_possible = false;

        // --- Pass 1: terrain ---------------------------------------------
        let s = CHUNK_SIZE as i32;
        for lz in 0..s {
            for lx in 0..s {
                let wx = origin.x + lx;
                let wz = origin.z + lz;
                let col = self.column(wx, wz);
                let surface = col.surface_height;
                let profile = col.biome.profile();
                if surface >= chunk_min_y {
                    any_solid_possible = true;
                }

                for ly in 0..s {
                    let wy = chunk_min_y + ly;
                    let block = self.terrain_block(wx, wy, wz, surface, &profile);
                    if !block.is_air() {
                        storage.set(
                            LocalPos::new(lx as u8, ly as u8, lz as u8),
                            block,
                        );
                    }
                }
            }
        }

        // --- Pass 2: decoration ------------------------------------------
        // Only relevant for chunks that intersect the surface band.
        if any_solid_possible && chunk_max_y >= SEA_LEVEL {
            self.decorate(pos, &mut storage);
        }

        storage
    }

    /// Choose the block for a single voxel during the terrain pass.
    #[inline]
    fn terrain_block(
        &self,
        wx: i32,
        wy: i32,
        wz: i32,
        surface: i32,
        profile: &crate::biome::BiomeProfile,
    ) -> BlockId {
        if wy > surface {
            // Above ground: water up to sea level, else air.
            if wy <= SEA_LEVEL {
                return blocks::WATER;
            }
            return BlockId::AIR;
        }
        // At or below surface.
        if self.is_cave(wx, wy, wz) {
            // Caves below the waterline stay flooded so we don't get dry holes
            // poking out of lakes.
            if wy <= SEA_LEVEL {
                return blocks::WATER;
            }
            return BlockId::AIR;
        }
        if wy == surface {
            // Underwater surfaces are sand/dirt, never grass.
            if surface < SEA_LEVEL {
                return blocks::DIRT;
            }
            return profile.surface;
        }
        if wy >= surface - 3 {
            return profile.subsurface;
        }
        blocks::STONE
    }

    /// Scatter trees and flora across a padded window of columns and stamp any
    /// blocks that land inside `pos` into `storage`.
    fn decorate(&self, pos: ChunkPos, storage: &mut ChunkStorage) {
        let origin = pos.origin();
        let s = CHUNK_SIZE as i32;
        for lz in -DECORATION_MARGIN..(s + DECORATION_MARGIN) {
            for lx in -DECORATION_MARGIN..(s + DECORATION_MARGIN) {
                let wx = origin.x + lx;
                let wz = origin.z + lz;
                let col = self.column(wx, wz);
                let profile = col.biome.profile();
                let surface = col.surface_height;
                if surface < SEA_LEVEL {
                    continue; // underwater: no land flora
                }
                let mut rng = SplitMix64::new(hash_coords(self.seed ^ 0x5EED, wx, wz));
                let roll = rng.next_f32();
                if roll < profile.tree_density {
                    self.place_tree(storage, origin, wx, wz, surface, &mut rng);
                } else if roll < profile.tree_density + profile.flora_density {
                    self.place_flora(storage, origin, wx, wz, surface, col.biome, &mut rng);
                }
            }
        }
    }

    /// Stamp a rounded cosy tree. The trunk sits at the column; the canopy is a
    /// small blob of leaves. Only blocks inside the target chunk are written.
    fn place_tree(
        &self,
        storage: &mut ChunkStorage,
        origin: pixelcraft_core::coords::BlockPos,
        wx: i32,
        wz: i32,
        surface: i32,
        rng: &mut SplitMix64,
    ) {
        let trunk_h = 4 + (rng.next_u64() % 3) as i32; // 4..6
        let base = surface + 1;
        // Trunk.
        for h in 0..trunk_h {
            self.stamp(storage, origin, wx, base + h, wz, blocks::TRUNK, true);
        }
        // Canopy: a squashed sphere of leaves around the top.
        let top = base + trunk_h;
        let r = 2;
        for dy in -1..=2 {
            // Narrow the canopy at the very top for a rounded silhouette.
            let layer_r = if dy >= 2 { 1 } else { r };
            for dz in -layer_r..=layer_r {
                for dx in -layer_r..=layer_r {
                    if dx * dx + dz * dz + (dy * dy) / 2 > layer_r * layer_r + 1 {
                        continue;
                    }
                    self.stamp(
                        storage,
                        origin,
                        wx + dx,
                        top + dy,
                        wz + dz,
                        blocks::LEAVES,
                        false, // don't overwrite trunk with leaves
                    );
                }
            }
        }
    }

    fn place_flora(
        &self,
        storage: &mut ChunkStorage,
        origin: pixelcraft_core::coords::BlockPos,
        wx: i32,
        wz: i32,
        surface: i32,
        biome: Biome,
        rng: &mut SplitMix64,
    ) {
        let pick = rng.next_u64() % 100;
        let block = match biome {
            Biome::Forest => {
                if pick < 55 {
                    blocks::MUSHROOM
                } else if pick < 80 {
                    blocks::FLOWER_PINK
                } else {
                    blocks::FLOWER_BLUE
                }
            }
            Biome::Dunes => blocks::PUMPKIN,
            _ => {
                if pick < 50 {
                    blocks::FLOWER_PINK
                } else {
                    blocks::FLOWER_BLUE
                }
            }
        };
        self.stamp(storage, origin, wx, surface + 1, wz, block, false);
    }

    /// Write a block at world coordinates if it falls within this chunk.
    /// `overwrite` controls whether an existing non-air block is replaced.
    #[inline]
    fn stamp(
        &self,
        storage: &mut ChunkStorage,
        origin: pixelcraft_core::coords::BlockPos,
        wx: i32,
        wy: i32,
        wz: i32,
        block: BlockId,
        overwrite: bool,
    ) {
        let lx = wx - origin.x;
        let ly = wy - origin.y;
        let lz = wz - origin.z;
        let s = CHUNK_SIZE as i32;
        if lx < 0 || ly < 0 || lz < 0 || lx >= s || ly >= s || lz >= s {
            return;
        }
        let lp = LocalPos::new(lx as u8, ly as u8, lz as u8);
        if !overwrite && !storage.get(lp).is_air() {
            return;
        }
        storage.set(lp, block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelcraft_core::block::RenderKind;
    use pixelcraft_core::block::BlockRegistry;

    #[test]
    fn generation_is_deterministic() {
        let g = WorldGenerator::new(123);
        let a = g.generate_chunk(ChunkPos::new(2, 1, -3));
        let b = g.generate_chunk(ChunkPos::new(2, 1, -3));
        for i in 0..pixelcraft_core::coords::CHUNK_VOLUME {
            assert_eq!(a.get_index(i), b.get_index(i));
        }
    }

    #[test]
    fn surface_height_is_reasonable() {
        let g = WorldGenerator::new(7);
        for x in -200..200 {
            let h = g.surface_height(x * 3, x * 5);
            assert!(h > 0 && h < 200, "implausible height {h} at x={x}");
        }
    }

    #[test]
    fn deep_underground_is_solid_stone_chunk_region() {
        // A chunk well below sea level should be (almost) entirely solid.
        let g = WorldGenerator::new(99);
        let c = g.generate_chunk(ChunkPos::new(0, 0, 0)); // world y 0..31
        let mut solid = 0;
        for i in 0..pixelcraft_core::coords::CHUNK_VOLUME {
            if !c.get_index(i).is_air() {
                solid += 1;
            }
        }
        // Allow for caves but expect mostly solid.
        assert!(
            solid > pixelcraft_core::coords::CHUNK_VOLUME * 3 / 4,
            "expected mostly solid deep chunk, got {solid}"
        );
    }

    #[test]
    fn high_sky_chunk_is_pure_air() {
        let g = WorldGenerator::new(5);
        // World y 320..351 — far above any terrain.
        let c = g.generate_chunk(ChunkPos::new(0, 10, 0));
        assert!(c.is_uniform());
        assert_eq!(c.uniform_block(), Some(BlockId::AIR));
    }

    #[test]
    fn trees_and_terrain_use_registered_blocks() {
        // Every block the generator emits must exist in the registry and, if
        // visible, be renderable — guards against id drift.
        let reg = BlockRegistry::with_defaults();
        let g = WorldGenerator::new(2024);
        let mut seen_leaves = false;
        for cx in -1..=1 {
            for cz in -1..=1 {
                let c = g.generate_chunk(ChunkPos::new(cx, 2, cz)); // y 64..95
                for i in 0..pixelcraft_core::coords::CHUNK_VOLUME {
                    let id = c.get_index(i);
                    let b = reg.get(id);
                    if id == blocks::LEAVES {
                        seen_leaves = true;
                        assert!(matches!(b.render, RenderKind::TransparentCube));
                    }
                }
            }
        }
        // Across 9 surface chunks we expect at least one tree somewhere.
        assert!(seen_leaves, "no trees generated across sampled chunks");
    }
}
