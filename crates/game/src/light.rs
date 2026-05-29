//! Baked voxel lighting: sky light and block (emitter) light, flood-filled with
//! a BFS over the chunk plus a margin so light bleeds correctly across borders.
//!
//! Sky light enters from open columns and falls straight down without
//! attenuation (sunbeams), decrementing when it spreads sideways or up — so
//! caves and overhangs darken smoothly. Block light radiates from emitters
//! (lanterns, crystals, glowing mushrooms), decrementing one level per step and
//! blocked by opaque blocks. The result is sampled per-face by the mesher and
//! combined with the live daylight in the shader, so lanterns glow at night
//! while open ground tracks the sun — without re-meshing.

use pixelcraft_core::block::{BlockId, BlockRegistry};
use pixelcraft_core::coords::{BlockPos, CHUNK_SIZE};
use pixelcraft_mesh::ChunkLight;
use std::collections::VecDeque;

/// Blocks of padding around the chunk; light from sources this far into a
/// neighbour still reaches the chunk edge. Beyond it, tiny seams are possible.
const M: i32 = 6;
const N: i32 = CHUNK_SIZE as i32;
const R: i32 = N + 2 * M;

#[inline]
fn ridx(x: i32, y: i32, z: i32) -> usize {
    (x + y * R + z * R * R) as usize
}

/// Bake the light for the chunk whose origin (world coords of its `(0,0,0)`) is
/// `origin`. `block_at` reports the block id at a world position; the registry
/// supplies its opacity and emission. Sampling each cell once (rather than once
/// per property) keeps the world lookups — the dominant cost — to a minimum.
pub fn bake(
    origin: BlockPos,
    block_at: impl Fn(BlockPos) -> BlockId,
    registry: &BlockRegistry,
) -> ChunkLight {
    let vol = (R * R * R) as usize;
    let mut solid = vec![false; vol];
    let mut sky = vec![0u8; vol];
    let mut block = vec![0u8; vol];

    // Sample the padded region's blocks once each.
    let mut queue_sky: VecDeque<usize> = VecDeque::new();
    let mut queue_block: VecDeque<usize> = VecDeque::new();
    for rz in 0..R {
        for ry in 0..R {
            for rx in 0..R {
                let wp = BlockPos::new(origin.x - M + rx, origin.y - M + ry, origin.z - M + rz);
                let i = ridx(rx, ry, rz);
                let def = registry.get(block_at(wp));
                if def.occludes() {
                    solid[i] = true;
                }
                if def.light_emission > 0 {
                    block[i] = def.light_emission;
                    queue_block.push_back(i);
                }
            }
        }
    }
    // Seed sky light at the top plane of the region (open sky above).
    let top = R - 1;
    for rz in 0..R {
        for rx in 0..R {
            let i = ridx(rx, top, rz);
            if !solid[i] {
                sky[i] = 15;
                queue_sky.push_back(i);
            }
        }
    }

    propagate(&mut sky, &solid, &mut queue_sky, true);
    propagate(&mut block, &solid, &mut queue_block, false);

    // Copy the chunk + one-block ring into the mesher's light grid.
    let mut light = ChunkLight::dark();
    for lz in -1..=N {
        for ly in -1..=N {
            for lx in -1..=N {
                let i = ridx(lx + M, ly + M, lz + M);
                light.set(lx, ly, lz, sky[i], block[i]);
            }
        }
    }
    light
}

/// Flood-fill light levels. For sky light, propagation straight *down* keeps a
/// full level (sunbeam); every other direction loses one level.
fn propagate(level: &mut [u8], solid: &[bool], queue: &mut VecDeque<usize>, sky: bool) {
    // Neighbour offsets as (dx, dy, dz).
    const DIRS: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    while let Some(i) = queue.pop_front() {
        let l = level[i];
        if l <= 1 {
            continue;
        }
        let x = (i as i32) % R;
        let y = ((i as i32) / R) % R;
        let z = (i as i32) / (R * R);
        for (dx, dy, dz) in DIRS {
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            if nx < 0 || ny < 0 || nz < 0 || nx >= R || ny >= R || nz >= R {
                continue;
            }
            let ni = ridx(nx, ny, nz);
            if solid[ni] {
                continue;
            }
            // Sky light falling straight down keeps full strength.
            let nl = if sky && dy == -1 && l == 15 {
                15
            } else {
                l - 1
            };
            if nl > level[ni] {
                level[ni] = nl;
                queue.push_back(ni);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelcraft_core::block::{blocks, BlockId, BlockRegistry};
    use pixelcraft_core::chunk::ChunkStorage;
    use pixelcraft_core::coords::{ChunkPos, LocalPos};
    use pixelcraft_core::world::World;

    fn world_light(world: &World, reg: &BlockRegistry, pos: ChunkPos) -> ChunkLight {
        bake(pos.origin(), |bp| world.block_at(bp), reg)
    }

    #[test]
    fn open_air_is_fully_sky_lit() {
        let reg = BlockRegistry::with_defaults();
        let mut world = World::new();
        world.insert_chunk(ChunkPos::new(0, 5, 0), ChunkStorage::empty()); // all air, high up
        let light = world_light(&world, &reg, ChunkPos::new(0, 5, 0));
        let (sky, _blk) = light.get(5, 5, 5);
        assert_eq!(sky, 15, "open air should be fully sky-lit");
    }

    #[test]
    fn under_solid_cover_is_dark() {
        let reg = BlockRegistry::with_defaults();
        let mut world = World::new();
        // A solid stone chunk with a small air pocket carved deep inside.
        let mut storage = ChunkStorage::filled(blocks::STONE);
        storage.set(LocalPos::new(16, 16, 16), BlockId::AIR);
        world.insert_chunk(ChunkPos::new(0, 0, 0), storage);
        let light = world_light(&world, &reg, ChunkPos::new(0, 0, 0));
        let (sky, _blk) = light.get(16, 16, 16);
        assert_eq!(sky, 0, "a sealed pocket gets no sky light");
    }

    #[test]
    fn lantern_emits_block_light_that_falls_off() {
        let reg = BlockRegistry::with_defaults();
        let mut world = World::new();
        let mut storage = ChunkStorage::empty();
        storage.set(LocalPos::new(16, 16, 16), blocks::LANTERN);
        world.insert_chunk(ChunkPos::new(0, 0, 0), storage);
        let light = world_light(&world, &reg, ChunkPos::new(0, 0, 0));
        // Adjacent air is bright; it dims with distance.
        let (_s1, near) = light.get(17, 16, 16);
        let (_s2, far) = light.get(20, 16, 16);
        assert!(near > 0, "air beside a lantern should be lit");
        assert!(near > far, "block light should fall off with distance");
    }
}
