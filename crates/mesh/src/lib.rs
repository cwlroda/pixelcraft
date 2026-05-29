//! Greedy meshing: turning a chunk of voxels into a compact triangle mesh.
//!
//! A naive mesher emits one quad per visible block face. Greedy meshing instead
//! merges adjacent, coplanar faces that share the same material into a single
//! large quad, which slashes vertex counts dramatically on flat terrain (a 32×32
//! grass plateau becomes one quad instead of 1024). This is the standard
//! technique behind high-performance voxel engines.
//!
//! The algorithm sweeps each of the three axes, builds a 2D "face mask" for each
//! slice, then greedily grows maximal rectangles over equal mask entries
//! (after Mikola Lysenko's classic formulation). Faces are split into an
//! **opaque** layer (face-culled cubes) and a **transparent** layer (water,
//! leaves, flora) so the renderer can draw them with the right blend state.

use bytemuck::{Pod, Zeroable};
use pixelcraft_core::block::{BlockId, BlockRegistry};
use pixelcraft_core::coords::CHUNK_SIZE;

/// One mesh vertex. Compact and `Pod` so it uploads straight to the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Pre-shaded linear colour (block tint × face/AO shading).
    pub color: [f32; 4],
}

/// A single draw layer's geometry.
#[derive(Default, Clone)]
pub struct MeshBuffers {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshBuffers {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn quad_count(&self) -> usize {
        self.indices.len() / 6
    }

    fn push_quad(&mut self, corners: [[f32; 3]; 4], normal: [f32; 3], color: [f32; 4], flip: bool) {
        let base = self.vertices.len() as u32;
        for &position in &corners {
            self.vertices.push(Vertex {
                position,
                normal,
                color,
            });
        }
        // Two triangles. `flip` reverses winding for negative-facing quads so
        // every front face is consistently counter-clockwise.
        if flip {
            self.indices
                .extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        } else {
            self.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
}

/// Result of meshing one chunk.
#[derive(Default, Clone)]
pub struct ChunkMesh {
    pub opaque: MeshBuffers,
    pub transparent: MeshBuffers,
}

impl ChunkMesh {
    pub fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.transparent.is_empty()
    }

    pub fn total_vertices(&self) -> usize {
        self.opaque.vertices.len() + self.transparent.vertices.len()
    }
}

/// Provides block ids for the chunk being meshed. Coordinates are chunk-local
/// but may extend one block past either edge (`-1 ..= CHUNK_SIZE`) so the mesher
/// can cull faces against blocks in neighbouring chunks.
pub trait BlockSampler {
    fn block_at(&self, x: i32, y: i32, z: i32) -> BlockId;
}

/// Which geometry layer a pass produces.
#[derive(Clone, Copy, PartialEq)]
enum Layer {
    Opaque,
    Transparent,
}

/// Identifies a face for greedy-merge equality: same block, same facing.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceKey {
    block: BlockId,
    positive: bool,
}

/// Per-face directional shading for a soft, cosy look. Top faces are brightest,
/// the underside darkest, with the four sides in between — cheap fake lighting
/// that reads well before real lighting lands.
fn face_shade(axis: usize, positive: bool) -> f32 {
    match (axis, positive) {
        (1, true) => 1.0,   // +Y top
        (1, false) => 0.5,  // -Y bottom
        (0, _) => 0.78,     // ±X
        (2, _) => 0.65,     // ±Z
        _ => 0.8,
    }
}

const N: i32 = CHUNK_SIZE as i32;

/// Mesh a chunk. `sampler` resolves blocks (including one ring of neighbours)
/// and `registry` supplies occlusion/visibility/colour.
pub fn mesh_chunk<S: BlockSampler>(sampler: &S, registry: &BlockRegistry) -> ChunkMesh {
    let mut mesh = ChunkMesh::default();
    greedy_pass(sampler, registry, Layer::Opaque, &mut mesh.opaque);
    greedy_pass(sampler, registry, Layer::Transparent, &mut mesh.transparent);
    mesh
}

fn greedy_pass<S: BlockSampler>(
    sampler: &S,
    registry: &BlockRegistry,
    layer: Layer,
    out: &mut MeshBuffers,
) {
    // Decide whether the face between voxels `a` (lower along axis d) and `b`
    // (higher) should be emitted in this layer, and to whom it belongs.
    let face_between = |a: BlockId, b: BlockId| -> Option<FaceKey> {
        let ba = registry.get(a);
        let bb = registry.get(b);
        match layer {
            Layer::Opaque => {
                // Opaque cube exposed to a non-opaque neighbour.
                if ba.occludes() && !bb.occludes() {
                    Some(FaceKey { block: a, positive: true })
                } else if bb.occludes() && !ba.occludes() {
                    Some(FaceKey { block: b, positive: false })
                } else {
                    None
                }
            }
            Layer::Transparent => {
                // Visible, non-occluding blocks (water/leaves/flora). Cull the
                // interface between two of the *same* such block, and anything
                // hidden behind an opaque neighbour.
                let a_t = ba.is_visible() && !ba.occludes();
                let b_t = bb.is_visible() && !bb.occludes();
                if a_t && a != b && !bb.occludes() {
                    Some(FaceKey { block: a, positive: true })
                } else if b_t && a != b && !ba.occludes() {
                    Some(FaceKey { block: b, positive: false })
                } else {
                    None
                }
            }
        }
    };

    for d in 0..3usize {
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;
        let mut x = [0i32; 3];
        let mut mask: Vec<Option<FaceKey>> = vec![None; (N * N) as usize];

        // Slices run from -1 (the boundary with the previous neighbour chunk)
        // up to N-1, so both outer faces of the chunk are produced.
        for slice in -1..N {
            x[d] = slice;
            // Build the mask for this slice.
            let mut n = 0usize;
            for j in 0..N {
                x[v] = j;
                for i in 0..N {
                    x[u] = i;
                    let lo = x;
                    let mut hi = x;
                    hi[d] += 1;
                    let a = sampler.block_at(lo[0], lo[1], lo[2]);
                    let b = sampler.block_at(hi[0], hi[1], hi[2]);
                    mask[n] = face_between(a, b);
                    n += 1;
                }
            }

            // Greedily emit maximal rectangles from the mask.
            let mut n = 0usize;
            for j in 0..N {
                let mut i = 0i32;
                while i < N {
                    let Some(key) = mask[n] else {
                        i += 1;
                        n += 1;
                        continue;
                    };
                    // Grow width along u.
                    let mut w = 1i32;
                    while i + w < N && mask[n + w as usize] == Some(key) {
                        w += 1;
                    }
                    // Grow height along v.
                    let mut h = 1i32;
                    'grow: while j + h < N {
                        for k in 0..w {
                            if mask[n + k as usize + (h * N) as usize] != Some(key) {
                                break 'grow;
                            }
                        }
                        h += 1;
                    }

                    emit_quad(out, registry, key, d, u, v, slice + 1, i, j, w, h);

                    // Clear the consumed cells.
                    for l in 0..h {
                        for k in 0..w {
                            mask[n + k as usize + (l * N) as usize] = None;
                        }
                    }
                    i += w;
                    n += w as usize;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_quad(
    out: &mut MeshBuffers,
    registry: &BlockRegistry,
    key: FaceKey,
    d: usize,
    u: usize,
    v: usize,
    plane: i32,
    i: i32,
    j: i32,
    w: i32,
    h: i32,
) {
    let block = registry.get(key.block);
    let shade = face_shade(d, key.positive);
    let c = block.color;
    let color = [
        (c.r as f32 / 255.0) * shade,
        (c.g as f32 / 255.0) * shade,
        (c.b as f32 / 255.0) * shade,
        c.a as f32 / 255.0,
    ];

    let mut base = [0f32; 3];
    base[d] = plane as f32;
    base[u] = i as f32;
    base[v] = j as f32;

    let mut du = [0f32; 3];
    du[u] = w as f32;
    let mut dv = [0f32; 3];
    dv[v] = h as f32;

    let p0 = base;
    let p1 = [base[0] + du[0], base[1] + du[1], base[2] + du[2]];
    let p2 = [
        base[0] + du[0] + dv[0],
        base[1] + du[1] + dv[1],
        base[2] + du[2] + dv[2],
    ];
    let p3 = [base[0] + dv[0], base[1] + dv[1], base[2] + dv[2]];

    let mut normal = [0f32; 3];
    normal[d] = if key.positive { 1.0 } else { -1.0 };

    out.push_quad([p0, p1, p2, p3], normal, color, !key.positive);
}

/// Convenience sampler backed by a single [`ChunkStorage`]: every position
/// outside the chunk reads as air. Useful for meshing isolated chunks and in
/// tests; real rendering uses a world-backed sampler that reaches into
/// neighbouring chunks.
pub struct IsolatedChunkSampler<'a> {
    pub storage: &'a pixelcraft_core::chunk::ChunkStorage,
}

impl BlockSampler for IsolatedChunkSampler<'_> {
    fn block_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        if x < 0 || y < 0 || z < 0 || x >= N || y >= N || z >= N {
            return BlockId::AIR;
        }
        self.storage.get(pixelcraft_core::coords::LocalPos::new(
            x as u8, y as u8, z as u8,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelcraft_core::block::blocks;
    use pixelcraft_core::chunk::ChunkStorage;
    use pixelcraft_core::coords::LocalPos;

    fn mesh_storage(s: &ChunkStorage) -> ChunkMesh {
        let reg = BlockRegistry::with_defaults();
        mesh_chunk(&IsolatedChunkSampler { storage: s }, &reg)
    }

    #[test]
    fn empty_chunk_produces_no_geometry() {
        let s = ChunkStorage::empty();
        let m = mesh_storage(&s);
        assert!(m.is_empty());
    }

    #[test]
    fn single_cube_has_six_quads() {
        let mut s = ChunkStorage::empty();
        s.set(LocalPos::new(5, 5, 5), blocks::STONE);
        let m = mesh_storage(&s);
        assert_eq!(m.opaque.quad_count(), 6, "a lone cube should expose 6 faces");
        assert_eq!(m.opaque.vertices.len(), 24);
        assert_eq!(m.opaque.indices.len(), 36);
    }

    #[test]
    fn flat_layer_merges_into_few_quads() {
        // Fill the bottom layer (y=0) fully with stone. Its top and bottom
        // should each greedily merge into a single 32×32 quad.
        let mut s = ChunkStorage::empty();
        for z in 0..CHUNK_SIZE as u8 {
            for x in 0..CHUNK_SIZE as u8 {
                s.set(LocalPos::new(x, 0, z), blocks::STONE);
            }
        }
        let m = mesh_storage(&s);
        // Top (+Y) = 1 quad, bottom (-Y) = 1 quad, and the four side strips =
        // 1 quad each → 6 total thanks to greedy merging.
        assert_eq!(m.opaque.quad_count(), 6);
    }

    #[test]
    fn solid_chunk_only_meshes_outer_shell() {
        // A completely solid chunk should only emit its 6 outer faces, never
        // any interior geometry.
        let s = ChunkStorage::filled(blocks::STONE);
        let m = mesh_storage(&s);
        assert_eq!(m.opaque.quad_count(), 6);
    }

    #[test]
    fn interior_faces_between_same_block_are_culled() {
        // Two adjacent stone cubes form a 2×1×1 bar. The shared face is culled,
        // and greedy merging fuses the two coplanar top/bottom/±Z faces, so the
        // bar reduces to just 6 quads (one per side).
        let mut s = ChunkStorage::empty();
        s.set(LocalPos::new(5, 5, 5), blocks::STONE);
        s.set(LocalPos::new(6, 5, 5), blocks::STONE);
        let m = mesh_storage(&s);
        assert_eq!(m.opaque.quad_count(), 6);
    }

    #[test]
    fn water_goes_to_transparent_layer() {
        let mut s = ChunkStorage::empty();
        s.set(LocalPos::new(8, 8, 8), blocks::WATER);
        let m = mesh_storage(&s);
        assert!(m.opaque.is_empty(), "water must not be in the opaque layer");
        assert_eq!(m.transparent.quad_count(), 6);
    }

    #[test]
    fn opaque_neighbour_culls_transparent_face() {
        // Water with stone on its +X side: that water face is hidden.
        let mut s = ChunkStorage::empty();
        s.set(LocalPos::new(8, 8, 8), blocks::WATER);
        s.set(LocalPos::new(9, 8, 8), blocks::STONE);
        let m = mesh_storage(&s);
        assert_eq!(m.transparent.quad_count(), 5, "hidden water face not culled");
        // Stone exposes all 6 faces (the -X side touches water, which does not
        // occlude).
        assert_eq!(m.opaque.quad_count(), 6);
    }

    #[test]
    fn all_quad_vertices_are_within_padded_bounds() {
        // Fill sparsely and assert geometry stays in [0, N].
        let mut s = ChunkStorage::empty();
        for i in (0..pixelcraft_core::coords::CHUNK_VOLUME).step_by(7) {
            s.set_index(i, blocks::DIRT);
        }
        let m = mesh_storage(&s);
        for v in m.opaque.vertices.iter() {
            for c in v.position {
                assert!((0.0..=N as f32).contains(&c), "vertex out of bounds: {c}");
            }
        }
    }
}
