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
///
/// `color` carries a luminance multiplier (face shade × ambient occlusion), not
/// the block tint — the tint comes from the texture array `layer` sampled at
/// `uv`. `uv` is measured in block units along the quad so it tiles correctly
/// across greedy-merged faces (the shader takes `fract(uv)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub layer: u32,
    /// Baked lighting, `[sky, block]` in `0..1`. The shader scales sky light by
    /// the current daylight and takes the max with block light, so lanterns
    /// stand out at night while open ground tracks the sun.
    pub light: [f32; 2],
}

/// Baked per-cell light for a chunk and its one-block border ring (so the
/// mesher can read the light of the air cell each face opens onto). Stores sky
/// and block light (0..15) for local coordinates in `-1..=CHUNK_SIZE`.
#[derive(Clone)]
pub struct ChunkLight {
    sky: Vec<u8>,
    block: Vec<u8>,
}

impl ChunkLight {
    const DIM: i32 = CHUNK_SIZE as i32 + 2;

    /// All cells fully sky-lit, no block light — the default for unlit meshing
    /// and tests.
    pub fn full_bright() -> Self {
        let n = (Self::DIM * Self::DIM * Self::DIM) as usize;
        Self {
            sky: vec![15; n],
            block: vec![0; n],
        }
    }

    /// All cells dark (used as a base when baking).
    pub fn dark() -> Self {
        let n = (Self::DIM * Self::DIM * Self::DIM) as usize;
        Self {
            sky: vec![0; n],
            block: vec![0; n],
        }
    }

    #[inline]
    fn index(x: i32, y: i32, z: i32) -> Option<usize> {
        if x < -1 || y < -1 || z < -1 || x > CHUNK_SIZE as i32 || y > CHUNK_SIZE as i32 || z > CHUNK_SIZE as i32 {
            return None;
        }
        Some(((x + 1) + (y + 1) * Self::DIM + (z + 1) * Self::DIM * Self::DIM) as usize)
    }

    /// Set the light at a local cell (`-1..=CHUNK_SIZE`).
    pub fn set(&mut self, x: i32, y: i32, z: i32, sky: u8, block: u8) {
        if let Some(i) = Self::index(x, y, z) {
            self.sky[i] = sky;
            self.block[i] = block;
        }
    }

    /// Light `(sky, block)` at a local cell; out-of-range reads as full sky.
    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> (u8, u8) {
        match Self::index(x, y, z) {
            Some(i) => (self.sky[i], self.block[i]),
            None => (15, 0),
        }
    }
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

    #[allow(clippy::too_many_arguments)]
    fn push_quad(
        &mut self,
        corners: [[f32; 3]; 4],
        colors: [[f32; 4]; 4],
        uvs: [[f32; 2]; 4],
        layer: u32,
        lights: [[f32; 2]; 4],
        normal: [f32; 3],
        reverse_winding: bool,
        flip_diagonal: bool,
    ) {
        let base = self.vertices.len() as u32;
        for (((position, color), uv), light) in
            corners.into_iter().zip(colors).zip(uvs).zip(lights)
        {
            self.vertices.push(Vertex {
                position,
                normal,
                color,
                uv,
                layer,
                light,
            });
        }
        // Choose the split diagonal (0–2 vs 1–3) for smooth AO interpolation.
        let mut idx = if flip_diagonal {
            [base + 1, base + 2, base + 3, base + 1, base + 3, base]
        } else {
            [base, base + 1, base + 2, base, base + 2, base + 3]
        };
        // Reverse winding for negative-facing quads so every front face stays
        // consistently counter-clockwise for back-face culling.
        if reverse_winding {
            idx.reverse();
        }
        self.indices.extend_from_slice(&idx);
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

/// Identifies a face for greedy-merge equality: same block, same facing, and
/// the same four-corner ambient-occlusion pattern. Including AO in the key means
/// faces only merge where the baked corner shadow is identical — flat open areas
/// still collapse into big quads, while creases keep their per-cell shading.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceKey {
    block: BlockId,
    positive: bool,
    /// Corner occlusion 0..3 (0 = darkest crevice, 3 = fully open), ordered
    /// `[(-u,-v), (+u,-v), (+u,+v), (-u,+v)]` to match emitted quad corners.
    ao: [u8; 4],
    /// Baked `(sky, block)` light at the four face corners (smooth lighting:
    /// each is the average of the cells touching that corner). Part of the merge
    /// key so faces only merge where the light gradient matches.
    light: [(u8, u8); 4],
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

/// Mesh a chunk with full-bright lighting (no baked light). Convenience for
/// tests and tools.
pub fn mesh_chunk<S: BlockSampler>(sampler: &S, registry: &BlockRegistry) -> ChunkMesh {
    mesh_chunk_lit(sampler, registry, &ChunkLight::full_bright())
}

/// Mesh a chunk. `sampler` resolves blocks (including one ring of neighbours),
/// `registry` supplies occlusion/visibility/colour, and `light` provides baked
/// per-cell sky/block light.
pub fn mesh_chunk_lit<S: BlockSampler>(
    sampler: &S,
    registry: &BlockRegistry,
    light: &ChunkLight,
) -> ChunkMesh {
    let mut mesh = ChunkMesh::default();
    greedy_pass(sampler, registry, light, Layer::Opaque, &mut mesh.opaque);
    greedy_pass(sampler, registry, light, Layer::Transparent, &mut mesh.transparent);
    cross_pass(sampler, registry, light, &mut mesh.transparent);
    mesh
}

/// Emit crossed-quad geometry for small plants (flowers, grass, mushrooms).
/// Each such voxel becomes two diagonal double-sided quads through the cell,
/// giving a soft billboard look instead of a solid translucent cube.
fn cross_pass<S: BlockSampler>(
    sampler: &S,
    registry: &BlockRegistry,
    light: &ChunkLight,
    out: &mut MeshBuffers,
) {
    for y in 0..N {
        for z in 0..N {
            for x in 0..N {
                let id = sampler.block_at(x, y, z);
                let block = registry.get(id);
                let pixelcraft_core::block::RenderKind::Cross { width, height } = block.render
                else {
                    continue;
                };
                // Full-brightness luminance; tint + shape come from the tile.
                let alpha = block.color.a as f32 / 255.0;
                let colors = [[1.0, 1.0, 1.0, alpha]; 4];
                let layer = id.0 as u32;
                let (sky, blk) = light.get(x, y, z);
                let lvc = [sky as f32 / 15.0, blk as f32 / 15.0];
                let lv = [lvc; 4];
                // The plant tile maps once across each quad.
                let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
                // Centre the billboard in the cell, sized per plant kind.
                let half = (width.clamp(0.1, 1.0)) * 0.5;
                let (lo, hi) = (0.5 - half, 0.5 + half);
                let top = height.clamp(0.1, 1.0);
                let (fx, fy, fz) = (x as f32, y as f32, z as f32);
                // Upward normal so plants read as bright and sky-lit.
                let normal = [0.0, 1.0, 0.0];
                // Diagonal A: (lo,lo) → (hi,hi).
                out.push_quad(
                    [
                        [fx + lo, fy, fz + lo],
                        [fx + hi, fy, fz + hi],
                        [fx + hi, fy + top, fz + hi],
                        [fx + lo, fy + top, fz + lo],
                    ],
                    colors,
                    uvs,
                    layer,
                    lv,
                    normal,
                    false,
                    false,
                );
                // Diagonal B: (hi,lo) → (lo,hi).
                out.push_quad(
                    [
                        [fx + hi, fy, fz + lo],
                        [fx + lo, fy, fz + hi],
                        [fx + lo, fy + top, fz + hi],
                        [fx + hi, fy + top, fz + lo],
                    ],
                    colors,
                    uvs,
                    layer,
                    lv,
                    normal,
                    false,
                    false,
                );
            }
        }
    }
}

fn greedy_pass<S: BlockSampler>(
    sampler: &S,
    registry: &BlockRegistry,
    light: &ChunkLight,
    layer: Layer,
    out: &mut MeshBuffers,
) {
    // Decide whether the face between voxels `a` (lower along axis d) and `b`
    // (higher) should be emitted in this layer, and to whom it belongs. Returns
    // `(block, positive)`; ambient occlusion is computed separately below.
    let face_between = |a: BlockId, b: BlockId| -> Option<(BlockId, bool)> {
        let ba = registry.get(a);
        let bb = registry.get(b);
        match layer {
            Layer::Opaque => {
                // Opaque cube exposed to a non-opaque neighbour.
                if ba.occludes() && !bb.occludes() {
                    Some((a, true))
                } else if bb.occludes() && !ba.occludes() {
                    Some((b, false))
                } else {
                    None
                }
            }
            Layer::Transparent => {
                // Visible, non-occluding *cube* blocks (water/leaves/glass).
                // Cross plants are handled separately. Cull the interface
                // between two of the same block, and anything behind an opaque
                // neighbour.
                let a_t = ba.is_cube() && !ba.occludes();
                let b_t = bb.is_cube() && !bb.occludes();
                if a_t && a != b && !bb.occludes() {
                    Some((a, true))
                } else if b_t && a != b && !ba.occludes() {
                    Some((b, false))
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
                    mask[n] = face_between(a, b).map(|(block, positive)| {
                        // AO only matters for the solid world; transparent
                        // foliage/water stays unshaded (full brightness).
                        let ao = if layer == Layer::Opaque {
                            compute_ao(sampler, registry, lo, d, u, v, positive)
                        } else {
                            [3; 4]
                        };
                        // Smooth lighting: average the cells touching each of
                        // the four face corners on the air side.
                        let light = compute_light(light, lo, d, u, v, positive);
                        FaceKey {
                            block,
                            positive,
                            ao,
                            light,
                        }
                    });
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
    let alpha = block.color.a as f32 / 255.0;

    // `color` is a luminance multiplier (face shade × AO); the tint comes from
    // the texture. Map AO level 0..3 to brightness, with a cosy non-black floor.
    let ao_factor = |level: u8| 0.5 + 0.5 * (level as f32 / 3.0);
    let corner_color = |level: u8| {
        let m = shade * ao_factor(level);
        [m, m, m, alpha]
    };
    let colors = [
        corner_color(key.ao[0]),
        corner_color(key.ao[1]),
        corner_color(key.ao[2]),
        corner_color(key.ao[3]),
    ];
    let layer = key.block.0 as u32;

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

    // UVs in block units so the per-block tile repeats across the merged quad.
    let (fw, fh) = (w as f32, h as f32);
    let uvs = [[0.0, 0.0], [fw, 0.0], [fw, fh], [0.0, fh]];

    // Flip the quad's split diagonal when AO is anisotropic, so the gradient
    // interpolates smoothly instead of producing a hard diagonal seam.
    let flip_diagonal = key.ao[0] as i32 + key.ao[2] as i32 > key.ao[1] as i32 + key.ao[3] as i32;

    let lights = [
        [key.light[0].0 as f32 / 15.0, key.light[0].1 as f32 / 15.0],
        [key.light[1].0 as f32 / 15.0, key.light[1].1 as f32 / 15.0],
        [key.light[2].0 as f32 / 15.0, key.light[2].1 as f32 / 15.0],
        [key.light[3].0 as f32 / 15.0, key.light[3].1 as f32 / 15.0],
    ];
    out.push_quad([p0, p1, p2, p3], colors, uvs, layer, lights, normal, !key.positive, flip_diagonal);
}

/// Smooth lighting: for each of the four face corners, average the `(sky,
/// block)` light of the (up to four) cells touching that corner on the air
/// side. Corners are ordered `[(-u,-v), (+u,-v), (+u,+v), (-u,+v)]`.
fn compute_light(
    light: &ChunkLight,
    lo: [i32; 3],
    d: usize,
    u: usize,
    v: usize,
    positive: bool,
) -> [(u8, u8); 4] {
    let outer_d = if positive { lo[d] + 1 } else { lo[d] };
    let i = lo[u];
    let j = lo[v];
    let sample = |ou: i32, ov: i32| -> (u16, u16) {
        let mut p = [0i32; 3];
        p[d] = outer_d;
        p[u] = i + ou;
        p[v] = j + ov;
        let (s, b) = light.get(p[0], p[1], p[2]);
        (s as u16, b as u16)
    };
    // Average the 2×2 block of cells around each corner.
    let corner = |su: i32, sv: i32| -> (u8, u8) {
        let a = sample(0, 0);
        let b = sample(su, 0);
        let c = sample(0, sv);
        let e = sample(su, sv);
        let sky = (a.0 + b.0 + c.0 + e.0) / 4;
        let blk = (a.1 + b.1 + c.1 + e.1) / 4;
        (sky as u8, blk as u8)
    };
    [
        corner(-1, -1),
        corner(1, -1),
        corner(1, 1),
        corner(-1, 1),
    ]
}

/// Compute four-corner ambient occlusion for a face. `lo` is the lower voxel's
/// local coords with `lo[d]` at the slice; `positive` selects which side the
/// face looks toward. Occluders are sampled in the layer the face opens into.
fn compute_ao<S: BlockSampler>(
    sampler: &S,
    registry: &BlockRegistry,
    lo: [i32; 3],
    d: usize,
    u: usize,
    v: usize,
    positive: bool,
) -> [u8; 4] {
    // The empty layer the face opens onto: slice+1 for +d faces, slice for -d.
    let outer_d = if positive { lo[d] + 1 } else { lo[d] };
    let i = lo[u];
    let j = lo[v];
    let occ = |ou: i32, ov: i32| -> u8 {
        let mut p = [0i32; 3];
        p[d] = outer_d;
        p[u] = i + ou;
        p[v] = j + ov;
        registry.get(sampler.block_at(p[0], p[1], p[2])).occludes() as u8
    };
    // Vertex order matches emitted corners: (-u,-v), (+u,-v), (+u,+v), (-u,+v).
    [
        ao_value(occ(-1, 0), occ(0, -1), occ(-1, -1)),
        ao_value(occ(1, 0), occ(0, -1), occ(1, -1)),
        ao_value(occ(1, 0), occ(0, 1), occ(1, 1)),
        ao_value(occ(-1, 0), occ(0, 1), occ(-1, 1)),
    ]
}

/// Classic Minecraft-style vertex AO: two fully-occluding edge neighbours
/// darken hardest; otherwise subtract the count of occluding neighbours.
#[inline]
fn ao_value(side1: u8, side2: u8, corner: u8) -> u8 {
    if side1 == 1 && side2 == 1 {
        0
    } else {
        3 - (side1 + side2 + corner)
    }
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
    fn ambient_occlusion_darkens_inner_corners() {
        // A lone block's top face, with two occluders placed diagonally above so
        // that the (+u,+v) corner sees both edge neighbours occluding (AO 0)
        // while the opposite corner stays fully open (AO 3).
        let mut s = ChunkStorage::empty();
        s.set(LocalPos::new(5, 5, 5), blocks::STONE);
        s.set(LocalPos::new(5, 6, 6), blocks::STONE);
        s.set(LocalPos::new(6, 6, 5), blocks::STONE);
        let m = mesh_storage(&s);

        // Gather the green channel of the target block's top face (y == 6,
        // facing +Y).
        let greens: Vec<f32> = m
            .opaque
            .vertices
            .iter()
            .filter(|v| v.normal[1] > 0.5 && (v.position[1] - 6.0).abs() < 1e-3)
            .map(|v| v.color[1])
            .collect();
        assert!(greens.len() >= 4, "expected a top face quad, got {}", greens.len());
        let min = greens.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = greens.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // Darkest corner is the AO floor (0.5×) of the brightest open corner.
        assert!(
            (min / max - 0.5).abs() < 0.05,
            "AO not applied as expected: min {min}, max {max}"
        );
    }

    #[test]
    fn cross_plants_make_two_quads_and_no_cube() {
        let mut s = ChunkStorage::empty();
        s.set(LocalPos::new(8, 8, 8), blocks::FLOWER_PINK);
        let m = mesh_storage(&s);
        // A flower contributes exactly two crossed quads, all in the transparent
        // layer, and nothing in the opaque layer.
        assert!(m.opaque.is_empty());
        assert_eq!(m.transparent.quad_count(), 2);
    }

    #[test]
    fn cross_plant_does_not_cull_block_below() {
        // Grass block with a flower on top: the grass top face must still render
        // (the flower doesn't occlude it).
        let mut s = ChunkStorage::empty();
        s.set(LocalPos::new(8, 8, 8), blocks::GRASS);
        s.set(LocalPos::new(8, 9, 8), blocks::TALL_GRASS);
        let m = mesh_storage(&s);
        // Grass cube: 6 faces. Plant: 2 quads in transparent.
        assert_eq!(m.opaque.quad_count(), 6);
        assert_eq!(m.transparent.quad_count(), 2);
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
