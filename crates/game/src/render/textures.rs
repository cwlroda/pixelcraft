//! Procedural block textures.
//!
//! Rather than ship image assets, we synthesise a small tile per block at
//! startup into a texture array (one layer per block id, plus a trailing solid
//! white tile for untextured geometry like critters). Patterns are chosen by
//! block name and tinted from the block's base colour, so the cosy pastel
//! palette is preserved while surfaces gain readable detail (grass blades, bark
//! grain, planks, cobble, roof tiles, crystal sparkle). Cross-plant tiles are
//! mostly transparent with a little plant shape painted in.

use pixelcraft_core::block::{BlockRegistry, Color};
use pixelcraft_worldgen::SplitMix64;

/// Edge length of each tile in texels.
pub const TILE: u32 = 16;

/// A generated texture array: `layers` tiles of `TILE×TILE` RGBA8, stored
/// contiguously (layer-major), plus the index of the solid white tile.
pub struct Atlas {
    pub data: Vec<u8>,
    pub layers: u32,
    pub white_layer: u32,
}

struct Canvas {
    px: [[u8; 4]; (TILE * TILE) as usize],
}

impl Canvas {
    fn new() -> Self {
        Self {
            px: [[0, 0, 0, 255]; (TILE * TILE) as usize],
        }
    }
    #[inline]
    fn set(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        self.px[(y * TILE + x) as usize] = rgba;
    }
}

#[inline]
fn shade(c: Color, f: f32) -> [u8; 4] {
    let g = |v: u8| ((v as f32 * f).clamp(0.0, 255.0)) as u8;
    [g(c.r), g(c.g), g(c.b), c.a]
}

#[inline]
fn mix(a: Color, b: Color, t: f32) -> Color {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color::rgba(l(a.r, b.r), l(a.g, b.g), l(a.b, b.b), a.a)
}

/// Build the texture array for every block in the registry.
pub fn build_atlas(registry: &BlockRegistry) -> Atlas {
    let n = registry.len();
    let layers = n as u32 + 1;
    let mut data = Vec::with_capacity((layers * TILE * TILE * 4) as usize);

    for id in 0..n {
        let block = registry.get(pixelcraft_core::block::BlockId(id as u16));
        let canvas = generate(block.name, block.color, id as u64);
        for p in canvas.px.iter() {
            data.extend_from_slice(p);
        }
    }
    // Trailing solid-white tile (layer `n`) for untextured geometry.
    for _ in 0..(TILE * TILE) {
        data.extend_from_slice(&[255, 255, 255, 255]);
    }

    Atlas {
        data,
        layers,
        white_layer: n as u32,
    }
}

/// Synthesise one tile for `name`, tinted from `base`.
fn generate(name: &str, base: Color, idx: u64) -> Canvas {
    let mut c = Canvas::new();
    let mut rng = SplitMix64::new(0x7E27_0000 ^ idx.wrapping_mul(2654435761));
    // Per-pixel noise helper in [-1, 1].
    let mut noise = move |_x: u32, _y: u32| rng.next_f32() * 2.0 - 1.0;

    match name {
        "grass" => {
            for y in 0..TILE {
                for x in 0..TILE {
                    let n = noise(x, y);
                    // Occasional brighter blade tips.
                    let f = 1.0 + n * 0.10 + if n > 0.85 { 0.18 } else { 0.0 };
                    c.set(x, y, shade(base, f));
                }
            }
        }
        "leaves" | "cherry_leaves" | "pine_leaves" | "autumn_leaves" => {
            for y in 0..TILE {
                for x in 0..TILE {
                    let n = noise(x, y);
                    let f = 1.0 + n * 0.18;
                    let mut px = shade(base, f);
                    // A few transparent gaps for a dappled canopy.
                    if n < -0.8 {
                        px[3] = 90;
                    }
                    c.set(x, y, px);
                }
            }
        }
        "trunk" => {
            // Vertical bark streaks: brightness varies by column, plus grain.
            for x in 0..TILE {
                let col = 1.0 + ((x as f32 * 1.7).sin()) * 0.12;
                for y in 0..TILE {
                    let f = col + noise(x, y) * 0.06;
                    c.set(x, y, shade(base, f));
                }
            }
        }
        "plank" => {
            for y in 0..TILE {
                let plank_row = (y / 4) % 2;
                let seam = y % 4 == 0;
                for x in 0..TILE {
                    // Stagger vertical seams between rows.
                    let vseam = (x + plank_row * 8) % 16 == 0;
                    let mut f = 1.0 + noise(x, y) * 0.05;
                    if seam || vseam {
                        f *= 0.78;
                    }
                    c.set(x, y, shade(base, f));
                }
            }
        }
        "cobble" | "stone" | "path" => {
            // Blobby cells with darker mortar from a coarse value field.
            for y in 0..TILE {
                for x in 0..TILE {
                    let cell = (((x + 2) / 5) * 7 + ((y + 1) / 5) * 13) as f32;
                    let f = 0.85 + (cell.sin() * 0.5 + 0.5) * 0.3 + noise(x, y) * 0.05;
                    let mortar = (x % 5 == 0) || (y % 5 == 0);
                    c.set(x, y, shade(base, if mortar { 0.7 } else { f }));
                }
            }
        }
        "coal_ore" => {
            for y in 0..TILE {
                for x in 0..TILE {
                    let f = 0.9 + noise(x, y) * 0.08;
                    let dark = noise(x, y) > 0.78;
                    c.set(x, y, if dark { [22, 22, 26, 255] } else { shade(base, f) });
                }
            }
        }
        "roof" => {
            // Overlapping clay tiles: rows of scalloped shingles.
            for y in 0..TILE {
                for x in 0..TILE {
                    let row = y / 4;
                    let edge = y % 4 == 0;
                    let offset = (x + row * 2) % 8;
                    let f = 1.0 + noise(x, y) * 0.05 - if edge { 0.18 } else { 0.0 }
                        + if offset == 0 { -0.1 } else { 0.0 };
                    c.set(x, y, shade(base, f));
                }
            }
        }
        "crystal" => {
            // Faceted gem: diagonal banding with bright sparkles.
            for y in 0..TILE {
                for x in 0..TILE {
                    let band = (((x + y) / 3) % 2) as f32;
                    let mut f = 0.8 + band * 0.35 + noise(x, y) * 0.06;
                    if noise(x, y) > 0.9 {
                        f += 0.5; // sparkle
                    }
                    c.set(x, y, shade(base, f));
                }
            }
        }
        "lantern" => {
            // Warm panes framed by darker metal.
            let frame = mix(base, Color::rgb(90, 60, 30), 0.6);
            for y in 0..TILE {
                for x in 0..TILE {
                    let border = x == 0 || y == 0 || x == TILE - 1 || y == TILE - 1 || x == 7 || y == 7;
                    if border {
                        c.set(x, y, shade(frame, 1.0));
                    } else {
                        let f = 1.0 + noise(x, y) * 0.05;
                        c.set(x, y, shade(base, f));
                    }
                }
            }
        }
        "glass" => {
            for y in 0..TILE {
                for x in 0..TILE {
                    let border = x == 0 || y == 0 || x == TILE - 1 || y == TILE - 1;
                    let mut px = shade(base, 1.0);
                    px[3] = if border { 180 } else { 60 };
                    // A diagonal glint.
                    if x == y || x + 1 == y {
                        px = [255, 255, 255, 150];
                    }
                    c.set(x, y, px);
                }
            }
        }
        "water" => {
            for y in 0..TILE {
                for x in 0..TILE {
                    let ripple = ((x as f32 * 0.8).sin() + (y as f32 * 0.6).cos()) * 0.06;
                    c.set(x, y, shade(base, 1.0 + ripple));
                }
            }
        }
        "pumpkin" => {
            for y in 0..TILE {
                for x in 0..TILE {
                    // Vertical ridges.
                    let ridge = if x % 4 == 0 { 0.78 } else { 1.0 };
                    c.set(x, y, shade(base, ridge + noise(x, y) * 0.04));
                }
            }
        }
        "flower_pink" | "flower_blue" => paint_flower(&mut c, base),
        "mushroom" => paint_mushroom(&mut c, base),
        "tall_grass" => paint_tall_grass(&mut c, base, &mut noise),
        "berry_bush" => paint_berry_bush(&mut c, base, &mut noise),
        _ => {
            // Generic speckle (sand, snow, dirt, unknown).
            for y in 0..TILE {
                for x in 0..TILE {
                    let f = 1.0 + noise(x, y) * 0.08;
                    c.set(x, y, shade(base, f));
                }
            }
        }
    }
    c
}

fn clear_transparent(c: &mut Canvas) {
    for p in c.px.iter_mut() {
        *p = [0, 0, 0, 0];
    }
}

/// A little flower: green stem with a coloured bloom on top.
fn paint_flower(c: &mut Canvas, base: Color) {
    clear_transparent(c);
    let stem = Color::rgb(96, 158, 80);
    // Stem up the centre.
    for y in 8..TILE {
        c.set(7, y, shade(stem, 1.0));
        c.set(8, y, shade(stem, 0.9));
    }
    // Bloom: a 6×6 rounded patch near the top.
    for y in 2..9 {
        for x in 4..12 {
            let dx = x as i32 - 7;
            let dy = y as i32 - 5;
            if dx * dx + dy * dy <= 11 {
                c.set(x, y, shade(base, 1.0));
            }
        }
    }
    // Bright centre.
    c.set(7, 5, [255, 240, 180, 255]);
    c.set(8, 5, [255, 240, 180, 255]);
}

fn paint_mushroom(c: &mut Canvas, base: Color) {
    clear_transparent(c);
    let stem = Color::rgb(238, 226, 200);
    for y in 8..14 {
        c.set(7, y, shade(stem, 1.0));
        c.set(8, y, shade(stem, 0.95));
    }
    // Domed cap.
    for y in 3..9 {
        let half = (y as i32 - 2).min(6);
        for x in (7 - half)..=(8 + half) {
            if x >= 0 && (x as u32) < TILE {
                c.set(x as u32, y, shade(base, 1.0));
            }
        }
    }
    // White spots.
    c.set(5, 6, [245, 245, 245, 255]);
    c.set(10, 5, [245, 245, 245, 255]);
}

fn paint_tall_grass(c: &mut Canvas, base: Color, noise: &mut impl FnMut(u32, u32) -> f32) {
    clear_transparent(c);
    // Several blades fanning up from the bottom.
    for blade in 0..5u32 {
        let bx = 2 + blade * 3;
        let lean = (noise(bx, 0) * 2.0) as i32;
        for y in 4..TILE {
            let x = bx as i32 + lean * (TILE - y) as i32 / 12;
            if x >= 0 && (x as u32) < TILE {
                let f = 0.85 + (TILE - y) as f32 / TILE as f32 * 0.4;
                c.set(x as u32, y, shade(base, f));
            }
        }
    }
}

fn paint_berry_bush(c: &mut Canvas, base: Color, noise: &mut impl FnMut(u32, u32) -> f32) {
    clear_transparent(c);
    let leaf = Color::rgb(86, 140, 78);
    // Round leafy clump.
    for y in 3..15 {
        for x in 2..14 {
            let dx = x as i32 - 8;
            let dy = y as i32 - 9;
            if dx * dx + dy * dy <= 36 {
                c.set(x, y, shade(leaf, 1.0 + noise(x, y) * 0.12));
            }
        }
    }
    // Berries dotted on top.
    for &(bx, by) in &[(5u32, 7u32), (9, 6), (11, 10), (6, 11)] {
        c.set(bx, by, shade(base, 1.1));
        c.set(bx + 1, by, shade(base, 0.9));
    }
}
