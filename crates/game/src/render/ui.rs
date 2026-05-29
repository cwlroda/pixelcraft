//! HUD geometry: builds the crosshair and hotbar as flat coloured quads in
//! normalised device coordinates, ready for the UI pipeline. Kept renderer-
//! agnostic (just produces vertices) so both the window and the headless
//! capturer show the same interface.

use bytemuck::{Pod, Zeroable};
use pixelcraft_core::block::BlockRegistry;

use crate::inventory::{Inventory, HOTBAR};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
}

/// Accumulates HUD quads in pixel space, converting to NDC on push.
struct UiBuilder {
    verts: Vec<UiVertex>,
    w: f32,
    h: f32,
}

impl UiBuilder {
    fn new(w: u32, h: u32) -> Self {
        Self {
            verts: Vec::new(),
            w: w.max(1) as f32,
            h: h.max(1) as f32,
        }
    }

    /// Push an axis-aligned rectangle given top-left pixel origin and size.
    fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        let to_ndc = |px: f32, py: f32| [px / self.w * 2.0 - 1.0, 1.0 - py / self.h * 2.0];
        let p0 = to_ndc(x, y);
        let p1 = to_ndc(x + w, y);
        let p2 = to_ndc(x + w, y + h);
        let p3 = to_ndc(x, y + h);
        for p in [p0, p1, p2, p0, p2, p3] {
            self.verts.push(UiVertex { pos: p, color });
        }
    }

    /// Draw `text` with the bitmap font; top-left at (x,y), one source pixel =
    /// `scale` screen pixels.
    fn text(&mut self, x: f32, y: f32, scale: f32, text: &str, color: [f32; 4]) {
        let mut cx = x;
        for ch in text.chars() {
            if let Some(bits) = super::font::glyph(ch) {
                for (row, bitmask) in bits.iter().enumerate() {
                    for col in 0..super::font::GLYPH_W {
                        if bitmask & (1 << (super::font::GLYPH_W - 1 - col)) != 0 {
                            self.rect(
                                cx + col as f32 * scale,
                                y + row as f32 * scale,
                                scale,
                                scale,
                                color,
                            );
                        }
                    }
                }
            }
            cx += (super::font::GLYPH_W as f32 + 1.0) * scale;
        }
    }

    /// Draw text with a 1px dark drop-shadow for legibility over any backdrop.
    fn text_shadow(&mut self, x: f32, y: f32, scale: f32, s: &str, color: [f32; 4]) {
        self.text(x + scale, y + scale, scale, s, [0.0, 0.0, 0.0, 0.55]);
        self.text(x, y, scale, s, color);
    }
}

/// Everything the HUD needs for a frame.
pub struct HudState<'a> {
    pub inventory: &'a Inventory,
    pub registry: &'a BlockRegistry,
    /// Time of day in `[0,1)` (0 = dawn, 0.25 = noon).
    pub time_of_day: f32,
    /// Optional quest objective line and its progress (done, total).
    pub objective: Option<String>,
    pub objective_progress: Option<(u32, u32)>,
}

/// Build the full HUD for the current frame.
pub fn build_hud(screen_w: u32, screen_h: u32, state: &HudState) -> Vec<UiVertex> {
    let mut b = UiBuilder::new(screen_w, screen_h);
    let (w, h) = (b.w, b.h);
    let inventory = state.inventory;
    let registry = state.registry;

    // --- Crosshair: a soft white plus at the screen centre ---------------
    let cx = w * 0.5;
    let cy = h * 0.5;
    let arm = 9.0;
    let thick = 2.0;
    let cross = [1.0, 1.0, 1.0, 0.7];
    b.rect(cx - arm, cy - thick * 0.5, arm * 2.0, thick, cross);
    b.rect(cx - thick * 0.5, cy - arm, thick, arm * 2.0, cross);

    // --- Top-left clock --------------------------------------------------
    // Map the cycle to a friendly 24h clock (dawn ≈ 06:00, noon = 12:00).
    let hours_f = (state.time_of_day * 24.0 + 6.0) % 24.0;
    let hh = hours_f as u32;
    let mm = ((hours_f - hh as f32) * 60.0) as u32;
    let clock = format!("{hh:02}:{mm:02}");
    let tag = if (6..20).contains(&hh) { "DAY" } else { "NIGHT" };
    b.text_shadow(14.0, 14.0, 3.0, &format!("{tag} {clock}"), [1.0, 1.0, 1.0, 0.95]);

    // --- Objective banner ------------------------------------------------
    if let Some(obj) = &state.objective {
        let line = if let Some((done, total)) = state.objective_progress {
            format!("{obj}  {done}/{total}")
        } else {
            obj.clone()
        };
        b.text_shadow(14.0, 40.0, 3.0, &line, [1.0, 0.93, 0.7, 0.95]);
    }

    // --- Hotbar: a centred row of slots near the bottom ------------------
    let slots = HOTBAR.len();
    let slot = 44.0;
    let gap = 6.0;
    let total = slots as f32 * slot + (slots as f32 - 1.0) * gap;
    let start_x = (w - total) * 0.5;
    let y = h - slot - 18.0;

    for (i, &id) in HOTBAR.iter().enumerate() {
        let sx = start_x + i as f32 * (slot + gap);
        let selected = i == inventory.selected;

        if selected {
            b.rect(sx - 3.0, y - 3.0, slot + 6.0, slot + 6.0, [1.0, 1.0, 1.0, 0.95]);
        }
        b.rect(sx, y, slot, slot, [0.12, 0.12, 0.16, 0.55]);

        // Block colour swatch inset in the slot.
        let c = registry.get(id).color;
        let swatch = [c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0, 1.0];
        let pad = 7.0;
        b.rect(sx + pad, y + pad, slot - pad * 2.0, slot - pad * 2.0, swatch);

        // Held count, bottom-right of the slot.
        let count = inventory.count(id);
        if count > 0 {
            let s = count.to_string();
            let scale = 2.0;
            let tw = super::font::text_width(&s, scale);
            b.text_shadow(sx + slot - tw - 3.0, y + slot - 7.0 * scale - 2.0, scale, &s, [1.0; 4]);
        }
        // Slot number key hint, top-left.
        b.text(sx + 3.0, y + 3.0, 1.5, &(i + 1).to_string(), [1.0, 1.0, 1.0, 0.5]);
    }

    // --- Selected block name, centred above the hotbar -------------------
    let name = registry.get(inventory.selected_block()).name.replace('_', " ");
    let scale = 3.0;
    let tw = super::font::text_width(&name, scale);
    b.text_shadow((w - tw) * 0.5, y - 26.0, scale, &name, [1.0, 1.0, 1.0, 0.95]);

    b.verts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hud_builds_triangulated_quads() {
        let reg = BlockRegistry::with_defaults();
        let mut inv = Inventory::new(reg.len());
        inv.add(HOTBAR[0], 32);
        inv.select(2);
        let state = HudState {
            inventory: &inv,
            registry: &reg,
            time_of_day: 0.25,
            objective: Some("COLLECT BERRIES".to_string()),
            objective_progress: Some((2, 5)),
        };
        let verts = build_hud(1280, 720, &state);
        assert!(!verts.is_empty());
        assert_eq!(verts.len() % 6, 0, "quads must be 2 triangles (6 verts) each");
        // All NDC positions stay within the clip volume.
        for v in &verts {
            assert!(v.pos[0] >= -1.0 && v.pos[0] <= 1.0);
            assert!(v.pos[1] >= -1.0 && v.pos[1] <= 1.0);
        }
    }
}
