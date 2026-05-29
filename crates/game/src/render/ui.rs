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
}

/// Build the full HUD for the current frame.
pub fn build_hud(
    screen_w: u32,
    screen_h: u32,
    inventory: &Inventory,
    registry: &BlockRegistry,
) -> Vec<UiVertex> {
    let mut b = UiBuilder::new(screen_w, screen_h);
    let (w, h) = (b.w, b.h);

    // --- Crosshair: a soft white plus at the screen centre ---------------
    let cx = w * 0.5;
    let cy = h * 0.5;
    let arm = 9.0;
    let thick = 2.0;
    let cross = [1.0, 1.0, 1.0, 0.7];
    b.rect(cx - arm, cy - thick * 0.5, arm * 2.0, thick, cross);
    b.rect(cx - thick * 0.5, cy - arm, thick, arm * 2.0, cross);

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

        // Slot backing (selected slot gets a bright frame).
        if selected {
            b.rect(sx - 3.0, y - 3.0, slot + 6.0, slot + 6.0, [1.0, 1.0, 1.0, 0.95]);
        }
        b.rect(sx, y, slot, slot, [0.12, 0.12, 0.16, 0.55]);

        // Block colour swatch inset in the slot.
        let c = registry.get(id).color;
        let swatch = [
            c.r as f32 / 255.0,
            c.g as f32 / 255.0,
            c.b as f32 / 255.0,
            1.0,
        ];
        let pad = 7.0;
        b.rect(sx + pad, y + pad, slot - pad * 2.0, slot - pad * 2.0, swatch);

        // Count pips: a tiny bar whose width reflects how many are held
        // (cheap stand-in for a number until text rendering lands).
        let count = inventory.count(id).min(64);
        if count > 0 {
            let frac = count as f32 / 64.0;
            b.rect(
                sx + pad,
                y + slot - pad - 3.0,
                (slot - pad * 2.0) * frac,
                3.0,
                [1.0, 0.95, 0.6, 0.9],
            );
        }
    }

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
        let verts = build_hud(1280, 720, &inv, &reg);
        assert!(!verts.is_empty());
        assert_eq!(verts.len() % 6, 0, "quads must be 2 triangles (6 verts) each");
        // All NDC positions stay within the clip volume.
        for v in &verts {
            assert!(v.pos[0] >= -1.0 && v.pos[0] <= 1.0);
            assert!(v.pos[1] >= -1.0 && v.pos[1] <= 1.0);
        }
    }
}
