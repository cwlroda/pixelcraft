//! A cosy little inventory: counts of collected blocks plus a selected build
//! material hotbar.

use pixelcraft_core::block::{blocks, BlockId};

/// The blocks the cat can hold/place, in hotbar order.
pub const HOTBAR: [BlockId; 8] = [
    blocks::DIRT,
    blocks::GRASS,
    blocks::STONE,
    blocks::SAND,
    blocks::PLANK,
    blocks::TRUNK,
    blocks::LEAVES,
    blocks::LANTERN,
];

#[derive(Clone)]
pub struct Inventory {
    /// Count per block id, indexed by `BlockId.0`.
    counts: Vec<u32>,
    /// Currently selected hotbar slot.
    pub selected: usize,
}

impl Inventory {
    pub fn new(registry_len: usize) -> Self {
        Self {
            counts: vec![0; registry_len.max(1)],
            selected: 0,
        }
    }

    pub fn count(&self, id: BlockId) -> u32 {
        self.counts.get(id.index()).copied().unwrap_or(0)
    }

    pub fn add(&mut self, id: BlockId, n: u32) {
        if let Some(c) = self.counts.get_mut(id.index()) {
            *c = c.saturating_add(n);
        }
    }

    /// Remove up to one of `id`; returns true if one was available and consumed.
    pub fn take_one(&mut self, id: BlockId) -> bool {
        self.take(id, 1)
    }

    /// Remove `n` of `id` if at least that many are held; returns whether it
    /// succeeded (all-or-nothing).
    pub fn take(&mut self, id: BlockId, n: u32) -> bool {
        if let Some(c) = self.counts.get_mut(id.index()) {
            if *c >= n {
                *c -= n;
                return true;
            }
        }
        false
    }

    pub fn selected_block(&self) -> BlockId {
        HOTBAR[self.selected % HOTBAR.len()]
    }

    pub fn select(&mut self, slot: usize) {
        self.selected = slot % HOTBAR.len();
    }

    /// Scroll the hotbar selection by `delta` (wrapping).
    pub fn scroll(&mut self, delta: i32) {
        let len = HOTBAR.len() as i32;
        let next = (self.selected as i32 + delta).rem_euclid(len);
        self.selected = next as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_count() {
        let mut inv = Inventory::new(32);
        inv.add(blocks::STONE, 3);
        assert_eq!(inv.count(blocks::STONE), 3);
        assert_eq!(inv.count(blocks::DIRT), 0);
    }

    #[test]
    fn take_one_decrements_until_empty() {
        let mut inv = Inventory::new(32);
        inv.add(blocks::PLANK, 2);
        assert!(inv.take_one(blocks::PLANK));
        assert!(inv.take_one(blocks::PLANK));
        assert!(!inv.take_one(blocks::PLANK));
        assert_eq!(inv.count(blocks::PLANK), 0);
    }

    #[test]
    fn hotbar_scroll_wraps() {
        let mut inv = Inventory::new(32);
        inv.select(0);
        inv.scroll(-1);
        assert_eq!(inv.selected, HOTBAR.len() - 1);
        inv.scroll(1);
        assert_eq!(inv.selected, 0);
    }
}
