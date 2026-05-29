//! A small crafting system: cosy recipes that turn gathered resources into
//! useful building blocks. Pure logic over the [`Inventory`], so it's testable.

use crate::inventory::Inventory;
use pixelcraft_core::block::{blocks, BlockId};

/// A recipe: consume `inputs`, produce `output`.
#[derive(Clone)]
pub struct Recipe {
    pub name: &'static str,
    pub inputs: &'static [(BlockId, u32)],
    pub output: (BlockId, u32),
}

impl Recipe {
    /// True if the inventory holds every ingredient.
    pub fn affordable(&self, inv: &Inventory) -> bool {
        self.inputs.iter().all(|&(id, n)| inv.count(id) >= n)
    }
}

/// The cosy recipe book.
pub fn recipes() -> &'static [Recipe] {
    &[
        Recipe {
            name: "PLANKS",
            inputs: &[(blocks::TRUNK, 1)],
            output: (blocks::PLANK, 4),
        },
        Recipe {
            name: "COBBLE",
            inputs: &[(blocks::STONE, 1)],
            output: (blocks::COBBLE, 1),
        },
        Recipe {
            name: "GLASS",
            inputs: &[(blocks::SAND, 2)],
            output: (blocks::GLASS, 2),
        },
        Recipe {
            name: "LANTERN",
            inputs: &[(blocks::PLANK, 3), (blocks::COAL_ORE, 1)],
            output: (blocks::LANTERN, 2),
        },
        Recipe {
            name: "ROOF TILES",
            inputs: &[(blocks::COBBLE, 2), (blocks::COAL_ORE, 1)],
            output: (blocks::ROOF, 4),
        },
        Recipe {
            name: "GLOW GLASS",
            inputs: &[(blocks::CRYSTAL, 1)],
            output: (blocks::GLASS, 4),
        },
    ]
}

/// Attempt to craft recipe `index`. Consumes inputs and grants the output on
/// success; returns whether it crafted.
pub fn craft(inv: &mut Inventory, index: usize) -> bool {
    let Some(recipe) = recipes().get(index) else {
        return false;
    };
    if !recipe.affordable(inv) {
        return false;
    }
    for &(id, n) in recipe.inputs {
        // Affordability was checked above, so these all succeed.
        let _ = inv.take(id, n);
    }
    inv.add(recipe.output.0, recipe.output.1);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn craft_planks_from_a_log() {
        let mut inv = Inventory::new(32);
        inv.add(blocks::TRUNK, 2);
        // Index 0 is PLANKS.
        assert!(craft(&mut inv, 0));
        assert_eq!(inv.count(blocks::TRUNK), 1);
        assert_eq!(inv.count(blocks::PLANK), 4);
    }

    #[test]
    fn cannot_craft_without_ingredients() {
        let mut inv = Inventory::new(32);
        // Lantern needs planks + coal; we have neither.
        let lantern_idx = recipes().iter().position(|r| r.name == "LANTERN").unwrap();
        assert!(!craft(&mut inv, lantern_idx));
        assert_eq!(inv.count(blocks::LANTERN), 0);
    }

    #[test]
    fn lantern_consumes_planks_and_coal() {
        let mut inv = Inventory::new(32);
        inv.add(blocks::PLANK, 3);
        inv.add(blocks::COAL_ORE, 1);
        let idx = recipes().iter().position(|r| r.name == "LANTERN").unwrap();
        assert!(craft(&mut inv, idx));
        assert_eq!(inv.count(blocks::PLANK), 0);
        assert_eq!(inv.count(blocks::COAL_ORE), 0);
        assert_eq!(inv.count(blocks::LANTERN), 2);
    }

    #[test]
    fn affordability_reflects_inventory() {
        let mut inv = Inventory::new(32);
        let planks = &recipes()[0];
        assert!(!planks.affordable(&inv));
        inv.add(blocks::TRUNK, 1);
        assert!(planks.affordable(&inv));
    }
}
