//! Block (voxel material) definitions and the runtime registry.
//!
//! Blocks are referred to by a small integer [`BlockId`] everywhere in the hot
//! paths; their rich properties live in a [`BlockRegistry`] that is built once
//! at startup. This indirection keeps chunk storage tiny while letting gameplay
//! code ask rich questions ("is this solid?", "what colour?").

/// Compact handle for a block type. `0` is always air.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BlockId(pub u16);

impl BlockId {
    pub const AIR: BlockId = BlockId(0);

    #[inline]
    pub const fn is_air(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// How light and rendering treat a block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderKind {
    /// Nothing is drawn (air).
    Invisible,
    /// A full opaque cube that hides the faces of neighbours touching it.
    OpaqueCube,
    /// A full cube that is see-through (leaves, glass): drawn, but does not
    /// cull neighbouring faces of *other* block types.
    TransparentCube,
    /// A translucent fluid surface (water).
    Fluid,
    /// A small plant drawn as two crossed quads (flowers, grass, mushrooms)
    /// rather than a full cube — avoids the "solid haze" of cube-shaped flora.
    Cross,
}

/// Linear RGB-ish colour used for vertex tinting before textures land.
/// Stored as 4 bytes to pack neatly into vertex data later.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
    pub const fn to_array(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

/// Full description of a block type.
#[derive(Clone, Debug)]
pub struct Block {
    pub name: &'static str,
    pub render: RenderKind,
    /// Blocks the player / entities collide against.
    pub solid: bool,
    /// Base tint. Top/side variation is applied by the mesher via face shading.
    pub color: Color,
    /// Light emitted (0..15); cute cosy glow sources like lanterns.
    pub light_emission: u8,
    /// Whether the player can mine and collect it.
    pub harvestable: bool,
}

impl Block {
    /// True if this block completely hides the touching face of an opaque
    /// neighbour (used for face culling during meshing).
    #[inline]
    pub fn occludes(&self) -> bool {
        matches!(self.render, RenderKind::OpaqueCube)
    }

    #[inline]
    pub fn is_visible(&self) -> bool {
        !matches!(self.render, RenderKind::Invisible)
    }

    #[inline]
    pub fn is_fluid(&self) -> bool {
        matches!(self.render, RenderKind::Fluid)
    }

    /// A crossed-quad plant rather than a cube.
    #[inline]
    pub fn is_cross(&self) -> bool {
        matches!(self.render, RenderKind::Cross)
    }

    /// A full-cube visible block (opaque, transparent or fluid) — i.e. anything
    /// the greedy cube mesher should generate faces for. Excludes cross plants.
    #[inline]
    pub fn is_cube(&self) -> bool {
        matches!(
            self.render,
            RenderKind::OpaqueCube | RenderKind::TransparentCube | RenderKind::Fluid
        )
    }
}

/// Stable well-known block ids. Keep in sync with [`BlockRegistry::with_defaults`].
pub mod blocks {
    use super::BlockId;
    pub const AIR: BlockId = BlockId(0);
    pub const GRASS: BlockId = BlockId(1);
    pub const DIRT: BlockId = BlockId(2);
    pub const STONE: BlockId = BlockId(3);
    pub const SAND: BlockId = BlockId(4);
    pub const WATER: BlockId = BlockId(5);
    pub const TRUNK: BlockId = BlockId(6);
    pub const LEAVES: BlockId = BlockId(7);
    pub const SNOW: BlockId = BlockId(8);
    pub const FLOWER_PINK: BlockId = BlockId(9);
    pub const FLOWER_BLUE: BlockId = BlockId(10);
    pub const PLANK: BlockId = BlockId(11);
    pub const LANTERN: BlockId = BlockId(12);
    pub const MUSHROOM: BlockId = BlockId(13);
    pub const PUMPKIN: BlockId = BlockId(14);
    pub const GLASS: BlockId = BlockId(15);
    pub const COBBLE: BlockId = BlockId(16);
    pub const PATH: BlockId = BlockId(17);
    pub const CRYSTAL: BlockId = BlockId(18);
    pub const TALL_GRASS: BlockId = BlockId(19);
    pub const BERRY_BUSH: BlockId = BlockId(20);
    pub const ROOF: BlockId = BlockId(21);
    pub const COAL_ORE: BlockId = BlockId(22);
}

/// Registry mapping [`BlockId`] to [`Block`] definitions.
#[derive(Clone)]
pub struct BlockRegistry {
    blocks: Vec<Block>,
}

impl BlockRegistry {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Register a block, returning its assigned id.
    pub fn register(&mut self, block: Block) -> BlockId {
        let id = BlockId(self.blocks.len() as u16);
        self.blocks.push(block);
        id
    }

    #[inline]
    pub fn get(&self, id: BlockId) -> &Block {
        // World-gen and meshing only ever produce ids we registered, so an
        // out-of-range id is a programming error; fall back to air defensively.
        self.blocks
            .get(id.index())
            .unwrap_or(&self.blocks[0])
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Build the registry for the cute, cosy cat world. Colours lean pastel and
    /// warm to fit the theme. Ids must match [`blocks`].
    pub fn with_defaults() -> Self {
        use RenderKind::*;
        let mut r = Self::new();
        let opaque = |name, color, solid| Block {
            name,
            render: OpaqueCube,
            solid,
            color,
            light_emission: 0,
            harvestable: true,
        };

        // 0: air
        r.register(Block {
            name: "air",
            render: Invisible,
            solid: false,
            color: Color::rgba(0, 0, 0, 0),
            light_emission: 0,
            harvestable: false,
        });
        // 1: grass — soft meadow green
        r.register(opaque("grass", Color::rgb(126, 200, 96), true));
        // 2: dirt — warm cocoa
        r.register(opaque("dirt", Color::rgb(150, 108, 74), true));
        // 3: stone — gentle slate
        r.register(opaque("stone", Color::rgb(150, 152, 160), true));
        // 4: sand — pale honey
        r.register(opaque("sand", Color::rgb(237, 222, 167), true));
        // 5: water — translucent aqua
        r.register(Block {
            name: "water",
            render: Fluid,
            solid: false,
            color: Color::rgba(90, 170, 220, 160),
            light_emission: 0,
            harvestable: false,
        });
        // 6: tree trunk — soft bark
        r.register(opaque("trunk", Color::rgb(140, 100, 70), true));
        // 7: leaves — rounded canopy, slightly see-through
        r.register(Block {
            name: "leaves",
            render: TransparentCube,
            solid: true,
            color: Color::rgba(96, 178, 88, 235),
            light_emission: 0,
            harvestable: true,
        });
        // 8: snow — bright but not harsh
        r.register(opaque("snow", Color::rgb(238, 244, 250), true));
        // 9: pink flower
        r.register(Block {
            name: "flower_pink",
            render: Cross,
            solid: false,
            color: Color::rgb(244, 162, 196),
            light_emission: 0,
            harvestable: true,
        });
        // 10: blue flower
        r.register(Block {
            name: "flower_blue",
            render: Cross,
            solid: false,
            color: Color::rgb(150, 180, 240),
            light_emission: 0,
            harvestable: true,
        });
        // 11: crafted plank — cosy build material
        r.register(opaque("plank", Color::rgb(206, 162, 110), true));
        // 12: lantern — warm glow, the cosy light source
        r.register(Block {
            name: "lantern",
            render: OpaqueCube,
            solid: true,
            color: Color::rgb(255, 214, 140),
            light_emission: 14,
            harvestable: true,
        });
        // 13: mushroom — toadstool red
        r.register(Block {
            name: "mushroom",
            render: Cross,
            solid: false,
            color: Color::rgb(214, 96, 96),
            light_emission: 2,
            harvestable: true,
        });
        // 14: pumpkin — autumn orange
        r.register(opaque("pumpkin", Color::rgb(232, 150, 70), true));
        // 15: glass — cottage windows, see-through and non-occluding
        r.register(Block {
            name: "glass",
            render: TransparentCube,
            solid: true,
            color: Color::rgba(205, 232, 240, 110),
            light_emission: 0,
            harvestable: true,
        });
        // 16: cobblestone — sturdy build base / ruins
        r.register(opaque("cobble", Color::rgb(124, 126, 132), true));
        // 17: path — packed earthy walkway
        r.register(opaque("path", Color::rgb(168, 138, 96), true));
        // 18: crystal — glowing cave gem, soft cosy light
        r.register(Block {
            name: "crystal",
            render: OpaqueCube,
            solid: true,
            color: Color::rgb(170, 224, 230),
            light_emission: 11,
            harvestable: true,
        });
        // 19: tall grass — wispy ground cover, non-solid
        r.register(Block {
            name: "tall_grass",
            render: Cross,
            solid: false,
            color: Color::rgb(120, 196, 104),
            light_emission: 0,
            harvestable: true,
        });
        // 20: berry bush — collectible snack, reddish-green
        r.register(Block {
            name: "berry_bush",
            render: Cross,
            solid: false,
            color: Color::rgb(176, 96, 110),
            light_emission: 0,
            harvestable: true,
        });
        // 21: roof — warm clay tile for cottages
        r.register(opaque("roof", Color::rgb(196, 104, 86), true));
        // 22: coal ore — speckled dark stone
        r.register(opaque("coal_ore", Color::rgb(78, 80, 86), true));

        r
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_well_known_ids() {
        let r = BlockRegistry::with_defaults();
        assert_eq!(r.get(blocks::AIR).name, "air");
        assert_eq!(r.get(blocks::GRASS).name, "grass");
        assert_eq!(r.get(blocks::WATER).name, "water");
        assert_eq!(r.get(blocks::LANTERN).name, "lantern");
        assert!(r.get(blocks::LANTERN).light_emission > 0);
    }

    #[test]
    fn air_is_not_solid_or_occluding() {
        let r = BlockRegistry::with_defaults();
        let air = r.get(BlockId::AIR);
        assert!(!air.solid);
        assert!(!air.occludes());
        assert!(!air.is_visible());
    }

    #[test]
    fn leaves_are_visible_but_do_not_occlude() {
        let r = BlockRegistry::with_defaults();
        let leaves = r.get(blocks::LEAVES);
        assert!(leaves.is_visible());
        assert!(!leaves.occludes());
    }
}
