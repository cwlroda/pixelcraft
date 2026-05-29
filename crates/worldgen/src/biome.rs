//! Biome classification for the cosy cat world.
//!
//! Biomes are chosen per world column from two low-frequency noise fields
//! (temperature and humidity), the classic Whittaker-diagram approach used by
//! Minecraft. Each biome contributes a surface palette, terrain shaping
//! parameters and decoration densities so the world feels varied but gentle.

use pixelcraft_core::block::{blocks, BlockId};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Biome {
    /// Rolling pastel-green hills — the friendly default.
    Meadow,
    /// Denser tree cover, cosy and shaded.
    Forest,
    /// Sandy honey shores near water.
    Beach,
    /// Cool snowy uplands.
    SnowyPeaks,
    /// Warm sandy dunes with the occasional pumpkin patch.
    Dunes,
}

/// Surface and shaping parameters derived from a biome.
pub struct BiomeProfile {
    pub surface: BlockId,
    pub subsurface: BlockId,
    /// Extra height multiplier applied to terrain relief.
    pub relief: f32,
    /// Baseline elevation offset in blocks.
    pub base_offset: f32,
    /// Probability (per column) of a tree, in `[0, 1]`.
    pub tree_density: f32,
    /// Probability (per column) of ground flora (flowers / mushrooms).
    pub flora_density: f32,
}

impl Biome {
    /// Classify from temperature & humidity, both in roughly `[-1, 1]`, plus the
    /// column's terrain height so coastlines and peaks override the climate.
    pub fn classify(temperature: f32, humidity: f32, height: i32, sea_level: i32) -> Biome {
        // Just above the waterline is always beach, regardless of climate.
        if height <= sea_level + 1 {
            return Biome::Beach;
        }
        if height >= sea_level + 40 && temperature < 0.1 {
            return Biome::SnowyPeaks;
        }
        if temperature > 0.45 && humidity < -0.1 {
            return Biome::Dunes;
        }
        if humidity > 0.15 {
            return Biome::Forest;
        }
        Biome::Meadow
    }

    pub fn profile(self) -> BiomeProfile {
        match self {
            Biome::Meadow => BiomeProfile {
                surface: blocks::GRASS,
                subsurface: blocks::DIRT,
                relief: 1.0,
                base_offset: 0.0,
                tree_density: 0.012,
                flora_density: 0.06,
            },
            Biome::Forest => BiomeProfile {
                surface: blocks::GRASS,
                subsurface: blocks::DIRT,
                relief: 1.15,
                base_offset: 1.0,
                tree_density: 0.085,
                flora_density: 0.05,
            },
            Biome::Beach => BiomeProfile {
                surface: blocks::SAND,
                subsurface: blocks::SAND,
                relief: 0.4,
                base_offset: -1.0,
                tree_density: 0.0,
                flora_density: 0.004,
            },
            Biome::SnowyPeaks => BiomeProfile {
                surface: blocks::SNOW,
                subsurface: blocks::STONE,
                relief: 1.9,
                base_offset: 6.0,
                tree_density: 0.02,
                flora_density: 0.0,
            },
            Biome::Dunes => BiomeProfile {
                surface: blocks::SAND,
                subsurface: blocks::SAND,
                relief: 0.8,
                base_offset: 0.0,
                tree_density: 0.0,
                flora_density: 0.012,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coast_is_beach() {
        let b = Biome::classify(0.0, 0.0, 63, 62);
        assert_eq!(b, Biome::Beach);
    }

    #[test]
    fn hot_dry_high_is_dunes() {
        let b = Biome::classify(0.8, -0.5, 90, 62);
        assert_eq!(b, Biome::Dunes);
    }

    #[test]
    fn cold_high_is_snowy() {
        let b = Biome::classify(-0.3, 0.0, 120, 62);
        assert_eq!(b, Biome::SnowyPeaks);
    }

    #[test]
    fn humid_is_forest() {
        let b = Biome::classify(0.2, 0.4, 80, 62);
        assert_eq!(b, Biome::Forest);
    }

    #[test]
    fn every_biome_has_a_profile() {
        for b in [
            Biome::Meadow,
            Biome::Forest,
            Biome::Beach,
            Biome::SnowyPeaks,
            Biome::Dunes,
        ] {
            let p = b.profile();
            assert!(p.tree_density >= 0.0 && p.tree_density <= 1.0);
            assert!(p.flora_density >= 0.0 && p.flora_density <= 1.0);
        }
    }
}
