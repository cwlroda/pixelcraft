//! Deterministic, infinite world generation for PixelCraft.
//!
//! Pure CPU-side simulation with no rendering dependencies, so terrain can be
//! generated on background threads and verified in headless tests.

pub mod biome;
pub mod generator;
pub mod noise;

pub use biome::{Biome, BiomeProfile};
pub use generator::{WorldGenerator, SEA_LEVEL};
pub use noise::{hash_coords, Perlin, SplitMix64};
