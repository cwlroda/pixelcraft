//! Core voxel data structures for PixelCraft: coordinates, block definitions,
//! palette-compressed chunk storage and the sparse world container.
//!
//! This crate is deliberately free of any rendering or platform dependencies so
//! the simulation can be exercised headlessly in tests and on servers.

pub mod block;
pub mod chunk;
pub mod coords;
pub mod world;

pub use block::{Block, BlockId, BlockRegistry, Color, RenderKind};
pub use chunk::ChunkStorage;
pub use coords::{BlockPos, ChunkPos, LocalPos, CHUNK_SIZE, CHUNK_VOLUME};
pub use world::{Chunk, World};
