//! Physics for PixelCraft: AABB voxel collision and the cat character
//! controller. Pure CPU simulation, fully headless-testable.

pub mod aabb;
pub mod collision;
pub mod player;

pub use aabb::Aabb;
pub use collision::{move_and_collide, CollisionFlags, SolidQuery};
pub use player::{MovementConfig, MovementInput, Player};
