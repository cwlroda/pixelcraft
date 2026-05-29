//! Axis-aligned bounding boxes and swept voxel collision resolution.

use glam::Vec3;

/// An axis-aligned box in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Build a box from its base centre (feet position) and half-extents on the
    /// horizontal axes plus full height — the natural way to describe a
    /// character standing on the ground.
    pub fn from_feet(feet: Vec3, half_width: f32, height: f32) -> Self {
        Self {
            min: Vec3::new(feet.x - half_width, feet.y, feet.z - half_width),
            max: Vec3::new(feet.x + half_width, feet.y + height, feet.z + half_width),
        }
    }

    pub fn translate(self, delta: Vec3) -> Self {
        Self {
            min: self.min + delta,
            max: self.max + delta,
        }
    }

    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// True if the two boxes overlap with a positive volume (touching faces do
    /// not count as overlap, so a character resting exactly on a surface is not
    /// considered intersecting it).
    pub fn intersects(self, other: Aabb) -> bool {
        self.min.x < other.max.x
            && self.max.x > other.min.x
            && self.min.y < other.max.y
            && self.max.y > other.min.y
            && self.min.z < other.max.z
            && self.max.z > other.min.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_feet_centers_horizontally() {
        let b = Aabb::from_feet(Vec3::new(2.0, 10.0, -3.0), 0.3, 0.8);
        assert_eq!(b.min, Vec3::new(1.7, 10.0, -3.3));
        assert_eq!(b.max, Vec3::new(2.3, 10.8, -2.7));
    }

    #[test]
    fn touching_faces_do_not_intersect() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0));
        assert!(!a.intersects(b));
    }

    #[test]
    fn overlap_detected() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(2.0, 2.0, 2.0));
        assert!(a.intersects(b));
    }
}
