//! Swept axis-by-axis voxel collision, the approach used by Minecraft-style
//! engines: move the box along one axis at a time, clipping the motion against
//! every solid unit-cube it would sweep through. Resolving axes independently
//! lets a character slide smoothly along walls instead of sticking.

use crate::aabb::Aabb;
use glam::Vec3;
use pixelcraft_core::coords::BlockPos;

/// Lets collision query whether a given block cell is solid.
pub trait SolidQuery {
    fn is_solid(&self, pos: BlockPos) -> bool;
}

/// Blanket impl so callers can pass a simple closure.
impl<F: Fn(BlockPos) -> bool> SolidQuery for F {
    fn is_solid(&self, pos: BlockPos) -> bool {
        self(pos)
    }
}

/// Which faces the box was blocked against during a move.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollisionFlags {
    pub neg_x: bool,
    pub pos_x: bool,
    pub neg_y: bool,
    pub pos_y: bool,
    pub neg_z: bool,
    pub pos_z: bool,
}

impl CollisionFlags {
    /// Standing on the ground this frame.
    pub fn on_ground(self) -> bool {
        self.neg_y
    }
    /// Bumped a ceiling.
    pub fn hit_ceiling(self) -> bool {
        self.pos_y
    }
    pub fn hit_wall(self) -> bool {
        self.neg_x || self.pos_x || self.neg_z || self.pos_z
    }
}

/// Move `aabb` by `delta`, clipping against solid voxels. Returns the resolved
/// box and the faces it collided with.
pub fn move_and_collide<Q: SolidQuery>(
    aabb: Aabb,
    delta: Vec3,
    world: &Q,
) -> (Aabb, CollisionFlags) {
    let mut flags = CollisionFlags::default();
    let mut current = aabb;

    // Resolve Y first so that landing/headbump is detected before horizontal
    // sliding, then X, then Z.
    let dy = clip_axis(current, delta.y, 1, world);
    current = current.translate(Vec3::new(0.0, dy, 0.0));
    if dy != delta.y {
        if delta.y < 0.0 {
            flags.neg_y = true;
        } else if delta.y > 0.0 {
            flags.pos_y = true;
        }
    }

    let dx = clip_axis(current, delta.x, 0, world);
    current = current.translate(Vec3::new(dx, 0.0, 0.0));
    if dx != delta.x {
        if delta.x < 0.0 {
            flags.neg_x = true;
        } else if delta.x > 0.0 {
            flags.pos_x = true;
        }
    }

    let dz = clip_axis(current, delta.z, 2, world);
    current = current.translate(Vec3::new(0.0, 0.0, dz));
    if dz != delta.z {
        if delta.z < 0.0 {
            flags.neg_z = true;
        } else if delta.z > 0.0 {
            flags.pos_z = true;
        }
    }

    (current, flags)
}

/// Clip a single-axis movement `dist` (along `axis` 0=x,1=y,2=z) so the box does
/// not pass into any solid voxel.
fn clip_axis<Q: SolidQuery>(aabb: Aabb, dist: f32, axis: usize, world: &Q) -> f32 {
    if dist == 0.0 {
        return 0.0;
    }
    // Broadphase: the union of the box now and where it wants to be.
    let swept = swept_bounds(aabb, dist, axis);
    let (lo, hi) = (swept.min, swept.max);

    let bx0 = lo.x.floor() as i32;
    let bx1 = (hi.x - 1e-4).floor() as i32;
    let by0 = lo.y.floor() as i32;
    let by1 = (hi.y - 1e-4).floor() as i32;
    let bz0 = lo.z.floor() as i32;
    let bz1 = (hi.z - 1e-4).floor() as i32;

    let mut allowed = dist;
    for bx in bx0..=bx1 {
        for by in by0..=by1 {
            for bz in bz0..=bz1 {
                if !world.is_solid(BlockPos::new(bx, by, bz)) {
                    continue;
                }
                // Unit cube occupied by this block.
                let cube = Aabb::new(
                    Vec3::new(bx as f32, by as f32, bz as f32),
                    Vec3::new(bx as f32 + 1.0, by as f32 + 1.0, bz as f32 + 1.0),
                );
                // Only blocks overlapping on the *other* two axes can stop us.
                if !overlaps_other_axes(aabb, cube, axis) {
                    continue;
                }
                allowed = clip_against(aabb, cube, allowed, axis);
            }
        }
    }
    allowed
}

fn swept_bounds(aabb: Aabb, dist: f32, axis: usize) -> Aabb {
    let mut min = aabb.min;
    let mut max = aabb.max;
    match axis {
        0 => {
            if dist > 0.0 {
                max.x += dist;
            } else {
                min.x += dist;
            }
        }
        1 => {
            if dist > 0.0 {
                max.y += dist;
            } else {
                min.y += dist;
            }
        }
        _ => {
            if dist > 0.0 {
                max.z += dist;
            } else {
                min.z += dist;
            }
        }
    }
    Aabb::new(min, max)
}

fn overlaps_other_axes(a: Aabb, b: Aabb, axis: usize) -> bool {
    let x = a.min.x < b.max.x && a.max.x > b.min.x;
    let y = a.min.y < b.max.y && a.max.y > b.min.y;
    let z = a.min.z < b.max.z && a.max.z > b.min.z;
    match axis {
        0 => y && z,
        1 => x && z,
        _ => x && y,
    }
}

/// Reduce `dist` so `a` moving along `axis` stops at the face of `b`.
fn clip_against(a: Aabb, b: Aabb, dist: f32, axis: usize) -> f32 {
    let (amin, amax, bmin, bmax) = match axis {
        0 => (a.min.x, a.max.x, b.min.x, b.max.x),
        1 => (a.min.y, a.max.y, b.min.y, b.max.y),
        _ => (a.min.z, a.max.z, b.min.z, b.max.z),
    };
    if dist > 0.0 {
        // Moving toward +: stop a's max face at b's min face.
        let gap = bmin - amax;
        if gap >= 0.0 && gap < dist {
            return gap;
        }
    } else {
        // Moving toward -: stop a's min face at b's max face.
        let gap = bmax - amin;
        if gap <= 0.0 && gap > dist {
            return gap;
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solid floor at y < 0 (everything below the plane y=0 is solid).
    fn floor(p: BlockPos) -> bool {
        p.y < 0
    }

    #[test]
    fn falls_onto_floor_and_stops() {
        // Box just above the floor, falling.
        let b = Aabb::from_feet(Vec3::new(0.5, 5.0, 0.5), 0.3, 0.8);
        let (resolved, flags) = move_and_collide(b, Vec3::new(0.0, -10.0, 0.0), &floor);
        assert!(
            (resolved.min.y - 0.0).abs() < 1e-3,
            "feet not at floor: {}",
            resolved.min.y
        );
        assert!(flags.on_ground());
    }

    #[test]
    fn unobstructed_move_is_exact() {
        let b = Aabb::from_feet(Vec3::new(0.5, 5.0, 0.5), 0.3, 0.8);
        let (resolved, flags) = move_and_collide(b, Vec3::new(1.0, 0.0, 0.0), &floor);
        assert!((resolved.min.x - 1.2).abs() < 1e-4);
        assert_eq!(flags, CollisionFlags::default());
    }

    #[test]
    fn slides_along_wall() {
        // Wall: all blocks with x >= 1 are solid.
        let wall = |p: BlockPos| p.x >= 1;
        let b = Aabb::from_feet(Vec3::new(0.5, 5.0, 0.5), 0.3, 0.8);
        // Try to move diagonally into the wall (+x, +z).
        let (resolved, flags) = move_and_collide(b, Vec3::new(5.0, 0.0, 2.0), &wall);
        // X is blocked (box max.x can't exceed 1.0), but Z slides freely.
        assert!(
            resolved.max.x <= 1.0 + 1e-4,
            "penetrated wall: {}",
            resolved.max.x
        );
        assert!(flags.pos_x);
        assert!(
            (resolved.min.z - (0.2 + 2.0)).abs() < 1e-3,
            "z did not slide: {}",
            resolved.min.z
        );
        assert!(!flags.pos_z);
    }

    #[test]
    fn ceiling_blocks_upward_motion() {
        // Solid ceiling for y >= 3.
        let ceil = |p: BlockPos| p.y >= 3;
        let b = Aabb::from_feet(Vec3::new(0.5, 0.0, 0.5), 0.3, 1.0); // top at y=1
        let (resolved, flags) = move_and_collide(b, Vec3::new(0.0, 10.0, 0.0), &ceil);
        assert!(
            resolved.max.y <= 3.0 + 1e-4,
            "passed ceiling: {}",
            resolved.max.y
        );
        assert!(flags.hit_ceiling());
    }

    #[test]
    fn no_collision_in_open_space() {
        let empty = |_p: BlockPos| false;
        let b = Aabb::from_feet(Vec3::new(0.0, 50.0, 0.0), 0.3, 0.8);
        let (resolved, flags) = move_and_collide(b, Vec3::new(3.0, -2.0, -4.0), &empty);
        assert_eq!(flags, CollisionFlags::default());
        assert!((resolved.min.x - (-0.3 + 3.0)).abs() < 1e-4);
    }
}
