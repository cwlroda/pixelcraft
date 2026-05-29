//! First-person camera: view + projection matrices and a frustum for culling.

use glam::{Mat4, Vec3, Vec4};

/// Perspective FPV camera derived from an eye position and look direction.
#[derive(Clone, Copy)]
pub struct Camera {
    pub eye: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(eye: Vec3, forward: Vec3, aspect: f32) -> Self {
        Self {
            eye,
            forward: forward.normalize_or_zero(),
            up: Vec3::Y,
            fov_y: 70f32.to_radians(),
            aspect,
            near: 0.05,
            far: 1000.0,
        }
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_to_rh(self.eye, self.forward, self.up)
    }

    pub fn projection(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect.max(0.01), self.near, self.far)
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection() * self.view()
    }

    /// Extract a frustum for chunk culling from the current view-projection.
    pub fn frustum(&self) -> Frustum {
        Frustum::from_view_projection(self.view_projection())
    }
}

/// Six world-space planes bounding the visible volume. Normals point inward, so
/// a point is inside when it is on the positive side of every plane.
#[derive(Clone, Copy)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// Gribb–Hartmann plane extraction from a view-projection matrix.
    pub fn from_view_projection(vp: Mat4) -> Self {
        // glam stores column-major; build rows for the standard formulation.
        let m = vp.to_cols_array_2d();
        let row = |i: usize| Vec4::new(m[0][i], m[1][i], m[2][i], m[3][i]);
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);
        let planes = [
            r3 + r0, // left
            r3 - r0, // right
            r3 + r1, // bottom
            r3 - r1, // top
            r3 + r2, // near
            r3 - r2, // far
        ]
        .map(normalize_plane);
        Self { planes }
    }

    /// True if the axis-aligned box `[min, max]` is at least partially inside.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for p in &self.planes {
            let n = p.truncate();
            // Pick the box corner furthest along the plane normal (the
            // "positive vertex"); if even that is outside, the box is culled.
            let positive = Vec3::new(
                if n.x >= 0.0 { max.x } else { min.x },
                if n.y >= 0.0 { max.y } else { min.y },
                if n.z >= 0.0 { max.z } else { min.z },
            );
            if n.dot(positive) + p.w < 0.0 {
                return false;
            }
        }
        true
    }
}

fn normalize_plane(p: Vec4) -> Vec4 {
    let len = p.truncate().length();
    if len > 0.0 {
        p / len
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Camera {
        // Looking down -Z from origin.
        Camera::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 16.0 / 9.0)
    }

    #[test]
    fn box_in_front_is_visible() {
        let f = cam().frustum();
        assert!(f.intersects_aabb(Vec3::new(-1.0, -1.0, -10.0), Vec3::new(1.0, 1.0, -8.0)));
    }

    #[test]
    fn box_behind_is_culled() {
        let f = cam().frustum();
        assert!(!f.intersects_aabb(Vec3::new(-1.0, -1.0, 8.0), Vec3::new(1.0, 1.0, 10.0)));
    }

    #[test]
    fn far_offscreen_box_is_culled() {
        let f = cam().frustum();
        // Way off to the right but at the camera plane depth.
        assert!(!f.intersects_aabb(Vec3::new(500.0, 0.0, -10.0), Vec3::new(520.0, 1.0, -9.0)));
    }

    #[test]
    fn view_projection_is_finite() {
        let vp = cam().view_projection();
        for c in vp.to_cols_array() {
            assert!(c.is_finite());
        }
    }
}
