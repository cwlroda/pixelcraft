//! The cat character controller: a kinematic body with gravity, jumping and
//! smooth wall-sliding movement driven by player input.

use crate::aabb::Aabb;
use crate::collision::{move_and_collide, SolidQuery};
use glam::Vec3;

/// Tunable movement constants. Defaults are chosen to feel bouncy and cute —
/// a nimble little cat rather than a heavy human.
#[derive(Clone, Copy)]
pub struct MovementConfig {
    pub walk_speed: f32,
    pub sprint_speed: f32,
    pub jump_speed: f32,
    pub gravity: f32,
    /// Horizontal acceleration toward the target velocity (1/s).
    pub accel: f32,
    /// Damping applied to horizontal velocity each second when grounded.
    pub ground_friction: f32,
    pub air_control: f32,
    pub half_width: f32,
    pub height: f32,
    /// Eye height as a fraction of body height (for the FPV camera).
    pub eye_fraction: f32,
    pub terminal_velocity: f32,
}

impl Default for MovementConfig {
    fn default() -> Self {
        Self {
            walk_speed: 4.6,
            sprint_speed: 7.2,
            jump_speed: 8.4,
            gravity: 24.0,
            accel: 12.0,
            ground_friction: 9.0,
            air_control: 2.5,
            half_width: 0.3,
            height: 0.9,
            eye_fraction: 0.85,
            terminal_velocity: 55.0,
        }
    }
}

/// Per-frame player intent, already resolved to world-space directions by the
/// caller (who knows the camera yaw).
#[derive(Clone, Copy, Default)]
pub struct MovementInput {
    /// Desired horizontal move direction in world space (need not be
    /// normalised; it will be clamped to unit length).
    pub wish_dir: Vec3,
    pub jump: bool,
    pub sprint: bool,
}

/// A controllable cat. Position is the feet (bottom-centre of the AABB).
#[derive(Clone)]
pub struct Player {
    pub position: Vec3,
    pub velocity: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub on_ground: bool,
    pub config: MovementConfig,
    /// Coyote time: brief window after leaving an edge where a jump still works,
    /// a classic platformer feel-good trick.
    coyote: f32,
}

impl Player {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            on_ground: false,
            config: MovementConfig::default(),
            coyote: 0.0,
        }
    }

    /// The collision box at the current position.
    pub fn aabb(&self) -> Aabb {
        Aabb::from_feet(self.position, self.config.half_width, self.config.height)
    }

    /// World-space eye position for the first-person camera.
    pub fn eye(&self) -> Vec3 {
        self.position + Vec3::new(0.0, self.config.height * self.config.eye_fraction, 0.0)
    }

    /// Forward look direction from yaw/pitch (right-handed, -Z forward at yaw 0).
    pub fn look_dir(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, -cp * cy).normalize()
    }

    /// Horizontal forward/right basis for translating input into world space.
    pub fn horizontal_basis(&self) -> (Vec3, Vec3) {
        let (sy, cy) = self.yaw.sin_cos();
        let forward = Vec3::new(sy, 0.0, -cy);
        let right = Vec3::new(cy, 0.0, sy);
        (forward, right)
    }

    /// Advance the simulation by `dt` seconds against the world collision query.
    pub fn update<Q: SolidQuery>(&mut self, input: MovementInput, dt: f32, world: &Q) {
        let cfg = self.config;

        // --- Horizontal acceleration toward the wish velocity --------------
        let mut wish = input.wish_dir;
        wish.y = 0.0;
        let wish_len = wish.length();
        if wish_len > 1.0 {
            wish /= wish_len;
        }
        let target_speed = if input.sprint {
            cfg.sprint_speed
        } else {
            cfg.walk_speed
        };
        let target = wish * target_speed;

        let accel = if self.on_ground { cfg.accel } else { cfg.air_control };
        let horiz = Vec3::new(self.velocity.x, 0.0, self.velocity.z);
        let mut new_horiz = horiz + (target - horiz) * (accel * dt).min(1.0);

        // Ground friction when there's no input, so the cat coasts to a stop.
        if self.on_ground && wish_len < 1e-3 {
            let damping = (1.0 - cfg.ground_friction * dt).max(0.0);
            new_horiz *= damping;
        }
        self.velocity.x = new_horiz.x;
        self.velocity.z = new_horiz.z;

        // --- Jump (with coyote time) ---------------------------------------
        if self.on_ground {
            self.coyote = 0.12;
        } else {
            self.coyote = (self.coyote - dt).max(0.0);
        }
        if input.jump && self.coyote > 0.0 {
            self.velocity.y = cfg.jump_speed;
            self.coyote = 0.0;
            self.on_ground = false;
        }

        // --- Gravity --------------------------------------------------------
        self.velocity.y -= cfg.gravity * dt;
        self.velocity.y = self
            .velocity
            .y
            .clamp(-cfg.terminal_velocity, cfg.terminal_velocity);

        // --- Integrate + collide -------------------------------------------
        let delta = self.velocity * dt;
        let (resolved, flags) = move_and_collide(self.aabb(), delta, world);
        self.position = Vec3::new(
            resolved.min.x + cfg.half_width,
            resolved.min.y,
            resolved.min.z + cfg.half_width,
        );

        // Cancel velocity on axes we collided with so it doesn't accumulate.
        if flags.neg_y || flags.pos_y {
            self.velocity.y = 0.0;
        }
        if flags.neg_x || flags.pos_x {
            self.velocity.x = 0.0;
        }
        if flags.neg_z || flags.pos_z {
            self.velocity.z = 0.0;
        }
        self.on_ground = flags.on_ground();
    }

    /// Apply a mouse-look delta (radians), clamping pitch to avoid flipping.
    pub fn apply_look(&mut self, delta_yaw: f32, delta_pitch: f32) {
        use std::f32::consts::FRAC_PI_2;
        self.yaw += delta_yaw;
        // Keep yaw in a sane range to avoid float drift over long sessions.
        let two_pi = std::f32::consts::TAU;
        if self.yaw > two_pi {
            self.yaw -= two_pi;
        } else if self.yaw < -two_pi {
            self.yaw += two_pi;
        }
        self.pitch = (self.pitch + delta_pitch).clamp(-FRAC_PI_2 + 0.01, FRAC_PI_2 - 0.01);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelcraft_core::coords::BlockPos;

    fn floor(p: BlockPos) -> bool {
        p.y < 0
    }

    #[test]
    fn player_falls_under_gravity_and_lands() {
        let mut p = Player::new(Vec3::new(0.5, 8.0, 0.5));
        let input = MovementInput::default();
        // Simulate ~3 seconds at 60 Hz.
        for _ in 0..180 {
            p.update(input, 1.0 / 60.0, &floor);
        }
        assert!(p.on_ground, "player should have landed");
        assert!((p.position.y - 0.0).abs() < 1e-2, "rest height wrong: {}", p.position.y);
        assert!(p.velocity.y.abs() < 1e-3);
    }

    #[test]
    fn player_can_jump_when_grounded() {
        let mut p = Player::new(Vec3::new(0.5, 0.0, 0.5));
        // Settle on the ground first.
        for _ in 0..10 {
            p.update(MovementInput::default(), 1.0 / 60.0, &floor);
        }
        assert!(p.on_ground);
        let mut jump = MovementInput::default();
        jump.jump = true;
        p.update(jump, 1.0 / 60.0, &floor);
        assert!(p.velocity.y > 0.0, "jump did not impart upward velocity");
        assert!(!p.on_ground);
    }

    #[test]
    fn player_walks_in_wish_direction() {
        let mut p = Player::new(Vec3::new(0.5, 0.0, 0.5));
        for _ in 0..10 {
            p.update(MovementInput::default(), 1.0 / 60.0, &floor);
        }
        let mut input = MovementInput::default();
        input.wish_dir = Vec3::new(1.0, 0.0, 0.0);
        let start_x = p.position.x;
        for _ in 0..60 {
            p.update(input, 1.0 / 60.0, &floor);
        }
        assert!(p.position.x > start_x + 1.0, "player did not walk: {}", p.position.x);
    }

    #[test]
    fn cannot_double_jump_in_air() {
        let mut p = Player::new(Vec3::new(0.5, 0.0, 0.5));
        for _ in 0..10 {
            p.update(MovementInput::default(), 1.0 / 60.0, &floor);
        }
        let mut jump = MovementInput::default();
        jump.jump = true;
        p.update(jump, 1.0 / 60.0, &floor);
        let vy_after_first = p.velocity.y;
        // Let coyote time expire, then attempt another jump mid-air.
        for _ in 0..20 {
            p.update(jump, 1.0 / 60.0, &floor);
        }
        assert!(p.velocity.y < vy_after_first, "gravity should reduce upward velocity");
    }

    #[test]
    fn pitch_is_clamped() {
        let mut p = Player::new(Vec3::ZERO);
        p.apply_look(0.0, 100.0);
        assert!(p.pitch < std::f32::consts::FRAC_PI_2);
        p.apply_look(0.0, -100.0);
        assert!(p.pitch > -std::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn look_dir_is_unit_length() {
        let mut p = Player::new(Vec3::ZERO);
        p.yaw = 1.3;
        p.pitch = -0.4;
        assert!((p.look_dir().length() - 1.0).abs() < 1e-5);
    }
}
