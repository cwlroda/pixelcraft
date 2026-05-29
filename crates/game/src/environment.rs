//! Sky, sun and a gentle day/night cycle. Pure math (no GPU), so it's testable
//! headlessly and shared by the renderer.

use glam::Vec3;

/// Drives sky colour, sun direction and ambient light from a time-of-day phase.
/// The cycle is deliberately slow and the palette kept warm and cosy.
#[derive(Clone)]
pub struct Environment {
    /// Time of day in `[0, 1)`: 0.0 = dawn, 0.25 = noon, 0.5 = dusk, 0.75 = midnight.
    pub time_of_day: f32,
    /// Length of a full day in seconds.
    pub day_length: f32,
}

impl Default for Environment {
    fn default() -> Self {
        // Start mid-morning so a fresh world looks bright and inviting.
        Self {
            time_of_day: 0.12,
            day_length: 600.0,
        }
    }
}

impl Environment {
    pub fn advance(&mut self, dt: f32) {
        self.time_of_day = (self.time_of_day + dt / self.day_length).fract();
    }

    /// Normalised direction pointing toward the sun.
    pub fn sun_dir(&self) -> Vec3 {
        let angle = std::f32::consts::TAU * self.time_of_day;
        // Sun arcs across the sky in the X/Y plane, with a slight Z tilt so
        // lighting isn't perfectly axis-aligned (nicer face shading).
        Vec3::new(angle.cos(), angle.sin(), 0.35).normalize()
    }

    /// `1.0` in full day, fading toward `0.0` at night — used to dim the sun.
    pub fn daylight(&self) -> f32 {
        (self.sun_dir().y * 1.5 + 0.3).clamp(0.0, 1.0)
    }

    /// Ambient light strength: never fully dark (cosy moonlight floor).
    pub fn ambient(&self) -> f32 {
        0.28 + 0.30 * self.daylight()
    }

    /// Warm sun tint that cools at night.
    pub fn sun_color(&self) -> Vec3 {
        let day = self.daylight();
        // Warm gold by day, soft blue moonlight by night.
        let day_c = Vec3::new(1.0, 0.96, 0.86);
        let night_c = Vec3::new(0.55, 0.62, 0.85);
        night_c.lerp(day_c, day)
    }

    /// Zenith (overhead) sky colour — deeper than the horizon for a gradient.
    pub fn zenith_color(&self) -> Vec3 {
        let day = self.daylight();
        let day_zenith = Vec3::new(0.30, 0.55, 0.88);
        let night_zenith = Vec3::new(0.02, 0.03, 0.09);
        night_zenith.lerp(day_zenith, day)
    }

    /// Horizon/fog colour blending dawn pink → day blue → dusk → night.
    pub fn sky_color(&self) -> Vec3 {
        let day = self.daylight();
        let day_sky = Vec3::new(0.55, 0.78, 0.95); // gentle cornflower
        let night_sky = Vec3::new(0.06, 0.08, 0.16); // dark cosy indigo
        let mut sky = night_sky.lerp(day_sky, day);
        // Add a warm sunrise/sunset blush when the sun is near the horizon.
        let horizon = (1.0 - self.sun_dir().y.abs()).clamp(0.0, 1.0);
        let blush = Vec3::new(0.95, 0.55, 0.45) * horizon * day.max(0.15) * 0.35;
        sky += blush;
        sky.clamp(Vec3::ZERO, Vec3::ONE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_wraps_time() {
        let mut e = Environment {
            time_of_day: 0.9,
            day_length: 10.0,
        };
        e.advance(2.0); // +0.2 → wraps to 0.1
        assert!((e.time_of_day - 0.1).abs() < 1e-5, "{}", e.time_of_day);
    }

    #[test]
    fn sun_dir_is_normalised() {
        let mut e = Environment::default();
        for _ in 0..50 {
            assert!((e.sun_dir().length() - 1.0).abs() < 1e-5);
            e.advance(20.0);
        }
    }

    #[test]
    fn ambient_never_pitch_black() {
        let mut e = Environment::default();
        let mut min = f32::INFINITY;
        for _ in 0..100 {
            min = min.min(e.ambient());
            e.advance(7.0);
        }
        assert!(min >= 0.25, "night got too dark: {min}");
    }

    #[test]
    fn daylight_brighter_at_noon_than_midnight() {
        let noon = Environment { time_of_day: 0.25, day_length: 600.0 };
        let midnight = Environment { time_of_day: 0.75, day_length: 600.0 };
        assert!(noon.daylight() > midnight.daylight());
    }

    #[test]
    fn zenith_is_darker_at_night_and_in_gamut() {
        let noon = Environment { time_of_day: 0.25, day_length: 600.0 };
        let midnight = Environment { time_of_day: 0.75, day_length: 600.0 };
        assert!(noon.zenith_color().length() > midnight.zenith_color().length());
        for e in [&noon, &midnight] {
            let z = e.zenith_color();
            for ch in [z.x, z.y, z.z] {
                assert!((0.0..=1.0).contains(&ch));
            }
        }
    }

    #[test]
    fn sky_color_stays_in_gamut() {
        let mut e = Environment::default();
        for _ in 0..200 {
            let c = e.sky_color();
            for ch in [c.x, c.y, c.z] {
                assert!((0.0..=1.0).contains(&ch), "sky channel out of gamut: {ch}");
            }
            e.advance(3.5);
        }
    }
}
