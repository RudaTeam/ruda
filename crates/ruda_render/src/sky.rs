//! The sky over a day: where the sun is, how bright the stars are, and how
//! much of the sky the camera sees. Its colours and light come from the air,
//! see `atmosphere.rs`.

use std::f32::consts::TAU;

use glam::Vec3;
use ruda_core::Light;

/// The least light anything gets, so caves aren't pitch black.
const LEAST_LIGHT: f32 = 0.01;

/// How the sky looks at one moment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SkyLook {
    /// Towards the sun; the moon is opposite.
    pub sun: Vec3,
    /// 0 at night, 1 by day.
    pub day: f32,
    pub stars: f32,
    /// How much of the sky the camera sees: 1 outside, little deep in a
    /// cave, where the sky and the haze go dark.
    pub eye: f32,
    pub least: f32,
}

impl SkyLook {
    /// `time_of_day` is the fraction of the day gone: 0 sunrise, 0.25 noon,
    /// 0.5 sunset, 0.75 midnight. `eye` is the light where the camera is.
    pub(crate) fn at(time_of_day: f32, eye: Light) -> Self {
        let angle = time_of_day.rem_euclid(1.0) * TAU;
        // East to west, tilted a little to the south.
        let sun = Vec3::new(angle.cos(), angle.sin(), 0.25).normalize();
        let day = smoothstep(-0.15, 0.15, sun.y);
        Self {
            sun,
            day,
            stars: 1.0 - smoothstep(-0.25, 0.05, sun.y),
            eye: 0.06 + 0.94 * brightness(f32::from(eye.sky()) / f32::from(Light::MAX)),
            least: LEAST_LIGHT,
        }
    }
}

/// How bright a light level looks, from 0 to 1: the same curve as in
/// `world.wgsl`, steep near full light.
fn brightness(level: f32) -> f32 {
    level / (4.0 - 3.0 * level)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sun_rises_at_dawn_and_sets_at_dusk() {
        let noon = SkyLook::at(0.25, Light::SKY);
        assert!(noon.sun.y > 0.9);
        assert_eq!(noon.day, 1.0);
        assert!(noon.stars < 0.01);

        let midnight = SkyLook::at(0.75, Light::SKY);
        assert!(midnight.sun.y < -0.9);
        assert_eq!(midnight.day, 0.0);
        assert!(midnight.stars > 0.99);

        let dawn = SkyLook::at(0.0, Light::SKY);
        assert!(dawn.sun.y.abs() < 0.01 && dawn.sun.x > 0.9);
        let dusk = SkyLook::at(0.5, Light::SKY);
        assert!(dusk.sun.y.abs() < 0.01 && dusk.sun.x < -0.9);
    }

    #[test]
    fn caves_hide_the_sky() {
        let outside = SkyLook::at(0.25, Light::SKY);
        let cave = SkyLook::at(0.25, Light::DARK);
        assert_eq!(outside.eye, 1.0);
        assert!(cave.eye < 0.1);
    }
}
