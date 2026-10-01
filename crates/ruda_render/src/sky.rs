//! The sky over a day: where the sun is, the colours of the sky and the fog,
//! and how bright and what colour sky light is.

use std::f32::consts::TAU;

use glam::Vec3;
use ruda_core::Light;

/// Colours here are sRGB, as picked; [`SkyLook::to_linear`] converts them
/// for rendering.
const DAY_ZENITH: Vec3 = Vec3::new(0.30, 0.52, 0.90);
const DAY_HORIZON: Vec3 = Vec3::new(0.66, 0.82, 0.95);
const DUSK_ZENITH: Vec3 = Vec3::new(0.30, 0.34, 0.58);
const NIGHT_ZENITH: Vec3 = Vec3::new(0.012, 0.016, 0.045);
const NIGHT_HORIZON: Vec3 = Vec3::new(0.04, 0.05, 0.10);
/// Sunrise and sunset colour the horizon on the sun's side.
const SUN_GLOW: Vec3 = Vec3::new(1.0, 0.52, 0.24);

/// Strength of sky light by day, at dusk and at night, as linear factors.
const DAYLIGHT: Vec3 = Vec3::new(1.0, 1.0, 1.0);
const DUSK_LIGHT: Vec3 = Vec3::new(1.0, 0.78, 0.62);
const MOONLIGHT: Vec3 = Vec3::new(0.16, 0.19, 0.30);
/// The least light anything gets, so caves aren't pitch black.
const AMBIENT: f32 = 0.02;

/// How the sky looks at one moment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SkyLook {
    /// Towards the sun; the moon is opposite.
    pub sun: Vec3,
    /// 0 at night, 1 by day.
    pub day: f32,
    pub zenith: Vec3,
    pub horizon: Vec3,
    /// The glow of sunrise and sunset, strongest at the horizon towards the
    /// sun.
    pub glow: Vec3,
    /// Fog and the colour behind everything.
    pub fog: Vec3,
    pub stars: f32,
    /// Colour and strength of full sky light, linear.
    pub sky_light: Vec3,
    pub ambient: f32,
}

impl SkyLook {
    /// `time_of_day` is the fraction of the day gone: 0 sunrise, 0.25 noon,
    /// 0.5 sunset, 0.75 midnight. `eye` is the light where the camera is:
    /// underground, the sky and fog darken.
    pub(crate) fn at(time_of_day: f32, eye: Light) -> Self {
        let angle = time_of_day.rem_euclid(1.0) * TAU;
        // East to west, tilted a little to the south.
        let sun = Vec3::new(angle.cos(), angle.sin(), 0.25).normalize();
        let day = smoothstep(-0.15, 0.15, sun.y);
        // Strongest with the sun at the horizon.
        let dusk = 1.0 - smoothstep(0.0, 0.35, sun.y.abs());

        let zenith = NIGHT_ZENITH
            .lerp(DAY_ZENITH, day)
            .lerp(DUSK_ZENITH, dusk * 0.5 * day);
        let horizon = NIGHT_HORIZON.lerp(DAY_HORIZON, day);
        let glow = SUN_GLOW * dusk * smoothstep(-0.25, 0.0, sun.y);
        let sky_light = MOONLIGHT
            .lerp(DAYLIGHT, day)
            .lerp(DUSK_LIGHT, dusk * 0.5 * day);

        // Seen from a cave, the sky and fog go dark.
        let eye = 0.06 + 0.94 * brightness(f32::from(eye.sky()) / f32::from(Light::MAX));
        Self {
            sun,
            day,
            zenith: zenith * eye,
            horizon: horizon * eye,
            glow: glow * eye,
            // The fog takes on some of the glow, as the horizon does on
            // average.
            fog: (horizon + glow * 0.35) * eye,
            stars: (1.0 - day) * eye,
            sky_light,
            ambient: AMBIENT,
        }
    }

    /// The colours as an sRGB render target expects them.
    pub(crate) fn to_linear(mut self) -> Self {
        let linear = |c: Vec3| c.map(|c| crate::srgb_to_linear(f64::from(c)) as f32);
        self.zenith = linear(self.zenith);
        self.horizon = linear(self.horizon);
        self.glow = linear(self.glow);
        self.fog = linear(self.fog);
        self
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
    fn noon_is_bright_and_midnight_dark() {
        let noon = SkyLook::at(0.25, Light::SKY);
        assert!(noon.sun.y > 0.9);
        assert_eq!(noon.day, 1.0);
        assert_eq!(noon.sky_light, DAYLIGHT);
        assert!(noon.stars < 0.01);

        let midnight = SkyLook::at(0.75, Light::SKY);
        assert!(midnight.sun.y < -0.9);
        assert_eq!(midnight.day, 0.0);
        assert!(midnight.sky_light.max_element() < 0.35);
        assert!(midnight.stars > 0.99);

        let sunset = SkyLook::at(0.5, Light::SKY);
        assert!(sunset.glow.x > 0.9);
    }

    #[test]
    fn caves_darken_the_fog() {
        let outside = SkyLook::at(0.25, Light::SKY);
        let cave = SkyLook::at(0.25, Light::DARK);
        assert!(cave.fog.max_element() < outside.fog.max_element() * 0.1);
    }
}
