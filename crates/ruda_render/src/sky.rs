//! The sky over a day: where the sun and the moon are, the light they and
//! the sky give, and how bright the picture should be. The colours come from
//! the air (see `atmosphere.rs`).

use std::f32::consts::TAU;

use glam::Vec3;
use ruda_core::Light;

use crate::atmosphere::{Lit, SkyTables};

/// The moon's light next to the sun's: far brighter than in life, so nights
/// are dark but not black.
const MOONLIGHT: f32 = 0.03;
/// Moonlight looks cold.
const MOON_TINT: Vec3 = Vec3::new(0.75, 0.88, 1.15);
/// Sky light on surfaces, more than the open sky alone gives: in a world of
/// blocks, light bounces between nearby faces too.
const SKY_BOUNCE: f32 = 1.4;
/// How much higher the sun stands for the clouds than for the ground, as the
/// sine of the angle: they keep glowing for a while after sunset.
const CLOUD_SUNSET: f32 = 0.05;
/// The least light anything gets, so caves aren't pitch black.
const AMBIENT: f32 = 0.004;
/// How bright full block light (a torch next to a face) is, next to the
/// noon sun's 1.
pub(crate) const BLOCK_LIGHT: f32 = 0.45;
/// The picture's brightness: what light the eye is adapted to shows as this.
const KEY: f32 = 1.0;
/// How much the eye adapts to darkness: 0 not at all, 1 fully, so night
/// would look as bright as day.
const ADAPTATION: f32 = 0.5;
/// The light the eye is adapted to in total darkness.
const DARK_ADAPTED: f32 = 0.002;
/// Seconds the eye takes to get most of the way used to new light.
const ADAPTING: f32 = 0.8;

/// The light where the camera is, as the eye has got used to it: how bright
/// sky light and block light are there, each from 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Eye {
    pub sky: f32,
    pub blocks: f32,
}

impl Eye {
    pub(crate) fn at(light: Light) -> Self {
        let level = |level: u8| brightness(f32::from(level) / f32::from(Light::MAX));
        Self {
            sky: level(light.sky()),
            blocks: level(light.red().max(light.green()).max(light.blue())),
        }
    }

    /// Part of the way to `seen` after `seconds`: stepping into a cave, the
    /// picture brightens over a moment instead of all at once.
    pub(crate) fn adapt(self, seen: Self, seconds: f32) -> Self {
        let t = 1.0 - (-seconds / ADAPTING).exp();
        Self {
            sky: self.sky + (seen.sky - self.sky) * t,
            blocks: self.blocks + (seen.blocks - self.blocks) * t,
        }
    }
}

/// Classic light, as in block games without shaders: colours picked by
/// hand for day, dusk and night. Picked in sRGB; [`ClassicSky::at`] returns
/// them linear.
const DAY_ZENITH: Vec3 = Vec3::new(0.30, 0.52, 0.90);
const DAY_HORIZON: Vec3 = Vec3::new(0.66, 0.82, 0.95);
const DUSK_ZENITH: Vec3 = Vec3::new(0.30, 0.34, 0.58);
const NIGHT_ZENITH: Vec3 = Vec3::new(0.012, 0.016, 0.045);
const NIGHT_HORIZON: Vec3 = Vec3::new(0.04, 0.05, 0.10);
/// Sunrise and sunset colour the horizon on the sun's side.
const SUN_GLOW: Vec3 = Vec3::new(1.0, 0.52, 0.24);
/// Strength of sky light by day, at dusk and at night, as linear factors.
const CLASSIC_DAYLIGHT: Vec3 = Vec3::new(1.0, 1.0, 1.0);
const CLASSIC_DUSK_LIGHT: Vec3 = Vec3::new(1.0, 0.78, 0.62);
const CLASSIC_MOONLIGHT: Vec3 = Vec3::new(0.16, 0.19, 0.30);
const CLASSIC_AMBIENT: f32 = 0.02;
/// Clouds: white by day, dark grey-blue by night, lit warm at dusk.
const DAY_CLOUD: Vec3 = Vec3::new(1.0, 1.0, 1.0);
const NIGHT_CLOUD: Vec3 = Vec3::new(0.07, 0.08, 0.12);
const DUSK_CLOUD: Vec3 = Vec3::new(1.0, 0.62, 0.42);

/// The sky and its light in classic light, linear.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ClassicSky {
    pub zenith: Vec3,
    pub horizon: Vec3,
    /// The glow of sunrise and sunset, strongest at the horizon towards the
    /// sun.
    pub glow: Vec3,
    /// Fog and the colour behind everything.
    pub fog: Vec3,
    pub stars: f32,
    /// Colour and strength of full sky light.
    pub light: Vec3,
    pub ambient: f32,
    /// Colour of a cloud's lit top.
    pub cloud: Vec3,
}

impl ClassicSky {
    /// With the sun towards `sun`, `day` from 0 at night to 1 by day, clouds
    /// covering `cover` of the sky and the sky and fog dimmed by `eye`.
    fn at(sun: Vec3, day: f32, cover: f32, eye: f32) -> Self {
        // Strongest with the sun at the horizon.
        let dusk = 1.0 - smoothstep(0.0, 0.35, sun.y.abs());
        let zenith = NIGHT_ZENITH
            .lerp(DAY_ZENITH, day)
            .lerp(DUSK_ZENITH, dusk * 0.5 * day);
        let horizon = NIGHT_HORIZON.lerp(DAY_HORIZON, day);
        let glow = SUN_GLOW * dusk * smoothstep(-0.25, 0.0, sun.y);
        let light = CLASSIC_MOONLIGHT
            .lerp(CLASSIC_DAYLIGHT, day)
            .lerp(CLASSIC_DUSK_LIGHT, dusk * 0.5 * day);
        let cloud = NIGHT_CLOUD
            .lerp(DAY_CLOUD, day)
            .lerp(DUSK_CLOUD, dusk * 0.6 * smoothstep(-0.25, 0.0, sun.y));
        // An overcast sky is greyer and its light weaker.
        let grey =
            |color: Vec3| color.lerp(Vec3::splat(color.dot(Vec3::splat(1.0 / 3.0))), cover * 0.5);
        let linear = |c: Vec3| c.map(|c| crate::srgb_to_linear(f64::from(c)) as f32);
        let (zenith, horizon) = (grey(zenith), grey(horizon));
        Self {
            zenith: linear(zenith) * eye,
            horizon: linear(horizon) * eye,
            glow: linear(glow) * eye,
            // The fog takes on some of the glow, as the horizon does on
            // average.
            fog: linear(horizon + glow * 0.35) * eye,
            stars: (1.0 - day) * eye,
            light: light * (1.0 - 0.3 * cover),
            ambient: CLASSIC_AMBIENT,
            cloud: linear(grey(cloud) * (1.0 - 0.35 * cover * cover)) * eye,
        }
    }
}

/// How the sky looks at one moment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SkyLook {
    /// Towards the sun; the moon is opposite.
    pub sun: Vec3,
    /// 0 at night, 1 by day.
    pub day: f32,
    /// Towards whichever of the sun and the moon lights the world.
    pub toward_light: Vec3,
    /// Its light on a white surface facing it.
    pub direct: Vec3,
    /// The sun's colour through the air, for its disc.
    pub sun_disc: Vec3,
    /// Towards the light the clouds get, and that light on a white surface
    /// facing it. High up, clouds still see the sun for a while after it
    /// has set on the ground, and glow red.
    pub cloud_light: Vec3,
    pub cloud_direct: Vec3,
    /// Sky light on a surface by its normal, divided by π: constant, x, y
    /// and z terms.
    pub ambient: [Vec3; 4],
    /// Layers of the sky table for the sun and the moon; below 0 for none.
    pub sun_layer: f32,
    pub moon_layer: f32,
    pub moonlight: f32,
    pub stars: f32,
    /// How much clouds cover the sky, 0 to 1.
    pub cover: f32,
    /// 1 outside; seen from a cave, the sky and fog go dark.
    pub eye: f32,
    /// How visible the sun and moon are: not from caves.
    pub discs: f32,
    pub ambient_floor: f32,
    /// What the light is multiplied by before it becomes a colour on screen.
    pub exposure: f32,
    /// The same moment in classic light.
    pub classic: ClassicSky,
}

impl SkyLook {
    /// `time_of_day` is the fraction of the day gone: 0 sunrise, 0.25 noon,
    /// 0.5 sunset, 0.75 midnight. `eye` is the light where the camera is:
    /// underground, the sky and fog darken and the eye adapts. `cover` is how
    /// much of the sky clouds cover, from 0 to 1: the more, the duller the
    /// light.
    pub(crate) fn at(time_of_day: f32, eye: Eye, cover: f32) -> Self {
        let angle = time_of_day.rem_euclid(1.0) * TAU;
        // East to west, tilted a little to the south.
        let sun = Vec3::new(angle.cos(), angle.sin(), 0.25).normalize();
        let day = smoothstep(-0.15, 0.15, sun.y);
        let tables = SkyTables::get();
        let by_sun = tables.light(sun, 1.0);
        let by_moon = tables.light(-sun, MOONLIGHT);

        let moonlight = by_moon.direct * MOON_TINT;
        let (toward_light, direct) = trade(sun, by_sun.direct, moonlight);
        let ambient: [Vec3; 4] =
            std::array::from_fn(|i| (by_sun.ambient[i] + by_moon.ambient[i]) * SKY_BOUNCE);

        let raised = (sun + Vec3::Y * CLOUD_SUNSET).normalize();
        let (cloud_light, cloud_direct) =
            trade(raised, tables.light(raised, 1.0).direct, moonlight);

        // Clouds cast their own shadows; an overcast sky also greys and dims
        // the light of the sky.
        let cover = cover.clamp(0.0, 1.0);
        let grey = |c: Vec3| c.lerp(Vec3::splat(luminance(c)), cover * 0.5) * (1.0 - 0.3 * cover);
        let ambient = ambient.map(grey);

        // What the eye is adapted to: the light on the ground outside, a
        // torch nearby, or darkness.
        let outside = luminance(direct * toward_light.y.max(0.0) + ambient[0] + ambient[2]);
        let adapted = eye.sky * outside + eye.blocks * BLOCK_LIGHT * 0.3 + DARK_ADAPTED;

        let layer = |lit: Lit| lit.layer.unwrap_or(-1.0);
        Self {
            sun,
            day,
            toward_light,
            direct,
            sun_disc: by_sun.direct,
            cloud_light,
            cloud_direct,
            ambient,
            sun_layer: layer(by_sun),
            moon_layer: layer(by_moon),
            moonlight: MOONLIGHT,
            // Only once the sun is down.
            stars: 1.0 - smoothstep(-0.2, 0.0, sun.y),
            cover,
            eye: 0.06 + 0.94 * eye.sky,
            discs: eye.sky,
            ambient_floor: AMBIENT,
            exposure: KEY / adapted.powf(ADAPTATION),
            classic: ClassicSky::at(sun, day, cover, 0.06 + 0.94 * eye.sky),
        }
    }
}

/// Direct light comes from one direction: whichever of the sun (towards
/// `sun`) and the moon (opposite) is brighter. As they trade places, the
/// light dims to nothing and comes back from the other side, so it never
/// jumps at sunset and sunrise.
fn trade(sun: Vec3, sunlight: Vec3, moonlight: Vec3) -> (Vec3, Vec3) {
    let (by_sun, by_moon) = (luminance(sunlight), luminance(moonlight));
    let fade = (by_sun - by_moon).abs() / (by_sun + by_moon).max(1e-9);
    let toward = if by_sun >= by_moon { sun } else { -sun };
    (toward, (sunlight + moonlight) * fade)
}

fn luminance(color: Vec3) -> f32 {
    color.dot(Vec3::new(0.2126, 0.7152, 0.0722))
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
        let noon = SkyLook::at(0.25, Eye::at(Light::SKY), 0.0);
        assert!(noon.sun.y > 0.9);
        assert_eq!(noon.day, 1.0);
        assert!(noon.direct.min_element() > 0.7, "{}", noon.direct);
        assert!(noon.stars < 0.01);
        assert!(noon.sun_layer > 0.0 && noon.moon_layer < 0.0);

        let midnight = SkyLook::at(0.75, Eye::at(Light::SKY), 0.0);
        assert!(midnight.sun.y < -0.9);
        assert_eq!(midnight.day, 0.0);
        assert_eq!(midnight.toward_light, -midnight.sun);
        assert!(midnight.direct.max_element() < 0.05, "{}", midnight.direct);
        // Moonlight is cold.
        assert!(midnight.direct.z > midnight.direct.x);
        assert!(midnight.stars > 0.99);
        assert!(midnight.sun_layer < 0.0 && midnight.moon_layer > 0.0);

        // The eye adapts to the dark, but not all the way.
        assert!(midnight.exposure > noon.exposure * 3.0);
        let seen = |look: SkyLook| luminance(look.direct) * look.exposure;
        assert!(seen(midnight) < seen(noon) * 0.5);
    }

    #[test]
    fn clouds_glow_after_sunset() {
        let dusk = SkyLook::at(0.505, Eye::at(Light::SKY), 0.0);
        assert!(dusk.sun.y < 0.0);
        // The ground gets only the rising moon's faint light.
        assert_eq!(dusk.toward_light, -dusk.sun);
        assert!(dusk.direct.max_element() < 0.01, "{}", dusk.direct);
        assert!(dusk.cloud_direct.x > 0.01, "{}", dusk.cloud_direct);
        assert!(dusk.cloud_direct.x > dusk.cloud_direct.z * 3.0);
    }

    #[test]
    fn the_light_fades_smoothly_through_sunset() {
        // A day is 20 minutes. Every tenth of a second through dusk and dawn,
        // the light on the ground and on the clouds changes only a little:
        // fast near the horizon, but without jumps.
        for (from, to) in [(0.47f32, 0.53f32), (0.97, 1.03)] {
            let steps = ((to - from) * 12_000.0) as usize;
            let at = |i: usize| {
                let time = from + (to - from) * i as f32 / steps as f32;
                SkyLook::at(time, Eye::at(Light::SKY), 0.0)
            };
            let mut last = at(0);
            for i in 1..=steps {
                let next = at(i);
                for (a, b) in [
                    (last.direct, next.direct),
                    (last.cloud_direct, next.cloud_direct),
                ] {
                    let step = (a - b).abs().max_element();
                    assert!(step < 0.006, "{a} → {b} at step {i}");
                }
                last = next;
            }
        }
    }

    #[test]
    fn classic_light_follows_the_day() {
        let noon = SkyLook::at(0.25, Eye::at(Light::SKY), 0.0).classic;
        let midnight = SkyLook::at(0.75, Eye::at(Light::SKY), 0.0).classic;
        let sunset = SkyLook::at(0.5, Eye::at(Light::SKY), 0.0).classic;
        assert_eq!(noon.light, CLASSIC_DAYLIGHT);
        assert!(midnight.light.max_element() < 0.35);
        assert!(midnight.stars > 0.99 && noon.stars < 0.01);
        assert!(sunset.glow.x > 0.5);
        assert!(
            sunset.cloud.x > sunset.cloud.z,
            "clouds glow warm at sunset"
        );
    }

    #[test]
    fn the_setting_sun_is_red() {
        let sunset = SkyLook::at(0.49, Eye::at(Light::SKY), 0.0);
        assert!(sunset.direct.x > sunset.direct.z * 3.0, "{}", sunset.direct);
        assert!(sunset.sun_disc.x > sunset.sun_disc.z * 3.0);
    }

    #[test]
    fn an_overcast_sky_dims_the_light() {
        let clear = SkyLook::at(0.25, Eye::at(Light::SKY), 0.0);
        let overcast = SkyLook::at(0.25, Eye::at(Light::SKY), 1.0);
        assert!(overcast.ambient[0].x < clear.ambient[0].x * 0.9);
        let saturation = |c: Vec3| c.max_element() - c.min_element();
        assert!(saturation(overcast.ambient[0]) < saturation(clear.ambient[0]));
    }

    #[test]
    fn caves_darken_the_fog_and_the_eye_adapts() {
        let outside = SkyLook::at(0.25, Eye::at(Light::SKY), 0.0);
        let cave = SkyLook::at(0.25, Eye::at(Light::DARK), 0.0);
        assert!(cave.eye < outside.eye * 0.1);
        assert!(cave.exposure > outside.exposure * 5.0);
        let torch = SkyLook::at(0.25, Eye::at(Light::rgb(14, 12, 8)), 0.0);
        assert!(torch.exposure < cave.exposure);
    }

    #[test]
    fn the_eye_adapts_over_a_moment() {
        let (outside, cave) = (Eye::at(Light::SKY), Eye::at(Light::DARK));
        let soon = outside.adapt(cave, 0.1);
        assert!(soon.sky < outside.sky && soon.sky > 0.8);
        let later = outside.adapt(cave, 5.0);
        assert!(later.sky < 0.01);
    }
}
